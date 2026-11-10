# Times each stage of a stereo depth pipeline on a Vulkan device and checks
# its results against the Nx code on EXLA: rectification, matching, the
# speckle and temporal filters, depth, the point cloud and the diagnostics,
# on a synthetic textured pair.
#
#     mix run bench/pipeline.exs [--size 544x728] [--frame 1088x1456]
#       [--search full|coarse_to_fine] [--disparities 64|128] [--paths 3|8]
#       [--coarse-paths 3|8] [--band 16] [--packed] [--kernels] [--runs 20]
#       [--device URI_SUBSTRING]
#
# Each stage is compiled alone and its inputs are on the device, so a
# stage's time is a call's: its kernels, the IREE code around them and the
# call itself. "Call" is the time of a call that does nothing, the part of
# each stage's time no kernel can remove. "Chain" runs every stage in one
# call, as a pipeline does. Times are medians, in milliseconds.
#
# --kernels also lists, under each stage, the time of each of its Vulkan
# kernels: each kernel compiled alone, its inputs the values the kernels
# before it in the stage produced from the stage's real inputs. A kernel's
# time includes a call's, so compare kernels against each other and against
# "Call", not their sum against the stage's.
#
# --packed passes the frames as 32-bit words, four pixels each, the way a
# GPU without 8-bit storage receives them. "Differing" counts the elements
# of a stage's output that differ from EXLA's by more than its tolerance,
# given the same inputs.

defmodule Bench.Kernels do
  @moduledoc false
  # The Vulkan kernels of a stage, each timed alone on the device.

  defmodule Capture do
    @moduledoc false
    # A compiler that records the NxIREE.Vulkan.Pipeline of each Nx.block the function
    # traces, and computes the blocks' inputs with EXLA.

    def __jit__(key, vars, fun, args_list, opts) do
      blocks = blocks(fun.(vars))
      Process.put(__MODULE__, Enum.map(blocks, &elem(&1, 0)))
      inputs = blocks |> Enum.flat_map(&elem(&1, 1)) |> List.to_tuple()
      # A key of its own: EXLA's cache would give the function's compiled outputs.
      EXLA.Defn.__jit__({__MODULE__, key}, vars, fn _vars -> inputs end, args_list, opts)
    end

    def __compile__(key, vars, fun, opts) do
      blocks = blocks(fun.(vars))
      Process.put(__MODULE__, Enum.map(blocks, &elem(&1, 0)))
      inputs = blocks |> Enum.flat_map(&elem(&1, 1)) |> List.to_tuple()
      EXLA.Defn.__compile__({__MODULE__, key}, vars, fn _vars -> inputs end, opts)
    end

    def __partitions_options__(opts), do: EXLA.Defn.__partitions_options__(opts)
    def __to_backend__(opts), do: EXLA.Defn.__to_backend__(opts)

    defp blocks(expr) do
      {blocks, _seen} = walk(Nx.Defn.Composite.flatten_list([expr]), {[], MapSet.new()})
      Enum.reverse(blocks)
    end

    defp walk(nodes, acc) when is_list(nodes), do: Enum.reduce(nodes, acc, &walk/2)

    defp walk(%Nx.Tensor{data: %Nx.Defn.Expr{id: id, op: op, args: args}}, {blocks, seen} = acc) do
      if MapSet.member?(seen, id) do
        acc
      else
        acc = {blocks, MapSet.put(seen, id)}

        case {op, args} do
          {:block, [%NxIREE.Vulkan.Pipeline{} = pipeline, inputs | _]} ->
            {blocks, seen} = walk(inputs, acc)
            {[{pipeline, inputs} | blocks], seen}

          _ ->
            walk(tensors(args), acc)
        end
      end
    end

    defp walk(_other, acc), do: acc

    defp tensors(%Nx.Tensor{} = tensor), do: [tensor]
    defp tensors(list) when is_list(list), do: Enum.flat_map(list, &tensors/1)
    defp tensors(tuple) when is_tuple(tuple), do: tuple |> Tuple.to_list() |> tensors()
    defp tensors(%{__struct__: _}), do: []
    defp tensors(map) when is_map(map), do: map |> Map.values() |> tensors()
    defp tensors(_other), do: []
  end

  # [{name, ms}] for each kernel of each pipeline `fun` runs on `args`.
  def rows(fun, args, device, runs) do
    inputs =
      case apply(Nx.Defn.jit(fun, compiler: Capture), args) do
        tuple when is_tuple(tuple) -> Tuple.to_list(tuple)
        one -> [one]
      end

    pipelines = Process.get(Capture)

    {rows, []} =
      Enum.flat_map_reduce(pipelines, inputs, fn pipeline, inputs ->
        {mine, rest} = Enum.split(inputs, length(pipeline.inputs))
        {time_kernels(pipeline, mine, device, runs), rest}
      end)

    rows
  end

  defp time_kernels(pipeline, inputs, device, runs) do
    gpu = [compiler: NxIREE.Compiler, iree_runtime_options: [device: device]]
    on_device = &Nx.backend_copy(&1, {NxIREE.Backend, device: device})
    values = Map.new(Enum.zip(pipeline.inputs, Enum.map(inputs, on_device)))

    {rows, _values} =
      Enum.map_reduce(pipeline.stages, values, fn {kernel, reads, writes} = stage, values ->
        single = %NxIREE.Vulkan.Pipeline{
          inputs: reads,
          stages: [stage],
          outputs: Keyword.keys(writes)
        }

        template =
          case Keyword.values(writes) do
            [one] -> one
            many -> List.to_tuple(many)
          end

        compiled =
          Nx.Defn.jit(
            fn args ->
              Nx.block(single, Tuple.to_list(args), template, zeros(tuple_size(args), template))
            end,
            gpu
          )

        args = reads |> Enum.map(&Map.fetch!(values, &1)) |> List.to_tuple()
        output = compiled.(args)
        ms = Bench.Pipeline.median_ms(fn -> compiled.(args) end, runs)
        outputs = if is_tuple(output), do: Tuple.to_list(output), else: [output]
        name = writes |> Keyword.keys() |> Enum.join(", ")

        {{"#{name} #{inspect(kernel.workgroups)}", ms},
         Map.merge(values, Map.new(Enum.zip(Keyword.keys(writes), outputs)))}
      end)

    rows
  end

  # The block's default, never run on Vulkan: Nx.block calls it with the struct and each input.
  defp zeros(count, template) do
    zero = fn ->
      Nx.Defn.Composite.traverse(template, &Nx.broadcast(Nx.tensor(0, type: &1.type), &1.shape))
    end

    case count do
      1 -> fn _, _ -> zero.() end
      2 -> fn _, _, _ -> zero.() end
      3 -> fn _, _, _, _ -> zero.() end
      4 -> fn _, _, _, _, _ -> zero.() end
      5 -> fn _, _, _, _, _, _ -> zero.() end
      6 -> fn _, _, _, _, _, _, _ -> zero.() end
      7 -> fn _, _, _, _, _, _, _, _ -> zero.() end
      8 -> fn _, _, _, _, _, _, _, _, _ -> zero.() end
    end
  end
end

defmodule Bench.Pipeline do
  alias NxStereo.{Depth, Diagnostics, Filter, Packing, Rectify}

  @defaults [
    size: "544x728",
    frame: nil,
    search: "coarse_to_fine",
    disparities: 128,
    paths: 8,
    coarse_paths: 3,
    band: 16,
    packed: false,
    kernels: false,
    runs: 20,
    device: ""
  ]

  def main(args) do
    {opts, _} =
      OptionParser.parse!(args,
        strict: [
          size: :string,
          frame: :string,
          search: :string,
          disparities: :integer,
          paths: :integer,
          coarse_paths: :integer,
          band: :integer,
          packed: :boolean,
          kernels: :boolean,
          runs: :integer,
          device: :string
        ]
      )

    opts = Keyword.merge(@defaults, opts)
    {height, width} = size = dimensions(opts[:size])
    frame_size = if opts[:frame], do: dimensions(opts[:frame]), else: {height * 2, width * 2}
    {:ok, devices} = NxIREE.list_devices("vulkan")

    device =
      Enum.find(devices, &String.contains?(&1.uri, opts[:device])) || raise "no Vulkan device"

    IO.puts(
      "#{device.uri}: #{width}x#{height} from #{elem(frame_size, 1)}x#{elem(frame_size, 0)} frames"
    )

    gpu = [compiler: NxIREE.Compiler, iree_runtime_options: [device: device]]
    inputs = inputs(frame_size, size, opts)
    stages = stages(size, opts)

    {rows, _outputs} =
      Enum.map_reduce(stages, inputs, fn {name, fun, args, tolerance}, values ->
        args = Enum.map(args, &Map.fetch!(values, &1))
        compiled = Nx.Defn.jit(fun, gpu)
        on_device = Enum.map(args, &Nx.backend_copy(&1, {NxIREE.Backend, device: device}))
        output = apply(compiled, on_device)
        ms = median_ms(fn -> apply(compiled, on_device) end, opts[:runs])
        reference = apply(Nx.Defn.jit(fun, compiler: EXLA), args)
        differing = differing(Nx.backend_transfer(output, Nx.BinaryBackend), reference, tolerance)

        kernels =
          if opts[:kernels],
            do:
              for(
                {kernel, ms} <- Bench.Kernels.rows(fun, args, device, opts[:runs]),
                do: ["  " <> kernel, ms, ""]
              ),
            else: []

        {[[name, ms, differing] | kernels], Map.put(values, name, reference)}
      end)

    rows = Enum.concat(rows)

    call = Nx.Defn.jit(&Nx.add(&1, 1), gpu)
    scalar = Nx.backend_transfer(Nx.f32(0), {NxIREE.Backend, device: device})
    apply(call, [scalar])
    call_ms = median_ms(fn -> call.(scalar) end, opts[:runs])

    chain = Nx.Defn.jit(&chain(&1, &2, &3, &4, &5, &6, stages), gpu)

    chain_args =
      Enum.map(
        chain_inputs(),
        &Nx.backend_copy(Map.fetch!(inputs, &1), {NxIREE.Backend, device: device})
      )

    apply(chain, chain_args)
    chain_ms = median_ms(fn -> apply(chain, chain_args) end, opts[:runs])

    print(
      [["Stage", "ms", "Differing"] | Enum.map(rows, &format/1)] ++
        [
          ["Sum of stages", ms(stage_sum(rows)), ""],
          ["Chain", ms(chain_ms), ""],
          ["Call", ms(call_ms), ""]
        ]
    )
  end

  defp dimensions(text) do
    [height, width] = text |> String.split("x") |> Enum.map(&String.to_integer/1)
    {height, width}
  end

  # A textured pair at frame size whose right frame sees the left one shifted, and the carried state.
  defp inputs({frame_height, frame_width}, {height, width}, opts) do
    shift = div(opts[:disparities], 4) * div(frame_width, width)

    {texture, _key} =
      Nx.Random.randint(Nx.Random.key(1), 0, 256,
        shape: {frame_height, frame_width + shift},
        type: :s32
      )

    texture = Nx.as_type(texture, :u8)
    left = texture[[.., 0..(frame_width - 1)]]
    right = texture[[.., shift..(frame_width + shift - 1)]]

    frame = fn image ->
      if opts[:packed],
        do: Packing.pack(Nx.to_binary(image), frame_width, frame_height),
        else: image
    end

    {fx, cx, cy} = {width * 0.6, width / 2, height / 2}
    k = [fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0]
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    p = [fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    camera = Rectify.camera(k, [-0.05, 0.01], identity, p)
    unmatched = Nx.broadcast(Nx.f32(-1.0), {height, width})

    %{
      left_frame: frame.(left),
      right_frame: frame.(right),
      camera: camera,
      previous: unmatched,
      missed: Nx.broadcast(Nx.f32(0.0), {height, width}),
      settings: Nx.tensor([fx, cx, cy, 0.2], type: :f32)
    }
  end

  defp chain_inputs, do: [:left_frame, :right_frame, :camera, :previous, :missed, :settings]

  # {name, function, its inputs by name, the largest difference from EXLA that counts as equal}.
  defp stages(size, opts) do
    rectify =
      if opts[:packed],
        do: &(Rectify.rectify_packed(&1, &2, size: size) |> elem(0)),
        else: &Rectify.rectify(&1, &2, size: size)

    match_opts = [disparities: opts[:disparities], paths: opts[:paths]]

    match =
      if opts[:search] == "coarse_to_fine",
        do:
          &NxStereo.sgm_coarse_to_fine(
            &1,
            &2,
            match_opts ++ [coarse_paths: opts[:coarse_paths], band: opts[:band]]
          ),
        else: &NxStereo.sgm(&1, &2, match_opts)

    [
      {:left, rectify, [:left_frame, :camera], 1},
      {:right, rectify, [:right_frame, :camera], 1},
      {:disparity, match, [:left, :right], 1.0e-3},
      {:despeckled, &Filter.despeckle/1, [:disparity], 0},
      {:filtered, &(Filter.temporal(&1, &2, &3) |> elem(0)), [:despeckled, :previous, :missed],
       1.0e-4},
      {:depth_m, &(Depth.depth(&1, &2[0], &2[3]) |> elem(1)), [:filtered, :settings], 1.0e-4},
      {:cloud, &(Depth.point_cloud(&1, &2[0], &2[1], &2[2], decimation: 4) |> elem(0)),
       [:depth_m, :settings], 1.0e-4},
      {:sharpness_sums, &Diagnostics.sharpness_sums/2, [:left, :right], 1.0e-3},
      {:row_alignment_sums, &Diagnostics.row_alignment_sums/3, [:left, :right, :filtered],
       1.0e-3},
      {:disparity_sums, &Diagnostics.disparity_sums/2, [:filtered, :previous], 1.0e-3}
    ]
  end

  # Every stage in one function, as a pipeline compiles them.
  defp chain(left_frame, right_frame, camera, previous, missed, settings, stages) do
    values = %{
      left_frame: left_frame,
      right_frame: right_frame,
      camera: camera,
      previous: previous,
      missed: missed,
      settings: settings
    }

    stages
    |> Enum.reduce(values, fn {name, fun, args, _tolerance}, values ->
      Map.put(values, name, apply(fun, Enum.map(args, &Map.fetch!(values, &1))))
    end)
    |> Map.take([:cloud, :depth_m, :sharpness_sums, :row_alignment_sums, :disparity_sums])
  end

  defp stage_sum(rows) do
    rows
    |> Enum.reject(&String.starts_with?(to_string(hd(&1)), " "))
    |> Enum.map(&Enum.at(&1, 1))
    |> Enum.sum()
  end

  def median_ms(fun, runs) do
    times = for _ <- 1..runs, do: elem(:timer.tc(fun), 0) / 1000
    times |> Enum.sort() |> Enum.at(div(runs, 2))
  end

  # Elements further than `tolerance` from the reference, relative above 1, NaN matching NaN.
  defp differing(got, want, tolerance) do
    {got, want} =
      {Nx.as_type(got, :f32), want |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type(:f32)}

    both_nan = Nx.logical_and(Nx.is_nan(got), Nx.is_nan(want))
    error = Nx.abs(Nx.subtract(got, want))
    bound = Nx.multiply(Nx.max(Nx.abs(want), 1.0), tolerance)

    Nx.greater(error, bound)
    |> Nx.logical_and(Nx.logical_not(both_nan))
    |> Nx.sum()
    |> Nx.to_number()
  end

  defp format([name, ms, differing]), do: [to_string(name), ms(ms), to_string(differing)]
  defp ms(value), do: :erlang.float_to_binary(value, decimals: 2)

  defp print(rows) do
    widths =
      rows
      |> Enum.zip()
      |> Enum.map(fn column ->
        column |> Tuple.to_list() |> Enum.map(&String.length/1) |> Enum.max()
      end)

    for row <- rows do
      row
      |> Enum.zip(widths)
      |> Enum.map_join("  ", fn {cell, width} -> String.pad_trailing(cell, width) end)
      |> IO.puts()
    end
  end
end

Bench.Pipeline.main(System.argv())
