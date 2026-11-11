# Times each stage of a stereo depth pipeline on a Vulkan device and checks
# its results against the Nx code on EXLA: rectification, matching, the
# speckle and temporal filters, depth, the point cloud and the diagnostics,
# on a synthetic textured pair.
#
#     mix run bench/pipeline.exs [--size 544x728] [--frame 1088x1456]
#       [--search full|coarse_to_fine] [--disparities 64|128] [--paths 3|8]
#       [--coarse-paths 3|8] [--band 16] [--margin 4] [--packed] [--kernels] [--runs 20]
#       [--device URI_SUBSTRING]
#
# Each stage is compiled alone and its inputs are on the device, so a
# stage's time is a call's: its kernels, the IREE code around them and the
# call itself. "Call" is the time of a call that does nothing, the part of
# each stage's time no kernel can remove. "Chain" runs the stages as one
# NxStereo.NxSegment, as an application does: their settings as inputs, the
# configured band and margin, the outputs in one download. Times are medians,
# in milliseconds.
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

Code.require_file("support/scene.exs", __DIR__)
Code.require_file("support/options.exs", __DIR__)
Code.require_file("support/compare.exs", __DIR__)

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
  alias NxStereo.{Depth, Diagnostics, Filter, NxSegment, Rectify, Segment}

  @defaults [
    size: "544x728",
    frame: nil,
    search: "coarse_to_fine",
    disparities: 128,
    paths: 8,
    coarse_paths: 3,
    fine_census: "sparse_9x7",
    band: 16,
    margin: 4,
    packed: false,
    cloud_decimation: 4,
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
          fine_census: :string,
          band: :integer,
          margin: :integer,
          packed: :boolean,
          cloud_decimation: :integer,
          kernels: :boolean,
          runs: :integer,
          device: :string
        ]
      )

    opts = @defaults |> Keyword.merge(opts) |> Bench.Options.fine_census()
    {height, width} = size = dimensions(opts[:size])
    opts = Bench.Options.cloud_decimation(opts, size)
    frame_size = if opts[:frame], do: dimensions(opts[:frame]), else: {height * 2, width * 2}
    {:ok, devices} = NxIREE.list_devices("vulkan")

    device =
      Enum.find(devices, &String.contains?(&1.uri, opts[:device])) || raise "no Vulkan device"

    IO.puts(
      "#{device.uri}: #{width}x#{height} from #{elem(frame_size, 1)}x#{elem(frame_size, 0)} frames"
    )

    gpu = [compiler: NxIREE.Compiler, iree_runtime_options: [device: device]]
    inputs = Bench.Scene.inputs(frame_size, size, opts)
    stages = stages(size, opts)

    {rows, _outputs} =
      Enum.map_reduce(stages, inputs, fn {name, fun, args, tolerance}, values ->
        args = Enum.map(args, &Map.fetch!(values, &1))
        compiled = Nx.Defn.jit(fun, gpu)
        on_device = Enum.map(args, &Nx.backend_copy(&1, {NxIREE.Backend, device: device}))
        output = apply(compiled, on_device)
        ms = median_ms(fn -> apply(compiled, on_device) end, opts[:runs])
        reference = apply(Nx.Defn.jit(fun, compiler: EXLA), args)

        differing =
          Bench.Compare.relative(
            Nx.backend_transfer(output, Nx.BinaryBackend),
            reference,
            tolerance
          )

        kernels =
          if opts[:kernels],
            do:
              for(
                {kernel, ms} <- Bench.Kernels.rows(fun, args, device, opts[:runs]),
                do: ["  " <> kernel, ms, ""]
              ),
            else: []

        {{[name, ms, differing], kernels}, Map.put(values, name, reference)}
      end)

    stage_sum = rows |> Enum.map(fn {[_name, ms, _differing], _kernels} -> ms end) |> Enum.sum()
    rows = Enum.flat_map(rows, fn {stage, kernels} -> [stage | kernels] end)

    call = Nx.Defn.jit(&Nx.add(&1, 1), gpu)
    scalar = Nx.backend_transfer(Nx.f32(0), {NxIREE.Backend, device: device})
    apply(call, [scalar])
    call_ms = median_ms(fn -> call.(scalar) end, opts[:runs])

    chain_ms = segment_ms(inputs, size, frame_size, opts, gpu)

    print(
      [["Stage", "ms", "Differing"] | Enum.map(rows, &format/1)] ++
        [
          ["Sum of stages", ms(stage_sum), ""],
          ["Chain", ms(chain_ms), ""],
          ["Call", ms(call_ms), ""]
        ]
    )
  end

  defp dimensions(text) do
    [height, width] = text |> String.split("x") |> Enum.map(&String.to_integer/1)
    {height, width}
  end

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
            match_opts ++
              [
                coarse_paths: opts[:coarse_paths],
                band: opts[:band],
                margin: opts[:margin],
                fine_census: opts[:fine_census]
              ]
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
      {:cloud,
       &(Depth.point_cloud(&1, &2[0], &2[1], &2[2], decimation: opts[:cloud_decimation])
         |> elem(0)), [:depth_m, :settings], 1.0e-4},
      {:sharpness_sums, &Diagnostics.sharpness_sums/2, [:left, :right], 1.0e-3},
      {:row_alignment_sums, &Diagnostics.row_alignment_sums/3, [:left, :right, :filtered],
       1.0e-3},
      {:disparity_sums, &Diagnostics.disparity_sums/2, [:filtered, :previous], 1.0e-3}
    ]
  end

  # The time of a pair through every stage as one NxStereo.NxSegment, as an application
  # compiles them: the stages' settings as inputs, their outputs in one download.
  defp segment_ms(inputs, {height, width}, {frame_height, frame_width}, opts, gpu) do
    {:ok, segment} =
      Bench.Scene.segment(inputs, {height, width}, {frame_height, frame_width}, opts, gpu)

    IO.puts(
      "cloud_decimation=#{opts[:cloud_decimation]} cloud_grid=#{inspect(Enum.find(segment.layout, &(elem(&1, 0) == :cloud_grid)))}"
    )

    entries = [left_frame: inputs.left_image, right_frame: inputs.right_image]
    {:ok, _outputs, segment} = Segment.run(NxSegment, entries, segment, :infinity)
    median_ms(fn -> Segment.run(NxSegment, entries, segment, :infinity) end, opts[:runs])
  end

  def median_ms(fun, runs) do
    times = for _ <- 1..runs, do: elem(:timer.tc(fun), 0) / 1000
    times |> Enum.sort() |> Enum.at(div(runs, 2))
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
