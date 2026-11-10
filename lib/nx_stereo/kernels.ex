defmodule NxStereo.Kernels do
  @moduledoc false
  # The Vulkan kernels of the stages around the matcher, as Nx.blocks: compiled
  # by NxIREE for Vulkan, a block runs its kernels, which read and write 32-bit
  # words only; everywhere else it runs `default`, the stage's Nx code. Without
  # NxIREE, `default` runs directly.

  @spirv_dir Path.expand("../../priv/spirv", __DIR__)

  # Each kernel's GLSL source in shaders/ and its macros, for `mix nx_stereo.shaders`.
  @variants [
    rectify_words: {"rectify", ["-DWORDS=1"]},
    rectify_bytes: {"rectify", ["-DWORDS=0"]},
    column_sums: {"column_sums", []},
    despeckle: {"despeckle", []},
    temporal_filter: {"temporal_filter", []},
    depth: {"depth", []},
    point_cloud_rows: {"point_cloud_rows", []}
  ]

  @spirv (for {kernel, _source} <- @variants, into: %{} do
            path = Path.join(@spirv_dir, "#{kernel}.spv")
            @external_resource path
            {kernel, File.read!(path)}
          end)

  @group 64

  # The values the kernels compile in; other values run the Nx code.
  @despeckle_radius 3
  @max_missed 255

  def variants, do: @variants
  def despeckle_radius, do: @despeckle_radius
  def max_missed, do: @max_missed

  # `frame` rectified to `{height, width}` by the camera `vector`: the u8 image
  # of a u8 frame, or, of a frame packed as u32 words, the s32 image and its
  # packing. The kernel writes 4 pixels per word.
  def rectify(frame, vector, {height, width}, default) do
    words? = Nx.type(frame) == {:u, 32}
    {frame_height, row} = Nx.shape(frame)
    frame_width = if words?, do: row * 4, else: row
    packed = Nx.template({height, div(width, 4)}, :u32)

    outputs =
      if words?,
        do: [image: Nx.template({height, width}, :s32), packed: packed],
        else: [packed: Nx.template({height, width}, :u8)]

    template =
      case Keyword.values(outputs) do
        [image, packed] -> {image, packed}
        [image] -> image
      end

    block([frame, vector], template, default, fn ->
      kernel = %NxIREE.Vulkan.Kernel{
        spirv: Map.fetch!(@spirv, if(words?, do: :rectify_words, else: :rectify_bytes)),
        workgroups: {ceil_div(div(width, 4), @group), height, 1},
        specialization: %{
          0 => frame_width,
          1 => frame_height,
          2 => width,
          3 => height,
          4 => frame_width / width,
          5 => frame_height / height
        }
      }

      %NxIREE.Vulkan.Pipeline{
        inputs: [:frame, :camera],
        stages: [{kernel, [:frame, :camera], outputs}],
        outputs: Keyword.keys(outputs)
      }
    end)
  end

  # The despeckled f32 `disparity`; `params` holds min_support and max_diff.
  def despeckle(disparity, params, default) do
    {height, width} = shape = Nx.shape(disparity)

    block([disparity, params], Nx.template(shape, :f32), default, fn ->
      # The shader's workgroups cover 64 x 16 pixels.
      kernel = %NxIREE.Vulkan.Kernel{
        spirv: Map.fetch!(@spirv, :despeckle),
        workgroups: {ceil_div(width, 64), ceil_div(height, 16), 1},
        specialization: %{0 => width, 1 => height}
      }

      %NxIREE.Vulkan.Pipeline{
        inputs: [:disparity, :params],
        stages: [{kernel, [:disparity, :params], [output: Nx.template(shape, :f32)]}],
        outputs: [:output]
      }
    end)
  end

  # The temporal filter's step: the filtered disparity and the new missed
  # counts; `params` holds alpha, delta and hold. The kernel filters four
  # pixels a thread.
  def temporal_filter(current, previous, missed, params, default) do
    shape = Nx.shape(current)
    pixels = Tuple.product(shape)
    template = Nx.template(shape, :f32)

    block([current, previous, missed, params], {template, template}, default, fn ->
      inputs = [:current, :previous, :missed, :params]

      kernel = %NxIREE.Vulkan.Kernel{
        spirv: Map.fetch!(@spirv, :temporal_filter),
        workgroups: {ceil_div(ceil_div(pixels, 4), @group), 1, 1},
        specialization: %{0 => pixels}
      }

      %NxIREE.Vulkan.Pipeline{
        inputs: inputs,
        stages: [{kernel, inputs, [filtered: template, missed_out: template]}],
        outputs: [:filtered, :missed_out]
      }
    end)
  end

  # The u16 millimetres and f32 metres of `disparity`; `params` holds the
  # focal length and the baseline. The kernel writes two u16 per word.
  def depth(disparity, params, default) do
    shape = Nx.shape(disparity)
    pixels = Tuple.product(shape)
    outputs = [depth_mm: Nx.template(shape, :u16), depth_m: Nx.template(shape, :f32)]

    block([disparity, params], {outputs[:depth_mm], outputs[:depth_m]}, default, fn ->
      kernel = %NxIREE.Vulkan.Kernel{
        spirv: Map.fetch!(@spirv, :depth),
        workgroups: {ceil_div(ceil_div(pixels, 2), @group), 1, 1},
        specialization: %{0 => pixels}
      }

      %NxIREE.Vulkan.Pipeline{
        inputs: [:disparity, :params],
        stages: [{kernel, [:disparity, :params], outputs}],
        outputs: [:depth_mm, :depth_m]
      }
    end)
  end

  # The point grid of `depth_m` and the number of points kept, an s32; `params`
  # holds the focal length, cx, cy and the depth range.
  def point_cloud(depth_m, params, decimation, default) do
    {height, width} = Nx.shape(depth_m)
    {grid_height, grid_width} = {div(height, decimation), div(width, decimation)}
    grid = Nx.template({grid_height, grid_width, 3}, :f32)

    {grid, kept} =
      block([depth_m, params], {grid, Nx.template({}, :f32)}, default, fn ->
        rows = %NxIREE.Vulkan.Kernel{
          spirv: Map.fetch!(@spirv, :point_cloud_rows),
          workgroups: {grid_height, 1, 1},
          specialization: %{0 => width, 1 => grid_width, 2 => decimation}
        }

        %NxIREE.Vulkan.Pipeline{
          inputs: [:depth_m, :params],
          stages: [
            {rows, [:depth_m, :params],
             [grid: grid, kept_rows: Nx.template({grid_height}, :f32)]},
            {column_sums(grid_height, 1), [:kept_rows], [kept: Nx.template({}, :f32)]}
          ],
          outputs: [:grid, :kept]
        }
      end)

    {grid, Nx.as_type(kept, :s32)}
  end

  # Scalars, numbers or tensors, as the f32 vector a kernel reads them from.
  def scalars(values) do
    values |> Enum.map(&(&1 |> Nx.as_type(:f32) |> Nx.reshape({1}))) |> Nx.concatenate()
  end

  # The values `scalars/1` packed, for a default function, which sees only the block's inputs.
  def unscalars(vector, count), do: for(i <- 0..(count - 1), do: vector[i])

  # `default` takes the block's inputs as a list.
  defp block(inputs, template, default, pipeline) do
    if Code.ensure_loaded?(NxIREE.Vulkan.Pipeline),
      do: Nx.block(pipeline.(), inputs, template, spread(length(inputs), default)),
      else: default.(inputs)
  end

  # Nx.block calls its default with the struct, then each input.
  defp spread(2, default), do: fn _, a, b -> default.([a, b]) end
  defp spread(3, default), do: fn _, a, b, c -> default.([a, b, c]) end
  defp spread(4, default), do: fn _, a, b, c, d -> default.([a, b, c, d]) end

  defp column_sums(rows, columns) do
    %NxIREE.Vulkan.Kernel{
      spirv: Map.fetch!(@spirv, :column_sums),
      workgroups: {columns, 1, 1},
      specialization: %{0 => rows, 1 => columns}
    }
  end

  defp ceil_div(a, b), do: div(a + b - 1, b)
end
