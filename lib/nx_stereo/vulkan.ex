defmodule NxStereo.Vulkan do
  @moduledoc false
  # The libSGM-derived compute shaders as an NxIREE.Vulkan.Pipeline. Stage
  # bindings, buffer sizes and workgroup counts follow the shaders' layouts:
  # features and byte volumes are stored in tiles of 1024 / disparities rows.

  @spirv_dir Path.expand("../../priv/spirv", __DIR__)
  @stages ~w(census cost_forward cost_reverse cost_down cost_up cost_down_right
              cost_down_left cost_up_right cost_up_left select select_8 consistency median)a
  # The 8-path scans after the horizontal ones, in order: a paired scan reads
  # its partner's volume and writes their sum (each path byte stays below
  # 128), so the selection reads four volumes, not eight.
  @lines [
    {:cost_up, :up, nil},
    {:cost_down_right, :up_down_right, :up},
    {:cost_down_left, :down_left, nil},
    {:cost_up_right, :down_left_up_right, :down_left},
    {:cost_down, :down, nil},
    {:cost_up_left, :down_up_left, :down}
  ]

  @spirv (for disparities <- [64, 128], stage <- @stages, into: %{} do
            path = Path.join(@spirv_dir, "#{stage}_d#{disparities}.spv")
            @external_resource path
            {{stage, disparities}, File.read!(path)}
          end)

  def pipeline(config) do
    if Code.ensure_loaded?(NxIREE.Vulkan.Pipeline), do: build(config)
  end

  defp build(%{height: height, width: width, disparities: disparities} = config) do
    tile = div(1024, disparities)
    tiled_pixels = width * ceil_div(height, tile) * tile
    pixels = width * height

    kernel = fn stage, workgroups ->
      %NxIREE.Vulkan.Kernel{
        spirv: Map.fetch!(@spirv, {stage, disparities}),
        workgroups: workgroups,
        specialization: %{0 => width, 1 => height}
      }
    end

    volume = Nx.template({div(tiled_pixels * disparities, 4)}, :u32)
    map = Nx.template({pixels}, :f32)
    rows = ceil_div(height, tile)

    # Vertical scans run one line per column, diagonal ones one per diagonal
    line = fn stage, name, partner ->
      count = if stage in [:cost_up, :cost_down], do: width, else: width + height - 1
      reads = if partner, do: [:params, :costs, partner], else: [:params, :costs]
      {kernel.(stage, {ceil_div(count, tile), 1, 1}), reads, [{name, volume}]}
    end

    {lines, select, summed} =
      if config.paths == 8,
        do:
          {for({stage, name, partner} <- @lines, do: line.(stage, name, partner)), :select_8,
           [:forward_reverse, :up_down_right, :down_left_up_right, :down_up_left]},
        else: {[line.(:cost_down, :down, nil)], :select, [:forward_reverse, :down]}

    %NxIREE.Vulkan.Pipeline{
      inputs: [:left, :right, :params],
      stages:
        [
          {kernel.(:census, {ceil_div(width, div(128, tile)), ceil_div(height, 4 * tile), 1}),
           [:left, :right], [features: Nx.template({tiled_pixels * 4}, :u32)]},
          {kernel.(:cost_forward, {rows, 1, 1}), [:params, :features],
           [forward: volume, costs: volume]},
          {kernel.(:cost_reverse, {rows, 1, 1}), [:params, :costs, :forward],
           [forward_reverse: volume]}
        ] ++
          lines ++
          [
            {kernel.(select, {width, rows, 1}), [:params | summed],
             [disparity: map, winners: Nx.template({pixels}, :u32)]},
            {kernel.(:consistency, {height, 1, 1}), [:disparity, :winners], [consistent: map]},
            {kernel.(:median, {ceil_div(pixels, 128), 1, 1}), [:consistent],
             [output: Nx.template({height, width}, :f32)]}
          ],
      outputs: [:output]
    }
  end

  defp ceil_div(a, b), do: div(a + b - 1, b)
end
