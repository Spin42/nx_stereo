defmodule NxStereo.Vulkan do
  @moduledoc false
  # The libSGM-derived compute shaders as an NxIREE.Vulkan.Pipeline. Stage
  # bindings, buffer sizes and workgroup counts follow the shaders' layouts:
  # features and byte volumes are stored in tiles of 1024 / disparities rows.

  @spirv_dir Path.expand("../../priv/spirv", __DIR__)
  @stages ~w(census cost_forward cost_reverse cost_vertical cost_vertical_8 cost_up
              cost_down_right cost_down_left cost_up_right cost_up_left consistency median)a
  # Scans added for 8 paths, each over its own byte volume
  @lines [
    cost_up: :up,
    cost_down_right: :down_right,
    cost_down_left: :down_left,
    cost_up_right: :up_right,
    cost_up_left: :up_left
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
    lines =
      if config.paths == 8 do
        for {stage, name} <- @lines do
          count = if stage == :cost_up, do: width, else: width + height - 1
          {kernel.(stage, {ceil_div(count, tile), 1, 1}), [:params, :costs], [{name, volume}]}
        end
      else
        []
      end

    {vertical, stored} =
      if config.paths == 8,
        do: {:cost_vertical_8, [:forward, :reverse | Keyword.values(@lines)]},
        else: {:cost_vertical, [:forward, :reverse]}

    %NxIREE.Vulkan.Pipeline{
      inputs: [:left, :right, :params],
      stages:
        [
          {kernel.(:census, {ceil_div(width, div(128, tile)), ceil_div(height, 4 * tile), 1}),
           [:left, :right], [features: Nx.template({tiled_pixels * 4}, :u32)]},
          {kernel.(:cost_forward, {rows, 1, 1}), [:params, :features],
           [forward: volume, costs: volume]},
          {kernel.(:cost_reverse, {rows, 1, 1}), [:params, :costs], [reverse: volume]}
        ] ++
          lines ++
          [
            {kernel.(vertical, {ceil_div(width, tile), 1, 1}), [:params, :costs | stored],
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
