defmodule NxStereo.Vulkan do
  @moduledoc false
  # The libSGM-derived compute shaders as an NxIREE.Vulkan.Pipeline. Stage
  # bindings, buffer sizes and workgroup counts follow the shaders' layouts:
  # features and byte volumes are stored in tiles of 1024 / disparities rows.

  @spirv_dir Path.expand("../../priv/spirv", __DIR__)
  @stages ~w(census cost_forward cost_reverse cost_vertical consistency median)a

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
        specialization: %{
          0 => width,
          1 => height,
          2 => config.p1,
          3 => config.p2,
          4 => config.uniqueness
        }
      }
    end

    volume = Nx.template({div(tiled_pixels * disparities, 4)}, :u32)
    map = Nx.template({pixels}, :f32)
    rows = ceil_div(height, tile)

    %NxIREE.Vulkan.Pipeline{
      inputs: [:left, :right],
      stages: [
        {kernel.(:census, {ceil_div(width, div(128, tile)), ceil_div(height, 4 * tile), 1}),
         [:left, :right], [features: Nx.template({tiled_pixels * 4}, :u32)]},
        {kernel.(:cost_forward, {rows, 1, 1}), [:features], [forward: volume, costs: volume]},
        {kernel.(:cost_reverse, {rows, 1, 1}), [:costs], [reverse: volume]},
        {kernel.(:cost_vertical, {ceil_div(width, tile), 1, 1}), [:costs, :forward, :reverse],
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
