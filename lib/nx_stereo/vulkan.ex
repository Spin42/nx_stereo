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

  @band_stages ~w(band_cost band_scan_forward band_scan_reverse band_scan_down band_scan_up
                   band_scan_down_right band_scan_down_left band_scan_up_right
                   band_scan_up_left band_select band_select_8)a

  @band_spirv (for band <- [8, 16, 32, 64], stage <- @band_stages, into: %{} do
                 path = Path.join(@spirv_dir, "#{stage}_b#{band}.spv")
                 @external_resource path
                 {{stage, band}, File.read!(path)}
               end)

  # sgm_band/4's scans, by direction, and how many lines each runs
  @band_scans [
    band_scan_forward: :rows,
    band_scan_reverse: :rows,
    band_scan_down: :columns,
    band_scan_up: :columns,
    band_scan_down_right: :diagonals,
    band_scan_down_left: :diagonals,
    band_scan_up_right: :diagonals,
    band_scan_up_left: :diagonals
  ]

  def pipeline(config) do
    if Code.ensure_loaded?(NxIREE.Vulkan.Pipeline), do: build(config)
  end

  def band_pipeline(config) do
    if Code.ensure_loaded?(NxIREE.Vulkan.Pipeline), do: build_band(config)
  end

  # Census in the 128-disparity tiles the band shaders read, the band's costs once,
  # a thread per path line, a thread per pixel for the selection; consistency and
  # median as build/1.
  defp build_band(%{height: height, width: width, disparities: disparities, band: band} = config) do
    tile = 8
    tiled_pixels = width * ceil_div(height, tile) * tile
    pixels = width * height
    path = Nx.template({div(pixels * band, 4)}, :u32)
    map = Nx.template({pixels}, :f32)

    kernel = fn spirv, workgroups, specialization ->
      %NxIREE.Vulkan.Kernel{
        spirv: spirv,
        workgroups: workgroups,
        specialization: Map.merge(%{0 => width, 1 => height}, specialization)
      }
    end

    scans = if config.paths == 8, do: @band_scans, else: Enum.take(@band_scans, 3)

    lines = %{rows: height, columns: width, diagonals: width + height - 1}

    scan_stages =
      for {stage, kind} <- scans do
        {kernel.(
           Map.fetch!(@band_spirv, {stage, band}),
           {ceil_div(lines[kind], 64), 1, 1},
           %{}
         ), [:params, :costs, :base], [{stage, path}]}
      end

    select = if config.paths == 8, do: :band_select_8, else: :band_select

    %NxIREE.Vulkan.Pipeline{
      inputs: [:left, :right, :params, :base],
      stages:
        [
          {kernel.(
             Map.fetch!(@spirv, {:census, 128}),
             {ceil_div(width, div(128, tile)), ceil_div(height, 4 * tile), 1},
             %{}
           ), [:left, :right], [features: Nx.template({tiled_pixels * 4}, :u32)]},
          {kernel.(
             Map.fetch!(@band_spirv, {:band_cost, band}),
             {ceil_div(width, div(64, tile)), ceil_div(height, tile), 1},
             %{}
           ), [:features, :base], [costs: path]}
        ] ++
          scan_stages ++
          [
            {kernel.(Map.fetch!(@band_spirv, {select, band}), {ceil_div(pixels, 64), 1, 1}, %{
               2 => disparities
             }), [:params, :base | Keyword.keys(scans)],
             [disparity: map, winners: Nx.template({pixels}, :u32)]},
            {kernel.(Map.fetch!(@spirv, {:consistency, 128}), {height, 1, 1}, %{}),
             [:disparity, :winners], [consistent: map]},
            {kernel.(Map.fetch!(@spirv, {:median, 128}), {ceil_div(pixels, 128), 1, 1}, %{}),
             [:consistent], [output: Nx.template({height, width}, :f32)]}
          ],
      outputs: [:output]
    }
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
