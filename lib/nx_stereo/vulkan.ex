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

  @band_stages ~w(band_cost band_transpose_costs band_scan_forward
                   band_scan_reverse band_scan_down band_scan_down_up band_scan_down_diagonals
                   band_scan_up_diagonals band_select band_select_8)a

  @band_spirv (for band <- [8, 16, 32, 64], stage <- @band_stages, into: %{} do
                 path = Path.join(@spirv_dir, "#{stage}_b#{band}.spv")
                 @external_resource path
                 {{stage, band}, File.read!(path)}
               end)

  # sgm_coarse_to_fine/3's steps between its two matchings
  @coarse_spirv (for stage <- ~w(shrink band_base)a, into: %{} do
                   path = Path.join(@spirv_dir, "#{stage}.spv")
                   @external_resource path
                   {stage, File.read!(path)}
                 end)

  # sgm_band/4's scans besides the horizontal ones, by path count: their lines
  # and how many directions each dispatch scans, two writing both volumes in turn.
  @band_scans %{
    3 => [band_scan_down: {:columns, 1}],
    8 => [
      band_scan_down_up: {:columns, 2},
      band_scan_down_diagonals: {:diagonals, 2},
      band_scan_up_diagonals: {:diagonals, 2}
    ]
  }

  @doc false
  # Both s32 images of {height, width} shrunk 2x, each pixel the rounded mean of a 2x2 block
  def shrink_pipeline(height, width) do
    if Code.ensure_loaded?(NxIREE.Vulkan.Pipeline) do
      {half_height, half_width} = {div(height, 2), div(width, 2)}
      half = Nx.template({half_height, half_width}, :s32)

      kernel = %NxIREE.Vulkan.Kernel{
        spirv: @coarse_spirv.shrink,
        workgroups: {ceil_div(half_height * half_width, 64), 1, 1},
        specialization: %{0 => half_width, 1 => half_height}
      }

      %NxIREE.Vulkan.Pipeline{
        inputs: [:left, :right],
        stages: [{kernel, [:left, :right], [left_half: half, right_half: half]}],
        outputs: [:left_half, :right_half]
      }
    end
  end

  @doc false
  # The s32 band bases of a {height, width} pair from its half-size f32 disparity
  def band_base_pipeline(height, width, margin, max_base) do
    if Code.ensure_loaded?(NxIREE.Vulkan.Pipeline) do
      kernel = %NxIREE.Vulkan.Kernel{
        spirv: @coarse_spirv.band_base,
        workgroups: {ceil_div(div(height, 2) * div(width, 2), 64), 1, 1},
        specialization: %{0 => width, 1 => height, 2 => margin, 3 => max_base}
      }

      %NxIREE.Vulkan.Pipeline{
        inputs: [:coarse],
        stages: [{kernel, [:coarse], [base: Nx.template({height, width}, :s32)]}],
        outputs: [:base]
      }
    end
  end

  def pipeline(config) do
    if Code.ensure_loaded?(NxIREE.Vulkan.Pipeline), do: build(config)
  end

  def band_pipeline(config) do
    if Code.ensure_loaded?(NxIREE.Vulkan.Pipeline), do: build_band(config)
  end

  # Census in the 128-disparity tiles the band shaders read, the band's costs once,
  # a thread per path line, a lane per word of a pixel's band for the selection;
  # consistency and median as build/1. The horizontal scans run over a column-major
  # copy of the costs, the reverse one adding the forward one's path bytes, and the
  # selection reads their column-major sum.
  defp build_band(%{height: height, width: width, disparities: disparities, band: band} = config) do
    tile = 8
    tiled_pixels = width * ceil_div(height, tile) * tile
    pixels = width * height
    # 6-bit costs, as band_common.glsl packs them.
    cost_words = ceil_div(band * 6, 32)
    map = Nx.template({pixels}, :f32)

    kernel = fn spirv, workgroups, specialization ->
      %NxIREE.Vulkan.Kernel{
        spirv: spirv,
        workgroups: workgroups,
        specialization: Map.merge(%{0 => width, 1 => height}, specialization)
      }
    end

    lines = %{rows: height, columns: width, diagonals: width + height - 1}
    tiles = {ceil_div(width, 8), ceil_div(height, 8), 1}

    scan = fn stage, kind, directions, inputs ->
      volume = Nx.template({div(directions * pixels * band, 4)}, :u32)

      {kernel.(
         Map.fetch!(@band_spirv, {stage, band}),
         {directions * ceil_div(lines[kind], 64), 1, 1},
         %{}
       ), inputs, [{stage, volume}]}
    end

    horizontal_stages = [
      {kernel.(Map.fetch!(@band_spirv, {:band_transpose_costs, band}), tiles, %{}),
       [:costs, :base], [columns: Nx.template({pixels * (cost_words + 1)}, :u32)]},
      scan.(:band_scan_forward, :rows, 1, [:params, :columns]),
      scan.(:band_scan_reverse, :rows, 1, [:params, :columns, :band_scan_forward])
    ]

    other_scans = Map.fetch!(@band_scans, config.paths)

    scan_stages =
      horizontal_stages ++
        for {stage, {kind, directions}} <- other_scans,
            do: scan.(stage, kind, directions, [:params, :costs, :base])

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
             {ceil_div(width, 64), ceil_div(height, 4), 1},
             %{2 => disparities}
           ), [:features, :base], [costs: Nx.template({pixels * cost_words}, :u32)]}
        ] ++
          scan_stages ++
          [
            {kernel.(
               Map.fetch!(@band_spirv, {select, band}),
               {ceil_div(pixels * div(band, 4), 64), 1, 1},
               %{
                 2 => disparities
               }
             ), [:params, :base, :band_scan_reverse | Keyword.keys(other_scans)],
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
