defmodule NxStereo.Vulkan do
  @moduledoc false
  # The libSGM-derived compute shaders as an NxIREE.Vulkan.Pipeline. Stage
  # bindings, buffer sizes and workgroup counts follow the shaders' layouts:
  # features and byte volumes are stored in tiles of 8 rows.

  import NxStereo.Kernels, only: [ceil_div: 2]

  # The shaders by stage: their GLSL source in shaders/ and its macros. The census,
  # consistency, median and coarse-to-fine shaders serve every disparity range and
  # band; the matcher's scans and selection exist per disparity range, sgm_band/4's
  # per band width.
  @shared [
    census: {"census_pair", []},
    census_dense: {"census_dense_pair", []},
    census_blend: {"census_dense_pair", ["-DBLEND=1"]},
    consistency: {"consistency_projected", []},
    median: {"filter", []},
    shrink: {"shrink", []},
    band_base: {"band_base", []}
  ]
  @by_disparities [
    cost_forward: {"cost_first", []},
    cost_reverse: {"cost_scan", ["-DPAIRED=1"]},
    cost_down: {"cost_line", ["-DSTEP_X=0", "-DSTEP_Y=1"]},
    cost_up: {"cost_line", ["-DSTEP_X=0", "-DSTEP_Y=-1"]},
    cost_down_select: {"cost_line", ["-DSTEP_X=0", "-DSTEP_Y=1", "-DSELECT=1"]},
    cost_down_right: {"cost_line", ["-DSTEP_X=1", "-DSTEP_Y=1", "-DPAIRED=1"]},
    cost_down_left: {"cost_line", ["-DSTEP_X=-1", "-DSTEP_Y=1"]},
    cost_up_right: {"cost_line", ["-DSTEP_X=1", "-DSTEP_Y=-1", "-DPAIRED=1"]},
    cost_up_left: {"cost_line", ["-DSTEP_X=-1", "-DSTEP_Y=-1", "-DPAIRED=1"]},
    select_8: {"select", []}
  ]
  # The band's costs, their column-major copy for the horizontal scans, the scans
  # ({DX, DY}, two directions with DX2, DY2) and the selection.
  @by_band [
    band_cost: {"band_cost", []},
    band_transpose_costs: {"band_transpose", []},
    band_scan_horizontal: {"band_scan", ["-DDX=1", "-DDY=0"]},
    band_scan_down: {"band_scan", ["-DDX=0", "-DDY=1"]},
    band_select: {"band_select", ["-DPATHS=3"]},
    band_select_8: {"band_select", ["-DPATHS=8"]}
  ]

  @narrow_scans [
    band_scan_down_up: {"band_scan", ["-DDX=0", "-DDY=1", "-DDX2=0", "-DDY2=-1"]},
    band_scan_down_diagonals: {"band_scan", ["-DDX=1", "-DDY=1", "-DDX2=-1", "-DDY2=1"]},
    band_scan_up_diagonals: {"band_scan", ["-DDX=1", "-DDY=-1", "-DDX2=-1", "-DDY2=-1"]}
  ]
  @wide_scans [
    band_aggregate_vertical: {"band_scan_aggregate", ["-DDX=0", "-DDY=1", "-DFIRST=1"]},
    band_aggregate_diagonal: {"band_scan_aggregate", ["-DDX=1", "-DDY=1"]},
    band_aggregate_antidiagonal: {"band_scan_aggregate", ["-DDX=-1", "-DDY=1"]}
  ]

  # {key, SPIR-V file name, {source, macros}}
  @variants [
              {{:cost_horizontal, 128}, "cost_horizontal_d128",
               {"cost_first", ["-DDISPARITIES=128", "-DFUSED=1"]}}
            ] ++
              for({stage, source} <- @shared, do: {stage, "#{stage}", source}) ++
              (for disparities <- [64, 128], {stage, {source, flags}} <- @by_disparities do
                 macros = ["-DDISPARITIES=#{disparities}" | flags]
                 {{stage, disparities}, "#{stage}_d#{disparities}", {source, macros}}
               end) ++
              (for band <- [8, 16, 32, 64],
                   {stage, {source, flags}} <-
                     @by_band ++ if(band >= 32, do: @wide_scans, else: @narrow_scans) do
                 source =
                   if source == "band_scan" and band >= 32, do: "band_scan_lanes", else: source

                 flags =
                   if stage == :band_select_8 and band >= 32,
                     do: ["-DAGGREGATED=1" | flags],
                     else: flags

                 {{stage, band}, "#{stage}_b#{band}", {source, ["-DBAND=#{band}" | flags]}}
               end)

  @spirv_dir Path.expand("../../shaders/spirv", __DIR__)
  @spirv (for {key, file, _source} <- @variants, into: %{} do
            path = Path.join(@spirv_dir, "#{file}.spv")
            @external_resource path
            {key, File.read!(path)}
          end)

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

  # Each shader's SPIR-V file name, GLSL source and macros, for `mix nx_stereo.shaders`.
  def variants, do: for({_key, file, source} <- @variants, do: {file, source})

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

  # Both s32 images of {height, width} shrunk 2x, each pixel the rounded mean of a 2x2 block
  def shrink_pipeline(height, width) do
    {half_height, half_width} = {div(height, 2), div(width, 2)}
    half = Nx.template({half_height, half_width}, :s32)

    kernel = %NxIREE.Vulkan.Kernel{
      spirv: Map.fetch!(@spirv, :shrink),
      workgroups: {ceil_div(half_height * half_width, 64), 1, 1},
      specialization: %{0 => half_width, 1 => half_height}
    }

    %NxIREE.Vulkan.Pipeline{
      inputs: [:left, :right],
      stages: [{kernel, [:left, :right], [left_half: half, right_half: half]}],
      outputs: [:left_half, :right_half]
    }
  end

  # The s32 band bases of a {height, width} pair from its half-size f32 disparity
  def band_base_pipeline(height, width, margin, max_base) do
    kernel = %NxIREE.Vulkan.Kernel{
      spirv: Map.fetch!(@spirv, :band_base),
      workgroups: {ceil_div(div(height, 2) * div(width, 2), 64), 1, 1},
      specialization: %{0 => width, 1 => height, 2 => margin, 3 => max_base}
    }

    %NxIREE.Vulkan.Pipeline{
      inputs: [:coarse],
      stages: [{kernel, [:coarse], [base: Nx.template({height, width}, :s32)]}],
      outputs: [:base]
    }
  end

  # The census of both images, in tiles of 8 rows.
  defp census_stage(height, width, census \\ :sparse_9x7) do
    stage =
      case census do
        :sparse_9x7 -> :census
        :dense_9x5 -> :census_dense
        :dense_9x5_blend -> :census_blend
      end

    tiled_pixels = width * ceil_div(height, 8) * 8

    kernel = %NxIREE.Vulkan.Kernel{
      spirv: Map.fetch!(@spirv, stage),
      workgroups: {ceil_div(width, div(128, 8)), ceil_div(height, 4 * 8), 1},
      specialization: %{0 => width, 1 => height}
    }

    {kernel, [:left, :right], [features: Nx.template({tiled_pixels * 4}, :u32)]}
  end

  # The left-right check of `:disparity` and `:winners`, then the 3x3 median, into `:output`.
  defp check_and_median_stages(height, width) do
    pixels = width * height
    specialization = %{0 => width, 1 => height}

    kernel =
      &%NxIREE.Vulkan.Kernel{
        spirv: Map.fetch!(@spirv, &1),
        workgroups: &2,
        specialization: specialization
      }

    [
      {kernel.(:consistency, {height, 1, 1}), [:disparity, :winners],
       [consistent: Nx.template({pixels}, :f32)]},
      {kernel.(:median, {ceil_div(pixels, 128), 1, 1}), [:consistent],
       [output: Nx.template({height, width}, :f32)]}
    ]
  end

  # Census in 8-row tiles, the band's costs once,
  # a thread per path line, a lane per word of a pixel's band for the selection;
  # consistency and median as pipeline/1. The horizontal scans run over a column-major
  # copy of the costs, adding the reverse path to the forward path in place, and the
  # selection reads their column-major sum. Wide eight-path bands accumulate the
  # remaining opposite pairs into ten-bit sums to bound scratch storage.
  def band_pipeline(
        %{height: height, width: width, disparities: disparities, band: band} = config
      ) do
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

    # Wide bands share a line across one subgroup lane per packed word.
    lines_per_group = if band >= 32, do: div(64, div(band, 4)), else: 64
    # A wide image's diagonal scans fold two diagonals into a line.
    diagonals = if width >= height, do: width, else: width + height - 1
    lines = %{rows: height, columns: width, diagonals: diagonals}
    tiles = {ceil_div(width, 8), ceil_div(height, 8), 1}

    scan = fn stage, kind, directions, inputs ->
      volume = Nx.template({div(directions * pixels * band, 4)}, :u32)

      {kernel.(
         Map.fetch!(@spirv, {stage, band}),
         {directions * ceil_div(lines[kind], lines_per_group), 1, 1},
         %{}
       ), inputs, [{stage, volume}]}
    end

    horizontal_stages = [
      {kernel.(Map.fetch!(@spirv, {:band_transpose_costs, band}), tiles, %{}), [:costs, :base],
       [columns: Nx.template({pixels * (cost_words + 1)}, :u32)]},
      scan.(:band_scan_horizontal, :rows, 1, [:params, :columns])
    ]

    {other_stages, select_inputs} =
      if config.paths == 8 and band >= 32 do
        aggregate = Nx.template({div(pixels * band * 10, 32)}, :u32)
        kinds = [:columns, :diagonals, :diagonals]

        {stages, last} =
          Enum.zip(@wide_scans, kinds)
          |> Enum.map_reduce(:band_scan_horizontal, fn {{stage, _source}, kind}, previous ->
            step =
              {kernel.(
                 Map.fetch!(@spirv, {stage, band}),
                 {ceil_div(lines[kind], lines_per_group), 1, 1},
                 %{}
               ), [:params, :costs, :base, previous], [{stage, aggregate}]}

            {step, stage}
          end)

        {stages, [last]}
      else
        other_scans = Map.fetch!(@band_scans, config.paths)

        stages =
          for {stage, {kind, directions}} <- other_scans,
              do: scan.(stage, kind, directions, [:params, :costs, :base])

        {stages, [:band_scan_horizontal | Keyword.keys(other_scans)]}
      end

    scan_stages = horizontal_stages ++ other_stages

    select = if config.paths == 8, do: :band_select_8, else: :band_select

    %NxIREE.Vulkan.Pipeline{
      inputs: [:left, :right, :params, :base],
      stages:
        [
          census_stage(height, width, Map.get(config, :census, :sparse_9x7)),
          {kernel.(
             Map.fetch!(@spirv, {:band_cost, band}),
             {ceil_div(width, 64), ceil_div(height, 4), 1},
             %{2 => disparities}
           ), [:features, :base], [costs: Nx.template({pixels * cost_words}, :u32)]}
        ] ++
          scan_stages ++
          [
            {kernel.(
               Map.fetch!(@spirv, {select, band}),
               linear_workgroups(ceil_div(pixels * div(band, 4), 64)),
               %{2 => disparities}
             ), [:params, :base | select_inputs],
             [disparity: map, winners: Nx.template({pixels}, :u32)]}
          ] ++ check_and_median_stages(height, width),
      outputs: [:output]
    }
  end

  @max_workgroups 65_535
  defp linear_workgroups(count) do
    rows = ceil_div(count, @max_workgroups)
    {ceil_div(count, rows), rows, 1}
  end

  def pipeline(%{height: height, width: width, disparities: disparities} = config) do
    tile = 8
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

    selected = [disparity: map, winners: Nx.template({pixels}, :u32)]

    # With 3 paths the vertical scan selects as it goes.
    aggregation =
      if config.paths == 8 do
        for({stage, name, partner} <- @lines, do: line.(stage, name, partner)) ++
          [
            {kernel.(:select_8, {width, rows, 1}),
             [:params, :forward_reverse, :up_down_right, :down_left_up_right, :down_up_left],
             selected}
          ]
      else
        [
          {kernel.(:cost_down_select, {ceil_div(width, tile), 1, 1}),
           [:params, :costs, :forward_reverse], selected}
        ]
      end

    # Fusion is retained only where complete-chain measurements show a repeatable gain.
    horizontal =
      if disparities == 128 and config.paths == 3 do
        [
          {kernel.(:cost_horizontal, {rows, 1, 1}), [:params, :features],
           [forward_reverse: volume, costs: volume]}
        ]
      else
        [
          {kernel.(:cost_forward, {rows, 1, 1}), [:params, :features],
           [forward: volume, costs: volume]},
          {kernel.(:cost_reverse, {rows, 1, 1}), [:params, :costs, :forward],
           [forward_reverse: volume]}
        ]
      end

    %NxIREE.Vulkan.Pipeline{
      inputs: [:left, :right, :params],
      stages:
        [census_stage(height, width)] ++
          horizontal ++
          aggregation ++ check_and_median_stages(height, width),
      outputs: [:output]
    }
  end
end
