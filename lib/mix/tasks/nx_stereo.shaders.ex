defmodule Mix.Tasks.NxStereo.Shaders do
  @shortdoc "Compiles the Vulkan shaders into priv/spirv"
  @moduledoc """
  Compiles the GLSL sources in `shaders/` into the SPIR-V binaries shipped in
  `priv/spirv`, one per stage and disparity range or band, and the kernels of
  the stages around the matcher. Requires
  `glslangValidator`; the binaries are committed, so users never run it.

      mix nx_stereo.shaders
  """

  use Mix.Task

  @stages [
    census: {"census_pair", []},
    cost_forward: {"cost_first", []},
    cost_reverse: {"cost_scan", ["-DPAIRED=1"]},
    cost_down: {"cost_line", ["-DSTEP_X=0", "-DSTEP_Y=1"]},
    cost_up: {"cost_line", ["-DSTEP_X=0", "-DSTEP_Y=-1"]},
    cost_down_select: {"cost_line", ["-DSTEP_X=0", "-DSTEP_Y=1", "-DSELECT=1"]},
    cost_down_right: {"cost_line", ["-DSTEP_X=1", "-DSTEP_Y=1", "-DPAIRED=1"]},
    cost_down_left: {"cost_line", ["-DSTEP_X=-1", "-DSTEP_Y=1"]},
    cost_up_right: {"cost_line", ["-DSTEP_X=1", "-DSTEP_Y=-1", "-DPAIRED=1"]},
    cost_up_left: {"cost_line", ["-DSTEP_X=-1", "-DSTEP_Y=-1", "-DPAIRED=1"]},
    select_8: {"select", []},
    consistency: {"consistency_projected", []},
    median: {"filter", []}
  ]

  # sgm_band/4's shaders, by band width: the costs, their column-major copy for the
  # horizontal scans, the scans ({DX, DY}, two directions with DX2, DY2) and the
  # selection.
  @band_stages [
    band_cost: {"band_cost", []},
    band_transpose_costs: {"band_transpose", []},
    band_scan_forward: {"band_scan", ["-DDX=1", "-DDY=0"]},
    band_scan_reverse: {"band_scan", ["-DDX=-1", "-DDY=0", "-DPAIRED=1"]},
    band_scan_down: {"band_scan", ["-DDX=0", "-DDY=1"]},
    band_scan_down_up: {"band_scan", ["-DDX=0", "-DDY=1", "-DDX2=0", "-DDY2=-1"]},
    band_scan_down_diagonals: {"band_scan", ["-DDX=1", "-DDY=1", "-DDX2=-1", "-DDY2=1"]},
    band_scan_up_diagonals: {"band_scan", ["-DDX=1", "-DDY=-1", "-DDX2=-1", "-DDY2=-1"]},
    band_select: {"band_select", ["-DPATHS=3"]},
    band_select_8: {"band_select", ["-DPATHS=8"]}
  ]

  @impl true
  def run(_args) do
    compiler =
      System.find_executable("glslangValidator") || Mix.raise("glslangValidator not found")

    File.mkdir_p!("priv/spirv")

    for disparities <- [64, 128], {stage, {source, flags}} <- @stages do
      output = "priv/spirv/#{stage}_d#{disparities}.spv"

      args =
        ~w(-V --target-env vulkan1.1 -S comp) ++
          [
            "-DDISPARITIES=#{disparities}",
            "-DBLOCK=8",
            "-DFEATURE_TILE=8"
          ] ++ flags ++ ["-o", output, "shaders/#{source}.comp"]

      compile!(compiler, args, source, output)
    end

    # sgm_coarse_to_fine/3's steps between its matchings take no build parameter.
    for source <- ["shrink", "band_base"] do
      output = "priv/spirv/#{source}.spv"
      args = ~w(-V --target-env vulkan1.1 -S comp -o) ++ [output, "shaders/#{source}.comp"]
      compile!(compiler, args, source, output)
    end

    # The band shaders read the census features in 8-row tiles.
    for band <- [8, 16, 32, 64], {stage, {source, flags}} <- @band_stages do
      output = "priv/spirv/#{stage}_b#{band}.spv"

      args =
        ~w(-V --target-env vulkan1.1 -S comp) ++
          ["-DBAND=#{band}", "-DFEATURE_TILE=8"] ++
          flags ++ ["-o", output, "shaders/#{source}.comp"]

      compile!(compiler, args, source, output)
    end

    # The kernels of the stages around the matcher.
    for {kernel, {source, flags}} <- NxStereo.Kernels.variants() do
      output = "priv/spirv/#{kernel}.spv"

      args =
        ~w(-V --target-env vulkan1.1 -S comp) ++ flags ++ ["-o", output, "shaders/#{source}.comp"]

      compile!(compiler, args, source, output)
    end
  end

  defp compile!(compiler, args, source, output) do
    case System.cmd(compiler, args, stderr_to_stdout: true) do
      {_, 0} -> Mix.shell().info("compiled #{output}")
      {log, _} -> Mix.raise("glslangValidator failed on #{source}:\n" <> log)
    end
  end
end
