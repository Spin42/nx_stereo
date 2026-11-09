defmodule Mix.Tasks.NxStereo.Shaders do
  @shortdoc "Compiles the Vulkan shaders into priv/spirv"
  @moduledoc """
  Compiles the GLSL sources in `shaders/` into the SPIR-V binaries shipped in
  `priv/spirv`, one per stage and disparity range. Requires
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
    cost_down_right: {"cost_line", ["-DSTEP_X=1", "-DSTEP_Y=1", "-DPAIRED=1"]},
    cost_down_left: {"cost_line", ["-DSTEP_X=-1", "-DSTEP_Y=1"]},
    cost_up_right: {"cost_line", ["-DSTEP_X=1", "-DSTEP_Y=-1", "-DPAIRED=1"]},
    cost_up_left: {"cost_line", ["-DSTEP_X=-1", "-DSTEP_Y=-1", "-DPAIRED=1"]},
    select: {"select", []},
    select_8: {"select", ["-DPATHS=8"]},
    consistency: {"consistency_projected", []},
    median: {"filter", []}
  ]

  # sgm_band/4's shaders, by band width: a scan per direction ({DX, DY}) and the selection.
  @band_stages [
    band_scan_forward: {"band_scan", ["-DDX=1", "-DDY=0"]},
    band_scan_reverse: {"band_scan", ["-DDX=-1", "-DDY=0"]},
    band_scan_down: {"band_scan", ["-DDX=0", "-DDY=1"]},
    band_scan_up: {"band_scan", ["-DDX=0", "-DDY=-1"]},
    band_scan_down_right: {"band_scan", ["-DDX=1", "-DDY=1"]},
    band_scan_down_left: {"band_scan", ["-DDX=-1", "-DDY=1"]},
    band_scan_up_right: {"band_scan", ["-DDX=1", "-DDY=-1"]},
    band_scan_up_left: {"band_scan", ["-DDX=-1", "-DDY=-1"]},
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
            "-DFEATURE_TILE=#{div(1024, disparities)}"
          ] ++ flags ++ ["-o", output, "shaders/#{source}.comp"]

      compile!(compiler, args, source, output)
    end

    # The band shaders read the census features in the 128-disparity tiles.
    for band <- [8, 16, 32, 64], {stage, {source, flags}} <- @band_stages do
      output = "priv/spirv/#{stage}_b#{band}.spv"

      args =
        ~w(-V --target-env vulkan1.1 -S comp) ++
          ["-DBAND=#{band}", "-DFEATURE_TILE=8"] ++
          flags ++ ["-o", output, "shaders/#{source}.comp"]

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
