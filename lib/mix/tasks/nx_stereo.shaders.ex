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
    cost_forward: {"cost_first", ["-DFUSE_VERTICAL=0"]},
    cost_reverse: {"cost_scan", ["-DFUSE_VERTICAL=0"]},
    cost_vertical: {"cost_scan", ["-DFUSE_VERTICAL=1"]},
    cost_vertical_8: {"cost_scan", ["-DFUSE_VERTICAL=1", "-DPATHS=8"]},
    cost_up: {"cost_line", ["-DSTEP_X=0", "-DSTEP_Y=-1"]},
    cost_down_right: {"cost_line", ["-DSTEP_X=1", "-DSTEP_Y=1"]},
    cost_down_left: {"cost_line", ["-DSTEP_X=-1", "-DSTEP_Y=1"]},
    cost_up_right: {"cost_line", ["-DSTEP_X=1", "-DSTEP_Y=-1"]},
    cost_up_left: {"cost_line", ["-DSTEP_X=-1", "-DSTEP_Y=-1"]},
    consistency: {"consistency_projected", []},
    median: {"filter", []}
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

      case System.cmd(compiler, args, stderr_to_stdout: true) do
        {_, 0} -> Mix.shell().info("compiled #{output}")
        {log, _} -> Mix.raise("glslangValidator failed on #{source}:\n" <> log)
      end
    end
  end
end
