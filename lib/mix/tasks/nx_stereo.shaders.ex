defmodule Mix.Tasks.NxStereo.Shaders do
  @shortdoc "Compiles the Vulkan shaders into shaders/spirv"
  @moduledoc """
  Compiles the GLSL sources in `shaders/` into the SPIR-V binaries shipped in
  `shaders/spirv`: each variant `NxStereo.Vulkan` and `NxStereo.Kernels` list,
  a source with its macros. Requires `glslangValidator`; the binaries are
  committed, so users never run it.

      mix nx_stereo.shaders
  """

  use Mix.Task

  @impl true
  def run(_args) do
    compiler =
      System.find_executable("glslangValidator") || Mix.raise("glslangValidator not found")

    File.mkdir_p!("shaders/spirv")

    for {name, {source, macros}} <- NxStereo.Vulkan.variants() ++ NxStereo.Kernels.variants() do
      output = "shaders/spirv/#{name}.spv"

      args =
        ~w(-V --target-env vulkan1.1 -S comp) ++
          macros ++ ["-o", output, "shaders/#{source}.comp"]

      case System.cmd(compiler, args, stderr_to_stdout: true) do
        {_, 0} -> Mix.shell().info("compiled #{output}")
        {log, _} -> Mix.raise("glslangValidator failed on #{source}:\n" <> log)
      end
    end
  end
end
