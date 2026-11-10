defmodule NxStereo.MixProject do
  use Mix.Project

  @version "0.1.0-dev"

  def project do
    [
      app: :nx_stereo,
      version: @version,
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      compilers: compilers(),
      make_targets: ["all"],
      make_clean: ["clean"],
      deps: deps(),
      description: "Stereo depth for Nx: Vulkan shaders through NxIREE and a native CPU backend",
      package: [
        licenses: ["Apache-2.0"],
        files:
          ~w(lib shaders native/Cargo.toml native/Cargo.lock native/stereo_match Makefile mix.exs README.md LICENSE OPENCV-LICENSE)
      ]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  # The native backend's Port is built when cargo is installed, unless
  # NX_STEREO_NATIVE=false; without it, only the Nx backend runs.
  defp compilers do
    if native?(), do: [:elixir_make] ++ Mix.compilers(), else: Mix.compilers()
  end

  defp native? do
    System.get_env("NX_STEREO_NATIVE") != "false" and System.find_executable("cargo") != nil
  end

  defp deps do
    [
      {:nx, "~> 1.0"},
      {:nx_iree, github: "Spin42/nx_iree", optional: true},
      {:elixir_make, "~> 0.7", runtime: false},
      {:exla, "~> 1.0", only: [:dev, :test]}
    ]
  end
end
