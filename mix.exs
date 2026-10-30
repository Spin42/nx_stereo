defmodule NxStereo.MixProject do
  use Mix.Project

  @version "0.1.0-dev"

  def project do
    [
      app: :nx_stereo,
      version: @version,
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "Semi-global stereo matching for Nx, with Vulkan shaders on NxIREE",
      package: [
        licenses: ["Apache-2.0"],
        files: ~w(lib priv shaders mix.exs README.md LICENSE OPENCV-LICENSE)
      ]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp deps do
    [
      {:nx, "~> 1.0"},
      {:nx_iree, github: "Spin42/nx_iree", optional: true},
      {:exla, "~> 1.0", only: [:dev, :test]}
    ]
  end
end
