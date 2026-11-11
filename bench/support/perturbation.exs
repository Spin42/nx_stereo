defmodule Bench.Perturbation do
  @moduledoc false
  @conditions ~w(clean noise2 noise4 noise8 gain0.8 gain1.2 blur_right)
  @noise %{"noise2" => 2.0, "noise4" => 4.0, "noise8" => 8.0}

  def validate!(condition) when condition in @conditions, do: :ok
  def validate!(_condition), do: raise(ArgumentError, "unknown --perturbation")

  def apply(left, right, "clean"), do: {left, right}

  def apply(left, right, condition) when condition in ~w(noise2 noise4 noise8) do
    scale = Map.fetch!(@noise, condition)
    {noisy(left, 271, scale), noisy(right, 723, scale)}
  end

  def apply(left, right, "gain0.8"), do: {left, exposure(right, 0.8, 8.0)}
  def apply(left, right, "gain1.2"), do: {left, exposure(right, 1.2, -8.0)}
  def apply(left, right, "blur_right"), do: {left, blur(right)}
  def apply(_left, _right, condition), do: validate!(condition)

  defp noisy(image, seed, scale) do
    {noise, _key} = Nx.Random.normal(Nx.Random.key(seed), shape: Nx.shape(image))
    image |> Nx.add(Nx.multiply(noise, scale)) |> intensities()
  end

  defp exposure(image, gain, offset) do
    image |> Nx.multiply(gain) |> Nx.add(offset) |> intensities()
  end

  defp blur(image) do
    {height, width} = Nx.shape(image)

    image
    |> Nx.as_type(:f32)
    |> Nx.reshape({1, 1, height, width})
    |> Nx.conv(Nx.broadcast(1.0 / 9, {1, 1, 3, 3}), padding: :same)
    |> Nx.reshape({height, width})
    |> intensities()
  end

  defp intensities(image), do: image |> Nx.round() |> Nx.clip(0, 255) |> Nx.as_type(:u8)
end
