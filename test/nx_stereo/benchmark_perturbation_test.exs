Code.require_file("../../bench/support/perturbation.exs", __DIR__)

defmodule NxStereo.BenchmarkPerturbationTest do
  use ExUnit.Case, async: true

  test "clean preserves both original images" do
    left = Nx.tensor([[0, 127, 255]], type: :u8)
    right = Nx.tensor([[255, 127, 0]], type: :u8)

    assert Bench.Perturbation.apply(left, right, "clean") == {left, right}
  end

  test "noise is repeatable and independent between eyes" do
    image = Nx.broadcast(Nx.u8(128), {64, 64})
    {left, right} = Bench.Perturbation.apply(image, image, "noise8")
    {again_left, again_right} = Bench.Perturbation.apply(image, image, "noise8")

    assert Nx.to_binary(left) == Nx.to_binary(again_left)
    assert Nx.to_binary(right) == Nx.to_binary(again_right)
    refute Nx.to_binary(left) == Nx.to_binary(right)
    assert Nx.shape(left) == {64, 64}
    assert Nx.type(left) == {:u, 8}

    for noisy <- [left, right] do
      assert_in_delta noisy |> Nx.mean() |> Nx.to_number(), 128, 1
      assert_in_delta noisy |> Nx.variance() |> Nx.to_number(), 64, 5
    end
  end

  test "right exposure rounds and clips without changing the left image" do
    image = Nx.tensor([[0, 10, 50, 100, 200, 255]], type: :u8)

    {left, dark} = Bench.Perturbation.apply(image, image, "gain0.8")
    {bright_left, bright} = Bench.Perturbation.apply(image, image, "gain1.2")

    assert left == image
    assert bright_left == image
    assert Nx.to_flat_list(dark) == [8, 16, 48, 88, 168, 212]
    assert Nx.to_flat_list(bright) == [0, 4, 52, 112, 232, 255]
  end

  test "right blur uses zero padding and preserves image dimensions" do
    image = Nx.broadcast(Nx.u8(90), {3, 3})

    {left, blurred} = Bench.Perturbation.apply(image, image, "blur_right")

    assert left == image
    assert Nx.shape(blurred) == {3, 3}
    assert Nx.type(blurred) == {:u, 8}
    assert Nx.to_flat_list(blurred) == [40, 60, 40, 60, 90, 60, 40, 60, 40]
  end

  test "unknown perturbations are rejected" do
    image = Nx.tensor([[0]], type: :u8)

    assert_raise ArgumentError, ~r/unknown --perturbation/, fn ->
      Bench.Perturbation.apply(image, image, "unknown")
    end
  end
end
