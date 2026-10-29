defmodule NxStereo.DepthTest do
  use ExUnit.Case, async: true

  alias NxStereo.Depth

  test "depth is f·B/d, 0 without a match, and saturates in millimetres" do
    disparity = Nx.tensor([[10.0, 0.0, -1.0, 0.0001]])
    {depth_mm, depth_m} = Nx.Defn.jit(&Depth.depth/3).(disparity, 400.0, 0.06)

    assert Nx.to_list(depth_mm) == [[2400, 0, 0, 65535]]
    [[near, none, unmatched, far]] = Nx.to_list(depth_m)
    assert_in_delta near, 2.4, 1.0e-6
    assert {none, unmatched} == {0.0, 0.0}
    assert_in_delta far, 240_000.0, 1.0
  end

  test "the cloud grid unprojects every decimation-th pixel" do
    depth_m = Nx.broadcast(Nx.tensor(2.0), {48, 64})
    grid = Nx.Defn.jit(&Depth.cloud_grid(&1, &2, &3, &4, 4)).(depth_m, 100.0, 32.0, 24.0)

    assert Nx.shape(grid) == {12, 16, 3}
    [x, y, z] = Nx.to_list(grid[1][1])
    assert_in_delta x, (4 - 32) * 2.0 / 100, 1.0e-6
    assert_in_delta y, (4 - 24) * 2.0 / 100, 1.0e-6
    assert z == 2.0
  end

  test "rows and columns are sampled as OpenCV's nearest resize when the size does not divide" do
    depth_m = Nx.iota({270, 8}, axis: 0, type: :f32) |> Nx.add(1.0)
    grid = Nx.Defn.jit(&Depth.cloud_grid(&1, &2, &3, &4, 4)).(depth_m, 100.0, 4.0, 135.0)

    assert Nx.shape(grid) == {67, 2, 3}
    sampled_rows = grid[[.., 0, 2]] |> Nx.to_list() |> Enum.map(&trunc(&1 - 1))
    assert sampled_rows == for(v <- 0..66, do: floor(v * (270 / 67)))
  end

  test "pixels without depth give zero points" do
    grid =
      Nx.Defn.jit(&Depth.cloud_grid(&1, &2, &3, &4, 4)).(
        Nx.broadcast(0.0, {8, 8}),
        100.0,
        4.0,
        4.0
      )

    assert grid |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() == 0.0
  end
end
