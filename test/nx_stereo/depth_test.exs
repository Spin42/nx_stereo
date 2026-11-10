defmodule NxStereo.DepthTest do
  use ExUnit.Case, async: true

  alias NxStereo.Depth

  test "depth is f·B/d, 0 without a match, and 0 in millimetres beyond 65535 mm" do
    disparity = Nx.tensor([[10.0, 0.0, -1.0, 0.0001, 0.36622]])
    {depth_mm, depth_m} = Nx.Defn.jit(&Depth.depth/3).(disparity, 400.0, 0.06)

    assert Nx.type(depth_mm) == {:u, 16}
    [[near, none, unmatched, far, _limit]] = Nx.to_list(depth_m)
    assert Nx.to_list(depth_mm) == [[2400, 0, 0, 0, 65534]]
    assert_in_delta near, 2.4, 1.0e-6
    assert {none, unmatched} == {0.0, 0.0}
    assert_in_delta far, 240_000.0, 1.0
  end

  test "the point cloud unprojects every decimation-th pixel from the first" do
    depth_m = Nx.broadcast(Nx.tensor(2.0), {48, 64})

    {grid, kept} =
      Nx.Defn.jit(&Depth.point_cloud(&1, &2, &3, &4, decimation: 4)).(depth_m, 100.0, 32.0, 24.0)

    assert Nx.shape(grid) == {12, 16, 3}
    [x, y, z] = Nx.to_list(grid[1][1])
    assert_in_delta x, (4 - 32) * 2.0 / 100, 1.0e-6
    assert_in_delta y, (4 - 24) * 2.0 / 100, 1.0e-6
    assert z == 2.0
    assert Nx.to_number(kept) == 12 * 16
  end

  test "a size that is not a multiple of the decimation samples the pixels its points stand for" do
    depth_m = Nx.iota({270, 8}, axis: 0, type: :f32) |> Nx.add(1.0)

    {grid, _kept} =
      Nx.Defn.jit(&Depth.point_cloud(&1, &2, &3, &4, decimation: 4, max_depth_m: 1000.0)).(
        depth_m,
        100.0,
        0.0,
        0.0
      )

    assert Nx.shape(grid) == {67, 2, 3}
    sampled_rows = grid[[.., 0, 2]] |> Nx.to_list() |> Enum.map(&trunc(&1 - 1))
    assert sampled_rows == for(v <- 0..66, do: v * 4)
    # y = v · z / f: the point's row is the row it sampled.
    [_x, y, z] = Nx.to_list(grid[[66, 0]])
    assert_in_delta y, 264 * z / 100, 1.0e-3
  end

  test "points without depth or outside the range are NaN and not kept" do
    depth_m = Nx.tensor([[0.0, 0.04, 0.06, 19.0, 21.0, 5.0]])

    {grid, kept} =
      Nx.Defn.jit(&Depth.point_cloud(&1, &2, &3, &4, min_depth_m: 0.05, max_depth_m: 20.0)).(
        depth_m,
        100.0,
        0.0,
        0.0
      )

    nan? = grid[[0, .., 2]] |> Nx.is_nan() |> Nx.to_flat_list()
    assert nan? == [1, 1, 0, 0, 1, 0]
    assert grid[[0, 0]] |> Nx.is_nan() |> Nx.to_flat_list() == [1, 1, 1]
    assert Nx.to_number(kept) == 3
  end

  test "by default every pixel with a positive depth is a point" do
    {grid, kept} =
      Nx.Defn.jit(&Depth.point_cloud/4).(Nx.tensor([[0.0, 0.01, 500.0]]), 100.0, 0.0, 0.0)

    assert Nx.shape(grid) == {1, 3, 3}
    assert Nx.to_number(kept) == 2
  end
end
