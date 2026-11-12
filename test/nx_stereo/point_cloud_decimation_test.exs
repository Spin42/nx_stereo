defmodule NxStereo.PointCloudDecimationTest do
  use ExUnit.Case, async: true

  alias NxStereo.Depth

  test "rejects a decimation that cannot form a nonempty sampling grid" do
    depth = Nx.broadcast(1.0, {2, 3})

    for decimation <- [0, -1, 1.5, 3, 4] do
      assert_raise ArgumentError, ~r/point_cloud.*decimation/, fn ->
        Depth.point_cloud(depth, 100.0, 0.0, 0.0, decimation: decimation)
      end
    end
  end
end
