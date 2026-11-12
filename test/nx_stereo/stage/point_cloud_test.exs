defmodule NxStereo.Stage.PointCloudTest do
  use ExUnit.Case, async: true

  alias NxStereo.Stage.PointCloud

  test "rejects decimation that exceeds either working dimension" do
    for {width, height, decimation} <- [{8, 3, 4}, {3, 8, 4}, {2, 3, 16}] do
      assert {:error, reason} =
               PointCloud.init([decimation: decimation], %{width: width, height: height})

      assert reason =~ "decimation"
      assert reason =~ "#{width}x#{height}"
    end
  end

  test "rejects the default decimation when the working image is too small" do
    assert {:error, _reason} = PointCloud.init([], %{width: 3, height: 2})
  end

  test "allows a single output row or column and discards incomplete sampling blocks" do
    for {width, height, decimation} <- [{1, 1, 1}, {7, 3, 3}, {2, 7, 2}] do
      geometry = %{
        width: width,
        height: height,
        focal_length_px: 100.0,
        principal_point: {0.0, 0.0}
      }

      assert {:ok, state} = PointCloud.init([decimation: decimation], geometry)

      outputs =
        PointCloud.step(
          %{depth_m: Nx.broadcast(1.0, {height, width})},
          PointCloud.params(state, geometry),
          state
        )

      assert Nx.shape(outputs.cloud_grid) ==
               {div(height, decimation), div(width, decimation), 3}

      assert Nx.type(outputs.cloud_points) == {:s, 32}

      assert Nx.to_number(outputs.cloud_points) ==
               div(height, decimation) * div(width, decimation)
    end
  end
end
