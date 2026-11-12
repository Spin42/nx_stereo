defmodule NxStereo.Stage.SettingsTest do
  use ExUnit.Case, async: true

  alias NxStereo.Stage

  @geometry %{width: 96, height: 16, frame_width: 96, frame_height: 16}

  for {stage, keys} <- [
        {Stage.Rectification, [:right_row_offset]},
        {Stage.Disparity,
         [:p1, :p2, :uniqueness, :num_disparities, :paths, :search, :band, :margin, :fine_census]},
        {Stage.MedianFilter, [:kernel_size]},
        {Stage.SpeckleFilter, [:min_support, :max_diff]},
        {Stage.TemporalFilter, [:alpha, :delta, :hold]},
        {Stage.PointCloud, [:decimation]}
      ] do
    test "#{inspect(stage)} rejects nil for required settings" do
      for key <- unquote(keys) do
        assert {:error, message} = unquote(stage).init([{key, nil}], @geometry)
        assert String.starts_with?(message, "#{key} must ")
      end
    end
  end

  test "optional settings still accept nil and resolve their defaults" do
    assert {:ok, state} = Stage.Disparity.init([coarse_paths: nil, matcher: nil], @geometry)
    assert state.coarse_paths == state.paths
    assert state.matcher == nil
  end
end
