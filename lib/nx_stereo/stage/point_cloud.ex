defmodule NxStereo.Stage.PointCloud do
  @moduledoc """
  Unprojects `:depth_m` into an organized point cloud with
  `NxStereo.Depth.point_cloud/5`: `:cloud_grid`, the `x, y, z` points of
  every `:decimation`-th pixel in the left camera's frame, NaN outside
  5 cm to 20 m, and `:cloud_points`, how many are kept.

  ## Settings

    * `:decimation` (`4`) - depth pixels per point in each direction,
      fixed when compiled.
  """
  @behaviour NxStereo.Stage

  @min_depth_m 0.05
  @max_depth_m 20.0

  @impl NxStereo.Stage
  def requires, do: [:depth_m]

  @impl NxStereo.Stage
  def provides, do: [:cloud_grid, :cloud_points]

  @impl NxStereo.Stage
  def init(settings, _geometry), do: NxStereo.Stage.settings(settings, decimation: 4)

  @impl NxStereo.Stage
  def params(_state, geometry) do
    {cx, cy} = geometry.principal_point
    %{focal_length_px: geometry.focal_length_px, cx: cx, cy: cy}
  end

  @impl NxStereo.Stage
  def step(tensors, params, state) do
    {grid, kept} =
      NxStereo.Depth.point_cloud(tensors.depth_m, params.focal_length_px, params.cx, params.cy,
        decimation: state.decimation,
        min_depth_m: @min_depth_m,
        max_depth_m: @max_depth_m
      )

    tensors
    |> Map.put(:cloud_grid, grid)
    |> Map.put(:cloud_points, kept)
  end

  @impl NxStereo.Stage
  def native_kind, do: "point_cloud"

  @impl NxStereo.Stage
  def native_settings(state), do: [decimation: state.decimation]

  @impl NxStereo.Stage
  def native_constants, do: %{min_depth_m: @min_depth_m, max_depth_m: @max_depth_m}
end
