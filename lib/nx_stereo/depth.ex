defmodule NxStereo.Depth do
  @moduledoc """
  Metric depth and points from a disparity map in pixels, where a
  disparity of 0 or less means no match.
  """
  import Nx.Defn

  @doc """
  `Z = f × B / d` for a rectified pair with focal length
  `focal_length_px` and baseline `baseline_m`. Returns the depth as
  rounded `u16` millimetres and as `f32` metres, both 0 where there is no
  match; the millimetres are 0 too beyond the 65535 mm they hold.
  """
  deftransform depth(disparity, focal_length_px, baseline_m) do
    params = NxStereo.Kernels.scalars([focal_length_px, baseline_m])

    if rem(Nx.size(disparity), 2) == 0 do
      NxStereo.Kernels.depth(disparity, params, fn [disparity, params] ->
        depth_n(disparity, params)
      end)
    else
      depth_n(disparity, params)
    end
  end

  defnp depth_n(disparity, params) do
    matched = disparity > 0
    depth_m = Nx.select(matched, params[0] * params[1] / Nx.select(matched, disparity, 1), 0)
    rounded_mm = Nx.round(depth_m * 1000)
    depth_mm = Nx.select(rounded_mm > 65_535, 0, Nx.max(rounded_mm, 0)) |> Nx.as_type(:u16)
    {depth_mm, depth_m}
  end

  @doc """
  The organized point cloud of `depth_m`: the `f32` `{height / decimation,
  width / decimation, 3}` grid of `x, y, z` points in the camera frame,
  point `(i, j)` unprojected from pixel `(i × decimation, j × decimation)`,
  and the number of points kept, an `s32` scalar. A point whose depth is
  not strictly between `:min_depth_m` and `:max_depth_m`, such as a pixel
  without depth, is NaN, as ROS organized clouds mark them.

  ## Options

    * `:decimation` - positive integer depth pixels per point in each
      direction, no greater than either image dimension. Defaults to 1.
    * `:min_depth_m` - defaults to 0.
    * `:max_depth_m` - defaults to infinity.
  """
  deftransform point_cloud(depth_m, focal_length_px, cx, cy, opts \\ []) do
    opts = Keyword.validate!(opts, decimation: 1, min_depth_m: 0.0, max_depth_m: :infinity)
    check_decimation!(Nx.shape(depth_m), opts[:decimation])

    max_depth_m =
      if opts[:max_depth_m] == :infinity,
        do: Nx.Constants.infinity(:f32),
        else: opts[:max_depth_m]

    params = NxStereo.Kernels.scalars([focal_length_px, cx, cy, opts[:min_depth_m], max_depth_m])

    NxStereo.Kernels.point_cloud(depth_m, params, opts[:decimation], fn [depth_m, params] ->
      point_cloud_n(depth_m, params, decimation: opts[:decimation])
    end)
  end

  defp check_decimation!({height, width}, decimation) do
    unless is_integer(decimation) and decimation > 0 and decimation <= min(height, width) do
      raise ArgumentError,
            "point_cloud/5 needs a positive integer decimation within the #{width}x#{height} image, " <>
              "got #{inspect(decimation)}"
    end
  end

  defnp point_cloud_n(depth_m, params, opts \\ []) do
    decimation = opts[:decimation]
    z = sample(depth_m, decimation)
    in_range = z > params[3] and z < params[4]
    # NaN depth unprojects to a NaN point: x and y scale with z.
    z = Nx.select(in_range, z, Nx.Constants.nan(:f32))
    u = Nx.iota(Nx.shape(z), axis: 1, type: :f32) * decimation
    v = Nx.iota(Nx.shape(z), axis: 0, type: :f32) * decimation
    z_over_f = z / params[0]
    grid = Nx.stack([(u - params[1]) * z_over_f, (v - params[2]) * z_over_f, z], axis: -1)
    {grid, in_range |> Nx.as_type(:f32) |> Nx.sum()}
  end

  deftransformp sample(depth_m, decimation) do
    {height, width} = Nx.shape(depth_m)
    lengths = [div(height, decimation) * decimation, div(width, decimation) * decimation]
    Nx.slice(depth_m, [0, 0], lengths, strides: [decimation, decimation])
  end
end
