defmodule NxStereo.Depth do
  @moduledoc """
  Metric depth and points from a disparity map in pixels, where a
  disparity of 0 or less means no match.
  """
  import Nx.Defn

  @doc """
  `Z = f × B / d` for a rectified pair with focal length
  `focal_length_px` and baseline `baseline_m`. Returns the depth as
  rounded `u16` millimetres, saturating at 65535, and as `f32` metres;
  both are 0 where there is no match.
  """
  defn depth(disparity, focal_length_px, baseline_m) do
    matched = disparity > 0

    depth_m =
      Nx.select(matched, focal_length_px * baseline_m / Nx.select(matched, disparity, 1), 0)

    depth_mm = (depth_m * 1000) |> Nx.round() |> Nx.clip(0, 65_535) |> Nx.as_type(:u16)
    {depth_mm, depth_m}
  end

  @doc """
  The `f32` `{height / decimation, width / decimation, 3}` grid of
  `x, y, z` points in the camera frame, one every `decimation` pixels in
  each direction, sampled as OpenCV's nearest-neighbour resize does.
  Pixels without depth give the point `0, 0, 0`.
  """
  deftransform cloud_grid(depth_m, focal_length_px, cx, cy, decimation) do
    {height, width} = Nx.shape(depth_m)
    {grid_height, grid_width} = {div(height, decimation), div(width, decimation)}

    z =
      depth_m
      |> Nx.reshape({height * width, 1})
      |> Nx.gather(sample_indices(height, width, grid_height, grid_width))
      |> Nx.reshape({grid_height, grid_width})

    unproject(z, focal_length_px, cx, cy, decimation: decimation)
  end

  defnp unproject(z, focal_length_px, cx, cy, opts \\ []) do
    decimation = opts[:decimation]
    u = Nx.iota(Nx.shape(z), axis: 1, type: :f32) * decimation
    v = Nx.iota(Nx.shape(z), axis: 0, type: :f32) * decimation
    z_over_f = z / focal_length_px
    Nx.stack([(u - cx) * z_over_f, (v - cy) * z_over_f, z], axis: -1)
  end

  # OpenCV's INTER_NEAREST: output pixel `i` reads input `floor(i × in / out)`.
  defp sample_indices(height, width, grid_height, grid_width) do
    rows = Enum.map(0..(grid_height - 1), &floor(&1 * (height / grid_height)))
    columns = Enum.map(0..(grid_width - 1), &floor(&1 * (width / grid_width)))

    for(row <- rows, column <- columns, do: [row * width + column])
    |> Nx.tensor(type: :s32)
  end
end
