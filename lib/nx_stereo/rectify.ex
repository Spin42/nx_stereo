defmodule NxStereo.Rectify do
  @moduledoc """
  Undistortion and rectification of a grayscale image, as OpenCV's
  `initUndistortRectifyMap` followed by a bilinear `remap`, in one
  `defn`.

  The maps are computed in the graph from a few camera values rather than
  passed as images, so a compiled function serves any calibration of the
  same size.

      camera = NxStereo.Rectify.camera(k, d, r, p)
      rectified = NxStereo.Rectify.rectify(image, camera)

  With `:size` smaller than the image, the calibration is the one scaled
  to that size, and the image is shrunk in the same pass: a camera frame
  goes to the matcher's size without a separate resize.
  """
  import Nx.Defn

  @typedoc """
  A camera as `rectify/2` takes it: `:k` the `{3, 3}` camera matrix,
  `:d` the five plumb_bob coefficients `k1, k2, p1, p2, k3`, and
  `:inverse_pr` the `{3, 3}` inverse of the rectified camera matrix (the
  left 3x3 of P) times the rectifying rotation R. All `f32`.
  """
  @type camera :: %{k: Nx.Tensor.t(), d: Nx.Tensor.t(), inverse_pr: Nx.Tensor.t()}

  @doc """
  The camera map for `rectify/2` from a calibration's row-major lists:
  `k` and `r` with 9 values, `p` with 12, and `d` the plumb_bob
  coefficients (missing ones are 0). More than five non-zero
  coefficients raise: `rectify/2` models plumb_bob only.
  """
  @spec camera([number()], [number()], [number()], [number()]) :: camera()
  def camera(k, d, r, p) when length(k) == 9 and length(r) == 9 and length(p) == 12 do
    p3 = [0, 1, 2, 4, 5, 6, 8, 9, 10] |> Enum.map(&Enum.at(p, &1))
    {coefficients, extra} = Enum.split(d ++ List.duplicate(0.0, 5), 5)

    if Enum.any?(extra, &(&1 != 0)) do
      raise ArgumentError,
            "rectify/2 models plumb_bob distortion (5 coefficients), got #{inspect(d)}"
    end

    %{
      k: k |> Nx.tensor(type: :f32) |> Nx.reshape({3, 3}),
      d: Nx.tensor(coefficients, type: :f32),
      inverse_pr:
        p3 |> multiply_3x3(r) |> invert_3x3() |> Nx.tensor(type: :f32) |> Nx.reshape({3, 3})
    }
  end

  @doc """
  The rectified `u8` image: each pixel samples `image` bilinearly where
  the camera model maps it, with 0 outside the image.

  ## Options

    * `:size` - the rectified image's `{height, width}`, at most the
      image's, which is the default. `camera` is the calibration at that
      size. A smaller size shrinks the image in the same pass, with one
      bilinear sample per pixel: faster on a GPU than a separate area
      resize, at the cost of some aliasing of fine texture.
  """
  deftransform rectify(image, camera, opts \\ []) do
    {height, width} = size = Keyword.get(opts, :size, Nx.shape(image))
    {in_height, in_width} = Nx.shape(image)

    if height > in_height or width > in_width do
      raise ArgumentError,
            "rectify/3 only shrinks, got #{in_height}x#{in_width} to #{height}x#{width}"
    end

    rectify_n(image, camera, size: size, scale: {in_width / width, in_height / height})
  end

  defnp rectify_n(image, camera, opts) do
    {map_x, map_y} = project(pixel_grid(opts[:size]), camera)
    {scale_x, scale_y} = opts[:scale]
    # Pixel centres line up, as in OpenCV's resize.
    source_x = (map_x + 0.5) * scale_x - 0.5
    source_y = (map_y + 0.5) * scale_y - 0.5

    image
    |> bilinear(source_x, source_y)
    |> Nx.round()
    |> Nx.clip(0, 255)
    |> Nx.as_type(:u8)
  end

  @doc """
  For each pixel of the rectified image, where `rectify/2` samples
  `image`: the `f32` `{map_x, map_y}` OpenCV's `initUndistortRectifyMap`
  computes.
  """
  defn maps(image, camera) do
    project(pixel_grid(Nx.shape(image)), camera)
  end

  defnp project(grid, camera) do
    {u, v} = grid
    ir = camera.inverse_pr

    x = ir[0][0] * u + ir[0][1] * v + ir[0][2]
    y = ir[1][0] * u + ir[1][1] * v + ir[1][2]
    w = ir[2][0] * u + ir[2][1] * v + ir[2][2]
    x = x / w
    y = y / w

    d = camera.d
    r2 = x * x + y * y
    radial = 1 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 * r2 * r2
    xd = x * radial + 2 * d[2] * x * y + d[3] * (r2 + 2 * x * x)
    yd = y * radial + d[2] * (r2 + 2 * y * y) + 2 * d[3] * x * y

    k = camera.k
    {k[0][0] * xd + k[0][1] * yd + k[0][2], k[1][1] * yd + k[1][2]}
  end

  deftransformp pixel_grid(shape) do
    {Nx.iota(shape, axis: 1, type: :f32), Nx.iota(shape, axis: 0, type: :f32)}
  end

  defnp bilinear(image, map_x, map_y) do
    {height, width} = Nx.shape(image)
    flat = image |> Nx.reshape({:auto}) |> Nx.as_type(:f32)

    x0 = Nx.floor(map_x)
    y0 = Nx.floor(map_y)
    fx = map_x - x0
    fy = map_y - y0
    x0 = Nx.as_type(x0, :s32)
    y0 = Nx.as_type(y0, :s32)

    top =
      (1 - fx) * sample(flat, x0, y0, height, width) +
        fx * sample(flat, x0 + 1, y0, height, width)

    bottom =
      (1 - fx) * sample(flat, x0, y0 + 1, height, width) +
        fx * sample(flat, x0 + 1, y0 + 1, height, width)

    (1 - fy) * top + fy * bottom
  end

  # Outside the image a neighbour reads 0, as OpenCV's BORDER_CONSTANT.
  defnp sample(flat, x, y, height, width) do
    inside = x >= 0 and x < width and y >= 0 and y < height
    index = Nx.clip(y, 0, height - 1) * width + Nx.clip(x, 0, width - 1)
    Nx.select(inside, Nx.take(flat, index), 0.0)
  end

  defp multiply_3x3(a, b) do
    for i <- 0..2, j <- 0..2 do
      Enum.sum(for n <- 0..2, do: Enum.at(a, i * 3 + n) * Enum.at(b, n * 3 + j))
    end
  end

  defp invert_3x3([a, b, c, d, e, f, g, h, i]) do
    cofactors = [
      e * i - f * h,
      c * h - b * i,
      b * f - c * e,
      f * g - d * i,
      a * i - c * g,
      c * d - a * f,
      d * h - e * g,
      b * g - a * h,
      a * e - b * d
    ]

    determinant =
      a * Enum.at(cofactors, 0) + b * Enum.at(cofactors, 3) + c * Enum.at(cofactors, 6)

    Enum.map(cofactors, &(&1 / determinant))
  end
end
