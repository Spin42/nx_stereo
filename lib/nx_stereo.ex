defmodule NxStereo do
  @moduledoc """
  Stereo matching for Nx.

  `sgm/3` computes a disparity map from a rectified pair of grayscale
  images with semi-global matching: a 9x7 Census transform that ignores
  one-level differences, Hamming matching costs, aggregation along three
  paths (left to right, right to left and top to bottom) with libSGM's
  normalized recurrence, winner selection with a uniqueness check and
  parabolic subpixel refinement, OpenCV's left-right consistency check, and
  a 3x3 median filter.

  It is written in `Nx.Defn` and runs on every Nx backend and compiler.
  Compiled by `NxIREE.Compiler` for Vulkan, it runs as six compute shaders
  ported from Fixstars' libSGM instead, with the same matches; subpixel
  values may differ in their last bit, as GPU division need not be
  correctly rounded.
  """

  import Nx.Defn

  @census_bound 48
  # Use as Nx.u32(@invalid_key): defn narrows untyped literals to s32
  @invalid_key 0xFFFFFFFF

  @doc """
  Computes the disparity map of the `left` image against the `right` one.

  Both images are `{height, width}` tensors of type `u8`, rectified so that
  matches lie on the same row and a point at column `x` in `left` appears at
  `x - d` in `right` for its disparity `d >= 0`. The result is an `f32`
  tensor of the same shape: the disparity of each left pixel, with a
  fractional part from subpixel refinement, or `-1.0` where no reliable match
  was found.

  ## Options

    * `:disparities` - the number of disparities searched, 64 or 128.
      Defaults to 64.
    * `:p1` - the penalty for a disparity change of one between neighbours.
      Defaults to 8.
    * `:p2` - the penalty for larger changes. Must exceed `:p1` and be at
      most 79. Defaults to 64.
    * `:uniqueness` - a match is kept when the second-best cost, ignoring
      the disparities next to the best one, exceeds the best cost times this
      factor. Defaults to 1.0526 (OpenCV's `uniquenessRatio` of 5).

  Images must be more than 8 pixels wide, more than 6 high, and at most 4096
  wide.
  """
  deftransform sgm(left, right, opts \\ []) do
    opts = Keyword.validate!(opts, disparities: 64, p1: 8, p2: 64, uniqueness: 1.0526315789473684)
    config = config!(left, right, opts)
    left = Nx.as_type(left, :s32)
    right = Nx.as_type(right, :s32)
    output = Nx.template({config.height, config.width}, :f32)

    case NxStereo.Vulkan.pipeline(config) do
      nil ->
        sgm_n(left, right, Map.to_list(config))

      pipeline ->
        Nx.block(pipeline, [left, right], output, fn _, l, r ->
          sgm_n(l, r, Map.to_list(config))
        end)
    end
  end

  defp config!(left, right, opts) do
    left = Nx.to_template(left)
    right = Nx.to_template(right)

    unless left.type == {:u, 8} and right.type == {:u, 8} do
      raise ArgumentError,
            "NxStereo.sgm/3 expects u8 images, got #{inspect(left.type)} and #{inspect(right.type)}"
    end

    {height, width} =
      case {left.shape, right.shape} do
        {{h, w}, {h, w}} when w > 8 and h > 6 and w <= 4096 ->
          {h, w}

        shapes ->
          raise ArgumentError,
                "NxStereo.sgm/3 expects two images of the same {height, width} shape, larger than 6x8 and at most 4096 wide, got #{inspect(shapes)}"
      end

    disparities = opts[:disparities]
    p1 = opts[:p1]
    p2 = opts[:p2]
    uniqueness = opts[:uniqueness]

    unless disparities in [64, 128],
      do: raise(ArgumentError, ":disparities must be 64 or 128, got #{inspect(disparities)}")

    # The shaders sum both horizontal paths bytewise, so each stays below 128
    unless is_integer(p1) and is_integer(p2) and p1 >= 0 and p1 < p2 and p2 + @census_bound <= 127 do
      raise ArgumentError,
            "expected integers 0 <= :p1 < :p2 <= #{127 - @census_bound}, got #{inspect({p1, p2})}"
    end

    unless is_number(uniqueness) and uniqueness >= 1.0 do
      raise ArgumentError,
            ":uniqueness must be a number of at least 1.0, got #{inspect(uniqueness)}"
    end

    %{
      height: height,
      width: width,
      disparities: disparities,
      p1: p1,
      p2: p2,
      uniqueness: uniqueness * 1.0
    }
  end

  # The reference implementation, and the block's default where the shaders
  # do not apply
  defnp sgm_n(left, right, opts) do
    disparities = opts[:disparities]
    costs = matching_costs(census(left), census(right), disparities)
    forward = scan_x(costs, opts)
    reverse = costs |> Nx.reverse(axes: [1]) |> scan_x(opts) |> Nx.reverse(axes: [1])
    vertical = scan_y(costs, opts)
    {disparity, keys} = select(forward + reverse + vertical, opts)
    disparity |> consistency(keys, disparities) |> median()
  end

  # 48 bits per pixel, one per neighbour of the 9x7 window outside the centre
  # row and column, set where the neighbour is more than one level below the
  # centre, so that sensor noise on flat surfaces does not flip bits. Borders
  # are replicated.
  deftransformp census(image) do
    centre = image

    neighbours()
    |> Enum.with_index()
    |> Enum.reduce(Nx.broadcast(Nx.u64(0), image), fn {{dy, dx}, bit}, acc ->
      below = Nx.less(Nx.add(shift(image, dy, dx), 1), centre) |> Nx.as_type(:u64)
      Nx.bitwise_or(acc, Nx.left_shift(below, bit))
    end)
  end

  defp neighbours do
    for dy <- -3..3, dy != 0, dx <- -4..4, dx != 0, do: {dy, dx}
  end

  # image[clamp(y + dy), clamp(x + dx)]
  deftransformp shift(image, dy, dx) do
    {height, width} = image.shape
    rows = Nx.clip(Nx.add(Nx.iota({height}), dy), 0, height - 1)
    columns = Nx.clip(Nx.add(Nx.iota({width}), dx), 0, width - 1)
    image |> Nx.take(rows, axis: 0) |> Nx.take(columns, axis: 1)
  end

  # costs[y, x, d] = popcount(left[y, x] ^ right[y, x - d]), right features
  # left of the image being zero
  deftransformp matching_costs(left, right, disparities) do
    {_height, width} = left.shape

    shifted =
      for d <- 0..(disparities - 1) do
        right
        |> Nx.pad(Nx.u64(0), [{0, 0, 0}, {d, 0, 0}])
        |> Nx.slice_along_axis(0, width, axis: 1)
      end
      |> Nx.stack(axis: 2)

    left |> Nx.new_axis(2) |> Nx.bitwise_xor(shifted) |> Nx.population_count() |> Nx.as_type(:s32)
  end

  defnp scan_x(costs, opts) do
    costs |> Nx.transpose(axes: [1, 0, 2]) |> scan(opts) |> Nx.transpose(axes: [1, 0, 2])
  end

  defnp(scan_y(costs, opts), do: scan(costs, opts))

  # libSGM's normalized path recurrence along the first axis
  defnp scan(costs, opts) do
    p1 = opts[:p1]
    p2 = opts[:p2]
    {steps, lines, disparities} = Nx.shape(costs)
    previous = Nx.broadcast(0, {lines, disparities})
    paths = Nx.broadcast(0, costs)

    {_previous, paths, _costs} =
      while {previous, paths, costs}, i <- 0..(steps - 1) do
        a = previous - Nx.reduce_min(previous, axes: [1], keep_axes: true)
        lower = Nx.pad(a, p2 - p1, [{0, 0, 0}, {1, -1, 0}])
        upper = Nx.pad(a, p2 - p1, [{0, 0, 0}, {-1, 1, 0}])
        cost = Nx.slice_along_axis(costs, i, 1, axis: 0) |> Nx.squeeze(axes: [0])
        value = Nx.min(Nx.min(a, p2), Nx.min(lower, upper) + p1) + cost
        {value, Nx.put_slice(paths, [i, 0, 0], Nx.new_axis(value, 0)), costs}
      end

    paths
  end

  # Winner, uniqueness check and subpixel refinement, as the shaders compute
  # them in f32. Disparities past the image's left edge never win.
  defnp select(sums, opts) do
    disparities = opts[:disparities]
    {_height, width, _} = Nx.shape(sums)
    d = Nx.iota({1, 1, disparities})
    x = Nx.iota({1, width, 1}, axis: 1)
    values = sums + Nx.select(d > x, 0x8000, 0)

    winner = Nx.argmin(values, axis: 2, keep_axis: true, tie_break: :low) |> Nx.as_type(:s32)
    minimum = Nx.take_along_axis(values, winner, axis: 2)

    second =
      Nx.select(Nx.abs(d - winner) <= 1, Nx.Constants.max_finite(:s32), values)
      |> Nx.reduce_min(axes: [2], keep_axes: true)

    lower = Nx.take_along_axis(values, Nx.max(winner - 1, 0), axis: 2) |> Nx.as_type(:f32)

    upper =
      Nx.take_along_axis(values, Nx.min(winner + 1, disparities - 1), axis: 2) |> Nx.as_type(:f32)

    minimum_f = Nx.as_type(minimum, :f32)

    accepted = Nx.as_type(second, :f32) > minimum_f * Nx.f32(opts[:uniqueness])
    denominator = lower + upper - 2 * minimum_f
    correction = Nx.clip((lower - upper) / (2 * Nx.max(denominator, 1)), -0.5, 0.5)
    x = Nx.iota({1, width, 1}, axis: 1)
    refine = denominator > 0 and winner > 0 and winner < Nx.min(disparities - 1, x)
    disparity = Nx.as_type(winner, :f32) + Nx.select(refine, correction, 0.0)

    key = Nx.bitwise_or(Nx.left_shift(Nx.as_type(minimum, :u32), 16), Nx.as_type(winner, :u32))

    {Nx.select(accepted, disparity, -1.0) |> Nx.squeeze(axes: [2]),
     Nx.select(accepted, key, Nx.u32(@invalid_key)) |> Nx.squeeze(axes: [2])}
  end

  # OpenCV's left-right check: each right pixel keeps the cheapest left match
  # projected onto it, ties favouring the larger disparity, and a left pixel
  # is rejected when neither adjacent integer disparity agrees within one.
  defnp consistency(disparity, keys, disparities) do
    {_height, width} = Nx.shape(keys)
    right = right_matches(keys, disparities)
    x = Nx.iota({1, width}, axis: 1)
    lo = Nx.floor(disparity) |> Nx.as_type(:s32)
    hi = Nx.ceil(disparity) |> Nx.as_type(:s32)
    rejected = disagrees(right, x - lo, lo) and disagrees(right, x - hi, hi)
    Nx.select(disparity >= 0 and not rejected, disparity, -1.0)
  end

  deftransformp right_matches(keys, disparities) do
    {_height, width} = keys.shape

    for d <- 0..(disparities - 1) do
      # key[y, rx + d], for left pixels within the image
      shifted =
        keys
        |> Nx.pad(Nx.u32(@invalid_key), [{0, 0, 0}, {0, d, 0}])
        |> Nx.slice_along_axis(d, width, axis: 1)

      matches =
        Nx.not_equal(shifted, Nx.u32(@invalid_key))
        |> Nx.logical_and(Nx.equal(Nx.bitwise_and(shifted, 0xFFFF), d))

      candidate = Nx.bitwise_or(Nx.bitwise_and(shifted, Nx.u32(0xFFFF0000)), 65_535 - d)
      Nx.select(matches, candidate, Nx.u32(@invalid_key))
    end
    |> Nx.stack(axis: 2)
    |> Nx.reduce_min(axes: [2])
  end

  defnp disagrees(right, rx, d) do
    {_height, width} = Nx.shape(right)
    inside = rx >= 0 and rx < width
    match = Nx.take_along_axis(right, Nx.clip(rx, 0, width - 1), axis: 1)
    right_d = 65_535 - Nx.as_type(Nx.bitwise_and(match, 0xFFFF), :s32)
    inside and match != Nx.u32(@invalid_key) and Nx.abs(right_d - d) > 1
  end

  # 3x3 median with replicated borders
  deftransformp median(disparity) do
    for(dy <- -1..1, dx <- -1..1, do: shift(disparity, dy, dx))
    |> Nx.stack(axis: 2)
    |> Nx.sort(axis: 2)
    |> Nx.slice_along_axis(4, 1, axis: 2)
    |> Nx.squeeze(axes: [2])
  end
end
