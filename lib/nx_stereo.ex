defmodule NxStereo do
  @moduledoc """
  Stereo matching for Nx.

  `sgm/3` computes a disparity map from a rectified pair of grayscale
  images with semi-global matching: a 9x7 Census transform that ignores
  one-level differences, Hamming matching costs, aggregation along eight
  paths (the horizontal, vertical and diagonal directions), or three with
  `paths: 3`, with libSGM's normalized recurrence, winner selection with a
  uniqueness check and parabolic subpixel refinement, OpenCV's left-right
  consistency check, and a 3x3 median filter.

  `sgm_band/4` does the same within a band of disparities per pixel, for a
  search refined around an earlier estimate, and `sgm_coarse_to_fine/3`
  combines both: a matching of the pair shrunk 2x, refined at full
  resolution within a band around it.

  It is written in `Nx.Defn` and runs on every Nx backend and compiler.
  Compiled by `NxIREE.Compiler` for Vulkan, it runs as compute shaders
  ported from Fixstars' libSGM instead, with the same matches; subpixel
  values may differ in their last bit, as GPU division need not be
  correctly rounded.
  """

  import Nx.Defn

  @census_bound 48
  @census_windows [:sparse_9x7, :dense_9x5, :dense_9x5_blend]
  @defaults [disparities: 64, p1: 8, p2: 64, uniqueness: 1.0526315789473684, paths: 8]
  # Use as Nx.u32(@invalid_key): defn narrows untyped literals to s32
  @invalid_key 0xFFFFFFFF

  @doc """
  Computes the disparity map of the `left` image against the `right` one.

  Both images are `{height, width}` tensors of the same type, `u8`, or
  `s32` holding 8-bit pixels for a device without 8-bit storage buffers,
  rectified so that matches lie on the same row and a point at column `x`
  in `left` appears at `x - d` in `right` for its disparity `d >= 0`. The result is an `f32`
  tensor of the same shape: the disparity of each left pixel, with a
  fractional part from subpixel refinement, or `-1.0` where no reliable match
  was found.

  ## Options

    * `:disparities` - the number of disparities searched, 64 or 128.
      Defaults to 64.
    * `:p1` - the penalty for a disparity change of one between neighbours.
      Defaults to 8. `:p1`, `:p2` and `:uniqueness` may be scalar tensors:
      they are then inputs of a compiled function, changed without
      compiling again, and checked by the caller.
    * `:p2` - the penalty for larger changes. Must exceed `:p1` and be at
      most 79. Defaults to 64.
    * `:uniqueness` - a match is kept when the second-best cost, ignoring
      the disparities next to the best one, exceeds the best cost times this
      factor. Defaults to 1.0526 (OpenCV's `uniquenessRatio` of 5).
    * `:paths` - the number of aggregation paths: 3 (left to right, right to
      left and top to bottom) or 8 (also bottom to top and the four
      diagonals), which is more accurate and slower. Defaults to 8.

  Images must be more than 8 pixels wide, more than 6 high, and at most 4096
  wide.
  """
  deftransform sgm(left, right, opts \\ []) do
    opts = Keyword.validate!(opts, @defaults)
    {config, left, right, penalties, output} = prepare!(left, right, opts, "NxStereo.sgm/3")

    NxStereo.Kernels.block(
      [left, right, penalties],
      output,
      fn [left, right, penalties] -> sgm_n(left, right, penalties, Map.to_list(config)) end,
      fn -> NxStereo.Vulkan.pipeline(config) end
    )
  end

  @doc false
  # The settings sgm/3 takes by default, which NxStereo.Stage.Disparity shares.
  def defaults, do: @defaults

  # What sgm/3 and sgm_band/4 start with: the matching's settings, both images as
  # s32, the penalties as the f32 vector the shaders read, and the output's template.
  defp prepare!(left, right, opts, caller) do
    config = config!(left, right, opts, caller)
    penalties = NxStereo.Kernels.scalars([opts[:p1], opts[:p2], opts[:uniqueness]])
    output = Nx.template({config.height, config.width}, :f32)
    {config, Nx.as_type(left, :s32), Nx.as_type(right, :s32), penalties, output}
  end

  defp config!(left, right, opts, caller) do
    left = Nx.to_template(left)
    right = Nx.to_template(right)

    unless left.type in [{:u, 8}, {:s, 32}] and right.type == left.type do
      raise ArgumentError,
            "#{caller} expects u8 or s32 images, got #{inspect(left.type)} and #{inspect(right.type)}"
    end

    {height, width} =
      case {left.shape, right.shape} do
        {{h, w}, {h, w}} when w > 8 and h > 6 and w <= 4096 ->
          {h, w}

        shapes ->
          raise ArgumentError,
                "#{caller} expects two images of the same {height, width} shape, larger than 6x8 and at most 4096 wide, got #{inspect(shapes)}"
      end

    disparities = opts[:disparities]
    p1 = opts[:p1]
    p2 = opts[:p2]
    uniqueness = opts[:uniqueness]

    unless disparities in [64, 128],
      do: raise(ArgumentError, ":disparities must be 64 or 128, got #{inspect(disparities)}")

    # The shaders sum pairs of paths bytewise, so each stays below 128.
    # Penalties given as tensors are inputs, checked by the caller.
    unless tensor?(p1) or tensor?(p2) or
             (is_integer(p1) and is_integer(p2) and p1 >= 0 and p1 < p2 and
                p2 + @census_bound <= 127) do
      raise ArgumentError,
            "expected integers 0 <= :p1 < :p2 <= #{127 - @census_bound}, got #{inspect({p1, p2})}"
    end

    unless tensor?(uniqueness) or (is_number(uniqueness) and uniqueness >= 1.0) do
      raise ArgumentError,
            ":uniqueness must be a number of at least 1.0, got #{inspect(uniqueness)}"
    end

    unless opts[:paths] in [3, 8],
      do: raise(ArgumentError, ":paths must be 3 or 8, got #{inspect(opts[:paths])}")

    %{height: height, width: width, disparities: disparities, paths: opts[:paths]}
  end

  defp tensor?(value), do: is_struct(value, Nx.Tensor)

  @doc """
  Computes the disparity map of `left` against `right` coarse to fine:
  `sgm/3` over the pair shrunk 2x with 64 disparities, then `sgm_band/4`
  at full resolution over the 128-disparity range, each pixel searched
  within a band of `:band` disparities.

  A pixel's band starts `:margin` disparities below twice the lowest
  coarse disparity around it, in its 3x3 neighbourhood at the coarse
  scale, so a pixel on a depth edge covers both sides when they are
  within `band - 2 * margin` disparities of each other. A pixel with no
  coarse match around it gets none. The matches are mostly `sgm/3`'s at
  full resolution, for a fraction of its cost; a structure too thin to
  survive the shrink can take the background's disparity.

  Takes `sgm/3`'s options, `:disparities` being 128, and:

    * `:band` - the band's width, 8, 16, 32 or 64. Defaults to 16.
    * `:margin` - how far below the coarse estimate a band starts.
      Defaults to 4.
    * `:coarse_paths` - the coarse matching's paths, 3 or 8. Defaults to
      `:paths`.
    * `:fine_census` - the fine descriptor, as `sgm_band/4`'s `:census`.
      Defaults to `:sparse_9x7`; the coarse pass always keeps that descriptor.

  Images must have an even height and width.
  """
  deftransform sgm_coarse_to_fine(left, right, opts \\ []) do
    defaults =
      Keyword.put(@defaults, :disparities, 128) ++
        [band: 16, margin: 4, coarse_paths: nil, fine_census: :sparse_9x7]

    opts = Keyword.validate!(opts, defaults)
    census_window!(opts[:fine_census], :fine_census)

    {height, width} = Nx.shape(left)

    if rem(height, 2) != 0 or rem(width, 2) != 0 do
      raise ArgumentError,
            "NxStereo.sgm_coarse_to_fine/3 expects images of even height and width, got #{inspect({height, width})}"
    end

    if opts[:disparities] != 128 do
      raise ArgumentError,
            ":disparities must be 128: the coarse matching takes 64, got #{inspect(opts[:disparities])}"
    end

    penalties = Keyword.take(opts, [:p1, :p2, :uniqueness])

    {left_half, right_half} = shrink_pair(left, right)

    coarse =
      sgm(
        left_half,
        right_half,
        [disparities: 64, paths: opts[:coarse_paths] || opts[:paths]] ++ penalties
      )

    base = coarse_base(coarse, opts[:band], opts[:margin])

    sgm_band(
      left,
      right,
      base,
      [disparities: 128, paths: opts[:paths], band: opts[:band], census: opts[:fine_census]] ++
        penalties
    )
  end

  # The pair uses 32-bit pixels throughout its rounded 2x2 average.
  deftransformp shrink_pair(left, right) do
    {height, width} = Nx.shape(left)
    half = Nx.template({div(height, 2), div(width, 2)}, :s32)

    NxStereo.Kernels.block(
      [Nx.as_type(left, :s32), Nx.as_type(right, :s32)],
      {half, half},
      fn [left, right] -> {shrink(left), shrink(right)} end,
      fn -> NxStereo.Vulkan.shrink_pipeline(height, width) end
    )
  end

  # The bases of band_base/3, as a shader on Vulkan
  deftransformp coarse_base(coarse, band, margin) do
    {height, width} = Nx.shape(coarse)
    {height, width} = {2 * height, 2 * width}

    NxStereo.Kernels.block(
      [coarse],
      Nx.template({height, width}, :s32),
      fn [coarse] -> band_base(coarse, band, margin) end,
      fn -> NxStereo.Vulkan.band_base_pipeline(height, width, margin, 128 - band) end
    )
  end

  # The 2x2 average, rounded, in the image's type
  deftransformp shrink(image) do
    {height, width} = Nx.shape(image)

    image
    |> Nx.as_type(:s32)
    |> Nx.reshape({div(height, 2), 2, div(width, 2), 2})
    |> Nx.sum(axes: [1, 3])
    |> Nx.add(2)
    |> Nx.quotient(4)
    |> Nx.as_type(Nx.type(image))
  end

  # Each full-resolution pixel's band base from the coarse disparity: margin
  # below twice the lowest match in its coarse 3x3 neighbourhood, -1 without one
  defnp band_base(coarse, band, margin) do
    {height, width} = Nx.shape(coarse)
    none = 1.0e9
    matched = Nx.select(coarse >= 0, coarse, none)
    lowest = matched |> Nx.pad(none, [{1, 1, 0}, {1, 1, 0}]) |> Nx.window_min({3, 3})
    start = (Nx.floor(lowest * 2) - margin) |> Nx.as_type(:s32) |> Nx.clip(0, 128 - band)
    base = Nx.select(lowest < none, start, -1)

    base
    |> Nx.reshape({height, 1, width, 1})
    |> Nx.broadcast({height, 2, width, 2})
    |> Nx.reshape({height * 2, width * 2})
  end

  @doc """
  Computes the disparity map of `left` against `right` as `sgm/3` does,
  but searching each pixel's disparity only within a band: the `:band`
  disparities from the pixel's `base`, an `s32` `{height, width}` tensor,
  up to `base + band - 1`.

  A path passing from one pixel to the next matches their candidates by
  disparity; a candidate outside the previous pixel's band is reached
  only with the `p2` penalty, as a jump. With a band covering every
  disparity and a base of zero, the result is `sgm/3`'s. A narrow band
  around a disparity estimated beforehand, such as a coarser matching's,
  costs the band's width instead of the full range. A negative base marks
  a pixel without an estimate: it is searched from 0 and gets no match.

  Takes `sgm/3`'s options and `:band`, 8, 16, 32 or 64; the caller keeps
  `base + band` within `:disparities`. Defaults to 16.

  `:census` chooses the descriptor: `:sparse_9x7` (default) has 48 comparisons
  outside the centre row and column; `:dense_9x5` has all 44 neighbours.
  `:dense_9x5_blend` uses the dense window and averages its centre with its
  rounded 3x3 Gaussian mean before comparison. The average is rounded too.
  All ignore one-level differences. Dense descriptors can improve detail;
  the blended centre reduces sensitivity to noise. Coverage and scene-level
  accuracy still vary, so select the option using representative images.
  """
  deftransform sgm_band(left, right, base, opts \\ []) do
    opts = Keyword.validate!(opts, @defaults ++ [band: 16, census: :sparse_9x7])
    census_window!(opts[:census], :census)
    band = opts[:band]

    {config, left, right, penalties, output} =
      prepare!(left, right, Keyword.drop(opts, [:band, :census]), "NxStereo.sgm_band/4")

    unless band in [8, 16, 32, 64] do
      raise ArgumentError, ":band must be 8, 16, 32 or 64, got #{inspect(band)}"
    end

    base_template = Nx.to_template(base)

    unless base_template.type == {:s, 32} and base_template.shape == {config.height, config.width} do
      raise ArgumentError,
            "NxStereo.sgm_band/4 expects base as an s32 tensor of shape #{inspect({config.height, config.width})}, " <>
              "got #{inspect(base_template.type)} #{inspect(base_template.shape)}"
    end

    config = config |> Map.put(:band, band) |> Map.put(:census, opts[:census])

    NxStereo.Kernels.block(
      [left, right, penalties, base],
      output,
      fn [left, right, penalties, base] ->
        band_n(left, right, base, penalties, Map.to_list(config))
      end,
      fn -> NxStereo.Vulkan.band_pipeline(config) end
    )
  end

  defnp band_n(left, right, base, penalties, opts) do
    disparities = opts[:disparities]
    searched = Nx.max(base, 0)

    costs =
      band_costs(census(left, opts[:census]), census(right, opts[:census]), searched, opts[:band])

    sums = band_aggregate(costs, searched, penalties, opts)
    {disparity, keys} = band_select(sums, base, penalties, opts)
    disparity |> consistency(keys, disparities) |> NxStereo.Image.median(3)
  end

  # costs[y, x, k] = popcount(left[y, x] ^ right[y, x - base[y, x] - k]), as matching_costs/3
  deftransformp band_costs(left, right, base, band) do
    {height, width} = left.shape
    d = Nx.add(Nx.new_axis(base, 2), Nx.iota({1, 1, band}))
    columns = Nx.subtract(Nx.iota({1, width, 1}, axis: 1), d)

    sampled =
      right
      |> Nx.take_along_axis(
        columns |> Nx.clip(0, width - 1) |> Nx.reshape({height, width * band}),
        axis: 1
      )
      |> Nx.reshape({height, width, band})

    shifted = Nx.select(Nx.less(columns, 0), Nx.u64(0), sampled)

    left
    |> Nx.new_axis(2)
    |> Nx.bitwise_xor(shifted)
    |> Nx.population_count()
    |> Nx.as_type(:s32)
  end

  # aggregate/3 over band costs, each path carrying the base of the pixel it left
  deftransformp band_aggregate(costs, base, penalties, opts) do
    down = fn shift -> scan_band(costs, base, penalties, Keyword.put(opts, :shift, shift)) end

    up = fn shift ->
      costs
      |> Nx.reverse(axes: [0])
      |> scan_band(Nx.reverse(base, axes: [0]), penalties, Keyword.put(opts, :shift, shift))
      |> Nx.reverse(axes: [0])
    end

    rows = Nx.transpose(costs, axes: [1, 0, 2])
    row_bases = Nx.transpose(base)

    horizontal = [
      rows |> scan_band(row_bases, penalties, opts) |> Nx.transpose(axes: [1, 0, 2]),
      rows
      |> Nx.reverse(axes: [0])
      |> scan_band(Nx.reverse(row_bases, axes: [0]), penalties, opts)
      |> Nx.reverse(axes: [0])
      |> Nx.transpose(axes: [1, 0, 2])
    ]

    paths =
      case opts[:paths] do
        3 -> horizontal ++ [down.(0)]
        8 -> horizontal ++ [down.(0), up.(0), down.(1), down.(-1), up.(1), up.(-1)]
      end

    Enum.reduce(paths, &Nx.add/2)
  end

  # scan/3 over band costs: the previous pixel's normalized path costs are
  # moved to this pixel's band by disparity, P2 standing for a candidate the
  # previous band lacks, as it stands for a neighbour past the range in scan/3.
  defnp scan_band(costs, base, penalties, opts) do
    p1 = Nx.as_type(penalties[0], :s32)
    p2 = Nx.as_type(penalties[1], :s32)
    {steps, lines, band} = Nx.shape(costs)
    previous = Nx.broadcast(0, {lines, band})
    previous_base = Nx.broadcast(-1, {lines})
    paths = Nx.broadcast(0, costs)
    k = Nx.iota({1, band + 2}) - 1

    {_previous, _previous_base, paths, _costs, _base, _p1, _p2} =
      while {previous, previous_base, paths, costs, base, p1, p2}, i <- 0..(steps - 1) do
        previous = follow(previous, opts[:shift])
        current_base = Nx.slice_along_axis(base, i, 1, axis: 0) |> Nx.squeeze(axes: [0])
        # A path entering the image starts from zero, in its own band
        previous_base = follow_base(previous_base, opts[:shift])
        previous_base = Nx.select(previous_base < 0, current_base, previous_base)

        a = previous - Nx.reduce_min(previous, axes: [1], keep_axes: true)
        index = k + Nx.new_axis(current_base - previous_base, 1)
        inside = index >= 0 and index < band
        moved = Nx.take_along_axis(a, Nx.clip(index, 0, band - 1), axis: 1)
        window = Nx.select(inside, moved, p2)

        own = Nx.slice_along_axis(window, 1, band, axis: 1)
        lower = Nx.slice_along_axis(window, 0, band, axis: 1)
        upper = Nx.slice_along_axis(window, 2, band, axis: 1)
        cost = Nx.slice_along_axis(costs, i, 1, axis: 0) |> Nx.squeeze(axes: [0])
        value = Nx.min(Nx.min(own, p2), Nx.min(lower, upper) + p1) + cost

        {value, current_base, Nx.put_slice(paths, [i, 0, 0], Nx.new_axis(value, 0)), costs, base,
         p1, p2}
      end

    paths
  end

  # follow/2 for the bases, -1 marking a line entering the image
  deftransformp follow_base(previous_base, shift) do
    case shift do
      s when s in [nil, 0] -> previous_base
      1 -> Nx.pad(previous_base, -1, [{1, -1, 0}])
      -1 -> Nx.pad(previous_base, -1, [{-1, 1, 0}])
    end
  end

  # select/3 within each pixel's band; a negative base gives no match
  defnp band_select(sums, base, penalties, opts) do
    disparities = opts[:disparities]
    estimated = Nx.new_axis(base >= 0, 2)
    base = Nx.max(base, 0)
    {_height, width, band} = Nx.shape(sums)
    k = Nx.iota({1, 1, band})
    d = Nx.new_axis(base, 2) + k
    x = Nx.iota({1, width, 1}, axis: 1)
    values = sums + Nx.select(d > x, 0x8000, 0)

    winner = Nx.argmin(values, axis: 2, keep_axis: true, tie_break: :low) |> Nx.as_type(:s32)
    minimum = Nx.take_along_axis(values, winner, axis: 2)

    second =
      Nx.select(Nx.abs(k - winner) <= 1, Nx.Constants.max_finite(:s32), values)
      |> Nx.reduce_min(axes: [2], keep_axes: true)

    lower = Nx.take_along_axis(values, Nx.max(winner - 1, 0), axis: 2) |> Nx.as_type(:f32)
    upper = Nx.take_along_axis(values, Nx.min(winner + 1, band - 1), axis: 2) |> Nx.as_type(:f32)
    minimum_f = Nx.as_type(minimum, :f32)
    winner_d = Nx.new_axis(base, 2) + winner

    accepted = estimated and winner_d <= x and Nx.as_type(second, :f32) > minimum_f * penalties[2]
    denominator = lower + upper - 2 * minimum_f
    correction = Nx.clip((lower - upper) / (2 * Nx.max(denominator, 1)), -0.5, 0.5)

    refine =
      denominator > 0 and winner > 0 and winner < band - 1 and
        winner_d < Nx.min(disparities - 1, x)

    disparity = Nx.as_type(winner_d, :f32) + Nx.select(refine, correction, 0.0)

    key =
      Nx.bitwise_or(Nx.left_shift(Nx.as_type(minimum, :u32), 16), Nx.as_type(winner_d, :u32))

    {Nx.select(accepted, disparity, -1.0) |> Nx.squeeze(axes: [2]),
     Nx.select(accepted, key, Nx.u32(@invalid_key)) |> Nx.squeeze(axes: [2])}
  end

  # The reference implementation, and the block's default where the shaders
  # do not apply
  defnp sgm_n(left, right, penalties, opts) do
    disparities = opts[:disparities]
    costs = matching_costs(census(left), census(right), disparities)
    {disparity, keys} = select(aggregate(costs, penalties, opts), penalties, opts)
    disparity |> consistency(keys, disparities) |> NxStereo.Image.median(3)
  end

  defp census_window!(window, key) do
    unless window in @census_windows,
      do:
        raise(
          ArgumentError,
          ":#{key} must be one of #{inspect(@census_windows)}, got #{inspect(window)}"
        )
  end

  # Neighbours more than one level below the anchor set a descriptor bit.
  deftransformp census(image, window \\ :sparse_9x7) do
    anchor = census_anchor(image, window)

    neighbours(window)
    |> Enum.with_index()
    |> Enum.reduce(Nx.broadcast(Nx.u64(0), image), fn {{dy, dx}, bit}, acc ->
      below = Nx.less(Nx.add(NxStereo.Image.shift(image, dy, dx), 1), anchor) |> Nx.as_type(:u64)
      Nx.bitwise_or(acc, Nx.left_shift(below, bit))
    end)
  end

  defp neighbours(:sparse_9x7) do
    for dy <- -3..3, dy != 0, dx <- -4..4, dx != 0, do: {dy, dx}
  end

  defp neighbours(_dense) do
    for dy <- -2..2, dx <- -4..4, dy != 0 or dx != 0, do: {dy, dx}
  end

  deftransformp census_anchor(image, :dense_9x5_blend) do
    weights = [{-1, 1}, {0, 2}, {1, 1}]

    weighted =
      for {dy, wy} <- weights,
          {dx, wx} <- weights,
          do: Nx.multiply(NxStereo.Image.shift(image, dy, dx), wy * wx)

    weighted
    |> Enum.reduce(&Nx.add/2)
    |> Nx.add(Nx.multiply(image, 16))
    |> Nx.add(24)
    |> Nx.quotient(32)
  end

  deftransformp(census_anchor(image, _window), do: image)

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

  # The sum of the path costs along every aggregation direction
  deftransformp aggregate(costs, penalties, opts) do
    down = fn shift -> scan(costs, penalties, Keyword.put(opts, :shift, shift)) end

    up = fn shift ->
      costs
      |> Nx.reverse(axes: [0])
      |> scan(penalties, Keyword.put(opts, :shift, shift))
      |> Nx.reverse(axes: [0])
    end

    rows = Nx.transpose(costs, axes: [1, 0, 2])

    horizontal = [
      rows |> scan(penalties, opts) |> Nx.transpose(axes: [1, 0, 2]),
      rows
      |> Nx.reverse(axes: [0])
      |> scan(penalties, opts)
      |> Nx.reverse(axes: [0])
      |> Nx.transpose(axes: [1, 0, 2])
    ]

    paths =
      case opts[:paths] do
        3 -> horizontal ++ [down.(0)]
        8 -> horizontal ++ [down.(0), up.(0), down.(1), down.(-1), up.(1), up.(-1)]
      end

    Enum.reduce(paths, &Nx.add/2)
  end

  # libSGM's normalized path recurrence along the first axis. With `:shift`
  # of 1 or -1 the path also moves one step along the second axis per step,
  # following a diagonal; it starts from zero where it enters the image.
  defnp scan(costs, penalties, opts) do
    p1 = Nx.as_type(penalties[0], :s32)
    p2 = Nx.as_type(penalties[1], :s32)
    {steps, lines, disparities} = Nx.shape(costs)
    previous = Nx.broadcast(0, {lines, disparities})
    paths = Nx.broadcast(0, costs)

    {_previous, paths, _costs, _p1, _p2} =
      while {previous, paths, costs, p1, p2}, i <- 0..(steps - 1) do
        previous = follow(previous, opts[:shift])
        a = previous - Nx.reduce_min(previous, axes: [1], keep_axes: true)
        lower = Nx.pad(a, p2 - p1, [{0, 0, 0}, {1, -1, 0}])
        upper = Nx.pad(a, p2 - p1, [{0, 0, 0}, {-1, 1, 0}])
        cost = Nx.slice_along_axis(costs, i, 1, axis: 0) |> Nx.squeeze(axes: [0])
        value = Nx.min(Nx.min(a, p2), Nx.min(lower, upper) + p1) + cost
        {value, Nx.put_slice(paths, [i, 0, 0], Nx.new_axis(value, 0)), costs, p1, p2}
      end

    paths
  end

  # Moves the path state of each line to the line it continues on
  deftransformp follow(previous, shift) do
    case shift do
      s when s in [nil, 0] -> previous
      1 -> Nx.pad(previous, 0, [{1, -1, 0}, {0, 0, 0}])
      -1 -> Nx.pad(previous, 0, [{-1, 1, 0}, {0, 0, 0}])
    end
  end

  # Winner, uniqueness check and subpixel refinement, as the shaders compute
  # them in f32. Disparities past the image's left edge never win.
  defnp select(sums, penalties, opts) do
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

    accepted = Nx.as_type(second, :f32) > minimum_f * penalties[2]
    denominator = lower + upper - 2 * minimum_f
    correction = Nx.clip((lower - upper) / (2 * Nx.max(denominator, 1)), -0.5, 0.5)
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
end
