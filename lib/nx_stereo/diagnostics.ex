defmodule NxStereo.Diagnostics do
  @moduledoc """
  Measures of a stereo rig's health: how sharp each camera's image is,
  whether the rectified images still line up row for row, and how much
  of the disparity matched and how steady it is.

  Each measure has two parts. The `*_sums` function runs inside a
  compiled function, next to the matching, and reduces the images to a
  few sums; compiled by `NxIREE.Compiler` for Vulkan it runs as a
  hand-written compute shader. The function of the same name without
  `_sums` turns the downloaded sums into the measure on the CPU.
  """
  import Nx.Defn

  @doc """
  The sums `sharpness/1` measures `left` and `right` from, rectified
  `{height, width}` images, both `u8` or both `s32`: per image, the
  squared differences, the greys and the number of the pairs of
  horizontal neighbours with neither pixel black (0), as an `f32` vector
  `[left energy, right energy, left grey, right grey, left pairs, right
  pairs]`.

  Leaving black pixels out keeps the borders rectification fills, where
  it reaches past a camera's frame, out of the measure.
  """
  deftransform sharpness_sums(left, right) do
    NxStereo.Kernels.sharpness(left, right, fn [left, right] -> sharpness_sums_n(left, right) end)
  end

  defnp sharpness_sums_n(left, right) do
    {left_energy, left_grey, left_pairs} = sharpness_planes(left)
    {right_energy, right_grey, right_pairs} = sharpness_planes(right)

    Nx.stack([left_energy, right_energy, left_grey, right_grey, left_pairs, right_pairs])
    |> image_sums()
  end

  # Per pair of horizontal neighbours both above black: their squared difference, the left one's grey, 1.
  defnp sharpness_planes(image) do
    {_height, width} = Nx.shape(image)
    image = Nx.as_type(image, :f32)
    pixels = Nx.slice_along_axis(image, 0, width - 1, axis: 1)
    neighbours = Nx.slice_along_axis(image, 1, width - 1, axis: 1)
    counted = Nx.as_type(pixels > 0 and neighbours > 0, :f32)
    difference = neighbours - pixels
    {difference * difference * counted, pixels * counted, counted}
  end

  @doc """
  Each image's sharpness from its `sharpness_sums/2`: the mean squared
  difference between horizontal neighbours over the square of the mean
  grey, as `%{left: float, right: float}`; 0.0 for an image without a
  counted pair. Dividing by the brightness keeps an exposure difference
  between the cameras out of it. The value depends on the scene: it is
  the ratio of the two that compares the cameras' focus.
  """
  @spec sharpness(Nx.Tensor.t()) :: %{left: float(), right: float()}
  def sharpness(sums) do
    [left_energy, right_energy, left_grey, right_grey, left_pairs, right_pairs] =
      Nx.to_flat_list(sums)

    measure = fn energy, grey, pairs ->
      if grey > 0, do: energy * pairs / (grey * grey), else: 0.0
    end

    %{
      left: measure.(left_energy, left_grey, left_pairs),
      right: measure.(right_energy, right_grey, right_pairs)
    }
  end

  @doc """
  Each image's mean grey from its `sharpness_sums/2`, over the pixels
  sharpness counts, black ones left out, as `%{left: float, right:
  float}`; 0.0 for an image without a counted pair. With both cameras at
  the same exposure and gain, a brighter image is a camera seeing
  differently, as through a hazy lens.
  """
  @spec brightness(Nx.Tensor.t()) :: %{left: float(), right: float()}
  def brightness(sums) do
    [_left_energy, _right_energy, left_grey, right_grey, left_pairs, right_pairs] =
      Nx.to_flat_list(sums)

    mean = fn grey, pairs -> if pairs > 0, do: grey / pairs, else: 0.0 end
    %{left: mean.(left_grey, left_pairs), right: mean.(right_grey, right_pairs)}
  end

  @doc """
  The sums `row_alignment/2` measures the row offset between the
  rectified `left` and `right` images from, both `u8` or both `s32`,
  given the `f32` `disparity` of `left`. On every `:row_step`-th row,
  each matched pixel is compared with the right image at its disparity on
  the rows `:reach` above to `:reach` below its own; the absolute
  differences are summed per row offset and per vertical band of the
  image, one of `:bands`. Returns the `f32` `{bands, 2 * reach + 2}`
  sums: per band, the sum at each offset from `-reach` to `reach`, then
  the number of matched pixels.

  ## Options

    * `:bands` - defaults to 4.
    * `:reach` - defaults to 2.
    * `:row_step` - defaults to 4.

  The shader computes the defaults; other values run this Nx code.
  """
  deftransform row_alignment_sums(left, right, disparity, opts \\ []) do
    defaults = NxStereo.Kernels.alignment()
    opts = Keyword.validate!(opts, Map.to_list(defaults))

    if Map.new(opts) == defaults do
      NxStereo.Kernels.row_alignment(left, right, disparity, fn [left, right, disparity] ->
        row_alignment_sums_n(left, right, disparity, opts)
      end)
    else
      row_alignment_sums_n(left, right, disparity, opts)
    end
  end

  defnp row_alignment_sums_n(left, right, disparity, opts \\ []) do
    row_step = opts[:row_step]
    {height, width} = Nx.shape(left)
    left = Nx.slice(left, [0, 0], [height, width], strides: [row_step, 1])
    disparity = Nx.slice(disparity, [0, 0], [height, width], strides: [row_step, 1])
    {sampled_height, _} = Nx.shape(left)
    matched = Nx.as_type(disparity > 0, :f32)
    columns = Nx.iota({sampled_height, width}, axis: 1)
    rows = Nx.iota({sampled_height, width}, axis: 0) * row_step
    source_columns = Nx.clip(columns - Nx.round(disparity), 0, width - 1)

    alignment_planes(
      Nx.reshape(Nx.as_type(right, :f32), {:auto, 1}),
      Nx.as_type(left, :f32),
      matched,
      rows,
      source_columns,
      opts
    )
  end

  # One plane per row offset, then the counts: a small last axis compiles to one-thread workgroups.
  deftransformp alignment_planes(right, left, matched, rows, source_columns, opts) do
    {sampled_height, width} = Nx.shape(left)
    height = div(Nx.axis_size(right, 0), width)
    %{bands: bands, reach: reach} = Map.new(opts)

    differences =
      for offset <- -reach..reach do
        indices =
          rows
          |> Nx.add(offset)
          |> Nx.clip(0, height - 1)
          |> Nx.multiply(width)
          |> Nx.add(source_columns)
          |> Nx.as_type(:s32)
          |> Nx.new_axis(-1)

        right
        |> Nx.gather(indices)
        |> Nx.reshape({sampled_height, width})
        |> Nx.subtract(left)
        |> Nx.abs()
        |> Nx.multiply(matched)
      end

    band_width = div(width, bands)
    planes = 2 * reach + 2

    # Down each column first: summing across a band first needs more GPU shared memory than there is.
    Nx.stack(differences ++ [matched])
    |> Nx.slice([0, 0, 0], [planes, sampled_height, band_width * bands])
    |> Nx.reshape({planes, sampled_height, bands, band_width})
    |> Nx.sum(axes: [1])
    |> Nx.sum(axes: [2])
    |> Nx.transpose()
  end

  @doc """
  The row offset from `row_alignment_sums/4`: how many rows lower the
  right image's match sits than its left pixel, over the whole image and
  per band from left to right, as `%{offset_rows: float, band_offset_rows:
  [float | nil]}`. The offset is the lowest sum's, refined between its
  neighbours by a parabola. A band, or the whole image, has no offset,
  `nil`, when fewer than `:min_matched` of its pixels matched or its
  lowest sum is not unique; without an offset over the whole image, the
  result is `nil`.

  A steady offset means a camera has moved; offsets that change across
  the bands mean it has turned.

  ## Options

    * `:min_matched` - defaults to 64.
  """
  @spec row_alignment(Nx.Tensor.t(), keyword()) ::
          %{offset_rows: float(), band_offset_rows: [float() | nil]} | nil
  def row_alignment(sums, opts \\ []) do
    opts = Keyword.validate!(opts, min_matched: 64)
    {_bands, planes} = Nx.shape(sums)
    reach = div(planes - 2, 2)

    {band_costs, band_counts} =
      sums
      |> Nx.to_list()
      |> Enum.map(&Enum.split(&1, 2 * reach + 1))
      |> Enum.unzip()

    band_counts = List.flatten(band_counts)
    costs = Enum.zip_with(band_costs, &Enum.sum/1)
    offset = fn costs, count -> offset(costs, count, reach, opts[:min_matched]) end

    case offset.(costs, Enum.sum(band_counts)) do
      nil ->
        nil

      rows ->
        %{offset_rows: rows, band_offset_rows: Enum.zip_with(band_costs, band_counts, offset)}
    end
  end

  # The row offset of the lowest cost, if enough pixels matched and no other offset shares it.
  defp offset(_costs, matched, _reach, min_matched) when matched < min_matched, do: nil

  defp offset(costs, _matched, reach, _min_matched) do
    lowest = Enum.min(costs)

    case for({cost, index} <- Enum.with_index(costs), cost == lowest, do: index) do
      [index] -> refine(costs, index, reach)
      _tied -> nil
    end
  end

  # The vertex of the parabola through the lowest cost and its neighbours: unique, it curves up.
  defp refine(costs, index, reach) when index > 0 and index < 2 * reach do
    [before, lowest, next] = Enum.slice(costs, index - 1, 3)
    index - reach + (before - next) / (2 * (before - 2 * lowest + next))
  end

  defp refine(_costs, index, reach), do: (index - reach) * 1.0

  @doc """
  The sums `disparity_stats/2` measures the `f32` disparity `current`
  from, given the `previous` one: the matched pixels, the sum of their
  disparity and of its square, the pixels matched in both, and the sum
  of their absolute change, as an `f32` vector of five.
  """
  deftransform disparity_sums(current, previous) do
    NxStereo.Kernels.disparity_sums(current, previous, fn [current, previous] ->
      disparity_sums_n(current, previous)
    end)
  end

  defnp disparity_sums_n(current, previous) do
    matched = Nx.as_type(current > 0, :f32)
    both = Nx.as_type(previous > 0, :f32) * matched
    values = current * matched

    Nx.stack([matched, values, values * values, both, Nx.abs(current - previous) * both])
    |> image_sums()
  end

  @doc """
  The disparity map's statistics from its `disparity_sums/2` and its
  number of `pixels`: the matched share, the mean disparity of the
  matched pixels and its standard deviation, and the jitter, the mean
  absolute change of the pixels matched in both maps. The jitter of a
  still scene is the matcher's noise.
  """
  @spec disparity_stats(Nx.Tensor.t(), pos_integer()) :: %{
          matched: float(),
          mean_px: float(),
          spread_px: float(),
          jitter_px: float()
        }
  def disparity_stats(sums, pixels) do
    [matched, total, squares, both, change] = Nx.to_flat_list(sums)
    mean = total / max(matched, 1)
    variance = max(squares / max(matched, 1) - mean * mean, 0.0)

    %{
      matched: matched / pixels,
      mean_px: mean,
      spread_px: :math.sqrt(variance),
      jitter_px: change / max(both, 1)
    }
  end

  # The sum of each image of a stack, down the columns first.
  defnp image_sums(images) do
    images |> Nx.sum(axes: [1]) |> Nx.sum(axes: [1])
  end
end
