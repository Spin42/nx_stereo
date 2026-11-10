defmodule NxStereo.Filter do
  @moduledoc """
  Filters of a disparity map, where a disparity of 0 or less means no
  match: one that drops speckles, and one that smooths the disparity
  over time.

  Compiled by `NxIREE.Compiler` for Vulkan, each runs as a hand-written
  compute shader; everywhere else, as this Nx code. Settings may be
  scalar tensors: they are then inputs of a compiled function, changed
  without compiling again.
  """
  import Nx.Defn

  @doc """
  Drops speckles, small islands of disparity that agree with nothing
  around them. A matched pixel is kept when at least `:min_support` of
  the other pixels in its square neighbourhood of radius `:radius`
  matched within `:max_diff` of it; the others become `-1.0`. Outside
  the image counts as unmatched. It plays the part of OpenCV's
  `filterSpeckles`, which labels connected regions instead, a poor fit
  for a GPU.

  ## Options

    * `:min_support` - neighbours that must agree. Defaults to 12; 0
      keeps every match.
    * `:max_diff` - how close, in pixels, a neighbour must be. Defaults
      to 1.0.
    * `:radius` - the neighbourhood's radius, fixed when compiled.
      Defaults to 3, a 7x7 window, which the shader computes; other
      radii run this Nx code.
  """
  deftransform despeckle(disparity, opts \\ []) do
    opts = Keyword.validate!(opts, min_support: 12, max_diff: 1.0, radius: 3)
    params = NxStereo.Kernels.scalars([opts[:min_support], opts[:max_diff]])
    radius = opts[:radius]

    if radius == NxStereo.Kernels.despeckle_radius() do
      NxStereo.Kernels.despeckle(disparity, params, fn [disparity, params] ->
        despeckle_n(disparity, params, radius: radius)
      end)
    else
      despeckle_n(disparity, params, radius: radius)
    end
  end

  # One gather of all neighbours, unmatched outside the image, compiles to one pass.
  defnp despeckle_n(disparity, params, opts \\ []) do
    radius = opts[:radius]
    {height, width} = Nx.shape(disparity)
    padded_width = width + 2 * radius

    flat =
      disparity
      |> Nx.pad(-1.0, [{radius, radius, 0}, {radius, radius, 0}])
      |> Nx.reshape({:auto, 1})

    centres =
      (Nx.iota({height, width, 1}, axis: 0) + radius) * padded_width +
        Nx.iota({height, width, 1}, axis: 1) + radius

    neighbours =
      flat
      |> Nx.gather(Nx.new_axis(centres + neighbour_offsets(radius, padded_width), -1))
      |> Nx.reshape({height, width, :auto})

    agree = neighbours > 0 and Nx.abs(neighbours - Nx.new_axis(disparity, -1)) <= params[1]
    support = Nx.sum(agree, axes: [2])
    Nx.select(disparity > 0 and support >= params[0], disparity, -1.0)
  end

  deftransformp neighbour_offsets(radius, padded_width) do
    offsets =
      for dy <- -radius..radius,
          dx <- -radius..radius,
          {dy, dx} != {0, 0},
          do: dy * padded_width + dx

    Nx.tensor([[offsets]], type: :s32)
  end

  @doc "The count of pairs missed at which `temporal/4` stops counting."
  @spec max_missed() :: pos_integer()
  def max_missed, do: NxStereo.Kernels.max_missed()

  @doc """
  One step of a temporal filter over successive disparity maps of a
  still or slowly moving scene. Returns the filtered disparity and the
  new count of pairs each pixel has gone unmatched, both `f32`; feed
  both back as `previous` and `missed` with the next pair, starting from
  `-1.0` and `0.0`.

  Where a pixel matched in `current` and `previous`, within `:delta`
  pixels of each other, it moves `:alpha` of the way from the previous
  value to the new one; elsewhere it takes the new value. A pixel that
  loses its match keeps its previous disparity for up to `:hold` pairs.
  The missed count stops at `max_missed/0`.

  ## Options

    * `:alpha` - the new value's weight; 1.0 turns smoothing off.
      Defaults to 0.4.
    * `:delta` - the largest change, in pixels, that is smoothed.
      Defaults to 2.0.
    * `:hold` - how many pairs a lost match is held; 0 turns holding
      off. Defaults to 2.
  """
  deftransform temporal(current, previous, missed, opts \\ []) do
    opts = Keyword.validate!(opts, alpha: 0.4, delta: 2.0, hold: 2)
    params = NxStereo.Kernels.scalars([opts[:alpha], opts[:delta], opts[:hold]])
    max_missed = max_missed()

    if rem(Nx.size(current), 4) == 0 do
      NxStereo.Kernels.temporal_filter(current, previous, missed, params, fn [c, p, m, params] ->
        temporal_n(c, p, m, params, max_missed: max_missed)
      end)
    else
      temporal_n(current, previous, missed, params, max_missed: max_missed)
    end
  end

  defnp temporal_n(current, previous, missed, params, opts \\ []) do
    alpha = params[0]
    agree = current > 0 and previous > 0 and Nx.abs(current - previous) <= params[1]
    smoothed = Nx.select(agree, alpha * current + (1 - alpha) * previous, current)
    matched = current > 0
    missed = Nx.select(matched, 0.0, missed + 1)
    held = not matched and previous > 0 and missed <= params[2]
    {Nx.select(held, previous, smoothed), Nx.min(missed, opts[:max_missed])}
  end
end
