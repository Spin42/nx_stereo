defmodule NxStereo.Stage.SpeckleFilter do
  @moduledoc """
  Drops speckles from `:disparity` with `NxStereo.Filter.despeckle/2`: a
  matched pixel is kept when at least `:min_support` of the other pixels
  in its 7x7 neighbourhood matched within `:max_diff` pixels of it.

  ## Settings

    * `:min_support` (`12`) - neighbours that must agree; 0 keeps every
      match.
    * `:max_diff` (`1.0`) - how close, in pixels, a neighbour must be.

  Both are params, so they change without compiling again.
  """
  @behaviour NxStereo.Stage

  @radius 3
  @defaults [min_support: 12, max_diff: 1.0]

  @impl NxStereo.Stage
  def requires, do: [:disparity]

  @impl NxStereo.Stage
  def provides, do: [:disparity]

  @impl NxStereo.Stage
  def init(settings, _geometry), do: NxStereo.Stage.settings(settings, @defaults)

  @impl NxStereo.Stage
  def params(state, _geometry), do: Map.take(state, [:min_support, :max_diff])

  @impl NxStereo.Stage
  def step(tensors, params, _state) do
    despeckled =
      NxStereo.Filter.despeckle(tensors.disparity,
        min_support: params.min_support,
        max_diff: params.max_diff,
        radius: @radius
      )

    Map.put(tensors, :disparity, despeckled)
  end

  @impl NxStereo.Stage
  def native_kind, do: "speckle_filter"

  @impl NxStereo.Stage
  def native_constants, do: %{radius: @radius}
end
