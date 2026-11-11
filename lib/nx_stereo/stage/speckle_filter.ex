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
  @impl NxStereo.Stage
  def requires, do: [:disparity]

  @impl NxStereo.Stage
  def provides, do: [:disparity]

  @impl NxStereo.Stage
  def settings do
    neighbours = (2 * @radius + 1) * (2 * @radius + 1) - 1

    [
      min_support: %{
        default: 12,
        type: :integer,
        range: {0, neighbours},
        live: true,
        doc:
          "Neighbours in the 7x7 window that must agree for a pixel to be kept; 0 turns the filter off"
      },
      max_diff: %{
        default: 1.0,
        type: :float,
        range: {0, 8},
        live: true,
        doc: "How close, in pixels, a neighbour's disparity must be to agree"
      }
    ]
  end

  @impl NxStereo.Stage
  def init(settings, _geometry), do: NxStereo.Stage.init_settings(__MODULE__, settings)

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
