defmodule NxStereo.Stage.TemporalFilter do
  @moduledoc """
  Smooths `:disparity` over successive pairs with
  `NxStereo.Filter.temporal/4`, carrying the previous disparity and each
  pixel's count of missed pairs from one pair to the next.

  ## Settings

    * `:alpha` (`0.4`) - the new value's weight; 1.0 turns smoothing off.
    * `:delta` (`2.0`) - the largest change, in pixels, that is smoothed.
    * `:hold` (`2`) - how many pairs a lost match is held; 0 turns
      holding off.

  All three are params, so they change without compiling again.
  """
  @behaviour NxStereo.Stage

  @impl NxStereo.Stage
  def requires, do: [:disparity]

  @impl NxStereo.Stage
  def provides, do: [:disparity]

  @impl NxStereo.Stage
  def settings do
    [
      alpha: %{
        default: 0.4,
        type: :float,
        range: {0.05, 1},
        live: true,
        doc:
          "Weight of a new disparity where it agrees with the previous one; 1 turns smoothing off"
      },
      delta: %{
        default: 2.0,
        type: :float,
        range: {0, 8},
        live: true,
        doc: "Largest disparity change, in pixels, that is smoothed"
      },
      hold: %{
        default: 2,
        type: :integer,
        range: {0, 10},
        live: true,
        doc: "Pairs a pixel that loses its match keeps its last disparity; 0 turns holding off"
      }
    ]
  end

  @impl NxStereo.Stage
  def init(settings, geometry) do
    with {:ok, settings} <- NxStereo.Stage.init_settings(__MODULE__, settings),
         do: {:ok, Map.put(settings, :size, NxStereo.Stage.size(geometry))}
  end

  @impl NxStereo.Stage
  def params(state, _geometry), do: Map.take(state, [:alpha, :delta, :hold])

  # `missed`: pairs since the pixel last matched.
  @impl NxStereo.Stage
  def carry(state, _geometry) do
    %{
      previous: Nx.broadcast(Nx.f32(-1), state.size),
      missed: Nx.broadcast(Nx.f32(0), state.size)
    }
  end

  @impl NxStereo.Stage
  def step(tensors, params, _state) do
    {filtered, missed} =
      NxStereo.Filter.temporal(tensors.disparity, params.previous, params.missed,
        alpha: params.alpha,
        delta: params.delta,
        hold: params.hold
      )

    tensors
    |> Map.put(:disparity, filtered)
    |> Map.put({:carry, :previous}, filtered)
    |> Map.put({:carry, :missed}, missed)
  end

  @impl NxStereo.Stage
  def native_kind, do: "temporal_filter"

  @impl NxStereo.Stage
  def native_constants, do: %{max_missed: NxStereo.Filter.max_missed() * 1.0}
end
