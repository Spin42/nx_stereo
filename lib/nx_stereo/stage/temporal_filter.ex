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

  @defaults [alpha: 0.4, delta: 2.0, hold: 2]

  @impl NxStereo.Stage
  def requires, do: [:disparity]

  @impl NxStereo.Stage
  def provides, do: [:disparity]

  @impl NxStereo.Stage
  def init(settings, geometry) do
    with {:ok, settings} <- NxStereo.Stage.settings(settings, @defaults),
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
