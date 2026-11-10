defmodule NxStereo.Stage.Quality do
  @moduledoc """
  Measures `:disparity` on every pair: adds `:quality_sums`, the sums
  `NxStereo.Diagnostics.disparity_stats/2` turns into the matched share,
  the mean and spread of the disparity and its jitter, carrying the
  previous disparity from one pair to the next. It takes no settings.
  """
  @behaviour NxStereo.Stage

  @impl NxStereo.Stage
  def requires, do: [:disparity]

  @impl NxStereo.Stage
  def provides, do: [:quality_sums]

  @impl NxStereo.Stage
  def init(settings, geometry) do
    with {:ok, _none} <- NxStereo.Stage.settings(settings, []),
         do: {:ok, %{size: {geometry.height, geometry.width}}}
  end

  @impl NxStereo.Stage
  def carry(state, _geometry), do: %{previous: Nx.broadcast(Nx.f32(-1), state.size)}

  @impl NxStereo.Stage
  def step(tensors, params, _state) do
    current = tensors.disparity

    tensors
    |> Map.put(:quality_sums, NxStereo.Diagnostics.disparity_sums(current, params.previous))
    |> Map.put({:carry, :previous}, current)
  end

  @impl NxStereo.Stage
  def native_kind, do: "quality"
end
