defmodule NxStereo.Stage.Sharpness do
  @moduledoc """
  Measures each rectified image's sharpness: adds `:sharpness_sums`, the
  sums `NxStereo.Diagnostics.sharpness/1` turns into each camera's
  sharpness, from `:left` and `:right`. It takes no settings.
  """
  @behaviour NxStereo.Stage

  @impl NxStereo.Stage
  def requires, do: [:left, :right]

  @impl NxStereo.Stage
  def provides, do: [:sharpness_sums]

  @impl NxStereo.Stage
  def init(settings, _geometry) do
    with {:ok, _none} <- NxStereo.Stage.settings(settings, []), do: {:ok, nil}
  end

  @impl NxStereo.Stage
  def step(tensors, _params, _state) do
    sums = NxStereo.Diagnostics.sharpness_sums(tensors.left, tensors.right)
    Map.put(tensors, :sharpness_sums, sums)
  end

  @impl NxStereo.Stage
  def native_kind, do: "sharpness"
end
