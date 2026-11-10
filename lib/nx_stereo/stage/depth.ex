defmodule NxStereo.Stage.Depth do
  @moduledoc """
  Turns `:disparity`, in pixels, into metric depth, `Z = f × B / d`,
  with `NxStereo.Depth.depth/3`: `:depth_mm`, `u16` millimetres, and
  `:depth_m`, `f32` metres, which it leaves for the stages after it on
  the device. Unmatched pixels have no depth, 0, and neither has the
  millimetres beyond the 65535 they hold. The focal length and baseline
  are params, so a new calibration needs no new compiled function. It
  takes no settings.
  """
  @behaviour NxStereo.Stage

  @impl NxStereo.Stage
  def requires, do: [:disparity]

  @impl NxStereo.Stage
  def provides, do: [:depth_mm, :depth_m]

  @impl NxStereo.Stage
  def init(settings, _geometry) do
    with {:ok, _none} <- NxStereo.Stage.settings(settings, []), do: {:ok, nil}
  end

  @impl NxStereo.Stage
  def params(_state, geometry),
    do: %{focal_length_px: geometry.focal_length_px, baseline_m: geometry.baseline_m}

  @impl NxStereo.Stage
  def step(tensors, params, _state) do
    {depth_mm, depth_m} =
      NxStereo.Depth.depth(tensors.disparity, params.focal_length_px, params.baseline_m)

    tensors
    |> Map.put(:depth_mm, depth_mm)
    |> Map.put(:depth_m, depth_m)
  end

  @impl NxStereo.Stage
  def device_only, do: [:depth_m]

  @impl NxStereo.Stage
  def native_kind, do: "depth"
end
