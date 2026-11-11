defmodule NxStereo.Stage.MedianFilter do
  @moduledoc """
  Filters `:disparity` with a median of each pixel's neighbourhood, with
  `NxStereo.Image.median/2`.

  ## Settings

    * `:kernel_size` (`5`) - the window's width, 3 or 5, fixed when
      compiled.
  """
  @behaviour NxStereo.Stage

  @impl NxStereo.Stage
  def requires, do: [:disparity]

  @impl NxStereo.Stage
  def provides, do: [:disparity]

  @impl NxStereo.Stage
  def settings do
    [
      kernel_size: %{
        default: 5,
        type: :integer,
        values: [3, 5],
        doc: "Width of the median filter's square window, in pixels"
      }
    ]
  end

  @impl NxStereo.Stage
  def init(settings, _geometry), do: NxStereo.Stage.init_settings(__MODULE__, settings)

  @impl NxStereo.Stage
  def step(tensors, _params, state) do
    filtered = NxStereo.Image.median(tensors.disparity, state.kernel_size)
    Map.put(tensors, :disparity, filtered)
  end

  @impl NxStereo.Stage
  def native_kind, do: "median_filter"

  @impl NxStereo.Stage
  def native_settings(state), do: [kernel_size: state.kernel_size]
end
