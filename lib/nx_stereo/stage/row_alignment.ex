defmodule NxStereo.Stage.RowAlignment do
  @moduledoc """
  Measures the row offset between the rectified images: adds
  `:alignment_sums`, the sums `NxStereo.Diagnostics.row_alignment/2`
  turns into the offset over the image and per vertical band, from
  `:left`, `:right` and `:disparity`, with the diagnostics' default
  bands, reach and row step. It takes no settings.
  """
  @behaviour NxStereo.Stage

  @bands 4
  @reach 2
  @row_step 4

  @impl NxStereo.Stage
  def requires, do: [:left, :right, :disparity]

  @impl NxStereo.Stage
  def provides, do: [:alignment_sums]

  @impl NxStereo.Stage
  def init(settings, _geometry) do
    with {:ok, _none} <- NxStereo.Stage.settings(settings, []), do: {:ok, nil}
  end

  @impl NxStereo.Stage
  def step(tensors, _params, _state) do
    sums =
      NxStereo.Diagnostics.row_alignment_sums(tensors.left, tensors.right, tensors.disparity,
        bands: @bands,
        reach: @reach,
        row_step: @row_step
      )

    Map.put(tensors, :alignment_sums, sums)
  end

  @impl NxStereo.Stage
  def native_kind, do: "row_alignment"

  @impl NxStereo.Stage
  def native_constants, do: %{bands: @bands, reach: @reach, row_step: @row_step}
end
