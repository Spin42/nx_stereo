defmodule NxStereo.Stage.Decode do
  @moduledoc """
  Turns both camera frames, `:left_frame` and `:right_frame`, into grey
  images at the working size, `:left` and `:right`, with an area resize
  (`NxStereo.Image.resize_area/2`): for frames that are already
  rectified. It takes no settings.
  """
  @behaviour NxStereo.Stage

  @impl NxStereo.Stage
  def requires, do: [:left_frame, :right_frame]

  @impl NxStereo.Stage
  def provides, do: [:left, :right]

  @impl NxStereo.Stage
  def init(settings, geometry) do
    with {:ok, _none} <- NxStereo.Stage.settings(settings, []),
         do: {:ok, %{size: {geometry.height, geometry.width}}}
  end

  @impl NxStereo.Stage
  def step(tensors, _params, %{size: size}) do
    tensors
    |> Map.put(:left, NxStereo.Image.resize_area(tensors.left_frame, size))
    |> Map.put(:right, NxStereo.Image.resize_area(tensors.right_frame, size))
  end
end
