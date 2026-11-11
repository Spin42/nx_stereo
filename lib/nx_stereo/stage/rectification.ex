defmodule NxStereo.Stage.Rectification do
  @moduledoc """
  Undistorts and rectifies both camera frames, `:left_frame` and
  `:right_frame`, into `:left` and `:right` with `NxStereo.Rectify`, so a
  point lies on the same row in each, shrinking frames larger than the
  working size in the same pass. Frames packed as 32-bit words
  (`NxStereo.Packing`) give `s32` images and their packing,
  `{:packed, :left}` and `{:packed, :right}`. Each camera, from the
  geometry's `:left_camera` and `:right_camera`, is a param, so a new
  calibration needs no new compiled function. It refuses frames smaller
  than the working size.

  ## Settings

    * `:right_row_offset` (`0.0`) - rows the right image is moved by, a
      correction for cameras that have shifted since their calibration; a
      param.
  """
  @behaviour NxStereo.Stage

  @impl NxStereo.Stage
  def requires, do: [:left_frame, :right_frame]

  @impl NxStereo.Stage
  def provides, do: [:left, :right]

  @impl NxStereo.Stage
  def settings do
    [
      right_row_offset: %{
        default: 0.0,
        type: :float,
        range: {-8, 8},
        live: true,
        doc: "Rows the right image is moved by to line up with the left one"
      }
    ]
  end

  @impl NxStereo.Stage
  def init(settings, geometry) do
    with :ok <- check_frame_size(geometry),
         {:ok, settings} <- NxStereo.Stage.init_settings(__MODULE__, settings),
         do: {:ok, Map.put(settings, :size, NxStereo.Stage.size(geometry))}
  end

  @impl NxStereo.Stage
  def params(state, geometry) do
    right = move_rows(geometry.right_camera, state.right_row_offset)
    %{left: geometry.left_camera, right: right}
  end

  @impl NxStereo.Stage
  def step(tensors, params, %{size: size}) do
    tensors
    |> put_image(:left, rectify(tensors.left_frame, params.left, size))
    |> put_image(:right, rectify(tensors.right_frame, params.right, size))
  end

  @impl NxStereo.Stage
  def native_kind, do: "rectification"

  defp put_image(tensors, key, {image, packed}),
    do: tensors |> Map.put(key, image) |> Map.put({:packed, key}, packed)

  defp put_image(tensors, key, image), do: Map.put(tensors, key, image)

  # A frame packed as 32-bit words gives the s32 image and its packing.
  defp rectify(frame, camera, size) do
    if Nx.type(frame) == {:u, 32},
      do: NxStereo.Rectify.rectify_packed(frame, camera, size: size),
      else: NxStereo.Rectify.rectify(frame, camera, size: size)
  end

  defp check_frame_size(geometry) do
    if geometry.frame_width < geometry.width or geometry.frame_height < geometry.height,
      do:
        {:error,
         "rectification only shrinks: #{geometry.frame_width}x#{geometry.frame_height} frames " <>
           "are smaller than the #{geometry.width}x#{geometry.height} working size"},
      else: :ok
  end

  # Sampling row v + rows: the inverse projection's row column, times rows, joins its constant one.
  defp move_rows(camera, rows) do
    inverse = camera.inverse_pr
    constant = Nx.add(inverse[[.., 2]], Nx.multiply(rows, inverse[[.., 1]]))
    %{camera | inverse_pr: Nx.put_slice(inverse, [0, 2], Nx.reshape(constant, {3, 1}))}
  end
end
