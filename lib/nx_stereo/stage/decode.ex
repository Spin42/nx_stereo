defmodule NxStereo.Stage.Decode do
  @moduledoc """
  Turns both camera frames, `:left_frame` and `:right_frame`, into grey
  images at the working size, `:left` and `:right`, with an area resize
  (`NxStereo.Image.resize_area/2`): for frames that are already
  rectified. Packed `u32` frames give `s32` pixels for devices without
  byte storage. Exact twofold resizing on Vulkan averages the camera
  words directly; other sizes use the area resizer. It takes no settings.
  """
  @behaviour NxStereo.Stage

  @impl NxStereo.Stage
  def requires, do: [:left_frame, :right_frame]

  @impl NxStereo.Stage
  def provides, do: [:left, :right]

  @impl NxStereo.Stage
  def init(settings, geometry) do
    with {:ok, _none} <- NxStereo.Stage.settings(settings, []),
         do: {:ok, %{size: NxStereo.Stage.size(geometry)}}
  end

  @impl NxStereo.Stage
  def step(tensors, _params, %{size: size}) do
    {left, right} = resize_pair(tensors.left_frame, tensors.right_frame, size)
    tensors |> Map.put(:left, left) |> Map.put(:right, right)
  end

  defp resize_pair(left, right, {height, width} = size) do
    type = Nx.type(left)
    {frame_height, columns} = Nx.shape(left)
    frame_width = columns * if(type == {:u, 32}, do: 4, else: 1)

    if type in [{:u, 8}, {:u, 32}] and Nx.type(right) == type and
         Nx.shape(right) == Nx.shape(left) and
         {frame_height, frame_width} == {2 * height, 2 * width} do
      {left, right} =
        NxStereo.Kernels.decode_half(left, right, size, fn [left, right] ->
          {resize(left, size) |> Nx.as_type(:s32), resize(right, size) |> Nx.as_type(:s32)}
        end)

      output_type = if type == {:u, 32}, do: :s32, else: :u8
      {Nx.as_type(left, output_type), Nx.as_type(right, output_type)}
    else
      {resize(left, size), resize(right, size)}
    end
  end

  defp resize(frame, size) do
    image =
      if Nx.type(frame) == {:u, 32},
        do: NxStereo.Packing.unpack(frame, :s32),
        else: frame

    NxStereo.Image.resize_area(image, size)
  end
end
