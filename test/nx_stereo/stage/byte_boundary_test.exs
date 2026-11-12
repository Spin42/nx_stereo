defmodule NxStereo.Stage.ByteBoundaryTest do
  use ExUnit.Case, async: true

  alias NxStereo.{Rectify, Stage}

  @moduletag :vulkan
  setup_all do: NxStereo.TestDevices.vulkan()

  defp run(stage, inputs, geometry, device) do
    {:ok, state} = stage.init([], geometry)
    params = Stage.params(stage, state, geometry)

    Nx.Defn.jit(fn inputs, params -> stage.step(inputs, params, state) end,
      compiler: NxIREE.Compiler,
      iree_runtime_options: [device: device]
    ).(inputs, params)
    |> Nx.backend_transfer(Nx.BinaryBackend)
  end

  test "sharpness counts the last neighbours of a partial byte word", %{devices: devices} do
    for device <- devices,
        {height, width} <- [{1, 2}, {1, 3}, {1, 5}, {1, 6}, {1, 7}, {3, 5}, {4, 5}] do
      pixels = height * width
      left = Nx.tensor(Enum.to_list(1..pixels), type: :u8) |> Nx.reshape({height, width})
      right = Nx.add(left, 10)
      pairs = height * (width - 1)
      grey = div(pixels * (pixels + 1) - width * height * (height + 1), 2)
      expected = [pairs, pairs, grey, grey + 10 * pairs, pairs, pairs]

      outputs =
        run(Stage.Sharpness, %{left: left, right: right}, %{height: height, width: width}, device)

      assert Nx.to_flat_list(outputs.sharpness_sums) == expected,
             "#{device.uri}, #{height}x#{width}"
    end
  end

  test "alignment reads the last row of a partial byte word", %{devices: devices} do
    for device <- devices, {height, width} <- [{1, 5}, {1, 6}, {1, 7}, {2, 5}, {3, 5}, {5, 7}] do
      image = Nx.broadcast(Nx.u8(100), {height, width})
      disparity = Nx.broadcast(Nx.f32(0.25), {height, width})
      inputs = %{left: image, right: image, disparity: disparity}
      matches = div(height + 3, 4) * div(width, 4)

      outputs = run(Stage.RowAlignment, inputs, %{height: height, width: width}, device)
      expected = List.duplicate([0, 0, 0, 0, 0, matches], 4)
      assert Nx.to_list(outputs.alignment_sums) == expected, "#{device.uri}, #{height}x#{width}"
    end
  end

  test "rectification samples a source frame ending in a partial byte word", %{devices: devices} do
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    projection = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    camera = Rectify.camera(identity, [], identity, projection)

    for device <- devices, {height, width} <- [{1, 5}, {1, 6}, {1, 7}, {3, 5}, {4, 5}] do
      frame = Nx.broadcast(Nx.u8(73), {height, width})

      geometry = %{
        height: height,
        width: 4,
        frame_height: height,
        frame_width: width,
        left_camera: camera,
        right_camera: camera
      }

      outputs =
        run(Stage.Rectification, %{left_frame: frame, right_frame: frame}, geometry, device)

      assert Nx.to_list(outputs.left) == List.duplicate([73, 73, 73, 73], height),
             "#{device.uri}, #{height}x#{width}"

      assert Nx.to_list(outputs.right) == Nx.to_list(outputs.left)
    end
  end
end
