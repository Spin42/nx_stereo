defmodule NxStereo.NxSegmentInputsTest do
  use ExUnit.Case, async: true

  alias NxStereo.{NxSegment, Rectify, Segment, Stage}

  @moduletag :vulkan
  setup_all do: NxStereo.TestDevices.vulkan()

  test "shared calibration and source frames survive placement and reload", %{devices: devices} do
    {camera, frame} =
      Nx.with_default_backend(EXLA.Backend, fn ->
        k = [10.0, 0.0, 4.0, 0.0, 10.0, 4.0, 0.0, 0.0, 1.0]
        identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        p = [10.0, 0.0, 4.0, 0.0, 0.0, 10.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        {Rectify.camera(k, [], identity, p), Nx.iota({8, 8}, type: :u8)}
      end)

    geometry = %{
      width: 8,
      height: 8,
      frame_width: 8,
      frame_height: 8,
      left_camera: camera,
      right_camera: camera
    }

    calibration = Nx.Defn.Composite.traverse(camera, &Nx.to_binary/1)
    expected = Nx.to_binary(frame)
    {:ok, members} = Segment.members([Stage.Rectification], geometry)

    for device <- devices do
      {:ok, segment} =
        NxSegment.build(members, geometry,
          defn_options: [compiler: NxIREE.Compiler, iree_runtime_options: [device: device]]
        )

      for current <- [segment, NxSegment.reload(segment, members, geometry)] do
        for _iteration <- 1..2 do
          {:ok, output, _segment} =
            Segment.run(NxSegment, [left_frame: frame, right_frame: frame], current, :infinity)

          assert Nx.to_binary(output.left) == expected
          assert Nx.to_binary(output.right) == expected
          assert Nx.to_binary(frame) == expected
          assert Nx.Defn.Composite.traverse(camera, &Nx.to_binary/1) == calibration
        end
      end
    end
  end
end
