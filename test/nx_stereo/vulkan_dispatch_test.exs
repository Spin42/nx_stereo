defmodule NxStereo.VulkanDispatchTest do
  use ExUnit.Case, async: true

  @moduletag :vulkan
  setup_all do: NxStereo.TestDevices.vulkan()

  test "selects wide bands when a one-dimensional dispatch would exceed Vulkan's limit", %{
    devices: devices
  } do
    {texture, _key} =
      Nx.Random.randint(Nx.Random.key(37), 0, 256, shape: {256, 1031}, type: :s32)

    left = Nx.slice(texture, [0, 0], [256, 1024])
    right = Nx.slice(texture, [0, 7], [256, 1024])
    base = Nx.broadcast(Nx.s32(0), {256, 1024})

    for device <- devices do
      disparity =
        Nx.Defn.jit(&NxStereo.sgm_band(&1, &2, &3, disparities: 64, band: 64, paths: 3),
          compiler: NxIREE.Compiler,
          iree_runtime_options: [device: device]
        ).(left, right, base)
        |> Nx.backend_transfer(Nx.BinaryBackend)

      matched = disparity |> Nx.slice([4, 70], [248, 950]) |> Nx.round() |> Nx.equal(7)
      assert Nx.all(matched) |> Nx.to_number() == 1
    end
  end
end
