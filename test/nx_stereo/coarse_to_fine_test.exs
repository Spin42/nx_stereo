defmodule NxStereo.CoarseToFineTest do
  use ExUnit.Case, async: true

  @moduletag :vulkan
  setup_all do: NxStereo.TestDevices.vulkan()

  test "byte images with partial reduction tiles match the reference on repeated pairs", %{
    devices: devices
  } do
    {texture, _key} =
      Nx.Random.randint(Nx.Random.key(37), 0, 256, shape: {128, 516}, type: :s32)

    left = texture |> Nx.slice([0, 0], [128, 258]) |> Nx.as_type(:u8)
    right = texture |> Nx.slice([0, 258], [128, 258]) |> Nx.as_type(:u8)
    match = &NxStereo.sgm_coarse_to_fine(&1, &2, coarse_paths: 3)
    reference = Nx.Defn.jit(match, compiler: EXLA).(left, right)

    for device <- devices do
      compiled =
        Nx.Defn.jit(match,
          compiler: NxIREE.Compiler,
          iree_runtime_options: [device: device]
        )

      for _pair <- 1..3 do
        disparity = compiled.(left, right) |> Nx.backend_transfer(Nx.BinaryBackend)
        error = Nx.abs(Nx.subtract(disparity, reference))
        assert Nx.reduce_max(error) |> Nx.to_number() < 1.0e-3
      end
    end
  end
end
