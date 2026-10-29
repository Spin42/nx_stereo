defmodule NxStereo.RectifyTest do
  use ExUnit.Case, async: true

  alias NxStereo.Rectify

  @k [100.0, 0.0, 32.0, 0.0, 100.0, 24.0, 0.0, 0.0, 1.0]
  @identity [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

  defp random_image(height, width, seed) do
    key = Nx.Random.key(seed)
    {image, _key} = Nx.Random.randint(key, 0, 256, shape: {height, width}, type: :s32)
    Nx.as_type(image, :u8)
  end

  defp projection(fx, cx, cy), do: [fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

  test "identity calibration leaves the image unchanged" do
    image = random_image(48, 64, 1)
    camera = Rectify.camera(@k, [], @identity, projection(100.0, 32.0, 24.0))

    assert Nx.to_list(Rectify.rectify(image, camera)) == Nx.to_list(image)
  end

  test "a principal point shift of P moves the image" do
    image = random_image(48, 64, 2)

    camera =
      Rectify.camera(@k, [0.0, 0.0, 0.0, 0.0, 0.0], @identity, projection(100.0, 35.0, 24.0))

    rectified = Rectify.rectify(image, camera)

    assert Nx.to_list(rectified[[.., 3..63]]) == Nx.to_list(image[[.., 0..60]])
    assert Nx.to_list(rectified[[.., 0..2]]) == List.duplicate([0, 0, 0], 48)
  end

  test "radial distortion maps a pixel where the formula says" do
    k1 = 0.5
    {u, v} = {60, 40}
    x = (u - 32) / 100
    y = (v - 24) / 100
    radial = 1 + k1 * (x * x + y * y)
    {source_x, source_y} = {100 * x * radial + 32, 100 * y * radial + 24}

    x0 = floor(source_x)
    y0 = floor(source_y)

    image =
      Nx.broadcast(Nx.tensor(0, type: :u8), {48, 64})
      |> Nx.put_slice([y0, x0], Nx.broadcast(Nx.tensor(255, type: :u8), {2, 2}))

    camera =
      Rectify.camera(@k, [k1, 0.0, 0.0, 0.0, 0.0], @identity, projection(100.0, 32.0, 24.0))

    rectified = Rectify.rectify(image, camera)

    assert Nx.to_number(rectified[v][u]) == 255

    assert Nx.to_number(
             Rectify.rectify(
               image,
               Rectify.camera(@k, [], @identity, projection(100.0, 32.0, 24.0))
             )[v][u]
           ) == 0
  end

  test "maps/2 gives, for each rectified pixel, where it samples the image" do
    image = Nx.broadcast(Nx.tensor(0, type: :u8), {48, 64})
    camera = Rectify.camera(@k, [], @identity, projection(100.0, 35.0, 24.0))
    {map_x, map_y} = Nx.Defn.jit(&Rectify.maps/2).(image, camera)

    assert Nx.to_number(map_x[[10, 20]]) == 17.0
    assert Nx.to_number(map_y[[10, 20]]) == 10.0
  end

  test "refuses distortion beyond plumb_bob's five coefficients" do
    rational = [0.1, 0.01, 0.0, 0.0, 0.001, 0.2, 0.0, 0.0]

    assert_raise ArgumentError, ~r/plumb_bob/, fn ->
      Rectify.camera(@k, rational, @identity, projection(100.0, 32.0, 24.0))
    end

    assert %{d: d} =
             Rectify.camera(
               @k,
               rational |> Enum.take(5) |> Kernel.++([0.0, 0.0, 0.0]),
               @identity,
               projection(100.0, 32.0, 24.0)
             )

    assert Nx.to_list(d) |> length() == 5
  end

  describe "on Vulkan through NxIREE" do
    @describetag :vulkan

    setup do
      {:ok, devices} = NxIREE.list_devices("vulkan")

      devices =
        Enum.filter(devices, &String.contains?(&1.uri, System.get_env("NX_STEREO_DEVICE", "")))

      assert devices != []
      {:ok, devices: devices}
    end

    test "matches the reference within one grey level", %{devices: devices} do
      image = random_image(270, 480, 3)

      camera =
        Rectify.camera(
          [350.0, 0.0, 243.0, 0.0, 351.0, 137.0, 0.0, 0.0, 1.0],
          [0.058, -0.07, 0.0023, 0.0027, 0.0],
          [0.998, -0.011, 0.0604, 0.0106, 0.9999, 0.0067, -0.0605, -0.006, 0.998],
          [379.0, 0.0, 220.0, -34.5, 0.0, 379.0, 139.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        )

      reference = Nx.Defn.jit(&Rectify.rectify/2, compiler: EXLA).(image, camera)

      for device <- devices do
        gpu =
          Nx.Defn.jit(&Rectify.rectify/2,
            compiler: NxIREE.Compiler,
            iree_runtime_options: [device: device]
          ).(image, camera)
          |> Nx.backend_transfer(Nx.BinaryBackend)

        difference =
          Nx.subtract(Nx.as_type(gpu, :s16), Nx.as_type(reference, :s16))
          |> Nx.abs()
          |> Nx.reduce_max()

        assert Nx.to_number(difference) <= 1, "differs on #{device.uri}"
      end
    end
  end
end
