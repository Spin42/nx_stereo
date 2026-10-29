defmodule NxStereo.ImageTest do
  use ExUnit.Case, async: true

  alias NxStereo.Image

  defp random_image(height, width, seed) do
    {image, _key} =
      Nx.Random.randint(Nx.Random.key(seed), 0, 256, shape: {height, width}, type: :s32)

    Nx.as_type(image, :u8)
  end

  describe "resize_area/2" do
    test "averages 2x2 blocks when halving" do
      image = Nx.tensor([[0, 4, 8, 8], [4, 8, 8, 8], [10, 10, 1, 3], [10, 10, 3, 1]], type: :u8)

      assert Nx.to_list(Image.resize_area(image, {2, 2})) == [[4, 8], [10, 2]]
    end

    test "keeps a uniform image uniform at a non-integer ratio" do
      image = Nx.broadcast(Nx.tensor(117, type: :u8), {432, 768})
      resized = Nx.Defn.jit(&Image.resize_area(&1, {270, 480})).(image)

      assert Nx.shape(resized) == {270, 480}
      assert resized |> Nx.equal(117) |> Nx.all() |> Nx.to_number() == 1
    end

    test "returns an image of the same size unchanged" do
      image = random_image(12, 16, 1)
      assert Nx.to_list(Image.resize_area(image, {12, 16})) == Nx.to_list(image)
    end

    test "refuses to enlarge" do
      assert_raise ArgumentError, ~r/only shrinks/, fn ->
        Image.resize_area(random_image(4, 4, 2), {8, 8})
      end
    end
  end

  describe "median/2" do
    test "is the middle of the sorted neighbourhood" do
      {values, _key} = Nx.Random.uniform(Nx.Random.key(9), shape: {20, 30})

      for kernel_size <- [3, 5] do
        radius = div(kernel_size, 2)
        padded = Nx.to_list(values)

        expected =
          for y <- 0..19 do
            for x <- 0..29 do
              neighbourhood =
                for dy <- -radius..radius, dx <- -radius..radius do
                  padded |> Enum.at(min(max(y + dy, 0), 19)) |> Enum.at(min(max(x + dx, 0), 29))
                end

              neighbourhood |> Enum.sort() |> Enum.at(div(kernel_size * kernel_size, 2))
            end
          end

        assert Nx.to_list(Image.median(values, kernel_size)) == expected
      end
    end

    test "a 5x5 median removes an outlier and keeps a step edge" do
      flat = Nx.broadcast(Nx.tensor(10.0), {9, 9}) |> Nx.put_slice([4, 4], Nx.tensor([[99.0]]))
      assert flat |> Image.median(5) |> Nx.equal(10.0) |> Nx.all() |> Nx.to_number() == 1

      step = Nx.select(Nx.less(Nx.iota({9, 9}, axis: 1), 4), 1.0, 5.0)
      assert Nx.to_list(Image.median(step, 5)) == Nx.to_list(step)
    end

    test "replicates the border" do
      image = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], type: :u8)
      assert Nx.to_list(Image.median(image, 3)) == [[2, 3, 3], [4, 5, 6], [7, 7, 8]]
    end

    test "takes 3 or 5" do
      assert_raise ArgumentError, ~r/3 or 5/, fn -> Image.median(Nx.iota({8, 8}), 7) end
    end
  end

  describe "turbo/2" do
    test "takes the range as an input of one compiled function" do
      depth = Nx.tensor([[2.0, 4.0]], type: :f32)
      colours = Nx.Defn.jit(&Image.turbo/2)

      at_4 = colours.(depth, Nx.f32(4.0)) |> Nx.to_list()
      at_8 = colours.(depth, Nx.f32(8.0)) |> Nx.to_list()

      assert at_4 == Nx.to_list(Image.turbo(depth, 4.0))
      refute at_4 == at_8
    end

    test "colours near red, far blue and no depth black" do
      depth = Nx.tensor([[0.0, 10.0, 0.001, 25.0]], type: :f32)
      colours = depth |> Image.turbo(10.0) |> Nx.to_list()

      assert colours == [[[0, 0, 0], [48, 18, 59], [122, 4, 3], [48, 18, 59]]]
    end
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

    test "turbo matches the reference", %{devices: devices} do
      depth = Nx.iota({27, 48}, type: :f32) |> Nx.divide(100)
      function = &Image.turbo(&1, 10.0)
      reference = Nx.Defn.jit(function, compiler: EXLA).(depth) |> Nx.to_list()

      for device <- devices do
        gpu =
          Nx.Defn.jit(function, compiler: NxIREE.Compiler, iree_runtime_options: [device: device]).(
            depth
          )
          |> Nx.backend_transfer(Nx.BinaryBackend)

        assert Nx.to_list(gpu) == reference, "differs on #{device.uri}"
      end
    end

    test "resize and median match the reference", %{devices: devices} do
      image = random_image(432, 768, 3)
      function = fn image -> image |> Image.resize_area({270, 480}) |> Image.median(5) end
      reference = Nx.Defn.jit(function, compiler: EXLA).(image) |> Nx.to_list()

      for device <- devices do
        gpu =
          Nx.Defn.jit(function, compiler: NxIREE.Compiler, iree_runtime_options: [device: device]).(
            image
          )
          |> Nx.backend_transfer(Nx.BinaryBackend)

        assert Nx.to_list(gpu) == reference, "differs on #{device.uri}"
      end
    end
  end
end
