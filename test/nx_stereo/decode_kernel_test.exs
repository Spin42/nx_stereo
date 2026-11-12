defmodule NxStereo.DecodeKernelTest do
  use ExUnit.Case, async: true

  alias NxStereo.{Kernels, Packing}

  @moduletag :vulkan
  setup_all do: NxStereo.TestDevices.vulkan()

  test "the half-size kernel averages every byte sum and handles unaligned rows", %{
    devices: devices
  } do
    for {height, width} = size <- [{1, 1024}, {3, 5}, {7, 126}] do
      totals = for pixel <- 0..(height * width - 1), do: rem(pixel * 17 + 3, 1021)
      indexed = List.to_tuple(totals)

      pixels =
        for row <- 0..(2 * height - 1), column <- 0..(2 * width - 1) do
          total = elem(indexed, div(row, 2) * width + div(column, 2))
          lane = rem(row, 2) * 2 + rem(column, 2)
          min(max(total - 255 * lane, 0), 255)
        end

      expected = Enum.map(totals, &div(&1 + 2, 4))
      left = Nx.tensor(pixels, type: :u8) |> Nx.reshape({2 * height, 2 * width})
      right = Nx.reverse(left, axes: [0, 1])
      zero = fn _inputs -> {Nx.broadcast(Nx.s32(0), size), Nx.broadcast(Nx.s32(0), size)} end

      for packed <- [false, true], not packed or rem(width, 2) == 0, device <- devices do
        inputs =
          for frame <- [left, right] do
            if packed,
              do: Packing.pack(Nx.to_binary(frame), 2 * width, 2 * height),
              else: frame
          end

        {decoded_left, decoded_right} =
          Nx.Defn.jit(&Kernels.decode_half(&1, &2, size, zero),
            compiler: NxIREE.Compiler,
            iree_runtime_options: [device: device]
          )
          |> apply(inputs)
          |> Nx.backend_transfer(Nx.BinaryBackend)

        assert Nx.type(decoded_left) == {:s, 32}
        assert Nx.type(decoded_right) == {:s, 32}
        assert Nx.shape(decoded_left) == size
        assert Nx.shape(decoded_right) == size
        assert Nx.to_flat_list(decoded_left) == expected, device.uri
        assert Nx.to_flat_list(decoded_right) == Enum.reverse(expected), device.uri
      end
    end
  end
end
