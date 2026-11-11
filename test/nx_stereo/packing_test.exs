defmodule NxStereo.PackingTest do
  use ExUnit.Case, async: true

  alias NxStereo.Packing

  test "pack then unpack gives the image back" do
    bytes = :rand.bytes(728 * 544)
    image = bytes |> Nx.from_binary(:u8) |> Nx.reshape({544, 728})

    packed = Packing.pack(bytes, 728, 544)
    assert Nx.shape(packed) == {544, 182}
    assert Nx.type(packed) == {:u, 32}

    unpacked = Nx.Defn.jit(&Packing.unpack(&1, {:u, 8})).(packed)
    assert Nx.to_list(unpacked) == Nx.to_list(image)
  end

  test "32-bit pixels packed in a compiled function unpack into the image" do
    bytes = :rand.bytes(16 * 8)
    image = bytes |> Nx.from_binary(:u8) |> Nx.reshape({8, 16})

    packed = Nx.Defn.jit(&Packing.pack_image/1).(Nx.as_type(image, :s32))
    assert Nx.to_list(Packing.unpack_download(packed)) == Nx.to_list(image)
  end

  test "CPU packing stays on the CPU with an EXLA default" do
    bytes = <<0, 1, 127, 255, 255, 127, 1, 0>>

    packed = Nx.with_default_backend(EXLA.Backend, fn -> Packing.pack(bytes, 4, 2) end)

    assert %Nx.BinaryBackend{} = packed.data
    assert Nx.shape(packed) == {2, 1}
    assert Nx.type(packed) == {:u, 32}
    assert Nx.to_binary(packed) == bytes
  end

  test "downloaded words become a CPU byte view with an EXLA default" do
    bytes = <<0, 1, 127, 255, 255, 127, 1, 0>>
    packed = Nx.from_binary(bytes, :u32, backend: Nx.BinaryBackend) |> Nx.reshape({2, 1})

    unpacked = Nx.with_default_backend(EXLA.Backend, fn -> Packing.unpack_download(packed) end)

    assert %Nx.BinaryBackend{} = unpacked.data
    assert Nx.shape(unpacked) == {2, 4}
    assert Nx.type(unpacked) == {:u, 8}
    assert Nx.to_binary(unpacked) == bytes
  end
end
