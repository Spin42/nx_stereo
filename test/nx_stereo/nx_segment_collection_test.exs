defmodule NxStereo.NxSegmentCollectionTest do
  use ExUnit.Case, async: false

  alias NxStereo.NxSegment

  test "collected tensors remain CPU views with EXLA configured globally" do
    fields = [
      {:disparity, Nx.tensor([[-1.0, 7.5]], type: :f32, backend: Nx.BinaryBackend)},
      {:depth_mm, Nx.tensor([[0, 1200]], type: :u16, backend: Nx.BinaryBackend)},
      {:cloud_grid,
       Nx.tensor([[[:nan, :nan, :nan], [1.0, 2.0, 3.0]]], backend: Nx.BinaryBackend)},
      {:cloud_points, Nx.tensor(1, type: :s32, backend: Nx.BinaryBackend)},
      {{:packed, :left}, NxStereo.Packing.pack(<<1, 2, 3, 4>>, 4, 1)}
    ]

    {sections, layout} =
      fields
      |> Enum.with_index()
      |> Enum.map(fn {{key, tensor}, index} ->
        bytes = Nx.to_binary(tensor)
        section = bytes <> :binary.copy(<<0>>, 256 - byte_size(bytes))
        {section, {key, tensor.shape, tensor.type, index * 64}}
      end)
      |> Enum.unzip()

    bundle = Nx.from_binary(IO.iodata_to_binary(sections), :u32, backend: Nx.BinaryBackend)

    segment =
      struct!(NxSegment,
        members: [],
        specs: %{},
        function: nil,
        params: %{},
        carried: %{},
        geometry: %{},
        defn_options: [],
        layout: layout
      )

    previous = Nx.global_default_backend(EXLA.Backend)
    on_exit(fn -> Nx.global_default_backend(previous) end)

    assert {:ok, outputs} = NxSegment.collect(%{published: bundle}, segment, :infinity)

    for {name, expected} <- fields do
      {key, expected} =
        case name do
          {:packed, key} -> {key, NxStereo.Packing.unpack_download(expected)}
          key -> {key, expected}
        end

      actual = Map.fetch!(outputs, key)
      assert %Nx.BinaryBackend{} = actual.data
      assert actual.shape == expected.shape
      assert actual.type == expected.type
      assert Nx.to_binary(actual) == Nx.to_binary(expected)
    end
  end
end
