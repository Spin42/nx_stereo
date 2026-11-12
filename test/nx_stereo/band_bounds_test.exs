defmodule NxStereo.BandBoundsTest do
  use ExUnit.Case, async: true

  @moduletag timeout: 180_000

  defp pair_and_base(band) do
    shape = {19, 101}
    {left, key} = Nx.Random.randint(Nx.Random.key(77), 0, 256, shape: shape, type: :s32)
    {right, key} = Nx.Random.randint(key, 0, 256, shape: shape, type: :s32)
    {base, _key} = Nx.Random.randint(key, -1, 129 - band, shape: shape, type: :s32)
    {left, right, base}
  end

  defp match(band, paths, options) do
    {left, right, base} = pair_and_base(band)
    function = &NxStereo.sgm_band(&1, &2, &3, band: band, paths: paths, uniqueness: 1.0)
    Nx.Defn.jit(function, options).(left, right, base) |> Nx.backend_transfer(Nx.BinaryBackend)
  end

  defp assert_in_bounds(disparity) do
    assert Nx.shape(disparity) == {19, 101}
    assert Nx.type(disparity) == {:f, 32}
    pixels = Nx.to_flat_list(disparity)
    assert Enum.any?(pixels, &(is_number(&1) and &1 > 0))

    invalid =
      pixels
      |> Enum.with_index()
      |> Enum.filter(fn {disparity, index} ->
        not is_number(disparity) or
          (disparity != -1.0 and (disparity < 0 or disparity > min(rem(index, 101), 127)))
      end)

    assert invalid == [], "matches outside the right image: #{inspect(Enum.take(invalid, 8))}"
  end

  for band <- [8, 16, 32, 64], paths <- [3, 8] do
    test "Nx rejects impossible candidates in band #{band} with #{paths} paths" do
      match(unquote(band), unquote(paths), compiler: EXLA) |> assert_in_bounds()
    end
  end

  describe "on Vulkan" do
    @describetag :vulkan
    setup do: NxStereo.TestDevices.vulkan()

    for band <- [8, 16, 32, 64], paths <- [3, 8] do
      test "rejects impossible candidates in band #{band} with #{paths} paths", %{
        devices: devices
      } do
        for device <- devices do
          match(unquote(band), unquote(paths),
            compiler: NxIREE.Compiler,
            iree_runtime_options: [device: device]
          )
          |> assert_in_bounds()
        end
      end
    end
  end
end
