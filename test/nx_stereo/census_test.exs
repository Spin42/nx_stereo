defmodule NxStereo.CensusTest do
  use ExUnit.Case, async: true

  defp pair do
    {texture, _key} = Nx.Random.randint(Nx.Random.key(35), 0, 256, shape: {24, 173}, type: :s32)
    left = Nx.slice(texture, [0, 0], [24, 160])
    right = Nx.slice(texture, [0, 13], [24, 160])
    noise = Nx.iota({24, 160}) |> Nx.remainder(7) |> Nx.subtract(3) |> Nx.multiply(3)
    {left, Nx.add(right, noise) |> Nx.clip(0, 255)}
  end

  test "keeps the sparse descriptor as the band search default" do
    {left, right} = pair()
    base = Nx.broadcast(Nx.s32(8), {24, 160})
    defaults = NxStereo.sgm_band(left, right, base, band: 16, paths: 3)
    explicit = NxStereo.sgm_band(left, right, base, band: 16, paths: 3, census: :sparse_9x7)

    assert Nx.to_binary(explicit) == Nx.to_binary(defaults)
  end

  test "uses a dense descriptor while preserving a known stereo shift" do
    {left, right} = pair()
    base = Nx.broadcast(Nx.s32(8), {24, 160})
    sparse = NxStereo.sgm_band(left, right, base, band: 16, paths: 3)
    dense = NxStereo.sgm_band(left, right, base, band: 16, paths: 3, census: :dense_9x5)
    interior = dense[[4..19, 24..151]] |> Nx.to_flat_list()

    assert Enum.count(interior, &(abs(&1 - 13) < 0.25)) > 0.98 * length(interior)
    refute Nx.to_binary(dense) == Nx.to_binary(sparse)
  end

  test "accepts a dense fine descriptor without changing the coarse search defaults" do
    {left, right} = pair()
    defaults = NxStereo.sgm_coarse_to_fine(left, right, coarse_paths: 3, paths: 3)

    explicit =
      NxStereo.sgm_coarse_to_fine(left, right,
        coarse_paths: 3,
        paths: 3,
        fine_census: :sparse_9x7
      )

    dense =
      NxStereo.sgm_coarse_to_fine(left, right,
        coarse_paths: 3,
        paths: 3,
        fine_census: :dense_9x5
      )

    assert Nx.to_binary(explicit) == Nx.to_binary(defaults)
    assert dense.shape == defaults.shape
    assert Nx.greater(dense, 0) |> Nx.sum() |> Nx.to_number() > 2000
    refute Nx.to_binary(dense) == Nx.to_binary(defaults)
  end

  test "rejects unknown census descriptors at the matching boundary" do
    {left, right} = pair()
    base = Nx.broadcast(Nx.s32(8), {24, 160})

    assert_raise ArgumentError, ~r/:census must be/, fn ->
      NxStereo.sgm_band(left, right, base, census: :unknown)
    end

    assert_raise ArgumentError, ~r/:fine_census must be/, fn ->
      NxStereo.sgm_coarse_to_fine(left, right, fine_census: :unknown)
    end
  end

  test "the blended centre keeps a known stereo shift with independent image noise" do
    {left, right} = pair()
    base = Nx.broadcast(Nx.s32(8), {24, 160})
    blended = NxStereo.sgm_band(left, right, base, band: 16, paths: 3, census: :dense_9x5_blend)
    dense = NxStereo.sgm_band(left, right, base, band: 16, paths: 3, census: :dense_9x5)
    interior = blended[[4..19, 24..151]] |> Nx.to_flat_list()

    assert Enum.count(interior, &(abs(&1 - 13) < 0.25)) > 0.98 * length(interior)
    refute Nx.to_binary(blended) == Nx.to_binary(dense)
  end
end

defmodule NxStereo.CensusVulkanTest do
  use ExUnit.Case, async: true

  @moduletag :vulkan
  setup_all do: NxStereo.TestDevices.vulkan()

  for band <- [8, 16, 32, 64], paths <- [3, 8], census <- [:dense_9x5, :dense_9x5_blend] do
    test "#{census} band #{band} with #{paths} paths preserves reference decisions", %{
      devices: devices
    } do
      {left, _key} = Nx.Random.randint(Nx.Random.key(65), 0, 256, shape: {19, 101}, type: :s32)
      {right, _key} = Nx.Random.randint(Nx.Random.key(82), 0, 256, shape: {19, 101}, type: :s32)

      {base, _key} =
        Nx.Random.randint(Nx.Random.key(29), -1, 129 - unquote(band),
          shape: {19, 101},
          type: :s32
        )

      fun =
        &NxStereo.sgm_band(&1, &2, &3,
          band: unquote(band),
          paths: unquote(paths),
          disparities: 128,
          census: unquote(census)
        )

      reference = Nx.Defn.jit(fun, compiler: EXLA).(left, right, base) |> Nx.to_flat_list()

      for device <- devices do
        result =
          Nx.Defn.jit(fun,
            compiler: NxIREE.Compiler,
            iree_runtime_options: [device: device]
          ).(left, right, base)
          |> Nx.to_flat_list()

        assert Enum.zip(result, reference)
               |> Enum.all?(fn {actual, expected} ->
                 actual < 0 == expected < 0 and abs(actual - expected) <= 1.0e-4
               end),
               device.uri
      end
    end
  end
end
