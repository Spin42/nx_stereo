defmodule NxStereo.Stage.DisparityTest do
  use ExUnit.Case, async: true

  alias NxStereo.Stage.Disparity

  @geometry %{width: 96, height: 16}

  test "uses the configured coarse-to-fine band and margin" do
    {texture, _key} = Nx.Random.randint(Nx.Random.key(17), 0, 256, shape: {16, 113}, type: :s32)
    left = Nx.slice(texture, [0, 0], [16, 96])
    right = Nx.slice(texture, [0, 17], [16, 96])

    settings = [
      search: "coarse_to_fine",
      num_disparities: 128,
      coarse_paths: 3,
      band: 8,
      margin: 0
    ]

    assert {:ok, state} = Disparity.init(settings, @geometry)
    params = Disparity.params(state, @geometry)

    actual =
      Nx.Defn.jit(fn left, right ->
        Disparity.step(%{left: left, right: right}, params, state).disparity
      end).(left, right)

    expected =
      Nx.Defn.jit(fn left, right ->
        disparity = NxStereo.sgm_coarse_to_fine(left, right, coarse_paths: 3, band: 8, margin: 0)
        Nx.select(Nx.greater(disparity, 0), disparity, -1.0)
      end).(left, right)

    assert Nx.to_binary(actual) == Nx.to_binary(expected)
    assert Enum.any?(Nx.to_flat_list(actual), &(&1 > 0))
  end

  test "declares the fine search's settings as applying to coarse to fine only" do
    settings = Disparity.settings()

    for key <- [:coarse_paths, :band, :margin, :fine_census],
        do: assert(settings[key].when == [search: "coarse_to_fine"], inspect(key))

    refute Map.has_key?(settings[:paths], :when)
  end

  test "rejects unsupported bands before compiling the segment" do
    settings = [search: "coarse_to_fine", num_disparities: 128, band: 12]

    assert {:error, message} = Disparity.init(settings, @geometry)
    assert message =~ "band must be one of [8, 16, 32, 64]"
  end

  test "rejects margins that do not keep the coarse estimate inside the band" do
    for margin <- [-1, 16, 1.5] do
      settings = [search: "coarse_to_fine", num_disparities: 128, band: 16, margin: margin]

      assert {:error, message} = Disparity.init(settings, @geometry)
      assert message =~ "margin must be"
    end
  end

  test "forwards both optional fine descriptors to the matcher" do
    {texture, _key} = Nx.Random.randint(Nx.Random.key(71), 0, 256, shape: {16, 113}, type: :s32)
    left = Nx.slice(texture, [0, 0], [16, 96])
    noise = Nx.iota({16, 96}) |> Nx.remainder(7) |> Nx.subtract(3)
    right = Nx.slice(texture, [0, 17], [16, 96]) |> Nx.add(noise) |> Nx.clip(0, 255)

    for census <- [:dense_9x5, :dense_9x5_blend] do
      settings = [
        search: "coarse_to_fine",
        num_disparities: 128,
        coarse_paths: 3,
        paths: 3,
        fine_census: census
      ]

      assert {:ok, state} = Disparity.init(settings, @geometry)
      params = Disparity.params(state, @geometry)

      {actual, expected} =
        Nx.Defn.jit(fn left, right ->
          actual = Disparity.step(%{left: left, right: right}, params, state).disparity

          disparity =
            NxStereo.sgm_coarse_to_fine(left, right,
              coarse_paths: 3,
              paths: 3,
              fine_census: census
            )

          {actual, Nx.select(Nx.greater(disparity, 0), disparity, -1.0)}
        end).(left, right)

      assert Nx.to_binary(actual) == Nx.to_binary(expected)
    end
  end

  test "rejects unsupported or unused fine descriptors at stage initialization" do
    assert {:error, message} =
             Disparity.init(
               [search: "coarse_to_fine", num_disparities: 128, fine_census: :unknown],
               @geometry
             )

    assert message =~ "fine_census must be"

    assert {:error, message} =
             Disparity.init([search: "full", fine_census: :dense_9x5], @geometry)

    assert message =~ "fine_census requires coarse_to_fine"
  end
end
