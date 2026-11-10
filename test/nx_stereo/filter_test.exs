defmodule NxStereo.FilterTest do
  use ExUnit.Case, async: true

  alias NxStereo.Filter

  # A surface of disparity 10 with one isolated pixel at 30 in it, and a lone match outside.
  defp speckled do
    Nx.broadcast(Nx.f32(10.0), {12, 12})
    |> Nx.put_slice([0, 8], Nx.broadcast(Nx.f32(-1.0), {12, 4}))
    |> Nx.put_slice([5, 3], Nx.tensor([[30.0]]))
    |> Nx.put_slice([6, 10], Nx.tensor([[12.0]]))
  end

  test "despeckle drops pixels too few neighbours agree with and keeps surfaces" do
    filtered = Nx.Defn.jit(&Filter.despeckle(&1, min_support: 12, max_diff: 1.0)).(speckled())

    assert Nx.to_number(filtered[[5, 3]]) == -1.0
    assert Nx.to_number(filtered[[6, 10]]) == -1.0
    assert Nx.to_number(filtered[[0, 0]]) == 10.0
    assert Nx.to_number(filtered[[11, 7]]) == 10.0
    assert Nx.to_number(filtered[[0, 9]]) == -1.0
  end

  test "despeckle with no support required keeps every match" do
    filtered = Nx.Defn.jit(&Filter.despeckle(&1, min_support: 0)).(speckled())
    assert Nx.to_list(filtered) == Nx.to_list(speckled())
  end

  test "despeckle counts neighbours within max_diff and its radius" do
    # Within 1.0 of 30.0: neighbours at 29.5 agree, 31.5 do not.
    image =
      Nx.broadcast(Nx.f32(-1.0), {9, 9})
      |> Nx.put_slice([3, 3], Nx.broadcast(Nx.f32(29.5), {3, 3}))
      |> Nx.put_slice([4, 4], Nx.tensor([[30.0]]))

    kept = fn opts ->
      Nx.Defn.jit(&Filter.despeckle(&1, opts)).(image)[[4, 4]] |> Nx.to_number()
    end

    assert kept.(min_support: 8, max_diff: 1.0) == 30.0
    assert kept.(min_support: 9, max_diff: 1.0) == -1.0
    assert kept.(min_support: 1, max_diff: 0.25) == -1.0
    # Radius 1's window holds only the 8 neighbours.
    assert kept.(min_support: 8, max_diff: 1.0, radius: 1) == 30.0
  end

  test "despeckle takes its settings as tensors" do
    filtered =
      Nx.Defn.jit(&Filter.despeckle(&1, min_support: &2, max_diff: &3)).(
        speckled(),
        Nx.s32(12),
        Nx.f32(1.0)
      )

    assert Nx.to_number(filtered[[5, 3]]) == -1.0
  end

  defp temporal(current, previous, missed, opts) do
    {filtered, missed} =
      Nx.Defn.jit(&Filter.temporal(&1, &2, &3, opts)).(
        Nx.tensor([current], type: :f32),
        Nx.tensor([previous], type: :f32),
        Nx.tensor([missed], type: :f32)
      )

    {hd(Nx.to_list(filtered)), hd(Nx.to_list(missed))}
  end

  test "temporal smooths small changes and takes large ones" do
    {filtered, missed} =
      temporal([12.0, 20.0, 5.0, -1.0], [10.0, 10.0, -1.0, -1.0], [0, 0, 3, 0],
        alpha: 0.5,
        delta: 2.0,
        hold: 0
      )

    assert filtered == [11.0, 20.0, 5.0, -1.0]
    assert missed == [0.0, 0.0, 0.0, 1.0]
  end

  test "temporal holds a lost match for hold pairs" do
    opts = [alpha: 0.4, delta: 2.0, hold: 2]
    assert temporal([-1.0], [8.0], [1.0], opts) == {[8.0], [2.0]}
    assert temporal([-1.0], [8.0], [2.0], opts) == {[-1.0], [3.0]}
  end

  test "temporal stops counting misses at max_missed/0" do
    assert Filter.max_missed() == 255
    assert temporal([-1.0], [-1.0], [255.0], hold: 2) == {[-1.0], [255.0]}
  end
end
