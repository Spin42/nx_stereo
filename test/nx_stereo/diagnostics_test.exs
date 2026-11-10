defmodule NxStereo.DiagnosticsTest do
  use ExUnit.Case, async: true

  alias NxStereo.Diagnostics

  defp texture(height, width, seed) do
    {image, _key} =
      Nx.Random.randint(Nx.Random.key(seed), 1, 256, shape: {height, width}, type: :s32)

    Nx.as_type(image, :u8)
  end

  defp sharpness(left, right) do
    left |> Nx.Defn.jit(&Diagnostics.sharpness_sums/2).(right) |> Diagnostics.sharpness()
  end

  test "sharpness is the mean squared neighbour difference over the squared mean grey" do
    left = Nx.tensor([[10, 20, 10], [10, 10, 10]], type: :u8)
    right = Nx.tensor([[10, 10, 10], [10, 10, 10]], type: :u8)

    # Left: differences 10, -10, 0, 0 over greys 10, 20, 10, 10.
    %{left: sharp, right: flat} = sharpness(left, right)
    assert_in_delta sharp, 50.0 / (12.5 * 12.5), 1.0e-6
    assert flat == 0.0
  end

  test "sharpness leaves out pairs with a black pixel and reads 0 without any" do
    left = Nx.tensor([[0, 200, 0, 10, 20]], type: :u8)
    right = Nx.broadcast(Nx.u8(0), {1, 5})

    # Only the pair 10, 20 counts.
    %{left: left_sharpness, right: right_sharpness} = sharpness(left, right)
    assert_in_delta left_sharpness, 100.0 / 100.0, 1.0e-6
    assert right_sharpness == 0.0
  end

  test "sharpness takes s32 images as u8 ones" do
    left = texture(16, 24, 1)
    right = texture(16, 24, 2)
    as_s32 = sharpness(Nx.as_type(left, :s32), Nx.as_type(right, :s32))
    assert as_s32 == sharpness(left, right)
  end

  # The left image sees the right one's pixel (y + rows, x - 4) at (y, x), at disparity 4.
  defp aligned_pair(rows) do
    right = texture(72, 96, 3)

    left =
      right
      |> Nx.pad(0, [{0, 8, 0}, {4, 0, 0}])
      |> Nx.slice([rows, 0], [72, 96])

    {left, right, Nx.broadcast(Nx.f32(4.0), {72, 96})}
  end

  defp row_alignment(left, right, disparity, opts \\ []) do
    Nx.Defn.jit(&Diagnostics.row_alignment_sums/3).(left, right, disparity)
    |> Diagnostics.row_alignment(opts)
  end

  test "row alignment finds no offset between rows that line up" do
    {left, right, disparity} = aligned_pair(0)
    %{offset_rows: offset, band_offset_rows: bands} = row_alignment(left, right, disparity)

    assert_in_delta offset, 0.0, 0.05
    assert length(bands) == 4
    for band <- bands, do: assert_in_delta(band, 0.0, 0.05)
  end

  test "row alignment measures how many rows lower the right image's match sits" do
    {left, right, disparity} = aligned_pair(1)
    %{offset_rows: offset} = row_alignment(left, right, disparity)
    assert_in_delta offset, 1.0, 0.1
  end

  test "row alignment gives nothing with too few matched pixels" do
    {left, right, _disparity} = aligned_pair(0)
    unmatched = Nx.broadcast(Nx.f32(-1.0), {72, 96})
    assert row_alignment(left, right, unmatched) == nil
  end

  test "row alignment's band offset is nil where the band matched too little" do
    {left, right, disparity} = aligned_pair(0)
    disparity = Nx.put_slice(disparity, [0, 0], Nx.broadcast(Nx.f32(-1.0), {72, 24}))

    %{band_offset_rows: [first | rest]} = row_alignment(left, right, disparity)
    assert first == nil
    assert Enum.all?(rest, &is_float/1)
  end

  test "disparity statistics measure the matched share, the spread and the jitter" do
    current = Nx.tensor([[2.0, 4.0, -1.0, 6.0]])
    previous = Nx.tensor([[2.5, -1.0, 3.0, 5.0]])

    stats =
      current
      |> Nx.Defn.jit(&Diagnostics.disparity_sums/2).(previous)
      |> Diagnostics.disparity_stats(4)

    assert stats.matched == 0.75
    assert_in_delta stats.mean_px, 4.0, 1.0e-6
    assert_in_delta stats.spread_px, :math.sqrt(8.0 / 3), 1.0e-5
    # Matched in both: 2.0 vs 2.5 and 6.0 vs 5.0.
    assert_in_delta stats.jitter_px, 0.75, 1.0e-6
  end

  test "disparity statistics of an unmatched map are zero" do
    unmatched = Nx.broadcast(Nx.f32(-1.0), {2, 4})

    stats =
      unmatched
      |> Nx.Defn.jit(&Diagnostics.disparity_sums/2).(unmatched)
      |> Diagnostics.disparity_stats(8)

    assert stats == %{matched: 0.0, mean_px: 0.0, spread_px: 0.0, jitter_px: 0.0}
  end
end
