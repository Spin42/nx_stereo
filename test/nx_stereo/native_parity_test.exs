defmodule NxStereo.NativeParityTest do
  # The native backend against the Nx code, stage by stage and chained, on
  # the stereo_match nx_stereo builds when cargo is installed.
  use ExUnit.Case, async: false

  alias NxStereo.{NativeSegment, NxSegment, Rectify, Segment, Stage}

  @moduletag :native

  setup_all do
    {:ok, matcher} = NxStereo.Matcher.start_link(threads: 2)
    {:ok, matcher: matcher}
  end

  @k [60.0, 0.0, 32.0, 0.0, 60.0, 24.0, 0.0, 0.0, 1.0]
  @identity [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

  defp geometry do
    p = fn tx -> [60.0, 0.0, 32.0, tx, 0.0, 60.0, 24.0, 0.0, 0.0, 0.0, 1.0, 0.0] end

    %{
      width: 64,
      height: 48,
      frame_width: 64,
      frame_height: 48,
      focal_length_px: 60.0,
      baseline_m: 0.1,
      principal_point: {32.0, 24.0},
      left_camera: Rectify.camera(@k, [0.02], @identity, p.(0.0)),
      right_camera: Rectify.camera(@k, [0.02], @identity, p.(-6.0))
    }
  end

  defp members(stages) do
    for {stage, settings} <- stages do
      {:ok, state} = stage.init(settings, geometry())
      %{module: stage, name: stage, state: state}
    end
  end

  defp texture(height, width, seed) do
    {image, _key} =
      Nx.Random.randint(Nx.Random.key(seed), 0, 256, shape: {height, width}, type: :s32)

    image |> Nx.as_type(:u8) |> Nx.backend_transfer(Nx.BinaryBackend)
  end

  # Frames of a textured scene the right camera sees 8 pixels to the left.
  defp frames(seed) do
    scene = texture(48, 72, seed)
    [left_frame: scene[[.., 0..63]], right_frame: scene[[.., 8..71]]]
  end

  # A disparity of slanted surfaces, with holes and isolated wrong pixels.
  defp disparity(seed) do
    key = Nx.Random.key(seed)
    {noise, key} = Nx.Random.uniform(key, shape: {48, 64})
    {wrong, _key} = Nx.Random.uniform(key, 0.0, 40.0, shape: {48, 64})
    surface = Nx.add(Nx.divide(Nx.iota({48, 64}, axis: 1), 8), 4.0)
    surface = Nx.select(Nx.less(noise, 0.05), wrong, surface)

    Nx.select(Nx.greater(noise, 0.9), -1.0, surface)
    |> Nx.as_type(:f32)
    |> Nx.backend_transfer(Nx.BinaryBackend)
  end

  defp run(module, segment, inputs) do
    {outputs, _segment} =
      Enum.map_reduce(inputs, segment, fn entries, segment ->
        {:ok, outputs, segment} = Segment.run(module, entries, segment, :infinity)
        {Map.new(outputs, fn {key, value} -> {key, Nx.backend_copy(value)} end), segment}
      end)

    outputs
  end

  defp both(stages, inputs, matcher, compiler \\ Nx.Defn.Evaluator) do
    members = members(stages)
    {:ok, nx} = NxSegment.build(members, geometry(), defn_options: [compiler: compiler])
    {:ok, native} = NativeSegment.build(members, geometry(), matcher: matcher)
    Enum.zip(run(NativeSegment, native, inputs), run(NxSegment, nx, inputs))
  end

  defp ulps(a, b) do
    to_int = fn t -> t |> Nx.bitcast(:s32) |> Nx.as_type(:s64) end

    Nx.subtract(to_int.(a), to_int.(b))
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
  end

  defp assert_same(native, nx, key) do
    {got, want} = {Map.fetch!(native, key), Map.fetch!(nx, key)}
    assert Nx.shape(got) == Nx.shape(want), "#{inspect(key)} shape"
    assert Nx.to_binary(Nx.as_type(got, Nx.type(want))) == Nx.to_binary(want), inspect(key)
  end

  # Sums reduced in another order agree to their rounding.
  defp assert_close(native, nx, key) do
    got = native |> Map.fetch!(key) |> Nx.as_type(:f64)
    want = nx |> Map.fetch!(key) |> Nx.as_type(:f64)
    bound = want |> Nx.abs() |> Nx.max(1.0) |> Nx.multiply(1.0e-5)

    assert Nx.to_number(Nx.all(Nx.less_equal(Nx.abs(Nx.subtract(got, want)), bound))) == 1,
           inspect(key)
  end

  # A sample that falls on a rounding half may round the other way: one
  # pixel of the second pair's left image does.
  defp assert_within_one_level(native, nx, key) do
    difference =
      Nx.subtract(
        Nx.as_type(Map.fetch!(native, key), :s32),
        Nx.as_type(Map.fetch!(nx, key), :s32)
      )
      |> Nx.abs()

    assert Nx.to_number(Nx.reduce_max(difference)) <= 1, inspect(key)
    assert Nx.to_number(Nx.mean(Nx.greater(difference, 0))) < 0.001, inspect(key)
  end

  test "rectification and sharpness", %{matcher: matcher} do
    stages = [{Stage.Rectification, [right_row_offset: 0.5]}, {Stage.Sharpness, []}]

    for {native, nx} <- both(stages, [frames(1), frames(2)], matcher) do
      assert_within_one_level(native, nx, :left)
      assert_within_one_level(native, nx, :right)
    end

    [{native, nx} | _] = both(stages, [frames(1)], matcher)
    assert_same(native, nx, :left)
    assert_same(native, nx, :right)
    assert_close(native, nx, :sharpness_sums)
  end

  test "disparity, full and with 3 or 8 paths", %{matcher: matcher} do
    images = fn seed -> [left: frames(seed)[:left_frame], right: frames(seed)[:right_frame]] end

    for paths <- [3, 8] do
      stages = [{Stage.Disparity, [num_disparities: 64, paths: paths]}]

      for {native, nx} <- both(stages, [images.(3)], matcher, EXLA),
          do: assert_same(native, nx, :disparity)
    end
  end

  test "the filters, depth, point cloud and measures after the disparity, chained", %{
    matcher: matcher
  } do
    stages = [
      {Stage.MedianFilter, [kernel_size: 3]},
      {Stage.SpeckleFilter, [min_support: 6, max_diff: 1.0]},
      {Stage.RowAlignment, []},
      {Stage.TemporalFilter, [alpha: 0.5, delta: 2.0, hold: 1]},
      {Stage.Depth, []},
      {Stage.PointCloud, [decimation: 2]},
      {Stage.Quality, []}
    ]

    inputs =
      for seed <- 4..6 do
        [
          left: frames(seed)[:left_frame],
          right: frames(seed)[:right_frame],
          disparity: disparity(seed)
        ]
      end

    for {native, nx} <- both(stages, inputs, matcher) do
      assert_same(native, nx, :disparity)
      assert_same(native, nx, :depth_mm)
      assert_same(native, nx, :cloud_points)
      assert ulps(native.cloud_grid, nx.cloud_grid) <= 1
      assert_close(native, nx, :alignment_sums)
      assert_close(native, nx, :quality_sums)
    end
  end

  test "a 5x5 median", %{matcher: matcher} do
    inputs = for seed <- 7..8, do: [disparity: disparity(seed)]

    for {native, nx} <- both([{Stage.MedianFilter, [kernel_size: 5]}], inputs, matcher),
        do: assert_same(native, nx, :disparity)
  end
end
