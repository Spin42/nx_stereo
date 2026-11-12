defmodule NxStereo.Stage.StatefulPipelineTest do
  use ExUnit.Case, async: true

  alias NxStereo.{Diagnostics, NativeSegment, NxSegment, Segment, Stage}

  @geometry %{
    width: 4,
    height: 4,
    frame_width: 4,
    frame_height: 4,
    focal_length_px: 64.0,
    baseline_m: 0.125,
    principal_point: {1.5, 1.5}
  }

  @stages [
    {Stage.SpeckleFilter, min_support: 0},
    Stage.Quality,
    {Stage.TemporalFilter, alpha: 0.5, delta: 2.0, hold: 2},
    Stage.Depth,
    {Stage.PointCloud, decimation: 1}
  ]

  defp run(module, segment, disparity) do
    input = Nx.broadcast(Nx.f32(disparity), {4, 4})
    {:ok, output, segment} = Segment.run(module, %{disparity: input}, segment, 10_000)
    assert Nx.to_flat_list(input) == List.duplicate(disparity * 1.0, 16)
    {output, segment}
  end

  defp assert_depth(output, disparity, focal_length) do
    assert Nx.to_flat_list(output.disparity) == List.duplicate(disparity * 1.0, 16)
    assert Nx.type(output.depth_mm) == {:u, 16}
    assert Nx.type(output.cloud_points) == {:s, 32}
    assert Nx.shape(output.cloud_grid) == {4, 4, 3}
    refute Map.has_key?(output, :depth_m)

    if disparity > 0 do
      depth = focal_length * 0.125 / disparity
      assert Nx.to_flat_list(output.depth_mm) == List.duplicate(round(depth * 1000), 16)
      assert Nx.to_number(output.cloud_points) == 16

      output.cloud_grid
      |> Nx.to_flat_list()
      |> Enum.chunk_every(3)
      |> Enum.with_index()
      |> Enum.each(fn {[x, y, z], index} ->
        assert_in_delta z, depth, 1.0e-6
        assert_in_delta x, (rem(index, 4) - 1.5) * depth / focal_length, 1.0e-6
        assert_in_delta y, (div(index, 4) - 1.5) * depth / focal_length, 1.0e-6
      end)
    else
      assert Nx.to_flat_list(output.depth_mm) == List.duplicate(0, 16)
      assert Nx.to_number(output.cloud_points) == 0
      assert Nx.to_flat_list(Nx.is_nan(output.cloud_grid)) == List.duplicate(1, 48)
    end
  end

  defp exercise(module, options) do
    {:ok, members} = Segment.members(@stages, @geometry)
    {:ok, segment} = module.build(members, @geometry, options)

    sequence = [
      {10.0, 10.0, 0.0},
      {12.0, 11.0, 2.0},
      {12.0, 11.5, 0.0},
      {-1.0, 11.5, 0.0},
      {-1.0, 11.5, 0.0},
      {-1.0, -1.0, 0.0},
      {50.0, 50.0, 0.0}
    ]

    segment =
      Enum.reduce(sequence, segment, fn {current, filtered, jitter}, segment ->
        {output, segment} = run(module, segment, current)
        assert_depth(output, filtered, 64.0)
        stats = Diagnostics.disparity_stats(output.quality_sums, 16)
        assert stats.mean_px == max(current, 0.0)
        assert stats.matched == if(current > 0, do: 1.0, else: 0.0)
        assert stats.spread_px == 0.0
        assert stats.jitter_px == jitter
        segment
      end)

    {:ok, state} = Stage.TemporalFilter.init([alpha: 1.0, delta: 2.0, hold: 0], @geometry)
    segment = module.put_member_state(segment, 2, state)
    {output, segment} = run(module, segment, 51.0)
    assert_depth(output, 51.0, 64.0)
    assert Diagnostics.disparity_stats(output.quality_sums, 16).jitter_px == 1.0
    {output, segment} = run(module, segment, -1.0)
    assert_depth(output, -1.0, 64.0)

    geometry = %{@geometry | focal_length_px: 128.0}
    {:ok, members} = Segment.members(@stages, geometry)
    segment = module.reload(segment, members, geometry)
    {output, _segment} = run(module, segment, 12.0)
    assert_depth(output, 12.0, 128.0)
    assert Diagnostics.disparity_stats(output.quality_sums, 16).jitter_px == 0.0
  end

  test "CPU pipelines keep diagnostic and temporal history separate through updates and reload" do
    exercise(NxSegment, defn_options: [compiler: EXLA])
  end

  @tag :native
  test "native pipelines keep diagnostic and temporal history separate through updates and reload" do
    matcher = start_supervised!({NxStereo.Matcher, name: nil, threads: 2})
    exercise(NativeSegment, matcher: matcher)
  end

  @tag :vulkan
  test "Vulkan pipelines keep diagnostic and temporal history separate through updates and reload" do
    {:ok, devices: devices} = NxStereo.TestDevices.vulkan()

    for device <- devices do
      exercise(NxSegment,
        defn_options: [compiler: NxIREE.Compiler, iree_runtime_options: [device: device]]
      )
    end
  end
end
