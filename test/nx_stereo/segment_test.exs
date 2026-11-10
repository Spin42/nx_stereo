defmodule NxStereo.SegmentTest do
  use ExUnit.Case, async: true

  alias NxStereo.Segment

  defmodule Rectifies do
    def requires, do: [:left_frame, :right_frame]
    def provides, do: [:left, :right]
  end

  defmodule Filters do
    def requires, do: [:left, :disparity]
    def provides, do: [:disparity]
  end

  defmodule ReadsDepth do
    def requires, do: [:depth]
    def provides, do: []
  end

  # Echoes its entries, as a device that has computed nothing.
  defmodule Echo do
    def send_request(entries, segment), do: {:ok, {Map.new(entries), segment}}
    def await({entries, segment}, _segment, _timeout), do: {:ok, %{echoed: entries}, segment, 0}
    def collect(outputs, _segment, _timeout), do: {:ok, outputs}
  end

  @geometry %{width: 4, height: 2, frame_width: 6, frame_height: 3}

  defp members(modules), do: for(module <- modules, do: %{module: module, name: module})

  test "takes the frames and images at the frame and working sizes, as u8, and the disparity as f32" do
    assert Segment.entry_specs(members([Rectifies]), @geometry) ==
             {:ok, [left_frame: {{3, 6}, {:u, 8}}, right_frame: {{3, 6}, {:u, 8}}]}

    assert Segment.entry_specs(members([Filters]), @geometry) ==
             {:ok, [left: {{2, 4}, {:u, 8}}, disparity: {{2, 4}, {:f, 32}}]}

    assert Segment.entry_specs(members([Rectifies, Filters]), @geometry) ==
             {:ok,
              [
                left_frame: {{3, 6}, {:u, 8}},
                right_frame: {{3, 6}, {:u, 8}},
                disparity: {{2, 4}, {:f, 32}}
              ]}
  end

  test "packed, takes frames and images as u32 words of four pixels, on a width that allows it" do
    geometry = %{@geometry | frame_width: 8}

    assert Segment.entry_specs(members([Rectifies]), geometry, packed: true) ==
             {:ok, [left_frame: {{3, 2}, {:u, 32}}, right_frame: {{3, 2}, {:u, 32}}]}

    assert Segment.entry_specs(members([Rectifies]), @geometry, packed: true) ==
             {:error, ":left_frame is 6 pixels wide: packing needs a multiple of 4"}
  end

  test "the stages, in depth order, read nothing but the camera frames from outside" do
    stages = [
      NxStereo.Stage.Rectification,
      NxStereo.Stage.Sharpness,
      NxStereo.Stage.Disparity,
      NxStereo.Stage.SpeckleFilter,
      NxStereo.Stage.RowAlignment,
      NxStereo.Stage.TemporalFilter,
      NxStereo.Stage.Depth,
      NxStereo.Stage.PointCloud,
      NxStereo.Stage.Quality
    ]

    assert Segment.entry_specs(members(stages), @geometry) ==
             {:ok, [left_frame: {{3, 6}, {:u, 8}}, right_frame: {{3, 6}, {:u, 8}}]}

    assert Segment.entry_fields(members([NxStereo.Stage.Decode, NxStereo.Stage.MedianFilter])) ==
             [:left_frame, :right_frame, :disparity]
  end

  test "cannot take a tensor no stage of a segment fills" do
    assert Segment.entry_specs(members([ReadsDepth]), @geometry) ==
             {:error, "a segment cannot read :depth from a CPU stage"}
  end

  test "takes entries as tensors of their spec, or says why it cannot" do
    {:ok, specs} = Segment.entry_specs(members([Rectifies, Filters]), @geometry)
    frame = Nx.broadcast(Nx.u8(7), {3, 6})
    disparity = Nx.broadcast(Nx.f64(1.5), {2, 4})
    entries = %{left_frame: frame, right_frame: frame, disparity: disparity}

    assert {:ok, [left_frame: ^frame, right_frame: ^frame, disparity: converted]} =
             Segment.entry_tensors(entries, specs)

    assert Nx.type(converted) == {:f, 32}

    assert Segment.entry_tensors(%{entries | right_frame: Nx.iota({2, 4}, type: :u8)}, specs) ==
             {:error, {:frame_size, :right_frame, {2, 4}, {3, 6}}}

    assert Segment.entry_tensors(%{entries | disparity: nil}, specs) ==
             {:error, {:not_a_tensor, :disparity, nil}}

    assert Segment.entry_tensors(Map.delete(entries, :disparity), specs) ==
             {:error, {:missing_entry, :disparity}}
  end

  test "packs an entry its spec gives as u32 words" do
    geometry = %{@geometry | frame_width: 8}
    {:ok, specs} = Segment.entry_specs(members([Rectifies]), geometry, packed: true)
    frame = Nx.iota({3, 8}, type: :u8)

    {:ok, [left_frame: packed, right_frame: _]} =
      Segment.entry_tensors([left_frame: frame, right_frame: frame], specs)

    assert {Nx.shape(packed), Nx.type(packed)} == {{3, 2}, {:u, 32}}
    assert Nx.to_binary(packed) == Nx.to_binary(frame)
  end

  test "runs a pair's entries through a segment, returning them but the frames with its outputs" do
    entries = [left_frame: Nx.u8(1), disparity: Nx.f32(2)]

    assert {:ok, outputs, :segment} = Segment.run(Echo, entries, :segment, :infinity)
    assert Map.keys(outputs) == [:disparity, :echoed]
  end

  test "initialises stages from their settings, or says which refuses" do
    geometry = Map.merge(@geometry, %{focal_length_px: 100.0, baseline_m: 0.1})

    assert {:ok, [depth, filter]} =
             Segment.members(
               [NxStereo.Stage.Depth, {NxStereo.Stage.SpeckleFilter, min_support: 4}],
               geometry
             )

    assert depth == %{module: NxStereo.Stage.Depth, name: NxStereo.Stage.Depth, state: nil}
    assert filter.state.min_support == 4

    assert {:error, "NxStereo.Stage.Depth: Depth takes no settings, got [:scale]"} =
             Segment.members([{NxStereo.Stage.Depth, scale: 2}], geometry)
  end
end
