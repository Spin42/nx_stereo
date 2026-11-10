defmodule NxStereo.NativeSegmentTest do
  use ExUnit.Case, async: true

  alias NxStereo.{NativeSegment, Segment, Stage}

  # A matcher that sends each run to `test` and answers it with `reply`.
  defmodule FakeMatcher do
    use GenServer

    def start_link({test, reply}), do: GenServer.start_link(__MODULE__, {test, reply})

    @impl GenServer
    def init(state), do: {:ok, state}

    @impl GenServer
    def handle_call({:run, :stop, _id, _version, _configuration, _request}, _from, state),
      do: {:stop, :boom, state}

    def handle_call({:run, owner, id, version, configuration, request}, _from, {test, reply}) do
      send(test, {:run, owner, id, version, configuration, IO.iodata_to_binary(request)})
      {:reply, reply, {test, reply}}
    end
  end

  # Takes no settings; reads a disparity and has two params.
  defmodule Scale do
    def requires, do: [:disparity]
    def provides, do: [:scaled]
    def params(_state, geometry), do: %{gain: 2.0, offset: geometry.offset}
    def native_kind, do: "scale"
    def native_settings(state), do: [taps: state]
    def native_constants, do: %{limit: 9, exact: true}
  end

  defmodule NxOnly do
    def requires, do: [:disparity]
    def provides, do: [:disparity]
  end

  @geometry %{width: 4, height: 2, frame_width: 6, frame_height: 3, offset: 0.5}

  defp segment(matcher, taps \\ 3) do
    member = %{module: Scale, name: :scale, state: taps}
    {:ok, segment} = NativeSegment.build([member], @geometry, matcher: matcher)
    segment
  end

  defp matcher(reply), do: start_supervised!({FakeMatcher, {self(), reply}})

  defp disparity, do: Nx.iota({2, 4}, type: :f32)

  # A reply of `times_us` and the f32 output `name`, 1x1.
  defp outputs_reply(times_us, name) do
    times = for time <- times_us, into: <<>>, do: <<time::little-32>>

    output =
      <<byte_size(name), name::binary, 2, 2, 1::little-32, 1::little-32, 1.5::little-float-32>>

    {:ok, <<length(times_us), times::binary, 1, output::binary>>}
  end

  test "configures the Port with the geometry, each stage's kind, settings and constants" do
    segment = segment(matcher(outputs_reply([7], "scaled")))
    {:ok, _request_id} = NativeSegment.send_request(%{disparity: disparity()}, segment)
    id = segment.id

    assert_receive {:run, owner, ^id, 0, configuration, _request}
    assert owner == self()

    assert configuration ==
             <<1, id::little-32, 4::little-16, 2::little-16, 6::little-16, 3::little-16, 1>> <>
               <<5, "scale", 3>> <>
               <<4, "taps", 3.0::little-float-64>> <>
               <<5, "exact", 1.0::little-float-64>> <>
               <<5, "limit", 9.0::little-float-64>>
  end

  test "sends a pair as its entries and each stage's params, as f32 values" do
    segment = segment(matcher(outputs_reply([7], "scaled")))
    {:ok, _request_id} = NativeSegment.send_request([disparity: disparity()], segment)
    id = segment.id

    assert_receive {:run, _owner, ^id, 0, _configuration, request}

    assert request ==
             <<2, id::little-32>> <>
               <<1, 2>> <>
               <<4, "gain", 1::little-16, 2.0::little-float-32>> <>
               <<6, "offset", 1::little-16, 0.5::little-float-32>> <>
               <<1, 9, "disparity", 2, 2, 2::little-32, 4::little-32>> <>
               Nx.to_binary(disparity())
  end

  test "returns the Port's outputs and each member's time, and a new state from the next pair" do
    segment = segment(matcher(outputs_reply([7], "scaled")))
    {:ok, request_id} = NativeSegment.send_request(%{disparity: disparity()}, segment)

    assert {:ok, %{scaled: scaled}, replied, done_ns} =
             NativeSegment.await(request_id, segment, 5_000)

    assert Nx.to_list(scaled) == [[1.5]]
    assert done_ns <= System.monotonic_time(:nanosecond)
    assert NativeSegment.member_times(replied) == [{:scale, 7_000}]
    assert NativeSegment.merge_carry(segment, replied) == segment

    changed = NativeSegment.put_member_state(segment, 0, 5)
    assert changed.configuration == segment.configuration
    assert changed.inputs == segment.inputs
    assert hd(changed.members).state == 5
  end

  test "recognises its reply among other messages" do
    segment = segment(matcher(outputs_reply([7], "scaled")))
    {:ok, request_id} = NativeSegment.send_request(%{disparity: disparity()}, segment)
    assert NativeSegment.response(:other, request_id, segment) == :no_reply

    receive do
      {:run, _owner, _id, _version, _configuration, _request} -> :ok
    end

    assert_receive message

    assert {:ok, %{scaled: _scaled}, _segment, _done_ns} =
             NativeSegment.response(message, request_id, segment)
  end

  test "runs entries through the Port, returning them but the frames with its outputs" do
    segment = segment(matcher(outputs_reply([7], "scaled")))

    assert {:ok, outputs, _segment} =
             Segment.run(NativeSegment, %{disparity: disparity()}, segment, 5_000)

    assert Map.keys(outputs) == [:disparity, :scaled]
  end

  test "a reload configures the stages again, from a new version" do
    segment = segment(matcher(outputs_reply([7], "scaled")))
    reloaded = NativeSegment.reload(segment, segment.members, %{@geometry | offset: 1.5})

    assert reloaded.version == 1
    refute reloaded.inputs == segment.inputs
  end

  test "fails a pair whose reply names an output no loaded module mentions" do
    segment = segment(matcher(outputs_reply([7], "output of no stage")))
    {:ok, request_id} = NativeSegment.send_request(%{disparity: disparity()}, segment)

    assert NativeSegment.await(request_id, segment, 5_000) ==
             {:error, {:unknown_output, "output of no stage"}}
  end

  test "fails a pair whose reply does not decode" do
    segment = segment(matcher({:ok, <<1, 2>>}))
    {:ok, request_id} = NativeSegment.send_request(%{disparity: disparity()}, segment)
    assert NativeSegment.await(request_id, segment, 5_000) == {:error, :bad_reply}
  end

  test "stops at a configuration the Port refuses, saying why" do
    segment = segment(matcher({:error, {:configuration_refused, "no scale"}}))

    assert Segment.run(NativeSegment, %{disparity: disparity()}, segment, 5_000) ==
             {:stop, "no scale"}
  end

  @tag :capture_log
  test "fails a pair whose matcher goes down" do
    segment = %{segment(matcher(:unused)) | owner: :stop}
    {:ok, request_id} = NativeSegment.send_request(%{disparity: disparity()}, segment)

    assert NativeSegment.await(request_id, segment, 5_000) == {:error, {:matcher_down, :boom}}
  end

  test "refuses entries of another size before they reach the Port" do
    segment = segment(matcher(:unused))

    assert NativeSegment.send_request(%{disparity: Nx.iota({2, 2}, type: :f32)}, segment) ==
             {:error, {:frame_size, :disparity, {2, 2}, {2, 4}}}

    refute_received {:run, _owner, _id, _version, _configuration, _request}
  end

  test "takes the stages that declare a native kind, to compute natively" do
    assert NativeSegment.accepts?(Scale, nil, :native)
    refute NativeSegment.accepts?(Scale, nil, :nx)
    refute NativeSegment.accepts?(NxOnly, nil, :native)

    for stage <- [Stage.Rectification, Stage.Disparity, Stage.Depth, Stage.PointCloud],
        do: assert(NativeSegment.accepts?(stage, nil, :native))

    refute NativeSegment.accepts?(Stage.Decode, nil, :native)
  end

  test "runs on the CPU, two pairs at a time, and refuses a build option it does not know" do
    segment = segment(matcher(:unused))
    assert NativeSegment.compute(segment) == %{runs_on: :cpu, implementation: :rust, backend: nil}
    assert NativeSegment.depth() == 2

    assert NativeSegment.build([], @geometry, defn_options: []) ==
             {:error, "unknown build options [:defn_options]"}
  end
end
