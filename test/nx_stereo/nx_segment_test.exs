defmodule NxStereo.NxSegmentTest do
  use ExUnit.Case, async: true

  alias NxStereo.{NxSegment, Segment}

  defmodule AddOne do
    def requires, do: [:left_frame]
    def provides, do: [:left]
    def step(tensors, _params, _state), do: Map.put(tensors, :left, Nx.add(tensors.left_frame, 1))
  end

  defmodule Double do
    def requires, do: [:left]
    def provides, do: [:left]
    def step(tensors, _params, _state), do: Map.put(tensors, :left, Nx.multiply(tensors.left, 2))
  end

  defmodule Offset do
    def requires, do: [:left]
    def provides, do: [:left]
    def params(_state, geometry), do: %{offset: geometry.focal_length_px}

    def step(tensors, params, _state),
      do: Map.put(tensors, :left, Nx.add(Nx.as_type(tensors.left, :f32), params.offset))
  end

  # Adds its `step`, a param, to a total it carries from pair to pair.
  defmodule Count do
    def requires, do: [:left]
    def provides, do: [:left]
    def params(step, _geometry), do: %{step: step}
    def carry(_step, _geometry), do: %{total: Nx.f32(0)}

    def step(tensors, params, _state) do
      total = Nx.add(params.total, params.step)

      tensors
      |> Map.put(:left, Nx.add(Nx.as_type(tensors.left, :f32), total))
      |> Map.put({:carry, :total}, total)
    end
  end

  # Fills a tensor that only later stages on the device read.
  defmodule Scratch do
    def requires, do: [:left]
    def provides, do: [:scratch]
    def device_only, do: [:scratch]

    def step(tensors, _params, _state),
      do: Map.put(tensors, :scratch, Nx.multiply(tensors.left, 3))
  end

  # Outputs of three widths: a 32-bit float and a 16-bit integer image from the 8-bit one.
  defmodule Widen do
    def requires, do: [:left]
    def provides, do: [:disparity, :depth]

    def step(tensors, _params, _state) do
      tensors
      |> Map.put(:disparity, Nx.multiply(Nx.as_type(tensors.left, :f32), 0.5))
      |> Map.put(:depth, tensors.left |> Nx.as_type(:u16) |> Nx.multiply(300))
    end
  end

  defmodule NoStep do
    def requires, do: [:left]
    def provides, do: [:left]
  end

  defmodule NoCompiler do
    @behaviour Nx.Defn.Compiler

    def __jit__(_key, _vars, _fun, _args, _opts), do: no_compiler()
    def __compile__(_key, _vars, _fun, _opts), do: no_compiler()

    defp no_compiler,
      do: raise(NxIREE.CompileError, message: "compiling defn functions needs EXLA")

    def __partitions_options__(opts), do: [opts]
    def __to_backend__(_opts), do: {Nx.BinaryBackend, []}
  end

  # Compiles, and fails every call, as a device that errors on a pair.
  defmodule FailingDevice do
    @behaviour Nx.Defn.Compiler

    def __jit__(_key, _vars, _fun, _args, _opts), do: raise("the device failed")
    def __compile__(_key, _vars, _fun, _opts), do: fn _args -> raise "the device failed" end
    def __partitions_options__(opts), do: [opts]
    def __to_backend__(_opts), do: {Nx.BinaryBackend, []}
  end

  # A device buffer whose download raises, or never ends.
  defmodule UnreadableBuffer do
    defstruct [:read]

    def backend_transfer(%Nx.Tensor{data: %__MODULE__{read: :raise}}, _backend, _opts),
      do: raise("the download failed")

    def backend_transfer(%Nx.Tensor{data: %__MODULE__{read: :hang}}, _backend, _opts),
      do: Process.sleep(:infinity)
  end

  # Compiles to a function whose published bundle, the `:left` image's 64 words, is an
  # UnreadableBuffer read as its `:read` says.
  defmodule UnreadableDevice do
    @behaviour Nx.Defn.Compiler

    def __jit__(_key, _vars, _fun, _args, _opts), do: raise("only compiled")

    def __compile__(_key, _vars, _fun, opts) do
      published = %Nx.Tensor{
        data: %UnreadableBuffer{read: opts[:read]},
        shape: {64},
        type: {:u, 32},
        names: [nil]
      }

      fn _args -> [%{published: published}] end
    end

    def __partitions_options__(opts), do: [opts]
    def __to_backend__(_opts), do: {Nx.BinaryBackend, []}
  end

  defp geometry(focal_length_px \\ 100.0),
    do: %{width: 4, height: 2, frame_width: 4, frame_height: 2, focal_length_px: focal_length_px}

  defp members(modules, state \\ nil),
    do: for(module <- modules, do: %{module: module, name: module, state: state})

  defp build(modules, opts \\ []), do: NxSegment.build(members(modules), geometry(), opts)

  defp entries(values) do
    frame = values |> Nx.tensor(type: :u8) |> Nx.reshape({2, 4})
    %{left_frame: frame, right_frame: frame}
  end

  defp zeros, do: entries(List.duplicate(0, 8))

  defp run(entries, segment), do: Segment.run(NxSegment, entries, segment, :infinity)

  defp uniq(tensor), do: tensor |> Nx.to_flat_list() |> Enum.uniq()

  # The reply to `entries`, its outputs left on the device.
  defp reply(entries, segment) do
    {:ok, task} = NxSegment.send_request(entries, segment)
    NxSegment.await(task, segment, 5_000)
  end

  defp await_response(task, segment) do
    receive do
      message ->
        case NxSegment.response(message, task, segment) do
          :no_reply -> await_response(task, segment)
          response -> response
        end
    after
      5_000 -> flunk("no response")
    end
  end

  test "runs consecutive steps as one function, returning the tensors they fill" do
    {:ok, segment} = build([AddOne, Double])
    {:ok, outputs, _segment} = run(entries([0, 1, 2, 3, 4, 5, 6, 7]), segment)

    assert Map.keys(outputs) == [:left]
    assert Nx.to_list(outputs.left) == [[2, 4, 6, 8], [10, 12, 14, 16]]
  end

  # Reads the left image, an entry, and fills a tensor of its own.
  defmodule Sum do
    def requires, do: [:left]
    def provides, do: [:sum]
    def step(tensors, _params, _state), do: Map.put(tensors, :sum, Nx.sum(tensors.left))
  end

  test "downloads only what its stages fill, not the entries it was given" do
    {:ok, segment} = NxSegment.build(members([Sum]), geometry(), [])
    left = Nx.iota({2, 4}, type: :u8)
    {:ok, outputs, segment, _done} = reply(%{left: left}, segment)
    {:ok, downloaded} = NxSegment.collect(outputs, segment, 5_000)

    assert Map.keys(downloaded) == [:sum]
    assert {:ok, %{left: ^left, sum: _}, _segment} = run(%{left: left}, segment)
  end

  test "passes each stage's params as inputs, and reload replaces them" do
    {:ok, segment} = NxSegment.build(members([AddOne, Offset]), geometry(100.0), [])
    {:ok, outputs, segment} = run(zeros(), segment)
    assert uniq(outputs.left) == [101.0]

    segment = NxSegment.reload(segment, members([AddOne, Offset]), geometry(5.0))
    {:ok, outputs, _segment} = run(zeros(), segment)
    assert uniq(outputs.left) == [6.0]
  end

  test "a new member state applies from the next pair, keeping what the stages carry" do
    {:ok, segment} = NxSegment.build(members([AddOne, Count], 1), geometry(), [])
    {:ok, _outputs, segment} = run(zeros(), segment)
    {:ok, _outputs, segment} = run(zeros(), segment)

    segment = NxSegment.put_member_state(segment, 1, 10)
    {:ok, outputs, _segment} = run(zeros(), segment)

    assert uniq(outputs.left) == [13.0]
  end

  test "a pair back after a member state change carries its values on, unless a reload came since" do
    {:ok, segment} = NxSegment.build(members([AddOne, Count], 1), geometry(), [])
    {:ok, _outputs, replied, _done_ns} = reply(zeros(), segment)

    changed = NxSegment.put_member_state(segment, 1, 10)
    {:ok, next, _segment} = run(zeros(), NxSegment.merge_carry(changed, replied))
    assert uniq(next.left) == [12.0]

    reloaded = NxSegment.reload(segment, members([AddOne, Count], 1), geometry())
    {:ok, next, _segment} = run(zeros(), NxSegment.merge_carry(reloaded, replied))
    assert uniq(next.left) == [2.0]
  end

  test "leaves a pair's outputs on the device as one tensor, unbundled once downloaded" do
    {:ok, segment} = build([AddOne, Widen])
    {:ok, outputs, segment, _done_ns} = reply(entries([0, 1, 2, 3, 4, 5, 6, 7]), segment)

    assert Map.keys(outputs) == [:published]
    assert Nx.type(outputs.published) == {:u, 32}
    assert NxSegment.published(segment) == [:depth, :disparity, :left]

    {:ok, downloaded} = NxSegment.collect(outputs, segment, 5_000)
    assert Nx.to_list(downloaded.left) == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert Nx.to_list(downloaded.disparity) == [[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]]
    assert Nx.type(downloaded.depth) == {:u, 16}
    assert Nx.to_list(downloaded.depth) == [[300, 600, 900, 1200], [1500, 1800, 2100, 2400]]
  end

  test "leaves the tensors a stage fills for the device only out of its outputs" do
    {:ok, segment} = build([AddOne, Scratch])
    {:ok, outputs, _segment} = run(zeros(), segment)

    assert Map.keys(outputs) == [:left]
    refute :scratch in NxSegment.published(segment)
  end

  test "sends a pair without waiting, and recognises its outputs among other messages" do
    {:ok, segment} = build([AddOne, Double])
    {:ok, task} = NxSegment.send_request(entries([0, 1, 2, 3, 4, 5, 6, 7]), segment)
    assert NxSegment.response(:other, task, segment) == :no_reply

    assert {:ok, outputs, segment, done_ns} = await_response(task, segment)
    assert done_ns <= System.monotonic_time(:nanosecond)
    {:ok, downloaded} = NxSegment.collect(outputs, segment, 5_000)
    assert Nx.to_list(downloaded.left) == [[2, 4, 6, 8], [10, 12, 14, 16]]
    refute_received {:DOWN, _ref, :process, _pid, _reason}
    assert NxSegment.member_times(segment) == []
    assert NxSegment.depth() == 1
  end

  test "returns what the device raises as the pair's error" do
    {:ok, segment} = build([AddOne], defn_options: [compiler: FailingDevice])
    error = %RuntimeError{message: "the device failed"}
    assert {:error, ^error} = run(zeros(), segment)

    {:ok, task} = NxSegment.send_request(zeros(), segment)
    assert {:error, ^error} = await_response(task, segment)
  end

  test "returns a download that fails as the pair's error, and one that does not end as a timeout" do
    for {read, outcome} <- [
          raise: {:error, %RuntimeError{message: "the download failed"}},
          hang: :timeout
        ] do
      {:ok, segment} = build([AddOne], defn_options: [compiler: UnreadableDevice, read: read])
      assert Segment.run(NxSegment, zeros(), segment, 200) == outcome
    end
  end

  test "refuses entries of another size, or missing, before they reach the device" do
    {:ok, segment} = build([AddOne])
    small = Nx.broadcast(Nx.u8(0), {2, 2})

    assert NxSegment.send_request(%{left_frame: small, right_frame: small}, segment) ==
             {:error, {:frame_size, :left_frame, {2, 2}, {2, 4}}}

    assert Segment.run(NxSegment, %{right_frame: small}, segment, :infinity) ==
             {:refused, {:missing_entry, :left_frame}}
  end

  test "takes its defn options among its build options, and refuses one it does not know" do
    assert {:ok, _segment} = build([AddOne], defn_options: [compiler: EXLA])
    assert build([AddOne], compiler: EXLA) == {:error, "unknown build options [:compiler]"}
  end

  test "says where its defn options run a segment, and with what" do
    evaluator = %{runs_on: :cpu, implementation: :nx, backend: :evaluator}
    assert NxSegment.compute([]) == evaluator
    assert NxSegment.compute(compiler: Nx.Defn.Evaluator) == evaluator

    assert NxSegment.compute(compiler: FailingDevice) ==
             %{runs_on: :cpu, implementation: :nx, backend: nil}

    assert NxSegment.compute(compiler: EXLA, client: :cuda) ==
             %{runs_on: :gpu, implementation: :nx, backend: :exla}

    vulkan = ["--iree-hal-target-backends=vulkan-spirv"]

    assert NxSegment.compute(compiler: NxIREE.Compiler, iree_compiler_flags: vulkan) ==
             %{runs_on: :gpu, implementation: :nx, backend: :vulkan}

    assert NxSegment.compute(compiler: NxIREE.Compiler) ==
             %{runs_on: :cpu, implementation: :nx, backend: :llvm_cpu}
  end

  test "takes every stage with a step to compile it, and none to compute natively" do
    assert NxSegment.accepts?(AddOne, nil, :nx)
    refute NxSegment.accepts?(AddOne, nil, :native)
    refute NxSegment.accepts?(NoStep, nil, :nx)
  end

  test "a module missing from the module cache is an error that says so" do
    assert {:error, "no compiled module" <> _rest} =
             build([AddOne], defn_options: [compiler: NoCompiler])
  end
end
