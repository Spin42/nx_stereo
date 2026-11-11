# Complete-chain throughput with changing CPU frames, carried state and one
# download overlapping the next computation. Camera capture and display are
# outside the measurement. Compare serial and overlap using the same workload:
#
#   mix run bench/stream.exs --size 544x728 --frames 300 --rounds 3
#     --band 32 --margin 8 --coarse-paths 8 --packed
#
# Compilation, fixture generation and warm-up are excluded. Every timed frame
# is uploaded and all published outputs are downloaded. Verification compares
# both modes, including carried tensors, over two repetitions of the sequence.
Code.require_file("support/scene.exs", __DIR__)
Code.require_file("support/options.exs", __DIR__)

defmodule Bench.Stream do
  @moduledoc false
  alias NxStereo.NxSegment

  @defaults [
    size: "544x728",
    frame: nil,
    search: "coarse_to_fine",
    disparities: 128,
    paths: 8,
    coarse_paths: 3,
    fine_census: "sparse_9x7",
    band: 16,
    margin: 4,
    packed: false,
    cloud_decimation: 4,
    frames: 300,
    warmup: 16,
    sequence: 8,
    rounds: 3,
    mode: "both",
    device: ""
  ]
  @budget_ms 1000 / 30

  def main(args) do
    {opts, []} =
      OptionParser.parse!(args,
        strict: [
          size: :string,
          frame: :string,
          search: :string,
          disparities: :integer,
          paths: :integer,
          coarse_paths: :integer,
          fine_census: :string,
          band: :integer,
          margin: :integer,
          packed: :boolean,
          cloud_decimation: :integer,
          frames: :integer,
          warmup: :integer,
          sequence: :integer,
          rounds: :integer,
          mode: :string,
          device: :string
        ]
      )

    opts = @defaults |> Keyword.merge(opts) |> Bench.Options.fine_census()
    validate!(opts)
    {height, width} = size = dimensions(opts[:size])
    opts = Bench.Options.cloud_decimation(opts, size)
    frame_size = if opts[:frame], do: dimensions(opts[:frame]), else: {height * 2, width * 2}
    {:ok, devices} = NxIREE.list_devices("vulkan")

    device =
      Enum.find(devices, &String.contains?(&1.uri, opts[:device])) || raise "no Vulkan device"

    gpu = [compiler: NxIREE.Compiler, iree_runtime_options: [device: device]]
    Nx.default_backend(Nx.BinaryBackend)
    inputs = Bench.Scene.inputs(frame_size, size, opts)
    sequence = Nx.with_default_backend(EXLA.Backend, fn -> sequence(frame_size, size, opts) end)
    {:ok, initial} = Bench.Scene.segment(inputs, size, frame_size, opts, gpu)

    IO.puts("device=#{device.uri} size=#{opts[:size]} frame=#{inspect(frame_size)}")
    IO.puts("configuration=#{inspect(Keyword.drop(opts, [:device, :size, :frame]))}")
    IO.puts("cloud_grid=#{inspect(Enum.find(initial.layout, &(elem(&1, 0) == :cloud_grid)))}")
    verify!(initial, sequence, opts[:sequence] * 2)
    IO.puts("serial/overlap outputs and final carry: identical across two sequence repetitions")

    IO.puts(
      "round,mode,frames,elapsed_ms,fps,mean_period_ms,period_p50_ms,period_p90_ms," <>
        "period_p99_ms,latency_p50_ms,latency_p90_ms,latency_p99_ms," <>
        "period_over_33ms,latency_over_33ms"
    )

    modes = if opts[:mode] == "both", do: [:serial, :overlap], else: [mode(opts[:mode])]

    for round <- 1..opts[:rounds], mode <- order(modes, round) do
      {warm, _records, _elapsed} = run(reset(initial), sequence, opts[:warmup], mode, false)
      {_segment, records, elapsed} = run(warm, sequence, opts[:frames], mode, false)
      report(round, mode, records, elapsed)
    end
  end

  defp validate!(opts) do
    for key <- [:frames, :warmup, :sequence, :rounds] do
      unless opts[key] > 0, do: raise(ArgumentError, "#{key} must be positive")
    end

    unless opts[:mode] in ["both", "serial", "overlap"],
      do: raise(ArgumentError, "mode must be both, serial or overlap")
  end

  defp dimensions(text) do
    case text |> String.split("x") |> Enum.map(&String.to_integer/1) do
      [height, width] when height > 0 and width > 0 -> {height, width}
      _ -> raise ArgumentError, "size must be positive HEIGHTxWIDTH"
    end
  end

  defp mode("serial"), do: :serial
  defp mode("overlap"), do: :overlap
  defp order(modes, round), do: if(rem(round, 2) == 0, do: Enum.reverse(modes), else: modes)

  # Vary both texture and disparity. All frames live on the CPU before timing.
  defp sequence(frame_size, size, opts) do
    scale = div(elem(frame_size, 1), elem(size, 1))

    for index <- 0..(opts[:sequence] - 1) do
      shift = (div(opts[:disparities], 4) + rem(index, 3) - 1) * scale
      settings = Keyword.merge(opts, seed: index + 1, shift: shift)
      inputs = Bench.Scene.inputs(frame_size, size, settings)

      [
        left_frame: Nx.backend_copy(inputs.left_image, Nx.BinaryBackend),
        right_frame: Nx.backend_copy(inputs.right_image, Nx.BinaryBackend)
      ]
    end
    |> List.to_tuple()
  end

  defp reset(segment), do: NxSegment.reload(segment, segment.members, segment.geometry)

  defp verify!(initial, sequence, frames) do
    {serial, first, _elapsed} = run(reset(initial), sequence, frames, :serial, true)
    {overlap, second, _elapsed} = run(reset(initial), sequence, frames, :overlap, true)

    for {a, b} <- Enum.zip(first, second) do
      unless a.digest == b.digest, do: raise("serial/overlap output mismatch at frame #{a.index}")
    end

    unless carry(serial) == carry(overlap), do: raise("serial/overlap final carry mismatch")

    unless first |> Enum.map(& &1.digest) |> Enum.uniq() |> length() > 1,
      do: raise("the changing sequence produced identical outputs")
  end

  defp carry(segment) do
    for {index, keys} <- segment.carried, key <- keys, into: %{} do
      {{index, key}, Nx.to_binary(segment.params[index][key])}
    end
  end

  defp run(segment, sequence, frames, mode, capture?) do
    started = now()
    {request, sent} = send_frame(segment, sequence, 0)

    context = %{sequence: sequence, frames: frames, mode: mode, capture?: capture?}
    {segment, records} = receive_frames(segment, context, {request, sent}, 0, started, [])

    {segment, Enum.reverse(records), now() - started}
  end

  defp send_frame(segment, sequence, index) do
    sent = now()

    {:ok, request} =
      NxSegment.send_request(elem(sequence, rem(index, tuple_size(sequence))), segment)

    {request, sent}
  end

  defp receive_frames(segment, context, {request, sent}, index, last, records) do
    %{sequence: sequence, frames: frames, mode: mode, capture?: capture?} = context
    {:ok, outputs, segment, _device_done} = NxSegment.await(request, segment, :infinity)
    more? = index + 1 < frames
    next = if more? and mode == :overlap, do: send_frame(segment, sequence, index + 1)
    {:ok, collected} = NxSegment.collect(outputs, segment, :infinity)
    completed = now()

    record = %{
      index: index,
      period: completed - last,
      latency: completed - sent,
      digest: if(capture?, do: digest(collected))
    }

    if more? do
      {request, sent} = next || send_frame(segment, sequence, index + 1)

      receive_frames(segment, context, {request, sent}, index + 1, completed, [record | records])
    else
      {segment, [record | records]}
    end
  end

  defp digest(outputs),
    do:
      Map.new(outputs, fn {name, tensor} ->
        {name, :crypto.hash(:sha256, Nx.to_binary(tensor))}
      end)

  defp report(round, mode, records, elapsed) do
    count = length(records)
    periods = Enum.map(records, &(&1.period / 1.0e6)) |> Enum.sort()
    latencies = Enum.map(records, &(&1.latency / 1.0e6)) |> Enum.sort()
    elapsed_ms = elapsed / 1.0e6

    [round, mode, count, elapsed_ms, 1000 * count / elapsed_ms, elapsed_ms / count]
    |> Kernel.++(
      for values <- [periods, latencies], p <- [0.5, 0.9, 0.99], do: percentile(values, p)
    )
    |> Kernel.++(for values <- [periods, latencies], do: Enum.count(values, &(&1 > @budget_ms)))
    |> Enum.map_join(",", &format/1)
    |> IO.puts()
  end

  defp percentile(sorted, fraction), do: Enum.at(sorted, ceil(length(sorted) * fraction) - 1)
  defp format(value) when is_float(value), do: :erlang.float_to_binary(value, decimals: 3)
  defp format(value), do: to_string(value)
  defp now, do: System.monotonic_time(:nanosecond)
end

Bench.Stream.main(System.argv())
