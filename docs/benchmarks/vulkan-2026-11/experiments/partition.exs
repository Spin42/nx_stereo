Code.require_file("partition_support.exs", __DIR__)
Code.require_file("bench/support/scene.exs")
Code.require_file("bench/support/options.exs")

source =
  File.read!("bench/stream.exs")
  |> String.replace("Code.require_file(\"support/scene.exs\", __DIR__)", "")
  |> String.replace("Code.require_file(\"support/options.exs\", __DIR__)", "")
  |> String.replace("Bench.Stream.main(System.argv())", "")

for name <- ["sequence", "run", "reset", "report", "carry"], reduce: source do
  text -> String.replace(text, "defp #{name}(", "def #{name}(")
end
|> Code.eval_string()

Nx.default_backend(Nx.BinaryBackend)
{:ok, [device | _]} = NxIREE.list_devices("vulkan")
gpu = [compiler: NxIREE.Compiler, iree_runtime_options: [device: device]]
smoke = System.get_env("SPLIT_SMOKE") == "1"

shapes =
  cond do
    smoke ->
      [{64, 96}]

    text = System.get_env("PARTITION_SIZE") ->
      [text |> String.split("x") |> Enum.map(&String.to_integer/1) |> List.to_tuple()]

    true ->
      [{544, 728}, {576, 768}, {600, 800}, {720, 960}]
  end

for {height, width} = size <- shapes do
  frame_size = {height * 2, width * 2}

  opts = [
    disparities: 128,
    paths: 8,
    coarse_paths: 8,
    band: 32,
    margin: 8,
    fine_census: :dense_9x5_blend,
    search: "coarse_to_fine",
    packed: true,
    sequence: 8,
    cloud_decimation: 4
  ]

  inputs = Bench.Scene.inputs(frame_size, size, opts)

  sequence =
    Nx.with_default_backend(EXLA.Backend, fn -> Bench.Stream.sequence(frame_size, size, opts) end)

  {:ok, baseline} = Bench.Scene.segment(inputs, size, frame_size, opts, gpu)

  {ref_segment, ref_frames, _} =
    Bench.Stream.run(Bench.Stream.reset(baseline), sequence, 16, :overlap, true)

  variants = [
    {:baseline, baseline}
    | for(
        {label, cuts} <- [{:after_rectify, [1]}, {:after_match, [3]}, {:three_parts, [1, 3]}],
        do: {label, SplitSegment.replace(baseline, cuts)}
      )
  ]

  for {label, segment} <- tl(variants), mode <- [:serial, :overlap] do
    {got_segment, got_frames, _} =
      Bench.Stream.run(Bench.Stream.reset(segment), sequence, 16, mode, true)

    unless Enum.map(ref_frames, & &1.digest) == Enum.map(got_frames, & &1.digest) and
             Bench.Stream.carry(ref_segment) == Bench.Stream.carry(got_segment),
           do: raise("split #{label}/#{mode} differs at #{height}x#{width}")

    IO.puts(
      "#{height}x#{width}/#{label}/#{mode}: all outputs and carry exact across 16 changing frames"
    )
  end

  IO.puts(
    "round,mode,frames,elapsed_ms,fps,mean_period_ms,period_p50_ms,period_p90_ms,period_p99_ms,latency_p50_ms,latency_p90_ms,latency_p99_ms,period_over_33ms,latency_over_33ms"
  )

  for round <- 1..if(smoke, do: 1, else: 4),
      {label, segment} <-
        Enum.drop(variants, rem(round - 1, 4)) ++ Enum.take(variants, rem(round - 1, 4)),
      mode <- if(rem(round, 2) == 0, do: [:overlap, :serial], else: [:serial, :overlap]) do
    {warm, _, _} = Bench.Stream.run(Bench.Stream.reset(segment), sequence, 24, mode, false)

    IO.puts(
      "BEGIN,#{height}x#{width}_#{label}_#{mode},#{round},#{System.system_time(:millisecond)}"
    )

    {_, records, elapsed} =
      Bench.Stream.run(warm, sequence, if(smoke, do: 8, else: 150), mode, false)

    IO.puts(
      "END,#{height}x#{width}_#{label}_#{mode},#{round},#{System.system_time(:millisecond)}"
    )

    Bench.Stream.report(round, "#{height}x#{width}_#{label}_#{mode}", records, elapsed)
  end
end
