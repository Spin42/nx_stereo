Code.require_file("bench/support/scene.exs")
Code.require_file("bench/support/options.exs")
original = System.fetch_env!("BASELINE_ROOT")

load = fn path, replacements ->
  source = File.read!(Path.join(original, path))

  source =
    Enum.reduce(replacements, source, fn {from, to}, text -> String.replace(text, from, to) end)

  source =
    String.replace(
      source,
      "Path.expand(\"../../shaders/spirv\", __DIR__)",
      inspect(Path.join(original, "shaders/spirv"))
    )

  Code.eval_string(source, file: Path.join(original, path))
end

load.("lib/nx_stereo/kernels.ex", [
  {"defmodule NxStereo.Kernels do", "defmodule OriginalKernels do"}
])

load.("lib/nx_stereo/vulkan.ex", [
  {"defmodule NxStereo.Vulkan do", "defmodule OriginalVulkan do"},
  {"NxStereo.Kernels", "OriginalKernels"}
])

load.("lib/nx_stereo.ex", [
  {"defmodule NxStereo do", "defmodule OriginalStereo do"},
  {"NxStereo.Vulkan", "OriginalVulkan"},
  {"NxStereo.Kernels", "OriginalKernels"}
])

load.("lib/nx_stereo/diagnostics.ex", [
  {"defmodule NxStereo.Diagnostics do", "defmodule OriginalDiagnostics do"},
  {"NxStereo.Kernels", "OriginalKernels"}
])

load.("lib/nx_stereo/nx_segment.ex", [
  {"defmodule NxStereo.NxSegment do", "defmodule OriginalSegment do"}
])

Code.eval_string(
  File.read!("lib/nx_stereo/stage/disparity.ex")
  |> String.replace("defmodule NxStereo.Stage.Disparity do", "defmodule OriginalDisparity do")
  |> String.replace("NxStereo.sgm_coarse_to_fine(", "OriginalStereo.sgm_coarse_to_fine(")
  |> String.replace("              fine_census: state.fine_census\n", "")
)

Code.eval_string(
  File.read!("lib/nx_stereo/stage/row_alignment.ex")
  |> String.replace(
    "defmodule NxStereo.Stage.RowAlignment do",
    "defmodule OriginalRowAlignment do"
  )
  |> String.replace("NxStereo.Diagnostics.", "OriginalDiagnostics.")
)

source =
  File.read!("bench/stream.exs")
  |> String.replace("Code.require_file(\"support/scene.exs\", __DIR__)", "")
  |> String.replace("Code.require_file(\"support/options.exs\", __DIR__)", "")
  |> String.replace("Bench.Stream.main(System.argv())", "")

source =
  Enum.reduce(["sequence", "run", "reset", "report", "carry"], source, fn name, text ->
    String.replace(text, "defp #{name}(", "def #{name}(")
  end)

Code.eval_string(source)

Code.eval_string(
  source
  |> String.replace("defmodule Bench.Stream do", "defmodule OriginalStream do")
  |> String.replace("alias NxStereo.NxSegment", "alias OriginalSegment, as: NxSegment")
)

Nx.default_backend(Nx.BinaryBackend)
{:ok, [device | _]} = NxIREE.list_devices("vulkan")
gpu = [compiler: NxIREE.Compiler, iree_runtime_options: [device: device]]
size = {544, 728}
frame_size = {1088, 1456}

for {band, coarse_paths} <- [{16, 3}, {16, 8}, {32, 8}] do
  opts = [
    disparities: 128,
    paths: 8,
    coarse_paths: coarse_paths,
    band: band,
    margin: div(band, 4),
    fine_census: :sparse_9x7,
    search: "coarse_to_fine",
    packed: true,
    sequence: 8
  ]

  inputs = Bench.Scene.inputs(frame_size, size, opts)

  sequence =
    Nx.with_default_backend(EXLA.Backend, fn -> Bench.Stream.sequence(frame_size, size, opts) end)

  {:ok, optimized} = Bench.Scene.segment(inputs, size, frame_size, opts, gpu)

  members =
    Enum.map(optimized.members, fn member ->
      case member.module do
        NxStereo.Stage.Disparity -> %{member | module: OriginalDisparity}
        NxStereo.Stage.RowAlignment -> %{member | module: OriginalRowAlignment}
        _ -> member
      end
    end)

  {:ok, baseline} =
    OriginalSegment.build(members, optimized.geometry, defn_options: [byte_storage: false] ++ gpu)

  variants = [{:original, OriginalStream, baseline}, {:optimized, Bench.Stream, optimized}]

  {reference, records, _} =
    OriginalStream.run(OriginalStream.reset(baseline), sequence, 16, :overlap, true)

  {checked, actual, _} =
    Bench.Stream.run(Bench.Stream.reset(optimized), sequence, 16, :overlap, true)

  unless Enum.map(records, & &1.digest) == Enum.map(actual, & &1.digest) and
           OriginalStream.carry(reference) == Bench.Stream.carry(checked),
         do: raise("baseline differs b#{band} c#{coarse_paths}")

  IO.puts(
    "CASE band#{band} coarse#{coarse_paths} sparse; original and optimized outputs/carry exact"
  )

  IO.puts(
    "round,mode,frames,elapsed_ms,fps,mean_period_ms,period_p50_ms,period_p90_ms,period_p99_ms,latency_p50_ms,latency_p90_ms,latency_p99_ms,period_over_33ms,latency_over_33ms"
  )

  for round <- 1..6,
      {name, bench, segment} <- if(rem(round, 2) == 0, do: Enum.reverse(variants), else: variants),
      mode <- [:serial, :overlap] do
    {warm, _, _} = bench.run(bench.reset(segment), sequence, 24, mode, false)
    {_, frames, elapsed} = bench.run(warm, sequence, 200, mode, false)
    bench.report(round, "#{name}_#{mode}", frames, elapsed)
  end
end
