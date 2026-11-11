[loader | _] = File.read!("bench/datasets.exs") |> String.split("\ndefmodule Bench do", parts: 2)
Code.eval_string(loader)
Code.require_file("bench/support/perturbation.exs")
Code.require_file("bench/support/quality.exs")
Nx.default_backend(EXLA.Backend)
{:ok, [device | _]} = NxIREE.list_devices("vulkan")
gpu = [compiler: NxIREE.Compiler, iree_runtime_options: [device: device]]
profiles = [{16, 2}, {32, 8}]

penalties = [
  {8, 64, 1.0526315789473684},
  {8, 64, 1.03},
  {9, 56, 1.04},
  {9, 56, 1.03},
  {10, 60, 1.04},
  {10, 60, 1.03}
]

functions =
  for {band, margin} <- profiles do
    fun = fn left, right, p1, p2, uniqueness ->
      NxStereo.sgm_coarse_to_fine(left, right,
        paths: 8,
        coarse_paths: 8,
        band: band,
        margin: margin,
        fine_census: :dense_9x5_blend,
        p1: p1,
        p2: p2,
        uniqueness: uniqueness
      )
    end

    {{band, margin}, Nx.Defn.jit(fun, gpu), Nx.Defn.jit(fun, compiler: EXLA)}
  end

{texture, _} = Nx.Random.randint(Nx.Random.key(974), 0, 256, shape: {64, 102}, type: :s32)
left = Nx.slice(texture, [0, 0], [64, 96]) |> Nx.as_type(:u8)
right = Nx.slice(texture, [0, 6], [64, 96]) |> Nx.as_type(:u8)
{unrelated, _} = Nx.Random.randint(Nx.Random.key(127), 0, 256, shape: {64, 96}, type: :s32)
unrelated = Nx.as_type(unrelated, :u8)

for {_profile, run, reference} <- functions,
    {p1, p2, uniqueness} <- penalties,
    r <- [right, unrelated] do
  a =
    run.(left, r, Nx.s32(p1), Nx.s32(p2), Nx.f32(uniqueness)) |> Nx.backend_transfer(EXLA.Backend)

  b = reference.(left, r, Nx.s32(p1), Nx.s32(p2), Nx.f32(uniqueness))

  unless Nx.all(Nx.less_equal(Nx.abs(Nx.subtract(a, b)), 1.0e-4)) |> Nx.to_number() == 1,
    do: raise("live penalty reference differs")
end

IO.puts("24_LIVE_PENALTY_REFERENCE_CHECKS_PASSED")

IO.puts(
  "dataset,scene,condition,coarse_paths,paths,band,margin,fine_census,p1,p2,uniqueness,scored_pixels,matched,bad2,bad_or_missing,mean_absolute_error"
)

for dataset <- [:middlebury, :eth3d, :kitti],
    {scene, left, right, truth} <- apply(Bench.Datasets, dataset, []) do
  {h, w} = Nx.shape(left)
  shape = [div(h, 2) * 2, div(w, 2) * 2]
  [left, right, truth] = Enum.map([left, right, truth], &Nx.slice(&1, [0, 0], shape))

  for condition <- ["clean", "noise4"] do
    {l, r} = Bench.Perturbation.apply(left, right, condition)

    for {{band, margin}, run, _} <- functions, {p1, p2, uniqueness} <- penalties do
      output =
        run.(l, r, Nx.s32(p1), Nx.s32(p2), Nx.f32(uniqueness))
        |> Nx.backend_transfer(EXLA.Backend)

      score = Bench.Quality.score(output, truth, 128)

      values = [
        score.scored_pixels,
        score.matched,
        score.bad,
        score.bad_or_missing,
        score.mean_absolute_error
      ]

      IO.puts(
        Enum.join(
          [dataset, scene, condition, 8, 8, band, margin, :dense_9x5_blend, p1, p2, uniqueness] ++
            values,
          ","
        )
      )
    end
  end
end
