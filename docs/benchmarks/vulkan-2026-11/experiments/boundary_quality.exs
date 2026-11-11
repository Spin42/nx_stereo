[loader | _] = File.read!("bench/datasets.exs") |> String.split("\ndefmodule Bench do", parts: 2)
Code.eval_string(loader)
Code.require_file("bench/support/perturbation.exs")
Code.require_file("bench/support/quality.exs")

defmodule BoundaryQuality do
  @moduledoc false
  import Nx.Defn

  defn masks(truth) do
    valid = truth >= 0 and truth < 128 and not Nx.is_nan(truth) and not Nx.is_infinity(truth)
    edges = edges(truth, valid)
    near = Nx.window_max(Nx.as_type(edges, :u8), {5, 5}, padding: :same) > 0
    {valid and near, valid and not near}
  end

  deftransformp edges(truth, valid) do
    Enum.reduce([{-1, 0}, {1, 0}, {0, -1}, {0, 1}], Nx.broadcast(Nx.u8(0), truth), fn {dy, dx},
                                                                                      found ->
      shifted = NxStereo.Image.shift(truth, dy, dx)
      other = NxStereo.Image.shift(valid, dy, dx)
      jump = Nx.greater(Nx.abs(Nx.subtract(truth, shifted)), 2)
      Nx.logical_or(found, Nx.logical_and(valid, Nx.logical_and(other, jump)))
    end)
  end
end

Nx.default_backend(EXLA.Backend)
masks = Nx.Defn.jit(&BoundaryQuality.masks/1, compiler: EXLA)
# A ten-pixel step places both endpoints of the jump in the edge set;
# dilation by two pixels gives exactly six boundary columns.
truth = Nx.concatenate([Nx.broadcast(10.0, {8, 8}), Nx.broadcast(20.0, {8, 8})], axis: 1)
{near, far} = masks.(truth)
unless Nx.sum(near) |> Nx.to_number() == 48, do: raise("boundary step control differs")
unless Nx.sum(far) |> Nx.to_number() == 80, do: raise("boundary interior control differs")
flat = Nx.broadcast(10.0, {8, 16})
{near, _} = masks.(flat)
unless Nx.sum(near) |> Nx.to_number() == 0, do: raise("flat depth has false boundaries")
unknown = Nx.concatenate([Nx.broadcast(10.0, {8, 8}), Nx.broadcast(-1.0, {8, 8})], axis: 1)
{near, _} = masks.(unknown)
unless Nx.sum(near) |> Nx.to_number() == 0, do: raise("unknown truth creates false boundaries")
IO.puts("BOUNDARY_MASK_CONTROLS_PASSED")
{:ok, [device | _]} = NxIREE.list_devices("vulkan")
gpu = [compiler: NxIREE.Compiler, iree_runtime_options: [device: device]]

settings = [
  {16, 4, :sparse_9x7},
  {16, 4, :dense_9x5_blend},
  {16, 2, :dense_9x5_blend},
  {32, 8, :sparse_9x7},
  {32, 8, :dense_9x5_blend},
  {32, 6, :dense_9x5_blend}
]

IO.puts(
  "dataset,scene,condition,coarse_paths,paths,band,margin,fine_census,region,scored_pixels,matched,bad2,bad_or_missing,mean_absolute_error"
)

for dataset <- [:middlebury, :eth3d, :kitti],
    {scene, left, right, truth} <- apply(Bench.Datasets, dataset, []) do
  {h, w} = Nx.shape(left)
  shape = [div(h, 2) * 2, div(w, 2) * 2]
  [left, right, truth] = Enum.map([left, right, truth], &Nx.slice(&1, [0, 0], shape))
  {near, far} = masks.(truth)
  regions = [{:boundary, Nx.select(near, truth, -1.0)}, {:interior, Nx.select(far, truth, -1.0)}]

  for condition <- ["clean", "noise4"] do
    {l, r} = Bench.Perturbation.apply(left, right, condition)

    for {band, margin, census} <- settings do
      output =
        Nx.Defn.jit(
          &NxStereo.sgm_coarse_to_fine(&1, &2,
            paths: 8,
            coarse_paths: 8,
            band: band,
            margin: margin,
            fine_census: census
          ),
          gpu
        ).(l, r)
        |> Nx.backend_copy(EXLA.Backend)

      for {region, masked_truth} <- regions do
        score = Bench.Quality.score(output, masked_truth, 128)

        values = [
          score.scored_pixels,
          score.matched,
          score.bad,
          score.bad_or_missing,
          score.mean_absolute_error
        ]

        IO.puts(
          Enum.join(
            [dataset, scene, condition, 8, 8, band, margin, census, region] ++ values,
            ","
          )
        )
      end
    end
  end
end
