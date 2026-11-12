defmodule Bench.Quality do
  @moduledoc false

  def score(disparity, truth, disparities) do
    truth_known = Nx.logical_and(Nx.greater_equal(truth, 0), finite(truth))
    scored = Nx.logical_and(truth_known, Nx.less(truth, disparities))
    prediction_known = Nx.logical_and(Nx.greater_equal(disparity, 0), finite(disparity))
    matched = Nx.logical_and(scored, prediction_known)
    error = Nx.abs(Nx.subtract(disparity, truth))
    bad = Nx.logical_and(matched, Nx.greater(error, 2))
    missing = Nx.logical_and(scored, Nx.logical_not(prediction_known))
    truth_pixels = count(truth_known)
    scored_pixels = count(scored)
    matched_pixels = count(matched)

    %{
      truth_pixels: truth_pixels,
      scored_pixels: scored_pixels,
      matched_pixels: matched_pixels,
      scored: rate(scored_pixels, truth_pixels),
      matched: rate(matched_pixels, scored_pixels),
      bad: rate(count(bad), matched_pixels),
      bad_or_missing: rate(count(Nx.logical_or(bad, missing)), scored_pixels),
      mean_absolute_error:
        rate(Nx.select(matched, error, 0) |> Nx.sum() |> Nx.to_number(), matched_pixels)
    }
  end

  defp finite(tensor),
    do: Nx.logical_not(Nx.logical_or(Nx.is_infinity(tensor), Nx.is_nan(tensor)))

  defp count(mask), do: mask |> Nx.sum() |> Nx.to_number()
  defp rate(_numerator, 0), do: nil
  defp rate(numerator, denominator), do: numerator / denominator
end
