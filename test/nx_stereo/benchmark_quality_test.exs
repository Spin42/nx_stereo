Code.require_file("../../bench/support/quality.exs", __DIR__)

defmodule NxStereo.BenchmarkQualityTest do
  use ExUnit.Case, async: true

  test "separates missing matches, wrong matches and unsupported ground truth" do
    truth = Nx.tensor([[0.0, 10.0, 20.0, 30.0, 127.0, 128.0, -1.0, :infinity]])
    disparity = Nx.tensor([[0.0, -1.0, 23.0, 32.0, 127.0, 0.0, 0.0, 0.0]])

    score = Bench.Quality.score(disparity, truth, 128)

    assert score.truth_pixels == 6
    assert score.scored_pixels == 5
    assert score.matched_pixels == 4
    assert_in_delta score.scored, 5 / 6, 1.0e-6
    assert_in_delta score.matched, 4 / 5, 1.0e-6
    assert_in_delta score.bad, 1 / 4, 1.0e-6
    assert_in_delta score.bad_or_missing, 2 / 5, 1.0e-6
    assert_in_delta score.mean_absolute_error, 5 / 4, 1.0e-6
  end

  test "leaves undefined rates empty when there is no supported ground truth" do
    score = Bench.Quality.score(Nx.tensor([[-1.0, 0.0]]), Nx.tensor([[-1.0, 128.0]]), 128)

    assert score.scored_pixels == 0
    assert score.matched_pixels == 0
    assert score.scored == 0.0
    assert score.matched == nil
    assert score.bad == nil
    assert score.bad_or_missing == nil
    assert score.mean_absolute_error == nil
  end

  test "missing every estimate has full bad-or-missing rate and no matched-pixel error" do
    score = Bench.Quality.score(Nx.tensor([[-1.0, -1.0]]), Nx.tensor([[1.0, 2.0]]), 64)

    assert score.matched == 0.0
    assert score.bad == nil
    assert score.bad_or_missing == 1.0
    assert score.mean_absolute_error == nil
  end
end
