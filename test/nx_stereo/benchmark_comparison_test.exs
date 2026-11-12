Code.require_file("../bench/support/compare.exs", __DIR__)

defmodule NxStereo.BenchmarkComparisonTest do
  use ExUnit.Case, async: true

  test "stage comparisons count unexpected NaNs on either side" do
    got = Nx.tensor([:nan, 7.0])
    want = Nx.tensor([7.0, :nan])
    assert Bench.Compare.relative(got, want, 0.01) == 2
  end

  test "stage comparisons accept paired NaNs and same-sign infinities" do
    got = Nx.tensor([:nan, :infinity, :neg_infinity, 1.0005])
    want = Nx.tensor([:nan, :infinity, :neg_infinity, 1.0])
    assert Bench.Compare.relative(got, want, 0.001) == 0
  end

  test "stage comparisons count infinity mismatches even with an infinite relative bound" do
    got = Nx.tensor([:infinity, :neg_infinity, 3.0])
    want = Nx.tensor([3.0, :infinity, :neg_infinity])
    assert Bench.Compare.relative(got, want, 0.01) == 3
  end

  test "stage comparisons preserve relative tolerances above one" do
    got = Nx.tensor([1000.009, 1.01, 0.00005, -1.0])
    want = Nx.tensor([1000.0, 1.0, 0.0, -1.0])
    assert Bench.Compare.relative(got, want, 0.0001) == 1
  end

  test "disparity comparisons reject nonfinite estimates and validity changes" do
    reference = Nx.tensor([-1.0, -1.0, 0.0, 1.0, -1.0, 0.0])
    output = Nx.tensor([-1.0, :nan, 0.0, 1.0005, :infinity, -0.0001])
    assert Bench.Compare.disparity(reference, output) == 3
    assert Bench.Compare.disparity(Nx.tensor([:nan, :infinity]), Nx.tensor([-1.0, 1.0])) == 2
  end

  test "disparity comparisons preserve absolute tolerance and missing matches" do
    reference = Nx.tensor([-1.0, 9.0, 10.0, 0.0])
    output = Nx.tensor([-1.0, 9.0005, 10.002, -0.0001])
    assert Bench.Compare.disparity(reference, output) == 2
  end
end
