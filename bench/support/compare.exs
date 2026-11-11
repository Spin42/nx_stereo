defmodule Bench.Compare do
  @moduledoc false

  def disparity(reference, output) do
    validity = Nx.not_equal(Nx.greater_equal(reference, 0), Nx.greater_equal(output, 0))
    error = Nx.greater(Nx.abs(Nx.subtract(reference, output)), 1.0e-3)
    invalid = Nx.logical_not(Nx.logical_and(finite(reference), finite(output)))

    validity |> Nx.logical_or(error) |> Nx.logical_or(invalid) |> count()
  end

  def relative(got, want, tolerance) do
    {got, want} =
      {Nx.as_type(got, :f32), want |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type(:f32)}

    both_nan = Nx.logical_and(Nx.is_nan(got), Nx.is_nan(want))
    error = Nx.abs(Nx.subtract(got, want))
    bound = Nx.multiply(Nx.max(Nx.abs(want), 1.0), tolerance)

    same = Nx.logical_or(Nx.equal(got, want), both_nan)

    within =
      Nx.logical_and(Nx.logical_and(finite(got), finite(want)), Nx.less_equal(error, bound))

    same |> Nx.logical_or(within) |> Nx.logical_not() |> count()
  end

  defp finite(tensor),
    do: Nx.logical_not(Nx.logical_or(Nx.is_nan(tensor), Nx.is_infinity(tensor)))

  defp count(mask), do: mask |> Nx.sum() |> Nx.to_number()
end
