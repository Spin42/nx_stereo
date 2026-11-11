defmodule Bench.Options do
  @moduledoc false
  @censuses %{
    "sparse_9x7" => :sparse_9x7,
    "dense_9x5" => :dense_9x5,
    "dense_9x5_blend" => :dense_9x5_blend
  }

  def cloud_decimation(opts, {height, width}) do
    decimation = Keyword.get(opts, :cloud_decimation, 4)

    unless is_integer(decimation) and decimation > 0 and decimation <= min(height, width),
      do: raise(ArgumentError, "--cloud-decimation must be positive and fit the working image")

    Keyword.put(opts, :cloud_decimation, decimation)
  end

  def fine_census(opts) do
    census =
      Map.get(@censuses, opts[:fine_census]) || raise ArgumentError, "unknown --fine-census"

    if opts[:search] != "coarse_to_fine" and census != :sparse_9x7,
      do: raise(ArgumentError, "--fine-census requires --search coarse_to_fine")

    Keyword.put(opts, :fine_census, census)
  end
end
