defmodule NxStereo.Stage.Disparity do
  @moduledoc """
  Matches the rectified pair, `:left` and `:right`, into `:disparity`
  with `NxStereo.sgm/3` or `NxStereo.sgm_coarse_to_fine/3`, unmatched
  pixels at `-1.0`.

  ## Settings

    * `:p1` (`8`), `:p2` (`64`), `:uniqueness` (`1.0526`) - as
      `NxStereo.sgm/3` takes them; params, so they change without
      compiling again.
    * `:num_disparities` (`64`) - 64 or 128.
    * `:paths` (`8`) - 3 or 8.
    * `:search` (`"full"`) - `"full"`, or `"coarse_to_fine"`, which needs
      128 disparities and an even working size.
    * `:coarse_paths` (`:paths`) - the half-size match's paths, with
      `"coarse_to_fine"`.
    * `:band` (`16`) - 8, 16, 32 or 64 candidates per pixel in the
      fine search. Wider bands cost more memory and computation.
    * `:margin` (`4`) - candidates below the coarse estimate, an
      integer from zero to `band - 1`.
    * `:fine_census` (`:sparse_9x7`) - `:sparse_9x7`, `:dense_9x5`, or
      `:dense_9x5_blend`, as `NxStereo.sgm_band/4` describes. Only used
      with `"coarse_to_fine"`; a static compilation choice.
    * `:matcher` (`nil`) - `:nx` or `:native` to pin where the matching
      is computed; `"coarse_to_fine"` has no native matcher and is
      computed in Nx.
  """
  @behaviour NxStereo.Stage

  @live [:p1, :p2, :uniqueness]
  @defaults [
    p1: 8,
    p2: 64,
    uniqueness: 1.0526315789473684,
    num_disparities: 64,
    paths: 8,
    search: "full",
    coarse_paths: nil,
    band: 16,
    margin: 4,
    fine_census: :sparse_9x7,
    matcher: nil
  ]

  @impl NxStereo.Stage
  def requires, do: [:left, :right]

  @impl NxStereo.Stage
  def provides, do: [:disparity]

  @impl NxStereo.Stage
  def init(settings, geometry) do
    with {:ok, settings} <- NxStereo.Stage.settings(settings, @defaults),
         settings = %{settings | coarse_paths: settings.coarse_paths || settings.paths},
         :ok <- check_matcher(settings.matcher),
         :ok <- check_search(settings, geometry),
         :ok <- check_census(settings),
         do: {:ok, settings}
  end

  @impl NxStereo.Stage
  def params(state, _geometry), do: Map.take(state, @live)

  @impl NxStereo.Stage
  def step(tensors, params, state) do
    opts = [
      disparities: state.num_disparities,
      paths: state.paths,
      p1: params.p1,
      p2: params.p2,
      uniqueness: params.uniqueness
    ]

    disparity =
      case state.search do
        "full" ->
          NxStereo.sgm(tensors.left, tensors.right, opts)

        "coarse_to_fine" ->
          NxStereo.sgm_coarse_to_fine(
            tensors.left,
            tensors.right,
            [
              coarse_paths: state.coarse_paths,
              band: state.band,
              margin: state.margin,
              fine_census: state.fine_census
            ] ++ opts
          )
      end

    Map.put(tensors, :disparity, Nx.select(Nx.greater(disparity, 0), disparity, -1.0))
  end

  @impl NxStereo.Stage
  def compute(%{search: "coarse_to_fine"} = state), do: state.matcher || :nx
  def compute(state), do: state.matcher

  @impl NxStereo.Stage
  def native_kind, do: "disparity"

  @impl NxStereo.Stage
  def native_settings(state), do: [num_disparities: state.num_disparities, paths: state.paths]

  defp check_search(%{search: "full"}, _geometry), do: :ok

  defp check_search(%{num_disparities: disparities}, _geometry) when disparities != 128,
    do: {:error, "coarse_to_fine searches 128 disparities, got #{disparities}"}

  defp check_search(_settings, %{width: width, height: height})
       when rem(width, 2) != 0 or rem(height, 2) != 0,
       do: {:error, "coarse_to_fine needs an even width and height, got #{width}x#{height}"}

  defp check_search(%{matcher: :native}, _geometry),
    do: {:error, "coarse_to_fine has no native matcher"}

  defp check_search(%{band: band}, _geometry) when band not in [8, 16, 32, 64],
    do: {:error, "band must be 8, 16, 32 or 64, got #{inspect(band)}"}

  defp check_search(%{margin: margin, band: band}, _geometry)
       when not is_integer(margin) or margin < 0 or margin >= band,
       do: {:error, "margin must be an integer from 0 to #{band - 1}, got #{inspect(margin)}"}

  defp check_search(_settings, _geometry), do: :ok

  defp check_census(%{fine_census: census})
       when census not in [:sparse_9x7, :dense_9x5, :dense_9x5_blend],
       do:
         {:error,
          "fine_census must be sparse_9x7, dense_9x5 or dense_9x5_blend, got #{inspect(census)}"}

  defp check_census(%{search: "full", fine_census: census}) when census != :sparse_9x7,
    do: {:error, "fine_census requires coarse_to_fine search"}

  defp check_census(_settings), do: :ok

  defp check_matcher(matcher) when matcher in [nil, :nx, :native], do: :ok

  defp check_matcher(matcher),
    do: {:error, "matcher must be :nx or :native, got #{inspect(matcher)}"}
end
