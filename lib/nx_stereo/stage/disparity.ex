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
            [coarse_paths: state.coarse_paths] ++ opts
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

  defp check_search(_settings, _geometry), do: :ok

  defp check_matcher(matcher) when matcher in [nil, :nx, :native], do: :ok

  defp check_matcher(matcher),
    do: {:error, "matcher must be :nx or :native, got #{inspect(matcher)}"}
end
