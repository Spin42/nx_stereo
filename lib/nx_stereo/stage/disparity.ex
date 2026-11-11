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

  # Path costs are bytes in the shaders: NxStereo.sgm/3 bounds P2 so they cannot overflow.
  @max_penalty 79

  @impl NxStereo.Stage
  def settings do
    [
      p1: %{
        default: 8,
        type: :integer,
        range: {0, @max_penalty - 1},
        live: true,
        doc: "Penalty for a disparity change of one between neighbours"
      },
      p2: %{
        default: 64,
        type: :integer,
        range: {1, @max_penalty},
        greater_than: :p1,
        live: true,
        doc: "Penalty for a larger disparity change; above p1"
      },
      uniqueness: %{
        default: 1.0526315789473684,
        type: :float,
        range: {1, 3},
        live: true,
        doc: "Factor the second-best cost must exceed the best by"
      },
      num_disparities: %{
        default: 64,
        type: :integer,
        values: [64, 128],
        doc: "Disparities searched, from 0"
      },
      paths: %{
        default: 8,
        type: :integer,
        values: [3, 8],
        doc: "Aggregation paths; 3 is faster and leaves more pixels unmatched"
      },
      search: %{
        default: "full",
        type: :string,
        values: ["full", "coarse_to_fine"],
        doc:
          "full searches every disparity; coarse_to_fine refines a half-size match, much faster"
      },
      coarse_paths: %{
        when: [search: "coarse_to_fine"],
        default: nil,
        type: :integer,
        values: [3, 8],
        doc:
          "Aggregation paths of coarse_to_fine's half-size match, paths by default; 3 is faster"
      },
      band: %{
        when: [search: "coarse_to_fine"],
        default: 16,
        type: :integer,
        values: [8, 16, 32, 64],
        doc: "Candidates per pixel in coarse_to_fine's fine search"
      },
      margin: %{
        when: [search: "coarse_to_fine"],
        default: 4,
        type: :integer,
        range: {0, 63},
        doc: "Candidates below the coarse estimate in the fine search, less than band"
      },
      fine_census: %{
        when: [search: "coarse_to_fine"],
        default: :sparse_9x7,
        type: :atom,
        values: [:sparse_9x7, :dense_9x5, :dense_9x5_blend],
        doc: "Census window of coarse_to_fine's fine search"
      },
      matcher: %{
        default: nil,
        type: :atom,
        values: [nil, :nx, :native],
        doc: "Where the matching is computed, when pinned"
      }
    ]
  end

  @impl NxStereo.Stage
  def requires, do: [:left, :right]

  @impl NxStereo.Stage
  def provides, do: [:disparity]

  @impl NxStereo.Stage
  def init(settings, geometry) do
    with {:ok, settings} <- NxStereo.Stage.init_settings(__MODULE__, settings),
         settings = %{settings | coarse_paths: settings.coarse_paths || settings.paths},
         :ok <- check_search(settings, geometry),
         :ok <- check_census(settings),
         do: {:ok, settings}
  end

  @impl NxStereo.Stage
  def params(state, _geometry), do: Map.take(state, [:p1, :p2, :uniqueness])

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

  defp check_search(%{margin: margin, band: band}, _geometry) when margin >= band,
    do: {:error, "margin must be below band (#{band}), got #{margin}"}

  defp check_search(_settings, _geometry), do: :ok

  defp check_census(%{search: "full", fine_census: census}) when census != :sparse_9x7,
    do: {:error, "fine_census requires coarse_to_fine search"}

  defp check_census(_settings), do: :ok
end
