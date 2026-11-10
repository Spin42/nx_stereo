defmodule NxStereo.Segment do
  @moduledoc """
  A backend that runs a list of stages on a device as one segment:
  `NxStereo.NxSegment` compiles them into one Nx function, and
  `NxStereo.NativeSegment` computes them natively in the `stereo_match`
  Port, through `NxStereo.Matcher`.

  A segment's members are initialised stages, `%{module:, name:,
  state:}`: `module` implements `NxStereo.Stage`'s computation callbacks
  with its `requires/0` and `provides/0`, and `state` is what its
  `init/2` returned. The geometry the segment is built for is a map
  (see `NxStereo.Stage`); a segment reads `:width`, `:height`,
  `:frame_width` and `:frame_height`, and its stages the keys they need.

  A pair goes through a segment in three steps: `c:send_request/2` takes
  its entries, the tensors the members read that no member before them
  provides (`entry_fields/1`), and hands them to the device without
  waiting; the device's reply comes back as a message, which
  `c:response/3` recognises and `c:await/3` waits for; and `c:collect/3`
  downloads what the device left there. `run/4` runs the three steps at
  once. Up to `c:depth/0` pairs may be at the device at once, computed in
  the order sent.

  A reply `{:error, reason}` fails that pair only; `{:stop, reason}`
  says the device refuses the segment itself, such as its configuration.
  """

  alias NxStereo.Packing

  @frames [:left_frame, :right_frame]
  @images [:left, :right]
  @maps [:disparity]
  @packable @frames ++ @images

  @type t :: %{required(:members) => [member()], optional(atom()) => term()}

  @typedoc "An initialised stage."
  @type member :: %{module: module(), name: term(), state: NxStereo.Stage.state()}

  @typedoc "Entries by name: a map or a keyword list of tensors."
  @type entries :: %{atom() => Nx.Tensor.t()} | [{atom(), Nx.Tensor.t()}]

  @typedoc """
  What a segment returns for a pair: its outputs, the segment to keep,
  and when the device was done; the error the device failed it with; or
  why the device refuses the segment itself.
  """
  @type response ::
          {:ok, outputs :: map(), t(), done_ns :: integer()}
          | {:error, term()}
          | {:stop, term()}

  @typedoc "Where a segment runs, and with what."
  @type compute :: %{
          runs_on: :cpu | :gpu,
          implementation: :nx | :rust,
          backend: atom() | nil
        }

  @typedoc "The shape and type an entry enters a segment with."
  @type entry_spec :: {tuple(), Nx.Type.t()}

  @doc """
  The segment of `members`, which `c:accepts?/3` took, for `geometry`,
  with the backend's options; or why it cannot start.
  """
  @callback build(members :: [member()], NxStereo.Stage.geometry(), opts :: keyword()) ::
              {:ok, t()} | {:error, String.t()}

  @doc """
  Whether the backend runs the stage `module`, initialised to
  `stage_state`, computing as `compute` says: compiled with Nx (`:nx`),
  or natively (`:native`).
  """
  @callback accepts?(module(), stage_state :: term(), compute :: :nx | :native) :: boolean()

  @doc """
  Hands a pair's `entries` to the device without waiting, returning the
  request that `c:response/3` and `c:await/3` recognise its reply by; or
  why the entries do not fit, before they reach the device.
  """
  @callback send_request(entries(), t()) :: {:ok, request :: term()} | {:error, term()}

  @doc """
  When `message` is the reply to `request`, sent with `segment`: the
  pair's outputs, the segment to keep, which carries the values its
  stages pass to the next pair, and the monotonic time in nanoseconds
  the device was done; `{:error, reason}` when the device failed the
  pair, `{:stop, reason}` when it refuses the segment itself; `:no_reply`
  for another message.
  """
  @callback response(message :: term(), request :: term(), segment :: t()) ::
              response() | :no_reply

  @doc """
  Waits up to `timeout` milliseconds for the reply to `request`, sent
  with `segment`, without taking other messages: what `c:response/3`
  returns for it, or `:timeout`.
  """
  @callback await(request :: term(), segment :: t(), timeout()) :: response() | :timeout

  @doc """
  The outputs a reply returned with `segment`, downloading those still
  on the device within `timeout` milliseconds; the download's error, or
  `:timeout`.
  """
  @callback collect(outputs :: map(), segment :: t(), timeout()) ::
              {:ok, map()} | {:error, term()} | :timeout

  @doc """
  How many pairs may be at the device at once: with one, a pair is
  collected while the next one's device work runs.
  """
  @callback depth() :: pos_integer()

  @doc """
  The segment for the next pair, from `current`, the segment in force,
  and `replied`, the one a pair came back with. `current` is usually the
  segment the pair was sent with, and the result must then carry
  `replied`'s values unchanged. A member state changed while the pair was
  at the device makes `current` another segment, whose inputs the result
  keeps, with `replied`'s carried values; a reload since then makes it
  one whose carried values started over, which the result keeps.
  """
  @callback merge_carry(current :: t(), replied :: t()) :: t()

  @doc "Each member's name and time on the device for the last pair, in nanoseconds, when the device reports them."
  @callback member_times(t()) :: [{term(), non_neg_integer()}]

  @doc "Where the segment runs, and with what."
  @callback compute(t()) :: compute()

  @doc "The segment with `members` re-initialised for a new `geometry` of the same size."
  @callback reload(t(), members :: [member()], NxStereo.Stage.geometry()) :: t()

  @doc "The segment with the member at `index` in a new `state`, from the next pair on."
  @callback put_member_state(t(), index :: non_neg_integer(), state :: term()) :: t()

  @doc """
  Runs a pair's `entries` through `segment`, of the backend `module`,
  waiting up to `timeout` milliseconds for each of its reply and its
  download: the outputs, with the entries other than the camera frames,
  and the segment to keep; `{:refused, reason}` when
  `c:send_request/2` refused the entries, `{:error, reason}` when the
  device failed the pair, `{:stop, reason}` when the device refuses the
  segment, or `:timeout`.
  """
  @spec run(module(), entries(), t(), timeout()) ::
          {:ok, map(), t()}
          | {:refused, term()}
          | {:error, term()}
          | {:stop, term()}
          | :timeout
  def run(module, entries, segment, timeout) do
    case module.send_request(entries, segment) do
      {:ok, request} ->
        with {:ok, outputs, segment, _done_ns} <- module.await(request, segment, timeout),
             {:ok, outputs} <- module.collect(outputs, segment, timeout) do
          entries = entries |> Map.new() |> Map.drop(@frames)
          {:ok, Map.merge(entries, outputs), segment}
        end

      {:error, reason} ->
        {:refused, reason}
    end
  end

  @doc """
  The members of a segment from `stages`, each a stage module or
  `{module, settings}`, initialised for `geometry` and named after its
  module; or the first stage's refusal, prefixed with its name.
  """
  @spec members([module() | {module(), keyword()}], NxStereo.Stage.geometry()) ::
          {:ok, [member()]} | {:error, String.t()}
  def members(stages, geometry) do
    Enum.reduce_while(stages, {:ok, []}, fn stage, {:ok, members} ->
      {module, settings} = if is_atom(stage), do: {stage, []}, else: stage

      case module.init(settings, geometry) do
        {:ok, state} -> {:cont, {:ok, members ++ [%{module: module, name: module, state: state}]}}
        {:error, message} -> {:halt, {:error, "#{inspect(module)}: #{message}"}}
      end
    end)
  end

  @doc "The tensors `members` read that no member before the reader provides."
  @spec entry_fields([member()]) :: [atom()]
  def entry_fields(members) do
    {entry, _provided} =
      Enum.reduce(members, {[], MapSet.new()}, fn member, {entry, provided} ->
        needed =
          Enum.reject(member.module.requires(), &(MapSet.member?(provided, &1) or &1 in entry))

        {entry ++ needed, MapSet.union(provided, MapSet.new(member.module.provides()))}
      end)

    entry
  end

  @doc """
  Each of the `entry_fields/1` of `members` with the shape and type it
  enters a segment with in `geometry`: the camera frames (`:left_frame`,
  `:right_frame`) as `u8` at the frame size, the rectified images
  (`:left`, `:right`) as `u8` and `:disparity` as `f32` at the working
  size. Another tensor cannot enter one.

  With `packed: true`, the frames and images enter as `u32` words of four
  pixels each (`NxStereo.Packing`), at a quarter of their width, which
  must be a multiple of 4.
  """
  @spec entry_specs([member()], NxStereo.Stage.geometry(), keyword()) ::
          {:ok, [{atom(), entry_spec()}]} | {:error, String.t()}
  def entry_specs(members, geometry, opts \\ []) do
    packed? = Keyword.get(opts, :packed, false)

    members
    |> entry_fields()
    |> Enum.reduce_while({:ok, []}, fn field, {:ok, specs} ->
      case entry_spec(field, geometry) do
        {:ok, spec} when packed? and field in @packable ->
          case packed_spec(field, spec) do
            {:ok, spec} -> {:cont, {:ok, specs ++ [{field, spec}]}}
            {:error, _message} = error -> {:halt, error}
          end

        {:ok, spec} ->
          {:cont, {:ok, specs ++ [{field, spec}]}}

        :error ->
          {:halt,
           {:error,
            "no stage before it provides #{inspect(field)}, and a segment cannot take it as an entry"}}
      end
    end)
  end

  defp packed_spec(_field, {{height, width}, {:u, 8}}) when rem(width, 4) == 0,
    do: {:ok, {{height, div(width, 4)}, {:u, 32}}}

  defp packed_spec(field, {{_height, width}, _type}),
    do: {:error, "#{inspect(field)} is #{width} pixels wide: packing needs a multiple of 4"}

  defp entry_spec(field, geometry) when field in @frames,
    do: {:ok, {{geometry.frame_height, geometry.frame_width}, {:u, 8}}}

  defp entry_spec(field, geometry) when field in @images,
    do: {:ok, {{geometry.height, geometry.width}, {:u, 8}}}

  defp entry_spec(field, geometry) when field in @maps,
    do: {:ok, {{geometry.height, geometry.width}, {:f, 32}}}

  defp entry_spec(_field, _geometry), do: :error

  @doc "The camera frames among the entry fields, which a segment reads and returns nothing of."
  @spec frame_fields() :: [atom()]
  def frame_fields, do: @frames

  @doc """
  The `entries` that `specs`, from `entry_specs/3`, name, each a tensor
  of its shape and type, packed when its spec says so; or why not: an
  entry of another shape (`{:frame_size, field, shape, expected}`), one
  missing (`{:missing_entry, field}`), or one that is no tensor
  (`{:not_a_tensor, field, value}`).
  """
  @spec entry_tensors(entries() | [{atom(), term()}], [{atom(), entry_spec()}]) ::
          {:ok, [{atom(), Nx.Tensor.t()}]} | {:error, term()}
  def entry_tensors(entries, specs) do
    entries = Map.new(entries)

    Enum.reduce_while(specs, {:ok, []}, fn {field, {expected, type}}, {:ok, tensors} ->
      with {:ok, tensor} <- fetch_tensor(entries, field),
           {:ok, tensor} <- entry_tensor(field, tensor, expected, type) do
        {:cont, {:ok, tensors ++ [{field, tensor}]}}
      else
        {:error, _reason} = error -> {:halt, error}
      end
    end)
  end

  defp fetch_tensor(entries, field) do
    case Map.fetch(entries, field) do
      {:ok, %Nx.Tensor{} = tensor} -> {:ok, tensor}
      {:ok, other} -> {:error, {:not_a_tensor, field, other}}
      :error -> {:error, {:missing_entry, field}}
    end
  end

  # A packed entry: the u8 image's bytes as u32 words.
  defp entry_tensor(field, tensor, {height, words}, {:u, 32}) when field in @packable do
    with :ok <- check_shape(field, Nx.shape(tensor), {height, words * 4}),
         do:
           {:ok,
            tensor |> Nx.as_type({:u, 8}) |> Nx.to_binary() |> Packing.pack(words * 4, height)}
  end

  defp entry_tensor(field, tensor, expected, type) do
    with :ok <- check_shape(field, Nx.shape(tensor), expected),
         do: {:ok, Nx.as_type(tensor, type)}
  end

  defp check_shape(_field, shape, shape), do: :ok

  defp check_shape(field, shape, expected),
    do: {:error, {:frame_size, field, shape, expected}}
end
