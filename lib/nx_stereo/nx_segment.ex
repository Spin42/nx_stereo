defmodule NxStereo.NxSegment do
  @moduledoc """
  A list of stages compiled into one Nx function when the segment is
  built, and run on each pair as one call: an `NxStereo.Segment`.

  The function takes the pair's entries (`NxStereo.Segment.entry_fields/1`,
  camera frames as `u8` images) and each stage's `params/2`, and returns
  every tensor but the frames and those the stages keep on the device
  (`c:NxStereo.Stage.device_only/0`). Outputs are published as one `u32`
  tensor and come back to the CPU in one download, since each device
  transfer has a fixed cost; `collect/3` unbundles them.

  It is compiled at the geometry's size with the `:defn_options` of its
  build options (`build/3`). Two of them are the segment's own:
  `byte_storage: false`, for a GPU without 8-bit storage, moves frames
  and images as `u32` words of four pixels (`NxStereo.Packing`); and
  `cache_only: true`, with `NxIREE.Compiler`, only writes the module to
  the module cache (`compile/3`). Since calibration values are inputs, a
  calibration reload (`reload/3`) needs no new function.

  The values the stages carry from one pair to the next
  (`c:NxStereo.Stage.carry/2`) stay on the device, in the segment
  `response/3` returns. On an `NxIREE` device the inputs stay there
  between pairs too.

  `send_request/2` runs the function in a `Task`; `collect/3` downloads
  the outputs in a `Task` of its own, so a pair's download overlaps the
  next pair's device work. What the function or the download raises
  fails that pair only.
  """

  @behaviour NxStereo.Segment

  alias NxStereo.{Packing, Segment, Stage}

  @enforce_keys [:members, :specs, :function, :params, :carried, :geometry, :defn_options]
  defstruct [
    :members,
    :specs,
    :function,
    :params,
    :carried,
    :geometry,
    :defn_options,
    packed: false,
    layout: [],
    placement: nil,
    version: 0
  ]

  @type t :: %__MODULE__{}

  @doc """
  Compiles `members` for `geometry`, with `opts`: `:defn_options` (`[]`,
  Nx's own defaults), how; see `compile/3`.
  """
  @impl NxStereo.Segment
  @spec build([Segment.member()], Stage.geometry(), keyword()) ::
          {:ok, t()} | {:error, String.t()}
  def build(members, geometry, opts) do
    case Keyword.validate(opts, defn_options: []) do
      {:ok, opts} -> compile_members(members, geometry, opts[:defn_options])
      {:error, unknown} -> {:error, "unknown build options #{inspect(unknown)}"}
    end
  end

  defp compile_members(members, geometry, defn_options) do
    params = params(members, geometry)
    {byte_storage?, defn_options} = Keyword.pop(defn_options, :byte_storage, true)
    packed? = not byte_storage?

    with {:ok, specs} <- Segment.entry_specs(members, geometry, packed: packed?),
         entry_templates =
           Map.new(specs, fn {field, {shape, type}} -> {field, Nx.template(shape, type)} end),
         run = &run_packed(members, &1, &2, packed?),
         {:ok, layout} <- layout(run, entry_templates, params, members),
         {:ok, function} <-
           compile(
             &bundle(run.(&1, &2), layout),
             [entry_templates, Nx.to_template(params)],
             defn_options
           ) do
      placement = placement(defn_options)

      {:ok,
       %__MODULE__{
         members: members,
         specs: specs,
         function: function,
         params: place(params, placement),
         carried: carried(members, geometry),
         geometry: geometry,
         defn_options: defn_options,
         packed: packed?,
         layout: layout,
         placement: placement
       }}
    end
  end

  @doc "The outputs a pair's published tensor holds, in the order they are packed."
  @spec published(t()) :: [term()]
  def published(%__MODULE__{layout: layout}), do: Enum.map(layout, &elem(&1, 0))

  # Each downloaded output's shape, type and offset in 32-bit words of the bundle.
  defp layout(run, entry_templates, params, members) do
    device_only = device_only(members)

    run
    |> Nx.Defn.debug_expr()
    |> apply([entry_templates, Nx.to_template(params)])
    |> Enum.reject(fn {key, _tensor} ->
      match?({:carry, _index, _name}, key) or key in device_only
    end)
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.map_reduce(0, fn {key, tensor}, offset ->
      words = words_for(Nx.shape(tensor), Nx.type(tensor))
      {{key, Nx.shape(tensor), Nx.type(tensor), offset}, offset + words}
    end)
    |> then(&{:ok, elem(&1, 0)})
  rescue
    error -> {:error, compile_error_message(error)}
  end

  # A section's words, rounded up so the next starts on a 256-byte boundary: IREE may
  # have a kernel write its output straight into the bundle, which needs an aligned offset.
  @section_words 64
  defp words_for(shape, {_kind, bits}) do
    words = div(Tuple.product(shape) * bits + 31, 32)
    div(words + @section_words - 1, @section_words) * @section_words
  end

  defp bundle(results, []),
    do: Map.filter(results, fn {key, _} -> match?({:carry, _, _}, key) end)

  defp bundle(results, layout) do
    words =
      Enum.map(layout, fn {key, shape, type, _offset} ->
        section = results |> Map.fetch!(key) |> to_words()
        Nx.pad(section, 0, [{0, words_for(shape, type) - Nx.size(section), 0}])
      end)

    results
    |> Map.filter(fn {key, _tensor} -> match?({:carry, _index, _name}, key) end)
    |> Map.put(:published, Nx.concatenate(words))
  end

  # A tensor's bytes as little-endian u32 words, its last word zero-padded.
  defp to_words(tensor) do
    {_kind, bits} = Nx.type(tensor)
    flat = tensor |> Nx.bitcast({:u, bits}) |> Nx.flatten()

    case bits do
      32 ->
        flat

      bits when bits in [8, 16] ->
        lanes = div(32, bits)
        padding = rem(lanes - rem(Nx.size(flat), lanes), lanes)

        flat
        |> Nx.pad(0, [{0, padding, 0}])
        |> Nx.as_type({:u, 32})
        |> Nx.reshape({:auto, lanes})
        |> then(fn grouped ->
          for lane <- 0..(lanes - 1) do
            grouped
            |> Nx.slice_along_axis(lane, 1, axis: 1)
            |> Nx.flatten()
            |> Nx.left_shift(lane * bits)
          end
        end)
        |> Enum.reduce(&Nx.bitwise_or/2)
    end
  end

  # On an IREE device, inputs live there between pairs, so each call reads them in place.
  defp placement(defn_options) do
    if Keyword.get(defn_options, :compiler) == NxIREE.Compiler and
         not Keyword.get(defn_options, :cache_only, false) do
      device = defn_options |> Keyword.get(:iree_runtime_options, []) |> Keyword.get(:device)
      {NxIREE.Backend, device: NxIREE.Device.fetch!(device)}
    end
  end

  defp place(container, nil), do: container

  defp place(container, placement) do
    Nx.Defn.Composite.traverse(container, fn
      %Nx.Tensor{data: %NxIREE.Backend{}} = tensor -> tensor
      %Nx.Tensor{} = tensor -> Nx.backend_transfer(tensor, placement)
    end)
  end

  @doc "Every stage with a `step/3`, to compile it."
  @impl NxStereo.Segment
  @spec accepts?(module(), term(), :nx | :native) :: boolean()
  def accepts?(module, _state, compute),
    do: compute == :nx and Code.ensure_loaded?(module) and function_exported?(module, :step, 3)

  @doc """
  Compiles `function` for `templates` with `defn_options`, an
  `:iree_module_cache` of `{app, path}` being a directory in that app.
  A module missing from the cache of a compiler that cannot compile is
  an error that starts with "no compiled module". With `cache_only:
  true` and `NxIREE.Compiler`, the module is only written to the module
  cache, for a device of another kind than the host: the function
  returned raises.
  """
  @spec compile(function(), [term()], keyword()) :: {:ok, function()} | {:error, String.t()}
  def compile(function, templates, defn_options) do
    {cache_only?, defn_options} = Keyword.pop(defn_options, :cache_only, false)
    defn_options = resolve_module_cache(defn_options)

    if cache_only? do
      {compiler, defn_options} = Keyword.pop(defn_options, :compiler)
      {:ok, _module} = compiler.to_bytecode(function, templates, defn_options)
      {:ok, cache_only_function(length(templates))}
    else
      {:ok, Nx.Defn.compile(function, templates, defn_options)}
    end
  rescue
    error -> {:error, compile_error_message(error)}
  end

  defp cache_only_function(1), do: fn _input -> raise "compiled for the module cache only" end

  defp cache_only_function(2),
    do: fn _input, _params -> raise "compiled for the module cache only" end

  @doc """
  The segment with `members` re-initialised for a new `geometry` of the
  same size: the same function, with new inputs, and what the stages
  carry started over.
  """
  @impl NxStereo.Segment
  @spec reload(t(), [Segment.member()], Stage.geometry()) :: t()
  def reload(%__MODULE__{} = segment, members, geometry) do
    %{
      segment
      | members: members,
        params: place(params(members, geometry), segment.placement),
        geometry: geometry,
        version: segment.version + 1
    }
  end

  @doc """
  The segment with the member at `index` in a new `state`: its new
  inputs, the same function, and what every stage carries kept.
  """
  @impl NxStereo.Segment
  @spec put_member_state(t(), non_neg_integer(), term()) :: t()
  def put_member_state(%__MODULE__{members: members} = segment, index, state) do
    members = List.update_at(members, index, &%{&1 | state: state})
    member = Enum.at(members, index)

    inputs =
      member.module
      |> Stage.params(member.state, segment.geometry)
      |> Map.new(fn {key, value} -> {key, to_input(value)} end)
      |> place(segment.placement)

    carried = Map.take(segment.params[index], segment.carried[index])

    %{
      segment
      | members: members,
        params: Map.put(segment.params, index, Map.merge(inputs, carried))
    }
  end

  @doc """
  `current` with the values `replied` carries, unless `current` was
  reloaded since `replied`'s pair was sent: the reload started them over.
  """
  @impl NxStereo.Segment
  @spec merge_carry(t(), t()) :: t()
  def merge_carry(
        %__MODULE__{version: version} = current,
        %__MODULE__{version: version} = replied
      ) do
    params =
      Map.new(current.params, fn {index, inputs} ->
        carried = Map.take(Map.fetch!(replied.params, index), current.carried[index])
        {index, Map.merge(inputs, carried)}
      end)

    %{current | params: params}
  end

  def merge_carry(%__MODULE__{} = current, %__MODULE__{}), do: current

  @doc """
  Takes a pair's `entries` and runs the compiled function on them in a
  `Task`, without waiting: returns the request, which `response/3` and
  `await/3` recognise its outputs by; or why the entries do not fit
  (`NxStereo.Segment.entry_tensors/2`). A pair sent before the previous
  one's response would start from the values carried before that pair.
  """
  @impl NxStereo.Segment
  @spec send_request(Segment.entries(), t()) :: {:ok, Task.t()} | {:error, term()}
  def send_request(entries, %__MODULE__{} = segment) do
    with {:ok, entry} <- Segment.entry_tensors(entries, segment.specs),
         do: {:ok, Task.async(fn -> dispatch(Map.new(entry), segment) end)}
  end

  # The outputs stay on the device; the segment carries the stages' values to the next pair.
  defp dispatch(entry, segment) do
    with {:ok, results} <- call(segment, place(entry, segment.placement)) do
      {carry, outputs} =
        results
        |> Map.drop(device_only(segment.members))
        |> Map.split_with(fn {key, _tensor} -> match?({:carry, _index, _name}, key) end)

      params =
        Enum.reduce(carry, segment.params, fn {{:carry, index, name}, tensor}, params ->
          put_in(params, [index, name], tensor)
        end)

      {:ok, outputs, %{segment | params: params}, System.monotonic_time(:nanosecond)}
    end
  end

  defp call(segment, entry) do
    {:ok, segment.function.(entry, segment.params)}
  rescue
    error -> {:error, error}
  end

  @doc """
  The outputs of the pair sent as `task`, left on the device, the
  segment carrying its stages' values to the next pair and the
  monotonic time the device was done, or what the compiled function
  raised, when `message` is its reply; `:no_reply` when it is another
  message.
  """
  @impl NxStereo.Segment
  @spec response(term(), Task.t(), t()) ::
          {:ok, map(), t(), integer()} | {:error, term()} | :no_reply
  def response({ref, dispatched}, %Task{ref: ref}, %__MODULE__{}) do
    Process.demonitor(ref, [:flush])
    dispatched
  end

  def response(_message, %Task{}, %__MODULE__{}), do: :no_reply

  @doc "What `response/3` returns for the pair sent as `task`, within `timeout` ms, or `:timeout`."
  @impl NxStereo.Segment
  @spec await(Task.t(), t(), timeout()) ::
          {:ok, map(), t(), integer()} | {:error, term()} | :timeout
  def await(%Task{} = task, %__MODULE__{}, timeout) do
    case Task.yield(task, timeout) do
      {:ok, dispatched} -> dispatched
      nil -> :timeout
    end
  end

  @doc """
  One pair: a pair's device work starts from the values the previous
  one carries.
  """
  @impl NxStereo.Segment
  @spec depth() :: pos_integer()
  def depth, do: 1

  @doc "Each member's time on its own: none, since the members run as one function."
  @impl NxStereo.Segment
  @spec member_times(t()) :: [{term(), non_neg_integer()}]
  def member_times(%__MODULE__{}), do: []

  @doc """
  Downloads `outputs` from the device, freeing them there, in a `Task`
  that has `timeout` ms: the outputs on the CPU, packed images unpacked
  to `u8`; or what the download raised, or `:timeout`.
  """
  @impl NxStereo.Segment
  @spec collect(map(), t(), timeout()) :: {:ok, map()} | {:error, term()} | :timeout
  def collect(outputs, %__MODULE__{} = segment, timeout) do
    download = Task.async(fn -> download(outputs, segment.layout) end)

    case Task.yield(download, timeout) do
      {:ok, result} -> result
      nil -> :timeout
    end
  end

  defp download(outputs, layout) do
    {bundle, loose} = Map.pop(outputs, :published)

    unbundled =
      if bundle, do: unbundle(Nx.to_binary(Nx.backend_transfer(bundle)), layout), else: []

    {:ok,
     (Enum.map(loose, fn {key, tensor} -> {key, Nx.backend_transfer(tensor)} end) ++ unbundled)
     |> Map.new(fn
       {{:packed, key}, tensor} -> {key, Packing.unpack_download(tensor)}
       entry -> entry
     end)}
  rescue
    error -> {:error, error}
  end

  defp unbundle(binary, layout) do
    for {key, shape, {_kind, bits} = type, offset} <- layout do
      bytes = div(Tuple.product(shape) * bits, 8)
      {key, binary |> binary_part(offset * 4, bytes) |> Nx.from_binary(type) |> Nx.reshape(shape)}
    end
  end

  # Without byte storage, frames and images cross as 32-bit words; frames go to the first stage packed.
  defp run_packed(members, tensors, params, false), do: run_steps(members, tensors, params)

  defp run_packed(members, tensors, params, true) do
    frames = Segment.frame_fields()

    outputs =
      tensors
      |> Map.new(fn {key, tensor} ->
        if key not in frames and Nx.type(tensor) == {:u, 32},
          do: {key, Packing.unpack(tensor, {:s, 32})},
          else: {key, tensor}
      end)
      |> then(&run_steps(members, &1, params))

    Enum.reduce([:left, :right], outputs, fn key, outputs ->
      case Map.pop(outputs, key) do
        {%Nx.Tensor{type: {:s, 32}} = image, rest} ->
          Map.put_new_lazy(rest, {:packed, key}, fn -> Packing.pack_image(image) end)

        _other ->
          outputs
      end
    end)
  end

  defp run_steps(members, tensors, params) do
    members
    |> Enum.with_index()
    |> Enum.reduce(tensors, fn {member, index}, tensors ->
      tensors
      |> member.module.step(Map.fetch!(params, index), member.state)
      |> Map.new(fn
        {{:carry, name}, tensor} -> {{:carry, index, name}, tensor}
        entry -> entry
      end)
    end)
    |> Map.drop(Segment.frame_fields())
  end

  defp params(members, geometry) do
    members
    |> Enum.with_index()
    |> Map.new(fn {member, index} ->
      values =
        Map.merge(
          Stage.params(member.module, member.state, geometry),
          Stage.carry(member.module, member.state, geometry)
        )

      {index, Map.new(values, fn {key, value} -> {key, to_input(value)} end)}
    end)
  end

  # The names of the values each member carries, by its index.
  defp carried(members, geometry) do
    members
    |> Enum.with_index()
    |> Map.new(fn {member, index} ->
      {index, Map.keys(Stage.carry(member.module, member.state, geometry))}
    end)
  end

  defp device_only(members), do: Enum.flat_map(members, &Stage.device_only(&1.module))

  defp to_input(value) when is_number(value), do: Nx.tensor(value, type: :f32)
  defp to_input(value), do: value

  @doc """
  Where the segment, or `defn_options`, run a segment, and with what:
  Nx's `:evaluator` without a compiler or with `Nx.Defn.Evaluator`; a
  compiler other than it, `NxIREE.Compiler` and EXLA has no backend name
  (`nil`).
  """
  @impl NxStereo.Segment
  @spec compute(t() | keyword()) :: Segment.compute()
  def compute(%__MODULE__{defn_options: defn_options}), do: compute(defn_options)

  def compute(defn_options) do
    {runs_on, backend} =
      case Keyword.get(defn_options, :compiler) do
        NxIREE.Compiler -> iree_target(Keyword.get(defn_options, :iree_compiler_flags, []))
        EXLA -> {exla_device(Keyword.get(defn_options, :client, :host)), :exla}
        evaluator when evaluator in [nil, Nx.Defn.Evaluator] -> {:cpu, :evaluator}
        _other -> {:cpu, nil}
      end

    %{runs_on: runs_on, implementation: :nx, backend: backend}
  end

  defp iree_target(flags) do
    if Enum.any?(flags, &String.contains?(&1, "vulkan")),
      do: {:gpu, :vulkan},
      else: {:cpu, :llvm_cpu}
  end

  defp exla_device(client) when client in [:cuda, :rocm], do: :gpu
  defp exla_device(_client), do: :cpu

  @doc "`defn_options` with an `:iree_module_cache` of `{app, path}` resolved to that app's directory."
  @spec resolve_module_cache(keyword()) :: keyword()
  def resolve_module_cache(defn_options) do
    case Keyword.fetch(defn_options, :iree_module_cache) do
      {:ok, {app, path}} ->
        Keyword.put(defn_options, :iree_module_cache, Application.app_dir(app, path))

      _other ->
        defn_options
    end
  end

  defp compile_error_message(%{__struct__: NxIREE.CompileError} = error),
    do:
      "no compiled module for it in the module cache: build it on a host " <>
        "(#{Exception.message(error)})"

  defp compile_error_message(error), do: "cannot compile it: " <> Exception.message(error)
end
