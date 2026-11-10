defmodule NxStereo.Stage do
  @moduledoc """
  One step of a stereo depth pipeline, such as rectification, matching or
  a filter, written so that several stages compile into one Nx function.

  A stage reads named tensors and adds its own: `step/3` takes the
  tensors so far and returns them with its outputs added or replaced.
  The function a run of stages compiles into takes each stage's
  `params/2` as inputs: what may change without compiling again, such as
  calibration values or a live setting. Everything in the stage's state
  is fixed in the compiled function.

  The geometry a stage is built for is a map with the working size
  (`:width`, `:height`), the camera frames' size (`:frame_width`,
  `:frame_height`), the rectified pair's `:focal_length_px`,
  `:baseline_m` and `:principal_point` (`{cx, cy}`), and, for
  rectification, each camera as `NxStereo.Rectify.camera/4` gives it
  (`:left_camera`, `:right_camera`). A stage reads the keys it needs.

  Tensors are named by atoms, such as `:left`, `:right`, `:disparity` or
  `:depth_m`: a stage reads its `requires/0` and adds or replaces its
  `provides/0`. A value carried from one pair to the next is named
  `{:carry, name}` (`carry/2`), and an image packed four pixels to a
  32-bit word for a GPU without 8-bit storage `{:packed, name}`
  (`NxStereo.Packing`).

  A stage may also be computed natively, outside Nx: `native_kind/0`
  names its native implementation and `native_constants/0` the constants
  both share.

  A list of stages runs on pairs as one segment (`NxStereo.Segment`):
  compiled as one Nx function (`NxStereo.NxSegment`) or computed
  natively in the `stereo_match` Port (`NxStereo.NativeSegment`).
  """

  @type state :: term()
  @type geometry :: %{optional(atom()) => term()}

  @doc """
  Builds the stage's state from its settings and the geometry, or says
  why it cannot: a setting it does not take, or a value it cannot
  compute with.
  """
  @callback init(settings :: keyword(), geometry()) :: {:ok, state()} | {:error, String.t()}

  @doc "The tensors the stage reads."
  @callback requires() :: [atom()]

  @doc """
  The tensors the stage adds or replaces, including those it leaves for
  the stages after it only (`c:device_only/0`).
  """
  @callback provides() :: [atom()]

  @doc """
  The compiled function's inputs for this stage: tensors, numbers, or
  maps of tensors such as a camera model. Without it, the stage takes
  none.
  """
  @callback params(state(), geometry()) ::
              %{atom() => Nx.Tensor.t() | number() | %{atom() => Nx.Tensor.t()}}

  @doc """
  Called while tracing: takes the named tensors so far and the stage's
  params, and returns the tensors with this stage's outputs added or
  replaced.
  """
  @callback step(tensors :: map(), params :: map(), state()) :: map()

  @doc """
  Values the stage carries from one pair to the next, with their values
  before the first pair. `step/3` reads them among its params and puts
  their next values in its tensors as `{:carry, name}`.
  """
  @callback carry(state(), geometry()) :: %{atom() => Nx.Tensor.t()}

  @doc """
  Outputs the stage leaves for the stages after it only: a run of stages
  on a device does not download them.
  """
  @callback device_only() :: [atom()]

  @doc """
  Where the stage asks to be computed: in Nx (`:nx`), natively
  (`:native`), or by whichever backend runs it (`nil`).
  """
  @callback compute(state()) :: :nx | :native | nil

  @doc """
  The name of the stage's native implementation; a stage without one is
  computed in Nx only.
  """
  @callback native_kind() :: String.t()

  @doc "The constants the stage's step shares with its native implementation, by name."
  @callback native_constants() :: %{atom() => number()}

  @doc """
  The numeric settings, fixed when compiled, that the stage's native
  implementation takes, by name. Its params go with each pair instead.
  """
  @callback native_settings(state()) :: keyword(number())

  @doc """
  `settings` as a map over `defaults`, for a stage's `init/2`: a setting
  not among `defaults` is an error.
  """
  @spec settings(keyword(), keyword()) :: {:ok, map()} | {:error, String.t()}
  def settings(settings, defaults) do
    case Keyword.validate(settings, defaults) do
      {:ok, settings} -> {:ok, Map.new(settings)}
      {:error, unknown} -> {:error, "unknown settings #{inspect(unknown)}"}
    end
  end

  @doc "The inputs `module`, in `state`, takes for `geometry`; see `c:params/2`."
  @spec params(module(), state(), geometry()) :: map()
  def params(module, state, geometry) do
    if function_exported?(module, :params, 2), do: module.params(state, geometry), else: %{}
  end

  @doc "The values `module`, in `state`, carries from pair to pair; see `c:carry/2`."
  @spec carry(module(), state(), geometry()) :: %{atom() => Nx.Tensor.t()}
  def carry(module, state, geometry) do
    if function_exported?(module, :carry, 2), do: module.carry(state, geometry), else: %{}
  end

  @doc "Where `module`, in `state`, asks to be computed; see `c:compute/1`."
  @spec compute(module(), state()) :: :nx | :native | nil
  def compute(module, state) do
    if function_exported?(module, :compute, 1), do: module.compute(state)
  end

  @doc "The outputs `module` leaves on the device; see `c:device_only/0`."
  @spec device_only(module()) :: [atom()]
  def device_only(module) do
    if function_exported?(module, :device_only, 0), do: module.device_only(), else: []
  end

  @doc "The working size of `geometry`, as `{height, width}`."
  @spec size(geometry()) :: {pos_integer(), pos_integer()}
  def size(geometry), do: {geometry.height, geometry.width}

  @doc "The native implementation `module` declares, or `nil`; see `c:native_kind/0`."
  @spec native_kind(module()) :: String.t() | nil
  def native_kind(module) do
    if Code.ensure_loaded?(module) and function_exported?(module, :native_kind, 0),
      do: module.native_kind()
  end

  @doc "The constants `module` shares with its native implementation; see `c:native_constants/0`."
  @spec native_constants(module()) :: %{atom() => number()}
  def native_constants(module) do
    if function_exported?(module, :native_constants, 0),
      do: module.native_constants(),
      else: %{}
  end

  @doc "The native build settings of `module` in `state`; see `c:native_settings/1`."
  @spec native_settings(module(), state()) :: keyword(number())
  def native_settings(module, state) do
    if function_exported?(module, :native_settings, 1),
      do: module.native_settings(state),
      else: []
  end

  @optional_callbacks params: 2,
                      carry: 2,
                      device_only: 0,
                      compute: 1,
                      native_kind: 0,
                      native_constants: 0,
                      native_settings: 1
end
