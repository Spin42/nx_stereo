# Vulkan tests run with `mix test --include vulkan`, on the devices whose URI
# contains NX_STEREO_DEVICE (all Vulkan devices by default); `--include iree`
# runs the tests that compile for the CPU through NxIREE.
# The :native tests run the stereo_match nx_stereo builds when cargo is installed.
Nx.Defn.global_default_options(compiler: EXLA)
native = if File.exists?(NxStereo.Matcher.default_path()), do: [], else: [:native]
ExUnit.start(exclude: [:vulkan, :iree | native])

defmodule NxStereo.TestDevices do
  @moduledoc false

  # The Vulkan devices whose URI contains NX_STEREO_DEVICE, as a setup's context.
  def vulkan do
    {:ok, devices} = NxIREE.list_devices("vulkan")
    filter = System.get_env("NX_STEREO_DEVICE", "")
    devices = Enum.filter(devices, &String.contains?(&1.uri, filter))
    ExUnit.Assertions.assert(devices != [], "no Vulkan device matches #{inspect(filter)}")
    {:ok, devices: devices}
  end
end
