# Vulkan tests run with `mix test --include vulkan`, on the devices whose URI
# contains NX_STEREO_DEVICE (all Vulkan devices by default); `--include iree`
# runs the tests that compile for the CPU through NxIREE.
# The :native tests run the stereo_match nx_stereo builds when cargo is installed.
Nx.Defn.global_default_options(compiler: EXLA)
native = if File.exists?(NxStereo.Matcher.default_path()), do: [], else: [:native]
ExUnit.start(exclude: [:vulkan, :iree | native])
