# Vulkan tests run with `mix test --include vulkan`, on the devices whose URI
# contains NX_STEREO_DEVICE (all Vulkan devices by default).
Nx.Defn.global_default_options(compiler: EXLA)
ExUnit.start(exclude: [:vulkan])
