# NxStereo

Semi-global stereo matching for [Nx](https://github.com/elixir-nx/nx).

```elixir
disparity = NxStereo.sgm(left, right, disparities: 64, p1: 8, p2: 64)
```

`left` and `right` are rectified `{height, width}` `u8` images; the result is
an `f32` disparity map with `-1.0` where no reliable match was found. The
algorithm is libSGM's: a 9x7 Census transform, Hamming costs, three
aggregation paths with the normalized recurrence, winner selection with a
uniqueness check and parabolic subpixel refinement, OpenCV's left-right
consistency check and a 3x3 median.

`NxStereo.sgm/3` is a `defn` transform and runs on any Nx backend or
compiler. Compiled by [NxIREE](https://github.com/elixir-nx/nx_iree) for
Vulkan, the same call runs as six compute shaders (an
`NxIREE.Vulkan.Pipeline` block) with the same matches; only subpixel values
may differ in their last bit.

```elixir
Nx.Defn.jit(&NxStereo.sgm/2,
  compiler: NxIREE.Compiler,
  iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv"]
).(left, right)
```

On the OKdo C100 (Jetson Nano GM20B, Mesa NVK) a 640x480 pair with 64
disparities takes 31 ms with the images already on the device, matching the
EXLA reference on every pixel.

## Shaders

The GLSL sources in `shaders/` are ported from
[Fixstars libSGM](https://github.com/fixstars/libSGM) (Apache-2.0) and
OpenCV's StereoSGBM consistency rules (BSD-3-Clause, see `OPENCV-LICENSE`).
Image size and penalties are specialization constants, so the SPIR-V in
`priv/spirv` is compiled once per disparity range; `mix nx_stereo.shaders`
regenerates it with `glslangValidator`.

## Tests

`mix test` checks the reference implementation on EXLA and Nx's evaluator.
`mix test --include vulkan` also compares the shaders with it on every
Vulkan device (`NX_STEREO_DEVICE` selects devices by URI substring).
