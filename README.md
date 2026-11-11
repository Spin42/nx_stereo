# NxStereo

Stereo depth for [Nx](https://github.com/elixir-nx/nx): from a calibrated
camera pair to a disparity map, metric depth and a point cloud, with
measures of the rig's health. Every step is an Nx function, so it runs
wherever Nx does, and has two faster implementations behind it:
hand-written Vulkan shaders, when [NxIREE](https://github.com/Spin42/nx_iree)
compiles it for a GPU, and native Rust, for a CPU.

## A depth pipeline

A pipeline is a list of stages, each a step such as rectification or
matching, that run on a pair of camera frames and add their outputs to it.
A segment runs a list of stages on a backend: `NxStereo.NxSegment`
compiles them into one Nx function, `NxStereo.NativeSegment` computes them
in a native program. From frames to depth and a point cloud:

```elixir
alias NxStereo.{Diagnostics, Rectify, Segment, Stage}

# The rectified pair's geometry: working size, frame size, and each camera
# from its calibration (camera matrix, distortion, rectification, projection).
geometry = %{
  width: 364,
  height: 272,
  frame_width: 728,
  frame_height: 544,
  focal_length_px: 436.5,
  baseline_m: 0.2,
  principal_point: {182.0, 136.0},
  left_camera: Rectify.camera(left_k, left_d, left_r, left_p),
  right_camera: Rectify.camera(right_k, right_d, right_r, right_p)
}

{:ok, members} =
  Segment.members(
    [
      Stage.Rectification,
      {Stage.Disparity, num_disparities: 64, paths: 8},
      Stage.SpeckleFilter,
      Stage.Depth,
      {Stage.PointCloud, decimation: 4},
      Stage.Quality
    ],
    geometry
  )

{:ok, segment} = NxStereo.NxSegment.build(members, geometry, defn_options: [compiler: EXLA])

# For each pair of u8 camera frames:
{:ok, outputs, segment} =
  Segment.run(NxStereo.NxSegment, [left_frame: left, right_frame: right], segment, 5_000)

outputs.disparity       # f32 {272, 364}, -1.0 where unmatched
outputs.depth_mm        # u16 {272, 364}, 0 where unmatched
outputs.cloud_grid      # f32 {68, 91, 3}, NaN outside 5 cm to 20 m
Diagnostics.disparity_stats(outputs.quality_sums, 364 * 272)
#=> %{matched: 0.95, mean_px: 16.16, spread_px: 0.93, jitter_px: 0.0}
```

Keep the segment `run/4` returns for the next pair: it carries what the
stages pass from one pair to the next, such as the temporal filter's
previous disparity. To compute the same stages natively, start the
native program's server and build a native segment instead:

```elixir
{:ok, _matcher} = NxStereo.Matcher.start_link()
{:ok, segment} = NxStereo.NativeSegment.build(members, geometry, [])
```

### The stages

Each stage reads tensors by name and adds its own. Its settings are its
options in the list; those marked live are inputs of the compiled function,
changed without compiling again, the others are fixed when it compiles.

| Stage | Reads | Adds | Settings | Native |
| --- | --- | --- | --- | --- |
| `Stage.Rectification` | `:left_frame`, `:right_frame` | `:left`, `:right` | `:right_row_offset` (live) | yes |
| `Stage.Decode` | `:left_frame`, `:right_frame` | `:left`, `:right`, area-resized, for frames already rectified | | no |
| `Stage.Sharpness` | `:left`, `:right` | `:sharpness_sums` | | yes |
| `Stage.Disparity` | `:left`, `:right` | `:disparity` | `:p1`, `:p2`, `:uniqueness` (live); `:num_disparities`, `:paths`, `:search`, `:coarse_paths`, `:band`, `:margin`, `:fine_census` | full search only |
| `Stage.MedianFilter` | `:disparity` | `:disparity` | `:kernel_size` | yes |
| `Stage.SpeckleFilter` | `:disparity` | `:disparity` | `:min_support`, `:max_diff` (live) | yes |
| `Stage.RowAlignment` | `:left`, `:right`, `:disparity` | `:alignment_sums` | | yes |
| `Stage.TemporalFilter` | `:disparity` | `:disparity`, carrying the previous one | `:alpha`, `:delta`, `:hold` (live) | yes |
| `Stage.Depth` | `:disparity` | `:depth_mm`, `:depth_m` | | yes |
| `Stage.PointCloud` | `:depth_m` | `:cloud_grid`, `:cloud_points` | `:decimation` | yes |
| `Stage.Quality` | `:disparity` | `:quality_sums`, carrying the previous disparity | | yes |

`NxStereo.Diagnostics` turns the `*_sums` into measures on the CPU:
`sharpness/1`, `row_alignment/2` and `disparity_stats/2`. A stage of your
own implements `NxStereo.Stage` and joins the list like the others.

### Which backend

| Backend | Runs on | Choose it for |
| --- | --- | --- |
| `NxSegment` with `NxIREE.Compiler` for Vulkan | a GPU, as the shaders in `shaders/spirv` | a board with a Vulkan GPU: 30 pairs a second at 728x544 over 128 disparities on an Arduino VENTUNO Q's Adreno 623 |
| `NativeSegment` | every core of a CPU, in `stereo_match` | a board without a usable GPU: 16 to 21 pairs a second at 480x270 over 64 disparities on a Raspberry Pi 5 |
| `NxSegment` with EXLA | a CPU or CUDA GPU, as XLA code | a workstation, and the reference the others are checked against |
| `NxSegment` with Nx alone | a CPU, in Elixir | checking results; far too slow for video |

A stage with no native implementation, or `Stage.Disparity` with
`search: "coarse_to_fine"`, runs in an `NxSegment`: `NativeSegment.build/3`
refuses it. To overlap pairs, send one with `send_request/2` while the
previous one's outputs are collected (`NxStereo.Segment`): a backend's
`depth/0` says how many pairs it keeps in flight, one on a GPU, two in
`stereo_match`.

### How the implementations are kept equal

Each stage's Nx code is the reference. The tests compile it with EXLA
and compare: every shader on each Vulkan device (`mix test --include
vulkan`), with a default that gives zeros, so only the shader can pass;
and every native stage, alone and chained, against the same stages on
Nx's evaluator. The native stages use the same `f32` operations in the
same order, so their integer outputs, rectified images included, and
most `f32` ones are identical; sums reduced in another order agree to
their rounding. EXLA's CPU code fuses some multiply-adds, so its
rectified pixels may differ by one grey level where a sample falls on a
rounding half.

## The matcher

`NxStereo.sgm/3` is the matching on its own:

```elixir
disparity = NxStereo.sgm(left, right, disparities: 64, p1: 8, p2: 64)
```

`left` and `right` are rectified `{height, width}` `u8` images (or `s32` ones
holding 8-bit pixels, for a GPU without 8-bit storage); the result is
an `f32` disparity map with `-1.0` where no reliable match was found. The
algorithm follows libSGM: a 9x7 Census transform that ignores one-level
differences, so that sensor noise on dark or flat surfaces does not count as
texture, Hamming costs, eight aggregation paths with the normalized
recurrence, winner selection with a uniqueness check and parabolic subpixel
refinement, OpenCV's left-right consistency check and a 3x3 median. For
faster frames, see [Speed or accuracy](#speed-or-accuracy).

`:disparities` is 64 or 128, and images are at most 4096 pixels wide. When
nearby objects shift by more than 128 pixels between the two images, run on
downscaled images.

### Coarse to fine

Searching 128 disparities at full resolution costs twice as much as 64.
`NxStereo.sgm_coarse_to_fine/3` gets most of the full search's matches for a
fraction of that:

```elixir
disparity = NxStereo.sgm_coarse_to_fine(left, right, disparities: 128, band: 16)
```

It matches the pair shrunk 2x over 64 disparities with `sgm/3`, then matches
the full-size pair with `NxStereo.sgm_band/4`, which searches each pixel
only within a band of `:band` disparities (8, 16, 32 or 64). A pixel's band
starts `:margin` disparities below twice the lowest coarse disparity in its
3x3 neighbourhood at the coarse scale, so a pixel on a depth edge covers
both sides. `:coarse_paths` sets the coarse matching's paths (3 or 8)
separately from the fine one's: the band only has to contain the right
disparity, so 3 coarse paths lose little. Images need an even height and
width.

`:fine_census` optionally changes only the fine descriptor. `:dense_9x5`
compares all 44 neighbours in a 9x5 window; `:dense_9x5_blend` also averages
the comparison centre with its rounded 3x3 Gaussian mean, rounding the
average. Both retain the one-level threshold. The blended option reduces
noise sensitivity in the tested scenes; coverage and individual scenes
still vary. The default remains `:sparse_9x7`, including in the coarse pass.
Direct `sgm_band/4` calls use the same choices under `:census`.

```elixir
disparity = NxStereo.sgm_coarse_to_fine(left, right,
  band: 32, margin: 8, coarse_paths: 8, paths: 8,
  fine_census: :dense_9x5_blend)
```

On the 2006 Middlebury pairs at half size (6 scenes), pixels both matchings
match agree with the full 128-disparity `sgm/3` within 1 pixel:

| Matching | Agreement with the full search |
| --- | --- |
| Coarse to fine, band 16 | 96.0–99.8% |
| Full search on the pair shrunk 2x | 86–99.4% |

A structure too thin to survive the shrink can take the background's
disparity.

`sgm_band/4` is usable on its own, with a band base from any earlier
estimate: an `s32` `{height, width}` tensor, `-1` where a pixel has no
estimate and should get no match.

## The functions under the stages

Each stage is a thin layer over Nx functions you can also call yourself,
in a `defn` of your own:

  * `NxStereo.Rectify`: undistortion and rectification, as OpenCV's
    `initUndistortRectifyMap` and `remap`, with the calibration as inputs.
  * `NxStereo.Filter`: a speckle filter, which drops matches their
    neighbours do not agree with, and a temporal filter, which smooths the
    disparity over successive pairs and holds lost matches.
  * `NxStereo.Depth`: metric depth and an organized point cloud from a
    disparity map.
  * `NxStereo.Diagnostics`: each camera's sharpness, the row offset
    between the rectified images, and the disparity's matched share,
    spread and jitter, to watch a rig's focus and calibration.
  * `NxStereo.Image`: area resizing, a 3x3 or 5x5 median and the turbo
    colour map.
  * `NxStereo.Packing`: 8-bit images as 32-bit words, four pixels each,
    for a GPU without 8-bit storage; `NxStereo.Rectify.rectify_packed/3`
    takes such frames.

Compiled for Vulkan, rectification, the filters, depth, the point cloud
and the diagnostics run as hand-written shaders too (see
[The stages around the matcher](#the-stages-around-the-matcher)).

## What runs where

Every stage and function is plain Nx. Compiled, how it runs depends on the
compiler; the native backend is the one way to run the stages outside Nx:

| Called through | Runs on | Needs |
| --- | --- | --- |
| Nx alone (`Nx.Defn.Evaluator`) | CPU, in Elixir; correct but far too slow for video | `nx` |
| `EXLA` | CPU, or CUDA GPUs, as XLA code | `exla` |
| `NxIREE.Compiler` for Vulkan | Vulkan GPUs, as the shaders in `shaders/spirv` | `nx_iree`, plus `exla` where it compiles |
| `NxStereo.NativeSegment` | CPU cores, in `stereo_match` | `cargo` where nx_stereo compiles |

Only `NxIREE.Compiler` uses the shaders; every other compiler or backend runs
the Nx implementation. All three give the same matches; subpixel values from
the shaders may differ in their last bit.

The libraries play these roles:

  * `nx_stereo`: the algorithms in Nx, their Vulkan shaders, and their
    native implementation.
  * `exla`: turns Nx code into StableHLO, which NxIREE compiles. Only
    needed on the machine that compiles.
  * `nx_iree`: a compiler, `iree-compile`, that builds GPU programs on the
    compiling machine, and a small runtime that runs them on the device.

## Installation

```elixir
def deps do
  [
    {:nx_stereo, github: "Spin42/nx_stereo"},
    # for Vulkan GPUs; NxStereo falls back to plain Nx without it
    {:nx_iree, github: "Spin42/nx_iree"},
    # only where functions are compiled
    {:exla, "~> 1.0"}
  ]
end
```

`nx_iree` is an optional dependency of `nx_stereo`, so it must be listed
explicitly. Its first build compiles the IREE runtime and needs `git`,
`cmake`, `ninja` and a C++17 compiler. NxIREE detects EXLA when it is
compiled: after adding `exla` to a project where `nx_iree` is already built,
run `mix deps.compile nx_iree --force`, or compiling for Vulkan fails.

Installing `nx_stereo` needs no shader compiler: the SPIR-V binaries in
`shaders/spirv` are committed (see [SPIR-V binaries](#spir-v-binaries)).

The native backend is built when nx_stereo compiles, if `cargo` is
installed: `stereo_match`, from `native/`, goes into the app's `priv`.
Without cargo, or with `NX_STEREO_NATIVE=false`, it is skipped and only
the Nx backends run. For a [Nerves](https://nerves-project.org) target it
is cross-built with the toolchain and sysroot Nerves sets while it
compiles the dependency, for the target's CPU: the target's Rust
standard library must be installed (`rustup target add
aarch64-unknown-linux-gnu` for a 64-bit Arm board).

## Running on a GPU

When one machine compiles and runs:

```elixir
sgm =
  Nx.Defn.jit(&NxStereo.sgm(&1, &2, disparities: 64),
    compiler: NxIREE.Compiler,
    iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv"],
    iree_runtime_options: [device: device]
  )

disparity = sgm.(left, right)
```

  * `device` is one of `NxIREE.list_devices("vulkan")`. Without it NxIREE
    uses the first GPU it finds, which may be an integrated one.
  * The first call for each image size compiles for a few seconds. For
    video, build `sgm` once per resolution and call it for every frame.
  * The result stays on the GPU. `Nx.backend_transfer/2` copies it to the
    CPU; passing it to another NxIREE call needs no copy.

The GPU needs Vulkan 1.1 with subgroup shuffles and a minimum subgroup size
of 8 for `sgm/3` over 64 disparities, 16 over 128, and `band / 4` for
`sgm_band/4`.
It needs no 8-bit storage buffers when the images are `s32`: every shader
reads and writes 32-bit words.

### Speed or accuracy

`NxStereo.sgm/3` aggregates along 8 paths by default. `paths: 3` keeps only
the horizontal and top-to-bottom paths, for about half the GPU time per
frame. In the `Nx.Defn.jit/2` call above, pass:

```elixir
&NxStereo.sgm(&1, &2, disparities: 64, paths: 3)
```

It leaves more pixels wrong or missing, mostly on dark or textureless
surfaces. On the [benchmark](#benchmark):

| Pixels wrong or missing | 8 paths (default) | 3 paths |
| --- | --- | --- |
| Middlebury | 12.4% | 13.0% |
| ETH3D | 8.7% | 11.4% |
| KITTI | 6.4% | 7.1% |

For 128 disparities, [coarse to fine](#coarse-to-fine) is the bigger saving;
`:paths` and `:coarse_paths` trade speed for accuracy there the same way.

## Running on a device without the compiler

A device can run the shaders with only `nx`, the `nx_iree` runtime and a
Vulkan driver. Compile on another machine with `NxIREE.export/3`, for the
image size the device will use:

```elixir
# compiling machine
template = Nx.template({480, 640}, :u8)

binary =
  NxIREE.export(&NxStereo.sgm(&1, &2, disparities: 64), [template, template],
    iree_compiler_flags: ["--iree-hal-target-backends=vulkan-spirv"]
  )

File.write!("sgm.bin", binary)

# device
sgm = NxIREE.load_exported(File.read!("sgm.bin"), device: device)
disparity = NxIREE.run(sgm, [left, right])
```

NxIREE's module cache works too: it keeps `NxStereo.sgm/3` calls unchanged on
the device, which then needs `nx_stereo` as well. See NxIREE's documentation
for both, and for cross-compiling with Nerves.

## How the shaders work

Each matching function wraps its Nx implementation in an `Nx.block`. When
`NxIREE.Compiler` compiles for Vulkan, the block becomes an
`NxIREE.Vulkan.Pipeline` (built in `NxStereo.Vulkan`): a sequence of compute
dispatches, each a SPIR-V shader from `shaders/spirv`, with the buffers between
them kept on the GPU. Every other compiler runs the block's Nx
implementation, which is also the reference the shaders are tested against.

The shaders share a few layouts:

  * **Census features.** A pixel's left and right 9x7 Census signatures sit
    side by side in one 128-bit element, stored in tiles of 8 consecutive
    rows per column, so a scan's step reads one contiguous run.
  * **Path volumes.** A path's cost for a pixel and disparity is one byte,
    four to a 32-bit word. The normalized recurrence keeps each value at or
    below the Census bound plus P2, 127 (which is why `:p2` is at most 79),
    so two paths' sum still fits in a byte: paired scans add their partner's
    volume to their own, and the selection reads half as many volumes.
  * **Band costs.** `sgm_band/4`'s Census distances are below 64, so a
    pixel's candidates are packed 6 bits each (3 words for a band of 16).

### `sgm/3`

| Stage | Shader | What it does |
| --- | --- | --- |
| Census | `census_pair.comp` | Both images' 9x7 Census transforms. A workgroup loads its window into shared memory; a thread computes four consecutive pixels of a row, reading each window row once for the four. A neighbour counts as darker only when it is more than one level below the centre. |
| Forward scan | `cost_first.comp` | The left-to-right path. Computes the Hamming matching costs from a sliding Census window in registers, stores them for the other scans, and stores its path. |
| Reverse scan | `cost_scan.comp` | The right-to-left path over the stored costs, written as the sum with the forward path. |
| Other scans (8 paths) | `cost_line.comp` | Bottom-to-top, top-to-bottom and the four diagonals, one line per lane group, in pairs whose sums go into one volume each. |
| Selection (8 paths) | `select.comp` | Sums the four paired volumes per pixel, picks the lowest-cost disparity, applies the uniqueness check and the parabolic subpixel refinement, and writes the winner's key for the consistency check. |
| Top-to-bottom scan and selection (3 paths) | `cost_line.comp` with `SELECT` | The last scan adds its path to the horizontal sum and selects there, without storing a volume. |
| Consistency | `consistency_projected.comp` | OpenCV SGBM's rule: each row's winners are projected into a pseudo right disparity map in shared memory, and a left match is dropped when the right map disagrees by more than one disparity. |
| Median | `filter.comp` | libSGM's 3x3 median selection network. |

For 128 disparities and three paths, `cost_first.comp` with `FUSED` performs
both horizontal scans in one dispatch and output volume. The other full-range
configurations retain separate forward and reverse dispatches.

The scans follow libSGM: a line's disparities are spread over 8 or 16 lanes
of a subgroup, 8 disparities each, and the recurrence's neighbours across
lanes arrive through subgroup shuffles. `cost_dp.glsl` holds the
recurrence, `cost_select.glsl` the selection.

### `sgm_band/4`

A pixel's candidates are the `:band` disparities from its base, and
neighbouring pixels' bands can differ. Bands of 8 or 16 stay in one
thread's registers; bands of 32 or 64 use one subgroup lane per four
candidates to keep the per-thread state small.

| Stage | Shader | What it does |
| --- | --- | --- |
| Census | `census_pair.comp`, `census_dense_pair.comp` | The default sparse window, or a dense 9x5 window with an optional blended centre. |
| Costs | `band_cost.comp` | Each pixel's matching costs for its band, once for every path, packed 6 bits a candidate. A workgroup covers 4 rows of 64 pixels and loads the right Census of every column they reach into shared memory once. Band 64 processes four signatures at a time to bound live state. |
| Transpose | `band_transpose.comp` | A column-major copy of the costs with each pixel's base appended, so a horizontal scan's lines are consecutive in memory, as a vertical scan's are. |
| Narrow scans | `band_scan.comp` | A thread per line, aligning previous costs with a packed-byte barrel shifter. A band of 16 skips alignment when the base stays the same. The horizontal traversal adds both directions in one output volume. |
| Wide scans | `band_scan_lanes.comp` | Four candidates per subgroup lane. A reduction finds the previous minimum and shuffles gather the shifted window; consecutive lanes write consecutive path words. Horizontal paths share one output volume. Bands of 32 skip realignment when the base is unchanged. |
| Wide aggregation | `band_scan_aggregate.comp` | Opposite vertical and diagonal paths share a dispatch. Their costs accumulate in 10 bits per candidate, retaining one aggregate instead of six path volumes. Four lanes share the extra high-byte word. |
| Selection | `band_select.comp` | `band / 4` lanes per pixel, each reading four aggregate costs or summing a word of the paths; the winner, the second best and the winner's neighbours are combined across those lanes with shuffles. Writes the same winner keys as `sgm/3`. |
| Consistency, median | `consistency_projected.comp`, `filter.comp` | As in `sgm/3`. |

`band_common.glsl` holds the band's layouts; `band_scan_geometry.glsl`
shares the line coordinates and folded diagonal restarts. `band_scan_lanes.glsl`
shares the wide-band recurrence, and `band_aggregate.glsl` decodes its packed
sums. Both scan implementations charge P2 for a missing previous candidate. The reverse
horizontal scan adds the forward path, and eight-path matching scans two
directions per dispatch. A negative base is searched from disparity 0 so
paths through it stay defined, but that pixel gets no match.

### `sgm_coarse_to_fine/3`

Between its two matchings run two small shaders: `shrink.comp` averages
each 2x2 block of both images after converting their pixels to `s32`,
and `band_base.comp` derives the full-size bases from the coarse
disparity, a thread per coarse pixel writing its 2x2 block of bases.

### The stages around the matcher

Each of these shaders replaces one function's Nx code, which stays the
default on every other compiler and below the sizes a shader takes:

| Function | Shader | What it does |
| --- | --- | --- |
| `Rectify.rectify/3`, `rectify_packed/3` | `rectify.comp` | Undistorts, rectifies and shrinks a frame read as 32-bit words, four pixels per thread, in the Nx code's order of operations, so a pixel differs only where a sample falls on a rounding half. |
| `Filter.despeckle/2` | `despeckle.comp` | A 64x16 tile and its 3-pixel border in shared memory; a thread counts the agreeing neighbours of four pixels of a row, reading each neighbourhood row once for the four. |
| `Filter.temporal/4` | `temporal_filter.comp` | Four pixels a thread, read and written as `vec4`s. |
| `Depth.depth/3` | `depth.comp` | Two pixels a thread: metres as a `vec2`, millimetres as two 16-bit halves of a word. |
| `Depth.point_cloud/5` | `point_cloud_rows.comp`, `column_sums.comp` | A workgroup per grid row unprojects its points and counts those kept; a second kernel adds the rows' counts. |
| `Diagnostics.*_sums` | `sharpness_rows.comp`, `row_alignment_rows.comp`, `disparity_rows.comp`, `column_sums.comp` | A workgroup per row reduces it to a few sums; row alignment uses one per sampled row and band. Counts and grey-level sums use exact 32-bit integers; `column_sums.comp` adds the rows. |

The rectification and the image diagnostics take `u8` images, read as
words, or `s32` ones, each a variant of the shader.

### Parameters

  * Image size, and for some shaders the disparity range or band margin, are
    specialization constants: a new image size needs no new SPIR-V.
  * The penalties P1 and P2 and the uniqueness ratio are a `{3}` `f32`
    input buffer (`params.glsl`), so they change between frames without a
    new pipeline.
  * Disparity range, band width, path count and scan direction are
    compile-time macros: each combination is its own binary, named after the
    stage and its disparity range (`_d64`, `_d128`) or band width (`_b8` to
    `_b64`). The census, consistency and median shaders serve them all.
    `NxStereo.Vulkan` and `NxStereo.Kernels` list every variant, which
    `mix nx_stereo.shaders` compiles.

### SPIR-V binaries

The binaries in `shaders/spirv` are required: `NxStereo.Vulkan` and
`NxStereo.Kernels` embed them when they compile, whether or not `nx_iree`
is installed, so nx_stereo does not compile without them and a release
does not carry them. They are committed so that installing nx_stereo
needs no shader compiler. After changing a shader in `shaders/`, regenerate them
with `glslangValidator` installed:

```shell
mix nx_stereo.shaders
```

Commit the sources and binaries together, and run `mix test --include vulkan`
to compare the shaders with the Nx implementation.

### Licences

The matching shaders are ported from
[Fixstars libSGM](https://github.com/fixstars/libSGM) (Apache-2.0), and
those files keep Fixstars' copyright line, as the Apache licence requires of
derived work. `consistency_projected.comp` follows OpenCV's StereoSGBM
consistency rules and keeps OpenCV's BSD-3-Clause notice (see
`OPENCV-LICENSE`). The rest of nx_stereo is Apache-2.0 (see `LICENSE`).

## The native backend

`stereo_match`, the Rust program in `native/stereo_match`, computes the
stages on every core of a CPU. `NxStereo.Matcher` runs it as a Port, a
separate process the BEAM talks to over its stdin and stdout, so a crash
in it cannot take the BEAM down; it reopens the Port when it exits, and
`:threads` sets how many cores it computes on (one per core by default).

A native segment is configured in the program on its first pair with
each stage's `native_kind/0`, its numeric build settings
(`native_settings/1`) and the constants it shares with its Nx code
(`native_constants/0`), such as the point cloud's depth range; a pair is
then one request with its entries and each stage's params, and one reply
with the outputs the Nx stages would have produced. The program keeps what
depends only on the settings, such as the rectification maps, and what
the stages carry from one pair to the next. It reads the next request and
writes the previous reply while it computes, so two pairs in flight keep
it busy. `native/stereo_match/src/protocol.rs` describes the messages.

## Tests

`mix test` checks the reference implementation on EXLA and Nx's evaluator,
and, when `stereo_match` was built, every native stage against it.
`mix test --include vulkan` also compares the shaders with it on every
Vulkan device (`NX_STEREO_DEVICE` selects devices by URI substring).
`cargo test` in `native/` runs the native program's own tests, among them
the matching against a step-by-step transcription of `NxStereo.sgm/3`.

## Benchmark

`mix run bench/pipeline.exs` times each stage of a depth pipeline on a
Vulkan device, on a synthetic textured pair: rectification, matching, the
filters, depth, the point cloud and the diagnostics. Each stage is
compiled alone, with its inputs on the device, and compared with EXLA's
result for the same inputs; the whole chain is also timed in one call, as
a pipeline compiles it, and so is a call that does nothing. Its options
set the size, the search and paths, and `--packed` passes the frames as
32-bit words, and `--kernels` lists, under each stage, the time of each of
its Vulkan kernels run alone. Run it after changing a shader to see what
the change is worth and that the results still match;
[Optimizing a stage](docs/optimizing.md) describes the whole method.

`mix run bench/stream.exs` measures the complete chain on a changing sequence
of CPU frames. It uploads every pair, propagates temporal state and downloads
all published outputs. Serial and overlapped modes must produce identical
outputs and carry before timing starts. The report separates sustained FPS,
completion intervals and frame latency, including counts above 33.33 ms.
Compilation, warm-up, camera capture and display are outside the measurement.

```shell
mix run bench/stream.exs --size 544x728 --band 32 --margin 8 --coarse-paths 8 --packed --frames 300 --rounds 3
```

`--mode serial|overlap|both` chooses how downloads run; `--sequence` controls
the number of distinct precomputed pairs. Both modes use the same sequence,
and their order alternates between rounds. `--warmup` sets warm-up frames.
Both `pipeline.exs` and `stream.exs` accept `--cloud-decimation` (default 4).
A value of 1 publishes a point at every depth pixel, producing 16 times as
many grid points as the default. The reported grid dimensions and complete
chain timings include that output density and its download.

`mix run bench/datasets.exs` caches the Middlebury MiddEval3 training scenes
in `_build/bench` and reports coverage, matched pixels more than 2 disparities
off, bad-or-missing pixels and mean absolute disparity error. On Vulkan it
also compares the output with EXLA and records median and p90 matcher times
after warmup, with resident inputs and the output downloaded each time.

```shell
mix run bench/datasets.exs --datasets middlebury,eth3d,kitti --search coarse_to_fine --paths 8 --coarse-paths 3 --band 16 --margin 4 --runs 20 --csv quality.csv
```

ETH3D needs `7z`; KITTI 2015 downloads 1.7 GB. `--device` selects a Vulkan
device by URI substring, `--scene` selects scene names containing a string,
and `--no-reference` skips the EXLA comparison. `--p1`, `--p2` and
`--uniqueness` set the penalties and rejection threshold. An explicit
`--disparities 64` or `128` scores only supported ground truth and reports
its share; coarse-to-fine always uses 128. Full search without that option
keeps automatic range selection and skips scenes beyond 127 disparities.
Dataset summaries weight scenes equally. Matcher times do not include the
other camera pipeline stages. All three benchmarks accept
`--fine-census sparse_9x7|dense_9x5|dense_9x5_blend` with coarse-to-fine
search, and CSV records the selected descriptor.

`--perturbation clean|noise2|noise4|noise8|gain0.8|gain1.2|blur_right`
selects reproducible image conditions, recorded in CSV. Noise is independent
between eyes, with Gaussian standard deviation 2, 4 or 8 grey levels and
fixed left/right seeds 271/723. Exposure changes only the right image: gain
0.8 adds 8, gain 1.2 subtracts 8. Blur applies a 3x3 box filter to the right
image with zero padding. Intensities are rounded and clipped to bytes;
ground truth is unchanged. These transforms run before compilation and timing.

## Examples

![Left image, ground truth and NxStereo disparity for four Middlebury scenes](docs/images/middlebury-examples.webp)

The ground truth was measured with structured light; NxStereo
(`NxStereo.sgm/3`, 64 disparities, default options) only sees the two
photos, so it loses detail at object edges, in structures thinner than its
9x7 window and on blank or repetitive surfaces, and leaves the left border
and occluded pixels without a value.
The scenes are Teddy and Cones from the Middlebury 2003 datasets, and
Adirondack and Motorcycle from the 2014 datasets at quarter resolution:

| Scene | Pixels matched | Matched pixels off by more than 2 px |
| --- | --- | --- |
| Teddy | 91% | 7.7% |
| Cones | 91% | 5.7% |
| Adirondack | 94% | 5.7% |
| Motorcycle | 94% | 7.9% |

The images come from the [Middlebury stereo datasets](https://vision.middlebury.edu/stereo/data/):
D. Scharstein and R. Szeliski, *High-accuracy stereo depth maps using
structured light*, CVPR 2003; D. Scharstein et al., *High-resolution stereo
datasets with subpixel-accurate ground truth*, GCPR 2014.
