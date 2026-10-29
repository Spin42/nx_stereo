defmodule NxStereo.ChainTest do
  use ExUnit.Case, async: true

  alias NxStereo.{Depth, Image, Rectify}

  @moduletag :tmp_dir

  # The OKdo Nano C100's Vulkan features (C100GPU.Target).
  @c100_features ~w(spirv:v1.3 cap:Shader cap:StorageBuffer8BitAccess cap:Int8 ext:SPV_KHR_8bit_storage
                    cap:GroupNonUniform cap:GroupNonUniformArithmetic cap:GroupNonUniformShuffle
                    cap:GroupNonUniformBallot cap:Int64 cap:Int16 cap:Float16 cap:StorageBuffer16BitAccess
                    ext:SPV_KHR_16bit_storage)

  defp chain(left, right, left_camera, right_camera, focal_length_px, baseline_m) do
    left = Rectify.rectify(left, left_camera)
    right = Rectify.rectify(right, right_camera)
    {left, right, match(left, right, focal_length_px, baseline_m)}
  end

  defp match(left, right, focal_length_px, baseline_m) do
    disparity = left |> NxStereo.sgm(right, disparities: 64) |> Image.median(5)
    {depth_mm, depth_m} = Depth.depth(disparity, focal_length_px, baseline_m)
    {disparity, depth_mm, Depth.cloud_grid(depth_m, focal_length_px, 96.0, 48.0, 4)}
  end

  defp textured_pair(height, width, shift) do
    {texture, _} =
      Nx.Random.randint(Nx.Random.key(7), 0, 256, shape: {height, width + shift}, type: :s32)

    texture = Nx.as_type(texture, :u8)
    {texture[[.., 0..(width - 1)]], texture[[.., shift..(width + shift - 1)]]}
  end

  @tag :vulkan
  test "rectify, match, filter, depth and cloud compile into one module and match the reference",
       %{tmp_dir: dir} do
    {:ok, devices} = NxIREE.list_devices("vulkan")
    device = Enum.find(devices, &String.contains?(&1.uri, System.get_env("NX_STEREO_DEVICE", "")))

    {left, right} = textured_pair(96, 192, 8)
    k = [200.0, 0.0, 96.0, 0.0, 200.0, 48.0, 0.0, 0.0, 1.0]
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    p = fn tx -> [200.0, 0.0, 96.0, tx, 0.0, 200.0, 48.0, 0.0, 0.0, 0.0, 1.0, 0.0] end
    left_camera = Rectify.camera(k, [0.01], identity, p.(0.0))
    right_camera = Rectify.camera(k, [0.01], identity, p.(-12.0))
    args = [left, right, left_camera, right_camera, Nx.f32(200.0), Nx.f32(0.06)]

    gpu =
      apply(
        Nx.Defn.jit(&chain/6,
          compiler: NxIREE.Compiler,
          iree_compiler_flags: [
            "--iree-hal-target-backends=vulkan-spirv",
            "--iree-vulkan-target-features=" <> Enum.join(@c100_features, ",")
          ],
          iree_module_cache: dir,
          iree_runtime_options: [device: device]
        ),
        args
      )

    assert [_module] = Path.wildcard(Path.join(dir, "*.nxiree"))

    {rectified_left, rectified_right, {disparity, depth_mm, cloud}} =
      Nx.backend_transfer(gpu, Nx.BinaryBackend)

    {reference_left, reference_right, _} = apply(Nx.Defn.jit(&chain/6, compiler: EXLA), args)

    # Rounding at half a grey level may differ on the GPU, and the
    # matcher's Census bits amplify any input difference: the rest of the
    # chain is checked on the GPU's own rectified images.
    assert max_difference(rectified_left, reference_left) <= 1
    assert max_difference(rectified_right, reference_right) <= 1

    {reference_disparity, reference_depth_mm, reference_cloud} =
      Nx.Defn.jit(&match/4, compiler: EXLA).(
        rectified_left,
        rectified_right,
        Nx.f32(200.0),
        Nx.f32(0.06)
      )

    # Subpixel values may differ in their last bits on the GPU.
    assert max_difference(disparity, reference_disparity) <= 1 / 16
    assert matched_share(disparity) > 0.5
    assert_in_delta median_matched(disparity), 8.0, 0.25
    assert max_difference(depth_mm, reference_depth_mm) <= 1
    assert max_relative_difference(cloud, reference_cloud) <= 1.0e-3
  end

  defp max_difference(a, b) do
    Nx.subtract(Nx.as_type(a, :f32), Nx.as_type(b, :f32))
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
  end

  defp max_relative_difference(a, b) do
    a
    |> Nx.subtract(b)
    |> Nx.abs()
    |> Nx.divide(Nx.add(Nx.abs(b), 1.0))
    |> Nx.reduce_max()
    |> Nx.to_number()
  end

  defp median_matched(disparity) do
    matched = disparity |> Nx.to_flat_list() |> Enum.filter(&(&1 >= 0)) |> Enum.sort()
    Enum.at(matched, div(length(matched), 2))
  end

  defp matched_share(disparity),
    do: disparity |> Nx.greater_equal(0) |> Nx.mean() |> Nx.to_number()

  # The Raspberry Pi 5's path: the Nx code itself, compiled for a CPU.
  @tag :iree
  @tag timeout: 600_000
  test "compiled for the CPU, the chain matches the reference" do
    {:ok, [device | _]} = NxIREE.list_devices("local-sync")
    {left, right} = textured_pair(96, 192, 8)
    k = [200.0, 0.0, 96.0, 0.0, 200.0, 48.0, 0.0, 0.0, 1.0]
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    p = fn tx -> [200.0, 0.0, 96.0, tx, 0.0, 200.0, 48.0, 0.0, 0.0, 0.0, 1.0, 0.0] end

    args = [
      left,
      right,
      Rectify.camera(k, [], identity, p.(0.0)),
      Rectify.camera(k, [], identity, p.(-12.0)),
      Nx.f32(200.0),
      Nx.f32(0.06)
    ]

    {cpu_left, _, {cpu_disparity, cpu_depth_mm, _}} =
      apply(
        Nx.Defn.jit(&chain/6, compiler: NxIREE.Compiler, iree_runtime_options: [device: device]),
        args
      )
      |> Nx.backend_transfer(Nx.BinaryBackend)

    {reference_left, _, {reference_disparity, reference_depth_mm, _}} =
      apply(Nx.Defn.jit(&chain/6, compiler: EXLA), args)

    assert max_difference(cpu_left, reference_left) <= 1
    assert max_difference(cpu_disparity, reference_disparity) <= 1 / 16
    assert max_difference(cpu_depth_mm, reference_depth_mm) <= 1
  end
end
