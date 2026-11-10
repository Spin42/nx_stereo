defmodule NxStereo.KernelsTest do
  # Each kernel against its function's Nx code on EXLA. The kernels run with a
  # default that gives zeros, so only the kernel can give the reference.
  use ExUnit.Case, async: true

  alias NxStereo.{Depth, Diagnostics, Filter, Kernels, Packing, Rectify}

  @moduletag :vulkan

  setup_all do
    {:ok, devices} = NxIREE.list_devices("vulkan")

    devices =
      Enum.filter(devices, &String.contains?(&1.uri, System.get_env("NX_STEREO_DEVICE", "")))

    assert devices != []
    {:ok, devices: devices}
  end

  defp on_gpus(devices, fun, args) do
    for device <- devices do
      result =
        fun
        |> Nx.Defn.jit(compiler: NxIREE.Compiler, iree_runtime_options: [device: device])
        |> apply(args)
        |> Nx.backend_transfer(Nx.BinaryBackend)

      {device.uri, result}
    end
  end

  defp reference(fun, args), do: fun |> Nx.Defn.jit(compiler: EXLA) |> apply(args)

  defp zeros(shape, type), do: Nx.broadcast(Nx.tensor(0, type: type), shape)

  defp texture(height, width, seed) do
    {image, _key} =
      Nx.Random.randint(Nx.Random.key(seed), 0, 256, shape: {height, width}, type: :s32)

    Nx.as_type(image, :u8)
  end

  # A disparity of slanted surfaces, with holes and isolated wrong pixels.
  defp disparity(height, width, seed) do
    key = Nx.Random.key(seed)
    {noise, key} = Nx.Random.uniform(key, shape: {height, width})
    {wrong, _key} = Nx.Random.uniform(key, 0.0, 128.0, shape: {height, width})
    surface = Nx.add(Nx.iota({height, width}, axis: 1) |> Nx.divide(width / 30), 4.0)
    surface = Nx.select(Nx.less(noise, 0.05), wrong, surface)
    Nx.select(Nx.greater(noise, 0.9), -1.0, surface) |> Nx.as_type(:f32)
  end

  # The share of the elements of two tensors within `tolerance` of each other, NaN matching NaN.
  defp close(got, want, tolerance, relative? \\ false) do
    {got, want} =
      {Nx.as_type(got, :f32), Nx.as_type(Nx.backend_copy(want, Nx.BinaryBackend), :f32)}

    nan = Nx.logical_and(Nx.is_nan(got), Nx.is_nan(want))
    bound = if relative?, do: Nx.multiply(Nx.abs(want), tolerance), else: tolerance
    near = Nx.less_equal(Nx.abs(Nx.subtract(got, want)), bound)
    nan |> Nx.logical_or(near) |> Nx.mean() |> Nx.to_number()
  end

  @k [436.0, 0.0, 364.0, 0.0, 436.0, 272.0, 0.0, 0.0, 1.0]
  @r [0.9998, -0.0175, 0.0, 0.0175, 0.9998, 0.0, 0.0, 0.0, 1.0]
  @p [218.0, 0.0, 182.0, 0.0, 0.0, 218.0, 136.0, 0.0, 0.0, 0.0, 1.0, 0.0]

  test "rectify's kernel samples as its Nx code does", %{devices: devices} do
    frame = texture(544, 728, 1)
    camera = Rectify.camera(@k, [-0.1, 0.02], @r, @p)
    want = reference(&Rectify.rectify(&1, &2, size: {272, 364}), [frame, camera])
    vector = Nx.concatenate([Nx.flatten(camera.k), camera.d, Nx.flatten(camera.inverse_pr)])

    fun = fn frame, vector ->
      Kernels.rectify(frame, vector, {272, 364}, fn _ -> zeros({272, 364}, :u8) end)
    end

    # A sample on a rounding half may round the other way on a GPU.
    for {uri, got} <- on_gpus(devices, fun, [frame, vector]),
        do: assert(close(got, want, 1) >= 0.999, uri)
  end

  test "rectify_packed's kernel gives the image and its packing", %{devices: devices} do
    frame = texture(544, 728, 2)
    camera = Rectify.camera(@k, [-0.1, 0.02], @r, @p)
    packed = Packing.pack(Nx.to_binary(frame), 728, 544)
    {want, _} = reference(&Rectify.rectify_packed(&1, &2, size: {272, 364}), [packed, camera])
    vector = Nx.concatenate([Nx.flatten(camera.k), camera.d, Nx.flatten(camera.inverse_pr)])

    fun = fn frame, vector ->
      Kernels.rectify(frame, vector, {272, 364}, fn _ ->
        {zeros({272, 364}, :s32), zeros({272, 91}, :u32)}
      end)
    end

    for {uri, {image, words}} <- on_gpus(devices, fun, [packed, vector]) do
      assert close(image, want, 1) >= 0.999, uri
      assert Nx.to_list(Packing.unpack_download(words)) == Nx.to_list(Nx.as_type(image, :u8))
    end
  end

  for type <- [:u8, :s32] do
    test "the sharpness kernel sums #{type} images as its Nx code does", %{devices: devices} do
      left = texture(272, 364, 3) |> Nx.put_slice([0, 0], Nx.broadcast(Nx.u8(0), {272, 10}))
      right = texture(272, 364, 4) |> Nx.max(1)
      [left, right] = Enum.map([left, right], &Nx.as_type(&1, unquote(type)))
      want = reference(&Diagnostics.sharpness_sums/2, [left, right])
      fun = &Kernels.sharpness(&1, &2, fn _ -> zeros({6}, :f32) end)

      for {uri, got} <- on_gpus(devices, fun, [left, right]),
          do: assert(close(got, want, 1.0e-5, true) == 1.0, uri)
    end

    test "the row alignment kernel sums #{type} images as its Nx code does", %{
      devices: devices
    } do
      right = texture(272, 364, 6)
      left = right |> Nx.pad(0, [{1, 0, 0}, {0, 4, 0}]) |> Nx.slice([0, 4], [272, 364])
      [left, right] = Enum.map([left, right], &Nx.as_type(&1, unquote(type)))
      disparity = disparity(272, 364, 7)
      want = reference(&Diagnostics.row_alignment_sums/3, [left, right, disparity])
      fun = &Kernels.row_alignment(&1, &2, &3, fn _ -> zeros({4, 6}, :f32) end)

      for {uri, got} <- on_gpus(devices, fun, [left, right, disparity]),
          do: assert(close(got, want, 0) == 1.0, uri)
    end
  end

  test "the speckle kernel keeps the pixels its Nx code keeps", %{devices: devices} do
    disparity = disparity(272, 364, 5)
    want = reference(&Filter.despeckle(&1, min_support: 12, max_diff: 1.0), [disparity])
    params = Kernels.scalars([12, 1.0])
    fun = &Kernels.despeckle(&1, &2, fn _ -> zeros({272, 364}, :f32) end)

    for {uri, got} <- on_gpus(devices, fun, [disparity, params]),
        do: assert(close(got, want, 0) == 1.0, uri)
  end

  test "the temporal kernel filters and counts misses as its Nx code does", %{devices: devices} do
    current = disparity(272, 364, 8)
    {jitter, _key} = Nx.Random.uniform(Nx.Random.key(9), -3.0, 3.0, shape: {272, 364})
    previous = Nx.select(Nx.greater(jitter, 2.5), -1.0, Nx.add(disparity(272, 364, 10), jitter))
    {missed, _key} = Nx.Random.randint(Nx.Random.key(11), 0, 4, shape: {272, 364}, type: :s32)
    args = [current, previous, Nx.as_type(missed, :f32)]
    settings = [Nx.f32(0.4), Nx.f32(2.0), Nx.f32(2)]

    # Op by op on the binary backend, as the kernel computes it: EXLA's vectorised loops fuse
    # the blend into a multiply-add.
    [current, previous, missed, alpha, delta, hold] =
      Enum.map(args ++ settings, &Nx.backend_copy(&1, Nx.BinaryBackend))

    # A block's Nx code, called with tensors, runs on the default compiler.
    defaults = Nx.Defn.default_options(compiler: Nx.Defn.Evaluator)

    {want_filtered, want_missed} =
      Filter.temporal(current, previous, missed, alpha: alpha, delta: delta, hold: hold)

    Nx.Defn.default_options(defaults)

    fun =
      &Kernels.temporal_filter(&1, &2, &3, &4, fn _ ->
        {zeros({272, 364}, :f32), zeros({272, 364}, :f32)}
      end)

    for {uri, {filtered, missed}} <-
          on_gpus(devices, fun, args ++ [Kernels.scalars(settings)]) do
      assert close(filtered, want_filtered, 0) == 1.0, uri
      assert close(missed, want_missed, 0) == 1.0, uri
    end
  end

  test "the depth kernel measures f·B/d as its Nx code does", %{devices: devices} do
    # Down to 0.001 px, whose depth is beyond 16-bit millimetres.
    disparity = disparity(272, 364, 12) |> Nx.put_slice([0, 0], Nx.tensor([[0.001, 0.0]]))
    {want_mm, want_m} = reference(&Depth.depth/3, [disparity, 436.5, 0.2])
    fun = &Kernels.depth(&1, &2, fn _ -> {zeros({272, 364}, :u16), zeros({272, 364}, :f32)} end)

    for {uri, {mm, m}} <- on_gpus(devices, fun, [disparity, Kernels.scalars([436.5, 0.2])]) do
      assert close(m, want_m, 1.0e-6, true) == 1.0, uri
      # A GPU's division need not be correctly rounded: a millimetre may round the other way.
      assert close(mm, want_mm, 1) == 1.0, uri
      assert close(mm, want_mm, 0) > 0.999, uri
      assert Nx.to_number(mm[0][0]) == 0, uri
    end
  end

  test "the point cloud kernel unprojects and counts as its Nx code does", %{devices: devices} do
    depth_m = disparity(272, 364, 13) |> Nx.divide(5) |> Nx.max(0.0)
    opts = [decimation: 4, min_depth_m: 0.05, max_depth_m: 20.0]

    {want_grid, want_kept} =
      reference(&Depth.point_cloud(&1, &2, &3, &4, opts), [depth_m, 436.5, 181.5, 135.5])

    params = Kernels.scalars([436.5, 181.5, 135.5, 0.05, 20.0])

    fun =
      &Kernels.point_cloud(&1, &2, 4, fn _ -> {zeros({68, 91, 3}, :f32), zeros({}, :f32)} end)

    for {uri, {grid, kept}} <- on_gpus(devices, fun, [depth_m, params]) do
      assert close(grid, want_grid, 1.0e-6, true) == 1.0, uri
      assert Nx.to_number(kept) == Nx.to_number(want_kept), uri
    end
  end

  test "the disparity sums kernel sums as its Nx code does", %{devices: devices} do
    current = disparity(272, 364, 14)
    previous = disparity(272, 364, 15)
    want = reference(&Diagnostics.disparity_sums/2, [current, previous])
    fun = &Kernels.disparity_sums(&1, &2, fn _ -> zeros({5}, :f32) end)

    for {uri, got} <- on_gpus(devices, fun, [current, previous]),
        do: assert(close(got, want, 1.0e-4, true) == 1.0, uri)
  end

  test "sizes the kernels do not take run the Nx code", %{devices: devices} do
    camera = Rectify.camera(@k, [-0.1], @r, @p)
    frame = texture(544, 728, 16)
    odd = disparity(3, 5, 17)

    fun = fn frame, camera, odd ->
      {Rectify.rectify(frame, camera, size: {272, 362}), Depth.depth(odd, 400.0, 0.2),
       Filter.temporal(odd, odd, Nx.broadcast(0.0, {3, 5}))}
    end

    {want_image, want_depth, want_filtered} = reference(fun, [frame, camera, odd])

    for {uri, {image, depth, filtered}} <- on_gpus(devices, fun, [frame, camera, odd]) do
      assert close(image, want_image, 1) >= 0.999, uri
      assert close(elem(depth, 1), elem(want_depth, 1), 1.0e-6, true) == 1.0, uri
      assert close(elem(filtered, 0), elem(want_filtered, 0), 1.0e-6, true) == 1.0, uri
    end
  end
end
