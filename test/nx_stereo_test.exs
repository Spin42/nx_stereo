defmodule NxStereoTest do
  use ExUnit.Case, async: true

  # A random texture and the same texture seen `shift` pixels further left
  defp pair(height, width, shift, seed \\ 1) do
    :rand.seed(:exsss, {seed, seed, seed})
    texture = for _ <- 1..(height * (width + shift)), into: <<>>, do: <<:rand.uniform(256) - 1>>
    scene = Nx.from_binary(texture, :u8) |> Nx.reshape({height, width + shift})

    {Nx.slice_along_axis(scene, 0, width, axis: 1),
     Nx.slice_along_axis(scene, shift, width, axis: 1)}
  end

  # A smooth background at disparity `back` behind a band of columns at
  # `front`. Fractional disparities next to occlusions exercise both integer
  # neighbours of the left-right consistency check.
  defp layered(height, width, back, front, seed \\ 1) do
    :rand.seed(:exsss, {seed, seed, seed})
    background = smooth_texture(height, width + 40)
    band = smooth_texture(height, div(width, 3) + 40)
    x = div(width, 3)
    left = Nx.put_slice(view(background, width, 0), [0, x], view(band, div(width, 3), 0))

    right =
      Nx.put_slice(
        view(background, width, back),
        [0, x - trunc(front)],
        view(band, div(width, 3), front - trunc(front))
      )

    {left |> Nx.round() |> Nx.as_type(:u8), right |> Nx.round() |> Nx.as_type(:u8)}
  end

  defp smooth_texture(height, width) do
    noise = for _ <- 1..(height * width), do: :rand.uniform() * 255.0
    kernel = Nx.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) |> Nx.divide(16)

    Nx.tensor(noise)
    |> Nx.reshape({1, 1, height, width})
    |> Nx.conv(Nx.reshape(kernel, {1, 1, 3, 3}), padding: :same)
    |> Nx.reshape({height, width})
  end

  # texture[y, x + shift], linearly interpolated
  defp view(texture, width, shift) do
    i = trunc(shift)
    f = shift - i
    a = Nx.slice_along_axis(texture, i, width, axis: 1)
    b = Nx.slice_along_axis(texture, i + 1, width, axis: 1)
    Nx.add(Nx.multiply(a, 1 - f), Nx.multiply(b, f))
  end

  defp interior(disparity, margin_x) do
    {height, width} = disparity.shape
    disparity |> Nx.slice([4, margin_x], [height - 8, width - margin_x - 4]) |> Nx.to_flat_list()
  end

  test "recovers the disparity of a shifted texture" do
    {left, right} = pair(24, 96, 5)
    disparity = NxStereo.sgm(left, right)

    assert disparity.shape == {24, 96}
    assert disparity.type == {:f, 32}
    values = interior(disparity, 70)
    assert Enum.all?(values, &(abs(&1 - 5.0) < 0.1)), inspect(Enum.frequencies(values))
  end

  test "marks pixels without a reliable match" do
    # Unrelated images have no zero-cost matches, which pass any uniqueness
    {left, _} = pair(16, 80, 0, 1)
    {_, right} = pair(16, 80, 0, 2)
    values = NxStereo.sgm(left, right, uniqueness: 1000.0) |> Nx.to_flat_list()

    assert Enum.uniq(values) == [-1.0]
  end

  test "evaluates the same on every compiler" do
    {left, right} = pair(16, 80, 3)
    fun = &NxStereo.sgm(&1, &2, p1: 10, p2: 60)
    evaluated = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(left, right)
    jitted = Nx.Defn.jit(fun, compiler: EXLA).(left, right)

    assert Nx.to_flat_list(jitted) == Nx.to_flat_list(evaluated)
  end

  test "takes the penalties as tensors, changed without compiling again" do
    {left, right} = layered(24, 96, 4.5, 14.25)

    compiled =
      Nx.Defn.compile(
        &NxStereo.sgm(&1, &2, p1: &3, p2: &4, uniqueness: &5),
        [left, right, Nx.template({}, :f32), Nx.template({}, :f32), Nx.template({}, :f32)],
        compiler: EXLA
      )

    results =
      for {p1, p2, uniqueness} <- [{10, 60, 1.05}, {3, 20, 1.3}] do
        expected = NxStereo.sgm(left, right, p1: p1, p2: p2, uniqueness: uniqueness)
        actual = compiled.(left, right, Nx.f32(p1), Nx.f32(p2), Nx.f32(uniqueness))
        assert Nx.to_flat_list(actual) == Nx.to_flat_list(expected)
        Nx.to_flat_list(actual)
      end

    assert Enum.uniq(results) |> length() == 2
  end

  test "matches 32-bit images as their 8-bit pixels" do
    {left, right} = pair(16, 80, 3)
    fun = &NxStereo.sgm(&1, &2, p1: 10, p2: 60)

    wide = Nx.Defn.jit(fun).(Nx.as_type(left, :s32), Nx.as_type(right, :s32))
    assert Nx.to_flat_list(wide) == Nx.to_flat_list(Nx.Defn.jit(fun).(left, right))
  end

  describe "sgm_coarse_to_fine/3" do
    test "matches a shifted texture as sgm/3 does at full resolution" do
      {left, right} = pair(32, 240, 37)
      disparity = Nx.Defn.jit(&NxStereo.sgm_coarse_to_fine(&1, &2, [])).(left, right)
      full = Nx.Defn.jit(&NxStereo.sgm(&1, &2, disparities: 128)).(left, right)

      assert disparity.shape == {32, 240}
      assert interior(disparity, 140) == interior(full, 140)
      assert Enum.count(interior(disparity, 140), &(abs(&1 - 37.0) < 0.1)) >= 2300
    end

    test "agrees with sgm/3 at full resolution across depth edges" do
      {left, right} = layered(48, 240, 20.5, 60.25)
      opts = [disparities: 128, p1: 7, p2: 70, uniqueness: 1.05]
      full = Nx.Defn.jit(&NxStereo.sgm(&1, &2, opts)).(left, right) |> Nx.to_flat_list()

      fine =
        Nx.Defn.jit(&NxStereo.sgm_coarse_to_fine(&1, &2, opts)).(left, right) |> Nx.to_flat_list()

      both = Enum.zip(fine, full) |> Enum.filter(fn {a, b} -> a >= 0 and b >= 0 end)
      assert length(both) >= 0.9 * Enum.count(full, &(&1 >= 0))
      assert Enum.count(both, fn {a, b} -> abs(a - b) <= 1 end) >= 0.95 * length(both)
    end

    test "validates its inputs" do
      {left, right} = pair(16, 80, 3)

      assert_raise ArgumentError, ~r/even height and width/, fn ->
        NxStereo.sgm_coarse_to_fine(left[0..14], right[0..14])
      end

      assert_raise ArgumentError, ~r/:disparities must be 128/, fn ->
        NxStereo.sgm_coarse_to_fine(left, right, disparities: 64)
      end
    end
  end

  describe "sgm_band/4" do
    for paths <- [3, 8] do
      test "a band covering every disparity matches as sgm/3 with #{paths} paths" do
        {left, right} = layered(24, 160, 4.5, 14.25)
        opts = [disparities: 64, p1: 7, p2: 70, uniqueness: 1.05, paths: unquote(paths)]
        base = Nx.broadcast(Nx.s32(0), {24, 160})

        band =
          Nx.Defn.jit(&NxStereo.sgm_band(&1, &2, &3, Keyword.put(opts, :band, 64))).(
            left,
            right,
            base
          )

        full = Nx.Defn.jit(&NxStereo.sgm(&1, &2, opts)).(left, right)
        assert Nx.to_flat_list(band) == Nx.to_flat_list(full)
      end
    end

    test "a band around each pixel's disparity finds the full range's matches" do
      {left, right} = layered(24, 160, 4.5, 14.25)
      opts = [disparities: 64, p1: 7, p2: 70, uniqueness: 1.05]
      full = Nx.Defn.jit(&NxStereo.sgm(&1, &2, opts)).(left, right)
      # Bands of 16 from 4 below each pixel's disparity: they differ across the depth edges.
      base =
        full |> Nx.max(0) |> Nx.floor() |> Nx.as_type(:s32) |> Nx.subtract(4) |> Nx.clip(0, 48)

      band =
        Nx.Defn.jit(&NxStereo.sgm_band(&1, &2, &3, Keyword.put(opts, :band, 16))).(
          left,
          right,
          base
        )

      pairs = Enum.zip(Nx.to_flat_list(band), Nx.to_flat_list(full))
      both = Enum.filter(pairs, fn {a, b} -> a >= 0 and b >= 0 end)
      agree = Enum.count(both, fn {a, b} -> abs(a - b) <= 0.5 end)

      assert length(both) >= 0.9 * Enum.count(pairs, fn {_a, b} -> b >= 0 end)
      assert agree >= 0.99 * length(both)
    end

    test "gives no match where the base is negative" do
      {left, right} = pair(16, 80, 3)

      base =
        Nx.broadcast(Nx.s32(0), {16, 80})
        |> Nx.put_slice([0, 40], Nx.broadcast(Nx.s32(-1), {16, 20}))

      opts = [band: 16, p1: 10, p2: 60]

      band = Nx.Defn.jit(&NxStereo.sgm_band(&1, &2, &3, opts)).(left, right, base)

      assert band[[.., 40..59]] |> Nx.to_flat_list() |> Enum.uniq() == [-1.0]
      assert band[[4..11, 10..30]] |> Nx.greater(0) |> Nx.all() |> Nx.to_number() == 1
    end

    test "validates the band" do
      {left, right} = pair(16, 80, 3)
      base = Nx.broadcast(Nx.s32(0), {16, 80})

      assert_raise ArgumentError, ~r/:band must be 8, 16, 32 or 64/, fn ->
        NxStereo.sgm_band(left, right, base, band: 12)
      end

      assert_raise ArgumentError, ~r/:band must be 8, 16, 32 or 64/, fn ->
        NxStereo.sgm_band(left, right, base, band: 128, disparities: 128)
      end

      assert_raise ArgumentError, ~r/base.*s32.*\{16, 80\}/, fn ->
        NxStereo.sgm_band(left, right, Nx.broadcast(Nx.s32(0), {16, 79}), band: 16)
      end
    end
  end

  test "validates its inputs" do
    {left, right} = pair(16, 80, 0)

    assert_raise ArgumentError, ~r/expects u8 or s32 images/, fn ->
      NxStereo.sgm(Nx.as_type(left, :f32), right)
    end

    assert_raise ArgumentError, ~r/same \{height, width\} shape/, fn ->
      NxStereo.sgm(left, left[0..9])
    end

    assert_raise ArgumentError, ~r/:disparities must be 64 or 128/, fn ->
      NxStereo.sgm(left, right, disparities: 32)
    end

    assert_raise ArgumentError, ~r/0 <= :p1 < :p2 <= 79/, fn ->
      NxStereo.sgm(left, right, p2: 100)
    end

    assert_raise ArgumentError, ~r/:uniqueness/, fn ->
      NxStereo.sgm(left, right, uniqueness: 0.5)
    end

    assert_raise ArgumentError, ~r/:paths must be 3 or 8/, fn ->
      NxStereo.sgm(left, right, paths: 4)
    end
  end

  describe "on Vulkan through NxIREE" do
    @describetag :vulkan

    setup do
      {:ok, devices} = NxIREE.list_devices("vulkan")

      devices =
        Enum.filter(devices, &String.contains?(&1.uri, System.get_env("NX_STEREO_DEVICE", "")))

      assert devices != []
      {:ok, devices: devices}
    end

    for {height, width, disparities} <- [{24, 160, 64}, {37, 203, 128}],
        scene <- [:shifted, :layered],
        paths <- [3, 8] do
      test "the shaders match the reference with #{paths} paths on a #{scene} scene at #{height}x#{width}, #{disparities} disparities",
           %{devices: devices} do
        {left, right} =
          case unquote(scene) do
            :shifted -> pair(unquote(height), unquote(width), 7)
            :layered -> layered(unquote(height), unquote(width), 4.5, 14.25)
          end

        opts = [
          disparities: unquote(disparities),
          p1: 7,
          p2: 70,
          uniqueness: 1.05,
          paths: unquote(paths)
        ]

        reference =
          Nx.Defn.jit(&NxStereo.sgm(&1, &2, opts), compiler: EXLA).(left, right)
          |> Nx.to_flat_list()

        for device <- devices do
          gpu =
            Nx.Defn.jit(&NxStereo.sgm(&1, &2, opts),
              compiler: NxIREE.Compiler,
              iree_runtime_options: [device: device]
            ).(left, right)
            |> Nx.backend_transfer(Nx.BinaryBackend)
            |> Nx.to_flat_list()

          # GLSL's f32 division need not be correctly rounded, so subpixel
          # values may differ in the last bit (here, at most two ulps); every
          # decision must agree.
          mismatches =
            Enum.zip(gpu, reference)
            |> Enum.with_index()
            |> Enum.reject(fn {{a, b}, _} ->
              a == -1.0 == (b == -1.0) and trunc(a) == trunc(b) and
                abs(a - b) <= 2.4e-7 * max(abs(b), 1.0)
            end)

          assert mismatches == [],
                 "pixels differ on #{device.uri}: #{inspect(Enum.take(mismatches, 20))}"
        end
      end
    end

    # A wide image's diagonal scans fold two diagonals into a line, a tall one's do not.
    for {height, width} <- [{37, 203}, {90, 70}], band <- [16, 64], paths <- [3, 8] do
      test "the band shaders match the reference at #{height}x#{width} with a band of #{band}, #{paths} paths",
           %{devices: devices} do
        {height, width} = {unquote(height), unquote(width)}
        {left, right} = layered(height, width, 4.5, 14.25)
        opts = [disparities: 64, p1: 7, p2: 70, uniqueness: 1.05, paths: unquote(paths)]
        full = Nx.Defn.jit(&NxStereo.sgm(&1, &2, opts), compiler: EXLA).(left, right)

        base =
          if unquote(band) == 64,
            do: Nx.broadcast(Nx.s32(0), {height, width}),
            else:
              full
              |> Nx.max(0)
              |> Nx.floor()
              |> Nx.as_type(:s32)
              |> Nx.subtract(4)
              |> Nx.clip(0, 48)
              # Pixels without an estimate
              |> Nx.put_slice([10, div(width, 2)], Nx.broadcast(Nx.s32(-1), {8, div(width, 7)}))

        band_opts = Keyword.put(opts, :band, unquote(band))
        sgm_band = &NxStereo.sgm_band(&1, &2, &3, band_opts)

        reference =
          Nx.Defn.jit(sgm_band, compiler: EXLA).(left, right, base) |> Nx.to_flat_list()

        for device <- devices do
          gpu =
            Nx.Defn.jit(sgm_band,
              compiler: NxIREE.Compiler,
              iree_runtime_options: [device: device]
            ).(left, right, base)
            |> Nx.backend_transfer(Nx.BinaryBackend)
            |> Nx.to_flat_list()

          mismatches =
            Enum.zip(gpu, reference)
            |> Enum.with_index()
            |> Enum.reject(fn {{a, b}, _} ->
              a == -1.0 == (b == -1.0) and trunc(a) == trunc(b) and
                abs(a - b) <= 2.4e-7 * max(abs(b), 1.0)
            end)

          assert mismatches == [],
                 "pixels differ on #{device.uri}: #{inspect(Enum.take(mismatches, 20))}"
        end
      end
    end

    for type <- [:u8, :s32] do
      test "coarse to fine on the shaders matches the reference, #{type} images", %{
        devices: devices
      } do
        {left, right} = layered(48, 240, 20.5, 60.25)
        {left, right} = {Nx.as_type(left, unquote(type)), Nx.as_type(right, unquote(type))}
        fun = &NxStereo.sgm_coarse_to_fine(&1, &2, p1: 7, p2: 70, uniqueness: 1.05)
        reference = Nx.Defn.jit(fun, compiler: EXLA).(left, right) |> Nx.to_flat_list()

        for device <- devices do
          gpu =
            Nx.Defn.jit(fun, compiler: NxIREE.Compiler, iree_runtime_options: [device: device]).(
              left,
              right
            )
            |> Nx.backend_transfer(Nx.BinaryBackend)
            |> Nx.to_flat_list()

          mismatches =
            Enum.zip(gpu, reference)
            |> Enum.with_index()
            |> Enum.reject(fn {{a, b}, _} ->
              a == -1.0 == (b == -1.0) and trunc(a) == trunc(b) and
                abs(a - b) <= 2.4e-7 * max(abs(b), 1.0)
            end)

          assert mismatches == [],
                 "pixels differ on #{device.uri}: #{inspect(Enum.take(mismatches, 20))}"
        end
      end
    end

    test "the shaders take the penalties as inputs", %{devices: devices} do
      {left, right} = layered(24, 160, 4.5, 14.25)
      templates = [Nx.template({}, :f32), Nx.template({}, :f32), Nx.template({}, :f32)]
      sgm = &NxStereo.sgm(&1, &2, p1: &3, p2: &4, uniqueness: &5)

      for device <- devices do
        compiled =
          Nx.Defn.compile(sgm, [left, right | templates],
            compiler: NxIREE.Compiler,
            iree_runtime_options: [device: device]
          )

        for {p1, p2, uniqueness} <- [{7, 70, 1.05}, {3, 20, 1.3}] do
          reference = NxStereo.sgm(left, right, p1: p1, p2: p2, uniqueness: uniqueness)

          gpu =
            compiled.(left, right, Nx.f32(p1), Nx.f32(p2), Nx.f32(uniqueness))
            |> Nx.backend_transfer(Nx.BinaryBackend)

          decisions = fn t -> t |> Nx.floor() |> Nx.to_flat_list() end
          assert decisions.(gpu) == decisions.(reference), "#{device.uri}, #{p1}/#{p2}"
        end
      end
    end
  end
end
