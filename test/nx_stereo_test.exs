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

  test "validates its inputs" do
    {left, right} = pair(16, 80, 0)

    assert_raise ArgumentError, ~r/expects u8 images/, fn ->
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
        scene <- [:shifted, :layered] do
      test "the shaders match the reference on a #{scene} scene at #{height}x#{width}, #{disparities} disparities",
           %{devices: devices} do
        {left, right} =
          case unquote(scene) do
            :shifted -> pair(unquote(height), unquote(width), 7)
            :layered -> layered(unquote(height), unquote(width), 4.5, 14.25)
          end

        opts = [disparities: unquote(disparities), p1: 7, p2: 70, uniqueness: 1.05]

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
          # values may differ in the last bit; every decision must agree.
          mismatches =
            Enum.zip(gpu, reference)
            |> Enum.with_index()
            |> Enum.reject(fn {{a, b}, _} ->
              a == -1.0 == (b == -1.0) and trunc(a) == trunc(b) and abs(a - b) <= 1.0e-6
            end)

          assert mismatches == [],
                 "pixels differ on #{device.uri}: #{inspect(Enum.take(mismatches, 20))}"
        end
      end
    end
  end
end
