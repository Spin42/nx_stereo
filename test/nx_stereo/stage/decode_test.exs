defmodule NxStereo.Stage.DecodeTest do
  use ExUnit.Case, async: true

  alias NxStereo.{NxSegment, Packing, Segment, Stage}

  defp decode(frames, size, defn_options) do
    {frame_height, frame_width} = Nx.shape(frames.left_frame)
    {height, width} = size

    geometry = %{
      width: width,
      height: height,
      frame_width: frame_width,
      frame_height: frame_height
    }

    {:ok, members} = Segment.members([Stage.Decode, Stage.Sharpness], geometry)
    {:ok, segment} = NxSegment.build(members, geometry, defn_options: defn_options)
    {:ok, outputs, _segment} = Segment.run(NxSegment, frames, segment, :infinity)
    outputs
  end

  defp frames do
    left = Nx.tensor([[0, 4, 8, 12, 16, 20, 24, 28], [4, 8, 12, 16, 20, 24, 28, 32]], type: :u8)
    right = Nx.add(left, 40)
    %{left_frame: left, right_frame: right}
  end

  test "packed frames decode to their pixels at the working size" do
    frames = frames()
    outputs = decode(frames, {2, 8}, compiler: EXLA, byte_storage: false)

    assert Nx.type(outputs.left) == {:u, 8}
    assert Nx.to_list(outputs.left) == Nx.to_list(frames.left_frame)
    assert Nx.to_list(outputs.right) == Nx.to_list(frames.right_frame)
  end

  test "packed frames average pixels rather than words when shrinking" do
    outputs = decode(frames(), {1, 4}, compiler: EXLA, byte_storage: false)

    assert Nx.to_list(outputs.left) == [[4, 12, 20, 28]]
    assert Nx.to_list(outputs.right) == [[44, 52, 60, 68]]
    assert Nx.shape(outputs.sharpness_sums) == {6}
  end

  test "noninteger resizing preserves pixel values and 32-bit device storage" do
    frame =
      Nx.tensor(for(row <- 0..3, do: for(column <- 0..11, do: 12 * row + 6 * column)), type: :u8)

    frames = %{left_frame: frame, right_frame: Nx.add(frame, 40)}
    expected = [[8, 16, 26, 34, 44, 52, 62, 70], [32, 40, 50, 58, 68, 76, 86, 94]]

    for byte_storage <- [true, false] do
      outputs = decode(frames, {2, 8}, compiler: EXLA, byte_storage: byte_storage)
      assert Nx.to_list(outputs.left) == expected

      assert Nx.to_list(outputs.right) ==
               Enum.map(expected, &Enum.map(&1, fn pixel -> pixel + 40 end))
    end

    resized = NxStereo.Image.resize_area(Nx.as_type(frame, :s32), {2, 8})
    assert Nx.type(resized) == {:s, 32}
    assert Nx.to_list(resized) == expected
  end

  describe "on Vulkan" do
    @describetag :vulkan
    setup do: NxStereo.TestDevices.vulkan()

    test "repeated resizes preserve exact area averages at a partial workgroup", %{
      devices: devices
    } do
      {height, width} = {7, 126}

      pairs =
        for phase <- 0..7 do
          totals = for index <- 0..(height * width - 1), do: rem(index * 37 + phase, 1021)
          totals_by_pixel = List.to_tuple(totals)

          pixels =
            for row <- 0..(2 * height - 1), column <- 0..(2 * width - 1) do
              total = elem(totals_by_pixel, div(row, 2) * width + div(column, 2))
              lane = rem(row, 2) * 2 + rem(column, 2)
              min(max(total - 255 * lane, 0), 255)
            end

          left = Nx.tensor(pixels, type: :u8) |> Nx.reshape({2 * height, 2 * width})
          right = Nx.reverse(left, axes: [0, 1])
          expected = Enum.map(totals, &div(&1 + 2, 4))
          {left, right, expected}
        end

      decode = fn left, right ->
        outputs =
          Stage.Decode.step(%{left_frame: left, right_frame: right}, %{}, %{size: {height, width}})

        {outputs.left, outputs.right}
      end

      generic = fn left, right ->
        resize = fn frame ->
          image = if Nx.type(frame) == {:u, 32}, do: Packing.unpack(frame, :s32), else: frame
          NxStereo.Image.resize_area(image, {height, width})
        end

        {resize.(left), resize.(right)}
      end

      for device <- devices,
          packed <- [false, true],
          {method, function} <- [decode: decode, generic: generic] do
        compiled =
          Nx.Defn.jit(function, compiler: NxIREE.Compiler, iree_runtime_options: [device: device])

        resident =
          Enum.map(pairs, fn {left, right, expected} ->
            {left, right} =
              if packed do
                {Packing.pack(Nx.to_binary(left), 2 * width, 2 * height),
                 Packing.pack(Nx.to_binary(right), 2 * width, 2 * height)}
              else
                {left, right}
              end

            {Nx.backend_copy(left, {NxIREE.Backend, device: device}),
             Nx.backend_copy(right, {NxIREE.Backend, device: device}), expected}
          end)
          |> List.to_tuple()

        for index <- 0..255 do
          {left, right, expected} = elem(resident, rem(index, 8))
          output = {decoded_left, decoded_right} = compiled.(left, right)
          actual = {Nx.to_flat_list(decoded_left), Nx.to_flat_list(decoded_right)}
          Nx.backend_deallocate(output)

          assert actual == {expected, Enum.reverse(expected)},
                 "#{device.uri}, #{method}, packed #{packed}, pair #{index}"
        end
      end
    end

    test "packed and byte frames have the same known decoded pixels", %{devices: devices} do
      for device <- devices, byte_storage <- [true, false] do
        options = [
          compiler: NxIREE.Compiler,
          iree_runtime_options: [device: device],
          byte_storage: byte_storage
        ]

        outputs = decode(frames(), {1, 4}, options)

        assert Nx.to_list(outputs.left) == [[4, 12, 20, 28]]
        assert Nx.to_list(outputs.right) == [[44, 52, 60, 68]]
      end
    end
  end
end
