defmodule NxStereo.Image do
  @moduledoc """
  Image operations around stereo matching, as OpenCV computes them:
  area resizing, a median filter and the turbo colour map.
  """
  import Nx.Defn

  # OpenCV's COLORMAP_TURBO, as RGB.
  @turbo_rgb [
    {48, 18, 59},
    {50, 21, 67},
    {51, 24, 74},
    {52, 27, 81},
    {53, 30, 88},
    {54, 33, 95},
    {55, 36, 102},
    {56, 39, 109},
    {57, 42, 115},
    {58, 45, 121},
    {59, 47, 128},
    {60, 50, 134},
    {61, 53, 139},
    {62, 56, 145},
    {63, 59, 151},
    {63, 62, 156},
    {64, 64, 162},
    {65, 67, 167},
    {65, 70, 172},
    {66, 73, 177},
    {66, 75, 181},
    {67, 78, 186},
    {68, 81, 191},
    {68, 84, 195},
    {68, 86, 199},
    {69, 89, 203},
    {69, 92, 207},
    {69, 94, 211},
    {70, 97, 214},
    {70, 100, 218},
    {70, 102, 221},
    {70, 105, 224},
    {70, 107, 227},
    {71, 110, 230},
    {71, 113, 233},
    {71, 115, 235},
    {71, 118, 238},
    {71, 120, 240},
    {71, 123, 242},
    {70, 125, 244},
    {70, 128, 246},
    {70, 130, 248},
    {70, 133, 250},
    {70, 135, 251},
    {69, 138, 252},
    {69, 140, 253},
    {68, 143, 254},
    {67, 145, 254},
    {66, 148, 255},
    {65, 150, 255},
    {64, 153, 255},
    {62, 155, 254},
    {61, 158, 254},
    {59, 160, 253},
    {58, 163, 252},
    {56, 165, 251},
    {55, 168, 250},
    {53, 171, 248},
    {51, 173, 247},
    {49, 175, 245},
    {47, 178, 244},
    {46, 180, 242},
    {44, 183, 240},
    {42, 185, 238},
    {40, 188, 235},
    {39, 190, 233},
    {37, 192, 231},
    {35, 195, 228},
    {34, 197, 226},
    {32, 199, 223},
    {31, 201, 221},
    {30, 203, 218},
    {28, 205, 216},
    {27, 208, 213},
    {26, 210, 210},
    {26, 212, 208},
    {25, 213, 205},
    {24, 215, 202},
    {24, 217, 200},
    {24, 219, 197},
    {24, 221, 194},
    {24, 222, 192},
    {24, 224, 189},
    {25, 226, 187},
    {25, 227, 185},
    {26, 228, 182},
    {28, 230, 180},
    {29, 231, 178},
    {31, 233, 175},
    {32, 234, 172},
    {34, 235, 170},
    {37, 236, 167},
    {39, 238, 164},
    {42, 239, 161},
    {44, 240, 158},
    {47, 241, 155},
    {50, 242, 152},
    {53, 243, 148},
    {56, 244, 145},
    {60, 245, 142},
    {63, 246, 138},
    {67, 247, 135},
    {70, 248, 132},
    {74, 248, 128},
    {78, 249, 125},
    {82, 250, 122},
    {85, 250, 118},
    {89, 251, 115},
    {93, 252, 111},
    {97, 252, 108},
    {101, 253, 105},
    {105, 253, 102},
    {109, 254, 98},
    {113, 254, 95},
    {117, 254, 92},
    {121, 254, 89},
    {125, 255, 86},
    {128, 255, 83},
    {132, 255, 81},
    {136, 255, 78},
    {139, 255, 75},
    {143, 255, 73},
    {146, 255, 71},
    {150, 254, 68},
    {153, 254, 66},
    {156, 254, 64},
    {159, 253, 63},
    {161, 253, 61},
    {164, 252, 60},
    {167, 252, 58},
    {169, 251, 57},
    {172, 251, 56},
    {175, 250, 55},
    {177, 249, 54},
    {180, 248, 54},
    {183, 247, 53},
    {185, 246, 53},
    {188, 245, 52},
    {190, 244, 52},
    {193, 243, 52},
    {195, 241, 52},
    {198, 240, 52},
    {200, 239, 52},
    {203, 237, 52},
    {205, 236, 52},
    {208, 234, 52},
    {210, 233, 53},
    {212, 231, 53},
    {215, 229, 53},
    {217, 228, 54},
    {219, 226, 54},
    {221, 224, 55},
    {223, 223, 55},
    {225, 221, 55},
    {227, 219, 56},
    {229, 217, 56},
    {231, 215, 57},
    {233, 213, 57},
    {235, 211, 57},
    {236, 209, 58},
    {238, 207, 58},
    {239, 205, 58},
    {241, 203, 58},
    {242, 201, 58},
    {244, 199, 58},
    {245, 197, 58},
    {246, 195, 58},
    {247, 193, 58},
    {248, 190, 57},
    {249, 188, 57},
    {250, 186, 57},
    {251, 184, 56},
    {251, 182, 55},
    {252, 179, 54},
    {252, 177, 54},
    {253, 174, 53},
    {253, 172, 52},
    {254, 169, 51},
    {254, 167, 50},
    {254, 164, 49},
    {254, 161, 48},
    {254, 158, 47},
    {254, 155, 45},
    {254, 153, 44},
    {254, 150, 43},
    {254, 147, 42},
    {254, 144, 41},
    {253, 141, 39},
    {253, 138, 38},
    {252, 135, 37},
    {252, 132, 35},
    {251, 129, 34},
    {251, 126, 33},
    {250, 123, 31},
    {249, 120, 30},
    {249, 117, 29},
    {248, 114, 28},
    {247, 111, 26},
    {246, 108, 25},
    {245, 105, 24},
    {244, 102, 23},
    {243, 99, 21},
    {242, 96, 20},
    {241, 93, 19},
    {240, 91, 18},
    {239, 88, 17},
    {237, 85, 16},
    {236, 83, 15},
    {235, 80, 14},
    {234, 78, 13},
    {232, 75, 12},
    {231, 73, 12},
    {229, 71, 11},
    {228, 69, 10},
    {226, 67, 10},
    {225, 65, 9},
    {223, 63, 8},
    {221, 61, 8},
    {220, 59, 7},
    {218, 57, 7},
    {216, 55, 6},
    {214, 53, 6},
    {212, 51, 5},
    {210, 49, 5},
    {208, 47, 5},
    {206, 45, 4},
    {204, 43, 4},
    {202, 42, 4},
    {200, 40, 3},
    {197, 38, 3},
    {195, 37, 3},
    {193, 35, 2},
    {190, 33, 2},
    {188, 32, 2},
    {185, 30, 2},
    {183, 29, 2},
    {180, 27, 1},
    {178, 26, 1},
    {175, 24, 1},
    {172, 23, 1},
    {169, 22, 1},
    {167, 20, 1},
    {164, 19, 1},
    {161, 18, 1},
    {158, 16, 1},
    {155, 15, 1},
    {152, 14, 1},
    {149, 13, 1},
    {146, 11, 1},
    {142, 10, 1},
    {139, 9, 2},
    {136, 8, 2},
    {133, 7, 2},
    {129, 6, 2},
    {126, 5, 2},
    {122, 4, 3}
  ]
  @turbo Enum.map(@turbo_rgb, &Tuple.to_list/1)

  # Median selection networks (Devillard, after Paeth): compare-exchanges
  # that leave the median of 9 and of 25 values in the middle position,
  # all element-wise min and max on a GPU.
  @median_9 [
    {1, 2},
    {4, 5},
    {7, 8},
    {0, 1},
    {3, 4},
    {6, 7},
    {1, 2},
    {4, 5},
    {7, 8},
    {0, 3},
    {5, 8},
    {4, 7},
    {3, 6},
    {1, 4},
    {2, 5},
    {4, 7},
    {4, 2},
    {6, 4},
    {4, 2}
  ]

  @median_25 [
    {0, 1},
    {3, 4},
    {2, 4},
    {2, 3},
    {6, 7},
    {5, 7},
    {5, 6},
    {9, 10},
    {8, 10},
    {8, 9},
    {12, 13},
    {11, 13},
    {11, 12},
    {15, 16},
    {14, 16},
    {14, 15},
    {18, 19},
    {17, 19},
    {17, 18},
    {21, 22},
    {20, 22},
    {20, 21},
    {23, 24},
    {2, 5},
    {3, 6},
    {0, 6},
    {0, 3},
    {4, 7},
    {1, 7},
    {1, 4},
    {11, 14},
    {8, 14},
    {8, 11},
    {12, 15},
    {9, 15},
    {9, 12},
    {13, 16},
    {10, 16},
    {10, 13},
    {20, 23},
    {17, 23},
    {17, 20},
    {21, 24},
    {18, 24},
    {18, 21},
    {19, 22},
    {8, 17},
    {9, 18},
    {0, 18},
    {0, 9},
    {10, 19},
    {1, 19},
    {1, 10},
    {11, 20},
    {2, 20},
    {2, 11},
    {12, 21},
    {3, 21},
    {3, 12},
    {13, 22},
    {4, 22},
    {4, 13},
    {14, 23},
    {5, 23},
    {5, 14},
    {15, 24},
    {6, 24},
    {6, 15},
    {7, 16},
    {7, 19},
    {13, 21},
    {15, 23},
    {7, 13},
    {7, 15},
    {1, 9},
    {3, 11},
    {5, 17},
    {11, 17},
    {9, 17},
    {4, 10},
    {6, 12},
    {7, 14},
    {4, 6},
    {4, 7},
    {12, 14},
    {10, 14},
    {6, 7},
    {10, 12},
    {6, 10},
    {6, 17},
    {12, 17},
    {7, 17},
    {7, 10},
    {12, 18},
    {7, 12},
    {10, 18},
    {12, 20},
    {10, 20},
    {10, 12}
  ]

  @doc """
  Shrinks a `u8` `{height, width}` image to `size` (`{height, width}`)
  by area averaging, as OpenCV's `INTER_AREA`: each output pixel is the
  mean of the input area it covers. The same size returns the image as it
  is.
  """
  deftransform resize_area(image, {height, width} = size) do
    {in_height, in_width} = Nx.shape(image)

    cond do
      {in_height, in_width} == size ->
        image

      height > in_height or width > in_width ->
        raise ArgumentError,
              "resize_area only shrinks, got #{in_height}x#{in_width} to #{height}x#{width}"

      true ->
        image
        |> Nx.as_type(:f32)
        |> shrink_rows(height)
        |> Nx.transpose()
        |> shrink_rows(width)
        |> Nx.transpose()
        |> Nx.round()
        |> Nx.clip(0, 255)
        |> Nx.as_type(:u8)
    end
  end

  # Each output row is the weighted sum of the few input rows it covers,
  # gathered from the flattened image: `Nx.gather`, as an axis `take`
  # does not compile on NxIREE.
  defp shrink_rows(image, size) do
    {in_size, width} = Nx.shape(image)
    {rows, weights} = area_weights(size, in_size)
    taps = Nx.axis_size(rows, 1)

    indices =
      rows
      |> Nx.reshape({size, taps, 1})
      |> Nx.multiply(width)
      |> Nx.add(Nx.iota({1, 1, width}))
      |> Nx.new_axis(-1)

    image
    |> Nx.reshape({in_size * width, 1})
    |> Nx.gather(indices)
    |> Nx.reshape({size, taps, width})
    |> Nx.multiply(Nx.reshape(weights, {size, taps, 1}))
    |> Nx.sum(axes: [1])
  end

  # For each output pixel, the input pixels it covers and their shares,
  # padded to the same count with zero weights.
  defp area_weights(size, in_size) do
    scale = in_size / size

    taps =
      for output <- 0..(size - 1) do
        {first, last} = {output * scale, (output + 1) * scale}

        for input <- floor(first)..(ceil(last) - 1),
            share = min(input + 1, last) - max(input, first),
            share > 0,
            do: {input, share / scale}
      end

    count = taps |> Enum.map(&length/1) |> Enum.max()
    padded = Enum.map(taps, &(&1 ++ List.duplicate({0, 0.0}, count - length(&1))))

    {Nx.tensor(Enum.map(padded, fn row -> Enum.map(row, &elem(&1, 0)) end), type: :s32),
     Nx.tensor(Enum.map(padded, fn row -> Enum.map(row, &elem(&1, 1)) end), type: :f32)}
  end

  @doc """
  The median of each pixel's `kernel_size` x `kernel_size` neighbourhood
  (3 or 5), with replicated borders, as OpenCV's `medianBlur`.
  """
  deftransform median(image, kernel_size) do
    network =
      case kernel_size do
        3 ->
          @median_9

        5 ->
          @median_25

        _ ->
          raise ArgumentError, "median takes a kernel_size of 3 or 5, got #{inspect(kernel_size)}"
      end

    radius = div(kernel_size, 2)
    neighbours = for dy <- -radius..radius, dx <- -radius..radius, do: shift(image, dy, dx)

    network
    |> Enum.reduce(List.to_tuple(neighbours), fn {a, b}, values ->
      {low, high} = {elem(values, a), elem(values, b)}
      values |> put_elem(a, Nx.min(low, high)) |> put_elem(b, Nx.max(low, high))
    end)
    |> elem(div(kernel_size * kernel_size, 2))
  end

  # `image` moved by `dy` rows and `dx` columns, the edge repeated: slices
  # rather than an axis `take`, which NxIREE cannot compile.
  defp shift(image, dy, dx), do: image |> shift_axis(dy, 0) |> shift_axis(dx, 1)

  defp shift_axis(image, 0, _axis), do: image

  defp shift_axis(image, offset, axis) when offset > 0 do
    size = Nx.axis_size(image, axis)
    edge = Nx.slice_along_axis(image, size - 1, 1, axis: axis)
    rest = Nx.slice_along_axis(image, offset, size - offset, axis: axis)
    Nx.concatenate([rest | List.duplicate(edge, offset)], axis: axis)
  end

  defp shift_axis(image, offset, axis) do
    size = Nx.axis_size(image, axis)
    edge = Nx.slice_along_axis(image, 0, 1, axis: axis)
    rest = Nx.slice_along_axis(image, 0, size + offset, axis: axis)
    Nx.concatenate(List.duplicate(edge, -offset) ++ [rest], axis: axis)
  end

  @doc """
  Colours an `f32` depth map in metres as `u8` RGB `{height, width, 3}`
  with OpenCV's turbo map: red at the camera, blue at `range_m` and
  beyond, black where there is no depth (0 or less).
  """
  deftransform turbo(depth_m, range_m) do
    turbo_n(depth_m, Nx.tensor(@turbo, type: :u8), range_m: range_m)
  end

  defnp turbo_n(depth_m, lut, opts \\ []) do
    range_m = opts[:range_m]
    index = 255 - Nx.round(Nx.min(depth_m, range_m) * (255 / range_m))
    # A gather rather than `Nx.take`, which NxIREE cannot compile here.
    colours = Nx.gather(lut, index |> Nx.clip(0, 255) |> Nx.as_type(:s32) |> Nx.new_axis(-1))
    Nx.select(Nx.broadcast(Nx.new_axis(depth_m > 0, -1), Nx.shape(colours)), colours, Nx.u8(0))
  end
end
