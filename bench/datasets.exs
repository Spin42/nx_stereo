# Accuracy of NxStereo.sgm/3 on stereo datasets with ground truth and, on a
# Vulkan device, how the shaders agree with the reference and their time per
# frame. Datasets are downloaded into _build/bench.
#
#     mix run bench/datasets.exs [--datasets middlebury,eth3d,kitti] [--paths 3|8] [--device URI_SUBSTRING]
#
#   * middlebury (default): the 15 MiddEval3 training scenes at quarter
#     resolution, non-occluded pixels, 45 MB.
#   * eth3d: the 27 two-view training scenes, non-occluded pixels; needs 7z.
#   * kitti: every 20th KITTI 2015 training pair at half resolution, 1.7 GB.
#
# Scenes whose ground truth exceeds 127 disparities are skipped. A pixel is
# bad when it is more than 2 disparities off.

defmodule Bench.Datasets do
  @cache "_build/bench"

  def middlebury do
    dir = Path.join(@cache, "middlebury")

    for zip <- ["MiddEval3-data-Q.zip", "MiddEval3-GT0-Q.zip"] do
      unzip(fetch("https://vision.middlebury.edu/stereo/submit3/zip/" <> zip, dir), dir)
    end

    for scene <- dir |> Path.join("MiddEval3/trainingQ/*") |> Path.wildcard() |> Enum.sort() do
      gt = scene |> Path.join("disp0GT.pfm") |> read_pfm()
      nonocc = scene |> Path.join("mask0nocc.png") |> read_png() |> Nx.equal(255)

      {Path.basename(scene), gray(Path.join(scene, "im0.png")), gray(Path.join(scene, "im1.png")),
       known(gt, nonocc)}
    end
  end

  def eth3d do
    dir = Path.join(@cache, "eth3d")
    System.find_executable("7z") || Mix.raise("ETH3D needs 7z")

    for archive <- ["two_view_training.7z", "two_view_training_gt.7z"] do
      path = fetch("https://www.eth3d.net/data/" <> archive, dir)
      {_, 0} = System.cmd("7z", ["x", "-y", "-o" <> dir, path], stderr_to_stdout: true)
    end

    for scene <- dir |> Path.join("*/disp0GT.pfm") |> Path.wildcard() |> Enum.sort() do
      scene = Path.dirname(scene)
      gt = scene |> Path.join("disp0GT.pfm") |> read_pfm()
      nonocc = scene |> Path.join("mask0nocc.png") |> read_png() |> Nx.equal(255)

      {Path.basename(scene), gray(Path.join(scene, "im0.png")), gray(Path.join(scene, "im1.png")),
       known(gt, nonocc)}
    end
  end

  # Half resolution keeps the disparities within 128.
  def kitti do
    dir = Path.join(@cache, "kitti")
    zip = fetch("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip", dir)
    frames = for i <- 0..180//20, do: :io_lib.format("~6..0B_10.png", [i]) |> to_string()
    folders = ["image_2", "image_3", "disp_occ_0"]
    unzip(zip, dir, for(f <- frames, d <- folders, do: "training/#{d}/#{f}"))

    for frame <- frames do
      read = &Path.join([dir, "training", &1, frame])
      gt = read.("disp_occ_0") |> read_png() |> Nx.as_type(:f32) |> Nx.divide(256) |> sample()

      {Path.rootname(frame), halve(gray(read.("image_2"))), halve(gray(read.("image_3"))),
       Nx.select(Nx.greater(gt, 0), Nx.divide(gt, 2), -1.0)}
    end
  end

  defp known(gt, mask) do
    Nx.select(Nx.logical_and(Nx.is_infinity(gt) |> Nx.logical_not(), mask), gt, -1.0)
  end

  defp even(t) do
    {h, w} = Nx.shape(t)
    Nx.slice(t, [0, 0], [div(h, 2) * 2, div(w, 2) * 2])
  end

  defp halve(image) do
    image = even(image)
    {h, w} = Nx.shape(image)

    image
    |> Nx.reshape({div(h, 2), 2, div(w, 2), 2})
    |> Nx.as_type(:f32)
    |> Nx.mean(axes: [1, 3])
    |> Nx.round()
    |> Nx.as_type(:u8)
  end

  defp sample(t) do
    {h, w} = t |> even() |> Nx.shape()
    t |> Nx.slice([0, 0], [h, w], strides: [2, 2])
  end

  defp gray(path) do
    image = read_png(path)

    case Nx.shape(image) do
      {_, _} ->
        image

      {_, _, _} ->
        # ITU-R 601-2 luma, as PIL computes it
        weights = Nx.tensor([19595, 38470, 7471], type: :u32)
        rgb = image[[.., .., 0..2]] |> Nx.as_type(:u32)

        rgb
        |> Nx.dot([2], weights, [0])
        |> Nx.add(32768)
        |> Nx.right_shift(16)
        |> Nx.as_type(:u8)
    end
  end

  # PFM rows are stored bottom to top; infinity marks unknown disparities.
  defp read_pfm(path) do
    [header, size, scale, data] = path |> File.read!() |> String.split("\n", parts: 4)
    "Pf" = String.trim(header)
    [w, h] = size |> String.split() |> Enum.map(&String.to_integer/1)

    type =
      if scale |> String.trim() |> Float.parse() |> elem(0) < 0,
        do: {:f, 32},
        else: {:f, 32, :big}

    data
    |> binary_part(0, w * h * 4)
    |> decode_floats(type)
    |> Nx.reshape({h, w})
    |> Nx.reverse(axes: [0])
  end

  defp decode_floats(data, {:f, 32}), do: Nx.from_binary(data, :f32)

  defp decode_floats(data, {:f, 32, :big}),
    do: Nx.tensor(for(<<v::float-32-big <- data>>, do: v), type: :f32)

  # Non-interlaced 8 and 16-bit grey, grey-alpha, RGB and RGBA PNGs.
  defp read_png(path) do
    <<137, "PNG", 13, 10, 26, 10, chunks::binary>> = File.read!(path)
    {<<w::32, h::32, depth, color, _, _, 0>>, data} = png_chunks(chunks, nil, [])
    channels = Map.fetch!(%{0 => 1, 2 => 3, 4 => 2, 6 => 4}, color)
    bpp = channels * div(depth, 8)
    raw = unfilter(:zlib.uncompress(data), w * bpp, bpp)

    image =
      case depth do
        8 -> Nx.from_binary(raw, :u8)
        16 -> Nx.tensor(for(<<v::16-big <- raw>>, do: v), type: :u16)
      end

    if channels == 1, do: Nx.reshape(image, {h, w}), else: Nx.reshape(image, {h, w, channels})
  end

  defp png_chunks(
         <<len::32, type::binary-4, body::binary-size(len), _crc::32, rest::binary>>,
         header,
         data
       ) do
    case type do
      "IHDR" -> png_chunks(rest, body, data)
      "IDAT" -> png_chunks(rest, header, [data | body])
      "IEND" -> {header, IO.iodata_to_binary(data)}
      _ -> png_chunks(rest, header, data)
    end
  end

  defp unfilter(raw, stride, bpp) do
    {rows, _} =
      for <<filter, row::binary-size(^stride) <- raw>>,
        reduce: {[], :binary.copy(<<0>>, stride)} do
        {rows, previous} ->
          row = unfilter_row(filter, row, previous, bpp)
          {[row | rows], row}
      end

    rows |> Enum.reverse() |> IO.iodata_to_binary()
  end

  defp unfilter_row(0, row, _previous, _bpp), do: row

  defp unfilter_row(filter, row, previous, bpp) do
    up = :binary.bin_to_list(previous)
    up_left = List.duplicate(0, bpp) ++ up

    {out, _} =
      Enum.zip([:binary.bin_to_list(row), up, up_left])
      |> Enum.with_index()
      |> Enum.reduce({[], {}}, fn {{x, b, c}, i}, {out, window} ->
        a = if i >= bpp, do: elem(window, rem(i, bpp)), else: 0

        value =
          case filter do
            1 -> x + a
            2 -> x + b
            3 -> x + div(a + b, 2)
            4 -> x + paeth(a, b, c)
          end
          |> Bitwise.band(255)

        window =
          if tuple_size(window) < bpp,
            do: Tuple.insert_at(window, tuple_size(window), value),
            else: put_elem(window, rem(i, bpp), value)

        {[value | out], window}
      end)

    out |> Enum.reverse() |> :binary.list_to_bin()
  end

  defp paeth(a, b, c) do
    p = a + b - c
    {pa, pb, pc} = {abs(p - a), abs(p - b), abs(p - c)}

    cond do
      pa <= pb and pa <= pc -> a
      pb <= pc -> b
      true -> c
    end
  end

  defp fetch(url, dir) do
    path = Path.join(dir, Path.basename(url))

    unless File.exists?(path) do
      File.mkdir_p!(dir)
      Mix.shell().info("downloading #{url}")
      {:ok, _} = Application.ensure_all_started([:inets, :ssl])
      download(url, path, 3)
    end

    path
  end

  defp download(url, path, attempts) do
    ssl = [
      verify: :verify_peer,
      cacerts: :public_key.cacerts_get(),
      customize_hostname_check: [match_fun: :public_key.pkix_verify_hostname_match_fun(:https)]
    ]

    part = path <> ".part"

    case :httpc.request(:get, {to_charlist(url), []}, [ssl: ssl], stream: to_charlist(part)) do
      {:ok, :saved_to_file} ->
        File.rename!(part, path)

      error when attempts > 1 ->
        Mix.shell().info("retrying after #{inspect(error)}")
        download(url, path, attempts - 1)

      error ->
        Mix.raise("could not download #{url}: #{inspect(error)}")
    end
  end

  defp unzip(zip, dir, files \\ nil) do
    opts =
      [cwd: to_charlist(dir)] ++
        if(files, do: [file_list: Enum.map(files, &to_charlist/1)], else: [])

    {:ok, _} = :zip.unzip(to_charlist(zip), opts)
  end
end

defmodule Bench do
  def run(args) do
    {opts, _} =
      OptionParser.parse!(args, strict: [datasets: :string, device: :string, paths: :integer])

    Nx.default_backend(EXLA.Backend)
    device = device(opts[:device])

    if device,
      do: IO.puts("Vulkan device: #{device.uri}"),
      else: IO.puts("No Vulkan device: reference only")

    datasets = opts |> Keyword.get(:datasets, "middlebury") |> String.split(",")

    for dataset <- datasets, dataset not in ~w(middlebury eth3d kitti) do
      Mix.raise("unknown dataset #{dataset}, expected middlebury, eth3d or kitti")
    end

    header =
      ["dataset", "scene", "size", "D", "matched", "bad", "bad or missing"] ++
        if(device, do: ["GPU decisions differing", "GPU ms/frame"], else: [])

    rows =
      for dataset <- datasets,
          {scene, left, right, gt} <- apply(Bench.Datasets, String.to_atom(dataset), []),
          row = scene_row(dataset, scene, left, right, gt, device, Keyword.get(opts, :paths, 8)),
          row != nil,
          do: row

    print([header | Enum.map(rows, &elem(&1, 0))])

    summary =
      for {dataset, scenes} <- Enum.group_by(rows, &elem(&1, 1)) do
        metrics = Enum.map(scenes, &elem(&1, 2))

        mean = fn i ->
          metrics |> Enum.map(&elem(&1, i)) |> Enum.sum() |> Kernel./(length(metrics))
        end

        [
          dataset,
          "#{length(metrics)} scenes",
          percent(mean.(0)),
          percent(mean.(1)),
          percent(mean.(2))
        ]
      end

    IO.puts("")
    print([["dataset", "", "matched", "bad", "bad or missing"] | summary])
  end

  defp scene_row(dataset, scene, left, right, gt, device, paths) do
    {h, w} = Nx.shape(left)
    top = gt |> Nx.reduce_max() |> Nx.to_number()

    disparities =
      cond do
        top < 63 -> 64
        top < 127 -> 128
        true -> nil
      end

    if disparities == nil do
      IO.puts("skipping #{dataset} #{scene}: ground truth reaches #{round(top)} disparities")
      nil
    else
      fun = &NxStereo.sgm(&1, &2, disparities: disparities, paths: paths)
      reference = Nx.Defn.jit(fun, compiler: EXLA).(left, right)
      metrics = score(reference, gt)

      gpu =
        if device do
          jitted =
            Nx.Defn.jit(fun, compiler: NxIREE.Compiler, iree_runtime_options: [device: device])

          frame = fn -> jitted.(left, right) |> Nx.backend_transfer(EXLA.Backend) end
          output = frame.()
          times = for _ <- 1..5, do: elem(:timer.tc(frame), 0) / 1000

          [
            Integer.to_string(differing(reference, output)),
            :erlang.float_to_binary(median(times), decimals: 1)
          ]
        else
          []
        end

      {[dataset, scene, "#{w}x#{h}", Integer.to_string(disparities)] ++
         Enum.map(Tuple.to_list(metrics), &percent/1) ++ gpu, dataset, metrics}
    end
  end

  defp score(disparity, gt) do
    known = Nx.greater_equal(gt, 0)
    valid = Nx.logical_and(known, Nx.greater_equal(disparity, 0))
    bad = Nx.greater(Nx.abs(Nx.subtract(disparity, gt)), 2)
    count = fn t -> t |> Nx.sum() |> Nx.to_number() end
    {known_n, valid_n} = {count.(known), count.(valid)}

    {valid_n / known_n, count.(Nx.logical_and(valid, bad)) / max(valid_n, 1),
     count.(Nx.logical_and(known, Nx.logical_or(bad, Nx.logical_not(valid)))) / known_n}
  end

  # Pixels where the shaders and the reference disagree on validity or on the
  # integer disparity; subpixel values may differ in their last bit.
  defp differing(a, b) do
    validity = Nx.not_equal(Nx.greater_equal(a, 0), Nx.greater_equal(b, 0))
    integer = Nx.logical_and(Nx.greater_equal(a, 0), Nx.not_equal(Nx.floor(a), Nx.floor(b)))
    Nx.logical_or(validity, integer) |> Nx.sum() |> Nx.to_number()
  end

  defp device(filter) do
    if Code.ensure_loaded?(NxIREE) do
      {:ok, devices} = NxIREE.list_devices("vulkan")
      Enum.find(devices, &String.contains?(&1.uri, filter || ""))
    end
  end

  defp median(values), do: values |> Enum.sort() |> Enum.at(div(length(values), 2))
  defp percent(value), do: :erlang.float_to_binary(value * 100, decimals: 1) <> "%"

  defp print(rows) do
    widths =
      rows
      |> Enum.zip_with(& &1)
      |> Enum.map(fn column -> column |> Enum.map(&String.length/1) |> Enum.max() end)

    for row <- rows do
      row
      |> Enum.zip(widths)
      |> Enum.map_join("  ", fn {cell, width} -> String.pad_trailing(cell, width) end)
      |> IO.puts()
    end
  end
end

Bench.run(System.argv())
