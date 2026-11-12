# Ground-truth quality and matcher timings. Datasets are cached in _build/bench.
#
#     mix run test/bench/datasets.exs --datasets middlebury,eth3d,kitti \
#       --search coarse_to_fine --paths 8 --coarse-paths 3 --band 16 --margin 4 \
#       --runs 20 --csv quality.csv
#
# Other options: --search full, --disparities 64|128, --p1 8, --p2 64,
# --uniqueness 1.0526315789473684, --warmup 3, --scene NAME_SUBSTRING,
# --device URI_SUBSTRING and --no-reference (GPU quality without EXLA parity).
# --perturbation clean|noise2|noise4|noise8|gain0.8|gain1.2|blur_right changes
# the images before timing; geometric ground truth stays unchanged.
#
# Middlebury: 15 quarter-resolution scenes, non-occluded truth, 45 MB.
# ETH3D: 27 training scenes, non-occluded truth; needs 7z.
# KITTI: every twentieth 2015 training pair at half resolution, 1.7 GB.
#
# An explicit disparity range scores all scenes, excluding unsupported truth
# and reporting the fraction scored. Coarse-to-fine always uses 128 disparities
# and crops odd dimensions. Full search without --disparities keeps automatic
# 64/128 selection and skips scenes beyond that range. Bad means >2px error;
# mean absolute error covers matched pixels. Summaries weight scenes equally.
# GPU times include output download with resident inputs, after compilation
# and warmup. They measure matching, not the complete camera pipeline.

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
  @moduledoc false
  Code.require_file("support/quality.exs", __DIR__)
  Code.require_file("support/compare.exs", __DIR__)
  Code.require_file("support/options.exs", __DIR__)
  Code.require_file("support/perturbation.exs", __DIR__)

  @defaults [
    datasets: "middlebury",
    perturbation: "clean",
    search: "full",
    paths: 8,
    coarse_paths: 3,
    fine_census: "sparse_9x7",
    band: 16,
    margin: 4,
    p1: 8,
    p2: 64,
    uniqueness: 1.0526315789473684,
    runs: 10,
    warmup: 3,
    reference: true
  ]
  @switches [
    datasets: :string,
    perturbation: :string,
    scene: :string,
    device: :string,
    search: :string,
    disparities: :integer,
    paths: :integer,
    coarse_paths: :integer,
    fine_census: :string,
    band: :integer,
    margin: :integer,
    p1: :integer,
    p2: :integer,
    uniqueness: :float,
    runs: :integer,
    warmup: :integer,
    reference: :boolean,
    csv: :string
  ]
  @loaders %{
    "middlebury" => &Bench.Datasets.middlebury/0,
    "eth3d" => &Bench.Datasets.eth3d/0,
    "kitti" => &Bench.Datasets.kitti/0
  }
  @rates [:scored, :matched, :bad, :bad_or_missing]

  def run(args) do
    {opts, []} = OptionParser.parse!(args, strict: @switches)
    opts = @defaults |> Keyword.merge(opts) |> Bench.Options.fine_census()
    validate!(opts)
    Bench.Perturbation.validate!(opts[:perturbation])
    Nx.default_backend(EXLA.Backend)
    device = device(opts[:device])

    IO.puts(
      if device, do: "Vulkan device: #{device.uri}", else: "No Vulkan device: reference only"
    )

    IO.puts(
      "Settings: #{inspect(Keyword.take(opts, [:search, :disparities, :paths, :coarse_paths, :band, :margin, :fine_census, :p1, :p2, :uniqueness]))}"
    )

    IO.puts("Image condition: #{opts[:perturbation]}")

    IO.puts(
      "GPU timings: resident inputs, output download, #{opts[:warmup]} warmups, #{opts[:runs]} measured runs"
    )

    rows =
      for dataset <- String.split(opts[:datasets], ","),
          {scene, left, right, truth} <- Map.fetch!(@loaders, dataset).(),
          opts[:scene] == nil or String.contains?(scene, opts[:scene]),
          row = scene_row(dataset, scene, left, right, truth, device, opts),
          row != nil,
          do: row

    if rows == [], do: Mix.raise("no scenes selected")

    header = [
      "dataset",
      "scene",
      "size",
      "D",
      "scored",
      "matched",
      "bad >2px",
      "bad or missing",
      "MAE px",
      "GPU differing",
      "GPU median ms",
      "GPU p90 ms"
    ]

    print([header | Enum.map(rows, &table_row/1)])
    print_summary(rows)
    if opts[:csv], do: write_csv(opts[:csv], rows)
  end

  defp validate!(opts) do
    unless opts[:search] in ["full", "coarse_to_fine"],
      do: Mix.raise("--search must be full or coarse_to_fine")

    unless opts[:disparities] in [nil, 64, 128], do: Mix.raise("--disparities must be 64 or 128")

    if opts[:search] == "coarse_to_fine" and opts[:disparities] == 64,
      do: Mix.raise("coarse_to_fine needs 128 disparities")

    unless opts[:runs] > 0 and opts[:warmup] >= 0,
      do: Mix.raise("--runs must be positive and --warmup nonnegative")

    unless opts[:band] in [8, 16, 32, 64] and opts[:margin] >= 0 and opts[:margin] < opts[:band],
      do: Mix.raise("use band 8, 16, 32 or 64 and 0 <= margin < band")

    for dataset <- String.split(opts[:datasets], ","),
        not Map.has_key?(@loaders, dataset),
        do: Mix.raise("unknown dataset #{dataset}")
  end

  defp scene_row(dataset, scene, left, right, truth, device, opts) do
    {left, right, truth} = prepare_pair(left, right, truth, opts[:search])
    {left, right} = Bench.Perturbation.apply(left, right, opts[:perturbation])
    {height, width} = Nx.shape(left)
    disparities = disparities(truth, opts)

    if disparities == nil do
      IO.puts(
        "skipping #{dataset} #{scene}: use --disparities 128 to score its supported ground truth"
      )

      nil
    else
      fun = matcher(disparities, opts)

      reference =
        if device == nil or opts[:reference], do: Nx.Defn.jit(fun, compiler: EXLA).(left, right)

      {output, differing, median, p90} =
        if device do
          measure_gpu(fun, left, right, reference, device, opts)
        else
          {reference, nil, nil, nil}
        end

      Bench.Quality.score(output, truth, disparities)
      |> Map.merge(%{
        dataset: dataset,
        scene: scene,
        width: width,
        height: height,
        disparities: disparities,
        device: if(device, do: device.uri, else: "EXLA"),
        differing: differing,
        gpu_median_ms: median,
        gpu_p90_ms: p90
      })
      |> Map.merge(
        opts
        |> Keyword.take([
          :search,
          :perturbation,
          :paths,
          :coarse_paths,
          :fine_census,
          :band,
          :margin,
          :p1,
          :p2,
          :uniqueness,
          :warmup,
          :runs
        ])
        |> Map.new()
      )
    end
  end

  defp prepare_pair(left, right, truth, "full"), do: {left, right, truth}

  defp prepare_pair(left, right, truth, "coarse_to_fine") do
    {height, width} = Nx.shape(left)
    shape = [div(height, 2) * 2, div(width, 2) * 2]
    [left, right, truth] = Enum.map([left, right, truth], &Nx.slice(&1, [0, 0], shape))
    {left, right, truth}
  end

  defp disparities(truth, opts) do
    cond do
      opts[:search] == "coarse_to_fine" ->
        128

      opts[:disparities] != nil ->
        opts[:disparities]

      true ->
        top = truth |> Nx.reduce_max() |> Nx.to_number()

        cond do
          top < 63 -> 64
          top < 127 -> 128
          true -> nil
        end
    end
  end

  defp matcher(disparities, opts) do
    matching = [disparities: disparities] ++ Keyword.take(opts, [:paths, :p1, :p2, :uniqueness])

    case opts[:search] do
      "full" ->
        &NxStereo.sgm(&1, &2, matching)

      "coarse_to_fine" ->
        matching = matching ++ Keyword.take(opts, [:coarse_paths, :band, :margin, :fine_census])
        &NxStereo.sgm_coarse_to_fine(&1, &2, matching)
    end
  end

  defp measure_gpu(fun, left, right, reference, device, opts) do
    compiled = Nx.Defn.jit(fun, compiler: NxIREE.Compiler, iree_runtime_options: [device: device])

    [left, right] =
      Enum.map([left, right], &Nx.backend_copy(&1, {NxIREE.Backend, device: device}))

    frame = fn -> compiled.(left, right) |> Nx.backend_transfer(EXLA.Backend) end
    output = frame.()
    Enum.each(List.duplicate(nil, opts[:warmup]), fn _ -> frame.() end)
    times = for _ <- 1..opts[:runs], do: elem(:timer.tc(frame), 0) / 1000
    differing = if reference, do: Bench.Compare.disparity(reference, output)
    {output, differing, percentile(times, 0.5), percentile(times, 0.9)}
  end

  defp device(filter) do
    found =
      if Code.ensure_loaded?(NxIREE) do
        {:ok, devices} = NxIREE.list_devices("vulkan")
        Enum.find(devices, &String.contains?(&1.uri, filter || ""))
      end

    if filter && found == nil, do: Mix.raise("no Vulkan device matches #{inspect(filter)}")
    found
  end

  defp table_row(row) do
    [row.dataset, row.scene, "#{row.width}x#{row.height}", to_string(row.disparities)] ++
      Enum.map(@rates, &percent(row[&1])) ++
      Enum.map(
        [row.mean_absolute_error, row.differing, row.gpu_median_ms, row.gpu_p90_ms],
        &number/1
      )
  end

  defp print_summary(rows) do
    summary =
      for {dataset, scenes} <- rows |> Enum.group_by(& &1.dataset) |> Enum.sort() do
        [dataset, "#{length(scenes)} scenes"] ++
          Enum.map(@rates, &percent(mean(scenes, &1))) ++
          [number(mean(scenes, :mean_absolute_error))]
      end

    IO.puts("\nEqual scene-weighted means:")

    print([
      ["dataset", "", "scored", "matched", "bad >2px", "bad or missing", "MAE px"] | summary
    ])
  end

  defp mean(rows, key) do
    scores = rows |> Enum.map(& &1[key]) |> Enum.reject(&is_nil/1)
    if scores != [], do: Enum.sum(scores) / length(scores)
  end

  defp write_csv(path, rows) do
    columns = [
      :dataset,
      :scene,
      :width,
      :height,
      :disparities,
      :search,
      :perturbation,
      :paths,
      :coarse_paths,
      :fine_census,
      :band,
      :margin,
      :p1,
      :p2,
      :uniqueness,
      :warmup,
      :runs,
      :device,
      :truth_pixels,
      :scored_pixels,
      :matched_pixels,
      :scored,
      :matched,
      :bad,
      :bad_or_missing,
      :mean_absolute_error,
      :differing,
      :gpu_median_ms,
      :gpu_p90_ms
    ]

    csv = [
      Enum.map_join(columns, ",", &to_string/1)
      | Enum.map(rows, fn row -> Enum.map_join(columns, ",", &csv_cell(row[&1])) end)
    ]

    File.write!(path, Enum.join(csv, "\n") <> "\n")
    IO.puts("CSV: #{path}")
  end

  defp csv_cell(nil), do: ""
  defp csv_cell(cell), do: "\"" <> String.replace(to_string(cell), "\"", "\"\"") <> "\""

  defp percentile(values, fraction),
    do: values |> Enum.sort() |> Enum.at(max(0, ceil(length(values) * fraction) - 1))

  defp number(nil), do: "-"
  defp number(value) when is_integer(value), do: Integer.to_string(value)
  defp number(value), do: :erlang.float_to_binary(value, decimals: 3)
  defp percent(nil), do: "-"
  defp percent(value), do: :erlang.float_to_binary(value * 100, decimals: 2) <> "%"

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
