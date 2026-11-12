defmodule Bench.Scene do
  @moduledoc false
  alias NxStereo.{NxSegment, Packing, Rectify, Segment, Stage}

  # A textured pair at frame size whose right frame sees the left one shifted, and the carried state.
  def inputs({frame_height, frame_width}, {height, width}, opts) do
    shift = Keyword.get(opts, :shift, div(opts[:disparities], 4) * div(frame_width, width))

    {texture, _key} =
      Nx.Random.randint(Nx.Random.key(Keyword.get(opts, :seed, 1)), 0, 256,
        shape: {frame_height, frame_width + shift},
        type: :s32
      )

    texture = Nx.as_type(texture, :u8)
    left = texture[[.., 0..(frame_width - 1)]]
    right = texture[[.., shift..(frame_width + shift - 1)]]

    frame = fn image ->
      if opts[:packed],
        do: Packing.pack(Nx.to_binary(image), frame_width, frame_height),
        else: image
    end

    {fx, cx, cy} = {width * 0.6, width / 2, height / 2}
    k = [fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0]
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    p = [fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    camera = Rectify.camera(k, [-0.05, 0.01], identity, p)
    unmatched = Nx.broadcast(Nx.f32(-1.0), {height, width})

    %{
      left_frame: frame.(left),
      right_frame: frame.(right),
      camera: camera,
      previous: unmatched,
      missed: Nx.broadcast(Nx.f32(0.0), {height, width}),
      settings: Nx.tensor([fx, cx, cy, 0.2], type: :f32),
      left_image: left,
      right_image: right,
      geometry: %{
        focal_length_px: fx,
        baseline_m: 0.2,
        principal_point: {cx, cy},
        left_camera: camera,
        right_camera: camera
      }
    }
  end

  # Compile the complete application stage chain with its normal published outputs.
  def segment(inputs, {height, width}, {frame_height, frame_width}, opts, gpu) do
    disparity =
      [num_disparities: opts[:disparities], paths: opts[:paths], search: opts[:search]] ++
        if(opts[:search] == "coarse_to_fine",
          do: [
            coarse_paths: opts[:coarse_paths],
            band: opts[:band],
            margin: opts[:margin],
            fine_census: Keyword.get(opts, :fine_census, :sparse_9x7)
          ],
          else: []
        )

    stages = [
      Stage.Rectification,
      Stage.Sharpness,
      {Stage.Disparity, disparity},
      Stage.SpeckleFilter,
      Stage.RowAlignment,
      Stage.TemporalFilter,
      Stage.Depth,
      {Stage.PointCloud, decimation: Keyword.get(opts, :cloud_decimation, 4)},
      Stage.Quality
    ]

    geometry =
      Map.merge(inputs.geometry, %{
        width: width,
        height: height,
        frame_width: frame_width,
        frame_height: frame_height
      })

    {:ok, members} = Segment.members(stages, geometry)
    options = [byte_storage: not opts[:packed]] ++ gpu
    NxSegment.build(members, geometry, defn_options: options)
  end
end
