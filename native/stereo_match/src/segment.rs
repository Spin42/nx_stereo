//! A run of consecutive `NxStereo.Stage`s computed natively: the
//! counterpart of `NxStereo.NxSegment`. It is
//! configured once with each stage's kind, its build settings and the
//! constants it shares with its Nx step, keeps what depends only on them
//! and the inputs it last saw (rectification maps, matching buffers) and
//! what stages carry from one pair to the next, and runs each pair through
//! every stage in order on named tensors.

use std::collections::HashMap;
use std::time::Instant;

use crate::depth::{self, Intrinsics};
use crate::filters::{self, Temporal, TemporalSettings};
use crate::measures::{self, Alignment, Quality};
use crate::rectify::{self, Camera, Map};
use crate::tensor::{f32_map, f32_map_mut, u8_image, Elements, Tensor, Tensors};
use crate::{Matcher, Parameters};

/// The sizes every stage works with: the rectified images' and the
/// submitted frames'.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub width: usize,
    pub height: usize,
    pub frame_width: usize,
    pub frame_height: usize,
}

/// A stage's build settings and constants, by name.
pub type Options = HashMap<String, f64>;
/// A stage's live inputs for one pair, its `params/2` values as `f32`.
pub type Inputs = HashMap<String, Vec<f32>>;
/// What a run returns.
#[derive(Debug)]
pub struct Outputs {
    /// The tensors the stages filled, by name.
    pub tensors: Vec<(String, Tensor)>,
    /// Each stage's time in microseconds.
    pub times_us: Vec<u32>,
}

enum Stage {
    Rectification {
        maps: Box<[Option<(Camera, Map)>; 2]>,
    },
    Sharpness,
    Disparity {
        disparities: usize,
        paths: u8,
        matcher: Box<Matcher>,
    },
    MedianFilter {
        kernel_size: usize,
    },
    SpeckleFilter {
        radius: usize,
    },
    RowAlignment(Alignment),
    TemporalFilter(Temporal),
    Depth,
    PointCloud {
        decimation: usize,
        depth_range: (f32, f32),
    },
    Quality(Quality),
}

pub struct Segment {
    geometry: Geometry,
    threads: usize,
    stages: Vec<Stage>,
}

/// The tensors each stage returns, once each. The depth in metres stays in
/// the segment, for the point cloud.
fn outputs(stage: &Stage) -> &'static [&'static str] {
    match stage {
        Stage::Rectification { .. } => &["left", "right"],
        Stage::Sharpness => &["sharpness_sums"],
        Stage::Disparity { .. }
        | Stage::MedianFilter { .. }
        | Stage::SpeckleFilter { .. }
        | Stage::TemporalFilter(_) => &["disparity"],
        Stage::RowAlignment(_) => &["alignment_sums"],
        Stage::Depth => &["depth_mm"],
        Stage::PointCloud { .. } => &["cloud_grid", "cloud_points"],
        Stage::Quality(_) => &["quality_sums"],
    }
}

fn option(options: &Options, kind: &str, name: &str) -> Result<f64, String> {
    options
        .get(name)
        .copied()
        .ok_or_else(|| format!("{kind} needs its {name} setting"))
}

/// Refuses the settings of `options` that `kind` does not take.
fn takes_only(options: &Options, kind: &str, names: &[&str]) -> Result<(), String> {
    let mut unknown: Vec<&str> = options
        .keys()
        .map(String::as_str)
        .filter(|name| !names.contains(name))
        .collect();
    if unknown.is_empty() {
        Ok(())
    } else {
        unknown.sort_unstable();
        Err(format!("{kind} has no setting {}", unknown.join(", ")))
    }
}

/// A whole-number setting from `lowest` to `highest`.
fn count_option(
    options: &Options,
    kind: &str,
    name: &str,
    lowest: usize,
    highest: usize,
) -> Result<usize, String> {
    let value = option(options, kind, name)?;
    if value.fract() == 0.0 && (lowest as f64..=highest as f64).contains(&value) {
        Ok(value as usize)
    } else {
        Err(format!(
            "{kind} {name} must be an integer from {lowest} to {highest}, got {value}"
        ))
    }
}

fn input<'a>(inputs: &'a Inputs, name: &str, length: usize) -> Result<&'a [f32], String> {
    match inputs.get(name) {
        Some(values) if values.len() == length => Ok(values),
        Some(values) => Err(format!(
            "{name} has {} values, expected {length}",
            values.len()
        )),
        None => Err(format!("no {name} among the stage's inputs")),
    }
}

fn scalar(inputs: &Inputs, name: &str) -> Result<f32, String> {
    input(inputs, name, 1).map(|values| values[0])
}

/// A penalty, an integer `NxStereo.sgm/3` takes as a byte.
fn penalty(inputs: &Inputs, name: &str) -> Result<u8, String> {
    let value = scalar(inputs, name)?;
    if value.fract() == 0.0 && (0.0..=255.0).contains(&value) {
        Ok(value as u8)
    } else {
        Err(format!(
            "{name} must be an integer from 0 to 255, got {value}"
        ))
    }
}

fn camera(inputs: &Inputs, side: &str) -> Result<Camera, String> {
    let values = |name: &str, length| input(inputs, &format!("{side}.{name}"), length);
    Ok(Camera {
        camera_matrix: values("k", 9)?.try_into().unwrap(),
        distortion: values("d", 5)?.try_into().unwrap(),
        inverse_pr: values("inverse_pr", 9)?.try_into().unwrap(),
    })
}

fn integer_option(
    options: &Options,
    kind: &str,
    name: &str,
    allowed: &[usize],
) -> Result<usize, String> {
    let value = option(options, kind, name)?;
    allowed
        .iter()
        .copied()
        .find(|&allowed| allowed as f64 == value)
        .ok_or_else(|| format!("{kind} {name} must be one of {allowed:?}, got {value}"))
}

impl Segment {
    /// A segment of `members`, each a stage kind (its name in the pipeline)
    /// and its options, run on `threads` threads.
    pub fn new(
        geometry: Geometry,
        members: &[(String, Options)],
        threads: usize,
    ) -> Result<Segment, String> {
        if geometry.width < 2 || geometry.height < 2 {
            return Err(format!("a segment works at 2x2 or more, got {geometry:?}"));
        }
        let stages = members
            .iter()
            .map(|(kind, options)| stage(&geometry, kind, options, threads))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Segment {
            geometry,
            threads: threads.max(1),
            stages,
        })
    }

    /// Runs `tensors` through every stage, `inputs` holding each stage's
    /// live inputs in order. Returns the tensors the stages filled, by name,
    /// and each stage's time in microseconds.
    pub fn run(&mut self, inputs: &[Inputs], mut tensors: Tensors) -> Result<Outputs, String> {
        if inputs.len() != self.stages.len() {
            return Err(format!(
                "{} stages, got the inputs of {}",
                self.stages.len(),
                inputs.len()
            ));
        }
        let mut filled: Vec<&'static str> = Vec::new();
        let mut times_us = Vec::with_capacity(self.stages.len());
        for (stage, inputs) in self.stages.iter_mut().zip(inputs) {
            let start = Instant::now();
            run_stage(stage, &self.geometry, self.threads, inputs, &mut tensors)?;
            times_us.push(u32::try_from(start.elapsed().as_micros()).unwrap_or(u32::MAX));
            for name in outputs(stage) {
                if !filled.contains(name) {
                    filled.push(name);
                }
            }
        }
        let outputs = filled
            .into_iter()
            .map(|name| {
                (
                    name.to_string(),
                    tensors.remove(name).expect("a stage fills its outputs"),
                )
            })
            .collect();
        Ok(Outputs {
            tensors: outputs,
            times_us,
        })
    }
}

/// The stage of `kind` with `options`, which must be exactly the ones it
/// takes.
fn stage(
    geometry: &Geometry,
    kind: &str,
    options: &Options,
    threads: usize,
) -> Result<Stage, String> {
    let (width, height) = (geometry.width, geometry.height);
    let pixels = width * height;
    let takes = |names: &[&str]| takes_only(options, kind, names);
    Ok(match kind {
        "rectification" => {
            takes(&[])?;
            if geometry.frame_width < width || geometry.frame_height < height {
                return Err(format!(
                    "rectification only shrinks: {}x{} frames are smaller than the {width}x{height} working size",
                    geometry.frame_width, geometry.frame_height
                ));
            }
            Stage::Rectification {
                maps: Box::new([None, None]),
            }
        }
        "sharpness" => {
            takes(&[])?;
            Stage::Sharpness
        }
        "disparity" => {
            takes(&["num_disparities", "paths"])?;
            let disparities = integer_option(options, kind, "num_disparities", &[64, 128])?;
            let paths = integer_option(options, kind, "paths", &[3, 8])? as u8;
            let parameters = Parameters {
                disparities,
                paths,
                p1: 8,
                p2: 64,
                uniqueness: 1.0,
            };
            parameters.check(width, height)?;
            Stage::Disparity {
                disparities,
                paths,
                matcher: Box::new(Matcher::new(threads)),
            }
        }
        "median_filter" => {
            takes(&["kernel_size"])?;
            Stage::MedianFilter {
                kernel_size: integer_option(options, kind, "kernel_size", &[3, 5])?,
            }
        }
        "speckle_filter" => {
            takes(&["radius"])?;
            Stage::SpeckleFilter {
                radius: count_option(options, kind, "radius", 1, width.min(height))?,
            }
        }
        "row_alignment" => {
            takes(&["bands", "reach", "row_step"])?;
            Stage::RowAlignment(Alignment {
                bands: count_option(options, kind, "bands", 1, width)?,
                reach: count_option(options, kind, "reach", 0, height)?,
                row_step: count_option(options, kind, "row_step", 1, height)?,
            })
        }
        "temporal_filter" => {
            takes(&["max_missed"])?;
            let max_missed = option(options, kind, "max_missed")? as f32;
            Stage::TemporalFilter(Temporal::new(pixels, max_missed))
        }
        "depth" => {
            takes(&[])?;
            Stage::Depth
        }
        "point_cloud" => {
            takes(&["decimation", "min_depth_m", "max_depth_m"])?;
            Stage::PointCloud {
                decimation: count_option(options, kind, "decimation", 1, width.min(height))?,
                depth_range: (
                    option(options, kind, "min_depth_m")? as f32,
                    option(options, kind, "max_depth_m")? as f32,
                ),
            }
        }
        "quality" => {
            takes(&[])?;
            Stage::Quality(Quality::new(pixels))
        }
        other => return Err(format!("no native implementation of {other}")),
    })
}

/// The map `slot` keeps for `camera`, built when the slot holds none or
/// one for another camera.
fn rectification_map<'a>(
    slot: &'a mut Option<(Camera, Map)>,
    camera: Camera,
    geometry: &Geometry,
) -> &'a Map {
    let kept = slot.take().filter(|(cached, _map)| *cached == camera);
    let (_camera, map) = slot.insert(kept.unwrap_or_else(|| {
        let map = Map::new(
            &camera,
            (geometry.width, geometry.height),
            (geometry.frame_width, geometry.frame_height),
        );
        (camera, map)
    }));
    map
}

fn run_stage(
    stage: &mut Stage,
    geometry: &Geometry,
    threads: usize,
    inputs: &Inputs,
    tensors: &mut Tensors,
) -> Result<(), String> {
    let (width, height) = (geometry.width, geometry.height);
    let size = [height, width];
    let image = |elements: Elements| Tensor::new(size.to_vec(), elements);
    match stage {
        Stage::Rectification { maps } => {
            let frame_shape = [geometry.frame_height, geometry.frame_width];
            let [left_slot, right_slot] = &mut **maps;
            let left_map = rectification_map(left_slot, camera(inputs, "left")?, geometry);
            let right_map = rectification_map(right_slot, camera(inputs, "right")?, geometry);
            let left_frame = u8_image(tensors, "left_frame", frame_shape)?;
            let right_frame = u8_image(tensors, "right_frame", frame_shape)?;
            let (mut left, mut right) = (vec![0; width * height], vec![0; width * height]);
            rectify::rectify_pair(
                [left_map, right_map],
                [left_frame, right_frame],
                geometry.frame_width,
                [&mut left, &mut right],
                threads,
            );
            tensors.insert("left".into(), image(Elements::U8(left)));
            tensors.insert("right".into(), image(Elements::U8(right)));
        }
        Stage::Sharpness => {
            let sums = measures::sharpness_sums(
                u8_image(tensors, "left", size)?,
                u8_image(tensors, "right", size)?,
                width,
            );
            tensors.insert(
                "sharpness_sums".into(),
                Tensor::new(vec![sums.len()], Elements::F32(sums.to_vec())),
            );
        }
        Stage::Disparity {
            disparities,
            paths,
            matcher,
        } => {
            let parameters = Parameters {
                disparities: *disparities,
                paths: *paths,
                p1: penalty(inputs, "p1")?,
                p2: penalty(inputs, "p2")?,
                uniqueness: scalar(inputs, "uniqueness")?,
            };
            parameters.check(width, height)?;
            // As the Nx stage, a match at disparity 0, which has no depth, is unmatched.
            let disparity = matcher
                .disparity(
                    u8_image(tensors, "left", size)?,
                    u8_image(tensors, "right", size)?,
                    width,
                    height,
                    &parameters,
                )
                .into_iter()
                .map(|disparity| if disparity > 0.0 { disparity } else { -1.0 })
                .collect();
            tensors.insert("disparity".into(), image(Elements::F32(disparity)));
        }
        Stage::MedianFilter { kernel_size } => {
            let filtered = filters::median(
                f32_map(tensors, "disparity", size)?,
                width,
                height,
                *kernel_size,
                threads,
            );
            tensors.insert("disparity".into(), image(Elements::F32(filtered)));
        }
        Stage::SpeckleFilter { radius } => {
            let filtered = filters::despeckle(
                f32_map(tensors, "disparity", size)?,
                width,
                height,
                scalar(inputs, "min_support")?,
                scalar(inputs, "max_diff")?,
                *radius,
                threads,
            );
            tensors.insert("disparity".into(), image(Elements::F32(filtered)));
        }
        Stage::RowAlignment(alignment) => {
            let sums = measures::alignment_sums(
                u8_image(tensors, "left", size)?,
                u8_image(tensors, "right", size)?,
                f32_map(tensors, "disparity", size)?,
                width,
                height,
                alignment,
            );
            tensors.insert(
                "alignment_sums".into(),
                Tensor::new(
                    vec![alignment.bands, alignment.planes()],
                    Elements::F32(sums),
                ),
            );
        }
        Stage::TemporalFilter(temporal) => {
            let settings = TemporalSettings {
                alpha: scalar(inputs, "alpha")?,
                delta: scalar(inputs, "delta")?,
                hold: scalar(inputs, "hold")?,
            };
            temporal.filter(f32_map_mut(tensors, "disparity", size)?, &settings);
        }
        Stage::Depth => {
            let (depth_mm, depth_m) = depth::depth(
                f32_map(tensors, "disparity", size)?,
                width,
                scalar(inputs, "focal_length_px")?,
                scalar(inputs, "baseline_m")?,
                threads,
            );
            tensors.insert("depth_mm".into(), image(Elements::U16(depth_mm)));
            tensors.insert("depth_m".into(), image(Elements::F32(depth_m)));
        }
        Stage::PointCloud {
            decimation,
            depth_range,
        } => {
            let intrinsics = Intrinsics {
                focal_length_px: scalar(inputs, "focal_length_px")?,
                cx: scalar(inputs, "cx")?,
                cy: scalar(inputs, "cy")?,
            };
            let (points, kept) = depth::cloud(
                f32_map(tensors, "depth_m", size)?,
                width,
                height,
                *decimation,
                &intrinsics,
                *depth_range,
            );
            let grid = vec![height / *decimation, width / *decimation, 3];
            tensors.insert(
                "cloud_grid".into(),
                Tensor::new(grid, Elements::F32(points)),
            );
            tensors.insert(
                "cloud_points".into(),
                Tensor::new(vec![], Elements::S32(vec![kept as i32])),
            );
        }
        Stage::Quality(quality) => {
            let sums = quality.sums(f32_map(tensors, "disparity", size)?, width);
            tensors.insert(
                "quality_sums".into(),
                Tensor::new(vec![5], Elements::F32(sums.to_vec())),
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> Geometry {
        Geometry {
            width: 32,
            height: 24,
            frame_width: 32,
            frame_height: 24,
        }
    }

    fn options(values: &[(&str, f64)]) -> Options {
        values
            .iter()
            .map(|&(name, value)| (name.to_string(), value))
            .collect()
    }

    /// Why a segment of the one stage `kind` with settings `values` is refused.
    fn refused(kind: &str, values: &[(&str, f64)]) -> String {
        Segment::new(geometry(), &[(kind.to_string(), options(values))], 2)
            .err()
            .unwrap()
    }

    #[test]
    fn refuses_unknown_kinds_and_settings() {
        assert!(refused("decode", &[]).contains("no native implementation of decode"));
        assert!(refused("median_filter", &[("kernel_size", 7.0)]).contains("[3, 5]"));
        assert!(refused("disparity", &[("num_disparities", 64.0)]).contains("paths"));
        assert!(refused("point_cloud", &[("decimation", 0.0)]).contains("decimation"));
        assert_eq!(
            refused(
                "temporal_filter",
                &[("max_missed", 255.0), ("hold", 2.0), ("alpha", 0.4)]
            ),
            "temporal_filter has no setting alpha, hold"
        );
        assert_eq!(
            refused("depth", &[("baseline_m", 0.1)]),
            "depth has no setting baseline_m"
        );
    }

    #[test]
    fn needs_the_constants_each_stage_shares_with_its_nx_step() {
        assert_eq!(
            refused("temporal_filter", &[]),
            "temporal_filter needs its max_missed setting"
        );
        assert_eq!(
            refused("speckle_filter", &[]),
            "speckle_filter needs its radius setting"
        );
        assert_eq!(
            refused("point_cloud", &[("decimation", 2.0), ("min_depth_m", 0.05)]),
            "point_cloud needs its max_depth_m setting"
        );
        assert_eq!(
            refused(
                "row_alignment",
                &[("bands", 0.0), ("reach", 2.0), ("row_step", 4.0)]
            ),
            "row_alignment bands must be an integer from 1 to 32, got 0"
        );
    }

    #[test]
    fn refuses_frames_smaller_than_the_working_size_to_rectify() {
        let rectification = [("rectification".to_string(), Options::new())];
        let frames = |frame_width, frame_height| Geometry {
            frame_width,
            frame_height,
            ..geometry()
        };
        assert_eq!(
            Segment::new(frames(32, 12), &rectification, 1)
                .err()
                .unwrap(),
            "rectification only shrinks: 32x12 frames are smaller than the 32x24 working size"
        );
        assert!(Segment::new(frames(16, 24), &rectification, 1).is_err());
        assert!(Segment::new(frames(32, 24), &rectification, 1).is_ok());
        assert!(Segment::new(frames(64, 48), &rectification, 1).is_ok());
        let depth = [("depth".to_string(), Options::new())];
        assert!(Segment::new(frames(16, 12), &depth, 1).is_ok());
    }

    #[test]
    fn returns_each_filled_tensor_once_with_its_last_value() {
        let geometry = geometry();
        let members = [
            (
                "median_filter".to_string(),
                options(&[("kernel_size", 3.0)]),
            ),
            ("depth".to_string(), Options::new()),
            ("speckle_filter".to_string(), options(&[("radius", 3.0)])),
        ];
        let mut segment = Segment::new(geometry, &members, 2).unwrap();
        let inputs: Vec<Inputs> = vec![
            Inputs::new(),
            [("focal_length_px", 100.0), ("baseline_m", 0.1)]
                .map(|(name, value)| (name.to_string(), vec![value]))
                .into(),
            [("min_support", 49.0), ("max_diff", 1.0)]
                .map(|(name, value)| (name.to_string(), vec![value]))
                .into(),
        ];
        let disparity = Tensor::new(vec![24, 32], Elements::F32(vec![10.0; 32 * 24]));
        let tensors = Tensors::from([("disparity".to_string(), disparity)]);
        let Outputs {
            tensors: outputs,
            times_us,
        } = segment.run(&inputs, tensors).unwrap();
        let names: Vec<&str> = outputs.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["disparity", "depth_mm"]);
        assert_eq!(times_us.len(), 3);
        assert_eq!(outputs[0].1.elements, Elements::F32(vec![-1.0; 32 * 24]));
        assert_eq!(outputs[1].1.elements, Elements::U16(vec![1000; 32 * 24]));
    }

    #[test]
    fn the_disparity_reads_a_match_at_disparity_0_as_unmatched() {
        let geometry = geometry();
        let (width, height) = (geometry.width, geometry.height);
        let members = [(
            "disparity".to_string(),
            options(&[("num_disparities", 64.0), ("paths", 3.0)]),
        )];
        let mut segment = Segment::new(geometry, &members, 2).unwrap();
        let parameters = crate::tests::parameters(64, 3);
        let inputs: Vec<Inputs> = vec![[
            ("p1", f32::from(parameters.p1)),
            ("p2", f32::from(parameters.p2)),
            ("uniqueness", parameters.uniqueness),
        ]
        .map(|(name, value)| (name.to_string(), vec![value]))
        .into()];
        // The upper half the same on both sides, at disparity 0; the lower half at 3.
        let left = crate::tests::noise(width, height, 4);
        let right: Vec<u8> = (0..width * height)
            .map(|i| match (i % width, i / width) {
                (_, y) if y < height / 2 => left[i],
                (x, y) => left[y * width + (x + 3).min(width - 1)],
            })
            .collect();
        let matched = Matcher::new(1).disparity(&left, &right, width, height, &parameters);
        assert!(matched.contains(&0.0) && matched.contains(&3.0));
        let tensors = Tensors::from([("left", left), ("right", right)].map(|(side, image)| {
            let image = Tensor::new(vec![height, width], Elements::U8(image));
            (side.to_string(), image)
        }));
        let outputs = segment.run(&inputs, tensors).unwrap().tensors;
        let unmatched_at_0 = matched
            .iter()
            .map(|&disparity| if disparity > 0.0 { disparity } else { -1.0 })
            .collect();
        assert_eq!(outputs[0].1.elements, Elements::F32(unmatched_at_0));
    }

    #[test]
    fn says_which_input_is_missing_or_of_the_wrong_shape() {
        let members = [("depth".to_string(), Options::new())];
        let mut segment = Segment::new(geometry(), &members, 1).unwrap();
        let wrong = Tensor::new(vec![2, 2], Elements::F32(vec![0.0; 4]));
        let error = segment
            .run(
                &[Inputs::new()],
                Tensors::from([("disparity".to_string(), wrong)]),
            )
            .unwrap_err();
        assert!(error.contains("expected a [24, 32] f32 map"), "{error}");
    }

    #[test]
    fn rectifies_with_the_maps_of_the_cameras_it_is_given() {
        let geometry = geometry();
        let (width, height) = (geometry.width, geometry.height);
        let members = [("rectification".to_string(), Options::new())];
        let mut segment = Segment::new(geometry, &members, 2).unwrap();
        let camera = |cx: f32| Camera {
            camera_matrix: [100.0, 0.0, cx, 0.0, 100.0, 12.0, 0.0, 0.0, 1.0],
            distortion: [0.01, 0.0, 0.0, 0.0, 0.0],
            inverse_pr: [0.01, 0.0, -0.16, 0.0, 0.01, -0.12, 0.0, 0.0, 1.0],
        };
        let inputs = |cameras: [&Camera; 2]| -> Vec<Inputs> {
            vec![["left", "right"]
                .into_iter()
                .zip(cameras)
                .flat_map(|(side, camera)| {
                    [
                        (format!("{side}.k"), camera.camera_matrix.to_vec()),
                        (format!("{side}.d"), camera.distortion.to_vec()),
                        (format!("{side}.inverse_pr"), camera.inverse_pr.to_vec()),
                    ]
                })
                .collect()]
        };
        let frames = [
            crate::tests::noise(width, height, 1),
            crate::tests::noise(width, height, 2),
        ];
        let image = |pixels: Vec<u8>| Tensor::new(vec![height, width], Elements::U8(pixels));
        // The right camera changes on the second pair, the left one does not.
        for cameras in [[camera(16.0), camera(16.0)], [camera(16.0), camera(18.5)]] {
            let entries = Tensors::from([
                ("left_frame".to_string(), image(frames[0].clone())),
                ("right_frame".to_string(), image(frames[1].clone())),
            ]);
            let cameras = [&cameras[0], &cameras[1]];
            let outputs = segment.run(&inputs(cameras), entries).unwrap();
            let maps = cameras.map(|camera| Map::new(camera, (width, height), (width, height)));
            let (mut left, mut right) = (vec![0; width * height], vec![0; width * height]);
            rectify::rectify_pair(
                [&maps[0], &maps[1]],
                [&frames[0], &frames[1]],
                width,
                [&mut left, &mut right],
                1,
            );
            assert_eq!(
                outputs.tensors,
                [
                    ("left".to_string(), image(left)),
                    ("right".to_string(), image(right))
                ]
            );
        }
    }

    #[test]
    fn filters_the_disparity_against_the_previous_pair() {
        let members = [(
            "temporal_filter".to_string(),
            options(&[("max_missed", 255.0)]),
        )];
        let mut segment = Segment::new(geometry(), &members, 1).unwrap();
        let inputs: Vec<Inputs> = vec![[("alpha", 0.5), ("delta", 2.0), ("hold", 1.0)]
            .map(|(name, value)| (name.to_string(), vec![value]))
            .into()];
        let settings = TemporalSettings {
            alpha: 0.5,
            delta: 2.0,
            hold: 1.0,
        };
        let mut temporal = Temporal::new(32 * 24, 255.0);
        let map = |values: Vec<f32>| Tensor::new(vec![24, 32], Elements::F32(values));
        let second: Vec<f32> = (0..32 * 24)
            .map(|i| if i % 3 == 0 { -1.0 } else { 11.0 })
            .collect();
        for pair in [vec![10.0; 32 * 24], second] {
            let entries = Tensors::from([("disparity".to_string(), map(pair.clone()))]);
            let outputs = segment.run(&inputs, entries).unwrap();
            let mut want = pair;
            temporal.filter(&mut want, &settings);
            assert_eq!(outputs.tensors, [("disparity".to_string(), map(want))]);
        }
    }
}
