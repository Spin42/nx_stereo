//! The named tensors a segment reads and fills: the counterpart of the Nx
//! tensors of `NxStereo.NxSegment`, row-major, little-endian on the
//! wire.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Deref;

#[derive(Clone, Debug, PartialEq)]
pub enum Elements {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
    S32(Vec<i32>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub elements: Elements,
}

pub type Tensors = HashMap<String, Tensor>;

/// The wire codes of the element types, as `NxStereo.NativeSegment`
/// maps them to Nx types.
const TYPE_U8: u8 = 0;
const TYPE_U16: u8 = 1;
const TYPE_F32: u8 = 2;
const TYPE_S32: u8 = 3;

/// The size on the wire of one element of type `type_code`.
pub fn element_bytes(type_code: u8) -> Result<usize, String> {
    match type_code {
        TYPE_U8 => Ok(1),
        TYPE_U16 => Ok(2),
        TYPE_F32 | TYPE_S32 => Ok(4),
        other => Err(format!("unknown element type {other}")),
    }
}

impl Tensor {
    pub fn new(shape: Vec<usize>, elements: Elements) -> Tensor {
        Tensor { shape, elements }
    }

    pub fn type_code(&self) -> u8 {
        match self.elements {
            Elements::U8(_) => TYPE_U8,
            Elements::U16(_) => TYPE_U16,
            Elements::F32(_) => TYPE_F32,
            Elements::S32(_) => TYPE_S32,
        }
    }

    /// The tensor from its wire bytes, which must hold exactly `shape`'s
    /// elements.
    pub fn from_bytes(type_code: u8, shape: Vec<usize>, bytes: &[u8]) -> Result<Tensor, String> {
        let elements: usize = shape.iter().product();
        let element_bytes = element_bytes(type_code)?;
        if bytes.len() != elements * element_bytes {
            return Err(format!(
                "a {shape:?} tensor of {element_bytes}-byte elements takes {} bytes, got {}",
                elements * element_bytes,
                bytes.len()
            ));
        }
        let elements = match type_code {
            TYPE_U8 => Elements::U8(bytes.to_vec()),
            TYPE_U16 => Elements::U16(from_little_endian(bytes, u16::from_le_bytes)),
            TYPE_F32 => Elements::F32(from_little_endian(bytes, f32::from_le_bytes)),
            _ => Elements::S32(from_little_endian(bytes, i32::from_le_bytes)),
        };
        Ok(Tensor { shape, elements })
    }

    /// The elements' wire bytes, borrowed where the CPU is little-endian.
    pub fn bytes(&self) -> Cow<'_, [u8]> {
        match &self.elements {
            Elements::U8(values) => Cow::Borrowed(values),
            Elements::U16(values) => little_endian_bytes(values, u16::to_le_bytes),
            Elements::F32(values) => little_endian_bytes(values, f32::to_le_bytes),
            Elements::S32(values) => little_endian_bytes(values, i32::to_le_bytes),
        }
    }
}

/// The values of `N` little-endian bytes each that `bytes` holds.
pub fn from_little_endian<T, const N: usize>(
    bytes: &[u8],
    from_le_bytes: fn([u8; N]) -> T,
) -> Vec<T> {
    bytes
        .chunks_exact(N)
        .map(|chunk| from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// `values` as little-endian bytes: their memory, on a little-endian CPU.
#[cfg(target_endian = "little")]
fn little_endian_bytes<T: Copy, const N: usize>(
    values: &[T],
    _to_le_bytes: fn(T) -> [u8; N],
) -> Cow<'_, [u8]> {
    // SAFETY: `T` is a plain number of `N` bytes, whose memory is its little-endian bytes.
    Cow::Borrowed(unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    })
}

#[cfg(not(target_endian = "little"))]
fn little_endian_bytes<T: Copy, const N: usize>(
    values: &[T],
    to_le_bytes: fn(T) -> [u8; N],
) -> Cow<'_, [u8]> {
    Cow::Owned(
        values
            .iter()
            .flat_map(|&value| to_le_bytes(value))
            .collect(),
    )
}

/// The `u8` image `name`, of `shape`.
pub fn u8_image<'a>(
    tensors: &'a Tensors,
    name: &str,
    shape: [usize; 2],
) -> Result<&'a [u8], String> {
    shaped(
        tensors.get(name),
        name,
        shape,
        "u8 image",
        |tensor| match &tensor.elements {
            Elements::U8(values) => Some(&values[..]),
            _ => None,
        },
    )
}

/// The `f32` map `name`, of `shape`.
pub fn f32_map<'a>(
    tensors: &'a Tensors,
    name: &str,
    shape: [usize; 2],
) -> Result<&'a [f32], String> {
    shaped(
        tensors.get(name),
        name,
        shape,
        "f32 map",
        |tensor| match &tensor.elements {
            Elements::F32(values) => Some(&values[..]),
            _ => None,
        },
    )
}

/// The `f32` map `name`, of `shape`, to change in place.
pub fn f32_map_mut<'a>(
    tensors: &'a mut Tensors,
    name: &str,
    shape: [usize; 2],
) -> Result<&'a mut [f32], String> {
    shaped(
        tensors.get_mut(name),
        name,
        shape,
        "f32 map",
        |tensor| match &mut tensor.elements {
            Elements::F32(values) => Some(&mut values[..]),
            _ => None,
        },
    )
}

/// The `values` of `tensor`, found as `name`, when it is the `shape`
/// tensor of the type `kind` names.
fn shaped<R: Deref<Target = Tensor>, V>(
    tensor: Option<R>,
    name: &str,
    shape: [usize; 2],
    kind: &str,
    values: impl FnOnce(R) -> Option<V>,
) -> Result<V, String> {
    let tensor = tensor.ok_or_else(|| format!("no {name} among the segment's tensors"))?;
    let type_code = tensor.type_code();
    let refusal = |found: &[usize]| {
        format!("{name} is a {found:?} tensor of type {type_code}, expected a {shape:?} {kind}")
    };
    if tensor.shape[..] != shape {
        return Err(refusal(&tensor.shape));
    }
    values(tensor).ok_or_else(|| refusal(&shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_its_bytes() {
        for tensor in [
            Tensor::new(vec![2, 2], Elements::U8(vec![1, 2, 3, 255])),
            Tensor::new(vec![3], Elements::U16(vec![0, 1000, 65535])),
            Tensor::new(vec![1, 2], Elements::F32(vec![-1.0, 2.5])),
            Tensor::new(vec![], Elements::S32(vec![-7])),
        ] {
            let bytes = tensor.bytes();
            let elements: usize = tensor.shape.iter().product();
            assert_eq!(
                bytes.len(),
                elements * element_bytes(tensor.type_code()).unwrap()
            );
            let back = Tensor::from_bytes(tensor.type_code(), tensor.shape.clone(), &bytes);
            assert_eq!(back, Ok(tensor));
        }
    }

    #[test]
    fn refuses_bytes_that_do_not_fill_the_shape() {
        assert!(Tensor::from_bytes(2, vec![2, 2], &[0; 15])
            .unwrap_err()
            .contains("takes 16 bytes"));
        assert!(Tensor::from_bytes(9, vec![1], &[0])
            .unwrap_err()
            .contains("unknown element type"));
    }
}
