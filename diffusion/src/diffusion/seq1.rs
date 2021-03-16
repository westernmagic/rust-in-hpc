use crate::halo::seq as update_halo;
use ndarray::prelude::*;
use std::mem::swap;
use std::ops::{Add, Mul};
use crate::num_halo;

/// `swap` instead of manual copy
#[inline]
pub fn apply_diffusion<'a, T>(in_field: &'a mut Array3<T>, out_field: &'a mut Array3<T>, alpha: T, num_iter: usize) where
    T: Mul<T, Output = T> + Add<T, Output = T> + From<f32> + Copy
{
    let nx = in_field.shape()[0] - 2 * num_halo;
    let ny = in_field.shape()[1] - 2 * num_halo;
    let nz = in_field.shape()[2];
    let alpha_20 = T::from(-20.0f32) * alpha + T::from(1.0f32);
    let alpha_08 = T::from(  8.0f32) * alpha;
    let alpha_02 = T::from( -2.0f32) * alpha;
    let alpha_01 = T::from( -1.0f32) * alpha;

    for iter in 0..num_iter {
        // update_halo(in_field, num_halo);

        for k in 0..nz {
            for j in num_halo..(ny + num_halo) {
                for i in num_halo..(nx + num_halo) {
                    out_field[[i, j, k]] =
                          alpha_20 * in_field[[i,     j,     k]]
                        + alpha_08 * in_field[[i - 1, j,     k]]
                        + alpha_08 * in_field[[i + 1, j,     k]]
                        + alpha_08 * in_field[[i,     j - 1, k]]
                        + alpha_08 * in_field[[i,     j + 1, k]]
                        + alpha_02 * in_field[[i - 1, j - 1, k]]
                        + alpha_02 * in_field[[i - 1, j + 1, k]]
                        + alpha_02 * in_field[[i + 1, j - 1, k]]
                        + alpha_02 * in_field[[i + 1, j + 1, k]]
                        + alpha_01 * in_field[[i - 2, j,     k]]
                        + alpha_01 * in_field[[i + 2, j,     k]]
                        + alpha_01 * in_field[[i,     j - 2, k]]
                        + alpha_01 * in_field[[i,     j + 2, k]];
                }
            }
        }

        swap(in_field, out_field);
    }

    // update_halo(out_field, num_halo);
}
