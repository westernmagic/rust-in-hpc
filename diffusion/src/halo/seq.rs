use ndarray::prelude::*;

#[inline]
pub fn update_halo<T>(field: &mut Array3<T>, num_halo: usize) where
    T: Copy
{
    let nx = field.shape()[0] - 2 * num_halo;
    let ny = field.shape()[1] - 2 * num_halo;
    let nz = field.shape()[2];

    // bottom edge (without corners)
    for k in 0..(nz - 1) {
        for j in 0..(num_halo - 1) {
            for i in num_halo..(nx + num_halo - 1) {
                field[[i, j, k]] = field[[i, j + ny, k]];
            }
        }
    }

    // top edge (excluding corners)
    for k in 0..(nz - 1) {
        for j in (ny + num_halo)..(ny + 2 * num_halo - 1) {
            for i in num_halo..(nx + num_halo - 1) {
                field[[i, j, k]] = field[[i, j - ny, k]];
            }
        }
    }

    // left edge (including corners)
    for k in 0..(nz - 1) {
        for j in 0..(ny + 2 * num_halo - 1) {
            for i in 0..(num_halo - 1) {
                field[[i, j, k]] = field[[i + nx, j, k]];
            }
        }
    }

    // right edge (including corners)
    for k in 0..(nz - 1) {
        for j in 0..(ny + 2 * num_halo - 1) {
            for i in (nx + num_halo)..(nx + 2 * num_halo - 1) {
                field[[i, j, k]] = field[[i - nx, j, k]];
            }
        }
    }
}
