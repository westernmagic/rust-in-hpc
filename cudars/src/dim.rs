#![cfg(target_os = "cuda")]

use core::arch::nvptx::*;
use core::convert::TryInto;

pub struct GridDim {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl GridDim {
    #[inline]
    pub fn new() -> Self {
        Self {
            x: unsafe { _grid_dim_x().try_into().unwrap() },
            y: unsafe { _grid_dim_y().try_into().unwrap() },
            z: unsafe { _grid_dim_z().try_into().unwrap() },
        }
    }
}

pub struct BlockIdx {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl BlockIdx {
    #[inline]
    pub fn new() -> Self {
        Self {
            x: unsafe { _block_idx_x().try_into().unwrap() },
            y: unsafe { _block_idx_y().try_into().unwrap() },
            z: unsafe { _block_idx_z().try_into().unwrap() },
        }
    }
}

pub struct BlockDim {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl BlockDim {
    #[inline]
    pub fn new() -> Self {
        Self {
            x: unsafe { _block_dim_x().try_into().unwrap() },
            y: unsafe { _block_dim_y().try_into().unwrap() },
            z: unsafe { _block_dim_z().try_into().unwrap() },
        }
    }
}

pub struct ThreadIdx {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl ThreadIdx {
    #[inline]
    pub fn new() -> Self {
        Self {
            x: unsafe { _thread_idx_x().try_into().unwrap() },
            y: unsafe { _thread_idx_y().try_into().unwrap() },
            z: unsafe { _thread_idx_z().try_into().unwrap() },
        }
    }
}
