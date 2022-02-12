//! Partitioner class
//!
//! Ported from m_partitioner.f (which in turn was ported from partitioner.py)

use anyhow::Result;
use ndarray::prelude::*;
use mpi::traits::*;
use mpi::datatype::{View, MutView};
use num_traits::identities::Zero;
use std::convert::{TryFrom, TryInto};
use std::ops::AddAssign;

/// 2-dimensional domain decomposition of a 3-dimensional computational grid among MPI ranks on a
/// communicator.
struct Partitioner<C> where C: AsCommunicator {
    comm: C,
    rank: usize,
    num_ranks: usize,
    num_halo: usize,
    size: [usize; 2],
    domains: Array2<usize>,
    shapes: Array2<usize>,
    domain: [usize; 4],
    shape: [usize; 3],
    max_shape: [usize; 3],
    periodic: [bool; 2],
    global_shape: [usize; 3],
}

impl<C> Partitioner<C> where C: AsCommunicator {
    pub fn new(comm: C, domain: [usize; 3], num_halo: usize, periodic: Option<[bool; 2]>) -> Result<Self> {
        let periodic = periodic.unwrap_or([true, true]);
        assert!(domain[0] > 0 && domain[1] > 0 && domain[2] > 0, "Invalid domain specification (negative size)");
        // assert!(num_halo >= 0, "Number of halo points must be zero or positive");

        let rank = usize::try_from(comm.as_communicator().rank())?;
        let num_ranks = usize::try_from(comm.as_communicator().size())?;
        let global_shape = [
            domain[0] + 2 * num_halo,
            domain[1] + 2 * num_halo,
            domain[2]
        ];

        let mut this = Self {
            comm,
            rank,
            num_ranks,
            num_halo,
            size: [0; 2],
            domains: Array::zeros((num_ranks, 4)),
            shapes: Array::zeros((num_ranks, 3)),
            domain: [0; 4],
            shape: [0; 3],
            max_shape: [0; 3],
            periodic,
            global_shape,
        };

        this.setup_grid();
        this.setup_domain(domain, num_halo);

        Ok(this)
    }

    /// Returns the MPI communicator used to setup the partitioner
    pub fn comm(&self) -> &C {
        &self.comm
    }

    /// Returns the number of halo points
    pub fn num_halo(&self) -> usize {
        self.num_halo
    }

    /// Returns the periodicity of all dimensions
    pub fn periodic(&self) -> [bool; 2] {
        self.periodic
    }

    /// Returns the rank of the current MPI worker
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the number of ranks that have been distributed by this partitioner
    pub fn num_ranks(&self) -> usize {
        self.num_ranks
    }

    /// Return the shape of a local field (including halo points)
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Returns the shape of a global field (including halo points)
    pub fn global_shape(&self) -> [usize; 3] {
        self.global_shape
    }

    /// Dimensions of the two-dimensional worker grid
    pub fn size(&self) -> [usize; 2] {
        self.size
    }

    /// Position of the current rank on two-dimensional worker grid
    pub fn position(&self) -> [usize; 2] {
        self.rank_to_position(self.rank)
    }

    /// Returns the rank of the left neighbor
    pub fn left(&self) -> usize {
        const position: [isize; 2] = [-1, 0];
        self.get_neighbor_rank(position)
    }

    /// Returns the rank of the right neighbor
    pub fn right(&self) -> usize {
        const position: [isize; 2] = [1, 0];
        self.get_neighbor_rank(position)
    }

    /// Returns the rank of the top neighbor
    pub fn top(&self) -> usize {
        const position: [isize; 2] = [0, 1];
        self.get_neighbor_rank(position)
    }

    /// Return the rank of the bottom neighbor
    pub fn bottom(&self) -> usize {
        const position: [isize; 2] = [0, -1];
        self.get_neighbor_rank(position)
    }

    /// Scatter a global field from a root rank to the workers
    pub fn scatter<T>(&self, field: &Array3<T>, root: Option<usize>) -> Array3<T> where
        T: Zero + Clone + Equivalence
    {
        let root = root.unwrap_or(0);
        assert!(field.shape() == self.global_shape, "Field does not have the correct shape");
        assert!(/* 0 <= root && */ root < self.num_ranks, "Root processor must be a valid rank");
        let root_process = self.comm.as_communicator().process_at_rank(root.try_into().unwrap());

        if self.num_ranks == 1 {
            return field.clone();
        }

        let datatype = T::equivalent_datatype();
        let mut recvbuf = Array3::<T>::zeros((self.max_shape[0], self.max_shape[1], self.max_shape[2]));
        let mut recvbufm = {
            let count = recvbuf.len().try_into().unwrap();
            let buffer = recvbuf.as_slice_memory_order_mut().unwrap();
            unsafe { MutView::with_count_and_datatype(
                buffer,
                count,
                &datatype
            )}
        };
        if self.rank == root {
            let mut sendbuf = Array4::<T>::zeros((self.max_shape[0], self.max_shape[1], self.max_shape[3], self.num_ranks));
            for rank in 0..self.num_ranks {
                let i_start = self.domains[[rank, 0]];
                let j_start = self.domains[[rank, 1]];
                let i_end   = self.domains[[rank, 2]];
                let j_end   = self.domains[[rank, 3]];

                sendbuf.slice_mut(s![..(i_end - i_start), ..(j_end - j_start), .., rank]).assign(&field.slice(s![i_start..i_end, j_start..j_end, ..]));
            }
            let sendbufm = unsafe { View::with_count_and_datatype(
                sendbuf.as_slice_memory_order().unwrap(),
                sendbuf.len().try_into().unwrap(),
                &datatype
            ) };
            root_process.scatter_into_root(&sendbufm, &mut recvbufm);
        } else {
            root_process.scatter_into(&mut recvbufm);
        }

        let i_start = self.domain[0];
        let j_start = self.domain[1];
        let i_end   = self.domain[2];
        let j_end   = self.domain[3];
        recvbuf.slice(s![..(i_end - i_start), ..(j_end - j_start), ..]).to_owned()
    }

    /// Gather a distributed field from workers to a single global field on a root rank
    pub fn gather<T>(&self, field: &Array3<T>, root: Option<isize>) -> Array3<T> where
        T: Zero + Clone + Equivalence
    {
        let root = root.unwrap_or(0);
        assert!(field.shape() == self.shape(), "Field does not have the correct shape");
        assert!(-1 <= root && root < self.num_ranks.try_into().unwrap(), "Root processor must be -1 (all) or a valid rank");

        if self.num_ranks == 1 {
            return field.clone();
        }

        let i_start = self.domain[0];
        let j_start = self.domain[1];
        let i_end   = self.domain[2];
        let j_end   = self.domain[3];

        let datatype = T::equivalent_datatype();
        let mut sendbuf = Array3::<T>::zeros((self.max_shape[0], self.max_shape[1], self.max_shape[2]));
        sendbuf.slice_mut(s![..(i_end - i_start), ..(j_end - j_start), ..]).assign(&field);
        let sendbufm = unsafe { View::with_count_and_datatype(
            sendbuf.as_slice_memory_order().unwrap(),
            sendbuf.len().try_into().unwrap(),
            &datatype
        )};

        if isize::try_from(self.rank).unwrap() == root || root == -1 {
            let mut recvbuf = Array4::<T>::zeros((self.max_shape[0], self.max_shape[1], self.max_shape[2], self.num_ranks));
            let mut recvbufm = {
                let count = recvbuf.len().try_into().unwrap();
                unsafe { MutView::with_count_and_datatype(
                    recvbuf.as_slice_memory_order_mut().unwrap(),
                    count,
                    &datatype
            )}};
            if root == -1 {
                self.comm.as_communicator().all_gather_into(&sendbufm, &mut recvbufm);
            } else if self.rank == root.try_into().unwrap() {
                let root_process = self.comm.as_communicator().process_at_rank(root.try_into().unwrap());
                root_process.gather_into_root(&sendbufm, &mut recvbufm);
            }

            let mut global_field = Array3::<T>::zeros((self.global_shape[0], self.global_shape[1], self.global_shape[2]));
            for rank in 0..self.num_ranks {
                let i_start = self.domains[[rank, 0]];
                let j_start = self.domains[[rank, 1]];
                let i_end   = self.domains[[rank, 2]];
                let j_end   = self.domains[[rank, 3]];

                global_field.slice_mut(s![i_start..i_end, j_start..j_end, ..]).assign(&recvbuf.slice(s![..(i_end - i_start), ..(j_end - j_start), .., rank]));
            }

            global_field
        } else {
            let root_process = self.comm.as_communicator().process_at_rank(root.try_into().unwrap());
            root_process.gather_into(&sendbufm);

            Array3::<T>::zeros((0, 0, 0))
        }
    }

    /// Return position of subdomain without halo on global domain
    pub fn compute_domain(&self) -> [usize; 4] {
        [
            self.domain[0] + self.num_halo,
            self.domain[1] + self.num_halo,
            self.domain[2] - self.num_halo,
            self.domain[3] - self.num_halo,
        ]
    }

    /// Distribute ranks onto a Cartesian grid of workers
    fn setup_grid(&mut self) {
        let ranks_x = (1..((self.num_ranks as f64).sqrt().floor() as usize)).rev().filter(|x| self.num_ranks % x == 0).take(1).collect::<Vec<_>>()[0];
        self.size = [ranks_x, self.num_ranks / ranks_x];
    }

    /// Get the rank ID of a neighboring rank at a certain offset relative to the current rank
    fn get_neighbor_rank(&self, offset: [isize; 2]) -> usize {
        let pos = self.rank_to_position(self.rank);
        let pos_offset = [
            self.cyclic_offset(pos[0], offset[0], self.size[0], self.periodic[0]).unwrap(),
            self.cyclic_offset(pos[1], offset[1], self.size[1], self.periodic[1]).unwrap()
        ];

        self.position_to_rank(pos_offset)
    }

    /// Add offset with cyclic boundary conditions
    fn cyclic_offset(&self, pos: usize, offset: isize, size: usize, periodic: bool) -> Option<usize> {
        // unimplemented!();
        let pos: isize = pos.try_into().unwrap();
        let size: isize = size.try_into().unwrap();
        let mut p: isize = pos + offset;
        if periodic {
            while p < 1 {
                p += size;
            }
            while p > size {
                p -= size;
            }
        }

        if p < 1 || p > size {
            None
        } else {
            Some(p.try_into().unwrap())
        }
    }

    /// Distribute the points of the computational grid onto the Cartesian grid of workers
    fn setup_domain(&mut self, shape: [usize; 3], num_halo: usize) {
        let size_x = Self::distribute_to_bins(shape[0], self.size[0]);
        let size_y = Self::distribute_to_bins(shape[1], self.size[1]);
        let size_z = self.size[2];

        let pos_x = Self::cumsum(&size_x, 1 + num_halo);
        let pos_y = Self::cumsum(&size_y, 1 + num_halo);

        for rank in 0..self.num_ranks {
            let pos = self.rank_to_position(rank);

            self.domains[[rank, 0]] = pos_x[pos[0]] - num_halo;
            self.domains[[rank, 1]] = pos_y[pos[1]] - num_halo;
            self.domains[[rank, 2]] = pos_x[pos[0] + 1] + num_halo - 1;
            self.domains[[rank, 3]] = pos_y[pos[1] + 1] + num_halo - 1;

            self.shapes[[rank, 0]] = self.domains[[rank, 2]] - self.domains[[rank, 0]];
            self.shapes[[rank, 1]] = self.domains[[rank, 3]] - self.domains[[rank, 1]];
            self.shapes[[rank, 2]] = size_z;
        }

        self.domain = self.domains.slice(s![self.rank, ..]).to_slice().unwrap().try_into().unwrap();
        self.shape = self.shapes.slice(s![self.rank, ..]).to_slice().unwrap().try_into().unwrap();

        self.max_shape = Self::find_max_shapes(&self.shapes);
    }

    /// Distribute a number of elements to a number of bins
    fn distribute_to_bins(num: usize, bins: usize) -> Vec<usize> {
        let n = num / bins;
        let mut bin_size = vec![n; bins];
        let extend = num - n * bins;
        if extend > 0 {
            let start_extend = bins / 2 - extend / 2;
            for i in start_extend..(start_extend + extend - 1) {
                bin_size[i] += 1usize;
            }
        }

        bin_size
    }

    /// Cumulative sum with an optional initial value
    fn cumsum<T>(array: &[T], initial_value: T) -> Vec<T> where
        T: AddAssign + Copy
    {
        array.iter().scan(initial_value, |cumsum, element| { *cumsum += *element; Some(*cumsum) }).collect()
    }

    /// Find maximum dimensions of subdomains across all ranks
    fn find_max_shapes(shapes: &Array2<usize>) -> [usize; 3] {
        assert!(shapes.shape()[1] == 3, "Wrong shapes size");
        let mut max_shape: [usize; 3] = shapes.slice(s![0, ..]).to_slice().unwrap().try_into().unwrap();
        for shape in 1..(shapes.shape()[0]) {
            max_shape[0] = max_shape[0].max(shapes[[shape, 0]]);
            max_shape[1] = max_shape[1].max(shapes[[shape, 1]]);
            max_shape[2] = max_shape[2].max(shapes[[shape, 2]]);
        }

        max_shape
    }

    /// Find position of rank on worker grid
    fn rank_to_position(&self, rank: usize) -> [usize; 2] {
        [rank % self.size[0] + 1, rank / self.size[1] + 1]
    }

    /// Find rank given a position on the worker grid
    fn position_to_rank(&self, pos: [usize; 2]) -> usize {
        (pos[1] - 1) * self.size[1] + (pos[0] - 1)
    }
}
