#include "array.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>

// using cooperative group API leads to weird errors

__global__
void diffuse_laplap_cuda_kernel1(
	float *     const in_field,
	float *     const out_field,
	std::size_t const nx,
	std::size_t const ny,
	std::size_t const nz,
	std::size_t const num_halo,
	float       const alpha,
	std::size_t const num_iter,
	std::size_t const iter,
	float *     const tmp1_field
) {
	assert(in_field != nullptr);
	assert(out_field != nullptr);
	assert(nx > 0);
	assert(ny > 0);
	assert(nz > 0);
	assert(num_halo > 0);
	assert(!std::isnan(alpha));
	assert(num_iter > 0);
	assert(0 <= iter && iter < num_iter);
	assert(tmp1_field != nullptr);

	ArrayView3 in_field_(  in_field,   nx + 2 * num_halo, ny + 2 * num_halo, nz);
	ArrayView3 out_field_( out_field,  nx + 2 * num_halo, ny + 2 * num_halo, nz);
	ArrayView3 tmp1_field_(tmp1_field, nx + 2 * num_halo, ny + 2 * num_halo, nz);

	std::size_t const i = blockDim.x * blockIdx.x + threadIdx.x;
	std::size_t const j = blockDim.y * blockIdx.y + threadIdx.y;
	std::size_t const k = blockDim.z * blockIdx.z + threadIdx.z;

	if (
		0            <= k && k < nz                &&
		num_halo - 1 <= j && j < ny + num_halo + 1 &&
		num_halo - 1 <= i && i < nx + num_halo + 1
	) {
		tmp1_field_(i, j, k) =
			-4.0f * in_field_(i,     j,     k)
			+       in_field_(i - 1, j,     k)
			+       in_field_(i + 1, j,     k)
			+       in_field_(i,     j - 1, k)
			+       in_field_(i,     j + 1, k)
		;
	}
}

__global__
void diffuse_laplap_cuda_kernel2(
	float *     const in_field,
	float *     const out_field,
	std::size_t const nx,
	std::size_t const ny,
	std::size_t const nz,
	std::size_t const num_halo,
	float       const alpha,
	std::size_t const num_iter,
	std::size_t const iter,
	float *     const tmp1_field
) {
	assert(in_field != nullptr);
	assert(out_field != nullptr);
	assert(nx > 0);
	assert(ny > 0);
	assert(nz > 0);
	assert(num_halo > 0);
	assert(!std::isnan(alpha));
	assert(num_iter > 0);
	assert(0 <= iter && iter < num_iter);
	assert(tmp1_field != nullptr);

	ArrayView3 in_field_(  in_field,   nx + 2 * num_halo, ny + 2 * num_halo, nz);
	ArrayView3 out_field_( out_field,  nx + 2 * num_halo, ny + 2 * num_halo, nz);
	ArrayView3 tmp1_field_(tmp1_field, nx + 2 * num_halo, ny + 2 * num_halo, nz);

	std::size_t const i = blockDim.x * blockIdx.x + threadIdx.x;
	std::size_t const j = blockDim.y * blockIdx.y + threadIdx.y;
	std::size_t const k = blockDim.z * blockIdx.z + threadIdx.z;

	if (
		0        <= k && k < nz &&
		num_halo <= j && j < ny + num_halo &&
		num_halo <= i && i < nx + num_halo
	) {
		float laplap =
			-4.0f * tmp1_field_(i,     j,     k)
			+       tmp1_field_(i - 1, j,     k)
			+       tmp1_field_(i + 1, j,     k)
			+       tmp1_field_(i,     j - 1, k)
			+       tmp1_field_(i,     j + 1, k)
		;
	
		if (iter != num_iter - 1) {
			in_field_(i, j, k) = in_field_(i, j, k) - alpha * laplap;
		} else {
			out_field_(i, j, k) = in_field_(i, j, k) - alpha * laplap;
		}
	}
}

extern "C" __host__ void diffuse_laplap_cuda(
	float *     const in_field,
	float *     const out_field,
	std::size_t const nx,
	std::size_t const ny,
	std::size_t const nz,
	std::size_t const num_halo,
	float       const alpha,
	std::size_t const num_iter
) noexcept {
	assert(in_field != nullptr);
	assert(out_field != nullptr);
	assert(nx > 0);
	assert(ny > 0);
	assert(nz > 0);
	assert(num_halo > 0);
	assert(!std::isnan(alpha));
	assert(num_iter > 0);

	std::size_t const size = (nx + 2 * num_halo) * (ny + 2 * num_halo) * nz * sizeof(float);
	float * in_field_d   = nullptr;
	float * out_field_d  = nullptr;
	float * tmp1_field_d = nullptr;

	cudaMalloc((void**)&in_field_d,   size);
	assert(cudaGetLastError() == cudaSuccess);
	cudaMalloc((void**)&out_field_d,  size);
	assert(cudaGetLastError() == cudaSuccess);
	cudaMalloc((void**)&tmp1_field_d, size);
	assert(cudaGetLastError() == cudaSuccess);

	cudaMemcpy(in_field_d, in_field, size, cudaMemcpyHostToDevice);
	assert(cudaGetLastError() == cudaSuccess);

	dim3 const blockDim(1, 1, 1);
	dim3 const gridDim((nx + 2 * num_halo + blockDim.x - 1) / blockDim.x, (ny + 2 * num_halo + blockDim.y - 1) / blockDim.y, (nz + blockDim.z - 1) / blockDim.z);

	assert(0 < gridDim.x  && gridDim.x  <= (1 << 31) - 1);
	assert(0 < gridDim.y  && gridDim.y  <= 65535);
	assert(0 < gridDim.z  && gridDim.z  <= 65535);
	assert(0 < blockDim.x && blockDim.x <= 1024);
	assert(0 < blockDim.y && blockDim.y <= 1024);
	assert(0 < blockDim.z && blockDim.z <= 64);

	for (std::size_t iter = 0; iter < num_iter; ++iter) {
		// Intel/Cray generate invalid stubs, pass all 0 for gridDim, blockDim and bad pointers for args
		diffuse_laplap_cuda_kernel1<<<gridDim, blockDim>>>(in_field_d, out_field_d, nx, ny, nz, num_halo, alpha, num_iter, iter, tmp1_field_d);
		assert(cudaPeekAtLastError() == cudaSuccess);
		cudaDeviceSynchronize();
		assert(cudaGetLastError() == cudaSuccess);
		diffuse_laplap_cuda_kernel2<<<gridDim, blockDim>>>(in_field_d, out_field_d, nx, ny, nz, num_halo, alpha, num_iter, iter, tmp1_field_d);
		assert(cudaPeekAtLastError() == cudaSuccess);
		cudaDeviceSynchronize();
		assert(cudaGetLastError() == cudaSuccess);
	}

	cudaMemcpy(out_field, out_field_d, size, cudaMemcpyDeviceToHost);
	assert(cudaGetLastError() == cudaSuccess);

	cudaFree(tmp1_field_d);
	assert(cudaGetLastError() == cudaSuccess);
	tmp1_field_d = nullptr;
	cudaFree(out_field_d);
	assert(cudaGetLastError() == cudaSuccess);
	out_field_d = nullptr;
	cudaFree(in_field_d);
	assert(cudaGetLastError() == cudaSuccess);
	in_field_d = nullptr;
}
