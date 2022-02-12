#include "array.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>

extern "C" void diffuse_laplap_openacc(
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

	std::size_t const size = (nx + 2 * num_halo) * (ny + 2 * num_halo) * nz;
	float * tmp1_field_ = new float[size];

	#pragma acc data \
		copyin(in_field[:size]) \
		copyout(out_field[:size]) \
		create(tmp1_field_[:size])
	for (std::size_t iter = 0; iter < num_iter; ++iter) {
		#pragma acc parallel loop gang \
			default(none) \
			firstprivate(nx, ny, nz, num_halo, alpha, num_iter) \
			present(in_field, out_field, tmp1_field_)
		for (std::size_t k = 0; k < nz; ++k) {
			ArrayView3 in_field_( in_field,    nx + 2 * num_halo, ny + 2 * num_halo, nz);
			ArrayView3 out_field_(out_field,   nx + 2 * num_halo, ny + 2 * num_halo, nz);
			ArrayView3 tmp1_field(tmp1_field_, nx + 2 * num_halo, ny + 2 * num_halo, nz);
			#pragma acc loop worker vector collapse(2)
			for (std::size_t j = num_halo - 1; j < ny + num_halo + 1; ++j) {
				for (std::size_t i = num_halo - 1; i < nx + num_halo + 1; ++i) {
					tmp1_field(i, j, k) =
						-4.0f * in_field_(i,     j,     k)
						+       in_field_(i - 1, j,     k)
						+       in_field_(i + 1, j,     k)
						+       in_field_(i,     j - 1, k)
						+       in_field_(i,     j + 1, k)
					;
				}
			}

			#pragma acc loop worker vector collapse(2)
			for (std::size_t j = num_halo; j < ny + num_halo; ++j) {
				for (std::size_t i = num_halo; i < nx + num_halo; ++i) {
					float laplap =
						-4.0f * tmp1_field(i,     j,     k)
						+       tmp1_field(i - 1, j,     k)
						+       tmp1_field(i + 1, j,     k)
						+       tmp1_field(i,     j - 1, k)
						+       tmp1_field(i,     j + 1, k)
					;
	
					if (iter != num_iter - 1) {
						in_field_(i, j, k) = in_field_(i, j, k) - alpha * laplap;
					} else {
						out_field_(i, j, k) = in_field_(i, j, k) - alpha * laplap;
					}
				}
			}
		}
		
	}
}
