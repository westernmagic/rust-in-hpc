#include "array.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>

extern "C" void diffuse_laplap_openmp(
	float * in_field,
	float * out_field,
	std::size_t nx,
	std::size_t ny,
	std::size_t nz,
	std::size_t num_halo,
	float alpha,
	std::size_t num_iter
) noexcept {
	assert(in_field != nullptr);
	assert(out_field != nullptr);
	assert(nx > 0);
	assert(ny > 0);
	assert(nz > 0);
	assert(num_halo > 0);
	assert(!std::isnan(alpha));
	assert(num_iter > 0);

	ArrayView3 in_field_( in_field,  nx + 2 * num_halo, ny + 2 * num_halo, nz);
	ArrayView3 out_field_(out_field, nx + 2 * num_halo, ny + 2 * num_halo, nz);
	Array2 tmp1_field = Array2::zeros(nx + 2 * num_halo, ny + 2 * num_halo);

	for (std::size_t iter = 0; iter < num_iter; ++iter) {
		// update_halo(in_field);
		#pragma omp parallel \
			default(none) \
			shared(iter, nx, ny, nz, num_halo, num_iter, alpha) \
			shared(in_field_, out_field_) \
			firstprivate(tmp1_field)
		#pragma omp for
		for (std::size_t k = 0; k < nz; ++k) {
			#pragma omp simd collapse(2)
			for (std::size_t j = num_halo - 1; j < ny + num_halo + 1; ++j) {
				for (std::size_t i = num_halo - 1; i < nx + num_halo + 1; ++i) {
					tmp1_field(i, j) =
						-4.0f * in_field_(i,     j,     k)
						+       in_field_(i - 1, j,     k)
						+       in_field_(i + 1, j,     k)
						+       in_field_(i,     j - 1, k)
						+       in_field_(i,     j + 1, k)
					;
				}
			}

			#pragma omp simd collapse(2)
			for (std::size_t j = num_halo; j < ny + num_halo; ++j) {
				for (std::size_t i = num_halo; i < nx + num_halo; ++i) {
					float laplap =
						-4.0f * tmp1_field(i,     j    )
						+       tmp1_field(i - 1, j    )
						+       tmp1_field(i + 1, j    )
						+       tmp1_field(i,     j - 1)
						+       tmp1_field(i,     j + 1)
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

	// update_halo(out_field);
}
