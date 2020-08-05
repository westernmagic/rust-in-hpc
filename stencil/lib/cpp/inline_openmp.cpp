#include "array.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>

extern "C" void diffuse_inline_openmp(
	float * in_field,
	float * out_field,
	std::size_t nx,
	std::size_t ny,
	std::size_t nz,
	std::size_t num_halo,
	float alpha,
	std::size_t num_iter
) noexcept {
	using std::swap;

	assert(in_field != nullptr);
	assert(out_field != nullptr);
	assert(nx > 0);
	assert(ny > 0);
	assert(nz > 0);
	assert(num_halo > 0);
	assert(!std::isnan(alpha));
	assert(num_iter > 0);

	ArrayView3 in_field_(in_field, nx + 2 * num_halo, ny + 2 * num_halo, nz);
	ArrayView3 out_field_(out_field, nx + 2 * num_halo, ny + 2 * num_halo, nz);

	float const alpha_20 = -20 * alpha + 1;
	float const alpha_08 =   8 * alpha;
	float const alpha_02 =  -2 * alpha;
	float const alpha_01 =  -1 * alpha;

	for (std::size_t iter = 0; iter < num_iter; ++iter) {
		// update_halo(in_field);
		#pragma omp parallel \
			default(none) \
			shared(iter, nx, ny, nz, num_halo, num_iter, alpha_20, alpha_08, alpha_02, alpha_01) \
			shared(in_field_, out_field_)
		#pragma omp for
		for (std::size_t k = 0; k < nz; ++k) {
			#pragma omp simd collapse(2)
			for (std::size_t j = num_halo; j < ny; ++j) {
				for (std::size_t i = num_halo; i < nx; ++i) {
					out_field_(i, j, k) =
						  alpha_20 * in_field_(i,     j,     k)
						+ alpha_08 * in_field_(i - 1, j,     k)
						+ alpha_08 * in_field_(i + 1, j,     k)
						+ alpha_08 * in_field_(i,     j - 1, k)
						+ alpha_08 * in_field_(i,     j + 1, k)
						+ alpha_02 * in_field_(i - 1, j - 1, k)
						+ alpha_02 * in_field_(i - 1, j + 1, k)
						+ alpha_02 * in_field_(i + 1, j - 1, k)
						+ alpha_02 * in_field_(i + 1, j + 1, k)
						+ alpha_01 * in_field_(i - 2, j,     k)
						+ alpha_01 * in_field_(i + 2, j,     k)
						+ alpha_01 * in_field_(i,     j - 2, k)
						+ alpha_01 * in_field_(i,     j + 2, k)
					;
				}
			}
		}

		swap(in_field_, out_field_);
	}

	// update_halo(out_field);
}
