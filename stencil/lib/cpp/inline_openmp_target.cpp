#include "array.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>

extern "C" void diffuse_inline_openmp_target(
	float *     const in_field,
	float *     const out_field,
	std::size_t const nx,
	std::size_t const ny,
	std::size_t const nz,
	std::size_t const num_halo,
	float       const alpha,
	std::size_t const num_iter
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

	#pragma omp target data \
		map(to: in_field_) \
		map(from: out_field_)
	for (std::size_t iter = 0; iter < num_iter; ++iter) {
		#if (__GNUC__ < 9) && !defined(__clang__) && !defined(__PGI)
			#pragma omp target teams distribute \
				default(none) \
				shared(in_field_, out_field_)
		#else
			#pragma omp target teams distribute \
				default(none) \
				shared(nx, ny, nz, num_halo, alpha_20, alpha_08, alpha_02, alpha_01) \
				shared(in_field_, out_field_)
		#endif
		for (std::size_t k = 0; k < nz; ++k) {
			#if (__GNUC__ < 9) && !defined(__clang__) && !defined(__PGI)
				#pragma omp parallel for collapse(2) schedule(static) \
					default(none) \
					shared(k, in_field_, out_field_)
			#else
				#pragma omp parallel for collapse(2) schedule(static) \
					default(none) \
					shared(nx, ny, num_halo, alpha_20, alpha_08, alpha_02, alpha_01) \
					shared(k, in_field_, out_field_)
			#endif
			for (std::size_t j = num_halo; j < ny + num_halo; ++j) {
				for (std::size_t i = num_halo; i < nx + num_halo; ++i) {
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
}
