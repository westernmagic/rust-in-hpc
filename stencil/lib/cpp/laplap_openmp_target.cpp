#include "array.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>

extern "C" void diffuse_laplap_openmp_target(
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

	ArrayView3 in_field_( in_field,  nx + 2 * num_halo, ny + 2 * num_halo, nz);
	ArrayView3 out_field_(out_field, nx + 2 * num_halo, ny + 2 * num_halo, nz);
	Array2 tmp1_field = Array2::zeros(nx + 2 * num_halo, ny + 2 * num_halo);

	#pragma omp target data \
		map(to: in_field_) \
		map(from: out_field_) \
		map(alloc: tmp1_field)
	for (std::size_t iter = 0; iter < num_iter; ++iter) {
		#if (__GNUC__ < 9) && !defined(__clang__) && !defined(__PGI)
			#pragma omp target teams distribute \
				default(none) \
				shared(iter, in_field_, out_field_, tmp1_field)
		#else
			#pragma omp target teams distribute \
				default(none) \
				shared(nx, ny, nz, num_halo, num_iter, alpha) \
				shared(iter, in_field_, out_field_, tmp1_field)
		#endif
		for (std::size_t k = 0; k < nz; ++k) {
			#if (__GNUC__ < 9) && !defined(__clang__) && !defined(__PGI)
				#pragma omp parallel for collapse(2) schedule(static) \
					default(none) \
					shared(k, in_field_, tmp1_field)
			#else
				#pragma omp parallel for collapse(2) schedule(static) \
					default(none) \
					shared(nx, ny, num_halo) \
					shared(k, in_field_, tmp1_field)
			#endif
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

			#if (__GNUC__ < 9) && !defined(__clang__) && !defined(__PGI)
				#pragma omp parallel for collapse(2) schedule(static) \
					default(none) \
					shared(iter, k, in_field_, out_field_, tmp1_field)
			#else
				#pragma omp parallel for collapse(2) schedule(static) \
					default(none) \
					shared(nx, ny, num_halo, num_iter, alpha) \
					shared(iter, k, in_field_, out_field_, tmp1_field)
			#endif
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
}
