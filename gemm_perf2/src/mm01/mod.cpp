#include "mod.hpp"
#include <cstddef>

void add_dot(
	std::size_t k,
	double const * x,
	std::size_t incx,
	double const * y,
	double * gamma
) noexcept {
	for (std::size_t p = 0; p < k; ++p) {
		*gamma += x[p * incx] * y[p];
	}
}

extern "C"
void mm01(
	std::size_t m,
	std::size_t n,
	std::size_t k,
	double const * a,
	std::size_t lda,
	double const * b,
	std::size_t ldb,
	double * c,
	std::size_t ldc
) noexcept {
	for (std::size_t j = 0; j < n; ++j) {
		for (std::size_t i = 0; i < m; ++i) {
			add_dot(k, &a[i + lda * 0], lda, &b[0 + ldb * j], &c[i + ldc * j]);
		}
	}
}
