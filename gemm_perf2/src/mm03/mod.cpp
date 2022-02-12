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

void add_dot_1x4(
	std::size_t k,
	double const * a,
	std::size_t lda,
	double const * b,
	std::size_t ldb,
	double * c,
	std::size_t ldc
) noexcept {
	add_dot(k, &a[0 + lda * 0], lda, &b[0 + ldb * 0], &c[0 + ldc * 0]);
	add_dot(k, &a[0 + lda * 0], lda, &b[0 + ldb * 1], &c[0 + ldc * 1]);
	add_dot(k, &a[0 + lda * 0], lda, &b[0 + ldb * 2], &c[0 + ldc * 2]);
	add_dot(k, &a[0 + lda * 0], lda, &b[0 + ldb * 3], &c[0 + ldc * 3]);
}

extern "C"
void mm03(
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
	for (std::size_t j = 0; j < n; j += 4) {
		for (std::size_t i = 0; i < m; ++i) {
			add_dot_1x4(k, &a[i + lda * 0], lda, &b[0 + ldb * j], ldb, &c[i + ldc * j], ldc);
		}
	}
}
