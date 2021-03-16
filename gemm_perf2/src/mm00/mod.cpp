#include "mod.hpp"
#include <cstddef>

extern "C"
void mm00(
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
	for (std::size_t i = 0; i < m; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			for (std::size_t p = 0; p < k; ++p) {
				c[i + ldc * j] += a[i + lda * p] * b[p + ldb * j];
			}
		}
	}
}
