#include "mod.hpp"
#include <cstddef>

void add_dot_1x4(
	std::size_t k,
	double const * a,
	std::size_t lda,
	double const * b,
	std::size_t ldb,
	double * c,
	std::size_t ldc
) noexcept {
	double c_00_reg = 0;
	double c_01_reg = 0;
	double c_02_reg = 0;
	double c_03_reg = 0;
	double const * b_p0_ptr = &b[0 + ldb * 0];
	double const * b_p1_ptr = &b[0 + ldb * 1];
	double const * b_p2_ptr = &b[0 + ldb * 2];
	double const * b_p3_ptr = &b[0 + ldb * 3];

	for (std::size_t p = 0; p < k; ++p) {
		double a_0p_reg = a[0 + lda * p];
		c_00_reg += a_0p_reg * (*b_p0_ptr++);
		c_01_reg += a_0p_reg * (*b_p1_ptr++);
		c_02_reg += a_0p_reg * (*b_p2_ptr++);
		c_03_reg += a_0p_reg * (*b_p3_ptr++);
	}

	c[0 + ldc * 0] += c_00_reg;
	c[0 + ldc * 1] += c_01_reg;
	c[0 + ldc * 2] += c_02_reg;
	c[0 + ldc * 3] += c_03_reg;
}

extern "C"
void mm07(
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
