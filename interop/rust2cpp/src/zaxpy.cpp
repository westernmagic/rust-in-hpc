#include "zaxpy.hpp"
#include <cassert>

extern "C" void zaxpy(double a, double const * x, std::size_t nx, double * y, std::size_t ny) noexcept {
	assert(nx == ny);
	for (std::size_t i = 0; i < nx; ++i) {
		y[i] += a * x[i];
	}
}
