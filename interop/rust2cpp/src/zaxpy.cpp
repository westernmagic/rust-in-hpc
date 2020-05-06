#include "zaxpy.hpp"
#include <cassert>
#include <complex>

extern "C" void zaxpy(std::complex<double> const * a, std::complex<double> const * x, std::size_t nx, std::complex<double> * y, std::size_t ny) noexcept {
	assert(nx == ny);
	for (std::size_t i = 0; i < nx; ++i) {
		y[i] += *a * x[i];
	}
}
