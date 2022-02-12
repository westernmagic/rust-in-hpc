#include <cstddef>
#include <complex>

extern "C" void zaxpy(std::complex<double> const * a, std::complex<double> const * x, std::size_t nx, std::complex<double> * y, std::size_t ny) noexcept;
