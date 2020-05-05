#include <cstddef>

extern "C" void zaxpy(double a, double const * x, std::size_t nx, double * y, std::size_t ny) noexcept;
