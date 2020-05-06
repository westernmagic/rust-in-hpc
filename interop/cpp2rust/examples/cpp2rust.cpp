#include <complex>
#include <iostream>

extern "C" void zaxpy(std::complex<double> const * a, std::complex<double> const * x, std::size_t nx, std::complex<double> * y, std::size_t ny) noexcept;

int main() {
	constexpr int const n = 3;
	std::complex<double> a = {1.0, 0.0};
	std::complex<double> x[n] = {{1.1, 2.2}, {3.3,  4.4},   {5.5,   6.6}};
	std::complex<double> y[n] = {{7.7, 8.8}, {9.9, 10.10}, {11.11, 12.12}};

	zaxpy(&a, x, n, y, n);

	std::cout << '['
		<< y[0] << ", "
		<< y[1] << ", "
		<< y[2] << "]\n";
}
