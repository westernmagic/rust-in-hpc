#include <iostream>

extern "C" void zaxpy(double a, double const * x, std::size_t nx, double * y, std::size_t ny) noexcept;

int main() {
	constexpr int const n = 3;
	double a = 10.0;
	double x[n] = {1.0, 2.0, 3.0};
	double y[n] = {4.0, 5.0, 6.0};

	zaxpy(a, x, n, y, n);

	std::cout << '['
		<< y[0] << ", "
		<< y[1] << ", "
		<< y[2] << "]\n";
}
