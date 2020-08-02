#pragma once

#include <cassert>
#include <cstddef>
#include <utility>

class ArrayView3 {
	protected:
		float * data;
		std::size_t nx;
		std::size_t ny;
		std::size_t nz;

	public:
		constexpr ArrayView3(float * data, std::size_t nx, std::size_t ny, std::size_t nz) noexcept :
			data(data), nx(nx), ny(ny), nz(nz)
		{
			assert(nx > 0);
			assert(ny > 0);
			assert(nz > 0);
		}

		constexpr float& operator()(std::size_t i, std::size_t j, std::size_t k) noexcept {
			assert(0 <= i && i < nx);
			assert(0 <= j && j < ny);
			assert(0 <= k && k < nz);

			return data[i + j * nx + k * nx * ny];
		}

		friend void swap(ArrayView3 & lhs, ArrayView3 & rhs) noexcept {
			using std::swap;
			swap(lhs.data, rhs.data);
			swap(lhs.nx, rhs.nx);
			swap(lhs.ny, rhs.ny);
			swap(lhs.nz, rhs.nz);
		}
};

class Array3 : public ArrayView3 {
	public:
		Array3(std::size_t nx, std::size_t ny, std::size_t nz) noexcept :
			ArrayView3(new float[nx * ny * nz], nx, ny, nz)
		{
		}

		~Array3() noexcept {
			delete[] data;
		}

		static Array3 zeros(std::size_t nx, std::size_t ny, std::size_t nz) noexcept {
			Array3 result(nx, ny, nz);
			for (std::size_t k = 0; k < nz; ++k) {
				for (std::size_t j = 0; j < ny; ++j) {
					for (std::size_t i = 0; i < nx; ++i) {
						result(i, j, k) = 0.0f;
					}
				}
			}

			return result;
		}
};

class ArrayView2 {
	protected:
		float * data;
	public:
		std::size_t const nx;
		std::size_t const ny;

	public:
		constexpr ArrayView2(float * data, std::size_t nx, std::size_t ny) noexcept :
			data(data), nx(nx), ny(ny)
		{
			assert(nx > 0);
			assert(ny > 0);
		}

		constexpr float& operator()(std::size_t i, std::size_t j) noexcept {
			assert(0 <= i && i < nx);
			assert(0 <= j && j < ny);

			return data[i + j * nx];
		}
};

class Array2: public ArrayView2 {
	public:
		Array2(std::size_t nx, std::size_t ny) noexcept :
			ArrayView2(new float[nx * ny], nx, ny)
		{}

		~Array2() noexcept {
			delete[] data;
		}

		static Array2 zeros(std::size_t nx, std::size_t ny) noexcept {
			Array2 result(nx, ny);
			for (std::size_t j = 0; j < ny; ++j) {
				for (std::size_t i = 0; i < nx; ++i) {
					result(i, j) = 0.0f;
				}
			}

			return result;
		}
};
