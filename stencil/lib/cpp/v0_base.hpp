#pragma once

#include <cstddef>

extern "C" void diffuse_v0_base(
	float * in_field,
	float * out_field,
	std::size_t nx,
	std::size_t ny,
	std::size_t nz,
	std::size_t num_halo,
	float alpha,
	std::size_t num_iter
) noexcept;
