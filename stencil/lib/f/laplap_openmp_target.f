!> v0: base version
!>
!> Ported from stencil2d-orig.F90
!>
!> @author Oliver Fuhrer
!> @author Michal Sudwoj
!> @date 2020-07-19
!> @licence LGPL-3.0
subroutine diffuse_laplap_openmp_target(in_field, out_field, nx, ny, nz, num_halo, alpha, num_iter) bind(C)
    use, intrinsic :: iso_c_binding, only: c_float, c_size_t, c_ptr, c_associated, c_f_pointer
    use m_assert, only: assert

    implicit none
    ! arguments
    type(c_ptr),              value, intent(in) :: in_field
    type(c_ptr),              value, intent(in) :: out_field
    integer(kind = c_size_t), value, intent(in) :: nx
    integer(kind = c_size_t), value, intent(in) :: ny
    integer(kind = c_size_t), value, intent(in) :: nz
    integer(kind = c_size_t), value, intent(in) :: num_halo
    real(kind = c_float),     value, intent(in) :: alpha
    integer(kind = c_size_t), value, intent(in) :: num_iter

    ! local
    real(kind = c_float), pointer     :: in_field_(:, :, :)
    real(kind = c_float), pointer     :: out_field_(:, :, :)
    real(kind = c_float), allocatable :: tmp1_field(:, :)
    real(kind = c_float)              :: laplap
    integer(kind = c_size_t)          :: iter
    integer(kind = c_size_t)          :: i
    integer(kind = c_size_t)          :: j
    integer(kind = c_size_t)          :: k

    call assert(c_associated(in_field),  "c_associated(in_field)")
    call assert(c_associated(out_field), "c_associated(out_field)")
    call assert(nx > 0, "nx > 0")
    call assert(ny > 0, "ny > 0")
    call assert(nz > 0, "nz > 0")
    call assert(num_halo > 0, "num_halo > 0")
    ! call assert(.not. isnan(alpha), ".not. isnan(alpha)")
    call assert(num_iter > 0, "num_iter > 0")

    call c_f_pointer(in_field,  in_field_,  [nx + 2 * num_halo, ny + 2 * num_halo, nz])
    call c_f_pointer(out_field, out_field_, [nx + 2 * num_halo, ny + 2 * num_halo, nz])

    allocate(tmp1_field(nx + 2 * num_halo, ny + 2 * num_halo))

    !$omp target data &
    !$omp   map(to: in_field_) &
    !$omp   map(from: out_field_) &
    !$omp   map(alloc: tmp1_field)
    do iter = 1, num_iter
        !$omp target teams distribute &
        !$omp   default(none) &
        !$omp   shared(iter, nx, ny, nz, num_halo, num_iter, alpha) &
        !$omp   shared(in_field_, out_field_, tmp1_field) &
        !$omp   private(i, j, k)
        do k = 1, nz
            !$omp parallel do collapse(2) schedule(static) &
            !$omp   default(none) &
            !$omp   shared(nx, ny, num_halo, k) &
            !$omp   shared(in_field_, tmp1_field) &
            !$omp   private(i, j)
            do j = 1 + num_halo - 1, ny + num_halo + 1
                do i = 1 + num_halo - 1, nx + num_halo + 1
                    tmp1_field(i, j) = &
                        -4.0_c_float * in_field_(i,     j,     k) &
                        +              in_field_(i - 1, j,     k) &
                        +              in_field_(i + 1, j,     k) &
                        +              in_field_(i,     j - 1, k) &
                        +              in_field_(i,     j + 1, k)
                end do
            end do

            !$omp parallel do collapse(2) schedule(static) &
            !$omp   default(none) &
            !$omp   shared(iter, nx, ny, num_halo, alpha, num_iter, k) &
            !$omp   shared(in_field_, out_field_, tmp1_field) &
            !$omp   private(i, j, laplap)
            do j = 1 + num_halo, ny + num_halo
                do i = 1 + num_halo, nx + num_halo
                    laplap = &
                        -4.0_c_float * tmp1_field(i,     j    ) &
                        +              tmp1_field(i - 1, j    ) &
                        +              tmp1_field(i + 1, j    ) &
                        +              tmp1_field(i,     j - 1) &
                        +              tmp1_field(i,     j + 1)

                    if (iter /= num_iter) then
                        in_field_(i, j, k) = in_field_(i, j, k) - alpha * laplap
                    else
                        out_field_(i, j, k) = in_field_(i, j, k) - alpha * laplap
                    end if
                end do
            end do
        end do
    end do
    !$omp end target data
end subroutine
