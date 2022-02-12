!> v1: inline version
!>
!>   - remove `tmp1_field` and simplify math
!>   - swap `in_field` and `out_field`
!>
!> @author Michal Sudwoj
!> @date 2020-07-19
!> @licence LGPL-3.0
subroutine diffuse_inline_openacc(in_field, out_field, nx, ny, nz, num_halo, alpha, num_iter) bind(C)
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
    real(kind = c_float), pointer     :: tmp_field(:, :, :)
    real(kind = c_float)              :: alpha_20
    real(kind = c_float)              :: alpha_08
    real(kind = c_float)              :: alpha_02
    real(kind = c_float)              :: alpha_01
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

    alpha_20 = -20 * alpha + 1
    alpha_08 =   8 * alpha
    alpha_02 =  -2 * alpha
    alpha_01 =  -1 * alpha

    !$acc data &
    !$acc   copyin(in_field_(nx + 2 * num_halo, ny + 2 * num_halo, nz)) &
    !$acc   copyout(out_field_(nx + 2 * num_halo, ny + 2 * num_halo, nz))
    do iter = 1, num_iter
        !$acc parallel loop gang worker vector collapse(3)
        do k = 1, nz
            do j = 1 + num_halo, ny + num_halo
                do i = 1 + num_halo, nx + num_halo
                    out_field_(i, j, k) = &
                        + alpha_20 * in_field_(i,     j,     k) &
                        + alpha_08 * in_field_(i - 1, j,     k) &
                        + alpha_08 * in_field_(i + 1, j,     k) &
                        + alpha_08 * in_field_(i,     j - 1, k) &
                        + alpha_08 * in_field_(i,     j + 1, k) &
                        + alpha_02 * in_field_(i - 1, j - 1, k) &
                        + alpha_02 * in_field_(i - 1, j + 1, k) &
                        + alpha_02 * in_field_(i + 1, j - 1, k) &
                        + alpha_02 * in_field_(i + 1, j + 1, k) &
                        + alpha_01 * in_field_(i - 2, j,     k) &
                        + alpha_01 * in_field_(i + 2, j,     k) &
                        + alpha_01 * in_field_(i,     j - 2, k) &
                        + alpha_01 * in_field_(i,     j + 2, k)
                end do
            end do
        end do
        !$acc end parallel loop

        tmp_field => in_field_
        in_field_ => out_field_
        out_field_ => tmp_field
    end do
    !$acc end data
end subroutine
