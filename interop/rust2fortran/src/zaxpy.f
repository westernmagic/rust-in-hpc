subroutine zaxpy(a, x, nx, y, ny)
    use iso_fortran_env
    implicit none

    real(real64),                 intent(in)    :: a
    real(real64),   dimension(*), intent(in)    :: x
    integer(int64),               intent(in)    :: nx
    real(real64),   dimension(*), intent(inout) :: y
    integer(int64),               intent(in)    :: ny
    integer(int64)                              :: i = 1

    if (nx .ne. ny) then 
        call exit(1)
    end if

    do i = 1, nx
        y(i) = a * x(i) + y(i)
    end do
end subroutine zaxpy
