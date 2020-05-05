program fortran2rust
    use iso_fortran_env
    implicit none

    interface
        subroutine zaxpy(a, x, nx, y, ny) bind(c)
            use iso_c_binding
            implicit none

            real(c_double),    value,        intent(in)    :: a
            real(c_double),    dimension(*), intent(in)    :: x
            integer(c_size_t), value,        intent(in)    :: nx
            real(c_double),    dimension(*), intent(inout) :: y
            integer(c_size_t), value,        intent(in)    :: ny
        end subroutine zaxpy
    end interface

    integer(int64), parameter    :: n = 3
    real(real64)                 :: a = 10.0
    real(real64),   dimension(n) :: x = [1.0, 2.0, 3.0]
    real(real64),   dimension(n) :: y = [4.0, 5.0, 6.0]

    call zaxpy(a, x, n, y, n)

    print *, "[", y(1), ", ", y(2), ", ", y(3), "]"
end program
