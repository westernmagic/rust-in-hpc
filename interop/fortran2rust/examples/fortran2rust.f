program fortran2rust
    use iso_fortran_env
    implicit none

    interface
        subroutine zaxpy(a, x, nx, y, ny) bind(c)
            use iso_c_binding
            implicit none

            complex(c_double_complex),               intent(in)    :: a
            complex(c_double_complex), dimension(*), intent(in)    :: x
            integer(c_size_t),         value,        intent(in)    :: nx
            complex(c_double_complex), dimension(*), intent(inout) :: y
            integer(c_size_t),         value,        intent(in)    :: ny
        end subroutine zaxpy
    end interface

    integer(int64),  parameter    :: n = 3
    complex(real64)               :: a = cmplx(1.0, 0.0)
    complex(real64), dimension(n) :: x = [cmplx(1.1, 2.2), cmplx(3.3,  4.4),  cmplx(5.5,    6.6)]
    complex(real64), dimension(n) :: y = [cmplx(7.7, 8.8), cmplx(9.9, 10.10), cmplx(11.11, 12.12)]

    call zaxpy(a, x, n, y, n)

    print *, "[", y(1), ", ", y(2), ", ", y(3), "]"
end program
