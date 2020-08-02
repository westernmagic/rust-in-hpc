module m_assert
    implicit none
    private

    public assert

    contains
        subroutine assert(cond, msg)
#ifdef _CRAYC
                !DIR$ INLINEALWAYS assert
#endif
#ifdef __INTEL_COMPILER
                !DIR$ ATTRIBUTES FORCEINLINE :: assert
#endif
            use, intrinsic :: iso_fortran_env, only: error_unit
            implicit none
        
            logical,                      intent(in) :: cond
            character(len = *), optional, intent(in) :: msg
        
            if (.not. cond) then
                if (present(msg)) then
                    write(error_unit, *) "Assertion failed:", msg
                else
                    write(error_unit, *) "Assertion failed"
                end if
                error stop
            end if
        end subroutine
end module
