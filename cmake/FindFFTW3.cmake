# Minimal FindFFTW3 module.
#
# Locates the FFTW3 library (double precision, real-to-real transforms)
# via pkg-config when available, otherwise by direct search. Creates the
# imported target `FFTW3::fftw3` on success.

include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_FFTW3 QUIET fftw3)
endif()

find_path(FFTW3_INCLUDE_DIR
  NAMES fftw3.h
  HINTS ${PC_FFTW3_INCLUDE_DIRS}
        /opt/homebrew/include
        /usr/local/include
        /usr/include)

find_library(FFTW3_LIBRARY
  NAMES fftw3
  HINTS ${PC_FFTW3_LIBRARY_DIRS}
        /opt/homebrew/lib
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu)

find_package_handle_standard_args(FFTW3
  REQUIRED_VARS FFTW3_INCLUDE_DIR FFTW3_LIBRARY)

if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3)
  add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
  set_target_properties(FFTW3::fftw3 PROPERTIES
    IMPORTED_LOCATION "${FFTW3_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}")
endif()

mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY)
