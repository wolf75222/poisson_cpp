pybind11_add_module(poisson_py bindings.cpp)
target_link_libraries(poisson_py PRIVATE poisson::poisson)
target_compile_features(poisson_py PRIVATE cxx_std_20)
set_target_properties(poisson_py PROPERTIES OUTPUT_NAME _core)

# Install rule used by scikit-build-core when packaging the wheel.
# The .so lands inside the poisson_cpp/ package so the __init__.py shim
# can re-export it and print install hints when FFTW is missing.
if(SKBUILD)
  install(TARGETS poisson_py LIBRARY DESTINATION poisson_cpp)
endif()
