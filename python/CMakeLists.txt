pybind11_add_module(poisson_py bindings.cpp)
target_link_libraries(poisson_py PRIVATE poisson::poisson)
target_compile_features(poisson_py PRIVATE cxx_std_20)
set_target_properties(poisson_py PROPERTIES OUTPUT_NAME poisson_cpp)
