add_executable(bench_solvers bench_solvers.cpp)
target_link_libraries(bench_solvers PRIVATE poisson::poisson)
