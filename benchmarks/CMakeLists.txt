add_executable(bench_solvers bench_solvers.cpp)
target_link_libraries(bench_solvers PRIVATE poisson::poisson)

add_executable(profile_cg profile_cg.cpp)
target_link_libraries(profile_cg PRIVATE poisson::poisson)
