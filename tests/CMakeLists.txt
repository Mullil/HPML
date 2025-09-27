enable_testing()
find_package(GTest REQUIRED)

add_executable(matmul_test matmul_tests.cpp)
target_link_libraries(matmul_test PRIVATE matmul GTest::gtest_main pthread)

include(GoogleTest)
gtest_discover_tests(matmul_test)
