add_library(matmul linalg/matmul.cpp)
target_include_directories(matmul PUBLIC ${PROJECT_SOURCE_DIR}/include)