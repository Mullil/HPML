#include <gtest/gtest.h>
#include <numeric>
#include <algorithm>
#include <vector>

#include "../src/matmul.hpp"


TEST(matmulSmall, 2x2) {
    double A_data[4] = {1,2,
                        2,4};
    double B_data[4] = {0,3,
                        1,2};
    double C_data[4] = {0,0,0,0};

    Matrix A{2,2,A_data};
    Matrix B{2,2,B_data};
    Matrix C{2,2,C_data};

    matmul(A,B,C);

    EXPECT_EQ(C.data[0], 2);
    EXPECT_EQ(C.data[1], 7);
    EXPECT_EQ(C.data[2], 4);
    EXPECT_EQ(C.data[3], 14);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}