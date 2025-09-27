add_test([=[matmulSmall.2x2]=]  /home/eetu/HPML/build/tests/matmul_test [==[--gtest_filter=matmulSmall.2x2]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[matmulSmall.2x2]=]  PROPERTIES WORKING_DIRECTORY /home/eetu/HPML/build/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  matmul_test_TESTS matmulSmall.2x2)
