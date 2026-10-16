include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_job_queue test_job_queue.cpp)
target_link_libraries(test_job_queue PRIVATE job_queue GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_job_queue)
