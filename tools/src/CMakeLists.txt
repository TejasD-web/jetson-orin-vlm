add_library(job_queue
    job_queue.cpp
)

target_include_directories(job_queue PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
