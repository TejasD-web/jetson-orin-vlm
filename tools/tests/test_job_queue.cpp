#include "job_queue.h"

#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using edgecloud::JobQueue;

TEST(JobQueueTest, PushPopSingleThread) {
    JobQueue q(4);
    int result = 0;

    q.push([&] { result = 42; });
    auto job = q.pop();
    ASSERT_TRUE(job.has_value());
    (*job)();
    EXPECT_EQ(result, 42);
}

TEST(JobQueueTest, FIFO_Order) {
    JobQueue q(4);
    std::vector<int> order;

    q.push([&] { order.push_back(1); });
    q.push([&] { order.push_back(2); });
    q.push([&] { order.push_back(3); });

    for (int i = 0; i < 3; ++i) {
        auto job = q.pop();
        (*job)();
    }
    EXPECT_EQ(order, (std::vector<int>{1, 2, 3}));
}

TEST(JobQueueTest, ShutdownUnblocksConsumer) {
    JobQueue q(4);

    std::thread consumer([&] {
        auto job = q.pop();  // blocks until shutdown
        EXPECT_FALSE(job.has_value());
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    q.shutdown();
    consumer.join();
}

TEST(JobQueueTest, BoundedCapacityBlocks) {
    JobQueue q(2);
    std::atomic<bool> pushed{false};

    q.push([] {});
    q.push([] {});

    // Third push should block until a pop frees space
    std::thread producer([&] {
        q.push([] {});
        pushed = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(pushed.load());

    q.pop();  // free one slot
    producer.join();
    EXPECT_TRUE(pushed.load());
}

TEST(JobQueueTest, ConcurrentProducersConsumers) {
    JobQueue q(8);
    std::atomic<int> count{0};
    constexpr int NUM_JOBS = 1000;

    std::vector<std::thread> producers;
    for (int i = 0; i < 4; ++i) {
        producers.emplace_back([&] {
            for (int j = 0; j < NUM_JOBS / 4; ++j) {
                q.push([&] { count.fetch_add(1); });
            }
        });
    }

    std::vector<std::thread> consumers;
    for (int i = 0; i < 4; ++i) {
        consumers.emplace_back([&] {
            while (true) {
                auto job = q.pop();
                if (!job.has_value()) break;
                (*job)();
            }
        });
    }

    for (auto& t : producers) t.join();
    q.shutdown();
    for (auto& t : consumers) t.join();

    EXPECT_EQ(count.load(), NUM_JOBS);
}
