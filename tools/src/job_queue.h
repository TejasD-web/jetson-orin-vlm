#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <cstddef>

namespace edgecloud {

/// Thread-safe, bounded job queue.
/// Producers block when the queue is full; consumers block when empty.
/// Call shutdown() to unblock all waiters and drain.
class JobQueue {
public:
    using Job = std::function<void()>;

    explicit JobQueue(std::size_t capacity = 64);

    /// Push a job. Blocks if queue is at capacity. Returns false if shut down.
    bool push(Job job);

    /// Pop a job. Blocks if queue is empty. Returns std::nullopt if shut down and empty.
    std::optional<Job> pop();

    /// Signal shutdown — unblocks all waiting push/pop calls.
    void shutdown();

    /// Current number of queued jobs.
    std::size_t size() const;

    bool is_shutdown() const;

private:
    mutable std::mutex mu_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    std::queue<Job> queue_;
    std::size_t capacity_;
    bool shutdown_ = false;
};

}  // namespace edgecloud
