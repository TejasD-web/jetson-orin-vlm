#include "job_queue.h"

namespace edgecloud {

JobQueue::JobQueue(std::size_t capacity) : capacity_(capacity) {}

bool JobQueue::push(Job job) {
    std::unique_lock lock(mu_);
    not_full_.wait(lock, [this] { return shutdown_ || queue_.size() < capacity_; });

    if (shutdown_) return false;

    queue_.push(std::move(job));
    not_empty_.notify_one();
    return true;
}

std::optional<JobQueue::Job> JobQueue::pop() {
    std::unique_lock lock(mu_);
    not_empty_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });

    if (queue_.empty()) return std::nullopt;

    auto job = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return job;
}

void JobQueue::shutdown() {
    std::lock_guard lock(mu_);
    shutdown_ = true;
    not_full_.notify_all();
    not_empty_.notify_all();
}

std::size_t JobQueue::size() const {
    std::lock_guard lock(mu_);
    return queue_.size();
}

bool JobQueue::is_shutdown() const {
    std::lock_guard lock(mu_);
    return shutdown_;
}

}  // namespace edgecloud
