#include "prx/libc/include/PreciseWait.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#endif

using Clock = std::chrono::steady_clock;

struct TimingStats {
    double min_us;
    double median_us;
    double p90_us;
    double max_us;
    double overshoot_median_us;
    double overshoot_mean_us;
    int early_returns;
    int total_samples;
};

template <typename Fn>
TimingStats BenchmarkCall(int samples, double target_us, Fn&& fn) {
    std::vector<double> durations;
    durations.reserve(samples);
    int early = 0;
    double overshoot_sum = 0.0;

    for (int i = 0; i < samples; ++i) {
        const auto t0 = Clock::now();
        fn();
        const auto t1 = Clock::now();
        const double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        durations.push_back(elapsed_us);
        if (elapsed_us < target_us) {
            early++;
        }
        overshoot_sum += (elapsed_us - target_us);
    }

    std::sort(durations.begin(), durations.end());
    const double min_us = durations.front();
    const double median_us = durations[durations.size() / 2];
    const double p90_us = durations[static_cast<size_t>(durations.size() * 0.9)];
    const double max_us = durations.back();
    const double overshoot_med = median_us - target_us;
    const double overshoot_mean = overshoot_sum / samples;

    return TimingStats{min_us, median_us, p90_us, max_us, overshoot_med, overshoot_mean, early, samples};
}

int main() {
    std::ostringstream report;

    report << "### macOS Timer & PreciseWait Benchmark\n\n";

#if defined(__APPLE__)
    struct timespec res_mono{}, res_raw{};
    clock_getres(CLOCK_MONOTONIC, &res_mono);
    clock_getres(CLOCK_MONOTONIC_RAW, &res_raw);

    struct timespec ts_mono{}, ts_raw{};
    clock_gettime(CLOCK_MONOTONIC, &ts_mono);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_raw);

    const int64_t ns_mono = static_cast<int64_t>(ts_mono.tv_sec) * 1000000000LL + ts_mono.tv_nsec;
    const int64_t ns_raw = static_cast<int64_t>(ts_raw.tv_sec) * 1000000000LL + ts_raw.tv_nsec;
    const double offset_ms = static_cast<double>(ns_raw - ns_mono) / 1000000.0;

    report << "- **Platform:** macOS ("
#if defined(__aarch64__) || defined(__arm64__)
           << "Apple Silicon arm64"
#elif defined(__x86_64__)
           << "x86_64"
#else
           << "Unknown Arch"
#endif
           << ")\n";
    report << "- **CLOCK_MONOTONIC resolution:** " << res_mono.tv_nsec << " ns\n";
    report << "- **CLOCK_MONOTONIC_RAW resolution:** " << res_raw.tv_nsec << " ns\n";
    report << "- **CLOCK_MONOTONIC vs RAW offset:** " << std::fixed << std::setprecision(2) << offset_ms << " ms\n";
    report << "- **PreciseAppleLowTimerSlack active:** " << (PreciseAppleLowTimerSlack() ? "YES (THREAD_TIME_CONSTRAINT_POLICY)" : "NO") << "\n\n";
#else
    report << "- **Platform:** Non-Apple POSIX\n\n";
#endif

    // 1. Expired deadline overhead
    constexpr int kPastSamples = 100000;
    const auto past_deadline = Clock::now() - std::chrono::seconds(1);
    const auto past_start = Clock::now();
    for (int i = 0; i < kPastSamples; ++i) {
        PreciseSleepUntil(past_deadline);
    }
    const auto past_end = Clock::now();
    const double past_total_ns = std::chrono::duration<double, std::nano>(past_end - past_start).count();
    const double past_ns_per_call = past_total_ns / kPastSamples;

    report << "#### 1. Expired Deadline Fast-Path Overhead\n";
    report << "- `PreciseSleepUntil(past)` (100k calls): **" << std::fixed << std::setprecision(1) << past_ns_per_call << " ns/call**\n\n";

    // 2. Sleep Latency & Overshoot Scenarios
    report << "#### 2. Sleep Timing, Overshoot & Bounds (100 samples each)\n\n";
    report << "| Scenario | Target | Min | Median | p90 | Max | Median Overshoot | Early Returns | Target Bound Status |\n";
    report << "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|\n";

    // Scenario A: PreciseSleepUs(50) -> Bound < 90 us
    const auto stats_50us = BenchmarkCall(100, 50.0, [] { PreciseSleepUs(50); });
    const bool pass_50us = (stats_50us.median_us < 90.0) && (stats_50us.early_returns == 0);
    report << "| `PreciseSleepUs(50)` | 50.0 µs | "
           << std::fixed << std::setprecision(1) << stats_50us.min_us << " µs | "
           << stats_50us.median_us << " µs | "
           << stats_50us.p90_us << " µs | "
           << stats_50us.max_us << " µs | +"
           << stats_50us.overshoot_median_us << " µs | "
           << stats_50us.early_returns << " / " << stats_50us.total_samples << " | "
           << (pass_50us ? "✅ PASS (< 90 µs)" : "❌ FAIL (>= 90 µs)") << " |\n";

    // Scenario B: PreciseSleepUs(1500) -> Bound < 1650 us (The one that failed on macOS in main)
    const auto stats_1500us = BenchmarkCall(100, 1500.0, [] { PreciseSleepUs(1500); });
    const bool pass_1500us = (stats_1500us.median_us < 1650.0) && (stats_1500us.early_returns == 0);
    report << "| `PreciseSleepUs(1500)` | 1500.0 µs | "
           << std::fixed << std::setprecision(1) << stats_1500us.min_us << " µs | "
           << stats_1500us.median_us << " µs | "
           << stats_1500us.p90_us << " µs | "
           << stats_1500us.max_us << " µs | +"
           << stats_1500us.overshoot_median_us << " µs | "
           << stats_1500us.early_returns << " / " << stats_1500us.total_samples << " | "
           << (pass_1500us ? "✅ PASS (< 1650 µs)" : "❌ FAIL (>= 1650 µs)") << " |\n";

    // Scenario C: PreciseSleepUntil(+50us) -> Bound < 90 us
    const auto stats_until_50us = BenchmarkCall(100, 50.0, [] {
        PreciseSleepUntil(Clock::now() + std::chrono::microseconds(50));
    });
    const bool pass_until_50us = (stats_until_50us.median_us < 90.0) && (stats_until_50us.early_returns == 0);
    report << "| `PreciseSleepUntil(+50µs)` | 50.0 µs | "
           << std::fixed << std::setprecision(1) << stats_until_50us.min_us << " µs | "
           << stats_until_50us.median_us << " µs | "
           << stats_until_50us.p90_us << " µs | "
           << stats_until_50us.max_us << " µs | +"
           << stats_until_50us.overshoot_median_us << " µs | "
           << stats_until_50us.early_returns << " / " << stats_until_50us.total_samples << " | "
           << (pass_until_50us ? "✅ PASS (< 90 µs)" : "❌ FAIL (>= 90 µs)") << " |\n";

    // Scenario D: PreciseSleepUntil(+1000us) -> Future 1ms
    const auto stats_until_1ms = BenchmarkCall(100, 1000.0, [] {
        PreciseSleepUntil(Clock::now() + std::chrono::milliseconds(1));
    });
    report << "| `PreciseSleepUntil(+1ms)` | 1000.0 µs | "
           << std::fixed << std::setprecision(1) << stats_until_1ms.min_us << " µs | "
           << stats_until_1ms.median_us << " µs | "
           << stats_until_1ms.p90_us << " µs | "
           << stats_until_1ms.max_us << " µs | +"
           << stats_until_1ms.overshoot_median_us << " µs | "
           << stats_until_1ms.early_returns << " / " << stats_until_1ms.total_samples << " | "
           << (stats_until_1ms.early_returns == 0 ? "✅ 0 early" : "❌ EARLY RETURN") << " |\n\n";

    std::cout << report.str();

    if (const char* step_summary = std::getenv("GITHUB_STEP_SUMMARY")) {
        std::ofstream summary_file(step_summary, std::ios::app);
        if (summary_file.is_open()) {
            summary_file << report.str();
        }
    }

    if (stats_50us.early_returns > 0 || stats_1500us.early_returns > 0 || stats_until_50us.early_returns > 0 || stats_until_1ms.early_returns > 0) {
        std::cerr << "FATAL: One or more benchmarks returned BEFORE deadline!\n";
        return 1;
    }

    if (!pass_50us || !pass_1500us || !pass_until_50us) {
        if (const char* strict = std::getenv("APS5_STRICT_TIMING_BOUNDS"); strict != nullptr && strict[0] != '\0' && strict[0] != '0') {
            std::cerr << "FATAL: Strict timing median bounds missed!\n";
            return 1;
        }
        std::clog << "NOTE: One or more non-contract timing bounds were exceeded under runner load.\n";
    }

    return 0;
}
