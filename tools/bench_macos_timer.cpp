#include "prx/libc/include/PreciseWait.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
#endif

using Clock = std::chrono::steady_clock;

template <typename T>
__attribute__((always_inline)) inline void DoNotOptimize(const T& val) {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" :: "r,m"(val) : "memory");
#endif
}

__attribute__((noinline)) void BenchmarkSleepUntil(Clock::time_point deadline) {
    PreciseSleepUntil(deadline);
}

inline std::uint64_t ReadTscTicks() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_lfence();
    const std::uint64_t val = __rdtsc();
    _mm_lfence();
    return val;
#else
    return 0;
#endif
}

inline std::uint64_t ReadCntvctTicks() {
#if defined(__aarch64__) || defined(__arm64__)
    std::uint64_t val = 0;
    asm volatile("isb; mrs %0, cntvct_el0; isb" : "=r"(val) :: "memory");
    return val;
#else
    return 0;
#endif
}

inline std::uint64_t ReadCntfrqHz() {
#if defined(__aarch64__) || defined(__arm64__)
    std::uint64_t freq = 0;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
#else
    return 0;
#endif
}

inline std::uint64_t ReadThreadCpuNanos() {
#if defined(CLOCK_THREAD_CPUTIME_ID)
    struct timespec ts{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
        return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
    }
#endif
    return 0;
}

struct BenchmarkMetrics {
    double wall_min_us;
    double wall_median_us;
    double wall_p90_us;
    double wall_max_us;
    double wall_mean_us;
    double overshoot_median_us;
    double overshoot_mean_us;

    double cpu_min_us;
    double cpu_median_us;
    double cpu_p90_us;
    double cpu_max_us;
    double cpu_mean_us;

    int early_returns;
    int total_samples;
};

template <typename Fn>
BenchmarkMetrics BenchmarkCall(int samples, double target_us, Fn&& fn) {
    std::vector<double> wall_durations;
    std::vector<double> cpu_durations;
    wall_durations.reserve(samples);
    cpu_durations.reserve(samples);
    int early = 0;
    double overshoot_sum = 0.0;
    double cpu_sum = 0.0;
    double wall_sum = 0.0;

    for (int i = 0; i < samples; ++i) {
        const std::uint64_t cpu_t0 = ReadThreadCpuNanos();
        const auto t0 = Clock::now();
        fn();
        const auto t1 = Clock::now();
        const std::uint64_t cpu_t1 = ReadThreadCpuNanos();

        const double wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        const double cpu_us = (cpu_t1 >= cpu_t0) ? static_cast<double>(cpu_t1 - cpu_t0) / 1000.0 : 0.0;

        wall_durations.push_back(wall_us);
        cpu_durations.push_back(cpu_us);
        wall_sum += wall_us;
        cpu_sum += cpu_us;
        if (wall_us < target_us) {
            early++;
        }
        overshoot_sum += (wall_us - target_us);
    }

    std::sort(wall_durations.begin(), wall_durations.end());
    std::sort(cpu_durations.begin(), cpu_durations.end());

    const double wall_min_us = wall_durations.front();
    const double wall_median_us = wall_durations[wall_durations.size() / 2];
    const double wall_p90_us = wall_durations[static_cast<size_t>(wall_durations.size() * 0.9)];
    const double wall_max_us = wall_durations.back();
    const double wall_mean_us = wall_sum / samples;

    const double overshoot_med = wall_median_us - target_us;
    const double overshoot_mean = overshoot_sum / samples;

    const double cpu_min_us = cpu_durations.front();
    const double cpu_median_us = cpu_durations[cpu_durations.size() / 2];
    const double cpu_p90_us = cpu_durations[static_cast<size_t>(cpu_durations.size() * 0.9)];
    const double cpu_max_us = cpu_durations.back();
    const double cpu_mean_us = cpu_sum / samples;

    return BenchmarkMetrics{
        wall_min_us, wall_median_us, wall_p90_us, wall_max_us, wall_mean_us,
        overshoot_med, overshoot_mean,
        cpu_min_us, cpu_median_us, cpu_p90_us, cpu_max_us, cpu_mean_us,
        early, samples
    };
}

struct PhaseStats {
    double min_val;
    double median_val;
    double p90_val;
    double mean_val;
    double max_val;
};

inline PhaseStats ComputePhaseStats(std::vector<double>& vals) {
    if (vals.empty()) {
        return PhaseStats{0.0, 0.0, 0.0, 0.0, 0.0};
    }
    std::sort(vals.begin(), vals.end());
    const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    return PhaseStats{
        vals.front(),
        vals[vals.size() / 2],
        vals[static_cast<size_t>(vals.size() * 0.9)],
        sum / vals.size(),
        vals.back()
    };
}

int main() {
    std::ostringstream report;

    report << "### macOS Timer, PreciseWait & Phase Instrumentation Benchmark\n\n";

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
#if defined(__aarch64__) || defined(__arm64__)
    report << "- **ARM64 cntfrq_el0:** " << ReadCntfrqHz() << " Hz\n";
#endif
    report << "- **PreciseAppleLowTimerSlack active:** " << (PreciseAppleLowTimerSlack() ? "YES (THREAD_TIME_CONSTRAINT_POLICY)" : "NO") << "\n\n";
#else
    report << "- **Platform:** Non-Apple POSIX\n\n";
#endif

    const auto baseline = BenchmarkCall(100, 0.0, [] {
        volatile int dummy = 0;
        DoNotOptimize(dummy);
    });
    report << "#### Measurement Baseline Calibration (100 samples empty loop)\n\n";
    report << "- **Measurement loop overhead (Wall):** Median = "
           << std::fixed << std::setprecision(3) << baseline.wall_median_us * 1000.0 << " ns, Mean = "
           << baseline.wall_mean_us * 1000.0 << " ns\n";
    report << "- **Measurement loop overhead (Thread CPU):** Median = "
           << baseline.cpu_median_us * 1000.0 << " ns, Mean = "
           << baseline.cpu_mean_us * 1000.0 << " ns\n\n";

    constexpr int kPastSamples = 100000;
    const auto past_deadline = Clock::now() - std::chrono::seconds(1);
    const auto past_us = std::chrono::time_point_cast<std::chrono::microseconds>(past_deadline);

    const std::uint64_t tsc0 = ReadTscTicks();
    const std::uint64_t cnt0 = ReadCntvctTicks();
    const std::uint64_t cpu0 = ReadThreadCpuNanos();
    const auto t0 = Clock::now();
    for (int i = 0; i < kPastSamples; ++i) {
        BenchmarkSleepUntil(past_deadline);
    }
    const auto t1 = Clock::now();
    const std::uint64_t cpu1 = ReadThreadCpuNanos();
    const std::uint64_t cnt1 = ReadCntvctTicks();
    const std::uint64_t tsc1 = ReadTscTicks();

    const double ns_per_call = std::chrono::duration<double, std::nano>(t1 - t0).count() / kPastSamples;
    const double cpu_ns_per_call = static_cast<double>(cpu1 - cpu0) / kPastSamples;
#if defined(__x86_64__) || defined(_M_X64)
    const double tsc_per_call = static_cast<double>(tsc1 - tsc0) / kPastSamples;
#elif defined(__aarch64__) || defined(__arm64__)
    const double cnt_per_call = static_cast<double>(cnt1 - cnt0) / kPastSamples;
#endif

    const std::uint64_t tsc0_us = ReadTscTicks();
    const std::uint64_t cnt0_us = ReadCntvctTicks();
    const std::uint64_t cpu0_us = ReadThreadCpuNanos();
    const auto t0_us = Clock::now();
    for (int i = 0; i < kPastSamples; ++i) {
        BenchmarkSleepUntil(past_us);
    }
    const auto t1_us = Clock::now();
    const std::uint64_t cpu1_us = ReadThreadCpuNanos();
    const std::uint64_t cnt1_us = ReadCntvctTicks();
    const std::uint64_t tsc1_us = ReadTscTicks();

    const double ns_per_call_us = std::chrono::duration<double, std::nano>(t1_us - t0_us).count() / kPastSamples;
    const double cpu_ns_per_call_us = static_cast<double>(cpu1_us - cpu0_us) / kPastSamples;
#if defined(__x86_64__) || defined(_M_X64)
    const double tsc_per_call_us = static_cast<double>(tsc1_us - tsc0_us) / kPastSamples;
#elif defined(__aarch64__) || defined(__arm64__)
    const double cnt_per_call_us = static_cast<double>(cnt1_us - cnt0_us) / kPastSamples;
#endif

    volatile int64_t runtime_zero_ns = 0;
    DoNotOptimize(runtime_zero_ns);
    const Clock::time_point zero_time{std::chrono::nanoseconds(runtime_zero_ns)};

    const std::uint64_t tsc0_z = ReadTscTicks();
    const std::uint64_t cnt0_z = ReadCntvctTicks();
    const std::uint64_t cpu0_z = ReadThreadCpuNanos();
    const auto t0_z = Clock::now();
    for (int i = 0; i < kPastSamples; ++i) {
        BenchmarkSleepUntil(zero_time);
    }
    const auto t1_z = Clock::now();
    const std::uint64_t cpu1_z = ReadThreadCpuNanos();
    const std::uint64_t cnt1_z = ReadCntvctTicks();
    const std::uint64_t tsc1_z = ReadTscTicks();

    const double ns_per_call_z = std::chrono::duration<double, std::nano>(t1_z - t0_z).count() / kPastSamples;
    const double cpu_ns_per_call_z = static_cast<double>(cpu1_z - cpu0_z) / kPastSamples;
#if defined(__x86_64__) || defined(_M_X64)
    const double tsc_per_call_z = static_cast<double>(tsc1_z - tsc0_z) / kPastSamples;
#elif defined(__aarch64__) || defined(__arm64__)
    const double cnt_per_call_z = static_cast<double>(cnt1_z - cnt0_z) / kPastSamples;
#endif

    report << "#### 1. Expired Deadline Fast-Path Overhead (100k calls, DCE protected)\n\n";
    report << "| Scenario | Wall Latency | Thread CPU Time | "
#if defined(__x86_64__) || defined(_M_X64)
           << "TSC ticks (lfenced) |\n";
#elif defined(__aarch64__) || defined(__arm64__)
           << "cntvct ticks (isb) |\n";
#else
           << "Hardware ticks |\n";
#endif
    report << "|---|---:|---:|---:|\n";
    report << "| `PreciseSleepUntil(past)` (nanos) | " << std::fixed << std::setprecision(1)
           << ns_per_call << " ns | " << cpu_ns_per_call << " ns | "
#if defined(__x86_64__) || defined(_M_X64)
           << tsc_per_call << " |\n";
#elif defined(__aarch64__) || defined(__arm64__)
           << cnt_per_call << " |\n";
#else
           << "n/a |\n";
#endif

    report << "| `PreciseSleepUntil(past_us)` (micros) | " << std::fixed << std::setprecision(1)
           << ns_per_call_us << " ns | " << cpu_ns_per_call_us << " ns | "
#if defined(__x86_64__) || defined(_M_X64)
           << tsc_per_call_us << " |\n";
#elif defined(__aarch64__) || defined(__arm64__)
           << cnt_per_call_us << " |\n";
#else
           << "n/a |\n";
#endif

    report << "| `PreciseSleepUntil(zero_time)` (runtime opaque) | " << std::fixed << std::setprecision(1)
           << ns_per_call_z << " ns | " << cpu_ns_per_call_z << " ns | "
#if defined(__x86_64__) || defined(_M_X64)
           << tsc_per_call_z << " |\n\n";
#elif defined(__aarch64__) || defined(__arm64__)
           << cnt_per_call_z << " |\n\n";
#else
           << "n/a |\n\n";
#endif

    report << "#### 2. Sleep Latency & CPU Time (100 samples each, matching statistics)\n\n";
    report << "| Scenario | Target | Wall Median (Overshoot) | Wall Mean | CPU Median | CPU Mean | Early Returns | Target Bound Status |\n";
    report << "|---|---:|---:|---:|---:|---:|:---:|:---:|\n";

    const auto stats_50us = BenchmarkCall(100, 50.0, [] { PreciseSleepUs(50); });
    const bool pass_50us = (stats_50us.wall_median_us < 90.0) && (stats_50us.early_returns == 0);
    report << "| `PreciseSleepUs(50)` | 50.0 µs | "
           << std::fixed << std::setprecision(1) << stats_50us.wall_median_us << " µs (+"
           << stats_50us.overshoot_median_us << " µs) | "
           << stats_50us.wall_mean_us << " µs | "
           << stats_50us.cpu_median_us << " µs | "
           << stats_50us.cpu_mean_us << " µs | "
           << stats_50us.early_returns << " / " << stats_50us.total_samples << " | "
           << (pass_50us ? "✅ PASS (< 90 µs)" : "❌ FAIL (>= 90 µs)") << " |\n";

    const auto stats_1500us = BenchmarkCall(100, 1500.0, [] { PreciseSleepUs(1500); });
    const bool pass_1500us = (stats_1500us.wall_median_us < 1650.0) && (stats_1500us.early_returns == 0);
    report << "| `PreciseSleepUs(1500)` | 1500.0 µs | "
           << std::fixed << std::setprecision(1) << stats_1500us.wall_median_us << " µs (+"
           << stats_1500us.overshoot_median_us << " µs) | "
           << stats_1500us.wall_mean_us << " µs | "
           << stats_1500us.cpu_median_us << " µs | "
           << stats_1500us.cpu_mean_us << " µs | "
           << stats_1500us.early_returns << " / " << stats_1500us.total_samples << " | "
           << (pass_1500us ? "✅ PASS (< 1650 µs)" : "❌ FAIL (>= 1650 µs)") << " |\n";

    const auto stats_until_50us = BenchmarkCall(100, 50.0, [] {
        PreciseSleepUntil(Clock::now() + std::chrono::microseconds(50));
    });
    const bool pass_until_50us = (stats_until_50us.wall_median_us < 90.0) && (stats_until_50us.early_returns == 0);
    report << "| `PreciseSleepUntil(+50µs)` | 50.0 µs | "
           << std::fixed << std::setprecision(1) << stats_until_50us.wall_median_us << " µs (+"
           << stats_until_50us.overshoot_median_us << " µs) | "
           << stats_until_50us.wall_mean_us << " µs | "
           << stats_until_50us.cpu_median_us << " µs | "
           << stats_until_50us.cpu_mean_us << " µs | "
           << stats_until_50us.early_returns << " / " << stats_until_50us.total_samples << " | "
           << (pass_until_50us ? "✅ PASS (< 90 µs)" : "❌ FAIL (>= 90 µs)") << " |\n";

    const auto stats_until_1ms = BenchmarkCall(100, 1000.0, [] {
        PreciseSleepUntil(Clock::now() + std::chrono::milliseconds(1));
    });
    report << "| `PreciseSleepUntil(+1ms)` | 1000.0 µs | "
           << std::fixed << std::setprecision(1) << stats_until_1ms.wall_median_us << " µs (+"
           << stats_until_1ms.overshoot_median_us << " µs) | "
           << stats_until_1ms.wall_mean_us << " µs | "
           << stats_until_1ms.cpu_median_us << " µs | "
           << stats_until_1ms.cpu_mean_us << " µs | "
           << stats_until_1ms.early_returns << " / " << stats_until_1ms.total_samples << " | "
           << (stats_until_1ms.early_returns == 0 ? "✅ 0 early" : "❌ EARLY RETURN") << " |\n\n";

#ifdef APS5_PRECISE_WAIT_TRACE
    report << "#### 3. Instrumented Phase Breakdown for PreciseSleepUs(1500) (100 samples)\n\n";
    std::vector<double> sleep_req_samples;
    std::vector<double> sleep_wall_samples;
    std::vector<double> sleep_overshoot_samples;
    std::vector<double> spin_wall_samples;
    std::vector<double> spin_cpu_samples;
    std::vector<double> spin_unaccounted_samples;

    sleep_req_samples.reserve(100);
    sleep_wall_samples.reserve(100);
    sleep_overshoot_samples.reserve(100);
    spin_wall_samples.reserve(100);
    spin_cpu_samples.reserve(100);
    spin_unaccounted_samples.reserve(100);

    g_preciseWaitTraceActive = true;
    for (int i = 0; i < 100; ++i) {
        PreciseSleepUs(1500);
        const double req_us = static_cast<double>(g_preciseWaitTraceData.requested_sleep_nanos) / 1000.0;
        const double sw_us = static_cast<double>(g_preciseWaitTraceData.t2_nanos - g_preciseWaitTraceData.t1_nanos) / 1000.0;
        const double so_us = sw_us - req_us;
        const double spw_us = static_cast<double>(g_preciseWaitTraceData.t3_nanos - g_preciseWaitTraceData.t2_nanos) / 1000.0;
        const double spc_us = (g_preciseWaitTraceData.cpu3_nanos >= g_preciseWaitTraceData.cpu2_nanos)
            ? static_cast<double>(g_preciseWaitTraceData.cpu3_nanos - g_preciseWaitTraceData.cpu2_nanos) / 1000.0
            : 0.0;
        const double unacc_us = (spw_us >= spc_us) ? (spw_us - spc_us) : 0.0;

        sleep_req_samples.push_back(req_us);
        sleep_wall_samples.push_back(sw_us);
        sleep_overshoot_samples.push_back(so_us);
        spin_wall_samples.push_back(spw_us);
        spin_cpu_samples.push_back(spc_us);
        spin_unaccounted_samples.push_back(unacc_us);
    }
    g_preciseWaitTraceActive = false;

    const auto s_req = ComputePhaseStats(sleep_req_samples);
    const auto s_sw = ComputePhaseStats(sleep_wall_samples);
    const auto s_so = ComputePhaseStats(sleep_overshoot_samples);
    const auto s_spw = ComputePhaseStats(spin_wall_samples);
    const auto s_spc = ComputePhaseStats(spin_cpu_samples);
    const auto s_unacc = ComputePhaseStats(spin_unaccounted_samples);

    report << "| Measurement Phase | Min | Median | p90 | Mean | Max |\n";
    report << "|---|---:|---:|---:|---:|---:|\n";
    report << "| `nanosleep` Requested | " << std::fixed << std::setprecision(1)
           << s_req.min_val << " µs | " << s_req.median_val << " µs | " << s_req.p90_val << " µs | "
           << s_req.mean_val << " µs | " << s_req.max_val << " µs |\n";
    report << "| `nanosleep` Actual Wall | "
           << s_sw.min_val << " µs | " << s_sw.median_val << " µs | " << s_sw.p90_val << " µs | "
           << s_sw.mean_val << " µs | " << s_sw.max_val << " µs |\n";
    report << "| `nanosleep` Overshoot (Actual - Req) | "
           << s_so.min_val << " µs | " << s_so.median_val << " µs | " << s_so.p90_val << " µs | "
           << s_so.mean_val << " µs | " << s_so.max_val << " µs |\n";
    report << "| Spin-phase Wall Duration | "
           << s_spw.min_val << " µs | " << s_spw.median_val << " µs | " << s_spw.p90_val << " µs | "
           << s_spw.mean_val << " µs | " << s_spw.max_val << " µs |\n";
    report << "| Spin-phase Thread CPU Time | "
           << s_spc.min_val << " µs | " << s_spc.median_val << " µs | " << s_spc.p90_val << " µs | "
           << s_spc.mean_val << " µs | " << s_spc.max_val << " µs |\n";
    report << "| Spin-phase Unaccounted Wall Time | "
           << s_unacc.min_val << " µs | " << s_unacc.median_val << " µs | " << s_unacc.p90_val << " µs | "
           << s_unacc.mean_val << " µs | " << s_unacc.max_val << " µs |\n\n";
#endif

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
