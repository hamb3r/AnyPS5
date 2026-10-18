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
#include <thread>
#include <vector>
#include <time.h>
#if defined(__APPLE__)
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <mach/mach_time.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <sys/sysctl.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
#endif

using Clock = std::chrono::steady_clock;

template <typename T>
[[gnu::always_inline]] inline void DoNotOptimize(const T& val) {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" :: "r,m"(val) : "memory");
#endif
}

[[gnu::noinline]] void BenchmarkSleepUntil(Clock::time_point deadline) {
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

struct PercentileStats {
    double p50;
    double p90;
    double p99;
    double mean;
    double max_val;
    double min_val;
};

inline PercentileStats ComputeStats(std::vector<double>& v) {
    if (v.empty()) {
        return PercentileStats{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    std::sort(v.begin(), v.end());
    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    const size_t n = v.size();
    return PercentileStats{
        v[n / 2],
        v[static_cast<size_t>(n * 0.90)],
        v[static_cast<size_t>(n * 0.99)],
        sum / n,
        v.back(),
        v.front()
    };
}

struct ModeResults {
    std::string backend_name;
    bool policy_enabled;
    std::string policy_status;
    std::uint32_t spin_tail_us;
    std::uint64_t total_samples;
    std::uint64_t warmup_samples;
    std::uint64_t early_returns;
    std::uint64_t outliers_count;
    std::uint64_t total_eintr;
    std::uint32_t last_syscall_code;
    std::uint64_t invalid_samples;

    PercentileStats total_wall_us;
    PercentileStats total_overshoot_us;
    PercentileStats sleep_wall_us;
    PercentileStats sleep_overshoot_us;
    PercentileStats spin_wall_us;
    PercentileStats spin_cpu_us;
    PercentileStats spin_unaccounted_us;
    PercentileStats total_cpu_us;
    PercentileStats wall_residual_us;
    PercentileStats cpu_residual_us;
};

inline ModeResults RunIsolatedMode(
    PreciseWaitBackend backend,
    bool enable_policy,
    unsigned long long sleep_micros,
    int warmup_count,
    int sample_count) {

    ModeResults res{};
    res.backend_name = (backend == PreciseWaitBackend::Nanosleep) ? "nanosleep" : "mach_wait_until";
    res.policy_enabled = enable_policy;
    res.total_samples = sample_count;
    res.warmup_samples = warmup_count;

    std::vector<double> v_total_wall;
    std::vector<double> v_total_overshoot;
    std::vector<double> v_sleep_wall;
    std::vector<double> v_sleep_overshoot;
    std::vector<double> v_spin_wall;
    std::vector<double> v_spin_cpu;
    std::vector<double> v_spin_unaccounted;
    std::vector<double> v_total_cpu;
    std::vector<double> v_wall_residual;
    std::vector<double> v_cpu_residual;

    v_total_wall.reserve(sample_count);
    v_total_overshoot.reserve(sample_count);
    v_sleep_wall.reserve(sample_count);
    v_sleep_overshoot.reserve(sample_count);
    v_spin_wall.reserve(sample_count);
    v_spin_cpu.reserve(sample_count);
    v_spin_unaccounted.reserve(sample_count);
    v_total_cpu.reserve(sample_count);
    v_wall_residual.reserve(sample_count);
    v_cpu_residual.reserve(sample_count);

    std::thread worker([&] {
        std::string status = "OFF";
#if defined(__APPLE__)
        if (enable_policy) {
            mach_timebase_info_data_t tb;
            if (mach_timebase_info(&tb) == KERN_SUCCESS && tb.numer > 0) {
                thread_time_constraint_policy_data_t policy;
                const uint64_t periodTicks = (1000000ULL * tb.denom) / tb.numer;
                const uint64_t computationTicks = (500000ULL * tb.denom) / tb.numer;
                if (periodTicks <= UINT32_MAX && computationTicks <= UINT32_MAX) {
                    policy.period = static_cast<uint32_t>(periodTicks);
                    policy.computation = static_cast<uint32_t>(computationTicks);
                    policy.constraint = static_cast<uint32_t>(periodTicks);
                    policy.preemptible = TRUE;
                    const mach_port_t threadPort = mach_thread_self();
                    const kern_return_t kr = thread_policy_set(
                        threadPort,
                        THREAD_TIME_CONSTRAINT_POLICY,
                        reinterpret_cast<thread_policy_t>(&policy),
                        THREAD_TIME_CONSTRAINT_POLICY_COUNT);
                    mach_port_deallocate(mach_task_self(), threadPort);
                    if (kr == KERN_SUCCESS) {
                        status = "ON (kr=0)";
                    } else {
                        status = "FAILED (kr=" + std::to_string(kr) + ")";
                    }
                } else {
                    status = "OVERFLOW";
                }
            } else {
                status = "TIMEBASE_ERR";
            }
        }
#endif
        res.policy_status = status;
        const bool lowSlack = (status.rfind("ON", 0) == 0);
        res.spin_tail_us = lowSlack ? 200 : 500;

        g_preciseWaitTraceBackend = backend;
        g_preciseWaitTraceActive = true;

        for (int i = 0; i < warmup_count; ++i) {
            PreciseSleepUs(sleep_micros);
        }

        const double target_us = static_cast<double>(sleep_micros);

        for (int i = 0; i < sample_count; ++i) {
            PreciseSleepUs(sleep_micros);

            const auto& s = g_preciseWaitTraceSample;
            const bool monotonic = (s.t0 <= s.t1) && (s.t1 <= s.t2) && (s.t2 <= s.t3) && (s.t3 <= s.t4);
            if (!monotonic) {
                res.invalid_samples++;
                continue;
            }

            res.total_eintr += s.eintr_count;
            res.last_syscall_code = s.syscall_return_code;

            const double total_w = static_cast<double>(s.t4 - s.t0) / 1000.0;
            const double prep_w  = static_cast<double>(s.t1 - s.t0) / 1000.0;
            const double sleep_w = static_cast<double>(s.t2 - s.t1) / 1000.0;
            const double spin_w  = static_cast<double>(s.t3 - s.t2) / 1000.0;
            const double post_w  = static_cast<double>(s.t4 - s.t3) / 1000.0;

            const double total_c = (s.cpu4 >= s.cpu0) ? static_cast<double>(s.cpu4 - s.cpu0) / 1000.0 : 0.0;
            const double prep_c  = (s.cpu1 >= s.cpu0) ? static_cast<double>(s.cpu1 - s.cpu0) / 1000.0 : 0.0;
            const double sleep_c = (s.cpu2 >= s.cpu1) ? static_cast<double>(s.cpu2 - s.cpu1) / 1000.0 : 0.0;
            const double spin_c  = (s.cpu3 >= s.cpu2) ? static_cast<double>(s.cpu3 - s.cpu2) / 1000.0 : 0.0;
            const double post_c  = (s.cpu4 >= s.cpu3) ? static_cast<double>(s.cpu4 - s.cpu3) / 1000.0 : 0.0;

            const double w_residual = total_w - (prep_w + sleep_w + spin_w + post_w);
            const double c_residual = total_c - (prep_c + sleep_c + spin_c + post_c);

            const double req_sleep = static_cast<double>(s.req_sleep_nanos) / 1000.0;
            const double sleep_overshoot = sleep_w - req_sleep;
            const double total_overshoot = total_w - target_us;
            const double spin_unacc = (spin_w >= spin_c) ? (spin_w - spin_c) : 0.0;

            const std::uint64_t target_nanos = sleep_micros * 1000ULL;
            if (s.t4 < s.t0 + target_nanos) {
                res.early_returns++;
            }
            if (s.t4 >= s.t0 + (target_nanos * 110ULL / 100ULL)) {
                res.outliers_count++;
            }

            v_total_wall.push_back(total_w);
            v_total_overshoot.push_back(total_overshoot);
            v_sleep_wall.push_back(sleep_w);
            v_sleep_overshoot.push_back(sleep_overshoot);
            v_spin_wall.push_back(spin_w);
            v_spin_cpu.push_back(spin_c);
            v_spin_unaccounted.push_back(spin_unacc);
            v_total_cpu.push_back(total_c);
            v_wall_residual.push_back(w_residual);
            v_cpu_residual.push_back(c_residual);
        }

        g_preciseWaitTraceActive = false;
    });

    worker.join();

    res.total_wall_us = ComputeStats(v_total_wall);
    res.total_overshoot_us = ComputeStats(v_total_overshoot);
    res.sleep_wall_us = ComputeStats(v_sleep_wall);
    res.sleep_overshoot_us = ComputeStats(v_sleep_overshoot);
    res.spin_wall_us = ComputeStats(v_spin_wall);
    res.spin_cpu_us = ComputeStats(v_spin_cpu);
    res.spin_unaccounted_us = ComputeStats(v_spin_unaccounted);
    res.total_cpu_us = ComputeStats(v_total_cpu);
    res.wall_residual_us = ComputeStats(v_wall_residual);
    res.cpu_residual_us = ComputeStats(v_cpu_residual);

    return res;
}

int main() {
    std::ostringstream report;

    report << "### macOS Timer 2x2 Matrix & Unified Trace Benchmark\n\n";

#if defined(__APPLE__)
    struct timespec res_mono{}, res_raw{};
    clock_getres(CLOCK_MONOTONIC, &res_mono);
    clock_getres(CLOCK_MONOTONIC_RAW, &res_raw);

    mach_timebase_info_data_t tb{};
    mach_timebase_info(&tb);

    report << "- **Platform:** macOS ("
#if defined(__aarch64__) || defined(__arm64__)
           << "Apple Silicon arm64"
#elif defined(__x86_64__)
           << "x86_64"
#else
           << "Unknown Arch"
#endif
           << ")\n";
    report << "- **Mach timebase:** " << tb.numer << " / " << tb.denom << "\n";
    report << "- **CLOCK_MONOTONIC resolution:** " << res_mono.tv_nsec << " ns\n";
    report << "- **CLOCK_MONOTONIC_RAW resolution:** " << res_raw.tv_nsec << " ns\n";
#if defined(__aarch64__) || defined(__arm64__)
    report << "- **ARM64 cntfrq_el0:** " << ReadCntfrqHz() << " Hz\n";
#endif
    report << "\n";
#else
    report << "- **Platform:** Non-Apple POSIX\n\n";
#endif

    std::vector<double> v_probe_wall;
    std::vector<double> v_probe_cpu;
    v_probe_wall.reserve(1000);
    v_probe_cpu.reserve(1000);

    for (int i = 0; i < 1000; ++i) {
        const std::uint64_t c0 = ReadThreadCpuNanos();
        const std::uint64_t t0 = PreciseMonotonicNanos();
        const std::uint64_t t1 = PreciseMonotonicNanos();
        const std::uint64_t c1 = ReadThreadCpuNanos();
        const std::uint64_t t2 = PreciseMonotonicNanos();
        const std::uint64_t c2 = ReadThreadCpuNanos();
        const std::uint64_t t3 = PreciseMonotonicNanos();
        const std::uint64_t c3 = ReadThreadCpuNanos();
        const std::uint64_t t4 = PreciseMonotonicNanos();
        const std::uint64_t c4 = ReadThreadCpuNanos();

        v_probe_wall.push_back(static_cast<double>(t4 - t0) / 1000.0);
        v_probe_cpu.push_back((c4 >= c0) ? static_cast<double>(c4 - c0) / 1000.0 : 0.0);
    }
    const auto probe_w_stats = ComputeStats(v_probe_wall);
    const auto probe_c_stats = ComputeStats(v_probe_cpu);

    report << "#### 1. Baseline Probe Overhead Calibration (1000 samples T0..T4 without sleep)\n\n";
    report << "| Probe Metric | p50 | p90 | p99 | Mean | Max |\n";
    report << "|---|---:|---:|---:|---:|---:|\n";
    report << "| T0..T4 Wall Probe Overhead | " << std::fixed << std::setprecision(3)
           << probe_w_stats.p50 * 1000.0 << " ns | " << probe_w_stats.p90 * 1000.0 << " ns | "
           << probe_w_stats.p99 * 1000.0 << " ns | " << probe_w_stats.mean * 1000.0 << " ns | "
           << probe_w_stats.max_val * 1000.0 << " ns |\n";
    report << "| CPU Clock Probe Overhead | "
           << probe_c_stats.p50 * 1000.0 << " ns | " << probe_c_stats.p90 * 1000.0 << " ns | "
           << probe_c_stats.p99 * 1000.0 << " ns | " << probe_c_stats.mean * 1000.0 << " ns | "
           << probe_c_stats.max_val * 1000.0 << " ns |\n\n";

    constexpr int kFastPathSamples = 100000;
    const auto past_deadline = Clock::now() - std::chrono::seconds(1);
    const auto past_us = std::chrono::time_point_cast<std::chrono::microseconds>(past_deadline);
    volatile int64_t runtime_zero_ns = 0;
    DoNotOptimize(runtime_zero_ns);
    const Clock::time_point zero_time{std::chrono::nanoseconds(runtime_zero_ns)};

    const std::uint64_t tsc0 = ReadTscTicks();
    const std::uint64_t cpu0 = ReadThreadCpuNanos();
    const auto ft0 = Clock::now();
    for (int i = 0; i < kFastPathSamples; ++i) {
        BenchmarkSleepUntil(past_deadline);
    }
    const auto ft1 = Clock::now();
    const std::uint64_t cpu1 = ReadThreadCpuNanos();
    const std::uint64_t tsc1 = ReadTscTicks();

    const std::uint64_t tsc0_z = ReadTscTicks();
    const std::uint64_t cpu0_z = ReadThreadCpuNanos();
    const auto ft0_z = Clock::now();
    for (int i = 0; i < kFastPathSamples; ++i) {
        BenchmarkSleepUntil(zero_time);
    }
    const auto ft1_z = Clock::now();
    const std::uint64_t cpu1_z = ReadThreadCpuNanos();
    const std::uint64_t tsc1_z = ReadTscTicks();

    report << "#### 2. Expired Deadline Fast-Path (100k calls on isolated thread, DCE protected)\n\n";
    report << "| Call | Wall Latency | Thread CPU Time | "
#if defined(__x86_64__) || defined(_M_X64)
           << "TSC ticks (lfenced) |\n";
#else
           << "Raw cycles/ticks |\n";
#endif
    report << "|---|---:|---:|---:|\n";
    report << "| `PreciseSleepUntil(past)` | " << std::fixed << std::setprecision(1)
           << (std::chrono::duration<double, std::nano>(ft1 - ft0).count() / kFastPathSamples) << " ns | "
           << (static_cast<double>(cpu1 - cpu0) / kFastPathSamples) << " ns | "
#if defined(__x86_64__) || defined(_M_X64)
           << (static_cast<double>(tsc1 - tsc0) / kFastPathSamples) << " |\n";
#else
           << "n/a |\n";
#endif
    report << "| `PreciseSleepUntil(zero_time)` | "
           << (std::chrono::duration<double, std::nano>(ft1_z - ft0_z).count() / kFastPathSamples) << " ns | "
           << (static_cast<double>(cpu1_z - cpu0_z) / kFastPathSamples) << " ns | "
#if defined(__x86_64__) || defined(_M_X64)
           << (static_cast<double>(tsc1_z - tsc0_z) / kFastPathSamples) << " |\n\n";
#else
           << "n/a |\n\n";
#endif

    report << "#### 3. 2x2 Matrix: Backend vs RT Policy for PreciseSleepUs(1500) (1000 samples, 50 warmup each)\n\n";

    std::vector<ModeResults> matrix_results;
    matrix_results.push_back(RunIsolatedMode(PreciseWaitBackend::Nanosleep, false, 1500, 50, 1000));
    matrix_results.push_back(RunIsolatedMode(PreciseWaitBackend::Nanosleep, true,  1500, 50, 1000));
#if defined(__APPLE__)
    matrix_results.push_back(RunIsolatedMode(PreciseWaitBackend::MachWaitUntil, false, 1500, 50, 1000));
    matrix_results.push_back(RunIsolatedMode(PreciseWaitBackend::MachWaitUntil, true,  1500, 50, 1000));
#endif

    report << "| Backend | Policy | Spin Tail | Wall p50 | Wall p90 | Wall p99 | Wall Max | Wall Mean | Overshoot p50 | CPU p50 | CPU Mean | Outliers (>1650µs) | Early | EINTR |\n";
    report << "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|\n";

    for (const auto& m : matrix_results) {
        report << "| `" << m.backend_name << "` | "
               << (m.policy_enabled ? "**ON**" : "OFF") << " | "
               << m.spin_tail_us << " µs | "
               << std::fixed << std::setprecision(1)
               << m.total_wall_us.p50 << " µs | "
               << m.total_wall_us.p90 << " µs | "
               << m.total_wall_us.p99 << " µs | "
               << m.total_wall_us.max_val << " µs | "
               << m.total_wall_us.mean << " µs | +"
               << m.total_overshoot_us.p50 << " µs | "
               << m.total_cpu_us.p50 << " µs | "
               << m.total_cpu_us.mean << " µs | "
               << m.outliers_count << " / " << m.total_samples << " | "
               << m.early_returns << " | "
               << m.total_eintr << " |\n";
    }
    report << "\n";

    report << "#### 4. Detailed Phase Deconstruction (p50 / p90 / p99 / Mean)\n\n";
    for (const auto& m : matrix_results) {
        report << "##### Mode: `" << m.backend_name << "` with Policy " << (m.policy_enabled ? "ON" : "OFF")
               << " (Residual Wall p50=" << std::setprecision(3) << m.wall_residual_us.p50 * 1000.0 << " ns, "
               << "Invalid samples=" << m.invalid_samples << ")\n\n";
        report << "| Phase | p50 | p90 | p99 | Mean | Max |\n";
        report << "|---|---:|---:|---:|---:|---:|\n";
        report << "| Syscall Wall Duration ($T_2 - T_1$) | " << std::fixed << std::setprecision(1)
               << m.sleep_wall_us.p50 << " µs | " << m.sleep_wall_us.p90 << " µs | "
               << m.sleep_wall_us.p99 << " µs | " << m.sleep_wall_us.mean << " µs | "
               << m.sleep_wall_us.max_val << " µs |\n";
        report << "| Syscall Overshoot ($Actual - Req$) | "
               << m.sleep_overshoot_us.p50 << " µs | " << m.sleep_overshoot_us.p90 << " µs | "
               << m.sleep_overshoot_us.p99 << " µs | " << m.sleep_overshoot_us.mean << " µs | "
               << m.sleep_overshoot_us.max_val << " µs |\n";
        report << "| Spin-phase Wall Duration ($T_3 - T_2$) | "
               << m.spin_wall_us.p50 << " µs | " << m.spin_wall_us.p90 << " µs | "
               << m.spin_wall_us.p99 << " µs | " << m.spin_wall_us.mean << " µs | "
               << m.spin_wall_us.max_val << " µs |\n";
        report << "| Spin-phase Thread CPU ($CPU_3 - CPU_2$) | "
               << m.spin_cpu_us.p50 << " µs | " << m.spin_cpu_us.p90 << " µs | "
               << m.spin_cpu_us.p99 << " µs | " << m.spin_cpu_us.mean << " µs | "
               << m.spin_cpu_us.max_val << " µs |\n";
        report << "| Spin-phase Unaccounted ($Wall - CPU$) | "
               << m.spin_unaccounted_us.p50 << " µs | " << m.spin_unaccounted_us.p90 << " µs | "
               << m.spin_unaccounted_us.p99 << " µs | " << m.spin_unaccounted_us.mean << " µs | "
               << m.spin_unaccounted_us.max_val << " µs |\n\n";
    }

    std::cout << report.str();

    if (const char* step_summary = std::getenv("GITHUB_STEP_SUMMARY")) {
        std::ofstream summary_file(step_summary, std::ios::app);
        if (summary_file.is_open()) {
            summary_file << report.str();
        }
    }

    for (const auto& m : matrix_results) {
        if (m.early_returns > 0) {
            std::cerr << "FATAL: Mode " << m.backend_name << " returned BEFORE deadline!\n";
            return 1;
        }
    }

    return 0;
}
