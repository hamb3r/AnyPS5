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

struct ModeConfig {
    PreciseWaitBackend backend;
    bool enable_policy;
    std::string backend_name;
};

struct ModeAccumulator {
    ModeConfig config;
    std::string policy_status;
    std::uint32_t spin_tail_us = 0;
    std::uint64_t total_samples = 0;
    std::uint64_t early_returns = 0;
    std::uint64_t outliers_count = 0;
    std::uint64_t total_eintr = 0;
    std::uint64_t syscall_errors = 0;
    std::uint64_t policy_failures = 0;
    std::uint64_t invalid_samples = 0;

    std::vector<double> v_total_wall;
    std::vector<double> v_total_overshoot;
    std::vector<double> v_sleep_wall;
    std::vector<double> v_sleep_overshoot;
    std::vector<double> v_spin_wall;
    std::vector<double> v_spin_cpu;
    std::vector<double> v_spin_unaccounted;
    std::vector<double> v_post_wall;
    std::vector<double> v_post_cpu;
    std::vector<double> v_total_cpu;
    std::vector<double> v_wall_residual;
    std::vector<double> v_cpu_residual;

    std::vector<double> v_cal_overshoot;
    std::vector<double> v_cal_wall;
    std::vector<double> v_cal_spin_cpu;
    std::vector<double> v_cal_total_cpu;

    PercentileStats s_total_wall;
    PercentileStats s_total_overshoot;
    PercentileStats s_sleep_wall;
    PercentileStats s_sleep_overshoot;
    PercentileStats s_spin_wall;
    PercentileStats s_spin_cpu;
    PercentileStats s_spin_unaccounted;
    PercentileStats s_post_wall;
    PercentileStats s_post_cpu;
    PercentileStats s_total_cpu;
    PercentileStats s_wall_residual;
    PercentileStats s_cpu_residual;

    PercentileStats s_cal_overshoot;
    PercentileStats s_cal_wall;
    PercentileStats s_cal_spin_cpu;
    PercentileStats s_cal_total_cpu;
};

struct ProbeCalibrationResults {
    PercentileStats total_wall;
    PercentileStats total_cpu;
    PercentileStats spin_wall;
    PercentileStats spin_cpu;
    PercentileStats post_wall;
    PercentileStats post_cpu;
};

inline ProbeCalibrationResults MeasureProbeCalibration(int warmup_count, int sample_count) {
    ProbeCalibrationResults res{};
    std::vector<double> v_total_w;
    std::vector<double> v_total_c;
    std::vector<double> v_spin_w;
    std::vector<double> v_spin_c;
    std::vector<double> v_post_w;
    std::vector<double> v_post_c;

    v_total_w.reserve(sample_count);
    v_total_c.reserve(sample_count);
    v_spin_w.reserve(sample_count);
    v_spin_c.reserve(sample_count);
    v_post_w.reserve(sample_count);
    v_post_c.reserve(sample_count);

    std::thread worker([&] {
        for (int i = 0; i < warmup_count; ++i) {
            const std::uint64_t t0 = PreciseMonotonicNanos();
            const std::uint64_t c0 = ReadThreadCpuNanos();
            const std::uint64_t t1 = PreciseMonotonicNanos();
            const std::uint64_t c1 = ReadThreadCpuNanos();
            const std::uint64_t t2 = PreciseMonotonicNanos();
            const std::uint64_t c2 = ReadThreadCpuNanos();
            const std::uint64_t t3 = PreciseMonotonicNanos();
            const std::uint64_t c3 = ReadThreadCpuNanos();
            const std::uint64_t t4 = PreciseMonotonicNanos();
            const std::uint64_t c4 = ReadThreadCpuNanos();
            DoNotOptimize(t0);
            DoNotOptimize(c0);
            DoNotOptimize(t1);
            DoNotOptimize(c1);
            DoNotOptimize(t2);
            DoNotOptimize(c2);
            DoNotOptimize(t3);
            DoNotOptimize(c3);
            DoNotOptimize(t4);
            DoNotOptimize(c4);
        }

        for (int i = 0; i < sample_count; ++i) {
            const std::uint64_t t0 = PreciseMonotonicNanos();
            const std::uint64_t c0 = ReadThreadCpuNanos();
            const std::uint64_t t1 = PreciseMonotonicNanos();
            const std::uint64_t c1 = ReadThreadCpuNanos();
            const std::uint64_t t2 = PreciseMonotonicNanos();
            const std::uint64_t c2 = ReadThreadCpuNanos();
            const std::uint64_t t3 = PreciseMonotonicNanos();
            const std::uint64_t c3 = ReadThreadCpuNanos();
            const std::uint64_t t4 = PreciseMonotonicNanos();
            const std::uint64_t c4 = ReadThreadCpuNanos();

            v_total_w.push_back(static_cast<double>(t4 - t0) / 1000.0);
            v_total_c.push_back((c4 >= c0) ? static_cast<double>(c4 - c0) / 1000.0 : 0.0);
            v_spin_w.push_back(static_cast<double>(t3 - t2) / 1000.0);
            v_spin_c.push_back((c3 >= c2) ? static_cast<double>(c3 - c2) / 1000.0 : 0.0);
            v_post_w.push_back(static_cast<double>(t4 - t3) / 1000.0);
            v_post_c.push_back((c4 >= c3) ? static_cast<double>(c4 - c3) / 1000.0 : 0.0);
        }
    });
    worker.join();

    res.total_wall = ComputeStats(v_total_w);
    res.total_cpu = ComputeStats(v_total_c);
    res.spin_wall = ComputeStats(v_spin_w);
    res.spin_cpu = ComputeStats(v_spin_c);
    res.post_wall = ComputeStats(v_post_w);
    res.post_cpu = ComputeStats(v_post_c);
    return res;
}

struct FastPathResults {
    double past_wall_ns;
    double past_cpu_ns;
    double past_tsc;
    double zero_wall_ns;
    double zero_cpu_ns;
    double zero_tsc;
};

inline FastPathResults MeasureFastPath(int count) {
    FastPathResults res{};
    std::thread worker([&] {
        const auto past_deadline = Clock::now() - std::chrono::seconds(1);
        volatile int64_t runtime_zero_ns = 0;
        DoNotOptimize(runtime_zero_ns);
        const Clock::time_point zero_time{std::chrono::nanoseconds(runtime_zero_ns)};

        for (int i = 0; i < 5000; ++i) {
            BenchmarkSleepUntil(past_deadline);
            BenchmarkSleepUntil(zero_time);
        }

        const std::uint64_t tsc0 = ReadTscTicks();
        const std::uint64_t cpu0 = ReadThreadCpuNanos();
        const auto ft0 = Clock::now();
        for (int i = 0; i < count; ++i) {
            BenchmarkSleepUntil(past_deadline);
        }
        const auto ft1 = Clock::now();
        const std::uint64_t cpu1 = ReadThreadCpuNanos();
        const std::uint64_t tsc1 = ReadTscTicks();

        const std::uint64_t tsc0_z = ReadTscTicks();
        const std::uint64_t cpu0_z = ReadThreadCpuNanos();
        const auto ft0_z = Clock::now();
        for (int i = 0; i < count; ++i) {
            BenchmarkSleepUntil(zero_time);
        }
        const auto ft1_z = Clock::now();
        const std::uint64_t cpu1_z = ReadThreadCpuNanos();
        const std::uint64_t tsc1_z = ReadTscTicks();

        res.past_wall_ns = std::chrono::duration<double, std::nano>(ft1 - ft0).count() / count;
        res.past_cpu_ns = static_cast<double>(cpu1 - cpu0) / count;
        res.past_tsc = static_cast<double>(tsc1 - tsc0) / count;

        res.zero_wall_ns = std::chrono::duration<double, std::nano>(ft1_z - ft0_z).count() / count;
        res.zero_cpu_ns = static_cast<double>(cpu1_z - cpu0_z) / count;
        res.zero_tsc = static_cast<double>(tsc1_z - tsc0_z) / count;
    });
    worker.join();
    return res;
}

inline void RunModeSlice(
    ModeAccumulator& acc,
    unsigned long long sleep_micros,
    int warmup_count,
    int sample_count,
    double probe_total_w_p50,
    double probe_total_c_p50,
    double probe_spin_c_p50) {

    std::thread worker([&] {
        std::string status = "OFF";
#if defined(__APPLE__)
        if (acc.config.enable_policy) {
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
        if (acc.config.enable_policy) {
            if (status != "ON (kr=0)") {
                acc.policy_failures++;
            }
            acc.policy_status = status;
            acc.spin_tail_us = 200;
        } else {
            acc.policy_status = "OFF";
            acc.spin_tail_us = 500;
        }

        g_preciseWaitTraceBackend = acc.config.backend;
        g_preciseWaitTraceAllowLowSlack = acc.config.enable_policy;
        g_preciseWaitTraceActive = true;

        for (int i = 0; i < warmup_count; ++i) {
            PreciseSleepUs(sleep_micros);
        }

        const double target_us = static_cast<double>(sleep_micros);
        const std::uint64_t target_nanos = sleep_micros * 1000ULL;
        const std::uint64_t expected_tail_nanos = acc.config.enable_policy ? 200000ULL : 500000ULL;

        for (int i = 0; i < sample_count; ++i) {
            PreciseSleepUs(sleep_micros);
            const std::uint64_t t4 = PreciseMonotonicNanos();
            const std::uint64_t cpu4 = ReadThreadCpuNanos();
            g_preciseWaitTraceSample.t4 = t4;
            g_preciseWaitTraceSample.cpu4 = cpu4;

            const auto& s = g_preciseWaitTraceSample;
            if (s.syscall_return_code != 0) {
                acc.syscall_errors++;
                acc.invalid_samples++;
                continue;
            }
            const bool monotonic = (s.t0 <= s.t1) && (s.t1 <= s.t2) && (s.t2 <= s.t3) && (s.t3 <= s.t4);
            if (!monotonic) {
                acc.invalid_samples++;
                continue;
            }
            if (s.spin_tail_nanos != expected_tail_nanos) {
                acc.invalid_samples++;
                continue;
            }

            acc.total_samples++;
            acc.total_eintr += s.eintr_count;

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

            const double cal_total_w = total_w - probe_total_w_p50;
            const double cal_overshoot = total_overshoot - probe_total_w_p50;
            const double cal_spin_c = spin_c - probe_spin_c_p50;
            const double cal_total_c = total_c - probe_total_c_p50;

            if (s.t4 < s.t0 + target_nanos) {
                acc.early_returns++;
            }
            if (s.t4 >= s.t0 + (target_nanos * 110ULL / 100ULL)) {
                acc.outliers_count++;
            }

            acc.v_total_wall.push_back(total_w);
            acc.v_total_overshoot.push_back(total_overshoot);
            acc.v_sleep_wall.push_back(sleep_w);
            acc.v_sleep_overshoot.push_back(sleep_overshoot);
            acc.v_spin_wall.push_back(spin_w);
            acc.v_spin_cpu.push_back(spin_c);
            acc.v_spin_unaccounted.push_back(spin_unacc);
            acc.v_post_wall.push_back(post_w);
            acc.v_post_cpu.push_back(post_c);
            acc.v_total_cpu.push_back(total_c);
            acc.v_wall_residual.push_back(w_residual);
            acc.v_cpu_residual.push_back(c_residual);

            acc.v_cal_wall.push_back(cal_total_w);
            acc.v_cal_overshoot.push_back(cal_overshoot);
            acc.v_cal_spin_cpu.push_back(cal_spin_c);
            acc.v_cal_total_cpu.push_back(cal_total_c);
        }

        g_preciseWaitTraceActive = false;
    });

    worker.join();
}

inline void FinalizeStats(ModeAccumulator& m) {
    m.s_total_wall = ComputeStats(m.v_total_wall);
    m.s_total_overshoot = ComputeStats(m.v_total_overshoot);
    m.s_sleep_wall = ComputeStats(m.v_sleep_wall);
    m.s_sleep_overshoot = ComputeStats(m.v_sleep_overshoot);
    m.s_spin_wall = ComputeStats(m.v_spin_wall);
    m.s_spin_cpu = ComputeStats(m.v_spin_cpu);
    m.s_spin_unaccounted = ComputeStats(m.v_spin_unaccounted);
    m.s_post_wall = ComputeStats(m.v_post_wall);
    m.s_post_cpu = ComputeStats(m.v_post_cpu);
    m.s_total_cpu = ComputeStats(m.v_total_cpu);
    m.s_wall_residual = ComputeStats(m.v_wall_residual);
    m.s_cpu_residual = ComputeStats(m.v_cpu_residual);

    m.s_cal_wall = ComputeStats(m.v_cal_wall);
    m.s_cal_overshoot = ComputeStats(m.v_cal_overshoot);
    m.s_cal_spin_cpu = ComputeStats(m.v_cal_spin_cpu);
    m.s_cal_total_cpu = ComputeStats(m.v_cal_total_cpu);
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

    const auto calibration = MeasureProbeCalibration(50, 1000);

    report << "#### 1. Baseline Probe Overhead Calibration (1000 samples on isolated worker thread)\n\n";
    report << "| Probe Metric | p50 | p90 | p99 | Mean | Max |\n";
    report << "|---|---:|---:|---:|---:|---:|\n";
    report << "| Total Probe Wall ($T_4 - T_0$) | " << std::fixed << std::setprecision(1)
           << calibration.total_wall.p50 * 1000.0 << " ns | " << calibration.total_wall.p90 * 1000.0 << " ns | "
           << calibration.total_wall.p99 * 1000.0 << " ns | " << calibration.total_wall.mean * 1000.0 << " ns | "
           << calibration.total_wall.max_val * 1000.0 << " ns |\n";
    report << "| Total Probe Thread CPU ($CPU_4 - CPU_0$) | "
           << calibration.total_cpu.p50 * 1000.0 << " ns | " << calibration.total_cpu.p90 * 1000.0 << " ns | "
           << calibration.total_cpu.p99 * 1000.0 << " ns | " << calibration.total_cpu.mean * 1000.0 << " ns | "
           << calibration.total_cpu.max_val * 1000.0 << " ns |\n";
    report << "| Spin Probe Wall ($T_3 - T_2$) | "
           << calibration.spin_wall.p50 * 1000.0 << " ns | " << calibration.spin_wall.p90 * 1000.0 << " ns | "
           << calibration.spin_wall.p99 * 1000.0 << " ns | " << calibration.spin_wall.mean * 1000.0 << " ns | "
           << calibration.spin_wall.max_val * 1000.0 << " ns |\n";
    report << "| Spin Probe Thread CPU ($CPU_3 - CPU_2$) | "
           << calibration.spin_cpu.p50 * 1000.0 << " ns | " << calibration.spin_cpu.p90 * 1000.0 << " ns | "
           << calibration.spin_cpu.p99 * 1000.0 << " ns | " << calibration.spin_cpu.mean * 1000.0 << " ns | "
           << calibration.spin_cpu.max_val * 1000.0 << " ns |\n";
    report << "| Post Probe Wall ($T_4 - T_3$) | "
           << calibration.post_wall.p50 * 1000.0 << " ns | " << calibration.post_wall.p90 * 1000.0 << " ns | "
           << calibration.post_wall.p99 * 1000.0 << " ns | " << calibration.post_wall.mean * 1000.0 << " ns | "
           << calibration.post_wall.max_val * 1000.0 << " ns |\n";
    report << "| Post Probe Thread CPU ($CPU_4 - CPU_3$) | "
           << calibration.post_cpu.p50 * 1000.0 << " ns | " << calibration.post_cpu.p90 * 1000.0 << " ns | "
           << calibration.post_cpu.p99 * 1000.0 << " ns | " << calibration.post_cpu.mean * 1000.0 << " ns | "
           << calibration.post_cpu.max_val * 1000.0 << " ns |\n\n";

    constexpr int kFastPathSamples = 100000;
    const auto fast_results = MeasureFastPath(kFastPathSamples);

    report << "#### 2. Expired Deadline Fast-Path (100k calls on isolated worker thread, DCE protected)\n\n";
    report << "| Call | Wall Latency | Thread CPU Time | "
#if defined(__x86_64__) || defined(_M_X64)
           << "TSC ticks (lfenced) |\n";
#else
           << "Raw cycles/ticks |\n";
#endif
    report << "|---|---:|---:|---:|\n";
    report << "| `PreciseSleepUntil(past)` | " << std::fixed << std::setprecision(1)
           << fast_results.past_wall_ns << " ns | "
           << fast_results.past_cpu_ns << " ns | "
#if defined(__x86_64__) || defined(_M_X64)
           << fast_results.past_tsc << " |\n";
#else
           << "n/a |\n";
#endif
    report << "| `PreciseSleepUntil(zero_time)` | "
           << fast_results.zero_wall_ns << " ns | "
           << fast_results.zero_cpu_ns << " ns | "
#if defined(__x86_64__) || defined(_M_X64)
           << fast_results.zero_tsc << " |\n\n";
#else
           << "n/a |\n\n";
#endif

    report << "#### 3. Raw Measurements: 2x2 Matrix for PreciseSleepUs(1500) (10 Interleaved Rounds, 1000 Samples per Mode)\n\n";

    std::vector<ModeAccumulator> modes;
    modes.push_back(ModeAccumulator{{PreciseWaitBackend::Nanosleep, false, "nanosleep"}});
    modes.push_back(ModeAccumulator{{PreciseWaitBackend::Nanosleep, true,  "nanosleep"}});
#if defined(__APPLE__)
    modes.push_back(ModeAccumulator{{PreciseWaitBackend::MachWaitUntil, false, "mach_wait_until"}});
    modes.push_back(ModeAccumulator{{PreciseWaitBackend::MachWaitUntil, true,  "mach_wait_until"}});
#endif

    const size_t num_modes = modes.size();
    const int kRounds = 10;
    const int kWarmupPerRound = 10;
    const int kSamplesPerRound = 100;

    const std::vector<std::vector<int>> round_permutations = {
        {0, 1, 2, 3},
        {3, 2, 1, 0},
        {1, 3, 0, 2},
        {2, 0, 3, 1},
        {0, 2, 3, 1},
        {3, 1, 0, 2},
        {2, 3, 1, 0},
        {1, 0, 2, 3},
        {0, 3, 1, 2},
        {2, 1, 0, 3}
    };

    for (int r = 0; r < kRounds; ++r) {
        const auto& perm = round_permutations[r % round_permutations.size()];
        for (int idx : perm) {
            if (static_cast<size_t>(idx) < num_modes) {
                RunModeSlice(modes[idx], 1500, kWarmupPerRound, kSamplesPerRound,
                    calibration.total_wall.p50,
                    calibration.total_cpu.p50,
                    calibration.spin_cpu.p50);
            }
        }
    }

    for (auto& m : modes) {
        FinalizeStats(m);
    }

    report << "| Backend | Policy | Spin Tail | Wall p50 | Wall p90 | Wall p99 | Wall Max | Wall Mean | Overshoot p50 | Total CPU p50 | Spin CPU p50 | Outliers (>1650µs) | Early | Syscall Errors | Policy Failures | Invalid |\n";
    report << "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|\n";

    for (const auto& m : modes) {
        report << "| `" << m.config.backend_name << "` | "
               << (m.config.enable_policy ? "**ON**" : "OFF") << " | "
               << m.spin_tail_us << " µs | "
               << std::fixed << std::setprecision(1)
               << m.s_total_wall.p50 << " µs | "
               << m.s_total_wall.p90 << " µs | "
               << m.s_total_wall.p99 << " µs | "
               << m.s_total_wall.max_val << " µs | "
               << m.s_total_wall.mean << " µs | +"
               << m.s_total_overshoot.p50 << " µs | "
               << m.s_total_cpu.p50 << " µs | "
               << m.s_spin_cpu.p50 << " µs | "
               << m.outliers_count << " / " << m.total_samples << " | "
               << m.early_returns << " | "
               << m.syscall_errors << " | "
               << m.policy_failures << " | "
               << m.invalid_samples << " |\n";
    }
    report << "\n";

    report << "#### 4. Calibrated Measurements (Probe Baseline Subtracted, Unclamped)\n\n";
    report << "| Backend | Policy | Calibrated Overshoot p50 | Calibrated Wall p50 | Calibrated Wall p90 | Calibrated Wall p99 | Calibrated Spin CPU p50 | Calibrated Total CPU p50 |\n";
    report << "|---|:---:|---:|---:|---:|---:|---:|---:|\n";

    for (const auto& m : modes) {
        report << "| `" << m.config.backend_name << "` | "
               << (m.config.enable_policy ? "**ON**" : "OFF") << " | "
               << std::fixed << std::setprecision(1)
               << (m.s_cal_overshoot.p50 >= 0 ? "+" : "") << m.s_cal_overshoot.p50 << " µs | "
               << m.s_cal_wall.p50 << " µs | "
               << m.s_cal_wall.p90 << " µs | "
               << m.s_cal_wall.p99 << " µs | "
               << m.s_cal_spin_cpu.p50 << " µs | "
               << m.s_cal_total_cpu.p50 << " µs |\n";
    }
    report << "\n";

    report << "#### 5. Detailed Phase Deconstruction (Raw)\n\n";
    for (const auto& m : modes) {
        report << "##### Mode: `" << m.config.backend_name << "` with Policy " << (m.config.enable_policy ? "ON" : "OFF")
               << " (Residual Wall p50=" << std::setprecision(3) << m.s_wall_residual.p50 * 1000.0 << " ns, "
               << "Invalid=" << m.invalid_samples << ", Syscall Errors=" << m.syscall_errors << ")\n\n";
        report << "| Phase | p50 | p90 | p99 | Mean | Max |\n";
        report << "|---|---:|---:|---:|---:|---:|\n";
        report << "| Syscall Wall Duration ($T_2 - T_1$) | " << std::fixed << std::setprecision(1)
               << m.s_sleep_wall.p50 << " µs | " << m.s_sleep_wall.p90 << " µs | "
               << m.s_sleep_wall.p99 << " µs | " << m.s_sleep_wall.mean << " µs | "
               << m.s_sleep_wall.max_val << " µs |\n";
        report << "| Syscall Overshoot ($Actual - Req$) | "
               << m.s_sleep_overshoot.p50 << " µs | " << m.s_sleep_overshoot.p90 << " µs | "
               << m.s_sleep_overshoot.p99 << " µs | " << m.s_sleep_overshoot.mean << " µs | "
               << m.s_sleep_overshoot.max_val << " µs |\n";
        report << "| Spin-phase Wall Duration ($T_3 - T_2$) | "
               << m.s_spin_wall.p50 << " µs | " << m.s_spin_wall.p90 << " µs | "
               << m.s_spin_wall.p99 << " µs | " << m.s_spin_wall.mean << " µs | "
               << m.s_spin_wall.max_val << " µs |\n";
        report << "| Spin-phase Thread CPU ($CPU_3 - CPU_2$) | "
               << m.s_spin_cpu.p50 << " µs | " << m.s_spin_cpu.p90 << " µs | "
               << m.s_spin_cpu.p99 << " µs | " << m.s_spin_cpu.mean << " µs | "
               << m.s_spin_cpu.max_val << " µs |\n";
        report << "| Spin-phase Unaccounted ($Wall - CPU$) | "
               << m.s_spin_unaccounted.p50 << " µs | " << m.s_spin_unaccounted.p90 << " µs | "
               << m.s_spin_unaccounted.p99 << " µs | " << m.s_spin_unaccounted.mean << " µs | "
               << m.s_spin_unaccounted.max_val << " µs |\n";
        report << "| Post-spin & Wrapper Exit Wall ($T_4 - T_3$) | "
               << m.s_post_wall.p50 << " µs | " << m.s_post_wall.p90 << " µs | "
               << m.s_post_wall.p99 << " µs | " << m.s_post_wall.mean << " µs | "
               << m.s_post_wall.max_val << " µs |\n";
        report << "| Post-spin & Wrapper Exit CPU ($CPU_4 - CPU_3$) | "
               << m.s_post_cpu.p50 << " µs | " << m.s_post_cpu.p90 << " µs | "
               << m.s_post_cpu.p99 << " µs | " << m.s_post_cpu.mean << " µs | "
               << m.s_post_cpu.max_val << " µs |\n";
        report << "| Total Call Wall ($T_4 - T_0$) | "
               << m.s_total_wall.p50 << " µs | " << m.s_total_wall.p90 << " µs | "
               << m.s_total_wall.p99 << " µs | " << m.s_total_wall.mean << " µs | "
               << m.s_total_wall.max_val << " µs |\n";
        report << "| Total Call Thread CPU ($CPU_4 - CPU_0$) | "
               << m.s_total_cpu.p50 << " µs | " << m.s_total_cpu.p90 << " µs | "
               << m.s_total_cpu.p99 << " µs | " << m.s_total_cpu.mean << " µs | "
               << m.s_total_cpu.max_val << " µs |\n\n";
    }

    std::cout << report.str();

    if (const char* step_summary = std::getenv("GITHUB_STEP_SUMMARY")) {
        std::ofstream summary_file(step_summary, std::ios::app);
        if (summary_file.is_open()) {
            summary_file << report.str();
        }
    }

    for (const auto& m : modes) {
        if (m.early_returns > 0) {
            std::cerr << "FATAL: Mode " << m.config.backend_name << " returned BEFORE deadline!\n";
            return 1;
        }
        if (m.syscall_errors > 0) {
            std::cerr << "FATAL: Mode " << m.config.backend_name << " encountered syscall errors!\n";
            return 1;
        }
        if (m.policy_failures > 0) {
            std::cerr << "FATAL: Mode " << m.config.backend_name << " encountered policy failures!\n";
            return 1;
        }
        if (m.invalid_samples > 0) {
            std::cerr << "FATAL: Mode " << m.config.backend_name << " encountered invalid samples!\n";
            return 1;
        }
    }

    return 0;
}
