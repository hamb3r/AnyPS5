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
    PreciseWaitSpinLoop spin_loop = PreciseWaitSpinLoop::MonotonicNanos;
    uint32_t policy_computation_us = 500;
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

    std::vector<double> round_p50_wall;

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
            const auto& tb = PreciseMachTimebase();
            if (tb.numer > 0 && tb.denom > 0) {
                thread_time_constraint_policy_data_t policy;
                const uint64_t periodTicks = (1000000ULL * tb.denom) / tb.numer;
                const uint64_t compUs = acc.config.policy_computation_us > 0 ? acc.config.policy_computation_us : 500;
                const uint64_t computationTicks = (compUs * 1000ULL * tb.denom) / tb.numer;
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
        g_preciseWaitTraceSpinLoop = acc.config.spin_loop;
        g_preciseWaitTracePolicyComputationUs = acc.config.policy_computation_us;
        g_preciseWaitTraceActive = true;

        for (int i = 0; i < warmup_count; ++i) {
            PreciseSleepUs(sleep_micros);
        }

        const double target_us = static_cast<double>(sleep_micros);
        const std::uint64_t target_nanos = sleep_micros * 1000ULL;
        const std::uint64_t expected_tail_nanos = acc.config.enable_policy ? 200000ULL : 500000ULL;

        std::vector<double> slice_wall;
        slice_wall.reserve(sample_count);

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

            slice_wall.push_back(total_w);

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

        if (!slice_wall.empty()) {
            acc.round_p50_wall.push_back(ComputeStats(slice_wall).p50);
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

#if !defined(__APPLE__)
struct mach_timebase_info_data_t {
    uint32_t numer{1};
    uint32_t denom{1};
};
#endif

struct ClockProbeStats {
    double min_ns;
    double p50_ns;
    double p90_ns;
    double p99_ns;
    double mean_ns;
    double max_ns;
};

inline ClockProbeStats MeasureClockProbe(int type, int chunks = 1000, int per_chunk = 100) {
    std::vector<double> v;
    v.reserve(chunks);
    for (int c = 0; c < chunks; ++c) {
        const auto t0 = Clock::now();
        for (int i = 0; i < per_chunk; ++i) {
            if (type == 0) {
                struct timespec ts{};
#if defined(CLOCK_MONOTONIC_RAW)
                clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#endif
                DoNotOptimize(ts.tv_nsec);
            } else if (type == 1) {
#if defined(__APPLE__)
                const uint64_t ticks = mach_absolute_time();
                DoNotOptimize(ticks);
#endif
            } else if (type == 2) {
#if defined(__APPLE__) && defined(CLOCK_MONOTONIC_RAW)
                const uint64_t ns = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
                DoNotOptimize(ns);
#endif
            }
        }
        const auto t1 = Clock::now();
        const double elapsed_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        v.push_back(elapsed_ns / static_cast<double>(per_chunk));
    }
    const auto s = ComputeStats(v);
    return ClockProbeStats{s.min_val, s.p50, s.p90, s.p99, s.mean, s.max_val};
}

inline ClockProbeStats MeasureSingleReadProbe(int type, int count = 10000) {
    std::vector<double> v;
    v.reserve(count);
    for (int i = 0; i < count; ++i) {
        const auto t0 = Clock::now();
        if (type == 0) {
            struct timespec ts{};
#if defined(CLOCK_MONOTONIC_RAW)
            clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#endif
            DoNotOptimize(ts.tv_nsec);
        } else if (type == 1) {
#if defined(__APPLE__)
            const uint64_t ticks = mach_absolute_time();
            DoNotOptimize(ticks);
#endif
        } else if (type == 2) {
#if defined(__APPLE__) && defined(CLOCK_MONOTONIC_RAW)
            const uint64_t ns = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
            DoNotOptimize(ns);
#endif
        }
        const auto t1 = Clock::now();
        v.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    const auto s = ComputeStats(v);
    return ClockProbeStats{s.min_val, s.p50, s.p90, s.p99, s.mean, s.max_val};
}

struct ClockSpinAccumulator {
    int clock_type;
    std::string clock_name;
    uint64_t target_micros;
    uint64_t total_samples = 0;
    uint64_t early_returns = 0;

    std::vector<double> v_wall;
    std::vector<double> v_cpu;
    std::vector<double> v_overshoot;
    std::vector<double> v_unaccounted;

    PercentileStats s_wall;
    PercentileStats s_cpu;
    PercentileStats s_overshoot;
    PercentileStats s_unaccounted;
};

inline void RunClockSpinSlice(
    ClockSpinAccumulator& acc,
    int warmup_count,
    int sample_count,
    mach_timebase_info_data_t tb) {
    (void)tb;
    std::thread worker([&] {
#if defined(__APPLE__)
        const auto& actual_tb = PreciseMachTimebase();
        if (actual_tb.numer > 0 && actual_tb.denom > 0) {
            thread_time_constraint_policy_data_t policy;
            const uint64_t periodTicks = (1000000ULL * actual_tb.denom) / actual_tb.numer;
            const uint64_t computationTicks = (500000ULL * actual_tb.denom) / actual_tb.numer;
            if (periodTicks <= UINT32_MAX && computationTicks <= UINT32_MAX) {
                policy.period = static_cast<uint32_t>(periodTicks);
                policy.computation = static_cast<uint32_t>(computationTicks);
                policy.constraint = static_cast<uint32_t>(periodTicks);
                policy.preemptible = TRUE;
                const mach_port_t threadPort = mach_thread_self();
                thread_policy_set(
                    threadPort,
                    THREAD_TIME_CONSTRAINT_POLICY,
                    reinterpret_cast<thread_policy_t>(&policy),
                    THREAD_TIME_CONSTRAINT_POLICY_COUNT);
                mach_port_deallocate(mach_task_self(), threadPort);
            }
        }
#endif
        const uint64_t target_nanos = acc.target_micros * 1000ULL;
        const double target_us = static_cast<double>(acc.target_micros);
#if defined(__APPLE__)
        const auto& spin_tb = PreciseMachTimebase();
        const uint64_t mach_delta_ticks = (acc.clock_type == 1 && spin_tb.numer > 0)
            ? (target_nanos * spin_tb.denom + spin_tb.numer - 1ULL) / spin_tb.numer
            : 0ULL;
#else
        const uint64_t mach_delta_ticks = 0ULL;
#endif

        auto do_spin = [&]() {
            if (acc.clock_type == 0) {
                const uint64_t deadline = PreciseMonotonicNanos() + target_nanos;
                while (PreciseMonotonicNanos() < deadline) {
                    PreciseSpinPause();
                }
            } else if (acc.clock_type == 1) {
#if defined(__APPLE__)
                const uint64_t deadlineTicks = mach_absolute_time() + mach_delta_ticks;
                while (mach_absolute_time() < deadlineTicks) {
                    PreciseSpinPause();
                }
#endif
            } else if (acc.clock_type == 2) {
#if defined(__APPLE__) && defined(CLOCK_MONOTONIC_RAW)
                const uint64_t deadline = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) + target_nanos;
                while (clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) < deadline) {
                    PreciseSpinPause();
                }
#endif
            }
        };

        for (int i = 0; i < warmup_count; ++i) {
            do_spin();
        }

        for (int i = 0; i < sample_count; ++i) {
            const uint64_t t0 = PreciseMonotonicNanos();
            const uint64_t c0 = ReadThreadCpuNanos();
            do_spin();
            const uint64_t t1 = PreciseMonotonicNanos();
            const uint64_t c1 = ReadThreadCpuNanos();

            acc.total_samples++;
            if (t1 < t0 + target_nanos) {
                acc.early_returns++;
            }
            const double wall_us = static_cast<double>(t1 - t0) / 1000.0;
            const double cpu_us = (c1 >= c0) ? static_cast<double>(c1 - c0) / 1000.0 : 0.0;
            const double overshoot_us = wall_us - target_us;
            const double unacc_us = (wall_us >= cpu_us) ? (wall_us - cpu_us) : 0.0;

            acc.v_wall.push_back(wall_us);
            acc.v_cpu.push_back(cpu_us);
            acc.v_overshoot.push_back(overshoot_us);
            acc.v_unaccounted.push_back(unacc_us);
        }
    });
    worker.join();
}

inline void FinalizeClockStats(ClockSpinAccumulator& acc) {
    acc.s_wall = ComputeStats(acc.v_wall);
    acc.s_cpu = ComputeStats(acc.v_cpu);
    acc.s_overshoot = ComputeStats(acc.v_overshoot);
    acc.s_unaccounted = ComputeStats(acc.v_unaccounted);
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

    report << "#### 3. Raw Measurements: 2x2 Matrix for PreciseSleepUs(1500) (10 Interleaved Rounds, 5000 Samples per Mode)\n\n";

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
    const int kSamplesPerRound = 500;

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

    report << "##### 3.1 Per-Round Wall p50 Breakdown (10 Rounds, 500 Samples/Round, Interleaved)\n\n";
    report << "| Backend | Policy | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | Overall p50 |\n";
    report << "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& m : modes) {
        report << "| `" << m.config.backend_name << "` | " << (m.config.enable_policy ? "**ON**" : "OFF") << " | ";
        for (double r_p50 : m.round_p50_wall) {
            report << std::fixed << std::setprecision(1) << r_p50 << " µs | ";
        }
        report << m.s_total_wall.p50 << " µs |\n";
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

#if defined(__APPLE__)
    report << "#### 6. Clock Source Comparison (Probe Latency & Spin Loop)\n\n";

    report << "##### 6.1a Direct Single Read Latency Distribution (10k individual unrolled invocations)\n\n";
    report << "| Clock Source | Min Latency | p50 Latency | p90 Latency | p99 Latency | Mean Latency | Max Latency |\n";
    report << "|---|---:|---:|---:|---:|---:|---:|\n";

    const auto s_gettime = MeasureSingleReadProbe(0);
    const auto s_mach = MeasureSingleReadProbe(1);
    const auto s_np = MeasureSingleReadProbe(2);

    report << "| `clock_gettime(CLOCK_MONOTONIC_RAW)` | "
           << std::fixed << std::setprecision(1)
           << s_gettime.min_ns << " ns | " << s_gettime.p50_ns << " ns | "
           << s_gettime.p90_ns << " ns | " << s_gettime.p99_ns << " ns | "
           << s_gettime.mean_ns << " ns | " << s_gettime.max_ns << " ns |\n";
    report << "| `mach_absolute_time()` | "
           << s_mach.min_ns << " ns | " << s_mach.p50_ns << " ns | "
           << s_mach.p90_ns << " ns | " << s_mach.p99_ns << " ns | "
           << s_mach.mean_ns << " ns | " << s_mach.max_ns << " ns |\n";
    report << "| `clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)` | "
           << s_np.min_ns << " ns | " << s_np.p50_ns << " ns | "
           << s_np.p90_ns << " ns | " << s_np.p99_ns << " ns | "
           << s_np.mean_ns << " ns | " << s_np.max_ns << " ns |\n\n";

    report << "##### 6.1b Amortized Per-Read Latency (Throughput: 100k samples, 1000 chunks x 100 reads)\n\n";
    report << "| Clock Source | Min Latency | p50 Latency | p90 Latency | p99 Latency | Mean Latency | Max Latency |\n";
    report << "|---|---:|---:|---:|---:|---:|---:|\n";

    const auto p_gettime = MeasureClockProbe(0);
    const auto p_mach = MeasureClockProbe(1);
    const auto p_np = MeasureClockProbe(2);

    report << "| `clock_gettime(CLOCK_MONOTONIC_RAW)` | "
           << std::fixed << std::setprecision(1)
           << p_gettime.min_ns << " ns | " << p_gettime.p50_ns << " ns | "
           << p_gettime.p90_ns << " ns | " << p_gettime.p99_ns << " ns | "
           << p_gettime.mean_ns << " ns | " << p_gettime.max_ns << " ns |\n";
    report << "| `mach_absolute_time()` | "
           << p_mach.min_ns << " ns | " << p_mach.p50_ns << " ns | "
           << p_mach.p90_ns << " ns | " << p_mach.p99_ns << " ns | "
           << p_mach.mean_ns << " ns | " << p_mach.max_ns << " ns |\n";
    report << "| `clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)` | "
           << p_np.min_ns << " ns | " << p_np.p50_ns << " ns | "
           << p_np.p90_ns << " ns | " << p_np.p99_ns << " ns | "
           << p_np.mean_ns << " ns | " << p_np.max_ns << " ns |\n\n";

    report << "##### 6.2 Spin Loop Comparison (Policy ON, Fresh Threads, Interleaved Rounds)\n\n";
    report << "| Target Duration | Clock Source | Wall p50 | Wall p90 | Wall p99 | Overshoot p50 | CPU Time p50 | Spin Unaccounted p50 | Early Returns |\n";
    report << "|---|---|---:|---:|---:|---:|---:|---:|---:|\n";

    std::vector<ClockSpinAccumulator> clock_modes;
    for (const uint64_t target_us : {50ULL, 1000ULL, 1500ULL}) {
        clock_modes.push_back(ClockSpinAccumulator{0, "clock_gettime", target_us});
        clock_modes.push_back(ClockSpinAccumulator{1, "mach_absolute_time", target_us});
        clock_modes.push_back(ClockSpinAccumulator{2, "clock_gettime_nsec_np", target_us});
    }

    const size_t num_clock_modes = clock_modes.size();
    const int kClockRounds = 5;
    const int kClockWarmup = 5;
    const int kClockSamples = 20;

    const std::vector<std::vector<int>> clock_permutations = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8},
        {8, 7, 6, 5, 4, 3, 2, 1, 0},
        {2, 5, 8, 1, 4, 7, 0, 3, 6},
        {6, 3, 0, 7, 4, 1, 8, 5, 2},
        {4, 0, 8, 2, 6, 1, 7, 3, 5}
    };

    for (int r = 0; r < kClockRounds; ++r) {
        const auto& perm = clock_permutations[r % clock_permutations.size()];
        for (int idx : perm) {
            if (static_cast<size_t>(idx) < num_clock_modes) {
                RunClockSpinSlice(clock_modes[idx], kClockWarmup, kClockSamples, tb);
            }
        }
    }

    for (auto& cm : clock_modes) {
        FinalizeClockStats(cm);
        report << "| " << cm.target_micros << " µs | `" << cm.clock_name << "` | "
               << std::fixed << std::setprecision(1)
               << cm.s_wall.p50 << " µs | "
               << cm.s_wall.p90 << " µs | "
               << cm.s_wall.p99 << " µs | +"
               << cm.s_overshoot.p50 << " µs | "
               << cm.s_cpu.p50 << " µs | "
               << cm.s_unaccounted.p50 << " µs | "
               << cm.early_returns << " |\n";
    }
    report << "\n";

    for (const auto& cm : clock_modes) {
        if (cm.early_returns > 0) {
            std::cerr << "FATAL: Clock source " << cm.clock_name << " returned BEFORE deadline for "
                      << cm.target_micros << " us!\n";
            return 1;
        }
    }

    report << "#### 7. Spin Loop Strategy Comparison: `clock_gettime` vs `mach_absolute_time` (Policy ON, 200 µs tail, PreciseSleepUs(1500), 5000 Samples per Mode)\n\n";
    report << "| Backend | Policy Status | Spin Strategy | Wall p50 | Wall p90 | Wall p99 | Wall Max | Overshoot p50 | Total CPU p50 | Spin CPU p50 | Spin Wall p50 | Outliers (>1650µs) | Early |\n";
    report << "|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n";

    std::vector<ModeAccumulator> spin_strategy_modes;
    spin_strategy_modes.push_back(ModeAccumulator{ModeConfig{PreciseWaitBackend::Nanosleep, true, "nanosleep", PreciseWaitSpinLoop::MonotonicNanos, 500}});
    spin_strategy_modes.push_back(ModeAccumulator{ModeConfig{PreciseWaitBackend::Nanosleep, true, "nanosleep", PreciseWaitSpinLoop::MachTicks, 500}});
    spin_strategy_modes.push_back(ModeAccumulator{ModeConfig{PreciseWaitBackend::MachWaitUntil, true, "mach_wait_until", PreciseWaitSpinLoop::MonotonicNanos, 500}});
    spin_strategy_modes.push_back(ModeAccumulator{ModeConfig{PreciseWaitBackend::MachWaitUntil, true, "mach_wait_until", PreciseWaitSpinLoop::MachTicks, 500}});

    const size_t num_spin_modes = spin_strategy_modes.size();
    const int kSpinStrategyRounds = 10;
    const int kSpinStrategyWarmup = 10;
    const int kSpinStrategySamples = 500;

    const std::vector<std::vector<int>> spin_permutations = {
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

    for (int r = 0; r < kSpinStrategyRounds; ++r) {
        const auto& perm = spin_permutations[r % spin_permutations.size()];
        for (int idx : perm) {
            if (static_cast<size_t>(idx) < num_spin_modes) {
                RunModeSlice(spin_strategy_modes[idx], 1500, kSpinStrategyWarmup, kSpinStrategySamples,
                    calibration.total_wall.p50,
                    calibration.total_cpu.p50,
                    calibration.spin_cpu.p50);
            }
        }
    }

    for (auto& sm : spin_strategy_modes) {
        FinalizeStats(sm);
        const std::string strat_name = (sm.config.spin_loop == PreciseWaitSpinLoop::MachTicks)
            ? "`mach_absolute_time` ticks"
            : "`clock_gettime` nanos";
        report << "| `" << sm.config.backend_name << "` | "
               << sm.policy_status << " | "
               << strat_name << " | "
               << std::fixed << std::setprecision(1)
               << sm.s_total_wall.p50 << " µs | "
               << sm.s_total_wall.p90 << " µs | "
               << sm.s_total_wall.p99 << " µs | "
               << sm.s_total_wall.max_val << " µs | +"
               << sm.s_total_overshoot.p50 << " µs | "
               << sm.s_total_cpu.p50 << " µs | "
               << sm.s_spin_cpu.p50 << " µs | "
               << sm.s_spin_wall.p50 << " µs | "
               << sm.outliers_count << " / " << sm.total_samples << " | "
               << sm.early_returns << " |\n";
    }
    report << "\n";

    report << "##### 7.1 Per-Round Wall p50 Breakdown (10 Rounds, 500 Samples/Round, Interleaved)\n\n";
    report << "| Backend | Strategy | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | Overall p50 |\n";
    report << "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& sm : spin_strategy_modes) {
        const std::string strat_name = (sm.config.spin_loop == PreciseWaitSpinLoop::MachTicks)
            ? "mach_ticks"
            : "monotonic_nanos";
        report << "| `" << sm.config.backend_name << "` | `" << strat_name << "` | ";
        for (double r_p50 : sm.round_p50_wall) {
            report << std::fixed << std::setprecision(1) << r_p50 << " µs | ";
        }
        report << sm.s_total_wall.p50 << " µs |\n";
    }
    report << "\n";

    for (const auto& sm : spin_strategy_modes) {
        if (sm.early_returns > 0) {
            std::cerr << "FATAL: Spin strategy mode " << sm.config.backend_name << " returned BEFORE deadline!\n";
            return 1;
        }
        if (sm.syscall_errors > 0) {
            std::cerr << "FATAL: Spin strategy mode " << sm.config.backend_name << " encountered syscall errors!\n";
            return 1;
        }
        if (sm.policy_failures > 0) {
            std::cerr << "FATAL: Spin strategy mode " << sm.config.backend_name << " encountered policy failures!\n";
            return 1;
        }
    }

    report << "#### 8. Policy Computation Budget Sweep: 250 µs vs 350 µs vs 500 µs (Fixed 200 µs Spin Tail, PreciseSleepUs(1500), 5000 Samples per Mode)\n\n";
    report << "| Backend | Budget | Policy Status | Wall p50 | Wall p90 | Wall p99 | Wall Max | Total CPU p50 | Spin CPU p50 | Spin CPU p90 | Spin CPU p99 | Syscall Wall p50 | Syscall Wall p90 | Syscall Wall p99 | Syscall Overshoot p50 | Syscall Overshoot p90 | Syscall Overshoot p99 | Outliers (>1650µs) | Early |\n";
    report << "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n";

    std::vector<ModeAccumulator> budget_modes;
    for (const uint32_t comp_us : {250U, 350U, 500U}) {
        budget_modes.push_back(ModeAccumulator{ModeConfig{PreciseWaitBackend::Nanosleep, true, "nanosleep", PreciseWaitSpinLoop::MonotonicNanos, comp_us}});
    }
    for (const uint32_t comp_us : {250U, 350U, 500U}) {
        budget_modes.push_back(ModeAccumulator{ModeConfig{PreciseWaitBackend::MachWaitUntil, true, "mach_wait_until", PreciseWaitSpinLoop::MonotonicNanos, comp_us}});
    }

    const size_t num_budget_modes = budget_modes.size();
    const int kBudgetRounds = 10;
    const int kBudgetWarmup = 10;
    const int kBudgetSamples = 500;

    const std::vector<std::vector<int>> budget_permutations = {
        {0, 1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1, 0},
        {1, 3, 5, 0, 2, 4},
        {4, 2, 0, 5, 3, 1},
        {2, 0, 4, 1, 5, 3},
        {3, 5, 1, 4, 0, 2},
        {0, 4, 2, 5, 1, 3},
        {5, 1, 3, 0, 4, 2},
        {2, 3, 0, 1, 5, 4},
        {4, 0, 5, 2, 3, 1}
    };

    for (int r = 0; r < kBudgetRounds; ++r) {
        const auto& perm = budget_permutations[r % budget_permutations.size()];
        for (int idx : perm) {
            if (static_cast<size_t>(idx) < num_budget_modes) {
                RunModeSlice(budget_modes[idx], 1500, kBudgetWarmup, kBudgetSamples,
                    calibration.total_wall.p50,
                    calibration.total_cpu.p50,
                    calibration.spin_cpu.p50);
            }
        }
    }

    for (auto& bm : budget_modes) {
        FinalizeStats(bm);
        report << "| `" << bm.config.backend_name << "` | "
               << bm.config.policy_computation_us << " µs | "
               << bm.policy_status << " | "
               << std::fixed << std::setprecision(1)
               << bm.s_total_wall.p50 << " µs | "
               << bm.s_total_wall.p90 << " µs | "
               << bm.s_total_wall.p99 << " µs | "
               << bm.s_total_wall.max_val << " µs | "
               << bm.s_total_cpu.p50 << " µs | "
               << bm.s_spin_cpu.p50 << " µs | "
               << bm.s_spin_cpu.p90 << " µs | "
               << bm.s_spin_cpu.p99 << " µs | "
               << bm.s_sleep_wall.p50 << " µs | "
               << bm.s_sleep_wall.p90 << " µs | "
               << bm.s_sleep_wall.p99 << " µs | +"
               << bm.s_sleep_overshoot.p50 << " µs | +"
               << bm.s_sleep_overshoot.p90 << " µs | +"
               << bm.s_sleep_overshoot.p99 << " µs | "
               << bm.outliers_count << " / " << bm.total_samples << " | "
               << bm.early_returns << " |\n";
    }
    report << "\n";

    report << "##### 8.1 Per-Round Wall p50 Breakdown (10 Rounds, 500 Samples/Round, Interleaved)\n\n";
    report << "| Backend | Budget | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | Overall p50 |\n";
    report << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& bm : budget_modes) {
        report << "| `" << bm.config.backend_name << "` | " << bm.config.policy_computation_us << " µs | ";
        for (double r_p50 : bm.round_p50_wall) {
            report << std::fixed << std::setprecision(1) << r_p50 << " µs | ";
        }
        report << bm.s_total_wall.p50 << " µs |\n";
    }
    report << "\n";

    for (const auto& bm : budget_modes) {
        if (bm.early_returns > 0) {
            std::cerr << "FATAL: Budget mode " << bm.config.backend_name << " returned BEFORE deadline!\n";
            return 1;
        }
        if (bm.syscall_errors > 0) {
            std::cerr << "FATAL: Budget mode " << bm.config.backend_name << " encountered syscall errors!\n";
            return 1;
        }
        if (bm.policy_failures > 0) {
            std::cerr << "FATAL: Budget mode " << bm.config.backend_name << " encountered policy failures!\n";
            return 1;
        }
    }
#endif

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
