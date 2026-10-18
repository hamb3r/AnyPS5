#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_PRECISEWAIT_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_PRECISEWAIT_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <type_traits>

inline constexpr std::uint64_t PreciseHundredNanosToQpcTicks(unsigned long long hundredNanos, std::uint64_t freq) {
    if (freq == 10000000ULL) [[likely]] {
        return hundredNanos;
    }
    const std::uint64_t wholeSeconds = hundredNanos / 10000000ULL;
    const std::uint64_t remHundredNanos = hundredNanos % 10000000ULL;
    return wholeSeconds * freq + (remHundredNanos * freq + 9999999ULL) / 10000000ULL;
}

inline constexpr std::uint64_t PrecisePauseThresholdTicks(std::uint64_t freq) {
    return PreciseHundredNanosToQpcTicks(2ULL, freq);
}

#ifdef _WIN32
#include <windows.h>
#include "prx/libc/include/HostThreadLocal.hpp"

inline void PreciseSleepWaitableTimerWithHandle(HANDLE handle, unsigned long long sleepHundredNanos) {
    bool slept = false;
    if (handle != nullptr && handle != INVALID_HANDLE_VALUE) [[likely]] {
        LARGE_INTEGER due;
        due.QuadPart = -static_cast<LONGLONG>(sleepHundredNanos);
        if (SetWaitableTimer(handle, &due, 0, nullptr, nullptr, FALSE) != FALSE) [[likely]] {
            slept = (WaitForSingleObject(handle, INFINITE) == WAIT_OBJECT_0);
        }
    }
    if (!slept) [[unlikely]] {
        std::this_thread::sleep_for(std::chrono::nanoseconds(sleepHundredNanos * 100ULL));
    }
}

inline HANDLE PreciseWaitThreadTimer() {
    struct Timer {
        HANDLE handle = CreateWaitableTimerExW(nullptr, nullptr,
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        ~Timer() {
            if (handle != nullptr) {
                CloseHandle(handle);
            }
        }
    };
    return HostThreadLocal<Timer, Timer>().handle;
}

inline std::uint64_t PreciseQpcFrequency() {
    static std::atomic<std::uint64_t> cachedFreq{0};
    std::uint64_t freq = cachedFreq.load(std::memory_order_relaxed);
    if (freq == 0) [[unlikely]] {
        LARGE_INTEGER f;
        if (QueryPerformanceFrequency(&f) && f.QuadPart > 0) {
            freq = static_cast<std::uint64_t>(f.QuadPart);
            cachedFreq.store(freq, std::memory_order_relaxed);
        }
    }
    return freq;
}

inline void PreciseSleepFromCounterWithTimer(
    LARGE_INTEGER counter,
    unsigned long long hundredNanos,
    HANDLE timerHandle,
    bool useCustomTimer,
    bool allowSpinTail = true) {
    if (hundredNanos == 0) {
        return;
    }
    constexpr unsigned long long kSpinTailHundredNanos = 2500;
    constexpr unsigned long long kMinTimerSpinSleepHundredNanos = 10000;
    const std::uint64_t freq = PreciseQpcFrequency();
    std::uint64_t deltaTicks = hundredNanos;
    std::uint64_t pauseThresholdTicks = 2ULL;
    if (freq != 10000000ULL) [[unlikely]] {
        deltaTicks = PreciseHundredNanosToQpcTicks(hundredNanos, freq);
        pauseThresholdTicks = PrecisePauseThresholdTicks(freq);
    }
    const std::uint64_t deadlineTicks = static_cast<std::uint64_t>(counter.QuadPart) + deltaTicks;
    const bool useSpinTail = allowSpinTail || hundredNanos >= kMinTimerSpinSleepHundredNanos;
    if (!useSpinTail || hundredNanos > kSpinTailHundredNanos) [[unlikely]] {
        const HANDLE handle = useCustomTimer ? timerHandle : PreciseWaitThreadTimer();
        const unsigned long long timerHundredNanos = useSpinTail ? (hundredNanos - kSpinTailHundredNanos) : hundredNanos;
        PreciseSleepWaitableTimerWithHandle(handle, timerHundredNanos);
    }
    for (;;) {
        QueryPerformanceCounter(&counter);
        const std::uint64_t nowTicks = static_cast<std::uint64_t>(counter.QuadPart);
        if (nowTicks >= deadlineTicks) {
            break;
        }
        if (deadlineTicks - nowTicks > pauseThresholdTicks) {
            YieldProcessor();
        }
    }
}

inline void PreciseSleepFromCounter(LARGE_INTEGER counter, unsigned long long hundredNanos) {
    PreciseSleepFromCounterWithTimer(counter, hundredNanos, nullptr, false, true);
}

inline void PreciseSleepHundredNanos(unsigned long long hundredNanos) {
    if (hundredNanos == 0) {
        return;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    PreciseSleepFromCounterWithTimer(counter, hundredNanos, nullptr, false, false);
}
#elif !defined(__EMSCRIPTEN__)
#include <cerrno>
#include <time.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

inline void PreciseSpinPause() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm64__)
    asm volatile("yield" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

inline std::uint64_t PreciseMonotonicNanos() {
    struct timespec ts;
#if defined(__APPLE__) && defined(_LIBCPP_VERSION) && defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<std::uint64_t>(ts.tv_nsec);
}

#ifdef __linux__
inline bool PreciseLinuxLowTimerSlack() {
    static thread_local const bool lowSlack = [] {
        const long current = prctl(PR_GET_TIMERSLACK, 0UL, 0UL, 0UL, 0UL);
        return current == 1L || prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL) == 0;
    }();
    return lowSlack;
}
#elif defined(__APPLE__)
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <mach/mach_time.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>

inline bool PreciseAppleLowTimerSlack() {
    static thread_local const bool lowSlack = [] {
        mach_timebase_info_data_t tb;
        if (mach_timebase_info(&tb) != KERN_SUCCESS || tb.numer == 0) {
            return false;
        }
        thread_time_constraint_policy_data_t policy;
        const uint64_t periodTicks = (1000000ULL * tb.denom) / tb.numer;
        const uint64_t computationTicks = (500000ULL * tb.denom) / tb.numer;
        if (periodTicks > UINT32_MAX || computationTicks > UINT32_MAX) {
            return false;
        }
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
        return kr == KERN_SUCCESS;
    }();
    return lowSlack;
}
#endif

#ifdef APS5_PRECISE_WAIT_TRACE
struct PreciseWaitTraceData {
    std::uint64_t t1_nanos;
    std::uint64_t t2_nanos;
    std::uint64_t t3_nanos;
    std::uint64_t cpu1_nanos;
    std::uint64_t cpu2_nanos;
    std::uint64_t cpu3_nanos;
    std::uint64_t requested_sleep_nanos;
};
inline thread_local PreciseWaitTraceData g_preciseWaitTraceData{};
inline thread_local bool g_preciseWaitTraceActive = false;

inline std::uint64_t PreciseThreadCpuNanosInternal() {
#if defined(CLOCK_THREAD_CPUTIME_ID)
    struct timespec ts{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
        return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
    }
#endif
    return 0;
}
#endif

inline void PreciseSleepFromNanosWithSlack(std::uint64_t startNanos, unsigned long long nanos, bool allowLowSlack) {
    if (nanos == 0) {
        return;
    }
    const std::uint64_t deadlineNanos = startNanos + nanos;
#ifdef __linux__
    const bool lowSlack = allowLowSlack && PreciseLinuxLowTimerSlack();
    const unsigned long long spinTailNanos = lowSlack ? 15000ULL : 100000ULL;
#ifdef APS5_PRECISE_WAIT_TRACE
    if (g_preciseWaitTraceActive) {
        g_preciseWaitTraceData.requested_sleep_nanos = (nanos > spinTailNanos) ? (nanos - spinTailNanos) : 0ULL;
        g_preciseWaitTraceData.t1_nanos = PreciseMonotonicNanos();
        g_preciseWaitTraceData.cpu1_nanos = PreciseThreadCpuNanosInternal();
    }
#endif
    if (nanos > spinTailNanos) [[unlikely]] {
        const std::uint64_t wakeNanos = deadlineNanos - spinTailNanos;
        struct timespec req;
        req.tv_sec = static_cast<time_t>(wakeNanos / 1000000000ULL);
        req.tv_nsec = static_cast<long>(wakeNanos % 1000000000ULL);
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &req, nullptr) == EINTR) {
        }
    }
#ifdef APS5_PRECISE_WAIT_TRACE
    if (g_preciseWaitTraceActive) {
        g_preciseWaitTraceData.t2_nanos = PreciseMonotonicNanos();
        g_preciseWaitTraceData.cpu2_nanos = PreciseThreadCpuNanosInternal();
    }
#endif
#elif defined(__APPLE__)
    const bool lowSlack = allowLowSlack && PreciseAppleLowTimerSlack();
    const unsigned long long spinTailNanos = lowSlack ? 200000ULL : 500000ULL;
#ifdef APS5_PRECISE_WAIT_TRACE
    if (g_preciseWaitTraceActive) {
        g_preciseWaitTraceData.requested_sleep_nanos = (nanos > spinTailNanos) ? (nanos - spinTailNanos) : 0ULL;
        g_preciseWaitTraceData.t1_nanos = PreciseMonotonicNanos();
        g_preciseWaitTraceData.cpu1_nanos = PreciseThreadCpuNanosInternal();
    }
#endif
    if (nanos > spinTailNanos) [[unlikely]] {
        const unsigned long long sleepNanos = nanos - spinTailNanos;
        struct timespec req;
        req.tv_sec = static_cast<time_t>(sleepNanos / 1000000000ULL);
        req.tv_nsec = static_cast<long>(sleepNanos % 1000000000ULL);
        while (nanosleep(&req, &req) == -1 && errno == EINTR) {
        }
    }
#ifdef APS5_PRECISE_WAIT_TRACE
    if (g_preciseWaitTraceActive) {
        g_preciseWaitTraceData.t2_nanos = PreciseMonotonicNanos();
        g_preciseWaitTraceData.cpu2_nanos = PreciseThreadCpuNanosInternal();
    }
#endif
#else
    (void)allowLowSlack;
    constexpr unsigned long long kSpinTailNanos = 100000ULL;
    if (nanos > kSpinTailNanos) [[unlikely]] {
        const unsigned long long sleepNanos = nanos - kSpinTailNanos;
        struct timespec req;
        req.tv_sec = static_cast<time_t>(sleepNanos / 1000000000ULL);
        req.tv_nsec = static_cast<long>(sleepNanos % 1000000000ULL);
        while (nanosleep(&req, &req) == -1 && errno == EINTR) {
        }
    }
#endif
    for (;;) {
        const std::uint64_t nowNanos = PreciseMonotonicNanos();
        if (nowNanos >= deadlineNanos) {
            break;
        }
        if (deadlineNanos - nowNanos > 200ULL) {
            PreciseSpinPause();
        }
    }
#ifdef APS5_PRECISE_WAIT_TRACE
    if (g_preciseWaitTraceActive) {
        g_preciseWaitTraceData.t3_nanos = PreciseMonotonicNanos();
        g_preciseWaitTraceData.cpu3_nanos = PreciseThreadCpuNanosInternal();
    }
#endif
}

inline void PreciseSleepFromNanos(std::uint64_t startNanos, unsigned long long nanos) {
    PreciseSleepFromNanosWithSlack(startNanos, nanos, true);
}

inline void PreciseSleepNanos(unsigned long long nanos) {
    if (nanos == 0) {
        return;
    }
    PreciseSleepFromNanos(PreciseMonotonicNanos(), nanos);
}
#endif

inline constexpr unsigned long long kSubTickWaitUs = 2000;

inline void PreciseSleepUs(unsigned long long micros) {
    if (micros == 0) {
        return;
    }
#ifdef _WIN32
    PreciseSleepHundredNanos(micros * 10ULL);
    return;
#elif !defined(__EMSCRIPTEN__)
    PreciseSleepNanos(micros * 1000ULL);
    return;
#else
    std::this_thread::sleep_for(std::chrono::microseconds(micros));
#endif
}

template <class TClock, class TDuration>
inline void PreciseSleepUntil(const std::chrono::time_point<TClock, TDuration>& deadline) {
#ifdef _WIN32
    if constexpr (std::is_same_v<TClock, std::chrono::steady_clock>) {
        if (PreciseQpcFrequency() == 10000000ULL) [[likely]] {
            const auto rawDeadline = deadline.time_since_epoch().count();
            if (rawDeadline <= 0) [[unlikely]] {
                return;
            }
            using HundredNanos = std::chrono::duration<std::int64_t, std::ratio<1, 10000000>>;
            std::uint64_t targetHns = 0;
            if constexpr (std::is_same_v<typename TDuration::period, std::nano>) {
                targetHns = (static_cast<std::uint64_t>(rawDeadline) + 99ULL) / 100ULL;
            } else {
                targetHns = static_cast<std::uint64_t>(
                    std::chrono::ceil<HundredNanos>(deadline.time_since_epoch()).count());
            }
            LARGE_INTEGER counter;
            QueryPerformanceCounter(&counter);
            const std::uint64_t nowTicks = static_cast<std::uint64_t>(counter.QuadPart);
            if (nowTicks < targetHns) {
                PreciseSleepFromCounter(counter, targetHns - nowTicks);
            }
            return;
        }
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    const auto now = TClock::now();
    if (now < deadline) {
        const auto remainingNs = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count();
        if (remainingNs > 0) {
            PreciseSleepFromCounter(counter, (static_cast<unsigned long long>(remainingNs) + 99ULL) / 100ULL);
        }
    }
    for (auto current = TClock::now(); current < deadline; current = TClock::now()) {
        if (deadline - current > std::chrono::nanoseconds(200)) {
            YieldProcessor();
        }
    }
#elif !defined(__EMSCRIPTEN__)
#if (defined(__linux__) && (defined(__GLIBCXX__) || (defined(_LIBCPP_VERSION) && !defined(_LIBCPP_HAS_TIMESPEC_GET)))) || \
    (defined(__APPLE__) && defined(_LIBCPP_VERSION) && defined(CLOCK_MONOTONIC_RAW))
    if constexpr (std::is_same_v<TClock, std::chrono::steady_clock>) {
        const auto rawDeadline = deadline.time_since_epoch().count();
        if (rawDeadline <= 0) [[unlikely]] {
            return;
        }
        std::uint64_t targetNanos = 0;
        if constexpr (std::is_same_v<typename TDuration::period, std::nano>) {
            targetNanos = static_cast<std::uint64_t>(rawDeadline);
        } else {
            targetNanos = static_cast<std::uint64_t>(
                std::chrono::ceil<std::chrono::nanoseconds>(deadline.time_since_epoch()).count());
        }
        const std::uint64_t startNanos = PreciseMonotonicNanos();
        if (startNanos < targetNanos) {
            PreciseSleepFromNanos(startNanos, targetNanos - startNanos);
        }
        return;
    }
#endif
    const std::uint64_t startNanos = PreciseMonotonicNanos();
    const auto now = TClock::now();
    if (now < deadline) {
        const auto remainingNs = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count();
        if (remainingNs > 0) {
            PreciseSleepFromNanos(startNanos, static_cast<unsigned long long>(remainingNs));
        }
    }
    for (auto current = TClock::now(); current < deadline; current = TClock::now()) {
        if (deadline - current > std::chrono::nanoseconds(200)) {
            PreciseSpinPause();
        }
    }
#else
    for (auto now = TClock::now(); now < deadline; now = TClock::now()) {
        PreciseSleepUs(static_cast<unsigned long long>(std::chrono::ceil<std::chrono::microseconds>(deadline - now).count()));
    }
#endif
}

#endif
