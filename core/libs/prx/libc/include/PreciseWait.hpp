#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_PRECISEWAIT_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_PRECISEWAIT_HPP

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ratio>
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

struct PreciseMachDeadlineResult {
    std::uint32_t status{0};
    std::uint64_t sleepTicks{0};
    std::uint64_t wakeDeadline{0};
};

inline constexpr std::uint64_t PreciseGcd(std::uint64_t a, std::uint64_t b) {
    while (b != 0) {
        const std::uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

template <class Period>
struct PreciseSafeNanoRatio {
    static constexpr std::uint64_t p_num = static_cast<std::uint64_t>(Period::num);
    static constexpr std::uint64_t p_den = static_cast<std::uint64_t>(Period::den);
    static constexpr std::uint64_t g1 = PreciseGcd(p_den, 1000000000ULL);
    static constexpr std::uint64_t k1 = 1000000000ULL / g1;
    static constexpr std::uint64_t d1 = p_den / g1;
    static constexpr std::uint64_t g2 = PreciseGcd(p_num, d1);
    static constexpr std::uint64_t n2 = p_num / g2;
    static constexpr std::uint64_t den = d1 / g2;
    static constexpr bool overflows_tick = (n2 > UINT64_MAX / k1);
    static constexpr std::uint64_t num = overflows_tick ? UINT64_MAX : (n2 * k1);
};

inline constexpr std::uint64_t PreciseMulDivCeilFallback64(std::uint64_t r, std::uint64_t num, std::uint64_t den) {
    if (r == 0 || num == 0 || den == 0) {
        return 0ULL;
    }
    const std::uint64_t wholeQ = r / den;
    const std::uint64_t remR = r % den;
    if (wholeQ > 0 && num > UINT64_MAX / wholeQ) {
        return UINT64_MAX;
    }
    const std::uint64_t base = wholeQ * num;
    if (remR == 0) {
        return base;
    }
    if (remR <= UINT64_MAX / num) {
        const std::uint64_t prod = remR * num;
        const std::uint64_t q = prod / den;
        const std::uint64_t rem = prod % den;
        const std::uint64_t extra = q + (rem != 0 ? 1ULL : 0ULL);
        if (UINT64_MAX - base < extra) {
            return UINT64_MAX;
        }
        return base + extra;
    }
    std::uint64_t q = 0;
    std::uint64_t rem = 0;
    const std::uint64_t addQ = num / den;
    const std::uint64_t addR = num % den;
    for (int i = 63; i >= 0; --i) {
        if ((remR >> i) & 1ULL) {
            if (UINT64_MAX - q < addQ) {
                return UINT64_MAX;
            }
            q += addQ;
            if (den - rem <= addR) {
                if (q == UINT64_MAX) {
                    return UINT64_MAX;
                }
                q += 1ULL;
                rem = addR - (den - rem);
            } else {
                rem += addR;
            }
        }
        if (i > 0) {
            if (q > UINT64_MAX / 2ULL) {
                return UINT64_MAX;
            }
            q <<= 1;
            if (rem >= den - rem) {
                if (q == UINT64_MAX) {
                    return UINT64_MAX;
                }
                q += 1ULL;
                rem = rem - (den - rem);
            } else {
                rem <<= 1;
            }
        }
    }
    const std::uint64_t extra = q + (rem != 0 ? 1ULL : 0ULL);
    if (UINT64_MAX - base < extra) {
        return UINT64_MAX;
    }
    return base + extra;
}

inline constexpr std::uint64_t PreciseMulDivCeil64(std::uint64_t r, std::uint64_t num, std::uint64_t den) {
#if defined(__SIZEOF_INT128__) && !defined(_WIN32)
    if (r == 0 || num == 0 || den == 0) {
        return 0ULL;
    }
    const unsigned __int128 prod = static_cast<unsigned __int128>(r) * num;
    const unsigned __int128 q = prod / den;
    const unsigned __int128 rem = prod % den;
    const unsigned __int128 res = q + (rem != 0 ? 1 : 0);
    return res > UINT64_MAX ? UINT64_MAX : static_cast<std::uint64_t>(res);
#else
    return PreciseMulDivCeilFallback64(r, num, den);
#endif
}

inline constexpr PreciseMachDeadlineResult PreciseComputeMachDeadline64(
    std::uint64_t sleepNanos,
    std::uint32_t numer,
    std::uint32_t denom,
    std::uint64_t nowTicks) {
    PreciseMachDeadlineResult result{};
    if (numer == 0 || denom == 0) [[unlikely]] {
        result.status = 0xFFFFFFFF;
        return result;
    }
    const std::uint64_t q = sleepNanos / numer;
    const std::uint64_t r = sleepNanos % numer;
    if (denom > 0 && q > UINT64_MAX / denom) [[unlikely]] {
        result.status = 0xFFFFFFFE;
        return result;
    }
    const std::uint64_t baseTicks = q * denom;
    const std::uint64_t remTicks = (r * static_cast<std::uint64_t>(denom)) / numer;
    if (remTicks > UINT64_MAX - baseTicks) [[unlikely]] {
        result.status = 0xFFFFFFFE;
        return result;
    }
    result.sleepTicks = baseTicks + remTicks;
    if (result.sleepTicks > UINT64_MAX - nowTicks) [[unlikely]] {
        result.status = 0xFFFFFFFD;
        return result;
    }
    result.wakeDeadline = nowTicks + result.sleepTicks;
    return result;
}

inline constexpr PreciseMachDeadlineResult PreciseComputeMachDeadline(
    std::uint64_t sleepNanos,
    std::uint32_t numer,
    std::uint32_t denom,
    std::uint64_t nowTicks) {
#if defined(__SIZEOF_INT128__) && !defined(_WIN32)
    PreciseMachDeadlineResult result{};
    if (numer == 0 || denom == 0) [[unlikely]] {
        result.status = 0xFFFFFFFF;
        return result;
    }
    const unsigned __int128 sleepTicks128 =
        (static_cast<unsigned __int128>(sleepNanos) * denom) / numer;
    if (sleepTicks128 > UINT64_MAX) [[unlikely]] {
        result.status = 0xFFFFFFFE;
        return result;
    }
    result.sleepTicks = static_cast<std::uint64_t>(sleepTicks128);
    if (result.sleepTicks > UINT64_MAX - nowTicks) [[unlikely]] {
        result.status = 0xFFFFFFFD;
        return result;
    }
    result.wakeDeadline = nowTicks + result.sleepTicks;
    return result;
#else
    return PreciseComputeMachDeadline64(sleepNanos, numer, denom, nowTicks);
#endif
}

template <class Rep, class Period>
constexpr std::uint64_t PreciseSafeDurationToNanos(const std::chrono::duration<Rep, Period>& d) {
    if constexpr (std::is_floating_point_v<Rep>) {
        const Rep c = d.count();
        if (c != c || !(c > static_cast<Rep>(0))) {
            return 0ULL;
        }
        if (c > std::numeric_limits<Rep>::max()) {
            return UINT64_MAX;
        }
        constexpr double kScale = (static_cast<double>(Period::num) / static_cast<double>(Period::den)) * 1000000000.0;
        const double scaled = static_cast<double>(c) * kScale;
        if (scaled > std::numeric_limits<double>::max() || scaled >= static_cast<double>(UINT64_MAX)) {
            return UINT64_MAX;
        }
        if (scaled <= 1.0) {
            return 1ULL;
        }
        const std::uint64_t floorVal = static_cast<std::uint64_t>(scaled);
        if (floorVal == UINT64_MAX) {
            return UINT64_MAX;
        }
        const double epsTol = (scaled * std::numeric_limits<double>::epsilon() * 8.0 > 1e-9)
                                  ? (scaled * std::numeric_limits<double>::epsilon() * 8.0)
                                  : 1e-9;
        if (scaled - static_cast<double>(floorVal) <= epsTol) {
            return floorVal;
        }
        if (static_cast<double>(floorVal + 1ULL) - scaled <= epsTol) {
            return floorVal + 1ULL;
        }
        return floorVal + 1ULL;
    } else {
        if (d.count() <= static_cast<Rep>(0)) {
            return 0ULL;
        }
        if constexpr (PreciseSafeNanoRatio<Period>::overflows_tick) {
            return UINT64_MAX;
        }
        if constexpr (sizeof(Rep) > sizeof(std::uint64_t)) {
            if (d.count() > static_cast<Rep>(UINT64_MAX)) {
                return UINT64_MAX;
            }
        }
        const std::uint64_t count = static_cast<std::uint64_t>(d.count());
        constexpr std::uint64_t num = PreciseSafeNanoRatio<Period>::num;
        constexpr std::uint64_t den = PreciseSafeNanoRatio<Period>::den;
        if constexpr (num == den) {
            return count;
        } else if constexpr (den == 1) {
            constexpr std::uint64_t kMaxCount = UINT64_MAX / num;
            if (count > kMaxCount) {
                return UINT64_MAX;
            }
            return count * num;
        } else {
            const std::uint64_t q = count / den;
            const std::uint64_t r = count % den;
            constexpr std::uint64_t kMaxQ = UINT64_MAX / num;
            if (q > kMaxQ) {
                return UINT64_MAX;
            }
            const std::uint64_t base = q * num;
            const std::uint64_t remTicks = PreciseMulDivCeil64(r, num, den);
            if (UINT64_MAX - base < remTicks) {
                return UINT64_MAX;
            }
            return base + remTicks;
        }
    }
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

#ifdef APS5_PRECISE_WAIT_TRACE
enum class PreciseWaitBackend {
    Nanosleep,
    MachWaitUntil
};

enum class PreciseWaitSpinLoop {
    MonotonicNanos,
    MachTicks
};

struct PreciseWaitTraceSample {
    std::uint64_t t0;
    std::uint64_t t1;
    std::uint64_t t2;
    std::uint64_t t3;
    std::uint64_t t4;
    std::uint64_t cpu0;
    std::uint64_t cpu1;
    std::uint64_t cpu2;
    std::uint64_t cpu3;
    std::uint64_t cpu4;
    std::uint64_t req_sleep_nanos;
    std::uint64_t spin_tail_nanos;
    std::uint32_t eintr_count;
    std::uint32_t syscall_return_code;
};

inline thread_local PreciseWaitTraceSample g_preciseWaitTraceSample{};
inline thread_local bool g_preciseWaitTraceActive = false;
inline thread_local PreciseWaitBackend g_preciseWaitTraceBackend = PreciseWaitBackend::Nanosleep;
inline thread_local PreciseWaitSpinLoop g_preciseWaitTraceSpinLoop = PreciseWaitSpinLoop::MonotonicNanos;
inline thread_local bool g_preciseWaitTraceAllowLowSlack = true;
inline thread_local std::uint32_t g_preciseWaitTracePolicyComputationUs = 500;

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

inline mach_timebase_info_data_t PreciseMachTimebase() {
    static std::atomic<std::uint64_t> cached{0};
    const std::uint64_t val = cached.load(std::memory_order_relaxed);
    if (val != 0) [[likely]] {
        mach_timebase_info_data_t tb;
        tb.numer = static_cast<std::uint32_t>(val);
        tb.denom = static_cast<std::uint32_t>(val >> 32);
        return tb;
    }
    mach_timebase_info_data_t t{0, 0};
    if (mach_timebase_info(&t) == KERN_SUCCESS && t.numer > 0 && t.denom > 0) [[likely]] {
        const std::uint64_t packed =
            static_cast<std::uint64_t>(t.numer) | (static_cast<std::uint64_t>(t.denom) << 32);
        cached.store(packed, std::memory_order_relaxed);
        return t;
    }
    return {0, 0};
}

inline bool PreciseAppleLowTimerSlack() {
    static thread_local std::int8_t lowSlackState = 0;
    if (lowSlackState != 0) [[likely]] {
        return lowSlackState > 0;
    }
    const auto tb = PreciseMachTimebase();
    if (tb.numer == 0 || tb.denom == 0) {
        return false;
    }
    thread_time_constraint_policy_data_t policy;
    const uint64_t periodTicks = (1000000ULL * tb.denom) / tb.numer;
#ifdef APS5_PRECISE_WAIT_TRACE
    const uint64_t compUs = g_preciseWaitTracePolicyComputationUs > 0 ? g_preciseWaitTracePolicyComputationUs : 500;
    const uint64_t computationTicks = (compUs * 1000ULL * tb.denom) / tb.numer;
#else
    const uint64_t computationTicks = (500000ULL * tb.denom) / tb.numer;
#endif
    if (periodTicks > UINT32_MAX || computationTicks > UINT32_MAX) {
        lowSlackState = -1;
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
    if (kr == KERN_SUCCESS) {
        lowSlackState = 1;
        return true;
    }
    if (kr == KERN_NOT_SUPPORTED || kr == KERN_PROTECTION_FAILURE || kr == KERN_INVALID_ARGUMENT) {
        lowSlackState = -1;
    }
    return false;
}
#endif

inline void PreciseSleepFromNanosWithSlack(std::uint64_t startNanos, unsigned long long nanos, bool allowLowSlack) {
    if (nanos == 0) {
        return;
    }
#ifdef APS5_PRECISE_WAIT_TRACE
    if (g_preciseWaitTraceActive) {
        g_preciseWaitTraceSample.t0 = startNanos;
        g_preciseWaitTraceSample.cpu0 = PreciseThreadCpuNanosInternal();
        g_preciseWaitTraceSample.eintr_count = 0;
        g_preciseWaitTraceSample.syscall_return_code = 0;
        allowLowSlack = g_preciseWaitTraceAllowLowSlack;
    }
#endif
    const std::uint64_t deadlineNanos = (nanos > UINT64_MAX - startNanos) ? UINT64_MAX : (startNanos + nanos);
#ifdef __linux__
    const bool lowSlack = allowLowSlack && PreciseLinuxLowTimerSlack();
    const unsigned long long spinTailNanos = lowSlack ? 15000ULL : 100000ULL;
#ifdef APS5_PRECISE_WAIT_TRACE
    if (g_preciseWaitTraceActive) {
        g_preciseWaitTraceSample.spin_tail_nanos = spinTailNanos;
    }
#endif
    if (nanos > spinTailNanos) [[unlikely]] {
        const std::uint64_t wakeNanos = deadlineNanos - spinTailNanos;
        struct timespec req;
        req.tv_sec = static_cast<time_t>(wakeNanos / 1000000000ULL);
        req.tv_nsec = static_cast<long>(wakeNanos % 1000000000ULL);
#ifdef APS5_PRECISE_WAIT_TRACE
        if (g_preciseWaitTraceActive) {
            g_preciseWaitTraceSample.req_sleep_nanos = (nanos - spinTailNanos);
            g_preciseWaitTraceSample.t1 = PreciseMonotonicNanos();
            g_preciseWaitTraceSample.cpu1 = PreciseThreadCpuNanosInternal();
        }
#endif
        for (;;) {
            const int err = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &req, nullptr);
            if (err == 0) [[likely]] {
                break;
            }
            if (err == EINTR) {
#ifdef APS5_PRECISE_WAIT_TRACE
                if (g_preciseWaitTraceActive) {
                    g_preciseWaitTraceSample.eintr_count++;
                }
#endif
                continue;
            }
#ifdef APS5_PRECISE_WAIT_TRACE
            if (g_preciseWaitTraceActive) {
                g_preciseWaitTraceSample.syscall_return_code = static_cast<std::uint32_t>(err);
            }
#endif
            break;
        }
#ifdef APS5_PRECISE_WAIT_TRACE
        if (g_preciseWaitTraceActive) {
            g_preciseWaitTraceSample.t2 = PreciseMonotonicNanos();
            g_preciseWaitTraceSample.cpu2 = PreciseThreadCpuNanosInternal();
        }
#endif
    } else {
#ifdef APS5_PRECISE_WAIT_TRACE
        if (g_preciseWaitTraceActive) {
            g_preciseWaitTraceSample.req_sleep_nanos = 0;
            g_preciseWaitTraceSample.t1 = PreciseMonotonicNanos();
            g_preciseWaitTraceSample.cpu1 = PreciseThreadCpuNanosInternal();
            g_preciseWaitTraceSample.t2 = g_preciseWaitTraceSample.t1;
            g_preciseWaitTraceSample.cpu2 = g_preciseWaitTraceSample.cpu1;
        }
#endif
    }
#elif defined(__APPLE__)
    const bool lowSlack = allowLowSlack && PreciseAppleLowTimerSlack();
    const unsigned long long spinTailNanos = lowSlack ? 200000ULL : 500000ULL;
#ifdef APS5_PRECISE_WAIT_TRACE
    if (g_preciseWaitTraceActive) {
        g_preciseWaitTraceSample.spin_tail_nanos = spinTailNanos;
    }
    std::uint64_t machSpinDeadlineTicks = 0;
    bool useMachSpinTicks = false;
    if (g_preciseWaitTraceActive && g_preciseWaitTraceSpinLoop == PreciseWaitSpinLoop::MachTicks) {
        const auto& tb = PreciseMachTimebase();
        if (tb.numer > 0 && tb.denom > 0) {
            const std::uint64_t nowNanos = PreciseMonotonicNanos();
            if (nowNanos < deadlineNanos) {
                const std::uint64_t remainingNanos = deadlineNanos - nowNanos;
                const std::uint64_t nowTicks = mach_absolute_time();
                const auto machDeadline =
                    PreciseComputeMachDeadline(remainingNanos, tb.numer, tb.denom, nowTicks);
                if (machDeadline.status == 0) {
                    machSpinDeadlineTicks = machDeadline.wakeDeadline;
                    useMachSpinTicks = true;
                }
            }
        }
    }
#endif
    if (nanos > spinTailNanos) [[unlikely]] {
        const unsigned long long sleepNanos = nanos - spinTailNanos;
#ifdef APS5_PRECISE_WAIT_TRACE
        if (g_preciseWaitTraceActive && g_preciseWaitTraceBackend == PreciseWaitBackend::MachWaitUntil) {
            const auto& tb = PreciseMachTimebase();
            if (tb.numer == 0 || tb.denom == 0) [[unlikely]] {
                g_preciseWaitTraceSample.t1 = PreciseMonotonicNanos();
                g_preciseWaitTraceSample.cpu1 = PreciseThreadCpuNanosInternal();
                g_preciseWaitTraceSample.t2 = g_preciseWaitTraceSample.t1;
                g_preciseWaitTraceSample.cpu2 = g_preciseWaitTraceSample.cpu1;
                g_preciseWaitTraceSample.t3 = g_preciseWaitTraceSample.t1;
                g_preciseWaitTraceSample.cpu3 = g_preciseWaitTraceSample.cpu1;
                g_preciseWaitTraceSample.syscall_return_code = 0xFFFFFFFF;
                return;
            }
            const std::uint64_t nowTicks = mach_absolute_time();
            const auto machDeadline =
                PreciseComputeMachDeadline(sleepNanos, tb.numer, tb.denom, nowTicks);
            if (machDeadline.status != 0) [[unlikely]] {
                g_preciseWaitTraceSample.t1 = PreciseMonotonicNanos();
                g_preciseWaitTraceSample.cpu1 = PreciseThreadCpuNanosInternal();
                g_preciseWaitTraceSample.t2 = g_preciseWaitTraceSample.t1;
                g_preciseWaitTraceSample.cpu2 = g_preciseWaitTraceSample.cpu1;
                g_preciseWaitTraceSample.t3 = g_preciseWaitTraceSample.t1;
                g_preciseWaitTraceSample.cpu3 = g_preciseWaitTraceSample.cpu1;
                g_preciseWaitTraceSample.syscall_return_code = machDeadline.status;
                return;
            }
            g_preciseWaitTraceSample.req_sleep_nanos = sleepNanos;
            g_preciseWaitTraceSample.t1 = PreciseMonotonicNanos();
            g_preciseWaitTraceSample.cpu1 = PreciseThreadCpuNanosInternal();
            const kern_return_t kr = mach_wait_until(machDeadline.wakeDeadline);
            g_preciseWaitTraceSample.t2 = PreciseMonotonicNanos();
            g_preciseWaitTraceSample.cpu2 = PreciseThreadCpuNanosInternal();
            g_preciseWaitTraceSample.syscall_return_code = static_cast<std::uint32_t>(kr);
        } else {
#endif
            struct timespec req;
            req.tv_sec = static_cast<time_t>(sleepNanos / 1000000000ULL);
            req.tv_nsec = static_cast<long>(sleepNanos % 1000000000ULL);
#ifdef APS5_PRECISE_WAIT_TRACE
            if (g_preciseWaitTraceActive) {
                g_preciseWaitTraceSample.req_sleep_nanos = sleepNanos;
                g_preciseWaitTraceSample.t1 = PreciseMonotonicNanos();
                g_preciseWaitTraceSample.cpu1 = PreciseThreadCpuNanosInternal();
            }
#endif
            for (;;) {
                const int ret = nanosleep(&req, &req);
                if (ret == 0) [[likely]] {
                    break;
                }
                if (errno == EINTR) {
#ifdef APS5_PRECISE_WAIT_TRACE
                    if (g_preciseWaitTraceActive) {
                        g_preciseWaitTraceSample.eintr_count++;
                    }
#endif
                    continue;
                }
#ifdef APS5_PRECISE_WAIT_TRACE
                if (g_preciseWaitTraceActive) {
                    g_preciseWaitTraceSample.syscall_return_code = static_cast<std::uint32_t>(errno != 0 ? errno : -1);
                }
#endif
                break;
            }
#ifdef APS5_PRECISE_WAIT_TRACE
            if (g_preciseWaitTraceActive) {
                g_preciseWaitTraceSample.t2 = PreciseMonotonicNanos();
                g_preciseWaitTraceSample.cpu2 = PreciseThreadCpuNanosInternal();
            }
#endif
#ifdef APS5_PRECISE_WAIT_TRACE
        }
#endif
    } else {
#ifdef APS5_PRECISE_WAIT_TRACE
        if (g_preciseWaitTraceActive) {
            g_preciseWaitTraceSample.req_sleep_nanos = 0;
            g_preciseWaitTraceSample.t1 = PreciseMonotonicNanos();
            g_preciseWaitTraceSample.cpu1 = PreciseThreadCpuNanosInternal();
            g_preciseWaitTraceSample.t2 = g_preciseWaitTraceSample.t1;
            g_preciseWaitTraceSample.cpu2 = g_preciseWaitTraceSample.cpu1;
        }
#endif
    }
#else
    (void)allowLowSlack;
    constexpr unsigned long long kSpinTailNanos = 100000ULL;
#ifdef APS5_PRECISE_WAIT_TRACE
    if (g_preciseWaitTraceActive) {
        g_preciseWaitTraceSample.spin_tail_nanos = kSpinTailNanos;
    }
#endif
    if (nanos > kSpinTailNanos) [[unlikely]] {
        const unsigned long long sleepNanos = nanos - kSpinTailNanos;
        struct timespec req;
        req.tv_sec = static_cast<time_t>(sleepNanos / 1000000000ULL);
        req.tv_nsec = static_cast<long>(sleepNanos % 1000000000ULL);
        for (;;) {
            const int ret = nanosleep(&req, &req);
            if (ret == 0) [[likely]] {
                break;
            }
            if (errno == EINTR) {
                continue;
            }
            break;
        }
    }
#endif
#if defined(__APPLE__) && defined(APS5_PRECISE_WAIT_TRACE)
    if (useMachSpinTicks) {
        const auto& tb = PreciseMachTimebase();
        const std::uint64_t pauseThresholdTicks =
            (200ULL * tb.denom + tb.numer - 1ULL) / tb.numer;
        for (;;) {
            const std::uint64_t curTicks = mach_absolute_time();
            if (curTicks >= machSpinDeadlineTicks) {
                break;
            }
            if (machSpinDeadlineTicks - curTicks > pauseThresholdTicks) {
                PreciseSpinPause();
            }
        }
    } else {
#endif
#if defined(__APPLE__) && !defined(APS5_PRECISE_WAIT_TRACE)
    std::uint64_t nowNanos =
        nanos <= spinTailNanos && deadlineNanos > startNanos ? startNanos : PreciseMonotonicNanos();
#else
    std::uint64_t nowNanos = PreciseMonotonicNanos();
#endif
    for (;;) {
        if (nowNanos >= deadlineNanos) {
            break;
        }
        if (deadlineNanos - nowNanos > 200ULL) {
            PreciseSpinPause();
        }
        nowNanos = PreciseMonotonicNanos();
    }
#if defined(__APPLE__) && defined(APS5_PRECISE_WAIT_TRACE)
    }
#endif
#ifdef APS5_PRECISE_WAIT_TRACE
    if (g_preciseWaitTraceActive) {
        g_preciseWaitTraceSample.t3 = PreciseMonotonicNanos();
        g_preciseWaitTraceSample.cpu3 = PreciseThreadCpuNanosInternal();
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
    if (micros > UINT64_MAX / 10ULL) [[unlikely]] {
        micros = UINT64_MAX / 10ULL;
    }
    PreciseSleepHundredNanos(micros * 10ULL);
    return;
#elif !defined(__EMSCRIPTEN__)
    if (micros > UINT64_MAX / 1000ULL) [[unlikely]] {
        micros = UINT64_MAX / 1000ULL;
    }
    PreciseSleepNanos(micros * 1000ULL);
    return;
#else
    constexpr unsigned long long kMaxMicros =
        static_cast<unsigned long long>((std::chrono::microseconds::max)().count());
    if (micros > kMaxMicros) [[unlikely]] {
        micros = kMaxMicros;
    }
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
            const std::uint64_t targetNanos = PreciseSafeDurationToNanos(deadline.time_since_epoch());
            const std::uint64_t targetHns =
                (targetNanos / 100ULL) + ((targetNanos % 100ULL) != 0 ? 1ULL : 0ULL);
            LARGE_INTEGER counter;
            QueryPerformanceCounter(&counter);
            const std::uint64_t nowTicks = static_cast<std::uint64_t>(counter.QuadPart);
            if (nowTicks < targetHns) {
                PreciseSleepFromCounter(counter, targetHns - nowTicks);
            }
            return;
        }
    }
    const std::uint64_t deadlineNanos = PreciseSafeDurationToNanos(deadline.time_since_epoch());
    if (deadlineNanos == 0) [[unlikely]] {
        return;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    const auto now = TClock::now();
    const std::uint64_t nowNanos = PreciseSafeDurationToNanos(now.time_since_epoch());
    if (nowNanos < deadlineNanos) {
        const std::uint64_t remainingNs = deadlineNanos - nowNanos;
        const std::uint64_t remainingHns =
            (remainingNs / 100ULL) + ((remainingNs % 100ULL) != 0 ? 1ULL : 0ULL);
        if (remainingHns > 0) {
            PreciseSleepFromCounter(counter, remainingHns);
        }
    }
    for (auto current = TClock::now(); PreciseSafeDurationToNanos(current.time_since_epoch()) < deadlineNanos; current = TClock::now()) {
        const std::uint64_t currentNanos = PreciseSafeDurationToNanos(current.time_since_epoch());
        if (deadlineNanos - currentNanos > 200ULL) {
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
        const std::uint64_t targetNanos = PreciseSafeDurationToNanos(deadline.time_since_epoch());
        const std::uint64_t startNanos = PreciseMonotonicNanos();
        if (startNanos < targetNanos) {
            PreciseSleepFromNanos(startNanos, targetNanos - startNanos);
        }
        return;
    }
#endif
    const std::uint64_t deadlineNanos = PreciseSafeDurationToNanos(deadline.time_since_epoch());
    if (deadlineNanos == 0) [[unlikely]] {
        return;
    }
    const std::uint64_t startNanos = PreciseMonotonicNanos();
    const auto now = TClock::now();
    const std::uint64_t nowNanos = PreciseSafeDurationToNanos(now.time_since_epoch());
    if (nowNanos < deadlineNanos) {
        PreciseSleepFromNanos(startNanos, deadlineNanos - nowNanos);
    }
    for (auto current = TClock::now(); PreciseSafeDurationToNanos(current.time_since_epoch()) < deadlineNanos; current = TClock::now()) {
        const std::uint64_t currentNanos = PreciseSafeDurationToNanos(current.time_since_epoch());
        if (deadlineNanos - currentNanos > 200ULL) {
            PreciseSpinPause();
        }
    }
#else
    const std::uint64_t deadlineNanos = PreciseSafeDurationToNanos(deadline.time_since_epoch());
    for (auto now = TClock::now(); PreciseSafeDurationToNanos(now.time_since_epoch()) < deadlineNanos; now = TClock::now()) {
        const std::uint64_t nowNanos = PreciseSafeDurationToNanos(now.time_since_epoch());
        const std::uint64_t diffNanos = deadlineNanos - nowNanos;
        const std::uint64_t diffMicros =
            (diffNanos / 1000ULL) + ((diffNanos % 1000ULL) != 0 ? 1ULL : 0ULL);
        PreciseSleepUs(diffMicros);
    }
#endif
}

#endif
