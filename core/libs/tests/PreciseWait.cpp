#include "prx/libc/include/PreciseWait.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <tuple>

namespace {

bool IsRunnerPreempted() {
#ifdef _WIN32
    char skip[8]{};
    const DWORD len = GetEnvironmentVariableA("APS5_SKIP_TIMING_TESTS", skip, sizeof(skip));
    if (len > 0 && skip[0] != '0') {
        return true;
    }
#else
    if (const char* skip = std::getenv("APS5_SKIP_TIMING_TESTS"); skip != nullptr && skip[0] != '\0' && skip[0] != '0') {
        return true;
    }
#endif
    const auto probeStart = std::chrono::steady_clock::now();
    auto prev = probeStart;
    std::chrono::nanoseconds maxGap{0};
    while (prev - probeStart < std::chrono::milliseconds(2)) {
        const auto cur = std::chrono::steady_clock::now();
        const auto gap = cur - prev;
        if (gap > maxGap) {
            maxGap = gap;
        }
        prev = cur;
    }
    return maxGap > std::chrono::microseconds(25);
}

template <class Fn>
bool CheckMedianSleepBound(
    const char* label,
    std::chrono::nanoseconds minDuration,
    std::chrono::nanoseconds maxMedian,
    Fn&& sleepFn) {
    std::array<std::chrono::nanoseconds, 65> samples{};
    std::chrono::nanoseconds bestMedian{};
    for (int attempt = 0; attempt < 3; ++attempt) {
        for (auto& sample : samples) {
            const auto start = std::chrono::steady_clock::now();
            sleepFn();
            sample = std::chrono::steady_clock::now() - start;
            if (sample < minDuration) {
                std::cerr << label << " returned before its requested duration\n";
                return false;
            }
        }
        std::sort(samples.begin(), samples.end());
        const auto median = samples[samples.size() / 2];
        if (attempt == 0 || median < bestMedian) {
            bestMedian = median;
        }
        if (bestMedian < maxMedian) {
            return true;
        }
    }
    if (IsRunnerPreempted() || samples.back() - samples.front() > std::chrono::milliseconds(1)) {
        std::clog << "Skipping strict median bound for " << label << " on busy runner ("
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(bestMedian).count() << " ns)\n";
        return true;
    }
    std::cerr << label << " median overslept target bound: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(bestMedian).count() << " ns\n";
    return false;
}

} // namespace

int main() {
    static_assert(PreciseHundredNanosToQpcTicks(500, 10000000ULL) == 500);
    static_assert(PreciseHundredNanosToQpcTicks(500, 3579545ULL) == 179);
    static_assert(PreciseHundredNanosToQpcTicks(500, 14318180ULL) == 716);
    static_assert(PreciseHundredNanosToQpcTicks(500, 24000000ULL) == 1200);
    static_assert(PrecisePauseThresholdTicks(3579545ULL) == 1);
    static_assert(PrecisePauseThresholdTicks(10000000ULL) == 2);
    static_assert(PrecisePauseThresholdTicks(24000000ULL) == 5);

    static_assert(PreciseSafeDurationToNanos(std::chrono::nanoseconds(0)) == 0ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::nanoseconds(-100)) == 0ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::nanoseconds(42)) == 42ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::microseconds(5)) == 5000ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::milliseconds(2)) == 2000000ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::seconds(1)) == 1000000000ULL);
    static_assert(PreciseSafeDurationToNanos((std::chrono::seconds::max)()) == UINT64_MAX);
    static_assert(PreciseSafeDurationToNanos((std::chrono::hours::max)()) == UINT64_MAX);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<int64_t, std::pico>(1000)) == 1ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<int64_t, std::pico>(1001)) == 2ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<int64_t, std::ratio<1, 3>>(1)) == 333333334ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<int64_t, std::ratio<1, 1000000000000000000LL>>(1000000000LL)) == 1ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<int64_t, std::ratio<INTMAX_MAX, 1>>(1)) == UINT64_MAX);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<int64_t, std::ratio<INTMAX_MAX, 1>>(0)) == 0ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<int64_t, std::ratio<INTMAX_MAX, 1>>(-5)) == 0ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<double>(0.067)) == 67000000ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<double>(1e-300)) == 1ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<double, std::pico>(1001.0)) == 2ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<double, std::ratio<INTMAX_MAX, 1>>(1e-20)) == 92233721ULL);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<double, std::ratio<INTMAX_MAX, 1>>(1.0)) == UINT64_MAX);
    static_assert(PreciseSafeDurationToNanos(std::chrono::duration<double, std::ratio<INTMAX_MAX, 1>>(-1.0)) == 0ULL);

    static_assert(PreciseMulDivCeilFallback64(0, 100, 10) == 0ULL);
    static_assert(PreciseMulDivCeilFallback64(10, 100, 10) == 100ULL);
    static_assert(PreciseMulDivCeilFallback64(1, 1, 3) == 1ULL);
    static_assert(PreciseMulDivCeilFallback64(5, 1000, 2000) == 3ULL);
    static_assert(PreciseMulDivCeilFallback64(1000000000000000ULL, 1000000000000000ULL, 10000000000000000ULL) == 100000000000000ULL);
    static_assert(PreciseMulDivCeil64(1000000000000000ULL, 1000000000000000ULL, 10000000000000000ULL) == 100000000000000ULL);
    static_assert(PreciseMulDivCeilFallback64(UINT64_MAX, 10, 1) == UINT64_MAX);
    static_assert(PreciseMulDivCeilFallback64(UINT64_MAX, 1, 1) == UINT64_MAX);
    static_assert(PreciseMulDivCeilFallback64(UINT64_MAX, UINT64_MAX, UINT64_MAX) == UINT64_MAX);
    static_assert(PreciseMulDivCeilFallback64(UINT64_MAX, UINT64_MAX, 1) == UINT64_MAX);
    static_assert(PreciseMulDivCeilFallback64(1, UINT64_MAX, UINT64_MAX) == 1ULL);
    static_assert(PreciseMulDivCeilFallback64(2, UINT64_MAX, UINT64_MAX) == 2ULL);
    static_assert(PreciseMulDivCeilFallback64(10, UINT64_MAX, 10) == UINT64_MAX);
    static_assert(PreciseMulDivCeilFallback64(UINT64_MAX / 2ULL + 1ULL, 3, 2) == 13835058055282163712ULL);
    static_assert(PreciseMulDivCeilFallback64(UINT64_MAX - 100ULL, 3, 2) == UINT64_MAX);
    static_assert(PreciseMulDivCeilFallback64(1ULL << 62, 1ULL << 62, 1ULL << 62) == 1ULL << 62);

    static_assert(PreciseMulDivCeil64(0, 100, 10) == 0ULL);
    static_assert(PreciseMulDivCeil64(10, 100, 10) == 100ULL);
    static_assert(PreciseMulDivCeil64(1, 1, 3) == 1ULL);
    static_assert(PreciseMulDivCeil64(5, 1000, 2000) == 3ULL);

    static_assert(PreciseComputeMachDeadline64(1000, 0, 1, 0).status == 0xFFFFFFFF);
    static_assert(PreciseComputeMachDeadline64(1000, 1, 0, 0).status == 0xFFFFFFFF);
    static_assert(PreciseComputeMachDeadline64(1500000ULL, 125, 3, 1000000ULL).status == 0);
    static_assert(PreciseComputeMachDeadline64(1500000ULL, 125, 3, 1000000ULL).sleepTicks == 36000ULL);
    static_assert(PreciseComputeMachDeadline64(1500000ULL, 125, 3, 1000000ULL).wakeDeadline == 1036000ULL);
    static_assert(PreciseComputeMachDeadline64(50000ULL, 1, 1, 200000ULL).status == 0);
    static_assert(PreciseComputeMachDeadline64(50000ULL, 1, 1, 200000ULL).sleepTicks == 50000ULL);
    static_assert(PreciseComputeMachDeadline64(50000ULL, 1, 1, 200000ULL).wakeDeadline == 250000ULL);
    static_assert(PreciseComputeMachDeadline64(UINT64_MAX, 1, 2, 0).status == 0xFFFFFFFE);
    static_assert(PreciseComputeMachDeadline64(1000ULL, 1, 1, UINT64_MAX - 500ULL).status == 0xFFFFFFFD);
    static_assert(PreciseComputeMachDeadline64(1000ULL, 1, 1, UINT64_MAX - 1000ULL).status == 0);
    static_assert(PreciseComputeMachDeadline64(1000ULL, 1, 1, UINT64_MAX - 1000ULL).wakeDeadline == UINT64_MAX);

    static_assert(PreciseComputeMachDeadline(1000, 0, 1, 0).status == 0xFFFFFFFF);
    static_assert(PreciseComputeMachDeadline(1000, 1, 0, 0).status == 0xFFFFFFFF);
    static_assert(PreciseComputeMachDeadline(1500000ULL, 125, 3, 1000000ULL).status == 0);
    static_assert(PreciseComputeMachDeadline(1500000ULL, 125, 3, 1000000ULL).sleepTicks == 36000ULL);
    static_assert(PreciseComputeMachDeadline(1500000ULL, 125, 3, 1000000ULL).wakeDeadline == 1036000ULL);
    static_assert(PreciseComputeMachDeadline(50000ULL, 1, 1, 200000ULL).status == 0);
    static_assert(PreciseComputeMachDeadline(50000ULL, 1, 1, 200000ULL).sleepTicks == 50000ULL);
    static_assert(PreciseComputeMachDeadline(50000ULL, 1, 1, 200000ULL).wakeDeadline == 250000ULL);
    static_assert(PreciseComputeMachDeadline(UINT64_MAX, 1, 2, 0).status == 0xFFFFFFFE);
    static_assert(PreciseComputeMachDeadline(1000ULL, 1, 1, UINT64_MAX - 500ULL).status == 0xFFFFFFFD);
    static_assert(PreciseComputeMachDeadline(1000ULL, 1, 1, UINT64_MAX - 1000ULL).status == 0);
    static_assert(PreciseComputeMachDeadline(1000ULL, 1, 1, UINT64_MAX - 1000ULL).wakeDeadline == UINT64_MAX);

    const double nanVal = std::numeric_limits<double>::quiet_NaN();
    if (PreciseSafeDurationToNanos(std::chrono::duration<double>(nanVal)) != 0ULL) {
        std::cerr << "PreciseSafeDurationToNanos NaN check failed\n";
        return 1;
    }
    const double infVal = std::numeric_limits<double>::infinity();
    if (PreciseSafeDurationToNanos(std::chrono::duration<double>(infVal)) != UINT64_MAX) {
        std::cerr << "PreciseSafeDurationToNanos +inf check failed\n";
        return 1;
    }
    if (PreciseSafeDurationToNanos(std::chrono::duration<double>(-infVal)) != 0ULL) {
        std::cerr << "PreciseSafeDurationToNanos -inf check failed\n";
        return 1;
    }
    if (PreciseSafeDurationToNanos(std::chrono::duration<double>(-100.5)) != 0ULL) {
        std::cerr << "PreciseSafeDurationToNanos negative float check failed\n";
        return 1;
    }
    if (PreciseSafeDurationToNanos(std::chrono::duration<double, std::milli>(1.5)) != 1500000ULL) {
        std::cerr << "PreciseSafeDurationToNanos float milli check failed\n";
        return 1;
    }
    if (PreciseSafeDurationToNanos(std::chrono::duration<double, std::ratio<INTMAX_MAX, 1>>(1e-20)) != 92233721ULL) {
        std::cerr << "PreciseSafeDurationToNanos tiny double huge period check failed\n";
        return 1;
    }
    if (PreciseSafeDurationToNanos(std::chrono::duration<double>(0.067)) != 67000000ULL) {
        std::cerr << "PreciseSafeDurationToNanos 0.067 float check failed\n";
        return 1;
    }
    if (PreciseSafeDurationToNanos(std::chrono::duration<double>(1e-300)) != 1ULL) {
        std::cerr << "PreciseSafeDurationToNanos underflow check failed\n";
        return 1;
    }

#if defined(__SIZEOF_INT128__)
    using Int128Nanos = std::chrono::duration<__int128, std::nano>;
    const __int128 huge128 = (static_cast<__int128>(1) << 65);
    if (PreciseSafeDurationToNanos(Int128Nanos(huge128)) != UINT64_MAX) {
        std::cerr << "PreciseSafeDurationToNanos wide Rep overflow check failed\n";
        return 1;
    }
    if (PreciseSafeDurationToNanos(Int128Nanos(500)) != 500ULL) {
        std::cerr << "PreciseSafeDurationToNanos wide Rep normal check failed\n";
        return 1;
    }
#endif

    for (const auto& [sleepNanos, numer, denom, nowTicks] : {
             std::tuple{1000ULL, 0U, 1U, 0ULL},
             std::tuple{1000ULL, 1U, 0U, 0ULL},
             std::tuple{1500000ULL, 125U, 3U, 1000000ULL},
             std::tuple{50000ULL, 1U, 1U, 200000ULL},
             std::tuple{UINT64_MAX, 1U, 2U, 0ULL},
             std::tuple{1000ULL, 1U, 1U, UINT64_MAX - 500ULL},
             std::tuple{1000ULL, 1U, 1U, UINT64_MAX - 1000ULL}}) {
        const auto r1 = PreciseComputeMachDeadline(sleepNanos, numer, denom, nowTicks);
        const auto r2 = PreciseComputeMachDeadline64(sleepNanos, numer, denom, nowTicks);
        if (r1.status != r2.status || r1.sleepTicks != r2.sleepTicks || r1.wakeDeadline != r2.wakeDeadline) {
            std::cerr << "PreciseComputeMachDeadline vs PreciseComputeMachDeadline64 mismatch\n";
            return 1;
        }
    }

#if defined(__SIZEOF_INT128__) && !defined(_WIN32)
    for (const auto& [r, num, den] : {
             std::tuple{0ULL, 100ULL, 10ULL},
             std::tuple{10ULL, 0ULL, 10ULL},
             std::tuple{10ULL, 10ULL, 0ULL},
             std::tuple{1ULL, 1ULL, 1ULL},
             std::tuple{1ULL, 1ULL, 3ULL},
             std::tuple{5ULL, 1000ULL, 2000ULL},
             std::tuple{1000000000000000ULL, 1000000000000000ULL, 10000000000000000ULL},
             std::tuple{UINT64_MAX, 1ULL, 1ULL},
             std::tuple{UINT64_MAX, 2ULL, 1ULL},
             std::tuple{UINT64_MAX, 10ULL, 2ULL},
             std::tuple{UINT64_MAX, UINT64_MAX, UINT64_MAX},
             std::tuple{UINT64_MAX, UINT64_MAX, 1ULL},
             std::tuple{1ULL, UINT64_MAX, UINT64_MAX},
             std::tuple{2ULL, UINT64_MAX, UINT64_MAX},
             std::tuple{10ULL, UINT64_MAX, 10ULL},
             std::tuple{UINT64_MAX, 10ULL, 10ULL},
             std::tuple{UINT64_MAX / 2ULL + 1ULL, 3ULL, 2ULL},
             std::tuple{UINT64_MAX - 100ULL, 3ULL, 2ULL},
             std::tuple{1ULL << 63, 1ULL << 63, 1ULL << 62},
             std::tuple{1ULL << 62, 1ULL << 62, 1ULL << 62}}) {
        const std::uint64_t f1 = PreciseMulDivCeilFallback64(r, num, den);
        const std::uint64_t f2 = PreciseMulDivCeil64(r, num, den);
        if (f1 != f2) {
            std::cerr << "PreciseMulDivCeilFallback64 vs PreciseMulDivCeil64 mismatch\n";
            return 1;
        }
    }
#endif

    for (const std::uint64_t freq : {3579545ULL, 10000000ULL, 14318180ULL, 19200000ULL, 24000000ULL}) {
        for (const unsigned long long hundredNanos : {1ULL, 5ULL, 100ULL, 300ULL, 500ULL, 10000ULL, 53330ULL}) {
            const std::uint64_t ticks = PreciseHundredNanosToQpcTicks(hundredNanos, freq);
            if (ticks * 10000000ULL < hundredNanos * freq) {
                std::cerr << "PreciseHundredNanosToQpcTicks rounded below requested duration\n";
                return 1;
            }
        }
    }

    PreciseSleepUs(0);

    for (const unsigned long long micros : {50ULL, 1000ULL}) {
        const auto start = std::chrono::steady_clock::now();
        PreciseSleepUs(micros);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed < std::chrono::microseconds(micros)) {
            std::cerr << "PreciseSleepUs returned before its requested duration\n";
            return 1;
        }
    }

#if !defined(_WIN32)
    if (!CheckMedianSleepBound(
            "PreciseSleepUs(50)",
            std::chrono::microseconds(50),
            std::chrono::microseconds(90),
            [] { PreciseSleepUs(50); })) {
        return 1;
    }
#endif

    if (!CheckMedianSleepBound(
            "PreciseSleepUntil(+50us)",
            std::chrono::microseconds(50),
            std::chrono::microseconds(90),
            [] { PreciseSleepUntil(std::chrono::steady_clock::now() + std::chrono::microseconds(50)); })) {
        return 1;
    }

    if (!CheckMedianSleepBound(
            "PreciseSleepUs(1500)",
            std::chrono::microseconds(1500),
            std::chrono::microseconds(1650),
            [] { PreciseSleepUs(1500); })) {
        return 1;
    }

    const auto subMicroDeadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(500);
    PreciseSleepUntil(subMicroDeadline);
    if (std::chrono::steady_clock::now() < subMicroDeadline) {
        std::cerr << "PreciseSleepUntil returned before sub-microsecond deadline\n";
        return 1;
    }

    const auto millisecondDeadline = std::chrono::steady_clock::now() + std::chrono::microseconds(1000);
    PreciseSleepUntil(millisecondDeadline);
    if (std::chrono::steady_clock::now() < millisecondDeadline) {
        std::cerr << "PreciseSleepUntil returned before millisecond deadline\n";
        return 1;
    }

    PreciseSleepUntil(std::chrono::steady_clock::time_point{});
    PreciseSleepUntil(std::chrono::time_point<std::chrono::steady_clock, std::chrono::seconds>{});
    const auto coarseDeadline = std::chrono::ceil<std::chrono::microseconds>(
        std::chrono::steady_clock::now() + std::chrono::microseconds(25));
    PreciseSleepUntil(coarseDeadline);
    if (std::chrono::steady_clock::now() < coarseDeadline) {
        std::cerr << "PreciseSleepUntil returned before coarse steady_clock deadline\n";
        return 1;
    }

#ifdef _WIN32
    for (const HANDLE badHandle : {static_cast<HANDLE>(nullptr), GetCurrentProcess()}) {
        const auto start = std::chrono::steady_clock::now();
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        PreciseSleepFromCounterWithTimer(counter, 10000ULL, badHandle, true);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed < std::chrono::microseconds(1000)) {
            std::cerr << "PreciseSleepFromCounterWithTimer fallback returned early\n";
            return 1;
        }
    }
#endif

#ifdef __linux__
    if (PreciseLinuxLowTimerSlack() && prctl(PR_GET_TIMERSLACK, 0UL, 0UL, 0UL, 0UL) != 1L) {
        std::cerr << "PreciseLinuxLowTimerSlack did not configure 1ns timer slack\n";
        return 1;
    }
    for (const unsigned long long nanos : {80000ULL, 1000000ULL}) {
        const auto start = std::chrono::steady_clock::now();
        PreciseSleepFromNanosWithSlack(PreciseMonotonicNanos(), nanos, false);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed < std::chrono::nanoseconds(nanos)) {
            std::cerr << "PreciseSleepFromNanosWithSlack conservative fallback returned early\n";
            return 1;
        }
    }
#endif

    return 0;
}
