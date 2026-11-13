/*
 * SPDX-FileCopyrightText: The LineageOS Project
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Vibrator.h"

#include <cutils/properties.h>
#include <inttypes.h>
#include <log/log.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

#include <android-base/properties.h>

#include "aac_vibra_function.h"
#include "ics_haptic_function.h"

#define RICHTAP_JND_EFFECT_CLICK 12296
#define RICHTAP_JND_EFFECT_TICK 12297
#define RICHTAP_STRENGTH_MAX 100

#define DOUBLE_CLICK_GAP_MS 70
#define DOUBLE_CLICK_DURATION_MS 88

#define SFDC_F0_MIN 50.0f
#define SFDC_F0_MAX 500.0f

#define HE_V1_VERSION 1
#define HE_V1_TYPE_TRANSIENT 0x1001
#define HE_V1_INTENSITY 100
#define HE_V1_FREQUENCY 65

#define COMPOSE_DELAY_MAX_MS 1000
#define COMPOSE_SIZE_MAX 16
#define LEVEL_GAMMA 1.2f

#define PRIMITIVE_CLICK_MS 18
#define PRIMITIVE_TICK_MS 12

// Continuous on() windows shorter than this are played as a plain transient.
#define ON_TRANSIENT_MAX_MS 100
// Only ring/notification-length windows get the attack click before the
// hold. Mid-size windows (app close, long press) hold without an attack
// so they feel like one clean pulse instead of a double tap.
#define ON_ATTACK_MIN_MS 400
// The framework sets the amplitude right after on(); wait this long for it.
#define ON_AMPLITUDE_WAIT_MS 5

namespace aidl {
namespace android {
namespace hardware {
namespace vibrator {

// Serialize cancellation with every hardware submission, including delayed work.
static std::mutex gLock;
static std::condition_variable gCondition;
static uint64_t gGeneration = 0;
static std::shared_ptr<IVibratorCallback> gCallback;
static bool gOnActive = false;
static int32_t gOnPendingMs = 0;
static float gAmplitude = 1.0f;

static void notifyComplete(const std::shared_ptr<IVibratorCallback>& callback) {
    if (callback != nullptr && !callback->onComplete().isOk()) {
        ALOGE("Vibrator completion callback failed");
    }
}

// Called with gLock held. Complete an expired operation even if its worker ran late.
static uint64_t beginLocked(const std::shared_ptr<IVibratorCallback>& callback) {
    auto previous = std::move(gCallback);
    if (previous != nullptr) {
        std::thread([previous] { notifyComplete(previous); }).detach();
    }
    ++gGeneration;
    gCallback = callback;
    gOnActive = false;
    gOnPendingMs = 0;
    gCondition.notify_all();
    return gGeneration;
}

// Called with gLock held. A canceled worker must not consume a newer callback.
static std::shared_ptr<IVibratorCallback> finishLocked(uint64_t generation) {
    if (gGeneration != generation) return nullptr;
    gOnActive = false;
    gOnPendingMs = 0;
    return std::move(gCallback);
}

static bool waitLocked(std::unique_lock<std::mutex>& lock, uint64_t generation,
                       std::chrono::steady_clock::time_point deadline) {
    return !gCondition.wait_until(lock, deadline,
                                 [generation] { return gGeneration != generation; });
}

static void sfdcF0Callback(float f0) {
    ALOGD("sfdc f0 update: %.1f", f0);
}

static bool primitiveToEffect(CompositePrimitive p, uint32_t* id, int32_t* durationMs) {
    switch (p) {
        case CompositePrimitive::CLICK:
        case CompositePrimitive::THUD:
            *id = RICHTAP_JND_EFFECT_CLICK;
            *durationMs = PRIMITIVE_CLICK_MS;
            return true;
        case CompositePrimitive::LIGHT_TICK:
        case CompositePrimitive::LOW_TICK:
            *id = RICHTAP_JND_EFFECT_TICK;
            *durationMs = PRIMITIVE_TICK_MS;
            return true;
        default:
            return false;
    }
}

// Map a 0..1 level to the RichTap prebaked strength (0..100).
static int32_t levelToStrength(float level) {
    level = std::clamp(level, 0.0f, 1.0f);
    int32_t strength = (int32_t)std::lround(powf(level, LEVEL_GAMMA) * RICHTAP_STRENGTH_MAX);
    if (level > 0.0f && strength < 1) {
        strength = 1;
    }
    return strength;
}

// The framework turns the user's intensity setting into an EffectStrength
// for prebaked effects.
static float effectStrengthToLevel(EffectStrength es) {
    switch (es) {
        case EffectStrength::LIGHT:
            return 0.5f;
        case EffectStrength::MEDIUM:
            return 0.75f;
        case EffectStrength::STRONG:
        default:
            return 1.0f;
    }
}

static int32_t setGain(uint8_t gain) {
    aac_vibra_dynamic_scale(gain);
    return aac_vibra_setAmplitude(gain);
}

// Play a prebaked transient at the given level. Keep the global gain at
// full so a stale continuous amplitude does not scale it a second time.
static int32_t playTransient(uint32_t id, float level) {
    setGain(0xff);
    return aac_vibra_looper_prebaked_effect(id, levelToStrength(level));
}

static void updateF0() {
    static const int32_t he[5] = {HE_V1_VERSION, HE_V1_TYPE_TRANSIENT, 0, HE_V1_INTENSITY,
                                  HE_V1_FREQUENCY};
    float f0;

    if (sfdc_calibrate(he, 5) >= 0) {
        f0 = sfdc_get_transient_fc();
    } else {
        f0 = sfdc_get_continuous_f0();
    }

    if (f0 >= SFDC_F0_MIN && f0 <= SFDC_F0_MAX) {
        aac_vibra_setting_f0(0, f0);
    }
}

// Continuous line mode has no transient calibration of its own; driving it at
// the transient-tuned frequency puts the LRA off resonance and makes its
// output weak. Re-seed the plain continuous f0 before every continuous play.
static void updateContinuousF0() {
    float f0 = sfdc_get_continuous_f0();

    if (f0 < SFDC_F0_MIN || f0 > SFDC_F0_MAX) {
        f0 = sfdc_get_manufactory_f0();
    }

    if (f0 >= SFDC_F0_MIN && f0 <= SFDC_F0_MAX) {
        aac_vibra_setting_f0(0, f0);
    }
}

// Short windows (every pulse of ring and notification waveforms) play as a
// transient click, which is far crisper than a stub of continuous drive.
// Longer windows get the click as an attack and hold with continuous drive.
static int32_t playOnWindow(int32_t timeoutMs, float amplitude) {
    // Short window: a single crisp transient, nothing more.
    if (timeoutMs <= ON_TRANSIENT_MAX_MS) {
        return playTransient(RICHTAP_JND_EFFECT_CLICK, amplitude);
    }

    int32_t ret = 0;
    // Attack click only for ring/notification-length windows; a mid-size
    // window (app close, long press) skips it so it is one clean pulse.
    if (timeoutMs > ON_ATTACK_MIN_MS) {
        ret = playTransient(RICHTAP_JND_EFFECT_CLICK, amplitude);
        if (ret < 0) {
            ALOGE("AAC on transient failed: %d\n", ret);
            return ret;
        }
        if (timeoutMs <= ret) {
            return ret;
        }
    }

    updateContinuousF0();
    setGain((uint8_t)std::lround(amplitude * 0xff));
    ret = aac_vibra_looper_on(timeoutMs - ret);
    if (ret < 0) {
        ALOGE("AAC on hold failed: %d\n", ret);
    }
    return ret;
}

Vibrator::Vibrator() {
    uint32_t deviceType = 0;

    std::string devicePath = ::android::base::GetProperty("vendor.vibrator.device", "");
    if (!devicePath.empty()) {
        setenv("RICHTAP_DEVICE_PATH", devicePath.c_str(), 1);
    } else {
        ALOGE("No vibrator device set!");
    }

    int32_t ret = aac_vibra_init(&deviceType);
    if (ret) {
        ALOGE("AAC init failed: %d\n", ret);
        return;
    }

    aac_vibra_looper_start();

    if (sfdc_initialize(sfdcF0Callback) < 0) {
        ALOGW("sfdc init failed, running without f0 tracking\n");
    } else {
        float f0 = sfdc_get_manufactory_f0();
        if (f0 >= SFDC_F0_MIN && f0 <= SFDC_F0_MAX) {
            aac_vibra_setting_f0(0, f0);
            ALOGI("seeded manufactory f0: %.1f\n", f0);
        }
    }

    ALOGI("AAC init success: %u\n", deviceType);
}

ndk::ScopedAStatus Vibrator::getCapabilities(int32_t* _aidl_return) {
    *_aidl_return = IVibrator::CAP_ON_CALLBACK | IVibrator::CAP_PERFORM_CALLBACK |
                    IVibrator::CAP_AMPLITUDE_CONTROL | IVibrator::CAP_COMPOSE_EFFECTS |
                    IVibrator::CAP_GET_RESONANT_FREQUENCY;

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::off() {
    std::shared_ptr<IVibratorCallback> callback;
    bool stopped;
    {
        std::lock_guard<std::mutex> lock(gLock);
        ++gGeneration;
        gOnActive = false;
        gOnPendingMs = 0;
        gAmplitude = 1.0f;
        callback = std::move(gCallback);
        stopped = aac_vibra_looper_stopPerformHe();
    }
    gCondition.notify_all();
    if (callback != nullptr) {
        std::thread([callback] { notifyComplete(callback); }).detach();
    }
    if (!stopped) {
        ALOGE("AAC stop failed\n");
        return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_SERVICE_SPECIFIC));
    }

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::on(int32_t timeoutMs,
                                const std::shared_ptr<IVibratorCallback>& callback) {
    if (timeoutMs <= 0) {
        return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_ILLEGAL_ARGUMENT));
    }

    timeoutMs = static_cast<int32_t>(std::min<double>(
            timeoutMs * 1.10, std::numeric_limits<int32_t>::max()));

    std::lock_guard<std::mutex> lock(gLock);
    uint64_t generation = beginLocked(callback);
    gOnActive = true;
    gOnPendingMs = timeoutMs;
    const auto start = std::chrono::steady_clock::now();

    // Fallback for callers that never follow up with setAmplitude().
    std::thread([generation, timeoutMs, start] {
        std::unique_lock<std::mutex> lock(gLock);
        if (!waitLocked(lock, generation, start + std::chrono::milliseconds(ON_AMPLITUDE_WAIT_MS))) {
            return;
        }
        if (gOnPendingMs > 0) {
            gOnPendingMs = 0;
            if (playOnWindow(timeoutMs, gAmplitude) < 0) {
                ALOGE("AAC pending on failed");
                auto callback = finishLocked(generation);
                lock.unlock();
                notifyComplete(callback);
                return;
            }
        }
        if (!waitLocked(lock, generation, start + std::chrono::milliseconds(timeoutMs))) return;
        auto callback = finishLocked(generation);
        lock.unlock();
        notifyComplete(callback);
    }).detach();

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::perform(Effect effect, EffectStrength es,
                                     const std::shared_ptr<IVibratorCallback>& callback,
                                     int32_t* _aidl_return) {
    uint32_t id;
    float level;

    // Keep the effects distinct like the reference Pixel tuning: ticks are
    // the light tick waveform, clicks sit below full drive and only heavy
    // clicks and thuds use all of it.
    switch (effect) {
        case Effect::TICK:
            id = RICHTAP_JND_EFFECT_TICK;
            level = 1.0f;
            break;
        case Effect::TEXTURE_TICK:
            id = RICHTAP_JND_EFFECT_TICK;
            level = 0.5f;
            break;
        case Effect::CLICK:
        case Effect::DOUBLE_CLICK:
            id = RICHTAP_JND_EFFECT_CLICK;
            level = 0.85f;
            break;
        case Effect::POP:
            id = RICHTAP_JND_EFFECT_CLICK;
            level = 0.7f;
            break;
        case Effect::THUD:
        case Effect::HEAVY_CLICK:
            id = RICHTAP_JND_EFFECT_CLICK;
            level = 1.0f;
            break;
        default:
            return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
    }

    if (es != EffectStrength::LIGHT && es != EffectStrength::MEDIUM &&
        es != EffectStrength::STRONG) {
        return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
    }

    level *= effectStrengthToLevel(es);

    std::lock_guard<std::mutex> lock(gLock);
    updateF0();

    int32_t ret = playTransient(id, level);
    if (ret < 0) {
        ALOGE("AAC perform failed: %d\n", ret);
        return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_SERVICE_SPECIFIC));
    }

    uint64_t generation = beginLocked(callback);
    const auto start = std::chrono::steady_clock::now();

    if (effect == Effect::DOUBLE_CLICK) ret = DOUBLE_CLICK_DURATION_MS;

    std::thread([generation, effect, id, level, ret, start] {
        std::unique_lock<std::mutex> lock(gLock);
        auto deadline = start + std::chrono::milliseconds(ret);
        if (effect == Effect::DOUBLE_CLICK) {
            if (!waitLocked(lock, generation, start + std::chrono::milliseconds(DOUBLE_CLICK_GAP_MS))) {
                return;
            }
            if (playTransient(id, level) < 0) {
                ALOGE("AAC second click failed");
                auto callback = finishLocked(generation);
                lock.unlock();
                notifyComplete(callback);
                return;
            }
            // A late second submission still needs its full nominal playback window.
            deadline = std::max(deadline, std::chrono::steady_clock::now() +
                                                  std::chrono::milliseconds(
                                                          DOUBLE_CLICK_DURATION_MS - DOUBLE_CLICK_GAP_MS));
        }
        if (!waitLocked(lock, generation, deadline)) return;
        auto callback = finishLocked(generation);
        lock.unlock();
        notifyComplete(callback);
    }).detach();

    *_aidl_return = ret;

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::getSupportedEffects(std::vector<Effect>* _aidl_return) {
    *_aidl_return = {Effect::CLICK, Effect::DOUBLE_CLICK, Effect::TICK,        Effect::THUD,
                     Effect::POP,   Effect::HEAVY_CLICK,  Effect::TEXTURE_TICK};

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::setAmplitude(float amplitude) {
    if (!std::isfinite(amplitude) || amplitude <= 0.0f || amplitude > 1.0f) {
        return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_ILLEGAL_ARGUMENT));
    }

    std::lock_guard<std::mutex> lock(gLock);
    gAmplitude = amplitude;
    int32_t pendingMs = gOnPendingMs;
    gOnPendingMs = 0;

    if (pendingMs > 0) {
        if (playOnWindow(pendingMs, amplitude) < 0) {
            auto callback = finishLocked(gGeneration);
            ++gGeneration;
            gCondition.notify_all();
            if (callback != nullptr) {
                std::thread([callback] { notifyComplete(callback); }).detach();
            }
            return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_SERVICE_SPECIFIC));
        }
        return ndk::ScopedAStatus::ok();
    }

    // Amplitude control applies only to on(), never prebaked/composed effects.
    if (!gOnActive) return ndk::ScopedAStatus::ok();

    int32_t ret = setGain((uint8_t)std::lround(amplitude * 0xff));
    if (ret) {
        ALOGE("AAC set amplitude failed: %d\n", ret);
        return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_SERVICE_SPECIFIC));
    }

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::setExternalControl(bool enabled __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

ndk::ScopedAStatus Vibrator::getCompositionDelayMax(int32_t* maxDelayMs) {
    *maxDelayMs = COMPOSE_DELAY_MAX_MS;

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::getCompositionSizeMax(int32_t* maxSize) {
    *maxSize = COMPOSE_SIZE_MAX;

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::getSupportedPrimitives(std::vector<CompositePrimitive>* supported) {
    *supported = {CompositePrimitive::NOOP, CompositePrimitive::CLICK, CompositePrimitive::THUD,
                  CompositePrimitive::LIGHT_TICK, CompositePrimitive::LOW_TICK};

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::getPrimitiveDuration(CompositePrimitive primitive,
                                                  int32_t* durationMs) {
    uint32_t id;

    if (primitive == CompositePrimitive::NOOP) {
        *durationMs = 0;
        return ndk::ScopedAStatus::ok();
    }

    if (!primitiveToEffect(primitive, &id, durationMs)) {
        return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
    }

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::compose(const std::vector<CompositeEffect>& composite,
                                     const std::shared_ptr<IVibratorCallback>& callback) {
    if (composite.empty() || composite.size() > COMPOSE_SIZE_MAX) {
        return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_ILLEGAL_ARGUMENT));
    }

    for (const auto& e : composite) {
        uint32_t id;
        int32_t ms;

        if (e.delayMs < 0 || e.delayMs > COMPOSE_DELAY_MAX_MS || !std::isfinite(e.scale) ||
            e.scale < 0.0f || e.scale > 1.0f) {
            return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_ILLEGAL_ARGUMENT));
        }
        if (e.primitive != CompositePrimitive::NOOP && !primitiveToEffect(e.primitive, &id, &ms)) {
            return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
        }
    }

    std::lock_guard<std::mutex> lock(gLock);
    uint64_t generation = beginLocked(callback);
    std::thread([generation, composite] {
        std::unique_lock<std::mutex> lock(gLock);
        if (gGeneration != generation) return;
        updateF0();
        for (const auto& e : composite) {
            uint32_t id;
            int32_t ms;

            if (!waitLocked(lock, generation,
                            std::chrono::steady_clock::now() + std::chrono::milliseconds(e.delayMs))) {
                return;
            }
            if (!primitiveToEffect(e.primitive, &id, &ms)) continue;
            if (playTransient(id, e.scale) < 0) {
                ALOGE("AAC composed primitive failed");
                break;
            }
            if (!waitLocked(lock, generation,
                            std::chrono::steady_clock::now() + std::chrono::milliseconds(ms))) {
                return;
            }
        }
        auto callback = finishLocked(generation);
        lock.unlock();
        notifyComplete(callback);
    }).detach();

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::getSupportedAlwaysOnEffects(
        std::vector<Effect>* _aidl_return __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

ndk::ScopedAStatus Vibrator::alwaysOnEnable(int32_t id __unused, Effect effect __unused,
                                            EffectStrength strength __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

ndk::ScopedAStatus Vibrator::alwaysOnDisable(int32_t id __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

ndk::ScopedAStatus Vibrator::getResonantFrequency(float* resonantFreqHz) {
    float f0 = sfdc_get_continuous_f0();
    if (f0 < SFDC_F0_MIN || f0 > SFDC_F0_MAX) {
        f0 = sfdc_get_manufactory_f0();
    }
    if (f0 < SFDC_F0_MIN || f0 > SFDC_F0_MAX) {
        return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_ILLEGAL_STATE));
    }

    *resonantFreqHz = f0;

    return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus Vibrator::getQFactor(float* qFactor __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

ndk::ScopedAStatus Vibrator::getFrequencyResolution(float* freqResolutionHz __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

ndk::ScopedAStatus Vibrator::getFrequencyMinimum(float* freqMinimumHz __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

ndk::ScopedAStatus Vibrator::getBandwidthAmplitudeMap(std::vector<float>* _aidl_return __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

ndk::ScopedAStatus Vibrator::getPwlePrimitiveDurationMax(int32_t* durationMs __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

ndk::ScopedAStatus Vibrator::getPwleCompositionSizeMax(int32_t* maxSize __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

ndk::ScopedAStatus Vibrator::getSupportedBraking(std::vector<Braking>* supported __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

ndk::ScopedAStatus Vibrator::composePwle(const std::vector<PrimitivePwle>& composite __unused,
                                         const std::shared_ptr<IVibratorCallback>& callback
                                                 __unused) {
    return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
}

}  // namespace vibrator
}  // namespace hardware
}  // namespace android
}  // namespace aidl
