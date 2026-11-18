#include <cstdlib>
// Android host (NativeActivity): memory (mmap at fixed addresses), pthreads, the app's window (ANativeWindow),
// game controllers + touch, AAudio, the Vulkan loader and android surface, a signal crash handler and
// android_main. Game data lives in the app's external files dir (see docs/ANDROID.md):
//   <files>/analysis/{main,sdk}.bin   module images (tools/nso2bin.py)
//   <files>/extract/romfs/...         the RomFS
//   <files>/user/...                  saves
//   <files>/config.txt                optional: resolution=WxH
#include <aaudio/AAudio.h>
#include <android/log.h>
#include <android/native_window.h>
#include <android/window.h>
#include <android_native_app_glue.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <jni.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>

#include <nx/game.h>
#include <nx/runtime.h>

#include "kernel.h"
#include "platform.h"
#include "platform_android.h"
#include "config.h"
#include "touch_layout.h"
#include "ui/ui.h"

// Android input: game controllers and touch (split from platform_android.cpp).
namespace nx::plat {

// ---- input: game controllers + touch --------------------------------------------------------------------------
// Touch: left half = virtual stick (relative to where the finger went down); right half = four buttons in a
// diamond around the thumb (Y attack up-left, X special up-right, B dash down-left, A interact down-right), top
// corners L / R, top centre +.

namespace {
std::mutex g_in;
PadState g_pad_buttons;  // controller state
struct Touch {
    int32_t id;
    float x0, y0, x, y;
    bool left;
    uint64_t button;
    bool run = false;      // the stick: holding the run button (read_pad)
    uint64_t past_ns = 0;  // the stick: when it went past the run distance, 0 while inside it
    // ACNH native touch (a finger on the free part of the screen, or the stick's: a tap clicks, a drag looks around).
    bool free = false;    // not on a button and not the stick: the camera / taps
    bool moved = false;   // went past the tap slop: no longer a tap
    uint64_t t0_ns = 0;   // when it went down
    float vx = 0, vy = 0; // camera: the swipe's smoothed speed (px/s), set by the moves, only read by read_pad
    uint64_t move_ns = 0; // when the last move came
    int dir = 0;          // once it moved: 1 mostly vertical (the camera, a menu's scroll), 2 mostly horizontal (a flick)
    float sy = 0;         // a menu's scroll: where the last step was taken
    bool pair = false;    // one of two fingers that went down together: a two-finger tap (B), not a click
};
std::vector<Touch> g_touches;
// Buttons the gestures press for a moment (a tap on the phone icon -> ZL, a flick -> d-pad, two fingers -> B, ...),
// one after another, applied in read_pad. Held long enough for a frame at 15 fps to see, with a gap so that two in a
// row are two presses.
struct Pulse {
    uint64_t button, from, until;
};
std::deque<Pulse> g_pulses;
uint64_t g_pulse_end = 0;
void pulse(uint64_t button) {
    if (g_pulses.size() >= 6) return;
    const uint64_t from = std::max(monotonic_ns(), g_pulse_end + 90'000'000);
    g_pulses.push_back({button, from, from + 110'000'000});
    g_pulse_end = from + 110'000'000;
}
std::atomic<bool> g_calibrate_phone{false};  // Settings > Controls > Set the phone icon: the next tap is its spot
uint64_t g_last_pair_ns = 0;

bool is_acnh() {
    static const bool acnh = std::strcmp(nx::game::kInfo.id, "acnh") == 0;
    return acnh;
}

// Settings (config.txt / Settings > Controls): input.touch_native (taps click the game's menus, a drag moves the camera),
// input.camera_sens (percent), input.camera_invert, input.touch_phone_rect = "x0,y0,x1,y1" (0..1 across the screen):
// a tap there, with the stick not in use, presses ZL (the NookPhone); empty: none.
bool native_touch() { return is_acnh() && config_get_bool("input.touch_native", true); }

bool phone_rect(float& x0, float& y0, float& x1, float& y1) {
    return std::sscanf(config_get("input.touch_phone_rect", "").c_str(), "%f,%f,%f,%f", &x0, &y0, &x1, &y1) == 4;
}
float g_w = 1, g_h = 1;
bool g_gamepad = false;  // the last input came from a controller: the drawn touch controls hide
bool g_pad_connected = false;  // a controller is connected (platform_android.cpp polls the system's input devices)
std::atomic<uint64_t> g_ui_taps{0};     // ui_taps
std::atomic<bool> g_ui_finger{false};  // ui_finger_down

// The button the on-screen stick holds to run (read_pad): Wonder's Y (dash), ACNH's B.
uint64_t run_button() {
    static const uint64_t run = std::strcmp(nx::game::kInfo.id, "acnh") == 0 ? kB : kY;
    return run;
}

// Settings > Controls > Confirm / cancel buttons (input.layout), as on the other hosts (platform_sdl.cpp): "console"
// keeps the game's A on the right face button, "xbox" puts it on the bottom one. Read again when the config changes.
bool xbox_layout() {
    static uint64_t gen = ~0ull;
    static bool xbox = false;
    if (uint64_t g = config_generation(); g != gen) {
        gen = g;
        xbox = config_get("input.layout", "console") == "xbox";
    }
    return xbox;
}

uint64_t key_to_button(int32_t k) {
    switch (k) {
        // Android names the face buttons by position, as Xbox labels them: BUTTON_A is the bottom one, BUTTON_B the
        // right one. Console layout: bottom = Nintendo B, right = Nintendo A; Xbox layout: the other way round.
        case AKEYCODE_BUTTON_A: return xbox_layout() ? kA : kB;
        case AKEYCODE_BUTTON_B: return xbox_layout() ? kB : kA;
        case AKEYCODE_BUTTON_X: return kY;
        case AKEYCODE_BUTTON_Y: return kX;
        case AKEYCODE_BUTTON_L1: return kL;
        case AKEYCODE_BUTTON_R1: return kR;
        case AKEYCODE_BUTTON_L2: return kZL;
        case AKEYCODE_BUTTON_R2: return kZR;
        case AKEYCODE_BUTTON_START: return kPlus;
        case AKEYCODE_BUTTON_SELECT: return kMinus;
        case AKEYCODE_BUTTON_THUMBL: return kStickL;
        case AKEYCODE_BUTTON_THUMBR: return kStickR;
        case AKEYCODE_DPAD_LEFT: return kLeft;
        case AKEYCODE_DPAD_UP: return kUp;
        case AKEYCODE_DPAD_RIGHT: return kRight;
        case AKEYCODE_DPAD_DOWN: return kDown;
        case AKEYCODE_BACK: return kMinus;
        default: return 0;
    }
}

uint64_t touch_button(float x, float y) {
    float nx_ = x / g_w, ny = y / g_h;
    if (ny < 0.2f) return nx_ < 0.6f ? (nx_ < 0.45f ? kL : kPlus) : (nx_ > 0.85f ? kR : kZR);
    bool up = ny < 0.6f, right = nx_ > 0.8f;
    if (up) return right ? kX : kY;
    return right ? kA : kB;
}

}  // namespace

int32_t on_input(android_app*, AInputEvent* e) {
    std::lock_guard lk(g_in);
    int32_t type = AInputEvent_getType(e);
    int32_t src = AInputEvent_getSource(e);
    if (type == AINPUT_EVENT_TYPE_KEY) {
        // The layout changed (the settings screen) while A or B was held: its release now names the other button.
        static bool layout_was = xbox_layout();
        if (xbox_layout() != layout_was) layout_was = !layout_was, g_pad_buttons.buttons &= ~(kA | kB);
        uint64_t b = key_to_button(AKeyEvent_getKeyCode(e));
        if (!b) return 0;
        g_gamepad = true;
        if (AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_DOWN)
            g_pad_buttons.buttons |= b;
        else if (AKeyEvent_getAction(e) == AKEY_EVENT_ACTION_UP)
            g_pad_buttons.buttons &= ~b;
        return 1;
    }
    if (type != AINPUT_EVENT_TYPE_MOTION) return 0;
    if ((src & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK) {
        g_gamepad = true;
        auto axis = [&](int32_t a) {
            float v = AMotionEvent_getAxisValue(e, a, 0);
            return std::fabs(v) < 0.15f ? 0 : int32_t(std::clamp(v, -1.f, 1.f) * kStickMax);
        };
        g_pad_buttons.lx = axis(AMOTION_EVENT_AXIS_X);
        g_pad_buttons.ly = -axis(AMOTION_EVENT_AXIS_Y);
        g_pad_buttons.rx = axis(AMOTION_EVENT_AXIS_Z);
        g_pad_buttons.ry = -axis(AMOTION_EVENT_AXIS_RZ);
        float hx = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_X, 0);
        float hy = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_Y, 0);
        g_pad_buttons.buttons &= ~(kLeft | kRight | kUp | kDown | kZL | kZR);
        if (hx < -0.5f) g_pad_buttons.buttons |= kLeft;
        if (hx > 0.5f) g_pad_buttons.buttons |= kRight;
        if (hy < -0.5f) g_pad_buttons.buttons |= kUp;
        if (hy > 0.5f) g_pad_buttons.buttons |= kDown;
        if (AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_LTRIGGER, 0) > 0.4f ||
            AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_BRAKE, 0) > 0.4f)
            g_pad_buttons.buttons |= kZL;
        if (AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RTRIGGER, 0) > 0.4f ||
            AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_GAS, 0) > 0.4f)
            g_pad_buttons.buttons |= kZR;
        return 1;
    }
    if ((src & AINPUT_SOURCE_TOUCHSCREEN) != AINPUT_SOURCE_TOUCHSCREEN) return 0;
    {  // The settings screen, while open, takes the first finger as its mouse (whatever the on-screen controls).
        int32_t kind0 = AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_MASK;
        bool up = kind0 == AMOTION_EVENT_ACTION_UP || kind0 == AMOTION_EVENT_ACTION_CANCEL;
        if (nx::ui::ui_touch(AMotionEvent_getX(e, 0), AMotionEvent_getY(e, 0), !up)) {
            g_touches.clear();
            if (up) g_ui_taps.fetch_add(1, std::memory_order_relaxed);
            g_ui_finger.store(!up, std::memory_order_relaxed);
            return 1;
        }
    }
    if (!touch_enabled()) return 0;  // NX_TOUCH_SHOW=0, or turned off in the settings
    int32_t action = AMotionEvent_getAction(e);
    int32_t kind = action & AMOTION_EVENT_ACTION_MASK;
    size_t index = size_t((action & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT);
    if (ANativeWindow* w = android_window()) {
        g_w = float(ANativeWindow_getWidth(w));
        g_h = float(ANativeWindow_getHeight(w));
    }
    if (kind == AMOTION_EVENT_ACTION_DOWN || kind == AMOTION_EVENT_ACTION_POINTER_DOWN) {
        float x = AMotionEvent_getX(e, index), y = AMotionEvent_getY(e, index);
        g_gamepad = false;
        // The first finger of a gesture: no other is down. A touch left from one whose end was not seen here (the
        // controls turned off meanwhile) would otherwise keep its button held, or share the new finger's id.
        if (kind == AMOTION_EVENT_ACTION_DOWN) g_touches.clear();
        size_t drawn = 0;
        touch_buttons(drawn, g_w / g_h);
        if (drawn) {  // a drawn layout (touch_layout.cpp): its buttons first, the rest of the left half is the stick
            uint64_t b = touch_hit(x / g_w, y / g_h, g_w / g_h);
            bool left = !b && (touch_mirrored() ? x > g_w * 0.5f : x < g_w * 0.5f);
            const bool fr = !b && !left && native_touch();  // the free part of the screen: taps and the camera
            if (b || left || fr) {
                g_touches.push_back({AMotionEvent_getPointerId(e, index), x, y, x, y, left, b});
                Touch& nt = g_touches.back();
                nt.free = fr;
                nt.t0_ns = monotonic_ns();
                nt.sy = y;
                // Two fingers down together on the free part of the screen: a two-finger tap is B (back, cancel).
                if (fr)
                    for (auto& o : g_touches)
                        if (&o != &nt && o.free && !o.moved && nt.t0_ns - o.t0_ns < 250'000'000) o.pair = nt.pair = true;
            }
        } else {
            bool left = x < g_w * 0.5f;
            g_touches.push_back({AMotionEvent_getPointerId(e, index), x, y, x, y, left, left ? 0 : touch_button(x, y)});
        }
    } else if (kind == AMOTION_EVENT_ACTION_UP || kind == AMOTION_EVENT_ACTION_POINTER_UP ||
               kind == AMOTION_EVENT_ACTION_CANCEL) {
        int32_t id = AMotionEvent_getPointerId(e, index);
        // A short touch that stayed put (the free part of the screen, or a stick's finger that never pushed) is a tap:
        // on the phone icon it presses ZL (the stick not in use), anywhere else it clicks, as the mouse does on the
        // other hosts (hle_hid.cpp: guiTouchManager, the focus managers and the message window take it from there).
        if (kind != AMOTION_EVENT_ACTION_CANCEL && native_touch() && g_w > 1 && g_h > 1)
            for (const auto& t : g_touches) {
                if (t.id != id) continue;
                const uint64_t tn = monotonic_ns();
                const float dx = t.x - t.x0, dy = t.y - t.y0;
                if (t.free && t.pair && !t.moved && tn - t.t0_ns < 400'000'000) {  // two fingers tapped: B
                    if (tn - g_last_pair_ns > 400'000'000) pulse(kB), g_last_pair_ns = tn;
                    continue;
                }
                if (t.free && t.moved && !t.pair && config_get_bool("input.touch_gestures", true)) {
                    const bool menu = nx::ui_menu_open();
                    if (t.dir == 2 && std::fabs(dx) > g_h * 0.10f && tn - t.t0_ns < 700'000'000) {
                        // A flick sideways: in a menu the tabs (L, R), outside it the tools (d-pad left, right).
                        pulse(menu ? (dx > 0 ? kL : kR) : (dx > 0 ? kLeft : kRight));
                    } else if (t.dir == 1 && !menu && t.y0 > g_h * 0.88f && dy < -g_h * 0.12f) {
                        pulse(kX);  // up from the bottom edge: the pockets
                    }
                }
                if (t.id != id || t.moved || (!t.free && !t.left) || tn - t.t0_ns > 350'000'000) continue;
                bool stick_in_use = false;
                for (const auto& o : g_touches) stick_in_use |= o.left && o.id != id && o.moved;
                float u = t.x / g_w, v = t.y / g_h, x0, y0, x1, y1;
                if (std::getenv("NX_TRACE_TOUCH")) nx::log("touch: tap at %.3f, %.3f of the screen", u, v);
                if (g_calibrate_phone.load() && !nx::ui::ui_visible()) {  // Settings > Set the phone icon: this tap
                    g_calibrate_phone = false;
                    const float hw = 0.07f * g_h / g_w;
                    char rect[96];
                    std::snprintf(rect, sizeof rect, "%.4f,%.4f,%.4f,%.4f", std::max(0.f, u - hw), std::max(0.f, v - 0.07f),
                                  std::min(1.f, u + hw), std::min(1.f, v + 0.07f));
                    config_set("input.touch_phone_rect", rect);
                    nx::log("touch: the phone icon is at %s", rect);
                } else if (!stick_in_use && phone_rect(x0, y0, x1, y1) && u >= x0 && u <= x1 && v >= y0 && v <= y1) {
                    pulse(kZL);
                } else {
                    set_mouse_touch(true, u, v);  // down and up before the next frame: still one touch (g_mouse_tap)
                    set_mouse_touch(false, u, v);
                }
            }
        if (kind == AMOTION_EVENT_ACTION_UP || kind == AMOTION_EVENT_ACTION_CANCEL) g_touches.clear();
        g_touches.erase(std::remove_if(g_touches.begin(), g_touches.end(), [&](const Touch& t) { return t.id == id; }),
                        g_touches.end());
    } else if (kind == AMOTION_EVENT_ACTION_MOVE) {
        for (size_t k = 0; k < AMotionEvent_getPointerCount(e); k++) {
            int32_t id = AMotionEvent_getPointerId(e, k);
            for (auto& t : g_touches)
                if (t.id == id) {
                    float nx_ = AMotionEvent_getX(e, k), ny = AMotionEvent_getY(e, k);
                    if (t.free) {  // the speed is kept here, not taken by read_pad: the game reads the pad more than once a frame
                        const uint64_t tn = monotonic_ns();
                        const float dts = std::max(0.004f, float(tn - (t.move_ns ? t.move_ns : t.t0_ns)) * 1e-9f);
                        t.vx = t.vx * 0.5f + (nx_ - t.x) / dts * 0.5f;
                        t.vy = t.vy * 0.5f + (ny - t.y) / dts * 0.5f;
                        t.move_ns = tn;
                    }
                    t.x = nx_;
                    t.y = ny;
                    if (!t.moved && std::hypot(t.x - t.x0, t.y - t.y0) > g_h * 0.03f) {  // past a tap's slop
                        t.moved = true;
                        t.dir = std::fabs(t.x - t.x0) > std::fabs(t.y - t.y0) ? 2 : 1;
                        t.sy = t.y0;
                    }
                    // In a menu a vertical drag scrolls it: a step of the d-pad every 6% of the screen's height, the
                    // list following the finger (dragging up shows what is below).
                    if (t.free && t.moved && t.dir == 1 && !t.pair && nx::ui_menu_open() &&
                        config_get_bool("input.touch_gestures", true)) {
                        const float step = g_h * 0.06f;
                        while (std::fabs(t.y - t.sy) > step) {
                            pulse(t.y < t.sy ? kDown : kUp);
                            t.sy += t.y < t.sy ? -step : step;
                        }
                    }
                }
        }
    }
    return 1;
}

uint64_t ui_taps() { return g_ui_taps.load(std::memory_order_relaxed); }
bool ui_finger_down() { return g_ui_finger.load(std::memory_order_relaxed); }

bool headless() {  // NX_HEADLESS (platform.h)
    static const bool on = std::getenv("NX_HEADLESS") != nullptr;
    return on;
}

bool read_player_pad(int, PadState&) { return false; }  // player 1 only on this host

void read_pad(PadState& p) {
    std::lock_guard lk(g_in);
    p.buttons |= g_pad_buttons.buttons;
    if (g_pad_buttons.lx || g_pad_buttons.ly) {
        p.lx = g_pad_buttons.lx;
        p.ly = g_pad_buttons.ly;
    }
    p.rx = g_pad_buttons.rx;
    p.ry = g_pad_buttons.ry;
    float radius = std::max(1.f, g_h * 0.12f);
    size_t drawn = 0;
    touch_buttons(drawn, g_w / g_h);
    // A drawn layout's stick pushed past its ring also holds the run button (Wonder: Y, dash; ACNH: B, run), so running
    // needs no second thumb on the right side. In ACNH's menus B is also Cancel, and in its software keyboard Delete:
    // the stick pushed to its edge there (0.9 of the ring before, which any full push reached) backed out of menus and
    // deleted the name being typed, a letter each time the thumb crossed that line (Discord, 0.8 and 0.9). So it runs
    // only past the ring (kRunOn) and after a moment there (kRunDelayNs): a push across a menu or the keyboard is
    // over before that. It stops back inside the ring (kRunOff), not at the same line, so a thumb resting near it does
    // not press B again and again. Never while the game says its text entry is open (2.0.7: games/acnh/v2_0_7.cpp);
    // input.touch_run = 0 (Settings > Controls) turns it off.
    constexpr float kRunOn = 1.3f, kRunOff = 1.1f;
    constexpr uint64_t kRunDelayNs = 250'000'000;
    const uint64_t run = run_button();
    const bool can_run = drawn && config_get_bool("input.touch_run", true) &&
                         !(keyboard_state_known() && keyboard_typing());
    const uint64_t now = monotonic_ns();
    for (auto& t : g_touches) {
        if (t.left) {
            float dx = (t.x - t.x0) / radius, dy = (t.y - t.y0) / radius;
            p.lx = int32_t(std::clamp(dx, -1.f, 1.f) * kStickMax);
            p.ly = int32_t(std::clamp(-dy, -1.f, 1.f) * kStickMax);
            const float d = std::hypot(dx, dy);
            if (!can_run || (t.run ? d < kRunOff : d <= kRunOn)) {
                t.run = false;
                t.past_ns = 0;
            } else if (!t.run) {
                if (!t.past_ns) t.past_ns = now;
                t.run = now - t.past_ns >= kRunDelayNs;
            }
            if (t.run) p.buttons |= run;
        } else {
            p.buttons |= t.button;
        }
    }
    // Native touch: a finger dragged over the free part of the screen is the right stick (the camera), its speed the
    // push: its speed (smoothed over the moves, so it does not depend on the frame rate). Full push at a swipe of 1.5 screen heights a second (input.camera_sens, percent, scales that).
    float cam_x = 0, cam_y = 0;
    for (const auto& t : g_touches)
        if (t.free && t.moved && t.dir == 1 && !t.pair && now - t.move_ns < 90'000'000) cam_x += t.vx, cam_y += t.vy;
    if (nx::ui_menu_open()) cam_x = cam_y = 0;  // in a menu the same drag scrolls it (on_input)
    if ((cam_x != 0 || cam_y != 0) && native_touch() && config_get_bool("input.touch_camera", true)) {
        const float full = g_h * 1.5f / (float(std::clamp(config_get_int("input.camera_sens", 100), 20, 400)) / 100.0f);
        const float inv = config_get_bool("input.camera_invert", false) ? -1.f : 1.f;
        p.rx = int32_t(std::clamp(cam_x / full, -1.f, 1.f) * kStickMax);
        p.ry = int32_t(std::clamp(-cam_y * inv / full, -1.f, 1.f) * kStickMax);
        if (std::getenv("NX_TRACE_TOUCH")) nx::log("touch: camera %d, %d", p.rx, p.ry);
    }
    while (!g_pulses.empty() && now >= g_pulses.front().until) g_pulses.pop_front();
    for (const auto& pl : g_pulses)
        if (now >= pl.from) p.buttons |= pl.button;  // a gesture's button press (pulse)
    input_script(p);  // NX_INPUT_SCRIPT (config.txt): scripted runs on the phone, as on the Mac
    // The gear opens and closes the settings screen; it is not a pad button.
    static bool gear_held = false;
    bool gear = p.buttons & kSettingsButton;
    if (gear && !gear_held) nx::ui::ui_toggle();
    gear_held = gear;
    p.buttons &= ~kSettingsButton;
    // The settings screen takes the controller while it is open (and watches for its Minus + Plus combo).
    bool ui = nx::ui::ui_pad(p.buttons, p.lx, p.ly);
    p.buttons = nx::ui::ui_filter_game_buttons(p.buttons, ui);
    if (ui) p = PadState{};
}

void touch_calibrate_phone() { g_calibrate_phone = true; }

bool gamepad_connected() {
    std::lock_guard lk(g_in);
    return g_pad_connected;
}

// A controller connected: the touch controls hide at once (as on its first input); disconnected: they come back.
void set_gamepad_connected(bool connected) {
    std::lock_guard lk(g_in);
    if (connected == g_pad_connected) return;
    g_pad_connected = connected;
    g_gamepad = connected;
    nx::log("input: a controller %s: touch controls %s", connected ? "connected" : "disconnected",
            connected ? "hidden" : "shown");
}

// The drawn layout's buttons (brighter while pressed) and the stick under the thumb; nothing while a controller is
// in use or for games without a drawn layout.
size_t touch_overlay(OverlayShape* out, size_t max) {
    size_t n = 0, k = 0;
    std::lock_guard lk(g_in);
    if (ANativeWindow* w = android_window()) {  // the layout fits the screen's shape (touch_layout.cpp)
        g_w = float(ANativeWindow_getWidth(w));
        g_h = float(ANativeWindow_getHeight(w));
    }
    const TouchButton* b = touch_buttons(n, g_w / g_h);
    if (!n) return 0;
    if (g_gamepad || !touch_enabled()) return 0;
    const float shade = touch_opacity_scale();
    const Touch* stick = nullptr;
    uint64_t held = 0;
    for (const auto& t : g_touches) {
        if (t.left) stick = &t;
        held |= t.button | (t.run ? run_button() : 0);
    }
    // A button's letters ("A", "ZR", "+") in the glyphs present.frag draws: "ABXYRLZ+-"; nothing for the d-pad and gear.
    auto label_code = [](const char* t) {
        static const char kGlyphs[] = "ABXYRLZ+-";
        uint32_t code = 0;
        for (int c = 0; c < 2 && t[c]; c++) {
            const char* at = std::strchr(kGlyphs, t[c]);
            if (!at) return 0u;  // a letter there is no glyph for: none rather than half a label
            code |= uint32_t(at - kGlyphs + 1) << (8 * c);
        }
        return code;
    };
    for (size_t i = 0; i < n && k < max; i++)
        if (b[i].r > 0)  // a hidden button (touch_layout.cpp: minimal buttons) has no radius
            out[k++] = {b[i].x, b[i].y, b[i].r, std::min(1.0f, (held & b[i].button ? 0.5f : 0.2f) * shade), label_code(touch_label(i))};
    if (stick && k + 2 <= max) {
        float r = 0.12f, dx = (stick->x - stick->x0) / g_h, dy = (stick->y - stick->y0) / g_h;
        float len = std::hypot(dx, dy), s = len > r ? r / len : 1.f;
        // The ring lights up while the stick holds the run button (read_pad), as that button does.
        out[k++] = {stick->x0 / g_w, stick->y0 / g_h, r, std::min(1.0f, (stick->run ? 0.3f : 0.14f) * shade)};
        out[k++] = {(stick->x0 / g_h + dx * s) * g_h / g_w, stick->y0 / g_h + dy * s, 0.055f, std::min(1.0f, 0.45f * shade)};
    } else if (k < max) {
        float sx, sy, sr;
        touch_stick_rest(g_w / g_h, touch_mirrored(), sx, sy, sr);
        out[k++] = {sx, sy, sr, std::min(1.0f, 0.08f * shade)};  // where the stick goes
    }
    return k;
}

}  // namespace nx::plat
