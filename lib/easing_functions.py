"""
Easing functions for animation curves.
All functions take a float t in [0,1] and return a float in [0,1].
Based on https://easings.net/
"""

import math


# --- Core easing functions ---

def linear(t):
    return t


def ease_in_quad(t):
    return t * t


def ease_out_quad(t):
    return 1.0 - (1.0 - t) ** 2


def ease_in_out_quad(t):
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def ease_in_cubic(t):
    return t ** 3


def ease_out_cubic(t):
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t):
    if t < 0.5:
        return 4.0 * t ** 3
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def ease_in_quart(t):
    return t ** 4


def ease_out_quart(t):
    return 1.0 - (1.0 - t) ** 4


def ease_in_out_quart(t):
    if t < 0.5:
        return 8.0 * t ** 4
    return 1.0 - (-2.0 * t + 2.0) ** 4 / 2.0


def ease_in_quint(t):
    return t ** 5


def ease_out_quint(t):
    return 1.0 - (1.0 - t) ** 5


def ease_in_out_quint(t):
    if t < 0.5:
        return 16.0 * t ** 5
    return 1.0 - (-2.0 * t + 2.0) ** 5 / 2.0


def ease_in_sine(t):
    return 1.0 - math.cos(t * math.pi / 2.0)


def ease_out_sine(t):
    return math.sin(t * math.pi / 2.0)


def ease_in_out_sine(t):
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def ease_in_expo(t):
    if t == 0.0:
        return 0.0
    return 2.0 ** (10.0 * t - 10.0)


def ease_out_expo(t):
    if t == 1.0:
        return 1.0
    return 1.0 - 2.0 ** (-10.0 * t)


def ease_in_out_expo(t):
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    if t < 0.5:
        return 2.0 ** (20.0 * t - 10.0) / 2.0
    return (2.0 - 2.0 ** (-20.0 * t + 10.0)) / 2.0


def ease_in_circ(t):
    return 1.0 - math.sqrt(1.0 - t * t)


def ease_out_circ(t):
    return math.sqrt(1.0 - (t - 1.0) ** 2)


def ease_in_out_circ(t):
    if t < 0.5:
        return (1.0 - math.sqrt(1.0 - (2.0 * t) ** 2)) / 2.0
    return (math.sqrt(1.0 - (-2.0 * t + 2.0) ** 2) + 1.0) / 2.0


def ease_in_back(t):
    c1 = 1.70158
    c3 = c1 + 1.0
    return c3 * t ** 3 - c1 * t ** 2


def ease_out_back(t):
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2


def ease_in_out_back(t):
    c1 = 1.70158
    c2 = c1 * 1.525
    if t < 0.5:
        return ((2.0 * t) ** 2 * ((c2 + 1.0) * 2.0 * t - c2)) / 2.0
    return ((2.0 * t - 2.0) ** 2 * ((c2 + 1.0) * (t * 2.0 - 2.0) + c2) + 2.0) / 2.0


def ease_in_elastic(t):
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    c4 = (2.0 * math.pi) / 3.0
    return -(2.0 ** (10.0 * t - 10.0)) * math.sin((t * 10.0 - 10.75) * c4)


def ease_out_elastic(t):
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    c4 = (2.0 * math.pi) / 3.0
    return 2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * c4) + 1.0


def ease_in_out_elastic(t):
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    c5 = (2.0 * math.pi) / 4.5
    if t < 0.5:
        return -(2.0 ** (20.0 * t - 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0
    return (2.0 ** (-20.0 * t + 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0 + 1.0


def ease_in_bounce(t):
    return 1.0 - ease_out_bounce(1.0 - t)


def ease_out_bounce(t):
    n1 = 7.5625
    d1 = 2.75
    if t < 1.0 / d1:
        return n1 * t * t
    elif t < 2.0 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


def ease_in_out_bounce(t):
    if t < 0.5:
        return (1.0 - ease_out_bounce(1.0 - 2.0 * t)) / 2.0
    return (1.0 + ease_out_bounce(2.0 * t - 1.0)) / 2.0


# --- Cubic Bezier solver (Newton-Raphson) ---

def create_bezier_easing(p1x, p1y, p2x, p2y):
    """
    Create an easing function from cubic bezier control points.
    The curve goes from (0,0) to (1,1) with control points (p1x,p1y) and (p2x,p2y).
    Uses Newton-Raphson iteration to solve for t given x, then evaluates y(t).
    """
    def _bezier_component(t, p1, p2):
        """Evaluate one component of the cubic bezier at parameter t."""
        return 3.0 * (1.0 - t) ** 2 * t * p1 + 3.0 * (1.0 - t) * t ** 2 * p2 + t ** 3

    def _bezier_derivative(t, p1, p2):
        """Derivative of bezier component with respect to t."""
        return 3.0 * (1.0 - t) ** 2 * p1 + 6.0 * (1.0 - t) * t * (p2 - p1) + 3.0 * t ** 2 * (1.0 - p2)

    def easing(x):
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0

        # Newton-Raphson to find t for given x
        t = x  # initial guess
        for _ in range(8):
            x_at_t = _bezier_component(t, p1x, p2x) - x
            dx = _bezier_derivative(t, p1x, p2x)
            if abs(dx) < 1e-7:
                break
            t -= x_at_t / dx
            t = max(0.0, min(1.0, t))

        # If Newton didn't converge well, fall back to bisection
        if abs(_bezier_component(t, p1x, p2x) - x) > 1e-4:
            lo, hi = 0.0, 1.0
            for _ in range(20):
                mid = (lo + hi) / 2.0
                if _bezier_component(mid, p1x, p2x) < x:
                    lo = mid
                else:
                    hi = mid
            t = (lo + hi) / 2.0

        return _bezier_component(t, p1y, p2y)

    return easing


# --- Name mapping ---

EASING_NAME_MAP = {
    "linear": linear,
    "easeInQuad": ease_in_quad,
    "easeOutQuad": ease_out_quad,
    "easeInOutQuad": ease_in_out_quad,
    "easeInCubic": ease_in_cubic,
    "easeOutCubic": ease_out_cubic,
    "easeInOutCubic": ease_in_out_cubic,
    "easeInQuart": ease_in_quart,
    "easeOutQuart": ease_out_quart,
    "easeInOutQuart": ease_in_out_quart,
    "easeInQuint": ease_in_quint,
    "easeOutQuint": ease_out_quint,
    "easeInOutQuint": ease_in_out_quint,
    "easeInSine": ease_in_sine,
    "easeOutSine": ease_out_sine,
    "easeInOutSine": ease_in_out_sine,
    "easeInExpo": ease_in_expo,
    "easeOutExpo": ease_out_expo,
    "easeInOutExpo": ease_in_out_expo,
    "easeInCirc": ease_in_circ,
    "easeOutCirc": ease_out_circ,
    "easeInOutCirc": ease_in_out_circ,
    "easeInBack": ease_in_back,
    "easeOutBack": ease_out_back,
    "easeInOutBack": ease_in_out_back,
    "easeInElastic": ease_in_elastic,
    "easeOutElastic": ease_out_elastic,
    "easeInOutElastic": ease_in_out_elastic,
    "easeInBounce": ease_in_bounce,
    "easeOutBounce": ease_out_bounce,
    "easeInOutBounce": ease_in_out_bounce,
}


def get_easing_function(name):
    """Get an easing function by its ComfyUI COMBO name."""
    return EASING_NAME_MAP.get(name)


def get_all_easing_names():
    """Return list of all easing preset names (for COMBO widget)."""
    return list(EASING_NAME_MAP.keys())
