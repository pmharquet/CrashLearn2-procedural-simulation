"""Pure numerical kernels: F110 integration, LiDAR and collision geometry."""

import numpy as np
from numba import njit
from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid

# Exact angular convention used by submission/champion/agent.py (ANGLES).
LIDAR_ANGLES = np.linspace(-np.pi, np.pi, 100)

PARAMS = (
    1.0,
    4.718,
    5.4562,
    0.15875,
    0.17145,
    0.074,
    3.74,
    0.04712,
    -0.4189,
    0.4189,
    -3.2,
    3.2,
    7.319,
    9.51,
    -5.0,
    20.0,
)


@njit(cache=True)
def advance(state, speed, steer, mu=1.0):
    acc, sv = pid(speed, steer, state[3], state[2], 3.2, 9.51, 20.0, -5.0)
    u = np.array([sv, acc])
    parameters = (mu,) + PARAMS[1:]
    k1 = vehicle_dynamics_st(state, u, *parameters)
    k2 = vehicle_dynamics_st(state + 0.005 * k1, u, *parameters)
    k3 = vehicle_dynamics_st(state + 0.005 * k2, u, *parameters)
    k4 = vehicle_dynamics_st(state + 0.01 * k3, u, *parameters)
    return state + 0.01 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@njit(cache=True)
def scan_walls(pos, yaw, left, right):
    scan = np.full(100, 15.0, dtype=np.float32)
    # Cull rails outside sensor range once, rather than for each of 100 rays.
    segments = np.empty((2 * (len(left) - 1), 4))
    count = 0
    for wall in (left, right):
        for j in range(len(wall) - 1):
            ax, ay = wall[j, 0] - pos[0], wall[j, 1] - pos[1]
            bx, by = wall[j + 1, 0] - pos[0], wall[j + 1, 1] - pos[1]
            if min(ax, bx) > 15 or max(ax, bx) < -15 or min(ay, by) > 15 or max(ay, by) < -15:
                continue
            segments[count] = (ax, ay, bx - ax, by - ay)
            count += 1
    for r in range(100):
        angle = yaw + LIDAR_ANGLES[r]
        dx, dy = np.cos(angle), np.sin(angle)
        for j in range(count):
            ax, ay, ex, ey = segments[j]
            den = dx * ey - dy * ex
            if abs(den) < 1e-10:
                continue
            t, u = (ax * ey - ay * ex) / den, (ax * dy - ay * dx) / den
            if t >= 0 and 0 <= u <= 1 and t < scan[r]:
                scan[r] = max(0.1, t)
    return scan


@njit(cache=True)
def scan_vehicles(pos, yaw, scan, states, status, agent_id):
    """Clip wall ranges against active opponents' collision rectangles.

    Hit IDs: -1 = no vehicle (wall or range limit), otherwise car slot.
    """
    hits = np.full(100, -1, dtype=np.int32)
    for j in range(len(states)):
        if j == agent_id or status[j] != 1:
            continue
        c, s = np.cos(states[j, 4]), np.sin(states[j, 4])
        delta = pos - states[j, :2]
        ox, oy = c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]
        for r in range(100):
            angle = yaw + LIDAR_ANGLES[r] - states[j, 4]
            dx, dy = np.cos(angle), np.sin(angle)
            enter, leave = 0.0, 15.0
            for origin, direction, half in ((ox, dx, 0.29), (oy, dy, 0.155)):
                if abs(direction) < 1e-10:
                    if abs(origin) > half:
                        leave = -1.0
                else:
                    a, b = (-half - origin) / direction, (half - origin) / direction
                    enter = max(enter, min(a, b))
                    leave = min(leave, max(a, b))
            if enter <= leave and enter < scan[r]:
                scan[r] = max(0.1, enter)
                hits[r] = j
    return hits


@njit(cache=True)
def project_center(pos, center):
    best = 1e30
    progress = 0.0
    for i in range(len(center) - 1):
        vx, vy = center[i + 1, 0] - center[i, 0], center[i + 1, 1] - center[i, 1]
        dx, dy = pos[0] - center[i, 0], pos[1] - center[i, 1]
        t = min(1.0, max(0.0, (dx * vx + dy * vy) / (vx * vx + vy * vy)))
        distance = (dx - t * vx) ** 2 + (dy - t * vy) ** 2
        if distance < best:
            best = distance
            progress = center[i, 3] + t * (center[i + 1, 3] - center[i, 3])
    return progress, np.sqrt(best)


@njit(cache=True)
def footprint_clear(state, center, width):
    c, s = np.cos(state[4]), np.sin(state[4])
    for x in (-0.29, 0.0, 0.29):
        for y in (-0.155, 0.155):
            _, distance = project_center(
                np.array([state[0] + c * x - s * y, state[1] + s * x + c * y]), center
            )
            if distance >= width / 2:
                return False
    return True


@njit(cache=True)
def contact_vector(a, b):
    """Rectangle SAT: minimal translation from b to a, or zero if disjoint."""
    delta = a[:2] - b[:2]
    ca, sa, cb, sb = np.cos(a[4]), np.sin(a[4]), np.cos(b[4]), np.sin(b[4])
    axes = np.array([[ca, sa], [-sa, ca], [cb, sb], [-sb, cb]])
    smallest = 1e20
    normal = np.zeros(2)
    for axis in axes:
        ra = 0.29 * abs(axis[0] * ca + axis[1] * sa) + 0.155 * abs(-axis[0] * sa + axis[1] * ca)
        rb = 0.29 * abs(axis[0] * cb + axis[1] * sb) + 0.155 * abs(-axis[0] * sb + axis[1] * cb)
        distance = delta[0] * axis[0] + delta[1] * axis[1]
        overlap = ra + rb - abs(distance)
        if overlap <= 0:
            return np.zeros(2)
        if overlap < smallest:
            smallest = overlap
            normal = axis * (1.0 if distance >= 0 else -1.0)
    return normal * (smallest + 1e-5)
