from controller import Robot
import json
import math
import os

# Must match src/config.py Webots mapping: X_w = x * SCALE - ARENA_HALF, Z_w = y * SCALE - ARENA_HALF
TIME_STEP = 32
MAX_SPEED = 6.28
TARGET_TOLERANCE = 0.055
SCALE = 0.01
ARENA_HALF = 1.5

robot = Robot()
name = robot.getName()

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

gps = robot.getDevice("gps")
gps.enable(TIME_STEP)
compass = robot.getDevice("compass")
compass.enable(TIME_STEP)


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def paper_to_webots(pt):
    x, y = pt
    return (x * SCALE - ARENA_HALF, y * SCALE - ARENA_HALF)


def load_waypoints():
    controller_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(controller_dir, "../../../../results/webots_paths.json"))
    if not os.path.exists(path):
        return [paper_to_webots((150.0, 25.0))]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = data.get(name, [])
    if not pts:
        return [paper_to_webots((150.0, 25.0))]
    return [paper_to_webots(p) for p in pts]


waypoints = load_waypoints()
idx = 0

while robot.step(TIME_STEP) != -1:
    x, _, z = gps.getValues()
    tx, tz = waypoints[idx]

    dx = tx - x
    dz = tz - z
    dist = math.hypot(dx, dz)

    north = compass.getValues()
    heading = math.atan2(-north[0], -north[2])
    target_heading = math.atan2(dx, dz)
    error = normalize_angle(target_heading - heading)

    if dist < TARGET_TOLERANCE:
        if idx < len(waypoints) - 1:
            idx += 1
            continue
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        continue

    if abs(error) > 0.5:
        base = 1.6
    else:
        base = 3.0

    turn = max(min(2.8 * error, 2.0), -2.0)
    left_motor.setVelocity(max(min(base - turn, MAX_SPEED), -MAX_SPEED))
    right_motor.setVelocity(max(min(base + turn, MAX_SPEED), -MAX_SPEED))
