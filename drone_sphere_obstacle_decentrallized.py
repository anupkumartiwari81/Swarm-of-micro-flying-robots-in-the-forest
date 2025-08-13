import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from matplotlib.animation import FuncAnimation

# === Parameters ===
num_drones = 5
num_obstacles = 10
drone_speed = 0.06                     # speed per frame
goal = np.array([-6.0, 6.0, 8.0])      # UPDATED Goal position
obstacle_radius = 0.3                  # obstacle/drone "size"
formation_spacing = 1.5                # spacing in the straight line
circle_radius = 2.0                    # radius of the circle around the goal
rotation_speed = 0.05                  # radians per frame (Phase 3)
arrival_threshold = 0.6                # threshold to say "we reached the goal area"
circle_snap_threshold = 0.07           # closeness to slot to consider "snapped" (Phase 2)
space_limits = [-8, 8, -8, 8, 0.5, 10] # expanded bounds to fit new goal

# === Initialization ===
np.random.seed(0)

# Initial scattered positions for drones
drone_positions = np.random.uniform(-5, 5, (num_drones, 3))
drone_positions[:, 2] = np.random.uniform(0.5, 2, num_drones)

# Random initial positions and velocities for obstacles
obstacle_positions = np.random.uniform([space_limits[0], space_limits[2], space_limits[4]],
                                       [space_limits[1], space_limits[3], space_limits[5]],
                                       (num_obstacles, 3))
obstacle_velocities = np.random.uniform(-0.03, 0.03, (num_obstacles, 3))

# Colors for drones (rainbow)
drone_colors = plt.cm.rainbow(np.linspace(0, 1, num_drones))

# === Plot setup ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(space_limits[0], space_limits[1])
ax.set_ylim(space_limits[2], space_limits[3])
ax.set_zlim(space_limits[4], space_limits[5])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

# Make geometry look correct so the circle is round
try:
    ax.set_box_aspect((1, 1, 1))
except Exception:
    pass
ax.view_init(elev=25, azim=45)

# Plot goal
ax.scatter(*goal, c='green', marker='*', s=220, label="Goal")

# Drone and obstacle scatters
drone_scatter = ax.scatter(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2],
                           c=drone_colors, s=60, depthshade=False)
obstacle_scatter = ax.scatter(obstacle_positions[:, 0], obstacle_positions[:, 1], obstacle_positions[:, 2],
                              c='blue', s=50, depthshade=True)

# === Phase control ===
# 0 = form line, 1 = move line to goal, 2 = capture circle slots (static), 3 = rotate circle
phase = 0
rotation_angle = 0.0
circle_slot_angles = None  # assigned when entering Phase 2

def safe_unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n

def clamp_to_bounds(pos):
    for axis in range(3):
        lo = space_limits[axis*2]
        hi = space_limits[axis*2 + 1]
        if pos[axis] < lo: pos[axis] = lo
        if pos[axis] > hi: pos[axis] = hi
    return pos

def update(frame):
    global drone_positions, obstacle_positions, obstacle_velocities
    global phase, rotation_angle, circle_slot_angles

    # === Move obstacles randomly and bounce at boundaries ===
    obstacle_positions += obstacle_velocities
    for i in range(num_obstacles):
        for axis in range(3):
            lo = space_limits[axis*2]
            hi = space_limits[axis*2 + 1]
            if obstacle_positions[i, axis] < lo:
                obstacle_positions[i, axis] = lo
                obstacle_velocities[i, axis] *= -1
            elif obstacle_positions[i, axis] > hi:
                obstacle_positions[i, axis] = hi
                obstacle_velocities[i, axis] *= -1

    # === PHASE 0: Align drones in a line perpendicular to goal direction ===
    if phase == 0:
        centroid = np.mean(drone_positions, axis=0)
        to_goal = goal - centroid
        gdir = safe_unit(to_goal)

        # Perpendicular in XY plane; fall back to x-axis if degenerate
        perp_xy = np.array([-gdir[1], gdir[0], 0.0])
        if np.linalg.norm(perp_xy[:2]) < 1e-6:
            perp_xy = np.array([1.0, 0.0, 0.0])
        perp_xy = safe_unit(perp_xy)

        # Line target positions (staggered heights for visibility)
        target_positions = np.array([
            centroid + perp_xy * (i - (num_drones - 1) / 2) * formation_spacing
            for i in range(num_drones)
        ])
        for i in range(num_drones):
            target_positions[i, 2] = 1.5 + (i - num_drones // 2) * 0.22

        reached = True
        for i in range(num_drones):
            delta = target_positions[i] - drone_positions[i]
            d = np.linalg.norm(delta)
            if d > 0.05:
                drone_positions[i] += safe_unit(delta) * drone_speed
                drone_positions[i] = clamp_to_bounds(drone_positions[i])
                reached = False

        if reached:
            phase = 1  # Proceed to approach goal

    # === PHASE 1: Move the line toward the goal with simple obstacle avoidance ===
    elif phase == 1:
        line_center = np.mean(drone_positions, axis=0)
        to_goal = goal - line_center
        dist_to_goal = np.linalg.norm(to_goal)

        if dist_to_goal > arrival_threshold:
            dir_to_goal = safe_unit(to_goal)

            # Keep drones roughly in their line formation by using same drive direction
            for i in range(num_drones):
                desired = drone_positions[i] + dir_to_goal * drone_speed

                # Mild obstacle repulsion
                for obs in obstacle_positions:
                    vec = obs - drone_positions[i]
                    d = np.linalg.norm(vec)
                    if d < (obstacle_radius + 0.8):
                        desired += safe_unit(drone_positions[i] - obs) * drone_speed

                # Flatten the formation height while approaching
                desired[2] = goal[2]
                drone_positions[i] = clamp_to_bounds(desired)
        else:
            # Enter circle capture: assign fixed, evenly spaced slot angles
            circle_slot_angles = np.array([2 * np.pi * i / num_drones for i in range(num_drones)])
            phase = 2

    # === PHASE 2: Move each drone to its circle slot (static circle) ===
    elif phase == 2:
        all_snapped = True
        for i in range(num_drones):
            theta = circle_slot_angles[i]
            target = np.array([
                goal[0] + circle_radius * np.cos(theta),
                goal[1] + circle_radius * np.sin(theta),
                goal[2]
            ])
            delta = target - drone_positions[i]
            d = np.linalg.norm(delta)
            if d > circle_snap_threshold:
                drone_positions[i] += safe_unit(delta) * drone_speed
                all_snapped = False
            else:
                drone_positions[i] = target  # snap cleanly when close

        if all_snapped:
            phase = 3  # begin rotation

    # === PHASE 3: Rotate the circle around the goal ===
    elif phase == 3:
        rotation_angle += rotation_speed
        for i in range(num_drones):
            theta = rotation_angle + circle_slot_angles[i]
            drone_positions[i, 0] = goal[0] + circle_radius * np.cos(theta)
            drone_positions[i, 1] = goal[1] + circle_radius * np.sin(theta)
            drone_positions[i, 2] = goal[2]

    # === Update scatters ===
    drone_scatter._offsets3d = (
        drone_positions[:, 0],
        drone_positions[:, 1],
        drone_positions[:, 2]
    )
    obstacle_scatter._offsets3d = (
        obstacle_positions[:, 0],
        obstacle_positions[:, 1],
        obstacle_positions[:, 2]
    )
    return drone_scatter, obstacle_scatter

# === Animation ===
ani = FuncAnimation(fig, update, frames=900, interval=50, blit=False)
plt.legend()
plt.show()
