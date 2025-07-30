"""
Drone Swarm Simulation
----------------------
3D simulation of autonomous drones inspired by the paper:
"Swarm of Micro Flying Robots in the Wild" (Science Robotics, 2022).

Features:
- Dynamic trajectory replanning with cubic splines (MINCO-like)
- Local obstacle avoidance and inter-drone collision avoidance
- Orbiting behavior around goal after reaching it
- Unique drone colors and trails
- Rotating 3D camera for cinematic visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

# ---------------------------
# Simulation Parameters
# ---------------------------
NUM_DRONES = 7
DT = 0.1                       # Time step
MAX_SPEED = 1.5                # Max drone speed
SENSING_RADIUS = 8.0           # Local obstacle detection radius
REPLAN_INTERVAL = 50           # Frames between replanning
GOAL = np.array([45, 45, 20])   # Goal position
NUM_OBSTACLES = 10
SPACE_SIZE = 50
TRAIL_LENGTH = 50              # Number of points to keep for trails
CAMERA_ROTATION_SPEED = 0.5    # Degrees per frame
GOAL_RADIUS = 3.0              # Distance to consider goal reached
CIRCLE_RADIUS = 5.0            # Radius for orbiting around goal

# ---------------------------
# Initialization
# ---------------------------
np.random.seed(42)  # For reproducibility
positions = np.random.rand(NUM_DRONES, 3) * 5
velocities = np.zeros((NUM_DRONES, 3))

# Static obstacles
obstacles = np.random.rand(NUM_OBSTACLES, 3) * SPACE_SIZE
dynamic_obstacles = []

# Trajectories (splines)
splines = [None] * NUM_DRONES
times = np.linspace(0, 1, 20)

# Trails and colors
trails = [[] for _ in range(NUM_DRONES)]
colors = plt.cm.rainbow(np.linspace(0, 1, NUM_DRONES))

# State variables
circle_mode = False
circle_angle = 0.0  # For orbiting

# ---------------------------
# Functions
# ---------------------------

def generate_spline(start, goal):
    """Generate cubic spline trajectory from start to goal with a random midpoint."""
    mid = (start + goal) / 2 + np.random.randn(3) * 2.0
    waypoints = np.vstack([start, mid, goal])
    t_points = [0, 0.5, 1]
    return (
        CubicSpline(t_points, waypoints[:, 0]),
        CubicSpline(t_points, waypoints[:, 1]),
        CubicSpline(t_points, waypoints[:, 2])
    )

def sample_spline(spline, t):
    """Sample spline at parameter t (0-1)."""
    sx, sy, sz = spline
    return np.array([sx(t), sy(t), sz(t)])

def detect_local_obstacles(pos):
    """Return obstacles within sensing radius."""
    all_obs = np.vstack([obstacles] + [dynamic_obstacles] if dynamic_obstacles else [obstacles])
    dists = np.linalg.norm(all_obs - pos, axis=1)
    return all_obs[dists < SENSING_RADIUS]

def avoidance_force(drone_idx):
    """Calculate avoidance force for drone against other drones and obstacles."""
    force = np.zeros(3)

    # Avoid other drones
    for j, pos in enumerate(positions):
        if j == drone_idx:
            continue
        vec = positions[drone_idx] - pos
        dist = np.linalg.norm(vec)
        if dist < 3.0:
            force += vec / (dist**2 + 1e-6) * 2.0

    # Avoid obstacles
    for obs in detect_local_obstacles(positions[drone_idx]):
        vec = positions[drone_idx] - obs
        dist = np.linalg.norm(vec)
        if dist < 5.0:
            force += vec / (dist**2 + 1e-6) * 4.0

    return force

def replan_trajectories():
    """Recompute splines for all drones toward the goal."""
    global splines
    for i in range(NUM_DRONES):
        splines[i] = generate_spline(positions[i], GOAL)

# ---------------------------
# Update Loop
# ---------------------------
frame_count = 0

def update_positions():
    """Update drone positions for each animation frame."""
    global positions, velocities, frame_count, dynamic_obstacles, circle_mode, circle_angle

    # Check if goal reached (leader triggers circle mode)
    if not circle_mode and np.linalg.norm(positions[0] - GOAL) < GOAL_RADIUS:
        circle_mode = True

    # Replan and add obstacles if not yet orbiting
    if frame_count % REPLAN_INTERVAL == 0 and frame_count > 0 and not circle_mode:
        replan_trajectories()
        new_obs = np.random.rand(3, 3) * SPACE_SIZE
        dynamic_obstacles.extend(new_obs)

    if circle_mode:
        # Orbiting behavior
        circle_angle += 0.05  # radians per frame
        for i in range(NUM_DRONES):
            angle_offset = (2 * np.pi / NUM_DRONES) * i
            angle = circle_angle + angle_offset
            positions[i] = np.array([
                GOAL[0] + CIRCLE_RADIUS * np.cos(angle),
                GOAL[1] + CIRCLE_RADIUS * np.sin(angle),
                GOAL[2]
            ])
            # Add to trail
            trails[i].append(positions[i].copy())
            if len(trails[i]) > TRAIL_LENGTH:
                trails[i].pop(0)
    else:
        # Normal trajectory following
        for i in range(NUM_DRONES):
            t = (frame_count % len(times)) / len(times)
            target = sample_spline(splines[i], t)
            vel = target - positions[i]
            vel += avoidance_force(i)
            speed = np.linalg.norm(vel)
            if speed > MAX_SPEED:
                vel = vel / speed * MAX_SPEED
            positions[i] += vel * DT

            trails[i].append(positions[i].copy())
            if len(trails[i]) > TRAIL_LENGTH:
                trails[i].pop(0)

    frame_count += 1

# ---------------------------
# Visualization
# ---------------------------
def run_simulation():
    """Run the matplotlib animation."""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, SPACE_SIZE)
    ax.set_ylim(0, SPACE_SIZE)
    ax.set_zlim(0, SPACE_SIZE / 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)

    # Obstacles
    obstacle_scatter = ax.scatter(obstacles[:,0], obstacles[:,1], obstacles[:,2], c='red', s=50)
    # Goal
    ax.scatter(GOAL[0], GOAL[1], GOAL[2], c='green', s=200, marker='*')

    # Drones
    drone_scatter = ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=colors, s=100)

    # Trails
    trail_lines = [ax.plot([], [], [], lw=2, c=colors[i])[0] for i in range(NUM_DRONES)]

    # Labels
    labels = [ax.text(positions[i,0], positions[i,1], positions[i,2]+1.0,
                      f"D{i+1}", color=colors[i], fontsize=10, weight='bold')
              for i in range(NUM_DRONES)]

    def init():
        drone_scatter._offsets3d = (positions[:,0], positions[:,1], positions[:,2])
        return [drone_scatter] + trail_lines + labels

    def animate(frame):
        update_positions()

        # Update drones
        drone_scatter._offsets3d = (positions[:,0], positions[:,1], positions[:,2])

        # Update trails
        for i, line in enumerate(trail_lines):
            if len(trails[i]) > 1:
                trail_arr = np.array(trails[i])
                line.set_data(trail_arr[:,0], trail_arr[:,1])
                line.set_3d_properties(trail_arr[:,2])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        # Update labels
        for i, label in enumerate(labels):
            label.set_x(positions[i,0])
            label.set_y(positions[i,1])
            label.set_3d_properties(positions[i,2] + 1.0)

        # Update dynamic obstacles
        if dynamic_obstacles:
            obs_all = np.vstack([obstacles] + [dynamic_obstacles])
        else:
            obs_all = obstacles
        obstacle_scatter._offsets3d = (obs_all[:,0], obs_all[:,1], obs_all[:,2])

        # Rotate camera
        azim_angle = (frame * CAMERA_ROTATION_SPEED) % 360
        ax.view_init(elev=30, azim=azim_angle)

        return [drone_scatter] + trail_lines + labels

    ani = FuncAnimation(fig, animate, init_func=init, frames=800, interval=50, blit=False)
    plt.show()

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    replan_trajectories()
    run_simulation()
