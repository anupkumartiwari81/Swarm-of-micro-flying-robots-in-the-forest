import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- PARAMETERS ----------------
num_drones = 5
num_obstacles = 15   # <<< UPDATED TO 15
speed = 0.05
goal = np.array([-8, 6, 8])   # goal position
circle_radius = 2.0
rotation_speed = 0.05
avoid_radius = 1.0       # drone-drone avoidance distance
obs_avoid_radius = 1.0   # obstacle avoidance distance

# ---------------- INITIAL SETUP ----------------
np.random.seed(42)

# random drone positions
drones = np.random.uniform(-6, 6, (num_drones, 3))
drones[:, 2] = np.random.uniform(1, 3, num_drones)

# random obstacle positions and velocities
obstacles = np.random.uniform(-7, 7, (num_obstacles, 3))
obstacles[:, 2] = np.random.uniform(1, 9, num_obstacles)
ob_vel = np.random.uniform(-0.03, 0.03, (num_obstacles, 3))

# rainbow colors for drones
colors = plt.cm.rainbow(np.linspace(0, 1, num_drones))

# Each drone has its own state: 0=go to goal, 1=circle, 2=orbit
states = np.zeros(num_drones, dtype=int)
rot_angles = np.zeros(num_drones)

# ---------------- PLOT ----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_zlim(0, 12)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.scatter(*goal, c="green", marker="*", s=200, label="Goal")
ax.view_init(25, 45)

drone_scatter = ax.scatter(drones[:,0], drones[:,1], drones[:,2], c=colors, s=60)
obstacle_scatter = ax.scatter(obstacles[:,0], obstacles[:,1], obstacles[:,2], c="blue", s=40)
plt.legend()

# ---------------- UPDATE FUNCTION ----------------
def update(frame):
    global drones, obstacles, ob_vel, states, rot_angles

    # --- Move obstacles (bounce on walls) ---
    obstacles[:] += ob_vel
    for i in range(num_obstacles):
        for axis, (lo, hi) in enumerate([(-10,10), (-10,10), (0,12)]):
            if obstacles[i,axis] < lo or obstacles[i,axis] > hi:
                ob_vel[i,axis] *= -1

    # --- Update each drone independently ---
    for i in range(num_drones):
        if states[i] == 0:  # go to goal
            to_goal = goal - drones[i]
            move = to_goal / (np.linalg.norm(to_goal)+1e-6) * speed

            # repulsion from other drones
            for j in range(num_drones):
                if i != j:
                    diff = drones[i] - drones[j]
                    dist = np.linalg.norm(diff)
                    if dist < avoid_radius:
                        move += diff/(dist+1e-6) * speed

            # repulsion from obstacles
            for obs in obstacles:
                diff = drones[i] - obs
                dist = np.linalg.norm(diff)
                if dist < obs_avoid_radius:
                    move += diff/(dist+1e-6) * speed

            drones[i] += move

            # switch to circle state if near goal
            if np.linalg.norm(to_goal) < 1.5:
                states[i] = 1

        elif states[i] == 1:  # circle formation
            vec = drones[i] - goal
            vec[2] = 0
            if np.linalg.norm(vec) > 1e-6:
                target = goal + vec/np.linalg.norm(vec) * circle_radius
                target[2] = goal[2]
                diff = target - drones[i]
                drones[i] += diff/np.linalg.norm(diff) * speed

            # repulsion to spread evenly
            for j in range(num_drones):
                if i != j:
                    diff = drones[i] - drones[j]
                    dist = np.linalg.norm(diff)
                    if dist < avoid_radius*2:
                        drones[i] += diff/(dist+1e-6) * speed

            # check if settled into circle → start orbit
            vec2 = drones[i] - goal
            vec2[2] = 0
            if abs(np.linalg.norm(vec2) - circle_radius) < 0.2:
                states[i] = 2

        elif states[i] == 2:  # orbit around goal
            rot_angles[i] += rotation_speed
            vec = drones[i] - goal
            vec[2] = 0
            r = np.linalg.norm(vec)
            if r < 1e-6:
                r = circle_radius
            angle = np.arctan2(vec[1], vec[0]) + rotation_speed
            drones[i,0] = goal[0] + r*np.cos(angle)
            drones[i,1] = goal[1] + r*np.sin(angle)
            drones[i,2] = goal[2]

    # --- Update plot ---
    drone_scatter._offsets3d = (drones[:,0], drones[:,1], drones[:,2])
    obstacle_scatter._offsets3d = (obstacles[:,0], obstacles[:,1], obstacles[:,2])
    return drone_scatter, obstacle_scatter

# ---------------- RUN ANIMATION ----------------
ani = FuncAnimation(fig, update, frames=600, interval=50, blit=False)
plt.show()
