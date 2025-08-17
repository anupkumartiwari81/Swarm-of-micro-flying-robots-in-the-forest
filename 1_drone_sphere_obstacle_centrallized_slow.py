import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- PARAMETERS ----------------
num_drones = 5
num_obstacles = 20
speed = 0.05
goal = np.array([-6, 6, 8])
circle_radius = 2.0
rotation_speed = 0.05

# ---------------- INITIAL SETUP ----------------
np.random.seed(42)

# random drone positions
drones = np.random.uniform(-5, 5, (num_drones, 3))
drones[:, 2] = np.random.uniform(1, 2, num_drones)

# random obstacle positions and velocities
obstacles = np.random.uniform(-7, 7, (num_obstacles, 3))
obstacles[:, 2] = np.random.uniform(1, 9, num_obstacles)
ob_vel = np.random.uniform(-0.03, 0.03, (num_obstacles, 3))

# rainbow colors for drones
colors = plt.cm.rainbow(np.linspace(0, 1, num_drones))

# ---------------- PLOT ----------------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_zlim(0, 10)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.scatter(*goal, c="green", marker="*", s=200, label="Goal")
ax.view_init(25, 45)

drone_scatter = ax.scatter(drones[:,0], drones[:,1], drones[:,2], c=colors, s=60)
obstacle_scatter = ax.scatter(obstacles[:,0], obstacles[:,1], obstacles[:,2], c="blue", s=40)
plt.legend()

# ---------------- PHASE CONTROL ----------------
phase = 0     # 0 = line, 1 = move line to goal, 2 = circle, 3 = rotate
rot_angle = 0
circle_slots = [2*np.pi*i/num_drones for i in range(num_drones)]

# ---------------- UPDATE FUNCTION ----------------
def update(frame):
    global drones, obstacles, ob_vel, phase, rot_angle

    # --- Move obstacles (bounce on walls) ---
    obstacles[:] += ob_vel
    for i in range(num_obstacles):
        for axis, (lo, hi) in enumerate([(-8,8), (-8,8), (0,10)]):
            if obstacles[i,axis] < lo or obstacles[i,axis] > hi:
                ob_vel[i,axis] *= -1

    # --- Phase 0: each drone aligns into line formation ---
    if phase == 0:
        x_line = np.linspace(-2, 2, num_drones)
        targets = np.array([[x, 0, 2+i*0.3] for i,x in enumerate(x_line)])
        close = True
        for i in range(num_drones):
            diff = targets[i] - drones[i]
            if np.linalg.norm(diff) > 0.1:
                drones[i] += diff/np.linalg.norm(diff) * speed
                close = False
        if close: phase = 1

    # --- Phase 1: move line formation toward goal ---
    elif phase == 1:
        center = np.mean(drones, axis=0)
        to_goal = goal - center
        if np.linalg.norm(to_goal) > 0.5:
            move = to_goal/np.linalg.norm(to_goal) * speed
            drones += move
        else:
            phase = 2

    # --- Phase 2: snap into circle around goal ---
    elif phase == 2:
        all_ready = True
        for i,theta in enumerate(circle_slots):
            target = goal + [circle_radius*np.cos(theta), circle_radius*np.sin(theta), 0]
            target[2] = goal[2]
            diff = target - drones[i]
            if np.linalg.norm(diff) > 0.1:
                drones[i] += diff/np.linalg.norm(diff)*speed
                all_ready = False
            else:
                drones[i] = target
        if all_ready: phase = 3

    # --- Phase 3: rotate around goal ---
    elif phase == 3:
        rot_angle += rotation_speed
        for i,theta in enumerate(circle_slots):
            drones[i,0] = goal[0] + circle_radius*np.cos(theta+rot_angle)
            drones[i,1] = goal[1] + circle_radius*np.sin(theta+rot_angle)
            drones[i,2] = goal[2]

    # --- Update plot ---
    drone_scatter._offsets3d = (drones[:,0], drones[:,1], drones[:,2])
    obstacle_scatter._offsets3d = (obstacles[:,0], obstacles[:,1], obstacles[:,2])
    return drone_scatter, obstacle_scatter

# ---------------- RUN ANIMATION ----------------
ani = FuncAnimation(fig, update, frames=600, interval=50, blit=False)
plt.show()
