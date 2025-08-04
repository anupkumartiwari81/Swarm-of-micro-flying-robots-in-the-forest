import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------- PARAMETERS -----------------
NUM_DRONES, NUM_OBSTACLES = 15, 30
SPACE_SIZE = 50
GOAL = np.array([45, 45, 20])
GOAL_RADIUS, CIRCLE_RADIUS = 3.0, 5.0
DT, MAX_SPEED = 0.1, 1.5
SENSING_RADIUS, OBSTACLE_SPEED = 8.0, 0.3
ALIGN_FRAME, ALIGN_DURATION = 100, 50  # smooth line formation

# ----------------- DRONES INIT -----------------
np.random.seed(42)
positions = np.array([5, 5, 0]) + (np.random.rand(NUM_DRONES, 3) - 0.5) * 4
velocities = np.zeros((NUM_DRONES, 3))
colors = plt.cm.rainbow(np.linspace(0, 1, NUM_DRONES))
trails = [[] for _ in range(NUM_DRONES)]

# ----------------- OBSTACLES INIT -----------------
obstacle_positions = np.random.rand(NUM_OBSTACLES, 3) * SPACE_SIZE
obstacle_velocities = (np.random.rand(NUM_OBSTACLES, 3) - 0.5) * OBSTACLE_SPEED

# ----------------- FLAGS -----------------
circle_mode, line_mode = False, False
circle_angle, frame_count, line_progress = 0.0, 0, 0.0

# ----------------- SPHERE CREATOR -----------------
def create_sphere(center, r=1.5, seg=8):
    phi, theta = np.linspace(0, np.pi, seg), np.linspace(0, 2*np.pi, seg)
    faces = []
    for i in range(len(phi)-1):
        for j in range(len(theta)-1):
            p1=[center[0]+r*np.sin(phi[i])*np.cos(theta[j]),
                center[1]+r*np.sin(phi[i])*np.sin(theta[j]),
                center[2]+r*np.cos(phi[i])]
            p2=[center[0]+r*np.sin(phi[i+1])*np.cos(theta[j]),
                center[1]+r*np.sin(phi[i+1])*np.sin(theta[j]),
                center[2]+r*np.cos(phi[i+1])]
            p3=[center[0]+r*np.sin(phi[i+1])*np.cos(theta[j+1]),
                center[1]+r*np.sin(phi[i+1])*np.sin(theta[j+1]),
                center[2]+r*np.cos(phi[i+1])]
            p4=[center[0]+r*np.sin(phi[i])*np.cos(theta[j+1]),
                center[1]+r*np.sin(phi[i])*np.sin(theta[j+1]),
                center[2]+r*np.cos(phi[i])]
            faces.append([p1,p2,p3,p4])
    return faces

# ----------------- UPDATE FUNCTIONS -----------------
def update_obstacles():
    """Move obstacles and bounce at boundaries."""
    global obstacle_positions, obstacle_velocities
    obstacle_positions += obstacle_velocities
    for i in range(NUM_OBSTACLES):
        for j in range(3):
            if obstacle_positions[i,j]<0 or obstacle_positions[i,j]>SPACE_SIZE:
                obstacle_velocities[i,j] *= -1
                obstacle_positions[i,j] = np.clip(obstacle_positions[i,j],0,SPACE_SIZE)

def detect_local_obstacles(pos):
    """Check nearby obstacles within sensing radius."""
    dists = np.linalg.norm(obstacle_positions - pos, axis=1)
    return obstacle_positions[dists < SENSING_RADIUS]

def reactive_avoidance(drone_idx, direction):
    """Simple avoidance: choose side with fewer obstacles."""
    local_obs = detect_local_obstacles(positions[drone_idx])
    if local_obs.size == 0: return np.zeros(3)
    forward = direction / (np.linalg.norm(direction)+1e-6)
    # Check four directions (left, right, up, down)
    directions = [[-forward[1],forward[0],0], [forward[1],-forward[0],0], [0,0,1], [0,0,-1]]
    best = np.zeros(3); min_count = float("inf")
    for d in directions:
        if len(detect_local_obstacles(positions[drone_idx]+np.array(d)*3)) < min_count:
            min_count, best = len(detect_local_obstacles(positions[drone_idx]+np.array(d)*3)), d
    return np.array(best)*2

def update_positions():
    """Main drone motion logic."""
    global positions, frame_count, circle_mode, circle_angle, line_mode, line_progress
    update_obstacles()

    # Start line formation after ALIGN_FRAME
    if frame_count >= ALIGN_FRAME and not line_mode:
        line_mode = True
        start_line = positions.mean(axis=0)
        end_line = start_line + np.array([28, 28, 10])
        global line_targets
        line_targets = np.array([start_line+(end_line-start_line)*i/(NUM_DRONES-1) for i in range(NUM_DRONES)])

    # Smooth move to line
    if line_mode and line_progress < 1.0:
        line_progress = min(1.0, line_progress+1.0/ALIGN_DURATION)
        positions = positions*(1-line_progress)+line_targets*line_progress

    # Enter circle mode when close to goal
    if not circle_mode and np.linalg.norm(positions[0]-GOAL) < GOAL_RADIUS:
        circle_mode = True

    # Circle mode motion
    if circle_mode:
        circle_angle += 0.05
        for i in range(NUM_DRONES):
            angle = circle_angle + 2*np.pi*i/NUM_DRONES
            positions[i] = [GOAL[0]+CIRCLE_RADIUS*np.cos(angle),
                            GOAL[1]+CIRCLE_RADIUS*np.sin(angle), GOAL[2]]
            trails[i].append(positions[i].copy())
            if len(trails[i]) > 50: trails[i].pop(0)
    else:  # Normal move toward goal
        for i in range(NUM_DRONES):
            direction = GOAL - positions[i]
            vel = direction/(np.linalg.norm(direction)+1e-6)*MAX_SPEED
            vel += reactive_avoidance(i, direction)
            if np.linalg.norm(vel) > MAX_SPEED: vel = vel/np.linalg.norm(vel)*MAX_SPEED
            positions[i] += vel*DT
            trails[i].append(positions[i].copy())
            if len(trails[i]) > 50: trails[i].pop(0)

    frame_count += 1

# ----------------- VISUALIZATION -----------------
def run_simulation():
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(0,SPACE_SIZE), ylim=(0,SPACE_SIZE), zlim=(0,SPACE_SIZE/2),
           xlabel="X", ylabel="Y", zlabel="Z")
    ax.scatter(*GOAL, c='green', s=200, marker='*')  # Goal point
    drone_scatter = ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=colors, s=100)
    trail_lines = [ax.plot([],[],[], lw=2, c=colors[i])[0] for i in range(NUM_DRONES)]
    labels = [ax.text(*positions[i], f"D{i+1}", color=colors[i], fontsize=9) for i in range(NUM_DRONES)]

    # Create obstacle spheres
    obstacle_polys = []
    for pos in obstacle_positions:
        poly = Poly3DCollection(create_sphere(pos), facecolors='blue', edgecolors='k', alpha=0.6)
        ax.add_collection3d(poly); obstacle_polys.append(poly)

    def animate(frame):
        update_positions()
        # Update drones
        drone_scatter._offsets3d = (positions[:,0], positions[:,1], positions[:,2])
        # Update trails
        for i, line in enumerate(trail_lines):
            arr = np.array(trails[i])
            line.set_data(arr[:,0], arr[:,1]) if len(arr)>1 else line.set_data([],[])
            line.set_3d_properties(arr[:,2]) if len(arr)>1 else line.set_3d_properties([])
        # Update labels
        for i, lbl in enumerate(labels):
            lbl.set_position((positions[i,0], positions[i,1])); lbl.set_3d_properties(positions[i,2]+1)
        # Update obstacle positions
        for i, poly in enumerate(obstacle_polys):
            poly.remove()
            new_poly = Poly3DCollection(create_sphere(obstacle_positions[i]), facecolors='blue', edgecolors='k', alpha=0.6)
            ax.add_collection3d(new_poly); obstacle_polys[i] = new_poly
        # Rotate camera
        ax.view_init(30, (frame*0.5)%360)
        return [drone_scatter]+trail_lines+labels+obstacle_polys

    # FIX: Assign to variable to avoid warning
    ani = FuncAnimation(fig, animate, frames=800, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    run_simulation()
