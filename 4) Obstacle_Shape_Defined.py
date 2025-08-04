import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------
# Simulation Parameters
# ---------------------------
NUM_DRONES = 15
NUM_OBSTACLES = 30
SPACE_SIZE = 50
GOAL = np.array([45, 45, 20])
GOAL_RADIUS = 3.0
CIRCLE_RADIUS = 5.0
DT = 0.1
MAX_SPEED = 1.5
SENSING_RADIUS = 8.0
OBSTACLE_SPEED = 0.3
ALIGN_FRAME = 100
ALIGN_DURATION = 50   # frames for smooth transition

# Shapes
OBSTACLE_SHAPES = [
    "cube", "sphere", "triangular_prism", "pentagonal_prism",
    "hexagonal_prism", "septagonal_prism", "octagonal_prism", "rectangular_prism"
]

# Colors for shapes
shape_colors = {
    "cube": "red",
    "sphere": "blue",
    "triangular_prism": "green",
    "pentagonal_prism": "purple",
    "hexagonal_prism": "orange",
    "septagonal_prism": "pink",
    "octagonal_prism": "cyan",
    "rectangular_prism": "brown"
}

# ---------------------------
# Drones Initialization
# ---------------------------
np.random.seed(42)
cluster_center = np.array([5, 5, 0])
positions = cluster_center + (np.random.rand(NUM_DRONES, 3) - 0.5) * 4
velocities = np.zeros((NUM_DRONES, 3))
colors = plt.cm.rainbow(np.linspace(0, 1, NUM_DRONES))
trails = [[] for _ in range(NUM_DRONES)]

# Obstacles Initialization
obstacle_positions = np.random.rand(NUM_OBSTACLES, 3) * SPACE_SIZE
obstacle_velocities = (np.random.rand(NUM_OBSTACLES, 3) - 0.5) * OBSTACLE_SPEED
obstacle_shapes = np.random.choice(OBSTACLE_SHAPES, NUM_OBSTACLES)

# ---------------------------
# Shape Generators
# ---------------------------

def create_cube(center, size=2):
    x,y,z=center; d=size/2
    v=np.array([
        [x-d,y-d,z-d],[x+d,y-d,z-d],[x+d,y+d,z-d],[x-d,y+d,z-d],
        [x-d,y-d,z+d],[x+d,y-d,z+d],[x+d,y+d,z+d],[x-d,y+d,z+d]
    ])
    faces=[
        [v[j] for j in [0,1,2,3]],[v[j] for j in [4,5,6,7]],
        [v[j] for j in [0,1,5,4]],[v[j] for j in [2,3,7,6]],
        [v[j] for j in [1,2,6,5]],[v[j] for j in [4,7,3,0]]
    ]
    return faces

def create_rectangular_prism(center, size=(2,1,1)):
    x,y,z=center; dx,dy,dz=size[0]/2,size[1]/2,size[2]/2
    v=np.array([
        [x-dx,y-dy,z-dz],[x+dx,y-dy,z-dz],[x+dx,y+dy,z-dz],[x-dx,y+dy,z-dz],
        [x-dx,y-dy,z+dz],[x+dx,y-dy,z+dz],[x+dx,y+dy,z+dz],[x-dx,y+dy,z+dz]
    ])
    faces=[
        [v[j] for j in [0,1,2,3]],[v[j] for j in [4,5,6,7]],
        [v[j] for j in [0,1,5,4]],[v[j] for j in [2,3,7,6]],
        [v[j] for j in [1,2,6,5]],[v[j] for j in [4,7,3,0]]
    ]
    return faces

def create_prism(center, n_sides=6, size=2):
    """Generic n-sided prism"""
    x,y,z=center; r=size/2
    angles=np.linspace(0,2*np.pi,n_sides,endpoint=False)
    bottom=[[x+r*np.cos(a), y+r*np.sin(a), z-r] for a in angles]
    top=[[x+r*np.cos(a), y+r*np.sin(a), z+r] for a in angles]
    faces=[bottom,top]
    for i in range(n_sides):
        faces.append([bottom[i], bottom[(i+1)%n_sides], top[(i+1)%n_sides], top[i]])
    return faces

def create_triangular_prism(center, size=2):
    return create_prism(center, n_sides=3, size=size)

def create_pentagonal_prism(center, size=2):
    return create_prism(center, n_sides=5, size=size)

def create_hexagonal_prism(center, size=2):
    return create_prism(center, n_sides=6, size=size)

def create_septagonal_prism(center, size=2):
    return create_prism(center, n_sides=7, size=size)

def create_octagonal_prism(center, size=2):
    return create_prism(center, n_sides=8, size=size)

def create_sphere(center, radius=1.5, segments=8):
    phi=np.linspace(0,np.pi,segments)
    theta=np.linspace(0,2*np.pi,segments)
    faces=[]
    for i in range(len(phi)-1):
        for j in range(len(theta)-1):
            p1=[center[0]+radius*np.sin(phi[i])*np.cos(theta[j]),
                center[1]+radius*np.sin(phi[i])*np.sin(theta[j]),
                center[2]+radius*np.cos(phi[i])]
            p2=[center[0]+radius*np.sin(phi[i+1])*np.cos(theta[j]),
                center[1]+radius*np.sin(phi[i+1])*np.sin(theta[j]),
                center[2]+radius*np.cos(phi[i+1])]
            p3=[center[0]+radius*np.sin(phi[i+1])*np.cos(theta[j+1]),
                center[1]+radius*np.sin(phi[i+1])*np.sin(theta[j+1]),
                center[2]+radius*np.cos(phi[i+1])]
            p4=[center[0]+radius*np.sin(phi[i])*np.cos(theta[j+1]),
                center[1]+radius*np.sin(phi[i])*np.sin(theta[j+1]),
                center[2]+radius*np.cos(phi[i])]
            faces.append([p1,p2,p3,p4])
    return faces

# Map shapes
shape_creators = {
    "cube": create_cube,
    "sphere": create_sphere,
    "triangular_prism": create_triangular_prism,
    "pentagonal_prism": create_pentagonal_prism,
    "hexagonal_prism": create_hexagonal_prism,
    "septagonal_prism": create_septagonal_prism,
    "octagonal_prism": create_octagonal_prism,
    "rectangular_prism": create_rectangular_prism
}

# ---------------------------
# Simulation State
# ---------------------------
circle_mode=False
circle_angle=0.0
frame_count=0
line_mode=False
line_progress=0.0   # 0 to 1 smooth alignment factor

# ---------------------------
# Obstacle & Drone Updates
# ---------------------------
def update_obstacles():
    global obstacle_positions, obstacle_velocities
    obstacle_positions+=obstacle_velocities
    for i in range(NUM_OBSTACLES):
        for j in range(3):
            if obstacle_positions[i,j]<0 or obstacle_positions[i,j]>SPACE_SIZE:
                obstacle_velocities[i,j]*=-1
                obstacle_positions[i,j]=np.clip(obstacle_positions[i,j],0,SPACE_SIZE)

def detect_local_obstacles(pos):
    dists=np.linalg.norm(obstacle_positions-pos,axis=1)
    return obstacle_positions[dists<SENSING_RADIUS]

def reactive_avoidance(drone_idx,direction):
    local_obs=detect_local_obstacles(positions[drone_idx])
    if local_obs.size==0: return np.zeros(3)
    forward=direction/(np.linalg.norm(direction)+1e-6)
    left=np.array([-forward[1],forward[0],0])
    right=-left; up=np.array([0,0,1]); down=np.array([0,0,-1])
    dirs=[left,right,up,down]
    min_count=float("inf"); best=np.zeros(3)
    for d in dirs:
        check=positions[drone_idx]+d*3
        if len(detect_local_obstacles(check))<min_count:
            min_count=len(detect_local_obstacles(check)); best=d
    return best*2.0

def update_positions():
    global positions, frame_count, circle_mode, circle_angle, line_mode, line_progress
    update_obstacles()

    # Start smooth line formation
    if frame_count>=ALIGN_FRAME and not line_mode:
        line_mode=True
        # Compute target line positions
        start_line=positions.mean(axis=0)
        end_line=start_line+np.array([28,28,10])
        global line_targets
        line_targets=np.array([
            start_line+(end_line-start_line)*i/(NUM_DRONES-1)
            for i in range(NUM_DRONES)
        ])

    if line_mode and line_progress<1.0:
        line_progress=min(1.0, line_progress+1.0/ALIGN_DURATION)
        # interpolate positions toward line_targets
        positions=positions*(1-line_progress)+line_targets*line_progress

    if not circle_mode and np.linalg.norm(positions[0]-GOAL)<GOAL_RADIUS:
        circle_mode=True

    if circle_mode:
        circle_angle+=0.05
        for i in range(NUM_DRONES):
            angle_offset=(2*np.pi/NUM_DRONES)*i
            angle=circle_angle+angle_offset
            positions[i]=np.array([
                GOAL[0]+CIRCLE_RADIUS*np.cos(angle),
                GOAL[1]+CIRCLE_RADIUS*np.sin(angle),
                GOAL[2]
            ])
            trails[i].append(positions[i].copy())
            if len(trails[i])>50: trails[i].pop(0)
    else:
        for i in range(NUM_DRONES):
            direction=GOAL-positions[i]
            vel=direction/(np.linalg.norm(direction)+1e-6)*MAX_SPEED
            vel+=reactive_avoidance(i,direction)
            speed=np.linalg.norm(vel)
            if speed>MAX_SPEED: vel=vel/speed*MAX_SPEED
            positions[i]+=vel*DT
            trails[i].append(positions[i].copy())
            if len(trails[i])>50: trails[i].pop(0)

    frame_count+=1

# ---------------------------
# Visualization
# ---------------------------
def run_simulation():
    fig=plt.figure(figsize=(12,12))
    ax=fig.add_subplot(111,projection='3d')
    ax.set_xlim(0,SPACE_SIZE); ax.set_ylim(0,SPACE_SIZE); ax.set_zlim(0,SPACE_SIZE/2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.grid(True)

    # Goal
    ax.scatter(GOAL[0],GOAL[1],GOAL[2],c='green',s=200,marker='*')

    # Drone scatter
    drone_scatter=ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=colors,s=100)

    # Trails
    trail_lines=[ax.plot([],[],[],lw=2,c=colors[i])[0] for i in range(NUM_DRONES)]

    # Labels
    labels=[ax.text(positions[i,0],positions[i,1],positions[i,2]+1.0,
                    f"D{i+1}",color=colors[i],fontsize=10,weight='bold')
            for i in range(NUM_DRONES)]

    # Obstacle shapes
    obstacle_collections=[]
    for i in range(NUM_OBSTACLES):
        shape=obstacle_shapes[i]; color=shape_colors[shape]
        faces=shape_creators[shape](obstacle_positions[i])
        poly=Poly3DCollection(faces,facecolors=color,edgecolors='k',alpha=0.6)
        ax.add_collection3d(poly)
        obstacle_collections.append(poly)

    def animate(frame):
        update_positions()

        # Update drones
        drone_scatter._offsets3d=(positions[:,0],positions[:,1],positions[:,2])

        # Trails
        for i,line in enumerate(trail_lines):
            if len(trails[i])>1:
                arr=np.array(trails[i]); line.set_data(arr[:,0],arr[:,1]); line.set_3d_properties(arr[:,2])
            else:
                line.set_data([],[]); line.set_3d_properties([])

        # Labels
        for i,label in enumerate(labels):
            label.set_x(positions[i,0]); label.set_y(positions[i,1]); label.set_3d_properties(positions[i,2]+1.0)

        # Obstacles update
        for i,poly in enumerate(obstacle_collections):
            poly.remove()
            shape=obstacle_shapes[i]; color=shape_colors[shape]
            faces=shape_creators[shape](obstacle_positions[i])
            new_poly=Poly3DCollection(faces,facecolors=color,edgecolors='k',alpha=0.6)
            ax.add_collection3d(new_poly)
            obstacle_collections[i]=new_poly

        # Camera rotate
        azim_angle=(frame*0.5)%360
        ax.view_init(elev=30,azim=azim_angle)

        return [drone_scatter]+trail_lines+labels+obstacle_collections

    ani=FuncAnimation(fig,animate,frames=800,interval=50,blit=False)
    plt.show()

if __name__=="__main__":
    run_simulation()
