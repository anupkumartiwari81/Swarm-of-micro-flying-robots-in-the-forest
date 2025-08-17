import sys
import random
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# ---------- Parameters ----------
NUM_DRONES = 40
STEP = 0.05
NEIGHBOR_THRESH = 9.0

drones = []
paths = [[] for _ in range(NUM_DRONES)]

camera_angle = 0.0
camera_radius = 15.0
camera_height = 7.0

# ---------- 3D Vector Utilities ----------
def subtract(a, b): return np.array(a) - np.array(b)
def add(a, b): return np.array(a) + np.array(b)
def scale(a, s): return np.array(a) * s
def dot(a, b): return float(np.dot(a, b))
def magnitude(v): return float(np.linalg.norm(v))
def normalize(v): return v / magnitude(v) if magnitude(v) > 1e-6 else np.zeros(3)

def randomPoint(r=2.0):
    return np.array([
        random.uniform(-r, r),
        random.uniform(-r, r),
        random.uniform(-r, r)
    ], dtype=float)

# ---------- PCA ----------
def computePCA(neighbors):
    if len(neighbors) < 2:
        return np.array([1, 0, 0], dtype=float)

    mat = np.array(neighbors)
    mean = np.mean(mat, axis=0)
    mat = mat - mean

    cov = np.dot(mat.T, mat)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    return normalize(principal)

# ---------- Centroid ----------
def getCentroid():
    return np.mean(drones, axis=0)

# ---------- Movement ----------
def moveDrone(idx):
    global drones
    p = drones[idx]

    # Neighbors
    neighbors = []
    for i in range(NUM_DRONES):
        if i == idx: continue
        dist = magnitude(drones[i] - p)
        if dist <= NEIGHBOR_THRESH:
            neighbors.append(drones[i])

    if len(neighbors) < 2:
        return

    lineDir = normalize(computePCA(neighbors))
    center = np.mean(neighbors, axis=0)

    # Projection onto line
    t = dot(p - center, lineDir)
    projection = center + scale(lineDir, t)
    toLine = projection - p
    distToLine = magnitude(toLine)

    if distToLine < 0.01:
        return
    speed = STEP
    if distToLine < 0.1:
        speed *= 0.2

    moveVec = normalize(toLine) * speed
    p = p + moveVec

    # Cohesion to centroid
    centroid = getCentroid()
    toCenter = centroid - p
    p = p + 0.01 * toCenter

    drones[idx] = p
    paths[idx].append(p)

# ---------- OpenGL Rendering ----------
def drawSphere(p, r, color):
    glPushMatrix()
    glTranslatef(p[0], p[1], p[2])
    glColor3fv(color)
    glutSolidSphere(r, 15, 15)
    glPopMatrix()

def drawGrid():
    glColor3f(0.85, 0.85, 0.85)
    glBegin(GL_LINES)
    for i in range(-10, 11):
        glVertex3f(i, 0, -10)
        glVertex3f(i, 0, 10)
        glVertex3f(-10, 0, i)
        glVertex3f(10, 0, i)
    glEnd()

def display():
    global camera_angle
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    camX = camera_radius * np.cos(camera_angle)
    camZ = camera_radius * np.sin(camera_angle)
    gluLookAt(camX, camera_height, camZ, 0, 0, 0, 0, 1, 0)

    drawGrid()

    # Trails
    for i in range(NUM_DRONES):
        if len(paths[i]) < 2: continue
        glBegin(GL_LINE_STRIP)
        for pt in paths[i]:
            glVertex3f(pt[0], pt[1], pt[2])
        glEnd()

    # Drones
    colors = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.6, 0.0, 0.6)
    ]
    for i in range(NUM_DRONES):
        color = colors[i % len(colors)]
        drawSphere(drones[i], 0.25, color)

    glutSwapBuffers()

def reshape(w, h):
    if h == 0: h = 1
    aspect = w / float(h)
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, aspect, 0.1, 100.0)

def timer(v):
    global camera_angle
    for i in range(NUM_DRONES):
        moveDrone(i)

    camera_angle += 0.01
    if camera_angle > 2*np.pi:
        camera_angle -= 2*np.pi

    glutPostRedisplay()
    glutTimerFunc(33, timer, 0)  # ~30 FPS

def initOpenGL():
    glEnable(GL_DEPTH_TEST)
    glClearColor(1.0, 1.0, 1.0, 1.0)

# ---------- Main ----------
if __name__ == "__main__":
    random.seed()
    for i in range(NUM_DRONES):
        drones.append(randomPoint(4.0))
        paths[i].append(drones[i])

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"Drone Swarm PCA Formation")
    initOpenGL()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutTimerFunc(33, timer, 0)
    glutMainLoop()
