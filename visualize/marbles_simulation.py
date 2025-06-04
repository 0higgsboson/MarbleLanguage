
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import random
from matplotlib import rcParams

box_size = 10
num_marbles = 20
marble_radius = 0.5
wall_thickness = 0.2

def initialize_non_overlapping_positions(num_marbles, radius, box_size):
    positions = []
    attempts = 0
    max_attempts = 1000
    while len(positions) < num_marbles and attempts < max_attempts:
        new_pos = np.random.rand(2) * (box_size - 2 * radius) + radius
        if all(np.linalg.norm(new_pos - np.array(p)) >= 2 * radius for p in positions):
            positions.append(new_pos)
        attempts += 1
    return np.array(positions)

positions = initialize_non_overlapping_positions(num_marbles, marble_radius, box_size)
velocities = []
colors = []
sizes = [marble_radius] * num_marbles
alive = [True] * num_marbles

for _ in range(num_marbles):
    speed = np.random.uniform(0.5, 2.5)
    angle = np.random.uniform(0, 2 * np.pi)
    velocities.append([speed * np.cos(angle), speed * np.sin(angle)])
    colors.append(np.random.rand(3,))

velocities = np.array(velocities)

rcParams['figure.dpi'] = 100
fig, ax = plt.subplots()

marbles = [patches.Circle(positions[i], sizes[i], color=colors[i], alpha=0.9)
           for i in range(num_marbles)]
for marble in marbles:
    ax.add_patch(marble)

ax.add_patch(patches.Rectangle((0, 0), wall_thickness, box_size, color='black'))  # Left
ax.add_patch(patches.Rectangle((0, 0), box_size, wall_thickness, color='red'))    # Bottom
ax.add_patch(patches.Rectangle((box_size - wall_thickness, 0), wall_thickness, box_size, color='black'))  # Right
ax.add_patch(patches.Rectangle((0, box_size - wall_thickness), box_size, wall_thickness, color='#FFD700'))  # Top

ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_aspect('equal')
ax.axis('off')

def spawn_new_marble():
    while True:
        new_pos = np.random.rand(2) * (box_size - 2 * marble_radius) + marble_radius
        if all(np.linalg.norm(new_pos - p) > 2 * marble_radius for p in positions):
            break
    new_vel_mag = np.random.uniform(0.5, 2.0)
    angle = np.random.uniform(0, 2 * np.pi)
    new_vel = [new_vel_mag * np.cos(angle), new_vel_mag * np.sin(angle)]
    new_color = np.random.rand(3,)
    new_size = marble_radius
    new_patch = patches.Circle(new_pos, new_size, color=new_color, alpha=0.9)
    ax.add_patch(new_patch)
    return new_pos, new_vel, new_color, new_size, new_patch

def update_with_respawn(frame):
    global positions, velocities, sizes, colors, alive, marbles

    positions[:] += velocities * 0.1

    for i in range(len(marbles)):
        if not alive[i]:
            continue
        if positions[i][0] - sizes[i] < 0 or positions[i][0] + sizes[i] > box_size:
            velocities[i][0] *= -1
        if positions[i][1] + sizes[i] > box_size:
            velocities[i][1] *= -1
            velocities[i] *= 2
            colors[i] = np.random.rand(3,)
            marbles[i].set_color(colors[i])
        if positions[i][1] - sizes[i] < 0:
            alive[i] = False
            marbles[i].set_visible(False)

    for i in range(len(marbles)):
        if not alive[i]: continue
        for j in range(i + 1, len(marbles)):
            if not alive[j]: continue
            delta_pos = positions[i] - positions[j]
            dist = np.linalg.norm(delta_pos)
            if dist < sizes[i] + sizes[j]:
                normal = delta_pos / (dist + 1e-6)
                rel_vel = velocities[i] - velocities[j]
                vel_along_normal = np.dot(rel_vel, normal)
                if vel_along_normal < 0:
                    velocities[i] -= vel_along_normal * normal
                    velocities[j] += vel_along_normal * normal
                overlap = sizes[i] + sizes[j] - dist
                positions[i] += normal * (overlap / 2)
                positions[j] -= normal * (overlap / 2)

    if sum(alive) < 5:
        new_pos, new_vel, new_color, new_size, new_patch = spawn_new_marble()
        positions = np.vstack([positions, new_pos])
        velocities = np.vstack([velocities, new_vel])
        colors.append(new_color)
        sizes.append(new_size)
        alive.append(True)
        marbles.append(new_patch)

    for i in range(len(marbles)):
        if alive[i]:
            marbles[i].center = positions[i]

    return marbles

ani = animation.FuncAnimation(fig, update_with_respawn, frames=600, interval=100, blit=True)
ani.save("marbles_simulation.mp4", writer='ffmpeg', fps=10)
print("Saved as marbles_simulation.mp4")
