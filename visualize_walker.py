import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Setup DMC environment
def initialize_dmc_environment():
    import ctypes.util
    import subprocess
    
    osmesa_available = ctypes.util.find_library('OSMesa') is not None
    try:
        subprocess.run(['which', 'Xvfb'], check=True, capture_output=True)
        xvfb_available = True
    except:
        xvfb_available = False
    
    if osmesa_available:
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    elif xvfb_available:
        os.environ['MUJOCO_GL'] = 'glfw'
    else:
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

initialize_dmc_environment()

from dm_control import suite
from dm_control.suite.wrappers import pixels


def visualize_walker(num_steps=10):
    """Visualize walker environment observations"""
    
    # Create environment
    env = suite.load(domain_name="walker", task_name="walk")
    env = pixels.Wrapper(
        env,
        pixels_only=True,
        render_kwargs={'height': 64, 'width': 64, 'camera_id': 0}
    )
    
    # Collect frames
    frames = []
    time_step = env.reset()
    frames.append(time_step.observation['pixels'])
    
    for _ in range(num_steps - 1):
        action = np.random.uniform(
            env.action_spec().minimum,
            env.action_spec().maximum
        )
        time_step = env.step(action)
        frames.append(time_step.observation['pixels'])
    
    env.close()
    
    # Plot
    fig, axes = plt.subplots(1, num_steps, figsize=(2 * num_steps, 2))
    
    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].set_title(f't={i}')
        axes[i].axis('off')
    
    plt.suptitle('Walker Environment Observations (64x64)', fontsize=14)
    plt.tight_layout()
    plt.savefig('walker_observations.png', dpi=150)
    plt.show()
    print("Saved to walker_observations.png")


if __name__ == "__main__":
    visualize_walker(num_steps=10)