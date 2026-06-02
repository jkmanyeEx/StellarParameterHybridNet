import numpy as np
import os

BASE_DIR = "/Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject"
WAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "standard_wave.npy")

if os.path.exists(WAVE_PATH):
    wave = np.load(WAVE_PATH)
    print(f"Length of standard_wave: {len(wave)}")
    print(f"Min wavelength: {wave[0]:.6f}")
    print(f"Max wavelength: {wave[-1]:.6f}")
    
    # Check if it's linear
    diffs = np.diff(wave)
    print(f"Min step: {np.min(diffs):.6f}")
    print(f"Max step: {np.max(diffs):.6f}")
    print(f"Mean step: {np.mean(diffs):.6f}")
    print(f"Std of steps: {np.std(diffs):.6f}")
    
    # Compare with np.linspace
    linear_grid = np.linspace(3650.0, 10250.0, len(wave))
    max_diff = np.max(np.abs(wave - linear_grid))
    print(f"Max difference from linear grid [3650, 10250]: {max_diff:.6f}")
else:
    print("standard_wave.npy not found")
