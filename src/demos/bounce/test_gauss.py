import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import math

# 0. Configuration
N_FUNCTIONS = 2
# Adjusted SIGMA_X for potentially sharper function.
# If F(x) is very sharp, a smaller sigma shows subtle smoothing,
# a larger sigma shows more aggressive smoothing. Let's try a slightly smaller one.
SIGMA_X = 0.1
SIGMA_X_INDIVIDUAL = 0.2 # For individual smoothing
X_MIN, X_MAX = 0, 2 * np.pi # Matching the x-range of your plot
NUM_POINTS = 2000 # Increased points for sharper details
np.random.seed(42) # For reproducibility

# 1. Generate parameters for random harmonic functions to make F(x) sharper
def generate_random_harmonic_params(n_functions):
    """Generates random parameters (A, B, C) for n_functions, aiming for a sharper F(x)."""
    params = []
    for i in range(n_functions):
        A = np.random.uniform(0.95, 1.15) # Amplitudes kept relatively stable
        if i == 0:
            # First function acting directly on x:
            # Increase frequency range to ensure more oscillations over [0, 2*pi]
            B = np.random.uniform(1.0, 5.0) # e.g., 5-15 cycles in the range
        else:
            # Subsequent functions act on y (output of previous func, approx in [-1,1]).
            # To make them sharp, B*y should cover many radians.
            # A B value of ~pi means one half-cycle if y goes from -1 to 1.
            # A B value of ~3*pi means ~1.5 cycles if y goes from -1 to 1.
            B = np.random.uniform(10.0, 15.0) # Approx 6.0 to 15.0
        C = np.random.uniform(0, 2 * np.pi)   # Full phase range for more variety
        params.append({'A': A, 'B': B, 'C': C})
    return params

# 2. Define function composition (same as before)
def apply_composed_functions(x_input, params_list):
    current_value = np.copy(x_input)
    for p in params_list:
        current_value = p['A'] * np.cos(p['B'] * current_value + p['C'])
    return current_value

# 3. Generate x coordinates (same as before)
x_coords = np.linspace(X_MIN, X_MAX, NUM_POINTS)
dx = x_coords[1] - x_coords[0]
sigma_pixels = SIGMA_X / dx

# 4. Generate original function parameters
original_params = generate_random_harmonic_params(N_FUNCTIONS)

# 5. Calculate Plot 1: Original composed function F(x)
y_original_composed = apply_composed_functions(x_coords, original_params)

# 6. Calculate Plot 2: Individually smoothed functions composed (H(x))
smoothed_params_list = []
for p_orig in original_params:
    A_smooth = p_orig['A'] * np.exp(-(p_orig['B']**2 * SIGMA_X_INDIVIDUAL**2) / 2.0)
    smoothed_params_list.append({'A': A_smooth, 'B': p_orig['B'], 'C': p_orig['C']})
y_individual_smoothed_composed = apply_composed_functions(x_coords, smoothed_params_list)

# 7. Calculate Plot 3: Original composed function, then smoothed (G_sigma(F(x)))
y_final_smoothed = gaussian_filter1d(y_original_composed, sigma=sigma_pixels)

# 8. Plotting (same structure as before)
fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True) # sharey can be True or False
title_str = (f'{N_FUNCTIONS} Composed Harmonic Functions Comparison (Sharper $F(x)$) ($\sigma_x={SIGMA_X}$)\n'
             f'X range: [{X_MIN:.2f}, {X_MAX:.2f}], Points: {NUM_POINTS}')
fig.suptitle(title_str, fontsize=16)

axs[0].plot(x_coords, y_original_composed, label='$F(x) = f_{10}(\dots f_1(x)\dots)$', color='blue', linewidth=1) # Thinner line for sharpness
axs[0].set_title('1. Original Composed Function $F(x)$')
axs[0].set_ylabel('Amplitude')
axs[0].legend(loc='upper right')
axs[0].grid(True, linestyle=':', alpha=0.7)

axs[1].plot(x_coords, y_individual_smoothed_composed, label='$H(x) = h_{10}(\dots h_1(x)\dots)$, $h_i = G_\sigma(f_i)$', color='green', linewidth=1)
axs[1].set_title('2. Composition of Individually Smoothed Functions $H(x)$')
axs[1].set_ylabel('Amplitude')
axs[1].legend(loc='upper right')
axs[1].grid(True, linestyle=':', alpha=0.7)

axs[2].plot(x_coords, y_final_smoothed, label='$G_\sigma(F(x))$', color='red', linewidth=1.5) # Smoothed can be slightly thicker
axs[2].set_title('3. Original Composed Function Smoothed at the End $G_\sigma(F(x))$')
axs[2].set_xlabel('x')
axs[2].set_ylabel('Amplitude')
axs[2].legend(loc='upper right')
axs[2].grid(True, linestyle=':', alpha=0.7)

# Optional: Common Y-axis limits if results are comparable, or individual if ranges vary widely.
# To make subtle features in potentially lower-amplitude smoothed versions visible, individual y-axes might be better.
# If you want them shared:
# common_ymin = min(np.min(y_original_composed), np.min(y_individual_smoothed_composed), np.min(y_final_smoothed)) - 0.1
# common_ymax = max(np.max(y_original_composed), np.max(y_individual_smoothed_composed), np.max(y_final_smoothed)) + 0.1
# for ax in axs:
#    ax.set_ylim(common_ymin, common_ymax)


plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.show()

# --- Optional: Print some parameter details for verification ---
print(f"Configuration: N_FUNCTIONS={N_FUNCTIONS}, SIGMA_X={SIGMA_X}")
print(f"X_RANGE=[{X_MIN:.2f}, {X_MAX:.2f}], NUM_POINTS={NUM_POINTS}")
print(f"Sigma for gaussian_filter1d (in pixels): {sigma_pixels:.2f}\n")

print("Original Function Parameters (A, B, C) for the first 3 functions:")
for i in range(min(3, N_FUNCTIONS)):
    p = original_params[i]
    print(f"  f_{i+1}: A={p['A']:.3f}, B={p['B']:.3f}, C={p['C']:.3f}")

print("\nSmoothed Function Parameters (A_smooth, B, C) for the first 3 functions (used in H(x)):")
for i in range(min(3, N_FUNCTIONS)):
    p_s = smoothed_params_list[i]
    p_o = original_params[i] # Corresponding original function
    reduction_factor = np.exp(-(p_o['B']**2 * SIGMA_X**2) / 2.0)
    print(f"  h_{i+1}: A_s={p_s['A']:.3f} (orig A={p_o['A']:.3f}, reduction={reduction_factor:.3f}), B={p_s['B']:.3f}, C={p_s['C']:.3f}")