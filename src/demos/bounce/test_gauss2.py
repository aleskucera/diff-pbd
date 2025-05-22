import numpy as np
from scipy.ndimage import gaussian_filter1d
from demos.utils import create_scientific_subplot_plot

# Configuration
SIGMA_X = 0.2
SIGMA_X_INDIVIDUAL = 0.2
X_MIN, X_MAX = 0, 2 * np.pi
NUM_POINTS = 2000

# Generate x coordinates
x_coords = np.linspace(X_MIN, X_MAX, NUM_POINTS)
dx = x_coords[1] - x_coords[0]
sigma_pixels = SIGMA_X / dx
sigma_pixels_individual = SIGMA_X_INDIVIDUAL / dx

# Define parameters (fixed, from output)
A_1, B_1, C_1 = 1.025, 4.803, 4.599  # f_1 parameters
A_2, B_2, C_2 = 1.070, 10.780, 0.980  # f_2 parameters

# Calculate Plot 1: Original composed function F(x) = f_2(f_1(x))
y_f1 = A_1 * np.cos(B_1 * x_coords + C_1)
y_original_composed = A_2 * np.cos(B_2 * y_f1 + C_2)

# Calculate Plot 2: Individually smoothed functions composed H(x) = h_2(h_1(x))
f_1 = A_1 * np.cos(B_1 * x_coords + C_1)
y_h1 = gaussian_filter1d(f_1, sigma=sigma_pixels_individual, mode='wrap')
f_2 = A_2 * np.cos(B_2 * y_h1 + C_2)
y_individual_smoothed_composed = gaussian_filter1d(f_2, sigma=sigma_pixels_individual, mode='wrap')

# Calculate Plot 2 (part 2): Original composed function smoothed at the end G_\sigma(F(x))
y_final_smoothed = gaussian_filter1d(y_original_composed, sigma=sigma_pixels, mode='wrap')

# Scientific plotting configuration
plot_data = {
    'original_function': {
        'original_series': {
            'data': y_original_composed,
            'label': r'$F(x) = f_2(f_1(x))$',
            'color': '#1f77b4',  # Blue
            'linewidth': 2.0,
            'alpha': 1.0
        }
    },
    'smoothed_functions': {
        'individual_smoothed_series': {
            'data': y_individual_smoothed_composed,
            'label': r'$H(x) = h_2(h_1(x))$, $h_i = G_{\sigma}(f_i)$',
            'color': '#ff7f0e',  # Orange
            'linewidth': 2.0,
            'alpha': 1.0
        },
        'final_smoothed_series': {
            'data': y_final_smoothed,
            'label': r'$G_{\sigma}(F(x))$',
            'color': '#d62728',  # Red
            'linewidth': 2.0,
            'alpha': 1.0
        }
    }
}

y_labels = {
    'original_function': r'Amplitude (-)',
    'smoothed_functions': r'Amplitude (-)'
}

plot_titles = {
    'original_function': r'Original Composed Function $F(x)$',
    'smoothed_functions': r'Smoothed Functions: $H(x)$ and $G_{\sigma}(F(x))$'
}

plot_config = {
    'plot_arrangement': 'vertical',
    'figsize': (10, 5),
    # 'suptitle': r'2 Composed Harmonic Functions Comparison ($\sigma_x=0.1$) \\ X range: $[0.00, 6.28]$, Points: $2000$',
    'suptitle': r'',
    'x_label': r'$x$ (rad)',
    'y_labels': y_labels,
    'plot_titles': plot_titles,
    'shared_x': True,
    'shared_y': False,
    'dpi': 200,
    'legend': {'show': True, 'location': 'upper right', 'fontsize': 12},
    'grid': {'show': True, 'linestyle': '--', 'alpha': 0.8},
    'font_sizes': {'axis_label': 14, 'tick_label': 14, 'title': 18, 'suptitle': 18},
    'tight_layout_params': {'rect': [0, 0.02, 1, 0.95]},
    'spines': {'top': False, 'right': False, 'left': True, 'bottom': True}
}

# LaTeX setup
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "text.latex.preamble": r'\usepackage{amsmath,amssymb,amsfonts}\usepackage{physics}\usepackage{siunitx}\AtBeginDocument{\RenewCommandCopy\qty\SI}'
})

# Generate and save plot
scientific_plot_filename = 'harmonic_smoothing_comparison.png'
fig, axes = create_scientific_subplot_plot(
    time_data=x_coords,
    plot_data=plot_data,
    config=plot_config,
    save_path=scientific_plot_filename
)
print(f"Scientific comparison plot saved to {scientific_plot_filename}")

# Print parameters for verification
print(f"\nConfiguration: N_FUNCTIONS=2, SIGMA_X={SIGMA_X}")
print(f"X_RANGE=[{X_MIN:.2f}, {X_MAX:.2f}], NUM_POINTS={NUM_POINTS}")
print(f"Sigma for gaussian_filter1d (in pixels): {sigma_pixels:.2f}\n")

print("Original Function Parameters (A, B, C):")
print(f"  f_1: A={A_1:.3f}, B={B_1:.3f}, C={C_1:.3f}")
print(f"  f_2: A={A_2:.3f}, B={B_2:.3f}, C={C_2:.3f}")

# Compute theoretical smoothed amplitudes for reporting
A_1_smooth = A_1 * np.exp(-(B_1**2 * SIGMA_X_INDIVIDUAL**2) / 2.0)
A_2_smooth = A_2 * np.exp(-(B_2**2 * SIGMA_X_INDIVIDUAL**2) / 2.0)
print("\nSmoothed Function Parameters (A_smooth, B, C) for H(x):")
print(f"  h_1: A_s={A_1_smooth:.3f} (orig A={A_1:.3f}, reduction={np.exp(-(B_1**2 * SIGMA_X_INDIVIDUAL**2) / 2.0):.3f}), B={B_1:.3f}, C={C_1:.3f}")
print(f"  h_2: A_s={A_2_smooth:.3f} (orig A={A_2:.3f}, reduction={np.exp(-(B_2**2 * SIGMA_X_INDIVIDUAL**2) / 2.0):.3f}), B={B_2:.3f}, C={C_2:.3f}")