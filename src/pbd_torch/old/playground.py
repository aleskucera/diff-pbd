import matplotlib

# Get the list of named colors in Matplotlib
colors = matplotlib.colors.TABLEAU_COLORS  # Alternatively, use matplotlib.colors.TABLEAU_COLORS

# Print the color names and their hex values
for name, hex_value in colors.items():
    print(f"{name}: {hex_value}")