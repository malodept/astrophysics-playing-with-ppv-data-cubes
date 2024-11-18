import numpy as np
import plotly.graph_objects as go
from load_data import load_data

# Load data
_, data_ds = load_data()

# Create grid
X, Y, Z = np.meshgrid(
    np.arange(data_ds.shape[0]),
    np.arange(data_ds.shape[1]),
    np.arange(data_ds.shape[2]),
    indexing='ij'
)

# Plot 3D visualization
fig_3d = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=data_ds.flatten(),
    opacity=0.2,
    isomin=data_ds.min(),
    isomax=data_ds.max(),
    surface_count=20,
    colorscale='Viridis',
))

fig_3d.update_layout(
    title="Interactive 3D Data Visualization",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis",
    ),
    margin=dict(l=0, r=0, t=50, b=0),
)

# Save and show
output_path = "D:/malo/Documents/projets/ppv cubes astro/fourier and pca/interactive_3d_viz.html"
fig_3d.write_html(output_path)
print(f"Saved interactive 3D visualization to {output_path}")
fig_3d.show()
