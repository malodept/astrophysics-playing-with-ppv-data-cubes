import numpy as np
import plotly.graph_objects as go

# Dummy data generation 
data = np.random.rand(50, 50, 50)  # Example 3D data

# Downsampling for performance
step = 5  # Adjust based on data size and desired resolution
data_np = np.array(data)
x = np.arange(0, data_np.shape[0], step)
y = np.arange(0, data_np.shape[1], step)
z = np.arange(0, data_np.shape[2], step)
values = data_np[::step, ::step, ::step]

# Flatten the 3D grid for plotly
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Create the interactive 3D visualization
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    opacity=0.2,  # Transparency level
    isomin=values.min(),
    isomax=values.max(),
    surface_count=25,  # Number of contour surfaces
    colorscale="Viridis"  # Color palette
))

# Layout adjustments 
fig.update_layout(
    title="Interactive 3D Data Cube",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis",
        xaxis=dict(backgroundcolor="white"),
        yaxis=dict(backgroundcolor="white"),
        zaxis=dict(backgroundcolor="white"),
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    coloraxis_colorbar=dict(title="Intensity")
)

# Export to HTML for interactive use
fig.write_html("interactive_cube.html")
fig.show()
