import h5py
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the .mat file
file_path = 'D:/malo/Documents/projets/ppv cubes astro/fourier and pca/velocity data cube for HW1 (2).mat'

with h5py.File(file_path, 'r') as file:
    # Inspect keys in the file
    print("Variables in the file .mat:")
    for key in file.keys():
        print(key)

    # Extract 3D data (assuming one variable for simplicity)
    key = list(file.keys())[0]  # Use the first variable as default
    data = np.array(file[key])  # Convert to a NumPy array
    print(f"Loaded data shape: {data.shape}")

# Downsampling for visualization performance
step = max(1, min(data.shape) // 50)
data_ds = data[::step, ::step, ::step]

# Viz 1: Interactive 2D slice viewer (with Plotly)
slice_index = 0  

fig_2d = go.Figure()
fig_2d.add_trace(go.Heatmap(
    z=data[:, :, slice_index],
    colorscale='Viridis',
    colorbar=dict(title="Intensity"),
))

fig_2d.update_layout(
    title=f"2D Slice {slice_index} Visualization",
    xaxis_title="X Axis",
    yaxis_title="Y Axis",
    autosize=True,
)
fig_2d.show()

# Viz 2: Interactive 3D volume visualization
X, Y, Z = np.meshgrid(
    np.arange(data_ds.shape[0]),
    np.arange(data_ds.shape[1]),
    np.arange(data_ds.shape[2]),
    indexing='ij'
)

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
fig_3d.show()

# Viz 3: 3D Animation (rotating cube)
rotation_frames = 60  # Number of frames for animation
angles = np.linspace(0, 2 * np.pi, rotation_frames)

frames = []
for angle in angles:
    frames.append(go.Frame(
        data=[go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data_ds.flatten(),
            opacity=0.2,
            isomin=data_ds.min(),
            isomax=data_ds.max(),
            surface_count=20,
            colorscale='Viridis',
        )],
        layout=go.Layout(
            scene=dict(camera=dict(
                eye=dict(x=np.cos(angle), y=np.sin(angle), z=0.5)
            ))
        )
    ))

fig_animation = go.Figure(
    data=[go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=data_ds.flatten(),
        opacity=0.2,
        isomin=data_ds.min(),
        isomax=data_ds.max(),
        surface_count=20,
        colorscale='Viridis',
    )],
    layout=go.Layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])]
        )]
    ),
    frames=frames
)
fig_animation.show()

# Turbulence analysis (FFT + spectrum)
fft_cube = np.fft.fftn(data)
power_spectrum = np.abs(fft_cube)**2

kx, ky, kz = np.meshgrid(
    np.fft.fftfreq(data.shape[0]),
    np.fft.fftfreq(data.shape[1]),
    np.fft.fftfreq(data.shape[2]),
    indexing='ij'
)

k = np.sqrt(kx**2 + ky**2 + kz**2)
k_flat = k.flatten()
ps_flat = power_spectrum.flatten()
bins = np.linspace(0, k.max(), 50)
k_bin_centers = 0.5 * (bins[:-1] + bins[1:])
ps_radial = np.histogram(k_flat, bins, weights=ps_flat)[0] / np.histogram(k_flat, bins)[0]

# Spectrum visualization
plt.figure()
plt.loglog(k_bin_centers, ps_radial)
plt.xlabel('Wave number k')
plt.ylabel('Power Spectrum')
plt.title('Turbulence Spectrum')
plt.show()

# PCA on the data
x, y, z = data.shape
data_reshaped = data.reshape(x * y, z)

pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_reshaped)

print("Explained variance ratio:", pca.explained_variance_ratio_)

data_pca_spatial = data_pca.reshape(x, y, -1)

# PCA Visualization
fig_pca = go.Figure()

for i in range(3):
    fig_pca.add_trace(go.Heatmap(
        z=data_pca_spatial[:, :, i],
        colorscale='Viridis',
        colorbar=dict(title=f"PC {i+1}"),
    ))

fig_pca.update_layout(
    title="Principal Components Visualization",
    autosize=True,
)
fig_pca.show()

# 3D PCA scatter
fig_pca_3d = go.Figure(data=go.Scatter3d(
    x=data_pca[:, 0],
    y=data_pca[:, 1],
    z=data_pca[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=data_pca[:, 0],  # Color by PC1
        colorscale='Viridis',
        opacity=0.7
    )
))

fig_pca_3d.update_layout(
    title="3D PCA Projection",
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3"
    )
)
fig_pca_3d.show()
