import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from load_data import load_data

# Load data
data, _ = load_data()
x, y, z = data.shape
data_reshaped = data.reshape(-1, z)  # Collapse spatial dimensions

# Perform PCA
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_reshaped)

# Prepare data for 3D plot
pc1, pc2, pc3 = data_pca[:, 0], data_pca[:, 1], data_pca[:, 2]

# Create 3D scatter plot
fig_pca_3d = px.scatter_3d(
    x=pc1,
    y=pc2,
    z=pc3,
    color=pc3,  
    title="Principal Components in 3D",
    labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
    color_continuous_scale='Viridis'
)

fig_pca_3d.update_layout(
    scene=dict(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        zaxis_title="Principal Component 3",
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

# Save and show
output_path = "D:/malo/Documents/projets/ppv cubes astro/fourier and pca/principal_components_3d_viz.html"
fig_pca_3d.write_html(output_path)
print(f"Saved PCA 3D visualization to {output_path}")
fig_pca_3d.show()
