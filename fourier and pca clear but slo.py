import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
from sklearn.decomposition import PCA


# Load the file .mat
with h5py.File('D:/malo/Documents/projets/ppv cubes astro/fourier and pca/velocity data cube for HW1 (2).mat', 'r') as file:
    # Display the keys (name of the variables) contained in the file
    print("Variables in the file .mat :")
    for key in file.keys():
        print(key)

    # Explore the structure of the data for each variable
    for key in file.keys():
        data = file[key]
        print(f"\nStructure of the variable '{key}' :")
        print(f"Type of data : {data.dtype}")
        print(f"Dimensions : {data.shape}")
        #print(f"Data : {data[:]}")

        # Visualize a slice of the data in 2D
        if len(data.shape) == 3:  # Verify if the data is in 3D
            # viz 1: Visualization of a single slice in 2D
            slice_index = 0  # Index of the slice to visualize
            slice_data = data[:, :, slice_index]
            plt.figure()
            plt.imshow(slice_data, cmap='viridis')
            plt.colorbar()
            plt.title(f'Visualization of the slice {slice_index} of the variable {key}')
            plt.show()

            # viz 2: 3D visualization of slices (downsampled)
            # Visualization 3D with optimized performance and style
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            data_np = np.array(data)  # Convert the data in numpy array

            # Downsample more efficiently
            step = max(1, min(data_np.shape) // 50)  # Dynamically decide step based on size
            x, y, z = data_np.shape
            X, Y, Z = np.meshgrid(range(x), range(y), range(z))
            X_sub = X[::step, ::step, ::step]
            Y_sub = Y[::step, ::step, ::step]
            Z_sub = Z[::step, ::step, ::step]
            data_sub = data_np[::step, ::step, ::step]

            # Modern scatter plot
            sc = ax.scatter(
                X_sub, Y_sub, Z_sub, 
                c=data_sub.flatten(), cmap='plasma', alpha=0.8, s=10  # Modern style
            )
            fig.colorbar(sc, ax=ax, shrink=0.6)
            ax.set_title("Modernized 3D Data Visualization", fontsize=14)
            plt.show()


            # viz 3: Animation of slices along the 3rd dimension
            fig, ax = plt.subplots()
            im = ax.imshow(data[:, :, 0], cmap='plasma', aspect='auto')
            plt.colorbar(im)
            plt.title('Dynamic Slices Animation', fontsize=14)

            def update(frame):
                im.set_array(data[:, :, frame])
                ax.set_title(f'Slice {frame}', fontsize=12)
                return im,

            ani = animation.FuncAnimation(fig, update, frames=data.shape[2], interval=30, blit=True)
            plt.show()


# 3D Fourier transform
fft_cube = np.fft.fftn(data_np)
power_spectrum = np.abs(fft_cube)**2  # Power spectrum

# Calculation of the associated frequencies 
kx, ky, kz = np.meshgrid(
    np.fft.fftfreq(data_np.shape[0]),
    np.fft.fftfreq(data_np.shape[1]),
    np.fft.fftfreq(data_np.shape[2]),
    indexing='ij'
)
k = np.sqrt(kx**2 + ky**2 + kz**2)

# Calculation of the radial spectrum
k_flat = k.flatten()
ps_flat = power_spectrum.flatten()
bins = np.linspace(0, k.max(), 50)
k_bin_centers = 0.5 * (bins[:-1] + bins[1:])
ps_radial = np.histogram(k_flat, bins, weights=ps_flat)[0] / np.histogram(k_flat, bins)[0]

# Visualization of the radial spectrum 
plt.figure()
plt.loglog(k_bin_centers, ps_radial)
plt.xlabel('Wave number k')
plt.ylabel('Power Spectrum')
plt.title('Turbulence spectrum')
plt.show()



# Example to access a specific value in the cube
valeur = data_np[2, 45, 124]  
print(f"Valeur à (2, 45, 124) : {valeur}")



print(f"Shape du cube : {data_np.shape}")
x, y, z = data_np.shape
data_reshaped = data_np.reshape(x * y, z)

pca = PCA(n_components=3)  # Keep the 3 first principal components
data_pca = pca.fit_transform(data_reshaped)

print("Explained variance ratio :", pca.explained_variance_ratio_)

data_pca_spatial = data_pca.reshape(x, y, -1)  # Keep the structure

# Visualization of the principal components
plt.figure(figsize=(10, 6))

# Display of each component
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(data_pca_spatial[:, :, i], cmap='viridis')
    plt.colorbar()
    plt.title(f'Principal component {i+1}')
    
plt.tight_layout()
plt.show()



# Improved PCA 3D Visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter with modern settings
scatter = ax.scatter(
    data_pca[:, 0], data_pca[:, 1], data_pca[:, 2],
    c=data_pca[:, 0], cmap='coolwarm', alpha=0.75, s=5
)

# Modernize axes
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_zlabel('PC3', fontsize=12)
fig.colorbar(scatter, ax=ax, shrink=0.5, label="Color Gradient")
plt.title("Modernized PCA Projection", fontsize=16)
plt.show()
