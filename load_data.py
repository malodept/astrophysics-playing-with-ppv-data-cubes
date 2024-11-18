import h5py
import numpy as np

# Load data
file_path = 'D:/malo/Documents/projets/ppv cubes astro/fourier and pca/velocity data cube for HW1 (2).mat'

def load_data(step=10):
    with h5py.File(file_path, 'r') as file:
        key = list(file.keys())[0]  # Assuming single key
        data = np.array(file[key])  # Load the full data
    # Downsample data
    data_ds = data[::step, ::step, ::step]
    return data, data_ds

if __name__ == "__main__":
    data, data_ds = load_data()
    print(f"Loaded full data shape: {data.shape}")
    print(f"Downsampled data shape: {data_ds.shape}")
