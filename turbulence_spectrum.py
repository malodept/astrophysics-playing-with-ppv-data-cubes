import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data

# Load data
data, _ = load_data()

# Compute FFT and spectrum
fft_cube = np.fft.fftn(data)
power_spectrum = np.abs(fft_cube) ** 2

kx, ky, kz = np.meshgrid(
    np.fft.fftfreq(data.shape[0]),
    np.fft.fftfreq(data.shape[1]),
    np.fft.fftfreq(data.shape[2]),
    indexing='ij'
)

k = np.sqrt(kx**2 + ky**2 + kz**2)
k_flat = k.flatten()
ps_flat = power_spectrum.flatten()

# Bin and compute radial spectrum
bins = np.linspace(0, k.max(), 50)
k_bin_centers = 0.5 * (bins[:-1] + bins[1:])
ps_radial = np.histogram(k_flat, bins, weights=ps_flat)[0] / np.histogram(k_flat, bins)[0]

# Plot spectrum
plt.figure()
plt.loglog(k_bin_centers, ps_radial, label='Power Spectrum')
plt.xlabel('Wave number k')
plt.ylabel('Power Spectrum')
plt.title('Turbulence Spectrum')
plt.legend()

# Save plot
output_path = "D:/malo/Documents/projets/ppv cubes astro/fourier and pca/turbulence_spectrum.png"
plt.savefig(output_path, dpi=300)
print(f"Saved turbulence spectrum to {output_path}")
plt.show()
