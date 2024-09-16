# Load your thermogram data T
# T = ...

# Define parameters
fs = 100.0          # Sampling frequency in Hz
f_stim = 10.0       # Stimulation frequency in Hz

# Phase Coherence Imaging
amplitude_map, phase_map, phase_coherence_map = phase_coherence_imaging(T, fs, f_stim)

# Visualize the results
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(amplitude_map, cmap='hot')
plt.title('Amplitude Map')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(phase_map, cmap='jet')
plt.title('Phase Map')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(phase_coherence_map, cmap='gray')
plt.title('Phase Coherence Map')
plt.colorbar()

plt.tight_layout()
plt.show()
