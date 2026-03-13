import matplotlib.pyplot as plt
import numpy as np

from aspcore.filter import create_filter

# Parameter choices
rng = np.random.default_rng()
block_len = 64
num_blocks = 8
num_samples = block_len * num_blocks
ir_len = 32
num_in = 2
num_out = 3

# Create signals and impulse response
ir = rng.normal(0, 1, size=(num_in, num_out, ir_len))
sig = rng.normal(0, 1, size=(num_in, num_samples))

# Filter using the filter class
filt = create_filter(ir=ir, sum_over_input=True)
filtered_signal = np.zeros((num_out, num_samples))

filtered_signal = [
    filt.process(sig[:, b * block_len : (b + 1) * block_len]) for b in range(num_blocks)
]
filtered_signal = np.concatenate(filtered_signal, axis=-1)


fig, ax = plt.subplots(num_in, 1, sharex=True)
for i in range(num_in):
    ax[i].plot(sig[i])
    ax[i].set_title(f"Input {i}")
ax[-1].set_xlabel("Sample")
plt.tight_layout()

fig, ax = plt.subplots(num_out, 1, sharex=True)
for i in range(num_out):
    ax[i].plot(filtered_signal[i])
    ax[i].set_title(f"Output {i}")
ax[-1].set_xlabel("Sample")
plt.tight_layout()

plt.show()
