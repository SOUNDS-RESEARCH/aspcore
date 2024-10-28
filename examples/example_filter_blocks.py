import numpy as np
import aspcore

# Parameter choices
rng = np.random.default_rng()
block_len = 256
num_blocks = 10
num_samples = block_len * num_blocks
ir_len = 32
num_in = 2
num_out = 3

# Create signals and impulse response
ir = rng.normal(0, 1, size = (num_in,num_out,ir_len))
sig = rng.normal(0, 1, size = (num_in, num_samples))

# Filter using the filter class
filt = aspcore.create_filter(ir = ir, sum_over_input=True)
filtered_signal = np.zeros((num_out, num_samples))

for b in range(num_blocks):
    idxs = slice(b*block_len, (b+1)*block_len)
    signal_block = sig[:,idxs]

    filtered_signal[:,idxs] = filt.process(signal_block)
