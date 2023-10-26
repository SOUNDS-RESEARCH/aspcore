import numpy as np
import aspcore.filterclasses as fc

# Parameter choices
rng = np.random.default_rng()
num_samples = 256
ir_len = 32
num_in = 2
num_out = 3

# Create signals and impulse responses
ir = rng.normal(0, 1, size = (num_samples, num_in,num_out,ir_len))
sig = rng.normal(0, 1, size = (num_in, num_samples))

# Filter using the dynamic filter class
filt = fc.create_filter(num_in=num_in, num_out=num_out, ir_len=ir_len, sum_over_input=True, dynamic=True)
filtered_signal_dyn = np.zeros((num_out, num_samples))
for n in range(num_samples):
    filt.update_ir(ir[n,...])
    filtered_signal_dyn[:,n:n+1] = filt.process(sig[:,n:n+1])

# Compare against the direct implementation of the definition
# of a time-varying filter
buffered_input = np.concatenate((np.zeros((num_in, ir_len-1)), sig), axis=-1)
filtered_signal = np.zeros((num_out, num_samples))
for n in range(num_samples):
    for out_ch in range(num_out):
        for in_ch in range(num_in):
            for i in range(ir_len):
                filtered_signal[out_ch,n] += ir[n, in_ch, out_ch, i] * buffered_input[in_ch,ir_len+n-i-1]

print(f"Both implementations are equal: {np.allclose(filtered_signal, filtered_signal_dyn)}")