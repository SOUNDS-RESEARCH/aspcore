# ASPCORE : Audio Signal Processing Core
## Introduction
The package contains classes and functions implementing different versions of linear convolutions. What makes them useful above what's already available in scipy and numpy is that they are intended to be used in a streaming manner, where only parts of the input signal is available at a time. All filters support multiple inputs and multiple outputs. There is also support for convolution with a time-varying impulse response. 

The package uses just-in-time compilation from numba to achieve lower computational cost. 

**[More info and complete API documentation](https://sounds-research.github.io/aspcore/)**

## Installation
The package can be installed via pip by running
```
pip install aspcore
```
Alternatively, the package can be installed by cloning the repository and running
```
pip install path/to/aspcore
```

## License
The software is distributed under the MIT license. See the LICENSE file for more information.

## Acknowledgements
The software has been developed during a PhD project as part of the [SOUNDS ETN](https://www.sounds-etn.eu) at KU Leuven. The SOUNDS project has recieved funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 956369.


## Usage
The main function of this package is aspcore.filterclasses.create_filter(). Using the keyword arguments, it will select and return the appropriate filter class. \
All filters can then be used convolve using its process() method, which returns the filtered signal. 

Signals are formatted with the time index as the last axis, with most filters accepting signals of the form (num_channels, num_samples). Some filters accepts signals with higher dimensional channels, such as (a, b, c, ..., num_samples). 

```python
import numpy as np
import aspcore.filterclasses as fc
rng = np.random.default_rng()

channels_in, channels_out, num_samples, ir_len = 5, 3, 128, 16

signal = rng.normal(0,1,size=(channels_in, num_samples))
ir = rng.normal(0,1,size=(channels_out, ir_len))

filt = fc.create_filter(ir=ir, sum_over_inputs=True)

filtered_signal = filt.process(signal)
```