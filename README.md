# ASPCORE : Adaptive / Audio Signal Processing Core
## Introduction
The package contains classes and functions implementing different versions of linear convolutions. What makes them useful above what's already available in scipy and numpy is that they are intended to be used in a streaming manner, and all filters support multiple inputs and multiple outputs. There is also support for convolution with a time-varying impulse response (soon). 

The package uses just-in-time compilation from numba to achieve lower computational cost. 

## Installation
I recommend an editable install. That basically means that python pretends that the package is installed like any other package, but it's instead just a link to a folder on your computer with the code. Then you can use the package from anywhere at the same time as you are developing it. It can be done by opening your favorite terminal and writing (changing out cd for the appropriate change-directory command, and the path to whatever you want)

```
cd c:/folder/parent_folder_to_code
git clone https://github.com/SOUNDS-RESEARCH/aspcore.git
pip install -e ./aspcore
```

## Dependencies
Python version is required to be 3.8 <= version <= 3.11, because that is supported by Numba (as of 0.57.0). \
All dependencies can be installed using
```
cd c:/folder/parent_folder_to_code
pip install -r aspcore/requirements.txt
```

Required packages are Numpy, Scipy and Numba. \
The unit tests are relying on pytest and hypothesis. 

## Usage
The main function of this package is aspcore.filterclasses.create_filter(). Using the keyword arguments, it will select and return the appropriate filter class. \
All filters can then be used convolve using its process() method, which returns the filtered signal. Signals are formatted with the time index as the last axis. 

```python
import numpy as np
import aspcore.filterclasses as fc

channels_in, channels_out, num_samples, ir_len = 5, 3, 128, 16

signal = np.random.normal(0,1,size=(channels_in, num_samples))
ir = np.random.normal(0,1,size=(channels_out, ir_len))

filt = fc.create_filter(ir=ir, sum_over_inputs=True)

filtered_signal = filt.process(signal)
```