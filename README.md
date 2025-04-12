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

In particular the lowrank module contains code developed for the following paper, considering citing it if relevant for your work.
@inproceedings{brunnstromFast2023,
  author={Brunnström, Jesper and Jälmby, Martin and Van Waterschoot, Toon and Moonen, Marc},
  booktitle={57th Asilomar Conference on Signals, Systems, and Computers}, 
  title={Fast Low-rank Filtered-x Least Mean Squares for Multichannel Active Noise Control}, 
  year={2023},
  month = oct,
  volume={},
  number={},
  pages={1085-1089},
  keywords={Loudspeakers;Computers;Convolution;Computational modeling;Frequency-domain analysis;Noise reduction;Adaptive filters;active noise control;filtered- x least mean squares;low rank;tensor decomposition;Kronecker decomposition},
  doi={10.1109/IEEECONF59524.2023.10477017}}
