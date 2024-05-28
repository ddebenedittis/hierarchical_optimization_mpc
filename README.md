# Hierarchical Optimization Model Predictive Control

ROS 2 packages to implement generic controllers based on Hierarchical Optimization (HO) and Model Predictive Control (MPC).

## Table of Contents

- [Hierarchical Optimization Model Predictive Control](#hierarchical-optimization-model-predictive-control)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [Scripts](#scripts)
  - [Known Bugs](#known-bugs)
  - [Author](#author)

## Overview

## Installation

These packages have been tested with ROS 2 Humble and ROS 2 Iron on an Ubuntu system.

To use Torch with an NVIDIA graphics card, it is necessary to install the NVIDIA drivers for Ubuntu. [Here](https://letmegooglethat.com/?q=Install+nvidia+drivers+ubuntu).

### Dependencies

Ubuntu packages:
- `git`

Python 3 packages:
- `casadi`
- `clarabel`
- `cvxpy`
- `matplotlib`
- `numpy`
- `osqp`
- `pandas`
- `proxsuite`
- `quadprog`
- `qpsolvers`
- `scipy`
- `reluqp`
- `torch`
- `torchaudio`
- `torchvision`

`Torch`, `torchaudio`, and `torchvision` are necessary for the neural network approximator and ReLuQP.

## Usage

When I write `<some_text>`, you have to change the value in the <>.

[Create a GitHub SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). Just do it.

Create a workspace and clone the repo in the source directory (don't fucking call your workspace a generic `<my_workspace>`)
```shell
mkdir <my_workspace>
cd <my_workspace>
git clone --recursive git@github.com:ddebenedittis/hierarchical_optimization_mpc.git src/hierarchical_optimization_mpc
```

Build the workspace with
```shell
colcon build --symlink-install
```

Source the workspace with (you have to add it to the `~/.bashrc` or do it on every newly opened terminal)
```shell
source install/setup.base
```

### Scripts

Single robot example
```shell
ros2 run hierarchical_optimization_mpc example_single_robot
```

Multi robot example
```shell
ros2 run hierarchical_optimization_mpc example_multi_robot
```

## Known Bugs

No known bugs.

## Author

[Davide De Benedittis](https://3.bp.blogspot.com/-xvFfjYBPegM/VvFp02nHUjI/AAAAAAAAIoc/Mysj-ESrXPQFQI_yOJFQQz2kwZuIQiAKA/s1600/He-Man.png)