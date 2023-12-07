# Hierarchical Optimization Model Predictive Control

ROS 2 packages to imlpement generic controllers based on Hierarchical Optimization (HO) and Model Predictive Control (MPC).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
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
- `matplotlib`
- `numpy`
- `quadprog`
- `scipy`
- `torch`
- `torchaudio`
- `torchvision`

`Torch`, `torchaudio`, and `torchvision` are necessary for the neural network approximator.

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

## Known Bugs

- HO-MPC for multi robots becomes extremely slow after a few steps.

## Author

[Davide De Benedittis](https://3.bp.blogspot.com/-xvFfjYBPegM/VvFp02nHUjI/AAAAAAAAIoc/Mysj-ESrXPQFQI_yOJFQQz2kwZuIQiAKA/s1600/He-Man.png)