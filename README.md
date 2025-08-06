# Hierarchical Optimization Model Predictive Control

ROS 2 packages to implement generic controllers based on Hierarchical Optimization (HO) and Model Predictive Control (MPC).

## Table of Contents

- [Hierarchical Optimization Model Predictive Control](#hierarchical-optimization-model-predictive-control)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Scripts](#scripts)
      - [With ROS](#with-ros)
      - [With Python](#with-python)
        - [Distributed](#distributed)
  - [Development](#development)
    - [Pre-Commit](#pre-commit)
    - [Tests](#tests)
  - [Known Bugs](#known-bugs)
  - [Author](#author)

## Overview

## Installation

These packages have been tested with ROS 2 Humble and ROS 2 Iron on an Ubuntu system.

To use Torch with an NVIDIA graphics card, it is necessary to install the NVIDIA drivers for Ubuntu. [Here](https://letmegooglethat.com/?q=Install+nvidia+drivers+ubuntu).

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

#### With ROS

Single robot example
```shell
ros2 run hierarchical_optimization_mpc example_single_robot
```

Multi robot example
```shell
ros2 run hierarchical_optimization_mpc example_multi_robot
```
<img src="https://raw.githubusercontent.com/ddebenedittis/media/main/hierarchical_optimization_mpc/coverage_9.webp" width="500">

Toy problem 1 (multiple conflicting tasks to one robot)
```shell
ros2 run hierarchical_optimization_mpc toy_problem_1
```

#### With Python

Python is more verbose than ROS, but you can pass options

Multi robot exmaple
```shell
python3 src/hierarchical_optimization_mpc/hierarchical_optimization_mpc/example_multi_robot.py [--hierarchical {True, False}] [--n_robots [int,int]] [--solver {clarabel, osqp, proxqp, quadprog, reluqp}] [--visual_method {plot, save, none}]
```
Parameters:
- `--hierarchical bool`: if True, uses the hierarchical approach.
- `--n_robots list[int]`: Number of unicycles and omnidirectional robots (default `[6,0]`).
- `--solver {clarabel, osqp, proxqp, quadprog, reluqp}`: QP solver to use.
- `--visual_method {plot, save, none}`: how to display the results.

##### Distributed

Distributed examples can be run with the scripts `network_simulation.py` in the `scenarios` folder in `distributed_ho_mpc` package.

## Development

### Pre-Commit

Install `pre-commit` and `ruff` (already installed in Docker)
```shell
pip3 install pre-commit ruff
```

Run
```shell
pre-commit install
```
This will format all the Python code with Ruff.

### Tests

Tests can be run with
```
colcon test
```

## Known Bugs

- **Problem**: `qpsolvers` does return both the solution and the cost. This breaks older centralized code. **Solution**: create another function `solve_qp_cost` that does also return the cost, and have `solve_qp` only return the solution. ToDo.

## Author

- [Davide De Benedittis](https://3.bp.blogspot.com/-xvFfjYBPegM/VvFp02nHUjI/AAAAAAAAIoc/Mysj-ESrXPQFQI_yOJFQQz2kwZuIQiAKA/s1600/He-Man.png)
- [Federico Iadarola](https://github.com/fedeiada)