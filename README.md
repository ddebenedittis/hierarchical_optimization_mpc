# Hierarchical Optimization Model Predictive Control

ROS 2 packages to implement generic controllers based on Hierarchical Optimization (HO) and Model Predictive Control (MPC).

## Table of Contents

- [Hierarchical Optimization Model Predictive Control](#hierarchical-optimization-model-predictive-control)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
    - [Docker](#docker)
    - [Docker Compose](#docker-compose)
  - [Usage](#usage)
    - [Scripts](#scripts)
      - [With ROS](#with-ros)
      - [With Python](#with-python)
    - [Distributed](#distributed)
      - [With Python](#with-python-1)
      - [With ROS](#with-ros-1)
      - [Examples](#examples)
  - [Development](#development)
    - [Pre-Commit](#pre-commit)
    - [Tests](#tests)
    - [Paper Figures](#paper-figures)
  - [Known Bugs](#known-bugs)
  - [Publications](#publications)
  - [Author](#author)

## Overview

## Installation

These packages have been tested with ROS 2 Humble and ROS 2 Iron on an Ubuntu system.

To use Torch with an NVIDIA graphics card, it is necessary to install the NVIDIA drivers for Ubuntu. [Here](https://letmegooglethat.com/?q=Install+nvidia+drivers+ubuntu).

The repository root is the colcon workspace: the ROS packages are in `src/`, the Docker files in `docker/`, and standalone scripts in `scripts/`.

### Docker

Install [Docker Community Edition](https://docs.docker.com/engine/install/ubuntu/) (ex Docker Engine).
You can follow the installation method through `apt`.
Note that it makes you verify the installation by running `sudo docker run hello-world`.
It is better to avoid running this command with `sudo` and instead follow the post installation steps first and then run the command without `sudo`.

Follow with the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) for Linux.
This will allow you to run Docker without `sudo`.

Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (nvidia-docker2).

From the repository root, build the docker image with
```shell
docker/build.bash [-a] [-f] [-h] [-l] [-r] [-t]
```
Where the optional arguments represent:
- `-a` or `--all`: build the image with all the dependencies
- `-f` or `--ffmpeg`: build the image with ffmpeg (for saving videos)
- `-h` or `--help`: show the help message
- `-l` or `--latex`: build the image with LaTeX
- `-r` or `--rebuild`: rebuild the image
- `-t` or `--torch`: build the image with PyTorch

Run the container with
```shell
docker/run.bash
```

This repo also supports VS Code devcontainers.
The Docker setup is based on [`docker_ros_nvidia`](https://github.com/ddebenedittis/docker_ros_nvidia).

### Docker Compose

As an alternative, you can use Docker Compose. First, set up X11 forwarding (once per login session):
```shell
xhost +local:docker
XAUTH=/tmp/.docker.xauth; [ -f "$XAUTH" ] || (touch "$XAUTH" && xauth nlist $DISPLAY | sed 's/^..../ffff/' | xauth -f "$XAUTH" nmerge -)
```

Then build, run in detached mode, and attach:
```shell
# Build the image (optional features via env vars: FFMPEG=1, LATEX=1, TORCH=1)
docker compose -f docker/docker-compose.yaml build

# Start the container in detached mode
docker compose -f docker/docker-compose.yaml up -d

# Open a shell inside the container (repeatable for multiple sessions)
docker compose -f docker/docker-compose.yaml exec ho_mpc bash

# Stop the container
docker compose -f docker/docker-compose.yaml down
```

## Usage

When I write `<some_text>`, you have to change the value in the <>.

[Create a GitHub SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). Just do it.

Clone the repo, which is itself the workspace, and enter it
```shell
git clone --recursive git@github.com:ddebenedittis/hierarchical_optimization_mpc.git
cd hierarchical_optimization_mpc
```

Build the workspace with
```shell
colcon build --symlink-install
```

Source the workspace with (you have to add it to the `~/.bashrc` or do it on every newly opened terminal)
```shell
source install/setup.bash
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

### Distributed
#### With Python
Distributed examples can be run with the scripts `network_simulation.py` in the `scenarios` folder in `distributed_ho_mpc` package.
The following scenarios with the relative setting are already settend in the folders:
- Radial Switching
- Movement in Formation with obstacle avoidance
- Coverage
- Passing through a narrow gap 


#### With ROS
The relative pkg are:
- dhqp: 
  run in different nodes the d-hqp algorithm
- limo_simulation
  provide a gazebo simulation for testing the algorithm. It spawns limo robots 

#### Examples
```shell
ros2 launch limo_simulation gazebo_models_diff.launch.py
```
then
```shell
ros2 launch dhqp network.launch.py
```

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

### Paper Figures

`scripts/omni_results/` turns the omnidirectional comparison campaign into tables and figures.
It runs with the host Python (numpy, matplotlib, and a LaTeX installation for the paper-style figures), no ROS needed.
```shell
python3 scripts/omni_results/make_results.py        # summary figures and tables
python3 scripts/omni_results/make_paper_figures.py  # paper-style figures
```
By default both read the campaign outputs in `out/omni_campaign_2026-09-25` and `out/omni_nl5_2026-09-26` and write to `out/omni_results/`; see `--help` for the options.

## Known Bugs

None.

## Publications

If you find this project useful in your research, please consider citing my related work (available [here](https://doi.org/10.1109/LRA.2025.3559843)):

```bibtex
@article{debenedittis2025managing,
  author={De Benedittis, Davide and Garabini, Manolo and Pallottino, Lucia},
  journal={IEEE Robotics and Automation Letters},
  title={Managing Conflicting Tasks in Heterogeneous Multi-Robot Systems Through Hierarchical Optimization}, 
  year={2025},
  volume={10},
  number={6},
  pages={5305-5312},
  doi={10.1109/LRA.2025.3559843}
}
```

## Author

- [Davide De Benedittis](https://3.bp.blogspot.com/-xvFfjYBPegM/VvFp02nHUjI/AAAAAAAAIoc/Mysj-ESrXPQFQI_yOJFQQz2kwZuIQiAKA/s1600/He-Man.png)
- [Federico Iadarola](https://github.com/fedeiada)