# MSc Dissertation — Catheter Navigation Using MPC and Reinforcement Learning

This repository contains the code and experiments for my MSc dissertation, where I'm exploring the use of **Model Predictive Control (MPC)** and **Reinforcement Learning (RL)** for catheter navigation in simulated blood vessels.

**Note:** This repository is still a work in progress. I'll continue updating it as the project develops.

## Repository Structure

The work is organised into three main Jupyter notebooks:

**`01_environment_development.ipynb`**

This is where I developed and tested the simulation environment, including an S-shaped vessel geometry, catheter dynamics, blood-flow effects and the functions needed to evaluate navigation performance.

**`02_MPC.ipynb`**

This notebook focuses on developing and testing the MPC controller. So far, I've implemented the controller, experimented with prediction horizons and solver settings, and carried out the initial baseline evaluation. The next steps involve cost-function sensitivity and robustness testing.

**`03_SAC.ipynb` (upcoming)**

This will focus on implementing and training a Soft Actor-Critic (SAC) reinforcement learning controller, followed by its evaluation and comparison with MPC.

## Reusable Code

Rather than keeping all the code inside the notebooks, I've moved most of the functions and classes into the **`catheter_sim/`** folder.

Most of these were developed in Notebook 01, with additional MPC-related code from Notebook 02. Any new reusable functions needed for SAC will also be added here.

This makes it easier to reuse the same environment and evaluation functions across both controllers.

Experimental results are saved in **`experiments/results/`**.

## A Note on Usage

This is my personal academic project, and I'm making the repository available mainly to document my progress.

You're welcome to look through the code, but please don't copy or submit any part of this work as your own. If you find something useful and decide to build on it, please give appropriate credit.

I'll update this README as the project progresses.