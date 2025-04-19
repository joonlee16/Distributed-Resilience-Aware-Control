# Distributed Resilience-Aware Control in Multi-Robot Networks
This repository provides an implementation of a **resilience-aware Control Barrier Function-based Quadratic Programming (CBF-QP) controller** for multi-robot systems. The controller is designed to guarantee sufficient **network resilience** in a robot networks with distance-based communication models, even in the presence of malicious robots. There exists at most F malicious robots that try to disrupt the consensus protocol by sharing faulty or even adversarial information. Unlike prior approaches that maintain network resilience by either: (1) assuming access to global network state (i.e., knowledge of all robot states), which becomes unreliable when malicious agents are present, or (2) enforcing fixed topologies with pre-known resilience guarantees, our approach enables robots to collectively maintain resilience in **dynamic and reconfigurable communication graphs** using **only locally available information**. In our paper, we provide theoretical guarantees showing that - under certain assumptions on adversary behavior - normal robots can achieve consensus despite the presence of malicious agents. The controller also includes additional CBF constraints to ensure inter-agent collision avoidance.

## About This Repository
This repository requires **numpy**, **matplotlib**, **cvxpy**, and **gurobi**.
It contains a simulation framework for multi-agent systems with different update scenarios. The simulation allows you to test how agents behave under various attack conditions such as nominal updates, overstatements, and understatements.
- `simulate.py`: Main script to run the simulation.
- `single_integrator.py`: Defines agent dynamics and communication models.
- `helper.py`: Contains utility functions used in the simulation.

## Running the Simulation
If you run `simulate.py`, it will prompt you to select a scenario using your keyboard:

Press `1` for Nominal Update

Press `2` for Overstatement

Press `3` for Understatement

## Scenarios
To demonstrate our work, we present three different scenarios, whose videos are shown below. 
1. Nominal Update: Malicious robots **share accurate** connectivity levels with neighbors.
2. Overstatement: Malicious robots **overstate** connectivity levels with neighbors.
3. Understatement: Malicious robots **understate** connectivity levels with neighbors.

In all scenarios, the robots are tasked with spreading out as much as possible, which inherently conflicts with their need to maintain resilient network formation. Our controller enables robots to spread **as far as the desired network resilience is maintained**.

## Scenario 1: Nominal Update

## Scenario 2: Overstatement

## Scenario 3: Understatement

As it can be seen from the videos, the type of adversarial behavior directly affects the mobility and performance of the robots. For full theoretical details, implementation insights, and results, please refer to our [paper](https://arxiv.org/abs/2504.03120).
