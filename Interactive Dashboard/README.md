# ADCS Dashboard

Interactive dashboard for the Satellite Attitude Dynamics and Control Simulator.

This dashboard provides a graphical interface for configuring spacecraft parameters, running attitude simulations, tuning controllers, and visualizing spacecraft behaviour in real time.

The long-term goal is to develop a modular ADCS simulation environment in which spacecraft models, actuators, disturbances, and control algorithms can be exchanged with minimal modification to the simulation framework.

---

# Features

## Spacecraft Configuration

Configure all spacecraft and simulation parameters without modifying source code.

### Spacecraft Parameters

* Inertia tensor input
* Spacecraft mass
* Reaction wheel configuration
* Actuator limits

### Initial Conditions

* Initial quaternion
* Initial angular velocity vector
* Desired target attitude
* Desired target angular velocity

### Simulation Parameters

* Simulation timestep (`dt`)
* Simulation duration
* Numerical integrator selection
* Propagation settings

---

## Controller Configuration

The dashboard architecture is designed around interchangeable controller modules.

### Currently Implemented

* PD Controller

### Supported Parameters

* Proportional gains (`Kp`)
* Derivative gains (`Kd`)
* Gain scheduling options
* Torque limits

### Planned Controllers

* PID Controller
* Linear Quadratic Regulator (LQR)
* Quaternion Feedback Regulator (Wie Controller)
* Bang-Bang Controller
* B-Dot Detumbling Controller
* Sliding Mode Controller (SMC)
* Model Predictive Control (MPC)
* H-Infinity Robust Controller
* Adaptive Controllers
* Reinforcement Learning Based Controllers



## Live Telemetry

The dashboard provides live access to simulation states and controller outputs.

Available telemetry includes:

* Quaternion components
* Quaternion norm
* Angular velocity vector
* Angular velocity magnitude
* Attitude error
* Commanded torque
* Reaction wheel speeds
* Reaction wheel torques
* Total wheel momentum
* Rotational kinetic energy
* Control effort



## Visualization

### 3D Attitude View

* Real-time spacecraft orientation
* Body frame visualization
* Inertial frame visualization
* Target attitude visualization

### Time History Plots

* Attitude error
* Angular velocity
* Commanded torque
* Reaction wheel speeds
* Wheel torques
* Wheel momentum
* Rotational kinetic energy
* Quaternion norm drift

### Performance Metrics

* Rise time
* Settling time
* Overshoot
* Steady-state error
* Peak control effort
* Total momentum usage

---

# Advanced Simulation Features

## Disturbance Modelling

* Gravity gradient torque
* Aerodynamic torque
* Solar radiation pressure
* Residual magnetic dipole effects
* Sensor noise
* Actuator noise
* External impulsive disturbances

## Actuator Models

* Reaction wheels
* Magnetorquers


## Verification Tools

* Angular momentum conservation checks
* Energy conservation checks
* Quaternion normalization monitoring
* Numerical integration error estimation

---

# Future Work

## Controller Research Platform

The modular architecture allows rapid implementation and comparison of multiple control strategies on a common spacecraft model.

Future work includes:

* Controller comparison mode
* Automated gain optimization
* Parameter sweep studies
* Robustness analysis
* Monte Carlo simulations
* Fault injection testing
* Hardware-in-the-loop simulation
* Digital twin architecture

## Mission Analysis Features

* Orbit-attitude coupled simulations
* Earth-pointing mission profiles
* Slew maneuver optimization
* Momentum management strategies
* Safe mode simulation

## Data Export

* CSV export
* MATLAB compatible output files
* Automatic figure generation
* PDF report generation
* Simulation session save/load functionality

---

# Dashboard Layout

## Sidebar

* Spacecraft configuration
* Initial conditions
* Controller configuration
* Simulation parameters

## Main Workspace

* 3D attitude visualization
* Live telemetry panel
* Time-history plots
* Performance metrics

---


# Design Philosophy

The project follows a modular architecture in which:

* Spacecraft dynamics are isolated from the user interface.
* Controllers are implemented as interchangeable modules.
* Disturbance models can be added independently.
* New actuators can be integrated without modifying controller logic.

This separation enables the simulator to evolve from a single-controller educational tool into a general-purpose ADCS research and development platform.

---

# Example Use Cases

* Controller gain tuning
* Controller benchmarking
* Reaction wheel sizing studies
* ADCS subsystem design
* Spacecraft attitude analysis
* Educational demonstrations
* Mission design trade studies


```
```
