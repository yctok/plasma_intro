# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:37:06 2024

@author: ychuang
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Physical Constants
e = 1.602e-19    # charge unit (Coulombs)
mp = 1.67e-27    # Mass of the nucleon (kg)

# Charge Particle Parameters (here for ion)
Z = 1            # Z of ion
q = Z*e          # charge of ion
N = 1           # total number of nucleons in ion
m = N*mp         # mass of ion


#Zeroth order magnetic field strength
B0 = 3e-5 #Tesla

#calculate cyclotron frequency and associated period
omega_c = q*B0/m
period = 2*math.pi/omega_c

# Initial conditions
initial_position = np.array([0, 0, -1])   # Initial position vector (m)
initial_velocity = np.array([1e2, 0, 1e2]) # Initial velocity vector (m/s)
s = np.linalg.norm(initial_velocity)    # Initial speed (m/s)
spar = initial_velocity[2]              # parallel velocity
sperp = np.sqrt(s**2 - spar**2)         # perpendicular velocity
rL = m*sperp/(q*B0)


# Time parameters
t_start = 0          # Start time (s)
t_end = 10*period    # End time (s)
dt = period*0.0001   # Time step (s)

# Number of time steps
num_steps = int((t_end - t_start) / dt)

# Preallocate arrays to store trajectory
positions = np.zeros((num_steps, 3))
velocities = np.zeros((num_steps, 3))
times = np.zeros(num_steps)

# Initial components of arrays
positions[0] = initial_position
velocities[0] = initial_velocity
times[0] = t_start

# Fields as functions of position
# Electric Field
def E(x, y, z, t):
    # Ex = 0
    # Ex = 1 + (300 * t)/(1000 * period)
    Ex = 0
    Ey = 0
    # Ey = (0.5*m*(sperp**2)/ q)*B0/(1000*rL)
    Ez = 0
    return np.array([Ex, Ey, Ez])

# Magnetic Field
def B(x, y, z, t):
    Bx = 0
    By = 0
    # Bz = B0
    # Bz = B0*(1- (y/(1000*rL)))
    # Bz = B0*(1- (x/(1000*rL)))
    # Bz = B0*(1- (5*z/(1000*rL)))
    Bz = B0*(1 + (50*z/(1000*rL))**2)
    # Bz = B0*(1 + np.sin(50*z/(1000*rL)))
    # Bz = B0*(1+ (100*t/ (1000*period)))
    return np.array([Bx, By, Bz])




# Simulation loop using Euler's method
for i in range(num_steps-1):
    # Current position and velocity
    r = positions[i]
    v = velocities[i]
    
    # Calculate electric and magnetic fields at current position
    E_field = E(*r, times[i])
    B_field = B(*r, times[i])
    
    # Calculate acceleration using Lorentz force equation
    acceleration = (q / m)*(E_field + np.cross(v, B_field))
    
    # Update velocity and position using Euler's method
    velocities[i+1] = v + dt * acceleration
    positions[i+1] = r + dt * v
    times[i+1] = times[i] + dt



    
# Plot trajectory

#3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Particle Trajectory')


#xy plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(positions[:, 0], positions[:, 1])
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title('Particle Trajectory')
ax2.scatter(positions[0,0],positions[0,1], color='green', label='Start')
ax2.scatter(positions[-1, 0],positions[-1,1], color='red', label='End')
plt.show()