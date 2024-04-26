# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:04:28 2024

@author: ychuang
"""

import numpy as np
import matplotlib.pyplot as plt


import functools

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy
from astropy.constants.si import c

from plasmapy.dispersion.analytical.stix_ import stix
from plasmapy.dispersion.analytical.two_fluid_ import two_fluid
from plasmapy.formulary import speeds
from plasmapy.particles import Particle



# Constants
epsilon_0 = 8.854e-12  # Vacuum permittivity
electron_charge = 1.602e-19  # Electron charge
electron_mass = 9.109e-31  # Electron mass
ion_mass = 1.673e-27  # Ion mass, assume H
c_light = 3.0e8   # speed of light


# Plasma Parameters
electron_density = 1e19  # Electron density (per cubic meter)
ion_density = electron_density  # Ion density (per cubic meter), assume H
electron_temperature = 1e12  # Electron temperature (in Kelvin)
ion_temperature = electron_temperature  # Ion temperature (in Kelvin)
B0 = 0.4  # Magnetic field strength (T)

# Calculate plasma frequencies
omega_pe = np.sqrt(electron_density * electron_charge ** 2 / (electron_mass * epsilon_0))
omega_pi = np.sqrt(ion_density * electron_charge ** 2 / (ion_mass * epsilon_0))


mu_0 = np.pi * 4e-7  # Vacuum permeability

# Calculate cyclotron frequencies
omega_ce = electron_charge * B0 / electron_mass
omega_ci = electron_charge * B0 / ion_mass

print(omega_ce/omega_pe)

# Angle of Wave Vector
angle = 90.0  # angle between k vector and B vector (in degrees)
angle_rad= angle*np.pi/180.0 # angle in radians

# define input parameters for PlasmaPy
inputs_1 = {
    "theta": angle_rad * u.rad,    # np.linspace(0, np.pi, 50) * u.rad,
    "ions": Particle("p"),
    "n_i": ion_density*1e-6 * u.cm**-3,
    "B": B0 * u.T,
    "w": np.linspace(omega_pe*1e-4, omega_pe*3, 5000)  * u.rad / u.s,
}

# define a meshgrid based on the number of theta values
omegas, thetas = np.meshgrid(
    inputs_1["w"].value, inputs_1["theta"].value, indexing="ij"
)
omegas = np.dstack((omegas,) * 4).squeeze()
thetas = np.dstack((thetas,) * 4).squeeze()

# compute k values
k = stix(**inputs_1)



# plot the results

plt.rcParams["figure.figsize"] = [10.5, 0.56 * 10.5]

fs = 20  # default font size
figwidth, figheight = plt.rcParams["figure.figsize"]
figheight = 1.6 * figheight
fig = plt.figure(figsize=[figwidth, figheight])

plt.scatter(k*c_light/omega_pe,
            omegas/omega_pe,
            s=8,
    label="",
)
# Add labels to the axes
plt.xlabel('kc/$\omega_{pe}$',fontsize=fs)
plt.ylabel('$\omega$/$\omega_{pe}$',fontsize=fs)
plt.title('High frequency dispersion relation')

# Set the ranges for the axes
plt.xlim(0.01, 3)  
plt.ylim(0.01, 3)

plt.show()



