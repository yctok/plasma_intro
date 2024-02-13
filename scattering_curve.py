# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:40:10 2024

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np


theta=np.arange(0,2*np.pi,2*np.pi/400)

a= 1.0      #semi-major axis
r = 10.
e1 = 2.
e2 = 3. 

def get_r(theta, e, a):
    return a*(e**2 - 1)/(1 + e*np.cos(theta))



fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, get_r(theta = theta, e = e1, a = a))
ax.set_rmax(30)
# ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()