#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:23:53 2021

@author: zammour
"""

import numpy as np
import matplotlib.pyplot as plt


def triangle(time, amplitude, freq, noise = False, noise_intensity = 0.1):
    
    freq /= 2
        
    y = 4*freq*(time - np.floor(2*time*freq+0.5)/2/freq)*(-1)**(np.floor(2*time*freq+0.5))
    
    if noise: y+= np.random.uniform(-noise_intensity, noise_intensity, size = y.shape)

    return y


def square(time, amplitude, freq, noise = False, noise_intensity = 0.1):
    
    y = amplitude*np.sign(np.sin(freq*time))
    
    if noise: y+= np.random.uniform(-noise_intensity, noise_intensity, size = y.shape)
    
    return y
    


def sinusoid(time, amplitudes, freqs, noise = False, noise_intensity = 0.1):
    
    y = np.zeros(len(time))
    
    for n in range(len(amplitudes)):
        
        y += amplitudes[n]*np.sin(freqs[n]*np.pi*time)
        
    if noise: y+= np.random.normal(0, noise_intensity, size = y.shape)
        
    return y


def lorenz_attractor(time, x_0, y_0, z_0, Prandtl_number = 10, Rayleigh_number = 28, beta = 8/3):
    
    dt = time[1] - time[0]
    
    x = [x_0]
    y = [y_0]
    z = [z_0]
    
    for t in range(len(time)):
        
        xp = (Prandtl_number * (y[t] - x[t])) * dt
        yp = (Rayleigh_number * x[t] - y[t] - x[t] * z[t]) * dt
        zp = (x[t] * y[t] - beta * z[t]) * dt
        
        x.append(x[-1] + xp)
        y.append(y[-1] + yp)
        z.append(z[-1] + zp)
        
    return x, y, z


nsecs=1440
dt = 0.1
simtime = np.arange(0, nsecs, dt)
plot_idx = len(simtime)//4

amp_figAC = 1
freq_figAC= 1/60
y_figAC = triangle(simtime, amp_figAC, freq_figAC)

amp_figD = [1, 0.5, 1/6, 1/3]
freq_figD = np.array([1, 2, 3, 4])/60
y_figD = sinusoid(simtime, amp_figD, freq_figD)

amp_figE = np.array([1, 1/4, 1/3, 1/3, 1/5, 1/10, 1/10, 1/12, 1/3, 1/2, 1/6, 1/2, -1/5, 1/4, 1/4, 1/10])
freq_figE = np.arange(1,17)/180
y_figE = sinusoid(simtime, amp_figE, freq_figE)

amp_figF = amp_figD
freq_figF = freq_figD
y_figF = sinusoid(simtime, amp_figF, freq_figF, noise = True, noise_intensity = 0.5)

amp_figG = 1
freq_figG = 2/60
y_figG = square(simtime, amp_figG, freq_figG, noise = True, noise_intensity=0.01)

x_figH, y_figH, z_figH = lorenz_attractor(simtime/10, 1, 1, 1, Rayleigh_number = 28, Prandtl_number = 10)

amp_figI = [1]
freq_figI1 = [1/6/dt]
y_figI1 = sinusoid(simtime, amp_figI, freq_figI1)

freq_figI2 = [1/800/dt]
y_figI2 = sinusoid(simtime, amp_figI, freq_figI2)


plt.subplot(2, 4, 1)
plt.plot(simtime[:plot_idx], y_figD[:plot_idx], c='brown')
plt.title('Periodic')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.plot(simtime[:plot_idx], y_figE[:plot_idx], c='brown')
plt.title('Complicated periodic')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.plot(simtime[:plot_idx], y_figF[:plot_idx], c='brown', linewidth=0.25)
plt.axis('off')
plt.title('Extremely noisy target')

plt.subplot(2, 4, 4)
plt.plot(simtime[:plot_idx*2], y_figG[:plot_idx*2], c='brown')
plt.axis('off')
plt.title('Discontinuous target')

plt.subplot(2, 4, 5)
plt.plot(simtime[:plot_idx], y_figAC[:plot_idx], c='brown')
plt.axis('off')
plt.title('Triangle')

plt.subplot(2, 4, 6)
plt.plot(simtime[:plot_idx], y_figH[:plot_idx], c = 'brown')
plt.axis('off')
plt.title('Lorenz attractor')


plt.subplot(2, 4, 7)
plt.plot(simtime[:plot_idx//20], y_figI1[:plot_idx//20], c ='brown')
plt.axis('off')
plt.title(r'Sine wave w/ period 6 $\tau$')

plt.subplot(2, 4, 8)
plt.plot(simtime[:plot_idx], y_figI2[:plot_idx], c ='brown')
plt.axis('off')
plt.title(r'Sine wave w/ period 800 $\tau$')
