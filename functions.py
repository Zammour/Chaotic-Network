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
        
    if noise: y+= np.random.uniform(-noise_intensity, noise_intensity, size = y.shape)
        
    return y

nsecs=1440
dt = 0.1
simtime = np.arange(0, nsecs, dt)

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
y_figF = sinusoid(simtime, amp_figF, freq_figF, noise = True)

amp_figG = 1
freq_figG = 2/60
y_figG = square(simtime, amp_figG, freq_figG, noise = True, noise_intensity=0.01)

plot_idx = len(simtime)//4

plt.subplot(2, 3, 1)
plt.plot(simtime[:plot_idx], y_figD[:plot_idx], c='brown')
plt.title('Periodic')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.plot(simtime[:plot_idx], y_figE[:plot_idx], c='brown')
plt.title('Complicated periodic')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.plot(simtime[:plot_idx], y_figF[:plot_idx], c='brown')
plt.axis('off')
plt.title('Extremely noisy target')

plt.subplot(2, 3, 4)
plt.plot(simtime[:plot_idx*2], y_figG[:plot_idx*2], c='brown')
plt.axis('off')
plt.title('Discontinuous target')

plt.subplot(2, 3, 5)
plt.plot(simtime[:plot_idx], y_figAC[:plot_idx], c='brown')
plt.axis('off')
plt.title('Triangle')

plt.subplot(2, 3, 6)
plt.axis('off')
plt.title('Lorenz attractor')


