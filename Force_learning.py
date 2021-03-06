#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:46:11 2021

@author: zammour
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sparse
from scipy import stats
from functions import triangle, sinusoid, square, lorenz_attractor


def create_network(N = 1000, g = 1.5, alpha = 1, p = .1, neurons_recorded = 10):
    
    """
    Parameters
    ----------
    N : int, optional
        Number fo neurons. The default is 1000.
    g : float, optional
        Chaotic factor. The default is 1.5.
    alpha : float, optional
        Inverse learning rate. The default is 1.
    p : float, optional
        Density of the non null weights of the network. The default is 1.

    Returns
    -------
    network : pd.Series
    Network containing, the weights, and the history of the weights and outputs after training or testing.
    """
    
    scale = 1/np.sqrt(N * p)
    rvs = stats.norm().rvs
    network = pd.Series({
        
        'x' : 0.5*np.random.randn(N,1),
        'z' : 0.5*np.random.randn(1,1),
        'y' : 0.5*np.random.randn(1,1),
        'J_GG' : sparse.random(N,N,p, data_rvs = rvs).todense()*g*scale,
        'alpha' : alpha,
        'J_Gz' : g*scale*np.random.normal(0, 1/N, (N//2, 1)),
        'J_FG' : 0*g*scale*np.random.normal(0, 1/N, (N//2, 1)), 	# synaptic strengths from internal pool to control unit
        'J_GF' : 2.0*(np.random.rand(N,1)-0.5),	# the feedback now comes from the control unit as opposed to the output
        'zidxs' : np.arange(N//2),
        'yidxs' : np.arange(N//2, N),
        'neurons' : np.zeros((neurons_recorded,1)),
        'history_z' : None,
        'history_y' : None,
        'history_J_Gz' : None,
        'history_J_FG' : None,
        'history_mae' : None,
        })
    
    network = network.append(pd.Series({
        'r' : np.tanh(network.x),
        'neurons_idxs' : np.random.randint(0, len(network.zidxs), size = neurons_recorded)
        }))

    return network


def train(network, time, learn_every, ft, plot_training = True, title = None):

    network.history_z = np.zeros((1, len(time)))
    network.history_y = np.zeros((1, len(time)))
    network.history_J_Gz = np.zeros((1, len(time)))
    network.history_J_FG = np.zeros((1, len(time)))
    network.history_mae = np.zeros((1, len(time)+1))    
    network.neurons = np.zeros((len(network.neurons_idxs),1))

    ti = 0;
    Pz = (1.0/network.alpha)*np.identity(len(network.zidxs));
    Py = (1.0/network.alpha)*np.identity(len(network.yidxs));

    if plot_training: plt.figure()
    if title != None: plt.suptitle(title, fontweight='bold')



        
    for t in simtime:
            
        if plot_training:
            if ti%(nsecs/2) == 0:
            	plt.subplot(211);
            	plt.plot(simtime, ft.T, linewidth = 3, color = 'green');
            	plt.plot(simtime, network.history_z.T, linewidth = 3, color = 'red');
            	plt.plot(simtime, network.history_y.T, linewidth = 3, color = 'magenta'); 
            	plt.title('training', fontsize = 14, fontweight = 'bold');
            	plt.xlabel('time', fontsize = 14, fontweight = 'bold');
            	plt.ylabel('f, z and y', fontsize = 14, fontweight = 'bold');
            	plt.legend(['f', 'z', 'y']);
            	
            	plt.subplot(212);
            	plt.plot(simtime, network.history_J_Gz.T, linewidth = 3); 
            	plt.plot(simtime, network.history_J_FG.T, linewidth = 3, color = 'green'); 
            	plt.xlabel('time', fontsize = 14, fontweight = 'bold');
            	plt.ylabel('|w_o|, |w_c|', fontsize = 14, fontweight = 'bold');
            	plt.legend(['|w_o|', '|w_c|'])
            	plt.pause(0.5)
                
            	if ti + nsecs/2 < len(simtime):
                    plt.clf()
                    
                
                # sim, so x(t) and r(t) are created.
        network.x = np.add(np.add(np.dot((1.0-dt), network.x), np.dot(np.dot(network.J_GG, network.r),dt)), np.dot(np.dot(network.J_Gz,network.z), dt))
        network.r = np.tanh(network.x);
        rz = network.r[network.zidxs]   		# the neurons that project to the output
        ry = network.r[network.yidxs]			# the neurons that project to the control unit
        network.z = network.J_Gz.T*rz;
        network.y = network.J_FG.T*ry;
        
    
        if ti % learn_every == 0:
        	# update inverse correlation matrix for the output unit
        	kz = np.dot(Pz,network.r);
        	rPrz = np.dot(network.r.T,kz);
        	cz = np.divide(1.0,np.add(1.0, rPrz));
        	Pz = np.subtract(Pz, np.dot(kz,np.dot(kz,cz).T))    
        	# update the error for the linear readout
        	e = np.subtract(network.z, ft[ti]);
        	# update the output weights
        	dJ_Gz = np.dot(np.dot(-float(e),kz),cz);
        	network.J_Gz += dJ_Gz;
            
    
        	# update inverse correlation matrix for the control unit
        	ky = Py*ry;
        	rPry = ry.T*ky;
        	cy = 1.0/(1.0 + rPry);
        	Py -= ky*(ky*cy).T;    
        
        	### NOTE WE USE THE OUTPUT'S ERROR ###
        	# update the output weights
        	dJ_FG = -float(e)*ky*cy;
        	network.J_FG += dJ_FG;
            
    
        # Store the output of the system.
        network.history_z[0,ti] = network.z;
        network.history_y[0,ti] = network.y;
        network.history_J_Gz[0,ti] = np.sqrt(np.dot(network.J_Gz.T, network.J_Gz));	
        network.history_J_FG[0,ti] = np.sqrt(np.dot(network.J_FG.T, network.J_FG));
        
        ti += 1;
        
        network.neurons = np.append(network.neurons, np.array(network.r[network.neurons_idxs]), axis = 1)

        
        network.history_mae[0, ti] = np.sum(np.abs(network.history_z[0, ti-learn_every:ti]-ft[ti-learn_every:ti]))/learn_every

    return network    

def test(network, time, ft, plot_test = True,  title = None):

    network.history_z = np.zeros((1, len(time)))
    network.history_y = np.zeros((1, len(time)))
    network.neurons = np.zeros((len(network.neurons_idxs),1))
    if np.all(network.history_J_Gz == None):
        network.history_J_Gz = np.repeat(np.sqrt(np.dot(network.J_Gz.T, network.J_Gz)), len(time))
        network.history_J_FG = np.repeat(np.sqrt(np.dot(network.J_FG.T, network.J_FG)), len(time))

    ti = 0;


    for t in simtime:				# don't want to subtract time in indices
        
        # sim, so x(t) and r(t) are created.
        network.x = np.add(np.add(np.dot((1.0-dt), network.x), np.dot(np.dot(network.J_GG,network.r),dt)), np.dot(np.dot(network.J_Gz,network.z),dt)); # note the y here.
        
    
        network.r = np.tanh(network.x);
        network.z = network.J_Gz.T*network.r[network.zidxs];
        network.y = network.J_FG.T*network.r[network.yidxs];
        
        network.history_z.T[ti] = network.z;
        network.history_y.T[ti] = network.y;
        
        network.neurons = np.append(network.neurons, np.array(network.r[network.neurons_idxs]), axis = 1)
        
        ti += 1;

    network.history_J_Gz = np.repeat(np.sqrt(np.dot(network.J_Gz.T, network.J_Gz)), len(time))
    network.history_J_FG = np.repeat(np.sqrt(np.dot(network.J_FG.T, network.J_FG)), len(time))


    error_avg = np.sum(np.abs(network.history_z-ft))/len(time)
    print('Testing MAE: ' + str(error_avg))
    
    if plot_test:
        
        plt.figure()
        if title != None: plt.suptitle(title, fontweight='bold')
        plt.plot(simtime, ft.T, color = 'green');
        plt.plot(simtime, network.history_z.T,  color = 'brown');
        plt.title('Testing', fontsize = 14, fontweight =  'bold');
        plt.xlabel('time', fontsize =  14, fontweight = 'bold');
        plt.axis('tight')
        plt.ylabel('f, z and y', fontsize = 14, fontweight = 'bold');
        plt.legend(['f', 'z', 'y']);
        
    return network
        

def plot_activity(network, time, title = None):
    
    nrows = 2 + network.neurons.shape[0]
    
    plt.figure()
    if title != None: plt.suptitle(title, fontweight='bold')
    plt.subplot(nrows, 1, 1)
    plt.plot(time, network.history_z.T, color = 'r')
    plt.axis('off')

    for row in range(network.neurons.shape[0]):
        plt.subplot(nrows, 1, 2 + row)
        plt.plot(time, network.neurons[row,:-1])
        plt.axis('off')    

    plt.subplot(nrows, 1, nrows)
    plt.plot(time[:-1], np.abs(np.diff(network.history_J_Gz)).T, color = 'orange')
    plt.axis('off')
    plt.text(-len(time)*0.01, 0, '$|\dot w|$', fontweight = 'bold')



nsecs = 1440;
dt = 0.1;
simtime = np.arange(0, nsecs, dt)
amp = 1.3;
freq = 1/60;
ft = triangle(simtime, amp, freq)


network = create_network()
network_tested = test(network, simtime, ft, plot_test = False)
plot_activity(network, simtime, title = 'A - Spontaneous Activity')

network_trained = train(network, simtime, 3, ft)
plot_activity(network_trained, simtime, title = 'B - Learning')

network_tested = test(network_trained, simtime, ft, plot_test = False)
plot_activity(network_tested, simtime, title = 'C - Post Learning')


amp_figD = [1, 0.5, 1/6, 1/3]
freq_figD = np.array([1, 2, 3, 4])/60
y_figD = sinusoid(simtime, amp_figD, freq_figD)
network_trained = train(network, simtime, 3, y_figD, plot_training=False)
network_tested = test(network_trained, simtime, y_figD, title = 'D - Periodic')


amp_figE = np.array([1, 1/4, 1/3, 1/3, 1/5, 1/10, 1/10, 1/12, 1/3, 1/2, 1/6, 1/2, -1/5, 1/4, 1/4, 1/10])/2
freq_figE = np.arange(1,17)/180
y_figE = sinusoid(simtime, amp_figE, freq_figE)
network_trained = train(network, simtime, 3, y_figE, plot_training=False)
network_tested = test(network_trained, simtime, y_figE, title = 'E - Complicated periodic')


amp_figF = amp_figD
freq_figF = freq_figD
y_figF = sinusoid(simtime, amp_figF, freq_figF, noise = True, noise_intensity = 0.5)
network_trained = train(network, simtime, 3, y_figF, plot_training=False)
network_tested = test(network_trained, simtime, y_figF, title = 'F - Extremely noisy target')


amp_figG = 1
freq_figG = 2/60
y_figG = square(simtime, amp_figG, freq_figG, noise = True, noise_intensity=0.01)
network_trained = train(network, simtime, 3, y_figG, plot_training=False)
network_tested = test(network_trained, simtime, y_figG, title = 'G - Discontinuous target')


x_figH, y_figH, z_figH = lorenz_attractor(simtime/10, 1, 1, 1, Rayleigh_number = 28, Prandtl_number = 10)
network_trained = train(network, simtime, 3, y_figH[:-1]/10, plot_training=False)
network_tested = test(network_trained, simtime, y_figH[:-1]/10, title = 'H - Lorenz attractor')


amp_figI = [1]
freq_figI1 = [1/6/dt]
y_figI1 = sinusoid(simtime, amp_figI, freq_figI1)
network_trained = train(network, simtime, 3, y_figI1, plot_training=False)
network_tested = test(network_trained, simtime, y_figI1, title = r'I - Sine wave w/ period 6 $\tau$')


freq_figI2 = [1/800/dt]
y_figI2 = sinusoid(simtime, amp_figI, freq_figI2)
network_trained = train(network, simtime, 3, y_figI2, plot_training=False)
network_tested = test(network_trained, simtime, y_figI2, title = r'I - Sine wave w/ period 800 $\tau$')


