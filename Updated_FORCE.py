import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sparse
from scipy import stats
import random


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
        
    return np.array(x), np.array(y), np.array(z) 

def create_network(N = 1000, g = 1.5, alpha = 1, p = 0.1, neurons_recorded = 10):
    
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
    random.seed(10);

    scale = 1/np.sqrt(N * p)
    rvs = stats.norm().rvs
    network = pd.Series({
        
        'x' : 0.5*np.random.randn(N,1),
        'z' : 0.5*np.random.randn(1,1),
        'J_GG' : sparse.random(N,N,p, data_rvs = rvs).todense()*g*scale,
        'alpha' : alpha,
        'wo' : 0.5*np.random.randn(N,1)/np.sqrt(N),
        'J_GF' : 2.0*(np.random.rand(N,1)-0.5), 	# the feedback now comes from the control unit as opposed to the output
        'zidxs' : np.arange(N),
        #'yidxs' : np.arange(N//2, N),
        'neurons' : np.zeros((neurons_recorded,1)),
        'history_z' : None,
        'history_wo' : None,
        #'history_J_FG' : None,
        'history_mae' : None,
        })
    
    network = network.append(pd.Series({
        'r' : np.tanh(network.x),
        'neurons_idxs' : np.random.randint(0, len(network.zidxs), size = neurons_recorded)
        }))

    return network


def train(network, time, learn_every, ft, plot_training = False):

    network.history_z = np.zeros((1, len(time)))
    #network.history_y = np.zeros((1, len(time)))
    network.history_wo = np.zeros((1, len(time)))
    #network.history_J_FG = np.zeros((1, len(time)))
    network.history_mae = np.zeros((1, len(time)+1))    
    network.neurons = np.zeros((len(network.neurons_idxs),1))

    ti = 0;
    Pz = (1.0/network.alpha)*np.identity(len(network.zidxs));
    #Py = (1.0/network.alpha)*np.identity(len(network.yidxs));

    if plot_training: plt.figure()


        
    for t in simtime:
            
        if plot_training:
            if ti%(nsecs/2) == 0:
            	plt.subplot(211);
            	plt.plot(simtime, ft.T, linewidth = 3, color = 'green');
            	plt.plot(simtime, network.history_z.T, linewidth = 3, color = 'red');
            	#plt.plot(simtime, network.history_y.T, linewidth = 3, color = 'magenta'); 
            	plt.title('training', fontsize = 14, fontweight = 'bold');
            	plt.xlabel('time', fontsize = 14, fontweight = 'bold');
            	plt.ylabel('f, z ', fontsize = 14, fontweight = 'bold');
            	plt.legend(['f', 'z']);
            	
            	plt.subplot(212);
            	plt.plot(simtime, network.history_wo.T, linewidth = 3); 
            	#plt.plot(simtime, network.history_J_FG.T, linewidth = 3, color = 'green'); 
            	plt.xlabel('time', fontsize = 14, fontweight = 'bold');
            	plt.ylabel('|w_o|', fontsize = 14, fontweight = 'bold');
            	plt.legend(['|w_o|'])
            	plt.pause(0.5)
                
            	if ti + nsecs/2 < len(simtime):
                    plt.clf()
                    
                
                # sim, so x(t) and r(t) are created.
        network.x = (1.0-dt)*network.x + network.J_GG*(network.r*dt) + network.J_GF*(network.z*dt); # note the y here.
        network.r = np.tanh(network.x);
        rz = network.r[network.zidxs]   		# the neurons that project to the output
        #ry = network.r[network.yidxs]			# the neurons that project to the control unit
        network.z = network.wo.T*rz;
        #network.y = network.J_FG.T*ry;
        
    
        if ti % learn_every == 0:
        	# update inverse correlation matrix for the output unit
        	kz = Pz*rz;
        	rPrz = rz.T*kz;
        	cz = 1.0/(1.0 + rPrz);
        	Pz = Pz - kz*(kz*cz).T;    
        	# update the error for the linear readout
        	e = network.z-ft[ti];
        	# update the output weights
        	dwo = -float(e)*kz*cz;
        	network.wo += dwo;

            
    
        # Store the output of the system.
        network.history_z[0,ti] = network.z;
        #network.history_y[0,ti] = network.y;
        network.history_wo[0,ti] = np.sqrt(np.dot(network.wo.T, network.wo));	
       # network.history_J_FG[0,ti] = np.sqrt(np.dot(network.J_FG.T, network.J_FG));
        
        ti += 1;
        
        network.neurons = np.append(network.neurons, np.array(network.r[network.neurons_idxs]), axis = 1)

        
        network.history_mae[0, ti] = np.sum(np.abs(network.history_z[0, ti-learn_every:ti]-ft[ti-learn_every:ti]))/learn_every

    if plot_training == True:
        plt.figure()
        plt.plot(simtime, ft.T, linewidth = 3, color = 'green');
     #plt.plot(simtime, network.history_y.T, linewidth = 3, color = 'magenta'); 
        plt.plot(simtime, network.history_z.T, linewidth = 3, color = 'brown');
        plt.title('output', fontsize = 14, fontweight =  'bold');
        plt.xlabel('time', fontsize =  14, fontweight = 'bold');
        plt.axis('tight')
        plt.ylabel('f, z', fontsize = 14, fontweight = 'bold');
        plt.legend(['f', 'z']);
    return network  ;  

def test(network, time, ft, plot_test = True, title = None):
    network.history_wo = np.repeat(np.sqrt(np.dot(network.wo.T, network.wo)), len(time))
    network.history_z = np.zeros((1, len(time)))
    #network.history_y = np.zeros((1, len(time)))
    network.neurons = np.zeros((len(network.neurons_idxs),1))
    if np.all(network.history_wo == None):
        network.history_wo = np.repeat(np.sqrt(np.dot(network.wo.T, network.wo)), len(time))
        #network.history_J_FG = np.repeat(np.sqrt(np.dot(network.J_FG.T, network.J_FG)), len(time))


    # Now test. 
    ti = 0;


    for t in simtime:				# don't want to subtract time in indices
        
        # sim, so x(t) and r(t) are created.
        network.x = (1.0-dt)*network.x + network.J_GG*(network.r*dt) + network.J_GF*(network.z*dt); # note the y here.
        
    
        network.r = np.tanh(network.x);
        network.z = network.wo.T*network.r[network.zidxs];
        #network.y = network.J_FG.T*network.r[network.yidxs];
        
        network.history_z.T[ti] = network.z;
        #network.history_y.T[ti] = network.y;
        
        network.neurons = np.append(network.neurons, np.array(network.r[network.neurons_idxs]), axis = 1)
        
        ti += 1;

    error_avg = np.sum(np.abs(network.history_z-ft))/len(time)
    print('Testing MAE: ' + str(error_avg))
    
    if plot_test:
        
        if title != None:
            plt.figure()
            plt.suptitle(title, fontweight='bold')
            plt.plot(simtime, ft.T, linewidth = 3, color = 'green');
            #plt.plot(simtime, network.history_y.T, linewidth = 3, color = 'magenta'); 
            plt.plot(simtime, network.history_z.T, linewidth = 3, color = 'brown');
            plt.title(title, fontsize = 14, fontweight =  'bold');
            plt.xlabel('time', fontsize =  14, fontweight = 'bold');
            plt.axis('tight')
            plt.ylabel('f, z', fontsize = 14, fontweight = 'bold');
            plt.legend(['f', 'z']);
        
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
    plt.plot(time[:-1], np.abs(np.diff(network.history_wo)).T, color = 'orange')
    plt.axis('off')
    plt.text(-len(time)*0.01, 0, '$|\dot w|$', fontweight = 'bold')



nsecs = 1440;
dt = 0.1;
simtime = np.arange(0, nsecs, dt)
simtime2 = np.arange(nsecs+10, 2*nsecs+10, dt)
amp = 1.3;
freq = 1/60;

amp_figAC = 1
freq_figAC= 1/60
ft = triangle(simtime, amp_figAC, freq_figAC)


network = create_network()
network_tested = test(network, simtime, ft, plot_test = False)
plot_activity(network, simtime, title = 'A - Spontaneous Activity')

network_trained = train(network, simtime, 3, ft)
plot_activity(network_trained, simtime, title = 'B - Learning')

network_tested = test(network_trained, simtime, ft, plot_test = False)
plot_activity(network_tested, simtime, title = 'C - Post Learning')


network = create_network();

amp_figD = [1, 0.5, 1/6, 1/3]
freq_figD = np.array([1, 2, 3, 4])/60
y_figD = sinusoid(simtime, amp_figD, freq_figD)
network_trained = train(network, simtime, 3, y_figD, plot_training=False)
network_tested = test(network_trained, simtime2, y_figD, title = 'D - Periodic');

network = create_network();
amp_figE = np.array([1, 1/4, 1/3, 1/3, 1/5, 1/10, 1/10, 1/12, 1/3, 1/2, 1/6, 1/2, -1/5, 1/4, 1/4, 1/10])/2
freq_figE = np.arange(1,17)/180
y_figE = sinusoid(simtime, amp_figE, freq_figE)
network_trained = train(network, simtime, 3, y_figE, plot_training=False);
network_tested = test(network_trained, simtime2, y_figE, title = 'E - Complicated periodic');

network = create_network();
amp_figF = amp_figD
freq_figF = freq_figD;
y_figF = sinusoid(simtime, amp_figF, freq_figF, noise = True, noise_intensity = 0.5)
network_trained = train(network, simtime, 3, y_figF, plot_training=False)
network_tested = test(network_trained, simtime2, y_figF, title = 'F - Extremely noisy target');

network = create_network();
amp_figG = 1
freq_figG = 2/60
y_figG = square(simtime, amp_figG, freq_figG, noise = True, noise_intensity=0.01);
network_trained = train(network, simtime, 3, y_figG, plot_training=False);
network_tested = test(network_trained, simtime2, y_figG, title = 'G - Discontinuous target');


network = create_network();
x_figH, y_figH, z_figH = lorenz_attractor(simtime/10, 1, 1, 1, Rayleigh_number = 28, Prandtl_number = 10)
network_trained = train(network, simtime, 3, y_figH[:-1]/10, plot_training=False)
network_tested = test(network_trained, simtime2, y_figH[:-1]/10, title = 'H - Lorenz attractor');

network = create_network();
amp_figI = [1]
freq_figI1 = [1/0.06]
y_figI1 = sinusoid(simtime, amp_figI, freq_figI1)
network_trained = train(network, simtime, 3, y_figI1, plot_training=False)
network_tested = test(network_trained, simtime2, y_figI1, title = r'J - Sine wave w/ period 6 $\tau$');

network = create_network();

freq_figI2 = [1/800/dt]
y_figI2 = sinusoid(simtime, amp_figI, freq_figI2)
network_trained = train(network, simtime, 3, y_figI2, plot_training=False)
network_tested = test(network_trained, simtime2, y_figI2, title = r'I2 - Sine wave w/ period 800 $\tau$');
