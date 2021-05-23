#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:46:11 2021

@author: zammour
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy import stats

g_GG=1.5;
g_Gz=1;
g_GF=0;
g_FF=1.2;
g_FG=1;

N_G=10;
N_F=9;
p_GG=0.1;
p_Z=1;
p_FG=0.025;
p_FF=0.25;
p_GF=0.25;
T=14400;
np.random.seed(1);

w=np.zeros(N_G);
w_nonzero=np.random.normal(0,1/(p_Z*N_G),(N_G));
for i in range(N_G):
#    for j in range(N_G):

        w[i]=np.random.choice([0,w_nonzero[i]]);
  # w[i]=w_nonzero[i];
   
#random.shuffle(w);

J_GG=np.random.normal(0,1/(p_Z*N_G),(N_G,N_G));
J_Gz=np.random.uniform(-1,1,(N_G));
J_FG=np.random.normal(0,1/(p_FG*N_G),(N_F,N_G));
J_FF=np.random.normal(0,1/(p_FF*N_F),(N_F,N_F));
J_GF=np.random.normal(0,1/(p_GF*N_F),(N_G,N_F));
tau=10;

x0=0.1

#N_I=0, so the last term is omitted
def g_GG_simul(N_G=N_G,N_F=N_F,tau=tau,T=T):
    yt=np.ones(N_F);

    xt=np.ones(N_G);
    r=np.ones(N_G);
    z=np.ones(T);
   # noise=sig*np.random.normal(0, 1, size=r)
    #xt.append(x0)
    #loop for z
    for t in range(T):        

    #initialize sigmas in the two formula
        first_sigma_x=1;
        first_sigma_y=1;
        second_sigma_y=1;
        second_sigma_x=1;

        if t>0:
           #loop for xi and ri
           for i in range(N_G):   
               #loop for 1st sigma x
               for j in range(N_G):
                   first_sigma_x=first_sigma_x+g_GG*J_GG[i][j]*np.tanh(xt[j]);
                   #loop for ya and 2nd sigma x
                   for n in range(N_F):
                       #loop for 2nd sigma y
                       for m in range(N_G):
                           second_sigma_y=second_sigma_y+g_FG*J_FG[n][m]*np.tan(xt[m]);
                       #loop for 1st sigma y    
                       for k in range(N_F):
                           first_sigma_y=first_sigma_y+g_FF*J_FF[n][k]*np.tanh(yt[k]);
                           
                       yt[n]=yt[n]+(-yt[n]+first_sigma_y+second_sigma_y)/tau;

                       second_sigma_x=second_sigma_x+np.dot(g_GF*J_GF[i][n],np.tanh(yt[n]));
  
    


               xt[i]=xt[i]+(-xt[i]+first_sigma_x+g_Gz*J_Gz[i]*z[t-1]+second_sigma_x)/tau;
               
               r[i]=(np.tanh(xt[i]));
            

        z[t]=(np.dot(w.T,r)); 

    return z;

z0=g_GG_simul();
plt.plot(z0);

linewidth = 3;
fontsize = 14;
fontweight = 'bold';

N = 2000;
p = 0.1;
g = 1.5;				# g greater than 1 leads to chaotic networks.
alpha = 1.0;
nsecs = 1440;
dt = 0.1;
learn_every = 2;

scale = 1.0/np.sqrt(p*N);
rvs = stats.norm().rvs
M = sparse.random(N,N,p, data_rvs = rvs).todense()*g*scale

nRec2Out = int(N/2)
nRec2Control = int(N/2)
neurons_recorded = 10

"""
% Allow output and control units to start with different ICs.  If you set beta greater than zero, then y will look
% different than z but still drive the network with the appropriate frequency content (because it will be driven with
% z).  A value of beta = 0 shows that the learning rules produce extremely similar signals for both z(t) and y(t),
% despite having no common pre-synaptic inputs.  Keep in mind that the vector norm of the output weights is 0.1-0.2 when
% finished, so if you make beta too big, things will eventually go crazy and learning won't converge.
%beta = 0.1;	
"""

beta = 0.0
wo = beta*np.random.randn(nRec2Out,1)/np.sqrt(N/2)		# synaptic strengths from internal pool to output unit
dwo = np.zeros((nRec2Out,1))
wc = beta*np.random.randn(nRec2Control, 1)/np.sqrt(N/2) 	# synaptic strengths from internal pool to control unit
dwc = np.zeros((nRec2Control, 1))

wf = 2.0*(np.random.rand(N,1)-0.5) 	# the feedback now comes from the control unit as opposed to the output

# Deliberatley set the pre-synaptic neurons to nonoverlapping between the output and control units.
zidxs = np.arange(round(N/2))
yidxs = np.arange(round(N/2), N)	 

print('   N: ' + str(N))
print('   g: '+str(g))
print('   p: '+ str(p))
print('   nRec2Out: '+ str(nRec2Out))
print('   nRec2Control: '+ str(nRec2Control))
print('   alpha: '+ str(alpha))
print('   nsecs: '+ str(nsecs))
print('   learn_every: '+ str(learn_every))


simtime = np.arange(0, nsecs, dt)
simtime_len = len(simtime)
simtime2 = np.arange(1*nsecs, 2*nsecs, dt)

amp = 1.3;
freq = 1/60;
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) + (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) + (amp/3.0)*np.sin(4.0*np.pi*freq*simtime);
ft = ft/1.5;

ft2 = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime2) + (amp/2.0)*np.sin(2.0*np.pi*freq*simtime2) + (amp/6.0)*np.sin(3.0*np.pi*freq*simtime2) + (amp/3.0)*np.sin(4.0*np.pi*freq*simtime2);
ft2 = ft2/1.5;

wo_len = np.zeros((1,simtime_len))  
wc_len = np.zeros((1,simtime_len))
zt = np.zeros((1,simtime_len))
zt= z0
yt = np.zeros((1,simtime_len))
zpt = np.zeros((1,simtime_len))
ypt = np.zeros((1,simtime_len))

x0 = 0.5*np.random.randn(N,1);
z0 = 0.5*np.random.randn(1,1);
y0 = 0.5*np.random.randn(1,1);

x = x0; 
r = np.tanh(x);
z = z0; 
y = y0;

plt.figure()
ti = 0;
Pz = (1.0/alpha)*np.identity(nRec2Out);
Py = (1.0/alpha)*np.identity(nRec2Control);


neurons = np.zeros((neurons_recorded,1))
pick_random_neurons = np.random.randint(0, N/2, size = neurons_recorded)


def Force_feedback(network, time):
    
    weights = network['weights']
    
    
    for t in time:
        
        if ti % learn_every == 0:
            kz = Pz*rz
            rPrz = rz.T*kz
            cz = 1.0/(1.0 + rPrz)
            Pz = Pz - kz*(kz*cz).T   
        	# update the error for the linear readout
            e = z-ft[ti]
        	# update the output weights
            dwo = -float(e)*kz*cz
            wo += dwo
        
        	# update inverse correlation matrix for the control unit
            ky = Py*ry
            rPry = ry.T*ky
            cy = 1.0/(1.0 + rPry)
            Py -= ky*(ky*cy).T  
        
        	### NOTE WE USE THE OUTPUT'S ERROR ###
        	# update the output weights
            dwc = -float(e)*ky*cy
            wc += dwc
            
            zt[0,ti] = z
            yt[0,ti] = y
            wo_len[0,ti] = np.sqrt(np.dot(wo.T, wo))	
            wc_len[0,ti] = np.sqrt(np.dot(wc.T, wc))
    
            neurons = np.append(neurons, np.array(rz[pick_random_neurons]), axis = 1)
    
            ti += 1





for t in simtime:
    
    if ti%(nsecs/2) == 0:
    	print('time: ' + str(t) +'.');
    	plt.subplot(211);
    	plt.plot(simtime, ft.T, linewidth = linewidth, color = 'green');
    	plt.plot(simtime, zt.T, linewidth = linewidth, color = 'red');
    	plt.plot(simtime, yt.T, linewidth = linewidth, color = 'magenta'); 
    	plt.title('training', fontsize = fontsize, fontweight = fontweight);
    	plt.xlabel('time', fontsize = fontsize, fontweight = fontweight);
    	plt.ylabel('f, z and y', fontsize = fontsize, fontweight = fontweight);
    	plt.legend(['f', 'z', 'y']);
    	
    	plt.subplot(212);
    	plt.plot(simtime, wo_len.T, linewidth = linewidth); 
    	plt.plot(simtime, wc_len.T, linewidth = linewidth, color = 'green'); 
    	plt.xlabel('time', fontsize = fontsize, fontweight = fontweight);
    	plt.ylabel('|w_o|, |w_c|', fontsize = fontsize, fontweight = fontweight);
    	plt.legend(['|w_o|', '|w_c|']);	
    	plt.pause(0.5)
        
    	if ti + nsecs/2 < len(simtime):
            plt.clf()
    
    # sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt) + wf*(y*dt); # note the y here.
    r = np.tanh(x);
    rz = r[zidxs]   		# the neurons that project to the output
    ry = r[yidxs]			# the neurons that project to the control unit
    z = wo.T*rz;
    y = wc.T*ry;
    
    if ti % learn_every == -1:
    	# update inverse correlation matrix for the output unit
    	kz = Pz*rz;
    	rPrz = rz.T*kz;
    	cz = 1.0/(1.0 + rPrz);
    	Pz = Pz - kz*(kz*cz).T;    
    	# update the error for the linear readout
    	e = z-ft[ti];
    	# update the output weights
    	dwo = -float(e)*kz*cz;
    	wo += dwo;
    
    	# update inverse correlation matrix for the control unit
    	ky = Py*ry;
    	rPry = ry.T*ky;
    	cy = 1.0/(1.0 + rPry);
    	Py -= ky*(ky*cy).T;    
    
    	### NOTE WE USE THE OUTPUT'S ERROR ###
    	# update the output weights
    	dwc = -float(e)*ky*cy;
    	wc += dwc;
        
    
    # Store the output of the system.
    zt[0,ti] = z;
    yt[0,ti] = y;
    wo_len[0,ti] = np.sqrt(np.dot(wo.T, wo));	
    wc_len[0,ti] = np.sqrt(np.dot(wc.T, wc));
    
    neurons = np.append(neurons, np.array(rz[pick_random_neurons]), axis = 1)
    
    ti += 1;	
        

error_avg = np.sum(np.abs(zt-ft))/simtime_len;
print('Training MAE: ' + str(error_avg));    
print('Now testing... please wait.');    

# Now test. 
ti = 0;

neurons_test = np.zeros((neurons_recorded,1))

for t in simtime:				# don't want to subtract time in indices
    
    # sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt) + wf*(y*dt); # note the y here.
    

    r = np.tanh(x);
    rz = r[zidxs];			#the neurons that project to the output
    ry = r[yidxs];			#the neurons that project to the control unit
    z = wo.T*rz;
    y = wc.T*ry;
    
    zpt.T[ti] = z;
    ypt.T[ti] = y;
    
    neurons_test = np.append(neurons_test, np.array(rz[pick_random_neurons]), axis = 1)
    
    ti += 1;    

error_avg = np.sum(np.abs(zpt-ft2))/simtime_len
print('Testing MAE: ' + str(error_avg))


plt.figure()
plt.subplot(211)
plt.plot(simtime, ft.T, linewidth = linewidth, color = 'green');
plt.plot(simtime, zt.T, linewidth = linewidth, color = 'red');
plt.plot(simtime, yt.T, linewidth = linewidth, color = 'magenta'); 
plt.title('training', fontsize = fontsize, fontweight =  fontweight);
plt.xlabel('time', fontsize =  fontsize, fontweight = fontweight);
plt.axis('tight')
plt.ylabel('f, z and y', fontsize = fontsize, fontweight = fontweight);
plt.legend(['f', 'z', 'y']);


plt.subplot(212)
plt.plot(simtime2, ft.T, linewidth = linewidth, color = 'green'); 
plt.plot(simtime2, zpt.T, linewidth = linewidth, color = 'red');
plt.plot(simtime2, ypt.T, linewidth = linewidth, color = 'magenta'); 
plt.axis('tight')
plt.title('simulation', fontsize = fontsize, fontweight = fontweight);
plt.xlabel('time', fontsize = fontsize, fontweight = fontweight);
plt.ylabel('f, z and y', fontsize = fontsize, fontweight = fontweight);
plt.legend(['f', 'z', 'y'])




def plot_activity(time, z, dw, neurons, title = None):
    
    nrows = 2 + neurons.shape[0]
    
    plt.figure()
    if title != None: plt.suptitle(title, fontweight='bold')
    plt.subplot(nrows, 1, 1)
    plt.plot(time, z, color = 'r')
    plt.axis('off')

    for row in range(neurons.shape[0]):
        plt.subplot(nrows, 1, 2 + row)
        plt.plot(time, neurons[row,:])
        plt.axis('off')    

    plt.subplot(nrows, 1, nrows)
    plt.plot(time[:-1], np.abs(dw), color = 'orange')
    plt.axis('off')
    plt.text(-len(dw)*0.01, 0, '$|\dot w|$', fontweight = 'bold')

    
plot_activity(simtime, zt.T, np.diff(wo_len).T, neurons[:,1:], title = 'Learning')
plot_activity(simtime, zpt.T, np.zeros(len(simtime)-1), neurons_test[:, 1:], title = 'Post_learning')