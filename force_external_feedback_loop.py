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

"""
% Allow output and control units to start with different ICs.  If you set beta greater than zero, then y will look
% different than z but still drive the network with the appropriate frequency content (because it will be driven with
% z).  A value of beta = 0 shows that the learning rules produce extremely similar signals for both z(t) and y(t),
% despite having no common pre-synaptic inputs.  Keep in mind that the vector norm of the output weights is 0.1-0.2 when
% finished, so if you make beta too big, things will eventually go crazy and learning won't converge.
%beta = 0.1;	
"""			

beta = 0.2;
wo = beta*np.random.randn(nRec2Out,1)/np.sqrt(N/2);			# synaptic strengths from internal pool to output unit
dwo = np.zeros((nRec2Out,1));
wc = beta*np.random.randn(nRec2Control, 1)/np.sqrt(N/2);		# synaptic strengths from internal pool to control unit
dwc = np.zeros((nRec2Control, 1));

wf = 2.0*(np.random.rand(N,1)-0.5);		# the feedback now comes from the control unit as opposed to the output

# Deliberatley set the pre-synaptic neurons to nonoverlapping between the output and control units.
zidxs = np.arange(round(N/2))
yidxs = np.arange(round(N/2), N)	 

print('   N: ' + str(N));
print('   g: '+str(g));
print('   p: '+ str(p));
print('   nRec2Out: '+ str(nRec2Out));
print('   nRec2Control: '+ str(nRec2Control));
print('   alpha: '+ str(alpha));
print('   nsecs: '+ str(nsecs));
print('   learn_every: '+ str(learn_every));

pre_training=np.arange(0, nsecs/2, dt);
simtime = np.arange(0, nsecs, dt);
simtime_len = len(simtime);
simtime2 = np.arange(1*nsecs, 2*nsecs, dt);
post_training=np.arange(0, nsecs/2, dt);


amp = 1.3;
freq = 1/60;
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) + (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) + (amp/3.0)*np.sin(4.0*np.pi*freq*simtime);
ft = ft/1.5;

ft2 = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime2) + (amp/2.0)*np.sin(2.0*np.pi*freq*simtime2) + (amp/6.0)*np.sin(3.0*np.pi*freq*simtime2) + (amp/3.0)*np.sin(4.0*np.pi*freq*simtime2);
ft2 = ft2/1.5;

wo_len = np.zeros((1,simtime_len))  
wc_len = np.zeros((1,simtime_len))
zpre = np.zeros((1,simtime_len))
ypre = np.zeros((1,simtime_len))
zt = np.zeros((1,simtime_len))
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



#pre-training

n1_pre = np.array([])
n10_pre = np.array([])
n50_pre = np.array([])
n100_pre = np.array([])
n200_pre = np.array([])
n500_pre = np.array([])
n600_pre = np.array([])
n700_pre = np.array([])
n800_pre = np.array([])
n900_pre = np.array([])

for t in simtime:				# don't want to subtract time in indices
     
    if ti%(nsecs/2) == 0:
    	print('time: ' + str(t) +'.');
      #  plt.figure('fig.training');
    	plt.subplot(211);
    	plt.plot(simtime, ft.T, linewidth = linewidth, color = 'green');
    	plt.plot(simtime, zpre.T, linewidth = linewidth, color = 'red');
    	plt.plot(simtime, ypre.T, linewidth = linewidth, color = 'magenta'); 
    	plt.title('pre-train', fontsize = fontsize, fontweight = fontweight);
    	plt.xlabel('time', fontsize = fontsize, fontweight = fontweight);
    	plt.ylabel('f, z and y', fontsize = fontsize, fontweight = fontweight);
    	plt.legend(['f', 'z', 'y']);
    	
    	plt.subplot(212);
    	plt.plot(simtime, wo_len.T, linewidth = linewidth); 
    	plt.plot(simtime, wc_len.T, linewidth = linewidth, color = 'green'); 
    	plt.xlabel('time', fontsize = fontsize, fontweight = fontweight);
    	plt.ylabel('|w_o|, |w_c|', fontsize = fontsize, fontweight = fontweight);
    	plt.legend(['|w_o|', '|w_c|']);	
    
    	plt.pause(0.5);	
    
    # sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt) + wf*(y*dt); # note the y here.
    
    n1_pre = np.append(n1_pre, x[1])
    n10_pre = np.append(n10_pre, x[50])
    n50_pre = np.append(n50_pre, x[1])
    n100_pre = np.append(n100_pre, x[100])
    n200_pre = np.append(n200_pre, x[20])
    n500_pre = np.append(n500_pre, x[500])
    n600_pre = np.append(n600_pre, x[600])
    n700_pre = np.append(n700_pre, x[700])
    n800_pre = np.append(n800_pre, x[800])
    n900_pre = np.append(n900_pre, x[900])

    r = np.tanh(x);
    rz = r[zidxs];			#the neurons that project to the output
    ry = r[yidxs];			#the neurons that project to the control unit
    z = wo.T*rz;
    y = wc.T*ry;
    
    zpre.T[ti] = z;
    ypre.T[ti] = y;
    wo_len[0,ti] = np.sqrt(np.dot(wo.T, wo));	
    wc_len[0,ti] = np.sqrt(np.dot(wc.T, wc));
    ti += 1;    

error_avg_pre = np.sum(np.abs(zpre-ft2))/simtime_len
print('Pre-training MAE: ' + str(error_avg_pre))

# FORCE learning
n1_train = np.array([])
n10_train = np.array([])
n50_train = np.array([])
n100_train = np.array([])
n200_train = np.array([])
n500_train = np.array([])
n600_train = np.array([])
n700_train = np.array([])
n800_train = np.array([])
n900_train = np.array([])



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
    
    	plt.pause(0.5);	
    
    # sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt) + wf*(y*dt); # note the y here.
    r = np.tanh(x);
    rz = r[zidxs]   		# the neurons that project to the output
    ry = r[yidxs]			# the neurons that project to the control unit
    z = wo.T*rz;
    y = wc.T*ry;
    
    if ti % learn_every == 0:
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
    
    ti += 1;	
    
    n1_train = np.append(n1_train, x[1])
    n10_train = np.append(n50_train, x[50])
    n50_train = np.append(n1_train, x[1])
    n100_train = np.append(n100_train, x[100])
    n200_train = np.append(n200_train, x[20])
    n500_train = np.append(n500_train, x[500])
    n600_train = np.append(n600_train, x[600])
    n700_train = np.append(n700_train, x[700])
    n800_train = np.append(n800_train, x[800])
    n900_train = np.append(n900_train, x[900])



error_avg = np.sum(np.abs(zt-ft))/simtime_len;
print('Training MAE: ' + str(error_avg));    
print('Now testing... please wait.');    

# Now test. 
ti = 0;
n1_test = np.array([])
n10_test = np.array([])
n50_test = np.array([])
n100_test = np.array([])
n200_test = np.array([])
n500_test = np.array([])
n600_test = np.array([])
n700_test = np.array([])
n800_test = np.array([])
n900_test = np.array([])

for t in simtime:				# don't want to subtract time in indices
    
    # sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt) + wf*(y*dt); # note the y here.
    
    n1_test = np.append(n1_test, x[1])
    n10_test = np.append(n50_test, x[50])
    n50_test = np.append(n1_test, x[1])
    n100_test = np.append(n100_test, x[100])
    n200_test = np.append(n200_test, x[20])
    n500_test = np.append(n500_test, x[500])
    n600_test = np.append(n600_test, x[600])
    n700_test = np.append(n700_test, x[700])
    n800_test = np.append(n800_test, x[800])
    n900_test = np.append(n900_test, x[900])

    r = np.tanh(x);
    rz = r[zidxs];			#the neurons that project to the output
    ry = r[yidxs];			#the neurons that project to the control unit
    z = wo.T*rz;
    y = wc.T*ry;
    
    zpt.T[ti] = z;
    ypt.T[ti] = y;
    
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
plt.plot(simtime2, ft2.T, linewidth = linewidth, color = 'green'); 
plt.plot(simtime2, zpt.T, linewidth = linewidth, color = 'red');
plt.plot(simtime2, ypt.T, linewidth = linewidth, color = 'magenta'); 
plt.axis('tight')
plt.title('simulation', fontsize = fontsize, fontweight = fontweight);
plt.xlabel('time', fontsize = fontsize, fontweight = fontweight);
plt.ylabel('f, z and y', fontsize = fontsize, fontweight = fontweight);
plt.legend(['f', 'z', 'y'])
	

