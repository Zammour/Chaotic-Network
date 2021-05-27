#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:06:43 2021

@author: zammour
"""

"""
% FORCE_INTERNAL_ALL2ALL.m
%
% This function generates the sum of 4 sine waves in figure 2D using the arcitecture of figure 1C (all-to-all
% connectivity) with the RLS learning rule.  The all-2-all connectivity allows a large optimization, in that we can
% maintain a single inverse correlation matrix for the network.  It's also not as a hard a learning problem as internal
% learning with sparse connectivity because there are no approximations of the eigenvectors of the correlation matrices,
% as there would be if this was sparse internal learning.  Note that there is no longer a feedback loop from the output
% unit.
%
% written by David Sussillo
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy import stats



linewidth = 3;
fontsize = 14;
fontweight = 'bold';

N = 1000;
p = 1.0;
g = 1.5;				# g greater than 1 leads to chaotic networks.
alpha = 1.0e-0;
nsecs = 1440;
dt = 0.1;
learn_every = 2;

scale = 1.0/np.sqrt(p*N);
rvs = stats.norm().rvs
M = sparse.random(N,N,p, data_rvs = rvs).todense()*g*scale


nRec2Out = N;
wo = np.zeros((nRec2Out,1))
dw = np.zeros((nRec2Out,1))

"""
disp(['   N: ', num2str(N)]);
disp(['   g: ', num2str(g)]);
disp(['   p: ', num2str(p)]);
disp(['   nRec2Out: ', num2str(nRec2Out)]);
disp(['   alpha: ', num2str(alpha,3)]);
disp(['   nsecs: ', num2str(nsecs)]);
disp(['   learn_every: ', num2str(learn_every)]);
"""

simtime = np.arange(0, nsecs, dt)
simtime_len = len(simtime)
simtime2 = np.arange(1*nsecs, 2*nsecs, dt)

amp = 0.7;
freq = 1/60;
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) + (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) + (amp/3.0)*np.sin(4.0*np.pi*freq*simtime);
ft = ft/1.5;

ft2 = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime2) + (amp/2.0)*np.sin(2.0*np.pi*freq*simtime2) + (amp/6.0)*np.sin(3.0*np.pi*freq*simtime2) + (amp/3.0)*np.sin(4.0*np.pi*freq*simtime2);
ft2 = ft2/1.5;

wo_len = np.zeros((1,simtime_len))  
M1_len = np.zeros((1,simtime_len));
rPr_len = np.zeros((1,simtime_len));
zt = np.zeros((1,simtime_len))
zpt = np.zeros((1,simtime_len))

x0 = 0.5*np.random.randn(N,1);
z0 = 0.5*np.random.randn(1,1);

x = x0; 
r = np.tanh(x);
xp = x0;
z = z0; 

plt.figure()
ti = 0
P = (1.0/alpha)*np.identity(nRec2Out)

for t in simtime:
    
    if ti % (nsecs/2) == 0:
    	plt.subplot(211)
    	plt.plot(simtime, ft.T, linewidth = linewidth, color = 'green');
    	plt.plot(simtime, zt.T, linewidth =  linewidth, color = 'red');
    	plt.title('training', fontsize = fontsize, fontweight = fontweight);
    	plt.legend(['f', 'z']);	
    	plt.xlabel('time', fontsize = fontsize, fontweight = fontweight);
    	plt.ylabel('f and z', fontsize = fontsize, fontweight = fontweight);
    	
    	plt.subplot(212)
    	plt.plot(simtime, wo_len.T, linewidth = linewidth);
    	plt.xlabel('time', fontsize = fontsize, fontweight = fontweight);
    	plt.ylabel('|w|', fontsize = fontsize, fontweight = fontweight);
    	plt.legend(['|w|']);
    	plt.pause(0.5);	
        
    	if ti + nsecs/2 < len(simtime):
            plt.clf()
        
    
    # sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt);
    r = np.tanh(x);
    z = wo.T*r;
    
    if ti % learn_every == 0:
    	# update inverse correlation matrix
    	k = P*r;
    	rPr = r.T*k;
    	c = 1.0/(1.0 + rPr);
    	P = P - k*(k*c).T;
    	
    	# update the error for the linear readout
    	e = z-ft[ti];
    	
    	# update the output weights
    	dw = -float(e)*k*c;
    	wo = wo + dw;
    	
    	# update the internal weight matrix using the output's error
    	M += dw.T
        

    # Store the output of the system.
    zt[0, ti] = z
    wo_len[0, ti] = np.sqrt(wo.T*wo)
    ti = ti+1
    

error_avg = np.sum(np.abs(zt-ft))/simtime_len;
print('Training MAE: ' +  str(error_avg))
print('Now testing... please wait.');    

# Now test. 
ti = 0;
for t in simtime:				# don't want to subtract time in indices 
    
    # sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M*(r*dt);
    r = np.tanh(x);
    z = wo.T*r;

    zpt[0, ti] = z;

    ti = ti+1;    

error_avg = sum(abs(zpt-ft2))/simtime_len;
print(['Testing MAE: ' + str(error_avg)]);


plt.figure();
plt.subplot(211);
plt.plot(simtime, ft.T, linewidth = linewidth, color = 'green');
plt.plot(simtime, zt.T, linewidth = linewidth, color =  'red');
plt.title('training', fontsize = fontsize, fontweight = fontweight);
plt.xlabel('time', fontsize = fontsize, fontweight =fontweight);
plt.ylabel('f and z', fontsize = fontsize, fontweight = fontweight);
plt.legend(['f', 'z']);


plt.subplot(212)
plt.plot(simtime2, ft2.T, linewidth = linewidth, color = 'green'); 
plt.axis('tight')
plt.plot(simtime2, zpt.T, linewidth = linewidth, color = 'red');
plt.axis('tight')
plt.title('simulation', fontsize = fontsize, fontweight = fontweight);
plt.xlabel('time', fontsize =fontsize, fontweight = fontweight);
plt.ylabel('f and z', fontsize = fontsize, fontweight = fontweight);
plt.legend(['f', 'z']);
	

