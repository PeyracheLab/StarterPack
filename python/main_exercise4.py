#!/usr/bin/env python

'''
	File name: main_exercise4.py
	Author: Guillaume Viejo
	Date created: 23/03/2020
	Python Version: 3.6

'''

###############################################################################
# 1. Import numpy, matplotlib, pandas and neuroseries
# Make sure to use the namespace np, pd and nts for numpy, pandas and neuroseries
###############################################################################
import scipy.stats

import numpy as np
from pylab import *
import pandas as pd
import neuroseries as nts
from functions import *

###############################################################################
# 2. Some data. Run it in the terminal after importing librairies
###############################################################################
neurons = np.arange(12)
n = 60
tuning_curves = pd.DataFrame.from_dict(
	{i : pd.Series(
			data = scipy.stats.norm.pdf(np.arange(-n/2, n/2), 0, 3)*np.random.uniform(100, 500),
			index = ((np.arange(n) + np.linspace(0, n, len(neurons)+1)[i])+n)%(n) 
		).sort_index() for i in neurons
	})
tuning_curves.index = pd.Index(np.linspace(0, 2*np.pi, n+1)[0:-1])

###############################################################################
# 3. Here is your lucky day. You have recorded 12 HD neurons together in the spinal cord
# Check their tuning curves by plotting them in a polar plot from the variable tuning_curves
# Don't hesitate to type the variable in your terminal to see how it's constructed
# Your plot should look like this : 
# https://www.dropbox.com/s/8mu7i1halsahyeq/figure_exercise4_1.png?dl=1
###############################################################################
figure()
subplot(111,projection = 'polar')
plot(tuning_curves)

###############################################################################
# 4. Supernova died and your rig burned. You have lost the spikes. Fortunately 
# you saved the angle of tracking of the animal during wake and sleep.
###############################################################################
fs = 10 # Hz
td = (20*60*fs, 10*60*fs, 15*60*fs) # minutes
angles = pd.Series(
	data = np.hstack((np.ones(td[0]),np.cumsum(np.random.randn(td[1])*0.5),np.zeros(td[2]))),
	index = np.arange(0, (1/fs)*np.sum(td), 1/fs))
angles = angles.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)	
angles = angles%(2*np.pi)
angles = nts.Tsd(t = angles.index.values, d = angles.values, time_units = 's')

sleep_ep = nts.IntervalSet(start = [0, (td[0]+td[1])/fs], end = [td[0]/fs, np.sum(td)/fs], time_units = 's')
wake_ep = nts.IntervalSet(start = [td[0]/fs], end = [(td[0]+td[1])/fs], time_units = 's')

###############################################################################
# 5. So you can generate some spikes and finish your phd with honors. Using the 
# variable tuning_curves, generate spikes for each neurons. Remember, the head-direction
# of the animal was tracked during wake and sleep.
# Start by defining the variables that will receive the spikes (i.e. a dict with lists inside
# accessible with the neuron number)
###############################################################################
spikes = {n:[] for n in neurons}
tmp = angles.restrict(wake_ep)

###############################################################################
# 6. The easiest way is to do a double loop. The first loop should be over 
# each time step of the wake epoch. Within each time step, you are gonna sample some 
# spike times according to the preferred direction of the neurons
# So start with the loop. Remember the epochs
###############################################################################
for t, a in zip(tmp.index, tmp.values):	

	###############################################################################
	# 7. Now the goal is to obtain firing rate as a function of the tunig curves
	# for each neuron based on the curent time step and angle(time step)
	# So something like : angle(time step) -> tuning_curves -> firing rate
	# Try to get it for all neurons at once. It's gonna speed up your code
	###############################################################################
	fr = tuning_curves.iloc[tuning_curves.index.get_loc(a, method = 'nearest')]
	# fr[fr < 0.001] = 0.0

	###############################################################################
	# 8. You have instantiated the variable fr as the firing rates for the time step t
	# for all neurons.
	# If you haven't noticed, the angle was sampled at 10Hz meaning that you want a 
	# number of spikes for a 100 ms window. You can use the np.random.poisson function
	# which will give you a spike count given a firing rate. Remember the window size 
	# to call the function
	# Hint : np.random.poisson([1,0,5]) will return you an array of the same size.
	# each element of [1,0,5] being firing rate of one neuron (i.e 1Hz, 0Hz, 5Hz) 
	###############################################################################
	n_spikes = np.random.poisson(fr.values*(1/fs))


	###############################################################################
	# 10. From np.random.poisson, you have an array called n_spikes for examples
	# telling you the number of spikes you should expect for each neuron given the time step
	# Since you want real spike times, 
	# you should assign to each spike a real timestamp uniformly distributed within 
	# your time window. You can use the function np.random.uniform called 
	# within the boundaries of the windows. Use a loop to go over each neuron and append
	# the spike times to the spikes list dictionnary defined outside of the two loops
	# Mind the time units
	###############################################################################
	for n in tuning_curves.columns:
		spikes[n].append(np.random.uniform(t, t + (1/fs) * 1e6, n_spikes[n]))
		
		

###############################################################################
# 11. You are now outside of the loop. For one neuron within you spike list dictionnary, 
# you have a list of spike times. You should concatenate them
# and make a nice array, put it in a Ts object.
# So loop over each element of your dictionnary and make a Ts oject
# Mind the time units again
###############################################################################
for  n in neurons: 
	spikes[n] = nts.Ts(np.hstack(spikes[n]))

###############################################################################
# 12. Now that you have some spikes for each neuron, you can see if they match their tuning curves
# Make a new tuning curve for each neuron by using the spikes you generated
# Remember to keep it tidy using the pandas dataframe structure
# You can use the function computeAngularTuningCurves in functions.py
# Mind the parameters to call the function
###############################################################################
tuning_curves2 = computeAngularTuningCurves(spikes, angles, wake_ep, nb_bins = 60, frequency = fs)

###############################################################################
# 7. Making a grid of subplot (i.e 3 times 4 for example), plot the first and 
# second tuning curve of each neuron
# Your plot should look like this :
# https://www.dropbox.com/s/lfdwewmc9tc802w/figure_exercise4_2.png?dl=1
###############################################################################
figure()
for i, n in enumerate(spikes):
	subplot(4,4,i+1)
	plot(tuning_curves[n])
	plot(tuning_curves2[n])


###############################################################################
# 8. Good but your boss wants to see a manifold decoding. 
# So do a manifold ring using UMAP (Yes you have to install a package now.
# It's called umap-learn)
# Usually a good manifold appears with 300 ms bins of wake
# You have to bin the spike trains of each neurons and downsample the angle (10Hz -> 3.3 Hz)
# with the same bins
# Start by defining the bins boundaries that spans the wake epoch in the time units you want
###############################################################################
bin_size = 300
bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1], bin_size)

###############################################################################
# 8. Bins the spike of each neuron (using a loop).
# Use the function np.histogram that takes as inputs your spike times and your bins boundaries
# You should obtain a pandas dataframe with index the timestep and columns the neuron number
# Each element of the dataframe should be equal to a spike count for one neuron over one time step
###############################################################################
spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = tuning_curves.columns)
for i in neurons:
	spks = spikes[i].as_units('ms').index.values
	spike_counts[i], _ = np.histogram(spks, bins)

###############################################################################
# 9. Transform your spike counts into the square root of a rate values by dividing by the bin size
# It appears that manifold projection works better with square root of firing rate
# Call your new variable rate_wake
###############################################################################
rate_wake = np.sqrt(spike_counts/(bin_size*1e-3))

###############################################################################
# 10. If you want to show the color code of your manifold you should bin the angle
# with the same bin boundaries array as your spike counts
# So bin the angle from a 100 ms time bins to a 300 ms time bins
# It's time to read the doc of np.digitize
# Make sure at the end of this step that your new wake angle is the same size
# as the rate array computed before
###############################################################################
wakangle = pd.Series(index = np.arange(len(bins)-1))
tmp = angles.groupby(np.digitize(angles.as_units('ms').index.values, bins)-1).mean()
wakangle.loc[wakangle.index] = tmp.iloc[wakangle.index]
wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)

###############################################################################
# 10. The next steps are given to help you.
# It imports UMAP to compute the manifold provided that you already installed it.
# If you called you new angles array as wakangle as a pandas series for exemple, 
# it will compute the RGB colors of each time point during wake
###############################################################################
from umap import UMAP
import sys
from matplotlib.colors import hsv_to_rgb

H = wakangle.values/(2*np.pi)
H[np.isnan(H)] = 0
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)


####################################################################################################################
# 11 The next step are given as well. It consists of smoothing slightly the squared root of the firign rate
# Make sure the variable is called rate_wake and that it is a pandas dataframe
# THe call to UMAP is given as well
####################################################################################################################
tmp = rate_wake.rolling(window=10,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1).values
ump = UMAP(n_neighbors = 100).fit_transform(tmp)

####################################################################################################################
# 12 Now you can plot the manifold. 
# If you type ump in your terminal, you will see a 2d array.
# Make a scatter plot of the first column versus the second column
# Use the array RGB as the color of each point
# You should obtain this :
# https://www.dropbox.com/s/ptfqw2c9l7v520j/figure_exercise4_3.png?dl=1
####################################################################################################################
figure()
scatter(ump[:,0].flatten(), ump[:,1].flatten(), c= RGB, marker = 'o', alpha = 0.5, linewidth = 0, s = 100)
show()

####################################################################################################################
# 13 Final step is to do the ring decoding
# You should use the function np.arctan2. Make sure you understand the doc
# The final plot should be the true angle versus the decoded angle from the ring
# Best is to make time series of it
# And remember, the manifold projection does not know where is the origin of the true angle
# and in which direction it's supposed to turn
# So there is an offset angle you should compute so that the two angles series match
# You should obtain something like this :
# 
####################################################################################################################
decodedangle = pd.Series(index = wakangle.index, data = np.arctan2(ump[:,1], ump[:,0]))
decodedangle += np.abs(wakangle.iloc[0] - decodedangle.iloc[0])
decodedangle += (2*np.pi)
decodedangle %= (2*np.pi)

figure()
plot(wakangle)
plot(decodedangle)
show()