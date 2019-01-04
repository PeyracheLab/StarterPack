#!/usr/bin/env python

'''	
	Author: Guillaume Viejo
	Date created: 03/01/2019
	Python Version: 3.5.2

This scripts will show you how to compute the auto and cross-correlograms of neurons
The data should be found in StarterPack/data_raw/A1110-180621/

The main function crossCorr is already written in StarterPack/python/functions.py

'''
import numpy as np
import pandas as pd
import neuroseries as nts
import os
from scipy.io import loadmat
from pylab import *


# First let's get some spikes
data_directory = '../data_raw2/A1110-180621'
from wrappers import loadSpikeData
spikes, shank = loadSpikeData(data_directory)

# Let's restrict the spikes to the wake episode
from wrappers import makeEpochs, loadEpoch
makeEpochs(data_directory, ['sleep', 'wake', 'sleep', 'wake'], file='Epoch_TS.csv')
wake_ep = loadEpoch(data_directory, 'wake')

# Let's make the autocorrelogram of the first neuron
neuron_0 = spikes[0]

# restricted for wake
neuron_0 = neuron_0.restrict(wake_ep)

# transforming the times in millisecond
neuron_0 = neuron_0.as_units('ms')

# and extracting the index to feed the function crossCorr
neuron_0_t = neuron_0.index.values

# Let's say you want to compute the autocorr with 5 ms bins
binsize = 5
# with 200 bins
nbins = 400

# Now we can call the function crossCorr
from functions import crossCorr
autocorr_0 = crossCorr(neuron_0_t, neuron_0_t, binsize, nbins)

# The corresponding times can be computed as follow 
times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2

# Let's make a time series
autocorr_0 = pd.Series(index = times, data = autocorr_0)

# We need to replace the value at 0
autocorr_0.loc[0] = 0.0

# Let's plot it
figure()
plot(autocorr_0)
show()

# The autocorr_0 is not normalized.
# To normalize, you need to divide by the mean firing rate
mean_fr_0 = len(neuron_0)/wake_ep.tot_length('s')
autocorr_0 = autocorr_0 / mean_fr_0

# Let's plot it again
figure()
plot(autocorr_0)
show()


# Now let's make a function to compute all autocorrs
def compute_AutoCorrs(spks, ep, binsize = 5, nbins = 400):
	# First let's prepare a pandas dataframe to receive the data
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2	
	autocorrs = pd.DataFrame(index = times, columns = np.arange(len(spikes)))
	firing_rates = pd.Series(index = np.arange(len(spikes)))

	# Now we can iterate over the dictionnary of spikes
	for i in spks:
		# First we extract the time of spikes in ms during wake
		spk_time = spks[i].restrict(ep).as_units('ms').index.values
		# Calling the crossCorr function
		autocorrs[i] = crossCorr(spk_time, spk_time, binsize, nbins)
		# Computing the mean firing rate
		firing_rates[i] = len(spk_time)/wake_ep.tot_length('s')

	# We can divide the autocorrs by the firing_rates
	autocorrs = autocorrs / firing_rates

	# And don't forget to replace the 0 ms for 0
	autocorrs.loc[0] = 0.0
	return autocorrs, firing_rates

# We can call the function
autocorrs, firing_rates = compute_AutoCorrs(spikes, wake_ep)

# Let's plot all the autocorrs
figure()
plot(autocorrs)
xlabel("Time lag (ms)")
show()

