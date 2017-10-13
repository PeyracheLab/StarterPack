#!/usr/bin/env python

'''
	File name: main4.py
	Author: Guillaume Viejo
	Date created: 13/10/2017    
	Python Version: 3.5.2

The goal of this script is to plot the tuning curve of one neuron of ADn
The steps are
1) load the data
2) integrate the data in pandas frame which is a data helper
3) construct the angular bins
4) compute the mean firing rate for each bin
5) plot

To start, I use the same function as in main3.py to load the spiking activity
'''
import numpy as np
import pandas as pd
import neuroseries as nts
import os
from function import *
import scipy.io
data_directory 	= '../data/'

files 			= os.listdir(data_directory) 
generalinfo 	= scipy.io.loadmat(data_directory+'Mouse12-120806_GeneralInfo.mat')
shankStructure 	= loadShankStructure(generalinfo)
spikes,shank 	= loadSpikeData(data_directory+'Mouse12-120806_SpikeData.mat', shankStructure['thalamus'])
my_thalamus_neuron_index = list(spikes.keys())
hd_neuron_index = loadHDCellInfo(data_directory+'Mouse12-120806_HDCells.mat', my_thalamus_neuron_index)
hd_spikes = {}
for neuron in hd_neuron_index:
	hd_spikes[neuron] = spikes[neuron]

wake_ep 		= loadEpoch(data_directory, 'wake')
sleep_ep 		= loadEpoch(data_directory, 'sleep')
sws_ep 			= loadEpoch(data_directory, 'sws')
rem_ep 			= loadEpoch(data_directory, 'rem')


# Next step is to load the angular value at each time steps
