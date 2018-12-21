import numpy as np

'''
Wrappers should be able to distinguish between raw data or matlab processed data
'''

def loadShankStructure(generalinfo):
	"""
	load Shank Structure from dictionnary 
	Only useful for matlab now
	Note : 
		TODO for raw data. 

	Args:
		generalinfo : dict        

	Returns: dict		    
	"""
	shankStructure = {}
	for k,i in zip(generalinfo['shankStructure'][0][0][0][0],range(len(generalinfo['shankStructure'][0][0][0][0]))):
		if len(generalinfo['shankStructure'][0][0][1][0][i]):
			shankStructure[k[0]] = generalinfo['shankStructure'][0][0][1][0][i][0]-1
		else :
			shankStructure[k[0]] = []
	
	return shankStructure	

def loadShankMapping(path):	
	import scipy.io	
	spikedata = scipy.io.loadmat(path)
	shank = spikedata['shank']
	return shank

def loadSpikeData(path, index):
	"""
	load the session_id_SpikeData.mat file that contains the times of the spikes
	Only useful for matlab now
	Note : 
		TODO for raw data. 

	Args:
		generalinfo : dict        

	Returns:
		dict, array
	"""	
	# units shoud be the value to convert in s 
	import scipy.io
	import neuroseries as nts
	spikedata = scipy.io.loadmat(path)
	shank = spikedata['shank'] - 1
	shankIndex = np.where(shank == index)[0]

	spikes = {}	
	for i in shankIndex:	
		spikes[i] = nts.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2], time_units = 's')

	a = spikes[0].as_units('s').index.values	
	if ((a[-1]-a[0])/60.)/60. > 20. : # VERY BAD		
		spikes = {}	
		for i in shankIndex:	
			spikes[i] = nts.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2]*0.0001, time_units = 's')

	return spikes, shank

def loadEpoch(path, epoch):
	"""
	load the epoch contained in path 
	Only useful for matlab now
	Note : 
		TODO for raw data. 

	Args:
		generalinfo : string, string

	Returns:
		intervalSet
	"""		
	import scipy.io
	import neuroseries as nts
	import os,sys
	sampling_freq = 1250	
	listdir = os.listdir(path)
	for f in listdir:
		if "BehavEpochs" in f:			
			behepochs = scipy.io.loadmat(path+f)		


	if epoch == 'wake':
		wake_ep = np.hstack([behepochs['wakeEp'][0][0][1],behepochs['wakeEp'][0][0][2]])
		return nts.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)

	elif epoch == 'sleep':
		sleep_pre_ep, sleep_post_ep = [], []
		if 'sleepPreEp' in behepochs.keys():
			sleep_pre_ep = behepochs['sleepPreEp'][0][0]
			sleep_pre_ep = np.hstack([sleep_pre_ep[1],sleep_pre_ep[2]])
			sleep_pre_ep_index = behepochs['sleepPreEpIx'][0]
		if 'sleepPostEp' in behepochs.keys():
			sleep_post_ep = behepochs['sleepPostEp'][0][0]
			sleep_post_ep = np.hstack([sleep_post_ep[1],sleep_post_ep[2]])
			sleep_post_ep_index = behepochs['sleepPostEpIx'][0]
		if len(sleep_pre_ep) and len(sleep_post_ep):
			sleep_ep = np.vstack((sleep_pre_ep, sleep_post_ep))
		elif len(sleep_pre_ep):
			sleep_ep = sleep_pre_ep
		elif len(sleep_post_ep):
			sleep_ep = sleep_post_ep						
		return nts.IntervalSet(sleep_ep[:,0], sleep_ep[:,1], time_units = 's')

	elif epoch == 'sws':
		for f in listdir:
			if 'sts.SWS' in f:
				sws = np.genfromtxt(path+f)/float(sampling_freq)
				return nts.IntervalSet.drop_short_intervals(nts.IntervalSet(sws[:,0], sws[:,1], time_units = 's'), 0.0)

			elif '-states.mat' in f:
				sws = scipy.io.loadmat(path+f)['states'][0]
				index = np.logical_or(sws == 2, sws == 3)*1.0
				index = index[1:] - index[0:-1]
				start = np.where(index == 1)[0]+1
				stop = np.where(index == -1)[0]
				return nts.IntervalSet.drop_short_intervals(nts.IntervalSet(start, stop, time_units = 's', expect_fix=True), 0.0)

	elif epoch == 'rem':
		for f in listdir:
			if 'sts.REM' in f:
				rem = np.genfromtxt(path+f)/float(sampling_freq)
				return nts.IntervalSet(rem[:,0], rem[:,1], time_units = 's').drop_short_intervals(0.0)

			elif '-states/m' in listdir:
				rem = scipy.io.loadmat(path+f)['states'][0]
				index = (rem == 5)*1.0
				index = index[1:] - index[0:-1]
				start = np.where(index == 1)[0]+1
				stop = np.where(index == -1)[0]
				return nts.IntervalSet(start, stop, time_units = 's', expect_fix=True).drop_short_intervals(0.0)

def loadHDCellInfo(path, index):
	"""
	load the session_id_HDCells.mat file that contains the index of the HD neurons
	Only useful for matlab now
	Note : 
		TODO for raw data. 

	Args:
		generalinfo : string, array

	Returns:
		array
	"""	
	# units shoud be the value to convert in s 	
	import scipy.io
	hd_info = scipy.io.loadmat(path)['hdCellStats'][:,-1]
	return np.where(hd_info[index])[0]