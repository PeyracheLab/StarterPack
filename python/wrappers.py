import numpy as np
import sys,os
import scipy.io
import neuroseries as nts
import pandas as pd
import scipy.signal

'''
Wrappers should be able to distinguish between raw data or matlab processed data
'''

def loadSpikeData(path, index=None, fs = 20000):
	"""
	if the path contains a folder named /Analysis, 
	the script will look into it to load either
		- SpikeData.mat saved from matlab
		- SpikeData.h5 saved from this same script
	if not, the res and clu file will be loaded 
	and an /Analysis folder will be created to save the data
	Thus, the next loading of spike times will be faster
	Notes :
		If the frequency is not givne, it's assumed 20kH
	Args:
		path : string

	Returns:
		dict, array	
	"""	
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()    
	new_path = os.path.join(path, 'Analysis/')
	if os.path.exists(new_path):
		new_path	= os.path.join(path, 'Analysis/')
		files		= os.listdir(new_path)
		if 'SpikeData.mat' in files:
			spikedata 	= scipy.io.loadmat(new_path+'SpikeData.mat')
			shank 		= spikedata['shank'] - 1
			if index is None:
				shankIndex 	= np.arange(len(shank))
			else:
				shankIndex 	= np.where(shank == index)[0]
			spikes 		= {}	
			for i in shankIndex:	
				spikes[i] 	= nts.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2], time_units = 's')
			a 			= spikes[0].as_units('s').index.values	
			if ((a[-1]-a[0])/60.)/60. > 20. : # VERY BAD		
				spikes 		= {}	
				for i in shankIndex:
					spikes[i] 	= nts.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2]*0.0001, time_units = 's')
			return spikes, shank
		elif 'SpikeData.h5' in files:			
			final_path = os.path.join(new_path, 'SpikeData.h5')			
			spikes = pd.read_hdf(final_path, mode='r')
			# Returning a dictionnary | can be changed to return a dataframe
			toreturn = {}
			for i,j in spikes:
				toreturn[j] = nts.Ts(t=spikes[(i,j)].replace(0,np.nan).dropna().index.values, time_units = 's')
			shank = spikes.columns.get_level_values(0).values[:,np.newaxis]
			return toreturn, shank
			
		else:			
			print("Couldn't find any SpikeData file in "+new_path)
			print("If clu and res files are present in "+path+", a SpikeData.h5 is going to be created")

	# Creating /Analysis/ Folder here if not already present
	if not os.path.exists(new_path): os.makedirs(new_path)
	files = os.listdir(path)
	clu_files 	= np.sort([f for f in files if 'clu' in f and f[0] != '.'])
	res_files 	= np.sort([f for f in files if 'res' in f and f[0] != '.'])
	clu1 		= np.sort([int(f.split(".")[-1]) for f in clu_files])
	clu2 		= np.sort([int(f.split(".")[-1]) for f in res_files])
	if len(clu_files) != len(res_files) or not (clu1 == clu2).any():
		print("Not the same number of clu and res files in "+path+"; Exiting ...")
		sys.exit()
	count = 0	
	spikes = pd.DataFrame()	
	for i in range(len(clu_files)):
		clu = np.genfromtxt(os.path.join(path,clu_files[i]))[1:]
		res = np.genfromtxt(os.path.join(path,res_files[i]))
		tmp = np.unique(clu).astype(int)		
		idx_clu = tmp[tmp>1]
		idx_col = np.arange(count, count+len(idx_clu))
		for j, k in zip(idx_clu, idx_col):
			tmp2 = pd.DataFrame(index=res[clu==j]/fs, data = k+1, columns = pd.MultiIndex.from_tuples([(i,k)]))
			spikes = pd.concat([spikes, tmp2], axis = 1)			
		count+=len(idx_clu)
	spikes = spikes.fillna(0)
	spikes = spikes.astype(np.int8)

	# Saving SpikeData.h5
	final_path = os.path.join(new_path, 'SpikeData.h5')
	spikes.columns.set_names(['shank', 'neuron'], inplace=True)	
	spikes.to_hdf(final_path, key='spikes', mode='w')

	# Returning a dictionnary
	toreturn = {}
	for i,j in spikes:
		toreturn[j] = nts.Ts(t=spikes[(i,j)].replace(0,np.nan).dropna().index.values, time_units = 's')

	shank = spikes.columns.get_level_values(0).values[:,np.newaxis]

	return toreturn, shank

def loadXML(path):
	"""
	path should be the folder session containing the XML file
	Function returns :
		1. the number of channels
		2. the sampling frequency of the dat file or the eeg file depending of what is present in the folder
			eeg file first if both are present or both are absent
		3. the mappings shanks to channels as a dict
	Args:
		path : string

	Returns:
		int, int, dict
	"""
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	listdir = os.listdir(path)
	xmlfiles = [f for f in listdir if f.endswith('.xml')]
	if not len(xmlfiles):
		print("Folder contains no xml files; Exiting ...")
		sys.exit()
	new_path = os.path.join(path, xmlfiles[0])
	
	from xml.dom import minidom	
	xmldoc 		= minidom.parse(new_path)
	nChannels 	= xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
	fs_dat 		= xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
	fs_eeg 		= xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data	
	if os.path.splitext(xmlfiles[0])[0] +'.dat' in listdir:
		fs = fs_dat
	elif os.path.splitext(xmlfiles[0])[0] +'.eeg' in listdir:
		fs = fs_eeg
	else:
		fs = fs_eeg
	shank_to_channel = {}
	groups 		= xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')
	for i in range(len(groups)):
		shank_to_channel[i] = np.sort([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
	return int(nChannels), int(fs), shank_to_channel

def downsampleDatFile(path, n_channels, fs):
	"""
	downsample .dat file to .eeg 1/16 (20000 -> 1250 Hz)
	
	Since .dat file can be very big, the strategy is to load one channel at the time,
	downsample it, and free the memory.

	Args:
		path: string
		n_channel: int
		fs: int
	Return: 
		none
	"""	
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	listdir 	= os.listdir(path)
	datfile 	= [f for f in listdir if f.endswith('.dat')]
	if not len(datfile):
		print("Folder contains no xml files; Exiting ...")
		sys.exit()
	new_path = os.path.join(path, datfile[0])

	f 			= open(new_path, 'rb')
	startoffile = f.seek(0, 0)
	endoffile 	= f.seek(0, 2)
	bytes_size 	= 2
	n_samples 	= int((endoffile-startoffile)/n_channels/bytes_size)
	duration 	= n_samples/fs
	f.close()

	chunksize 	= 100000
	eeg 		= np.zeros((int(n_samples/16),n_channels))

	for n in range(n_channels):		
		# Loading
		rawchannel = np.zeros(n_samples, np.int16)		
		count = 0
		while count < n_samples:
			f 			= open(new_path, 'rb')
			seekstart 	= count*n_channels*bytes_size
			f.seek(seekstart)
			block 		= np.fromfile(f, np.int16, n_channels*np.minimum(chunksize, n_samples-count))
			f.close()
			block 		= block.reshape(np.minimum(chunksize, n_samples-count), n_channels)
			rawchannel[count:count+np.minimum(chunksize, n_samples-count)] = np.copy(block[:,n])
			count 		+= chunksize
		# Downsampling		
		eeg[:,n] 	= scipy.signal.resample_poly(rawchannel, 1, 16)
		del rawchannel		
	
	# Saving
	eeg_path 	= os.path.join(path, os.path.splitext(datfile[0])[0]+'.eeg')
	with open(eeg_path, 'wb') as f:
		eeg.astype('int16').tofile(f)
		
	return

def makeEpochs(path, order, file = None, start=None, end = None, time_units = 's'):
	"""
	The pre-processing pipeline should spit out a csv file containing all the successive epoch of sleep/wake
	This function will load the csv and write neuroseries.IntervalSet of wake and sleep in /Analysis/BehavEpochs.h5
	If no csv exists, it's still possible to give by hand the start and end of the epochs
	Notes:
		The function assumes no header on the csv file
	Args:
		path: string
		order: list
		file: string
		start: list/array (optional)
		end: list/array (optional)
		time_units: string (optional)
	Return: 
		none
	"""		
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()	
	if file:
		listdir 	= os.listdir(path)	
		if file not in listdir:
			print("The file "+file+" cannot be found in the path "+path)
			sys.exit()			
		filepath 	= os.path.join(path, file)
		epochs 		= pd.read_csv(filepath, header = None)
	elif file is None and len(start) and len(end):
		epochs = pd.DataFrame(np.vstack((start, end)).T)
	elif file is None and start is None and end is None:
		print("You have to specify either a file or arrays of start and end; Exiting ...")
		sys.exit()
	
	# Creating /Analysis/ Folder here if not already present
	new_path	= os.path.join(path, 'Analysis/')
	if not os.path.exists(new_path): os.makedirs(new_path)
	# Writing to BehavEpochs.h5
	new_file 	= os.path.join(new_path, 'BehavEpochs.h5')
	store 		= pd.HDFStore(new_file, 'a')
	epoch 		= np.unique(order)
	for i, n in enumerate(epoch):
		idx = np.where(np.array(order) == n)[0]
		ep = nts.IntervalSet(start = epochs.loc[idx,0],
							end = epochs.loc[idx,1],
							time_units = time_units)
		store[n] = pd.DataFrame(ep)
	store.close()

	return None

def loadEpoch(path, epoch):
	"""
	load the epoch contained in path	
	If the path contains a folder analysis, the function will load either the BehavEpochs.mat or the BehavEpochs.h5
	Run makeEpochs(data_directory, ['sleep', 'wake', 'sleep', 'wake'], file='Epoch_TS.csv') to create the BehavEpochs.h5

	Args:
		path: string
		epoch: string

	Returns:
		neuroseries.IntervalSet
	"""			
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()		
	filepath 	= os.path.join(path, 'Analysis')
	listdir		= os.listdir(filepath)
	file 		= [f for f in listdir if 'BehavEpochs' in f]
	if len(file) == 0:
		print("Couldn't find a BehavEpochs file in "+filepath+"; Exiting ...")
		sys.exit()
	elif file[0] == 'BehavEpochs.h5':
		new_file = os.path.join(filepath, 'BehavEpochs.h5')
		store 		= pd.HDFStore(new_file, 'r')
		if '/'+epoch in store.keys():
			ep = store[epoch]
			store.close()
			return nts.IntervalSet(ep)
		else:
			print("The file BehavEpochs.h5 does not contain the key "+epoch+"; Exiting ...")
			sys.exit()
	elif file[0] == 'BehavEpochs.mat':
		behepochs = scipy.io.loadmat(os.path.join(filepath,file[0]))
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
		###################################
		# WORKS ONLY FOR MATLAB FROM HERE #
		###################################		
		elif epoch == 'sws':
			sampling_freq = 1250
			new_listdir = os.listdir(path)
			for f in new_listdir:
				if 'sts.SWS' in f:
					sws = np.genfromtxt(os.path.join(path,f))/float(sampling_freq)
					return nts.IntervalSet.drop_short_intervals(nts.IntervalSet(sws[:,0], sws[:,1], time_units = 's'), 0.0)

				elif '-states.mat' in f:
					sws = scipy.io.loadmat(os.path.join(path,f))['states'][0]
					index = np.logical_or(sws == 2, sws == 3)*1.0
					index = index[1:] - index[0:-1]
					start = np.where(index == 1)[0]+1
					stop = np.where(index == -1)[0]
					return nts.IntervalSet.drop_short_intervals(nts.IntervalSet(start, stop, time_units = 's', expect_fix=True), 0.0)

		elif epoch == 'rem':
			sampling_freq = 1250
			new_listdir = os.listdir(path)
			for f in new_listdir:
				if 'sts.REM' in f:
					rem = np.genfromtxt(os.path.join(path,f))/float(sampling_freq)
					return nts.IntervalSet(rem[:,0], rem[:,1], time_units = 's').drop_short_intervals(0.0)

				elif '-states/m' in listdir:
					rem = scipy.io.loadmat(path+f)['states'][0]
					index = (rem == 5)*1.0
					index = index[1:] - index[0:-1]
					start = np.where(index == 1)[0]+1
					stop = np.where(index == -1)[0]
					return nts.IntervalSet(start, stop, time_units = 's', expect_fix=True).drop_short_intervals(0.0)

def loadPosition(path, file, ep = None, names = ['x', 'y', 'ry', 'rx', 'rz']):
	"""
	load the position from the csv file contains in path

	Notes:
		The order of the columns in the csv file is assumed to be
			['x', 'y', 'ry', 'rx', 'rz']	
	Args:
		path: string
		epoch: string

	Returns:
		neuroseries.IntervalSet
	"""		
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()		
	if '.csv' not in file:
		print("The file is not .csv; The wrappers does not support other format yet; Exiting ...")
		sys.exit()
	position = pd.read_csv(os.path.join(path, 'Tracking_data.csv'), header = None, index_col = 0, names = names)
	position = position[~position.index.duplicated(keep='first')]
	position = nts.TsdFrame(t = position.index.values, d = position.values, time_units = 's', columns = names)
	position[['ry', 'rx', 'rz']] *= (np.pi/180)
	position[['ry', 'rx', 'rz']] += 2*np.pi
	position[['ry', 'rx', 'rz']] %= 2*np.pi
	if ep is not None:
		position = position.restrict(ep)
	return position

##########################################################################################################
# TODO
##########################################################################################################

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
	spikedata = scipy.io.loadmat(path)
	shank = spikedata['shank']
	return shank



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



def loadLFP(path, n_channels=90, channel=64, frequency=1250.0, precision='int16'):
	import neuroseries as nts
	if type(channel) is not list:
		f = open(path, 'rb')
		startoffile = f.seek(0, 0)
		endoffile = f.seek(0, 2)
		bytes_size = 2		
		n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
		duration = n_samples/frequency
		interval = 1/frequency
		f.close()
		with open(path, 'rb') as f:
			data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:,channel]
		timestep = np.arange(0, len(data))/frequency
		return nts.Tsd(timestep, data, time_units = 's')
	elif type(channel) is list:
		f = open(path, 'rb')
		startoffile = f.seek(0, 0)
		endoffile = f.seek(0, 2)
		bytes_size = 2
		
		n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
		duration = n_samples/frequency
		f.close()
		with open(path, 'rb') as f:
			data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:,channel]
		timestep = np.arange(0, len(data))/frequency
		return nts.TsdFrame(timestep, data, time_units = 's')

def loadBunch_Of_LFP(path,  start, stop, n_channels=90, channel=64, frequency=1250.0, precision='int16'):
	import neuroseries as nts	
	bytes_size = 2		
	start_index = int(start*frequency*n_channels*bytes_size)
	stop_index = int(stop*frequency*n_channels*bytes_size)
	fp = np.memmap(path, np.int16, 'r', start_index, shape = (stop_index - start_index)//bytes_size)
	data = np.array(fp).reshape(len(fp)//n_channels, n_channels)

	if type(channel) is not list:
		timestep = np.arange(0, len(data))/frequency
		return nts.Tsd(timestep, data[:,channel], time_units = 's')
	elif type(channel) is list:
		timestep = np.arange(0, len(data))/frequency		
		return nts.TsdFrame(timestep, data[:,channel], time_units = 's')
