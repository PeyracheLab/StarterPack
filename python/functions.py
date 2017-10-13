import numpy as np

#########################################################
# CORRELATION
#########################################################
def crossCorr(t1, t2, binsize, nbins):
	''' 
		Fast crossCorr 
	'''
	nt1 = len(t1)
	nt2 = len(t2)
	if np.floor(nbins/2)*2 == nbins:
		nbins = nbins+1

	m = -binsize*((nbins+1)/2)
	B = np.zeros(nbins)
	for j in range(nbins):
		B[j] = m+j*binsize

	w = ((nbins/2) * binsize)
	C = np.zeros(nbins)
	i2 = 1

	for i1 in range(nt1):
		lbound = t1[i1] - w
		while i2 < nt2 and t2[i2] < lbound:
			i2 = i2+1
		while i2 > 1 and t2[i2-1] > lbound:
			i2 = i2-1

		rbound = lbound
		l = i2
		for j in range(nbins):
			k = 0
			rbound = rbound+binsize
			while l < nt2 and t2[l] < rbound:
				l = l+1
				k = k+1

			C[j] += k

	# for j in range(nbins):
	# C[j] = C[j] / (nt1 * binsize)
	C = C/(nt1 * binsize/1000)

	return C

def crossCorr2(t1, t2, binsize, nbins):
	'''
		Slow crossCorr
	'''
	window = np.arange(-binsize*(nbins/2),binsize*(nbins/2)+2*binsize,binsize) - (binsize/2.)
	allcount = np.zeros(nbins+1)
	for e in t1:
		mwind = window + e
		# need to add a zero bin and an infinite bin in mwind
		mwind = np.array([-1.0] + list(mwind) + [np.max([t1.max(),t2.max()])+binsize])	
		index = np.digitize(t2, mwind)
		# index larger than 2 and lower than mwind.shape[0]-1
		# count each occurences 
		count = np.array([np.sum(index == i) for i in range(2,mwind.shape[0]-1)])
		allcount += np.array(count)
	allcount = allcount/(float(len(t1))*binsize / 1000)
	return allcount

def xcrossCorr_slow(t1, t2, binsize, nbins, nbiter, jitter, confInt):		
	times 			= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	H0 				= crossCorr(t1, t2, binsize, nbins)	
	H1 				= np.zeros((nbiter,nbins+1))
	t2j	 			= t2 + 2*jitter*(np.random.rand(nbiter, len(t2)) - 0.5)
	t2j 			= np.sort(t2j, 1)
	for i in range(nbiter):			
		H1[i] 		= crossCorr(t1, t2j[i], binsize, nbins)
	Hm 				= H1.mean(0)
	tmp 			= np.sort(H1, 0)
	HeI 			= tmp[int((1-confInt)/2*nbiter),:]
	HeS 			= tmp[int((confInt + (1-confInt)/2)*nbiter)]
	Hstd 			= np.std(tmp, 0)

	return (H0, Hm, HeI, HeS, Hstd, times)

def xcrossCorr_fast(t1, t2, binsize, nbins, nbiter, jitter, confInt):		
	times 			= np.arange(0, binsize*(nbins*2+1), binsize) - (nbins*2*binsize)/2
	# need to do a cross-corr of double size to convolve after and avoid boundary effect
	H0 				= crossCorr(t1, t2, binsize, nbins*2)	
	window_size 	= 2*jitter//binsize
	window 			= np.ones(window_size)*(1/window_size)
	Hm 				= np.convolve(H0, window, 'same')
	Hstd			= np.sqrt(np.var(Hm))	
	HeI 			= np.NaN
	HeS 			= np.NaN	
	return (H0, Hm, HeI, HeS, Hstd, times)	


#########################################################
# WRAPPERS
#########################################################
def loadShankStructure(generalinfo):
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
	# units shoud be the value to convert in s 
	import scipy.io
	import neuroseries as nts
	spikedata = scipy.io.loadmat(path)
	shank = spikedata['shank']
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

			elif '-states.mat' in listdir:
				rem = scipy.io.loadmat(path+f)['states'][0]
				index = (rem == 5)*1.0
				index = index[1:] - index[0:-1]
				start = np.where(index == 1)[0]+1
				stop = np.where(index == -1)[0]
				return nts.IntervalSet(start, stop, time_units = 's', expect_fix=True).drop_short_intervals(0.0)

def loadHDCellInfo(path, index):
	import scipy.io
	hd_info = scipy.io.loadmat(path)['hdCellStats'][:,-1]
	return np.where(hd_info[index])[0]