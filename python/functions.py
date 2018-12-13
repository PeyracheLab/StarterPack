import numpy as np

'''
Utilities functions
Feel free to add your own
'''


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
