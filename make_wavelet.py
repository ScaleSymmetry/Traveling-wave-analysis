import numpy as np
from scipy.optimize import curve_fit

def wavelet(data_cts,N_cycles,frequency,tdel,minf):
    #data_cts is the time series data in shape cases (trials, epochs), times (samples), sensors (contacts, electrodes)
    #N_cycles is the number of cycles in the Morlet wavelet; typically 2 in Alexander et al.
    #frequency is the centre frequency, typically chosen from oversampled, logarithmically spaced frequncies
    #tdel is the time interval between samples e.g. 1ms
    #minf is the minimum frequency used in the analysis, need to ensure the final phase/power estimates all have the same number of samples
    nSamples = data_cts.shape[1]
    nSensors = data_cts.shape[2]
    N_cycles_of_samples = int(N_cycles*1000.0/(frequency*tdel))
    #the extra samples of data required at both start and beginning of estimated phase/power values
    PH_unpad_at_minf = int(N_cycles*500.0/(minf*tdel))
    #the number of samples of phase/power that will be generated
    PHSamples = nSamples - 2*PH_unpad_at_minf
    #half the wavelet cycles at lowest frequency minus half the wavelet cycles at this frequency
    PH_pad_at_f =  PH_unpad_at_minf - int(N_cycles*500.0/(frequency*tdel))
    W = gausian_window(N_cycles_of_samples) #size of Morlet window
    f_indexes = []
    #how many sets of indexes to make? one for each sample in phase
    for t in range(PHSamples):
        #make a list of all data samples that are used in calculating each phase estimate
        padded_range = range(t+PH_pad_at_f,t+PH_pad_at_f+N_cycles_of_samples)
        f_indexes += padded_range
    f_indexes = np.asarray(f_indexes)
    power_cTs = np.zeros((data_cts.shape[0],PHSamples,nSensors),float)
    phi_cTs = np.zeros((data_cts.shape[0],PHSamples,nSensors),complex)
    for c,data_ts in enumerate(data_cts):
        for s in range(nSensors):
            data_tw = data_ts[f_indexes,s].reshape(PHSamples,N_cycles_of_samples) #tw
            base = data_tw.mean(1) #over w
            data_tw = data_tw - base[:,np.newaxis]
            t_f = 2.0*np.pi*N_cycles*np.arange(N_cycles_of_samples)/N_cycles_of_samples
            t_phi = np.exp(1j*(t_f)) * W
            fourier_components = (data_tw * t_phi[np.newaxis,:]).sum(1)
            phi_cTs[c,:,s] = np.exp(1j*np.angle(fourier_components)) #over w
            power_cTs[c,:,s] =  np.absolute(fourier_components)
    return power_cTs,phi_cTs

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def gausian_window(x):
    if type(x) is int:
        n = x
        x = np.arange(n)
    else:
        n = len(x)
    cosine_window = 0.5*(1.0 - np.cos(2*np.pi*x/(n-1)))
    mean = sum(x*cosine_window)/n
    sigma = np.sqrt(sum(cosine_window*(x-mean)**2)/n)
    popt,pcov = curve_fit(gaus,x,cosine_window,p0=[1,mean,sigma])
    G = gaus(x,*popt)
    return G
