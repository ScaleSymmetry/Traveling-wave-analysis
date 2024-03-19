import numpy as np

def decompose_rphi(rphi_cts,tdel,frequency):
    #tdel is 1/sampling_frequency in ms
    #frequency in Hertz
    #rphi_cts is complex valued phase, possibly with magnitude weighting
    one_cycle_of_samples = int(1000.0/(tdel*frequency))
    rphi_Cs = rphi_cts.reshape(-1,rphi_cts.shape[2])
    vel_C = [] #normalized velocity, 0 is pure SW, 1 is pure TW
    TWf_C = [] #forward spatial component
    for C in range(rphi_Cs.shape[0]):
        rphi_Ts = np.exp(1j*np.linspace(-np.pi,np.pi,one_cycle_of_samples,endpoint=False))[:,np.newaxis] * rphi_Cs[C,:][np.newaxis,:]
        u_r,s_r,vt_r = np.linalg.svd(rphi_Ts.real)
        TWf_s = vt_r[0,:] + 1j*vt_r[1,:] #throw away the time dimension we added, only need the spatial dimension
        TWf_C += [TWf_s[np.newaxis]]
        vel_C += [np.array([s_r[1]/s_r[0]])[np.newaxis]]
    vel_ct = np.concatenate(vel_C,axis=0).reshape(rphi_cts.shape[0],rphi_cts.shape[1]) 
    TWf_cts = np.concatenate(TWf_C,axis=0).reshape(rphi_cts.shape[0],rphi_cts.shape[1],rphi_cts.shape[2])
    return vel_ct,TWf_cts