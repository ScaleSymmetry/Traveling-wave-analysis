# phi_cts are complex valued, unit length phases. 
# they should be calculated using very short time-series methods, I prefer 2 cycle Morlet wavelets. 
# choice of wavelet size between 1 and 5 does not make too much difference, but larger window sizes can wash out fast changing waves.

# the matrix shapes are given by the indexes in the variable name. 
# c is the number of cases or trials, t is the number of samples in a trial, s is the number of sensors. C is c*t. b is the number of bases.

# bases_sb are the singular vectors, comprising orthonormal maps of phase. the first three bases are generally long wavelength (wavenumber=1), and explain about half the variance in the phase data.
# fit_ct is the fit of the TW to the phase data, which is a surrogate for spatio-temporal power, which we cannot calculate directly because the Fourier analysis is done empirically via the svd, enabling us to do the Fourier analysis on arbitrarily shaped arrays.
# betas_ctb are the time-series of the weights of each basis. useful for assessing changes in mean direction in an event-related paradigm.

def C_TW_bases_betas(phi_cts,nBases=3):
    phi_Cs = phi_cts.reshape(-1,phi_cts.shape[-1])
    phi_cent = phi_Cs - phi_Cs.mean(0)
    COV = phi_cent.T.conj()@phi_cent
    u,s,vh = np.linalg.svd(COV)
    print(100.0*s[:nBases]/s.sum())
    bases_sb = vh[:nBases].T
    betas_Cb = phi_cent.dot(bases_sb)
    model_Cs = np.exp(1j*np.angle(bases_sb.dot(betas_Cb.T).T))
    fit_C = (phi_Cs/model_Cs).mean(-1).real
    fit_ct = fit_C.reshape(phi_cts.shape[0],phi_cts.shape[1])
    betas_ctb = betas_Cb.reshape(phi_cts.shape[0],-1,bases_sb.shape[1])
    return bases_sb,fit_ct,betas_ctb