import numpy as np
import scipy as sp
import Zcriterion.plasma as plasma

tau_T = 4500*24*60*60 # Tritium half-life [s]
W_max = 18.6e3 # Maximum beta energy [eV]
r_e = plasma.e**2/(4*np.pi*plasma.eps0*plasma.m_e*plasma.c**2) # Classic electron radius [m]
sigma_T = 8/3*np.pi*r_e**2 # Thompson cross-section [m^-2]

# Fluid avalanche generation rate in a partially ionized plasma. Follows (C.11) in Hoppe et al. CPC (2021) to imporve
# accuracy in nearly neutral low-Z plasmas

def calc_Gamma(Z,Z0,n_j,T_e,E,B = 0, reltol = 1e-3, maxIter = 10, analytical = False):
    """
    Compute fluid avalanche generation rate.

    :param array_like Z: 1D array of charge numbers.
    :param array_like Z0: 1D array of charge states.
    :param array_like n_j: Charge state distribution. Last axis must match the size of Z and Z0.
    :param array_like T_e: Electron temperature.
    :param array_like E: Electric field in units of Ec.
    :param array_like B: Magnetic field used to compute synchrotron radiation [T] (default: 0).
    :param float reltol: Relative tolerence when iterating (default: 1e-3).
    :param int maxIter: Maximum number of iterations when evaluating pStar (default: 10).
    :param bool analytical: Evaluate using analytical formulation, which includes first order corrections for partial screening (default: False).
    """

    # Constants
    e = plasma.e # Electron charge
    c = plasma.c # Speed of light
    m_e = plasma.m_e # Electron mass


    n_e_free = np.sum(Z0*n_j,axis = -1) # Free electron density
    n_e_tot = np.sum(Z*n_j,axis = -1) # Total electron density
    lnL = plasma.lnLc(T_e,n_e_free)
    Ec = plasma.calc_Ec(T_e,n_e_free)
    if analytical:
        Z_eff_RP = plasma.calc_Zeff(Z0,Z,n_j,includeBoundElectrons = True)
        n_e_bound = n_e_tot - n_e_free
        E_C_RP = (1 + n_e_bound/(2*n_e_free))
        # Gamma as a function of normalized electric field E = Epar/Ec
        Gamma = Ec*e/(m_e*c*lnL)*n_e_tot/n_e_free*(np.maximum(E/E_C_RP - 1,0))/np.sqrt(5 + Z_eff_RP)
    else:
        Eceff = plasma.calc_Eceff(Z,Z0,n_j,T_e,B,reltol = reltol,maxIter = maxIter)
        pStar = plasma.calc_pStar(Z,Z0,n_j,T_e,E,reltol = reltol,maxIter = maxIter)
        nu_Dbar = plasma.calc_nu_Dbar(Z,Z0,n_j,T_e,pStar)
        nu_Sbar = plasma.calc_nu_Sbar(Z,Z0,n_j,T_e,pStar)
        # Gamma as a function of normalized electric field E = Epar/Ec
        Gamma = Ec*e/(m_e*c*lnL)*n_e_tot/n_e_free*(np.maximum(E - Eceff*n_e_tot/n_e_free,0))/np.sqrt(4*nu_Sbar**2 + nu_Dbar*nu_Sbar)
           
    return Gamma

# Exponent in avalanche multiplication, derived in Hesslow et al. NF (2019)

def calc_N_ava(Z,Z0,n_j,T_e,j0,a_wall,B = 0,reltol = 1e-3,maxIter = 10, analytical = False):
    """
    Compute exponent for avalanche multiplication factor.

    :param array_like Z: 1D array of charge numbers.
    :param array_like Z0: 1D array of charge states.
    :param array_like n_j: Charge state distribution. Last axis must match the size of Z and Z0.
    :param array_like T_e: Electron temperature.
    :param array_like j0: Initial current density.
    :param array_like a_wall: Minor radius of wall [m]
    :param array_like B: Magnetic field used to compute synchrotron radiation [T] (default: 0).
    :param float reltol: Relative tolerence when iterating (default: 1e-3).
    :param int maxIter: Maximum number of iterations when evaluating pStar (default: 10).
    :param bool analytical: Evaluate using analytical formulation, which includes first order corrections for partial screening (default: False).
    """

    # Constants
    e = plasma.e # Electron charge
    c = plasma.c # Speed of light
    m_e = plasma.m_e # Electron mass

    n_e_free = np.sum(Z0*n_j,axis = -1) # Free electron density
    Z_eff = np.sum(Z0**2*n_j,axis = -1)/n_e_free # Effective charge
    n_e_tot = np.sum(Z*n_j,axis = -1) # Total electron density
    sigma = plasma.calc_spitzerCond(T_e,n_e_free,Z_eff)
    E_init = j0/(sigma*plasma.calc_Ec(T_e,n_e_free)) # Initial electric field in units of Ec
    Ec = plasma.calc_Ec(T_e,n_e_free)
    # Effective critical electric field in units of Ec
    if analytical:
        Eceff = 1
        lnL = plasma.lnLc(T_e,n_e_free) 
        Z_eff_RP = plasma.calc_Zeff(Z0,Z,n_j,includeBoundElectrons = True)
        n_e_bound = n_e_tot - n_e_free
        E_C_RP = (1 + n_e_bound/(2*n_e_free))
        inteGammaOverE = (Ec*e/(m_e*c*lnL)*n_e_tot/n_e_free*(np.maximum(E_init/E_C_RP,1) - np.log(np.maximum(E_init/E_C_RP,1)))/np.sqrt(5 + Z_eff_RP) 
                         - Ec*e/(m_e*c*lnL)*n_e_tot/n_e_free/np.sqrt(5 + Z_eff_RP))
    else:
        Eceff = np.minimum(plasma.calc_Eceff(Z,Z0,n_j,T_e,B,reltol = reltol, maxIter = maxIter)*n_e_tot/n_e_free,E_init)
        E = np.linspace(Eceff,E_init,1000)
        Gamma = calc_Gamma(Z,Z0,n_j,T_e,E,B,reltol = reltol, maxIter = maxIter)
        inteGammaOverE = sp.integrate.simpson(Gamma/E,E,axis = 0)
        
    tau_CQ = plasma.calc_tau_CQ(T_e,n_e_free,Z_eff,a_wall) 

    return tau_CQ*inteGammaOverE

# Critical runaway momentum consistent with the definition of Gamma

def calc_pCrit(Z,Z0,n_j,T_e,E,B = 0,reltol = 1e-3,maxIter = 10):
    """
    Compute critical momentum used to evaluate seed currents. 

    :param array_like Z: 1D array of charge numbers.
    :param array_like Z0: 1D array of charge states.
    :param array_like n_j: Charge state distribution. Last axis must match the size of Z and Z0.
    :param array_like T_e: Electron temperature.
    :param array_like E: Electric field in units of Ec.
    :param array_like B: Magnetic field used to compute synchrotron radiation [T] (default: 0).
    :param float reltol: Relative tolerence when iterating (default: 1e-3).
    :param int maxIter: Maximum number of iterations when evaluating pStar (default: 10).
    """
    n_e_free = np.sum(Z0*n_j,axis = -1)
    n_e_tot = np.sum(Z*n_j,axis = -1)

    Eceff = plasma.calc_Eceff(Z,Z0,n_j,T_e,B,reltol = reltol, maxIter = maxIter)
    pStar = plasma.calc_pStar(Z,Z0,n_j,T_e,E,reltol = reltol, maxIter = maxIter)
    nu_Dbar = plasma.calc_nu_Dbar(Z,Z0,n_j,T_e,pStar)
    nu_Sbar = plasma.calc_nu_Sbar(Z,Z0,n_j,T_e,pStar)

    pCritInvSq = np.maximum(E - Eceff*n_e_tot/n_e_free,0)/np.sqrt(4*nu_Sbar**2 + nu_Dbar*nu_Sbar)
    pCrit = np.zeros(pCritInvSq.shape)
    pCrit[pCritInvSq == 0] = np.inf
    pCrit[pCritInvSq > 0] = 1/np.sqrt(pCritInvSq[pCritInvSq > 0])
    return pCrit

# Gamma photon spectrum from activated wall. Consistent with Martin-Solis et al. NF (2017) and Ekmark et al. JPP (2025)

def calc_comptonSpectrum(W,C):
    """
    Compute gamma photon spectrum.

    :param array_like W: Photon energies in eV
    :param list C: List or array of length 3 with fitting parameters for the photon spectrum
    """
    z  = (np.log(W/1e6) + C[0])/C[1] + C[2]*(W/1e6)**2
    return np.exp(-np.exp(-z) - z + 1)

# Critical angle previously used in the Klein-Nishina cross-section (see Martin-Solis et al. NF (2017) for details)

# def calc_cosTheta_c(x,xc):
#    # Find x that corresponds to cosTheta_c = -1
#    x_min = np.zeros(np.shape(xc)) 
#    x_min_lambda = lambda xc : xc/2*(1 + np.sqrt(1 + 2/xc))
#    x_min[xc > 0] = x_min_lambda(xc[xc > 0])
#    cosTheta_c = (1 - 1./np.maximum(x,x_min)*(xc/(np.maximum(x,x_min) - xc)))
#
#    ind = np.asarray(np.invert(np.isfinite(cosTheta_c))).nonzero()
#    cosTheta_c[ind] = -1
#    return cosTheta_c

# Integrated Klein-Nishina cross-section in units of the Thomson cross-section (see Martin-Solis et al. NF (2017) for details)

def calc_comptonCrossSection(x,xc):
    """
    Compute integrated Compton cross section in units of the Thomson cross-section. 

    :param array_like x: Photon energy in units the electron rest energy m_e*c^2
    :param array_like xc: Critical energy in units the electron rest energy m_e*c^2
    """
    xc = np.atleast_1d(xc)
    x_min = np.zeros(xc.shape)
    x_min[xc > 0] = xc[xc > 0]/2*(1 + np.sqrt(1 + 2/xc[xc > 0]))
    cosTheta_c = (1 - 1./np.maximum(x,x_min)*(xc/(np.maximum(x,x_min) - xc)))
    # Replace non-finite values with -1
    ind = np.nonzero(np.isinf(cosTheta_c) | np.isnan(cosTheta_c))
    cosTheta_c[ind] = -1

    crossSection = 3/8 * (((x**2-2*x-2)/x**3)*np.log((1 + 2*x)/(1 + x*(1-cosTheta_c))) 
                   + 1/(2*x)*(1/(1 + x*(1-cosTheta_c))**2 - 1/(1 + 2*x)**2) 
                   - 1/x**3*(1 - x - (1 + 2*x)/(1 + x*(1 - cosTheta_c)) - x*cosTheta_c))
    
    return np.maximum(crossSection,0) # Negative cross-sections may occur due to numerical noise 

# Cross-section averaged over the compton spectrum (in units of Thomson scattering cross-section).
# The "effective" cross-section is a function of critcal energy only. The implementation below is slow,
# since it is integrated over the photon energy spectrum for each data-point. For a more efficient
# implementation based in interpolation, see calc_sigmaEff_alt.

def calc_sigmaEff(Wc,C):
    """
    Computes the average of the Compton cross-section over the Compton spectrum, resulting in an
    effective Compton cross-section that is only a function of the critical energy Wc.

    :param array_like Wc: Critical energy in eV.
    :param list C: List of fitting parameters for the Compton spectrum.
    """
    # Constants
    We = plasma.We # Electron energy in eV

    Wc = np.array(Wc)

    # Photon energy normalized by electron rest energy
    # ITER Compton spectrum (from Martin-Solis et al.) converges very slowly, hence the high values of x
    w_gamma = np.logspace(-2,25,1350) # 50 grid points per decade
    # Add extra dimension for x to integrate over
    w_gamma = np.expand_dims(w_gamma, axis = tuple(range(1,Wc.ndim+1)))
    Wc = Wc[np.newaxis]
    comptonSpectrum = calc_comptonSpectrum(w_gamma*We,C)
    integrand = comptonSpectrum*calc_comptonCrossSection(w_gamma,Wc/We)
    sigmaEff = sp.integrate.simpson(integrand,w_gamma,axis = 0)/sp.integrate.simpson(comptonSpectrum,w_gamma,axis = 0)
    return sigmaEff

# A faster implementation of the effective cross-section. Evalute the integral for a range critical energies, and
# then interpolate in log-log space.

def calc_sigmaEff_alt(Wc,C):
    """
    Computes the average of the Compton cross-section over the Compton spectrum, resulting in an
    effective Compton cross-section that is only a function of the critical energy Wc.

    :param array_like Wc: Critical energy in eV.
    :param list C: List of fitting parameters for the Compton spectrum.
    """
    # Constants
    We = plasma.We # Electron rest energy in eV

    Wc = np.atleast_1d(Wc)
    # Photon energy normalized by electron rest energy
    # ITER Compton spectrum (from Martin-Solis et al.) converges very slowly, hence the high values of x
    x = np.logspace(-2,25,1350) # 50 grid points per decade
    Wc_p = np.logspace(0,7,1000) # Critical energy grid to create interpolation data in eV
    # Add extra dimension for x to integrate over
    x = np.expand_dims(x, axis = 1)
    comptonSpectrum = calc_comptonSpectrum(x*We,C)
    integrand = comptonSpectrum*calc_comptonCrossSection(x,Wc_p[np.newaxis]/We)
    sigmaEff_p = sp.integrate.simpson(integrand,x,axis = 0)/sp.integrate.simpson(comptonSpectrum,x,axis = 0)

    # Interpolation in log-log space
    log_x = np.log10(Wc)
    log_y = np.zeros(log_x.shape) # Preallocate output
    log_xp = np.log10(Wc_p)
    log_yp = np.log10(sigmaEff_p)

    # Crate masks for extrapolation and interpolation domains
    leftInd = np.nonzero(log_x < log_xp[0]) 
    rightInd = np.nonzero(log_x > log_xp[-1])
    middleInd = np.nonzero((log_x >= log_xp[0]) & (log_x <= log_xp[-1]))
    
    log_y[middleInd] = np.interp(log_x[middleInd],log_xp,log_yp)
    # Calculate slopes at the boundaries and extrapolate linearly
    k_left = (log_yp[1]-log_yp[0])/(log_xp[1] - log_xp[0])
    log_y[leftInd] = k_left*(log_x[leftInd] - log_xp[0]) + log_yp[0]
    k_right = (log_yp[-1] - log_yp[-2])/(log_xp[-1] - log_xp[-2])
    log_y[rightInd] = k_right*(log_x[rightInd] - log_xp[-1]) + log_yp[-1]
    
    return 10**log_y

# Evaluate Compton scattering seed current in units of j0/ec

def calc_comptonSeed(Z,Z0,n_j,T_e,j0,a,Gamma_flux,C = np.array([1.2, 0.8, 0]),
                     B = 0,maxIter = 10,reltol = 1e-3, analytical = False):
    """
    Compute integrated Compton scattering seed in units of j0/ec.

    :param array_like Z: Atomic number, length should match the total number of charge states and species.
    :param array_like Z0: Charge states, length should match the total number of charge states and species.
    :param array_like n_j: Densities (in m^-3) for each species and charge state, last dimension should match the total number of charge states and species.
    :param array_like T_e: Electron temperature [eV]
    :param array_like j0: Initial Ohmic current density [A/m^2]
    :param array_like a: Minor radius of plasma [m]
    :param array_like Gamma_flux: Total flux of gamma-photons [m^-2 s^-1]
    :param list C: Fitting parameters for Compton spectrum
    :param array_like B: Magnetic field used for syncrotron radiation when evaluating Eceff (default: 0)
    :param int maxIter: Maximum number of iterations when evaluating p_star and Eceff (default: 10)
    :param float reltol: Relative tolerence when evaluating p_star and Eceff (default: 1e-3)
    :param bool analytical: Bolean that determines whether to evaluate simplified model for the Compton seed. Fast, but overestimates the Compton seed (default: False)
    """
    
    # Constants
    global sigma_T
    e = plasma.e # Electron charge
    c = plasma.c # Speed of light
    x1 = plasma.x1 # First zero of the Bessel function j0
    We = plasma.We # Electron rest energy in eV

    n_e_free = np.sum(Z0*n_j,axis = -1) # Free electron density
    Z_eff = np.sum(Z0**2*n_j,axis = -1)/n_e_free # Effective charge
    n_e_tot = np.sum(Z*n_j,axis = -1) # Total electron density
    sigma = plasma.calc_spitzerCond(T_e,n_e_free,Z_eff)
    tau_CQ = plasma.calc_tau_CQ(T_e,n_e_free,Z_eff,a)

    E_init = j0/(sigma*plasma.calc_Ec(T_e,n_e_free)) # Initial electric field in units of Ec
    # Effective critical electric field in units of Ec
    Eceff = plasma.calc_Eceff(Z,Z0,n_j,T_e,B = B,reltol = reltol, maxIter = maxIter)*n_e_tot/n_e_free
    # Ensure that E_init is not smaller than Eceff
    E_init = np.maximum(Eceff,E_init)
    E = np.linspace(Eceff,E_init,1000)

    pCrit = calc_pCrit(Z,Z0,n_j,T_e,E,B = B,reltol = reltol, maxIter = maxIter)    
    Wc = We*(np.sqrt(pCrit**2 + 1) - 1) # Critial energy [eV]

    if analytical:
        Eceff = 1
        intSigmaEffOverE = 0.5*calc_sigmaEff_alt(1e-6,C)*np.log(E_init/Eceff)
    else:
        intSigmaEffOverE = sp.integrate.simpson(calc_sigmaEff_alt(Wc,C)/E,E,axis = 0)

    # x1^2/2 compensates for 1D effects in tau_CQ. It is possible to remove this factor, but 
    # it makes the criterion less consevative
    nSeed = x1**2/2*e*c*n_e_tot/j0*tau_CQ*Gamma_flux*sigma_T*intSigmaEffOverE

    return nSeed

# Evaluate Tritium beta decay seed current in units of j0/ec

def calc_tritiumSeed(Z,Z0,n_j,T_e,n_T,j0,a,B = 0,maxIter = 10,reltol = 1e-3,analytical = False):
    """
    Compute integrated Tritium beta decay seed in units of j0/ec.

    :param array_like Z: Atomic number, length should match the total number of charge states and species.
    :param array_like Z0: Charge states, length should match the total number of charge states and species.
    :param array_like n_j: Densities (in m^-3) for each species and charge state, last dimension should match the total number of charge states and species.
    :param array_like T_e: Electron temperature [eV]
    :param array_like n_T: Total tritium density [m^-3]
    :param array_like j0: Initial Ohmic current density [A/m^2]
    :param array_like a: Minor radius of plasma [m]
    :param array_like Gamma_flux: Total flux of gamma-photons [m^-2 s^-1]
    :param list C: Fitting parameters for Compton spectrum
    :param array_like B: Magnetic field used for syncrotron radiation when evaluating Eceff (default: 0)
    :param int maxIter: Maximum number of iterations when evaluating p_star and Eceff (default: 10)
    :param float reltol: Relative tolerence when evaluating p_star and Eceff (default: 1e-3)
    :param bool analytical: Bolean that determines whether to evaluate simplified model for the Tritium seed, neglects effects of partial screening (default: False)
    """
    # Constants
    global tau_T, W_max
    e = plasma.e # Electron charge
    c = plasma.c # Speed of light
    x1 = plasma.x1 # First zero of the Bessel function j0
    We = plasma.We # Electron rest energy in eV

    n_e_free = np.sum(Z0*n_j,axis = -1) # Free electron density
    Z_eff = np.sum(Z0**2*n_j,axis = -1)/n_e_free # Effective charge
    n_e_tot = np.sum(Z*n_j,axis = -1) # TOtal electron density
    sigma = plasma.calc_spitzerCond(T_e,n_e_free,Z_eff)
    tau_CQ = plasma.calc_tau_CQ(T_e,n_e_free,Z_eff,a)
    E_init = j0/(sigma*plasma.calc_Ec(T_e,n_e_free)) # Initial electric field in units of Ec

    if analytical:
        w_max = W_max/We
        Z_eff = plasma.calc_Zeff(Z0,Z,n_j,includeBoundElectrons = False)
        E_min = 1 + np.sqrt(5+Z_eff)/(w_max**2 + 2*w_max)
        intFoverE = calc_analyticalTritiumSeedIntegral(np.maximum(E_init,E_min),Z_eff) - calc_analyticalTritiumSeedIntegral(E_min,Z_eff)
        intFoverE = np.maximum(intFoverE,0)
    else:
        # Effective critical electric field in units of Ec
        Eceff = plasma.calc_Eceff(Z,Z0,n_j,T_e, B = B, reltol = reltol, maxIter = maxIter)*n_e_tot/n_e_free
        # Ensure that E_init is not smaller than Eceff
        E_init = np.maximum(Eceff,E_init)
        E = np.linspace(Eceff,E_init,1000)

        pCrit = calc_pCrit(Z,Z0,n_j,T_e,E,B = B, reltol = reltol, maxIter = maxIter)
        Wc = We*(np.sqrt(pCrit**2 + 1) - 1) # Critial energy [eV]
        # Cap w at 1, since this implies that dn/dt = 0 if Wc > W_max
        w = np.minimum(Wc/W_max,1)
        dndt = np.maximum(1 - 35/8*w**(3/2) + 21/4*w**(5/2) - 15/8*w**(7/2),0)
        intFoverE = sp.integrate.simpson(dndt/E,E,axis = 0) 

    # x1^2/2 compensates for 1D effects in tau_CQ. It is possible to remove this factor, but 
    # it makes the criterion less consevative  
    nSeed = np.log(2)*x1**2/2*e*c*n_T/j0*tau_CQ/tau_T*intFoverE
    return nSeed

# This function evaluates the indefinite integral of F_beta/E in the tritium seed over the 
# normalized electric field E. This expression is only valid in a fully ionized plasma,
# or when the effects of parital screening are neglected.

def calc_analyticalTritiumSeedIntegral(E,Zeff):
    """
    Computes the indefinite integral of F_beta/E used in the analytical evaluation of the analytical tritium seed

    :param array_like E: Normalized electric field in units of Ec.
    :param array_like Zeff: Effective charge.
    """
    global W_max
    We = plasma.We # Electron rest energy in eV
    w_max = W_max/We
    c1 = -35/4
    c2 = 21/2
    c3 = -15/4
    a = np.sqrt(5 + Zeff)
    wc = np.sqrt(a/(E - 1) + 1) - 1
    term0 = np.log(E)
    term1 = c1*w_max**(-3/2)*Gn(1,wc,Zeff)
    term2 = -21*a*np.sqrt(wc)*w_max**(-5/2) + c2*w_max**(-5/2)*Gn(2,wc,Zeff)
    term3 = w_max**(-7/2)*a*(-45/2*np.sqrt(wc) + 5/2*np.sqrt(wc)**3) + c3*w_max**(-7/2)*Gn(3,wc,Zeff)

    return term0 + term1 + term2 + term3

def Gn(n,wc,Zeff):
    """
    Evaluates the function G_n used in the evaluation of the indefinite integral of F_beta/E

    :param integer n: Integer that determines order of Chebyshev polynomials (1,2 or 3).
    :param array_like wc: Critical energy normalized to the electron rest energy.
    :param array_like Zeff: Effective charge.
    """
    a = np.sqrt(5+Zeff)
    u = 1/np.sqrt(2)*np.sqrt(1 + 1/np.sqrt(a))
    v = 1/np.sqrt(2)*np.sqrt(1 - 1/np.sqrt(a))
    out = (-1)**n*(2**((2*n+1)/2)*np.atan(np.sqrt(wc/2))
             - a**((2*n+1)/4)*sp.special.chebyt(2*n + 1)(u)*np.arctan(2*np.sqrt(wc)*a**(1/4)*u/(np.sqrt(a) - wc))
             - a**((2*n+1)/4)*v*sp.special.chebyu(2*n)(u)*np.arctanh(2*np.sqrt(wc)*a**(1/4)*v/(np.sqrt(a) + wc)))
    return out