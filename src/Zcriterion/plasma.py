import numpy as np
import warnings

atomicSymbolsList_I = np.array(['H','Ne','Ar'])
atomicNumberList_I = np.array([1,10,18])
atomicSymbolsList_aBar = np.array(['Ne','Ar'])
atomicNumberList_aBar = np.array([10,18])

# Constants
e = 1.60217663e-19 # Electron charge [C]
c = 299792458 # Speed of light [m/s]
m_e = 9.1093837139e-31 # Electron mass [kg]
mu0 = 4*np.pi*1e-7 # Vacuum permeability [H/m]
eps0 = 1/(mu0*c**2) # Vacuum permittivity [F/m]
x1 = 2.4048255576 # First zero of the zeroth order Bessel function J
We = m_e*c**2/e # Electron rest energy in [eV]
alpha = 1/137.035999177 # Fine structure constant


# Tabulated mean excitation energies used to compute slowing-down frequencies when electrons
# collide with partially ionized ions. Additional values for Z < 19 available in Sauer et al. 
# J. Chem. Phys. (2018). Extrapolation formula for Z > 18 is available but not implemented.

def get_ln_I(atomicSymbol = None):
    ln_I =  {'Ar0': -7.9050,
            'Ar1': -7.7532,
            'Ar2': -7.6076,
            'Ar3': -7.4626,
            'Ar4': -7.3178,
            'Ar5': -7.1665,
            'Ar6': -7.0055,
            'Ar7': -6.8020,
            'Ar8': -6.5538,
            'Ar9': -6.4647,
            'Ar10': -6.3644,
            'Ar11': -6.2465,
            'Ar12': -6.1070,
            'Ar13': -5.9219,
            'Ar14': -5.6535,
            'Ar15': -5.3213,
            'Ar16': -4.6937,
            'Ar17': -4.6598,
            'Ne0': -8.2227,
            'Ne1': -8.0370,
            'Ne2': -7.8614,
            'Ne3': -7.6837,
            'Ne4': -7.4994,
            'Ne5': -7.2788,
            'Ne6': -6.9808,
            'Ne7': -6.5976,
            'Ne8': -5.8933,
            'Ne9': -5.8320,
            'H0': -10.4367}
    if atomicSymbol is None:
        return ln_I
    else:
        return ln_I[atomicSymbol]
    
# Tabulated effective-size parameter for neon and argon obtained from https://github.com/hesslow/Eceff
# For non-tabulated species, the formula (3/2*alpha)(pi/3)^(1/3)*(Z−Z0)^(2/3)/Z is used (see eqn 
# (2.28) in Hesslow et al. JPP (2018)).

def get_ln_aBar(atomicSymbol = None):
    ln_aBar = {'Ar0': 4.5677,
            'Ar1': 4.5007,
            'Ar2': 4.4289,
            'Ar3': 4.3513,
            'Ar4': 4.2698,
            'Ar5': 4.1815,
            'Ar6': 4.0829,
            'Ar7': 3.9748,
            'Ar8': 3.8511,
            'Ar9': 3.7882,
            'Ar10': 3.7203,
            'Ar11': 3.6436,
            'Ar12': 3.5579,
            'Ar13': 3.4525,
            'Ar14': 3.3093,
            'Ar15': 3.0503,
            'Ar16': 2.5578,
            'Ar17': 2.5332,
            'Ne0': 4.7064,
            'Ne1': 4.6097,
            'Ne2': 4.5028,
            'Ne3': 4.3858,
            'Ne4': 4.2638,
            'Ne5': 4.1266,
            'Ne6': 3.9568,
            'Ne7': 3.6810,
            'Ne8': 3.1777,
            'Ne9': 3.1278}
    if atomicSymbol is None:
        return ln_aBar
    else:
        return ln_aBar[atomicSymbol]

def lnLc(T_e,n_e):
    """
    Compute relativistic Coulomb logarithm.

    :param array_like T_e: Electron temperature in eV.
    :param array_like n_e: Electron density in m^-3.
    """
    out = 14.6 + 0.5*np.log(T_e/(n_e/1e20))
    return out

def lnLt(T_e,n_e):
    """
    Compute thermal Coulomb logarithm.

    :param array_like T_e: Electron temperature in eV.
    :param array_like n_e: Electron density in m^-3.
    """
    out = 14.9 - 0.5*np.log(n_e/1e20) + np.log(T_e/1e3)
    return out

# Coulomb logarithm for electron-electron collisions

def lnLee(T_e,n_e,p):
    """
    Compute electron-electron Coulomb logarithm.

    :param array_like T_e: Electron temperature in eV.
    :param array_like n_e: Electron density in m^-3.
    :param array_like p: Electron momentum in units of m_e*c.
    """
    # Constants
    global We
    k = 5

    p_Te = np.sqrt(2*T_e/We)
    gamma = np.sqrt(p**2 + 1)
    # out = lnLc(T_e,n_e) + np.log(np.sqrt(gamma - 1)) 
    out = lnLt(T_e,n_e) + 1/k*np.log(1 + (2*(gamma - 1)/p_Te**2)**(k/2))
    return out

# Coulomb logarithm for electron-ion collisions

def lnLei(T_e,n_e,p):
    """
    Compute electron-ion Coulomb logarithm.

    :param array_like T_e: Electron temperature in eV.
    :param array_like n_e: Electron density in m^-3.
    :param array_like p: Electron momentum in units of m_e*c.
    """
    # Constants
    global We
    k = 5
    p_Te = np.sqrt(2*T_e/We)

    #out = lnLc(T_e,n_e) + np.log(np.sqrt(2)*p)
    out = lnLt(T_e,n_e) + 1/k*np.log(1 + (2*p/p_Te)**k)
    return out

# Effective charge
def calc_Zeff(Z0,Z,n_j,includeBoundElectrons = False):
    n_e_free = np.sum(Z0*n_j,axis = -1) # Free electron density
    if includeBoundElectrons:
        PI_mask = Z0 < Z
        CI_mask = Z0 == Z
        Z_eff = (np.sum(Z0[PI_mask]**2*n_j[...,PI_mask],axis = -1)/2 + np.sum(Z0[CI_mask]**2*n_j[...,CI_mask],axis = -1))/n_e_free
    else:
        Z_eff = np.sum(Z0**2*n_j,axis = -1)/n_e_free # Effective charge
    return Z_eff

# Relativistic collision time

def calc_tau_c(T_e,n_e):
    """
    Compute relativistic collision time

    :param array_like T_e: Electron temperature in eV.
    :param array_like n_e: Electron density in m^-3.
    """
    # Constants
    global eps0, m_e, c, e

    lnL = lnLc(T_e,n_e)
    constFactor = 4*np.pi*eps0**2*m_e**2*c**3/e**4
    tau_c = constFactor/(n_e*lnL)
    return tau_c

# Current quench time, accounts for radial diffusion of electric fields.
# See Hesslow et al. NF (2019)

def calc_tau_CQ(T_e,n_e,Z_eff,a_wall):
    """
    Compute current quench time.

    :param array_like T_e: Electron temperature in eV.
    :param array_like n_e: Electron density in m^-3.
    :param array_like Z_eff: Effective charge.
    :param array_like a_wall: Minor radius of the wall.
    """
    # Constants
    global mu0, x1

    sigma = calc_spitzerCond(T_e,n_e,Z_eff)
    tau_CQ = sigma*mu0*a_wall**2/x1**2
    return tau_CQ

# Critical electric field

def calc_Ec(T_e,n_e):
    """
    Compute critical/Connor-Hastie field.

    :param array_like T_e: Electron temperature in eV.
    :param array_like n_e: Electron density in m^-3.
    """
    # Constants
    global e, c, m_e

    Ec = m_e*c/(e*calc_tau_c(T_e,n_e))
    return Ec

# Parallel Spitzer conductivity

def calc_spitzerCond(T_e,n_e,Z_eff):
    """
    Compute parallel Spitzer conductivity.

    :param array_like T_e: Electron temperature in eV.
    :param array_like n_e: Electron density in m^-3.
    :param array_like Z_eff: Effective charge.
    """
    # Constants
    global e, c, m_e, mu0, eps0

    prefactor = (1 + 2.966*Z_eff + 0.753*Z_eff**2)/(1 + 1.198*Z_eff + 0.222*Z_eff**2)
    sigma = prefactor*3*(2*np.pi)**(3/2)*eps0**2*(e*T_e)**(3/2)/(Z_eff*e**2*m_e**(1/2)*lnLc(T_e,n_e))

    return sigma

# Deflection frequency nu_Dbar (normalized by p/(gamma^3*tau_c)) in the presence of partially ionized ions
# as defined by Hesslow et al. The ion contribution is defined in (2.22) in Hesslow et al. JPP (2018), 
# and the electron contribution is given by lnLambda_ee/lnLambda_c

def calc_nu_Dbar(Z,Z0,n_j,T_e,p,partialScreening = True):
    """
    Compute normalized deflection frequency.

    :param array_like Z: 1D array of charge numbers.
    :param array_like Z0: 1D array of charge states.
    :param array_like n_j: Charge state distribution. Last axis must match the size of Z and Z0.
    :param array_like T_e: Electron temperature.
    :param array_like p: Electron momentum in units of m_e*c.
    :param bool partialScreening: Include effects of partial screening (default: True).
    """
    global alpha

    n_e_free = np.sum(Z0*n_j,axis = -1) # Free electron density
    Z_eff = np.sum(Z0**2*n_j,axis = -1)/n_e_free # Effective charge

    if partialScreening == False:
        return np.ones(np.shape(T_e/n_e_free + p))*(1 + Z_eff)
    
    N_e = Z - Z0 # Number of bound electrons
    ln_aBar = get_ln_aBar() # Get tabulated values of ln(aBar)
    out = lnLee(T_e,n_e_free,p) + lnLei(T_e,n_e_free,p)*Z_eff # Complete screening limit
    partiallyIonizedSpecies = np.flatnonzero(N_e > 0) 
    for i in partiallyIonizedSpecies: # For each partially ionized species, add partial screening correction
        if Z[i] in atomicNumberList_aBar: # If tabulated value exists
            atomicSymbolInd = np.flatnonzero(Z[i] == atomicNumberList_aBar)
            atomicSymbolStr = atomicSymbolsList_aBar[atomicSymbolInd] + str(int(Z0[i]))
            aBar = np.exp(ln_aBar[atomicSymbolStr.item()])  
        else: # If tabulated value does not exist, use Kirillov formula (see Hesslow et al. JPP (2018))
            aBar = 3/(2*alpha)*(np.pi/3)**(1/3)*N_e[i]**(2/3)/Z[i]
        out += n_j[...,i]/n_e_free * 2/3 * ((Z[i]**2 - Z0[i]**2) * np.log((aBar*p)**(3/2) + 1) - 2/3 * N_e[i]**2 * (p*aBar)**(3/2) / ((p*aBar)**(3/2) + 1) )   

    return out/lnLee(T_e,n_e_free,p)

# Slowing down frequency nu_Sbar (normalized by p^2/(gamma^3*tau_c)) in the presence of partially ionized ions
# as defined in (2.31) in Hesslow et al. JPP (2018)
    
def calc_nu_Sbar(Z,Z0,n_j,T_e,p,partialScreening = True):
    """
    Compute normalized slowing-down frequency.

    :param array_like Z: 1D array of charge numbers.
    :param array_like Z0: 1D array of charge states.
    :param array_like n_j: Charge state distribution. Last axis must match the size of Z and Z0.
    :param array_like T_e: Electron temperature.
    :param array_like p: Electron momentum in units of m_e*c.
    :param bool partialScreening: Include effects of partial screening (default: True).
    """
    n_e_free = np.sum(Z0*n_j,axis = -1) # Free electron density

    if partialScreening == False: # Low momentum and complete screening limit
        return np.ones(np.shape(T_e/n_e_free + p))
    
    N_e = Z - Z0 # Number of bound electrons
    gamma = np.sqrt(p**2 + 1)
    ln_I = get_ln_I() # Get tabulated values of ln(I)
    out = lnLee(T_e,n_e_free,p)
    partiallyIonizedSpecies = np.flatnonzero(N_e > 0)
    for i in partiallyIonizedSpecies:
        if Z[i] in atomicNumberList_I: # If tabulated value exists
            atomicSymbolInd = np.flatnonzero(Z[i] == atomicNumberList_I)
            atomicSymbolStr = atomicSymbolsList_I[atomicSymbolInd] + str(int(Z0[i]))
            I_j = np.exp(ln_I[atomicSymbolStr.item()]) # Mean excitation energy in units of electron rest energy
            # Else statement for extrapolation formula for I_j goes here
            h_j = p*np.sqrt(gamma - 1)/I_j
            k = 5 # Ad-hoc interpolation parameter
            out += n_j[...,i]/n_e_free * N_e[i] * (1/k*np.log(1 + h_j**k) - (p/gamma)**2)
    return out/lnLee(T_e,n_e_free,p)

# The effective critical momentum at which to evaluate slowing-down and deflection frequencies, evaluated iteratively

def calc_pStar(Z,Z0,n_j,T_e,E,pGuess = None,reltol = 1e-3,maxIter = 10):
    """
    Compute effective normalized momentum pStar used when evaluating nu_Dbar and nu_Sbar.
    pStar is evaluated iteratively.

    :param array_like Z: 1D array of charge numbers.
    :param array_like Z0: 1D array of charge states.
    :param array_like n_j: Charge state distribution. Last axis must match the size of Z and Z0.
    :param array_like T_e: Electron temperature.
    :param array_like E: Electric field in units of Ec.
    :param array_like pGuess: Initial guess when evaluating pStar (default: (1+Zeff)^(1/4)/sqrt(E)).
    :param float reltol: Relative tolerence when iterating (default: 1e-3).
    :param int maxIter: Maximum number of iterations when evaluating pStar (default: 10).
    """
    n_e_free = np.sum(Z0*n_j,axis = -1) # Free electron energy
    Z_eff = np.sum(Z0**2*n_j,axis = -1)/n_e_free # Effective charge
    if pGuess is None: # Initial guess is complete screening limit
        pGuess = np.sqrt(np.sqrt(1+Z_eff)/E)

    pStar0 = pGuess

    for i in range(0,maxIter):
        nu_Sbar = calc_nu_Sbar(Z,Z0,n_j,T_e,pStar0)
        nu_Dbar = calc_nu_Dbar(Z,Z0,n_j,T_e,pStar0)
        pStar1 = (nu_Sbar*nu_Dbar)**(1/4)/np.sqrt(E)
        relError = np.linalg.norm(pStar1-pStar0)/np.linalg.norm(pStar0)
        if relError < reltol:
            return pStar1
        pStar0 = pStar1
    warnings.warn('Maximum number of iterations exceeded, pStar not converged. Returning value from last iteration')
    return pStar1

# Effective critical electric field as defined in Hesslow et al. PPCF (2018) in units of Ectot. The effective critical electric field is
# obtained by iteration between equations (23) and (24) until convergence. Note that the large momentum approximation may break down 
# at high densities

def calc_Eceff(Z,Z0,n_j,T_e,B,reltol = 1e-3,maxIter = 10):
    """
    Compute effective electric field. Ec_eff is evaluated iteratively.

    :param array_like Z: 1D array of charge numbers.
    :param array_like Z0: 1D array of charge states.
    :param array_like n_j: Charge state distribution. Last axis must match the size of Z and Z0.
    :param array_like T_e: Electron temperature.
    :param array_like B: Magnetic field used to compute synchrotron radiation [T].
    :param float reltol: Relative tolerence when iterating (default: 1e-3).
    :param int maxIter: Maximum number of iterations when evaluating pStar (default: 10).
    """
    global alpha # Fine struture constant

    N_e = Z - Z0 # Number of bound electrons
    n_e_free = np.sum(Z0*n_j,axis = -1) # Free electron density
    n_e_tot = np.sum(Z*n_j,axis = -1) # Total electron density
    Z_eff = np.sum(Z0**2*n_j,axis = -1)/n_e_free # Effective charge
    Z_tot_eff = np.sum(Z**2*n_j,axis = -1)/n_e_free
    lnL = lnLc(T_e,n_e_free)
    ln_aBar = get_ln_aBar()
    ln_I = get_ln_I()

    # High momentum limit of nu_D and nu_S given in equations (7), (8), (11), and (12)
    # in Hesslow et al. PPCF (2018)
    nu_D0 = 1 + Z_eff - 2/3*np.sum(N_e**2*n_j,axis = -1)/(n_e_free*lnL)
    nu_D1 = Z_tot_eff/lnL
    nu_S0 = 1
    nu_S1 = (1 + 3*np.sum(N_e*n_j,axis = -1)/n_e_free)/(2*lnL)

    partiallyIonizedSpecies = np.flatnonzero(N_e > 0)
    for i in partiallyIonizedSpecies:
        if Z[i] in atomicNumberList_I:
            atomicSymbolInd = np.flatnonzero(Z[i] == atomicNumberList_I)
            atomicSymbolStr = atomicSymbolsList_I[atomicSymbolInd] + str(int(Z0[i]))
            ln_I_j = ln_I[atomicSymbolStr.item()]
        if Z[i] in atomicNumberList_aBar:
            atomicSymbolInd = np.flatnonzero(Z[i] == atomicNumberList_I)
            atomicSymbolStr = atomicSymbolsList_I[atomicSymbolInd] + str(int(Z0[i]))
            ln_aBar_j = ln_aBar[atomicSymbolStr.item()]
        else:
            ln_aBar_j = np.log(3/(2*alpha)*(np.pi/3)**(1/3)*N_e[i]**(2/3)/Z[i])
        nu_D0 += n_j[...,i]/n_e_free * (Z[i]**2 - Z0[i]**2) * ln_aBar_j/lnL
        nu_S0 += n_j[...,i]/n_e_free * N_e[i] * (-1 - ln_I_j)/lnL
    # Synchrotron radiation-damping time scale in units of tau_c
    tauRadInv = B**2/(n_e_free/1e20)/(15.44*lnL)

    # Coefficients for bremsstrahlung (see eqn (18) and (24) in Hesslow et al. PPFC (2018))
    phi_b1 = alpha*Z_tot_eff/lnL*0.35
    phi_b2 = alpha*Z_tot_eff/lnL*0.20

    p_c0 = nu_D0/(2*nu_S1)

    # Eceff in units of E_c (equations (23) and (24) in Hesslow et al. PPFC (2018))
    lambda_Eceff = lambda delta : nu_S0 + nu_S1*((1 + nu_D1/nu_D0)*np.log(p_c0) + np.sqrt(2*delta + 1))
    lambda_delta = lambda Eceff : (nu_D0/nu_S1**2)*(nu_D0*tauRadInv/Eceff + phi_b1 + phi_b2*np.log(p_c0))

    Ectot = n_e_tot/n_e_free # In units of E_c
    Eceff0 = Ectot
    delta = lambda_delta(Eceff0)
    for i in range(0,maxIter): # Iterate until convergence
        delta = lambda_delta(Eceff0)
        Eceff1 = lambda_Eceff(delta)
        Eceff0 = Eceff1
        relError = np.linalg.norm(Eceff1-Eceff0)/np.linalg.norm(Eceff0)
        if relError < reltol:
            return Eceff1/Ectot
        
    warnings.warn('Maximum number of iterations exceeded, Eceff not converged. Returning value from last iteration')
    return Eceff1/Ectot

