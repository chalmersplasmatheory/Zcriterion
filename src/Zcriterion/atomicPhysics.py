import numpy as np
import scipy as sp
import warnings
import Zcriterion.plasma as plasma

def coronalEquilibrium(n_Z,T_e,ionRate,recombRate,reltol = 1e-4, maxIter = 10):
    """
    Computes densities of ion charge states with self-consistent electron 
    density using ionization and recombination rates from openADAS.

    :param array_like n_Z: Total ion density for a single species (or list several).
    :param array_like T_e: Electron temperature.
    :param ADASrate ionRate: Ionization rate interpolator (or list thereof).
    :param ADASrate recombRate: Recombination rate interpolator (or list thereof).
    :param float reltol: Relative tolerence when iterating between electron density charge state distribution (default 1e-4).
    :param int maxIter: Maximum number of iterations when iterating between electron density charge state distribution (default 10).
    """

    # If only one species, convert to lists
    if not isinstance(n_Z,list):
        n_Z = [np.array(n_Z)]
        ionRate = [ionRate]
        recombRate = [recombRate]

    # Guess electron density (assume that everything is singly ionized)
    n_e = sum(n_Z)

    for iter in range(0,maxIter):
        Z0_Z = [] # Create list of charge states
        n_j_Z = [] # Crate list of densities for each charge state
        # For a given electron density and temperature, evaluate collisional-radiative equilibrium of 
        # charge states 
        for spec in range(0,len(ionRate)): 
            frac_temp, Z0_temp = coronalEquilibriumSingleSpecies(n_e,T_e,ionRate[spec],recombRate[spec])
            n_j_Z.append(frac_temp*n_Z[spec][...,np.newaxis])
            Z0_Z.append(Z0_temp)
        n_j = np.concatenate(n_j_Z,axis = -1) # Convert list of arrays to numpy array
        Z0 = np.concatenate(Z0_Z,axis = -1) # Convert list of arrays to numpy array
        n_e = np.sum(Z0*n_j,axis = -1) # Compute new electron density
        if iter >= 1:
            relError = np.linalg.norm(n_j-n_j0)/np.linalg.norm(n_j0)
            if relError < reltol:
                return n_j,Z0
        n_j0 = n_j
    warnings.warn('Maximum number of iterations exceeded, n_j not converged. Returning value from last iteration')
    return n_j,Z0


def coronalEquilibriumSingleSpecies(n_e,T_e,ionRate,recombRate):
    """
    Computes relative densities of ion charge states for a given electron density
    and electron temperature. Note that the electron density is not evaluated 
    self-consistently.

    :param array_like n_e: Electron density.
    :param array_like T_e: Electron temperature.
    :param ADASrate ionRate: Ionization rate interpolator.
    :param ADASrate recombRate: Recombination rate interpolator.
    """

    Z = ionRate.Z
    Z0 = np.arange(0,Z+1)
    # Pre-allocate ionization and recombination coefficients
    I = np.full(np.broadcast_shapes(np.array(n_e).shape,np.array(T_e).shape) + (Z,),np.nan)
    R = np.full(np.broadcast_shapes(np.array(n_e).shape,np.array(T_e).shape) + (Z,),np.nan)
    frac = np.full(np.broadcast_shapes(np.array(n_e).shape,np.array(T_e).shape) + (Z+1,),np.nan)
    # Interpolate rates for each charge state
    for i in Z0[0:-1]:
        I[...,i] = ionRate.eval(i,n_e,T_e)
        R[...,i] = recombRate.eval(i+1,n_e,T_e)

    # Evaluate stationary solution
    for i in Z0:
        term1 = np.cumprod(np.flip(R[...,0:i]/I[...,0:i],axis = -1),axis = -1)
        term2 = np.cumprod(I[...,i::]/R[...,i::],axis = -1)
        frac[...,i] = 1/(1 + np.sum(term1, axis = -1) + np.sum(term2,axis = -1))
    
    return frac.squeeze(), Z0


def radiatedPower(n_Z,T,ionRate,recombRate,lineRadRate,bremsRate = None,n_j = None,Z0 = None):
    """
    Computes the radiated power for a given list of ion densities and a given temperature.
    radiatedPower evaluates the charge state density and electron density self-consistently
    using coronalEquilibrium.

    :param array_like n_Z: Total ion density for a given species (or list thereof).
    :param array_like T_e: Electron temperature.
    :param ADASrate ionRate: Ionization rate interpolator (or list thereof).
    :param ADASrate recombRate: Recombination rate interpolator (or list thereof).
    :param ADASrate lineRadRate: Line radiation rate interpolator (or list thereof).
    :param ADASrate bremsRate: Bremsstrahlung rate interpolator (or list thereof) (optional).
    :param array_like n_j: Optional parameter to pass converged charge state distribution (default None).
    :param array_like Z0: Corresponding charge state (default None).
    """

    if not isinstance(n_Z,list):
        n_Z = [np.array(n_Z)]
        ionRate = [ionRate]
        recombRate = [recombRate]
        lineRadRate = [lineRadRate]
        if bremsRate is not None:
            bremsRate = [bremsRate]

    if n_j is None or Z0 is None:
        n_j,Z0 = coronalEquilibrium(n_Z,T,ionRate,recombRate)
    n_e = np.sum(n_j*Z0,axis = -1)
    PLT_list = []
    PRB_list = []

    # Evaluate line radiation rates and bremsstrahlung rates (if interpolator is passed).
    # L is combined radiation rate
    for spec in range(0,len(ionRate)):
        Z = lineRadRate[spec].Z
        PLT_Z = np.full(np.broadcast_shapes(np.array(n_e).shape,np.array(T).shape) + (Z+1,),np.nan)
        for Z0 in range(0,Z):
            PLT_Z[...,Z0] = lineRadRate[spec].eval(Z0,n_e,T)
        PLT_Z[...,-1] = np.zeros(np.shape(PLT_Z[...,0]))
        PLT_list.append(PLT_Z)
        L = np.concatenate(PLT_list,axis = -1)
        
        if bremsRate is not None:
            PRB_Z = np.full(np.broadcast_shapes(np.array(n_e).shape,np.array(T).shape) + (Z+1,),np.nan)
            PRB_Z[...,0] = np.zeros(np.shape(PRB_Z[...,0]))
            for Z0 in range(1,Z+1):
                PRB_Z[...,Z0] = bremsRate[spec].eval(Z0,n_e,T)
            PRB_list.append(PRB_Z)
            L += np.concatenate(PRB_list,axis = -1)
        
    P_rad = n_e*np.sum(n_j*L,axis = -1)
    return P_rad

def ohmicPower(n_Z,T,ionRate,recombRate,j0,n_j = None,Z0 = None):
    """
    Computes the Ohmic power for a given list of ion densities and a given temperature.
    ohmicPower evaluates the charge state density and electron density self-consistently
    using coronalEquilibrium.

    :param array_like n_Z: Total ion density for a given species (or list thereof).
    :param array_like T_e: Electron temperature.
    :param ADASrate ionRate: Ionization rate interpolator (or list thereof).
    :param ADASrate recombRate: Recombination rate interpolator (or list thereof).
    :param array_like j0: Ohmic current density.
    :param array_like n_j: Optional parameter to pass converged charge state distribution (default None).
    :param array_like Z0: Corresponding charge state (default None).
    """
    if n_j is None or Z0 is None:
        n_j,Z0 = coronalEquilibrium(n_Z,T,ionRate,recombRate)
    n_e = np.sum(n_j*Z0,axis = -1)
    Z_eff = np.sum(Z0**2*n_j,axis = -1)/n_e
    sigma = plasma.calc_spitzerCond(T,n_e,Z_eff)
    P_ohm = j0**2/sigma
    return P_ohm

def powerBalance(n_Z,T,j0,ionRate,recombRate,lineRadRate,bremsRate = None,solveInLogScale = False):
    """
    Computes the power balance between radiated power and Ohmic power for a given list of ion densities 
    and a given temperature. The power balance can be returned in both logarithmic and linear scale.

    :param array_like n_Z: Total ion density for a given species (or list thereof).
    :param array_like T_e: Electron temperature.
    :param array_like j0: Ohmic current density.
    :param ADASrate ionRate: Ionization rate interpolator (or list thereof).
    :param ADASrate recombRate: Recombination rate interpolator (or list thereof).
    :param ADASrate lineRadRate: Line radiation rate interpolator (or list thereof).
    :param ADASrate bremsRate: Bremsstrahlung rate interpolator (or list thereof) (optional).
    :param bool solveInLogScale: Parameter that deterimnes whether to return log10(P_ohm/P_rad) or P_ohm - P_rad (default False).
    """

    n_j,Z0 = coronalEquilibrium(n_Z,T,ionRate,recombRate) # Compute and pass charge state distribution when evaluating P_ohm and P_rad
    P_ohm = ohmicPower(n_Z,T,ionRate,recombRate,j0,n_j = n_j,Z0 = Z0)
    P_rad = radiatedPower(n_Z,T,ionRate,recombRate,lineRadRate,bremsRate = bremsRate,n_j = n_j,Z0 = Z0)
    if solveInLogScale:
        return np.log10(P_ohm/P_rad)
    else:
        return P_ohm - P_rad

def equilibriumTemperature(n_Z,j0,T_guess,ionRate,recombRate,lineRadRate,bremsRate = None,solveInLogScale = True,solver = 'brentq'):
    """
    Computes the temperature assuming power balance between radiated power and Ohmic power for a given current 
    density and list of ion densities and a given temperature.

    :param array_like n_Z: Total ion density for a given species (or list thereof).
    :param array_like j0: Ohmic current density.
    :param array_like T_guess: Guess for temperature solution
    :param ADASrate ionRate: Ionization rate interpolator (or list thereof).
    :param ADASrate recombRate: Recombination rate interpolator (or list thereof).
    :param ADASrate lineRadRate: Line radiation rate interpolator (or list thereof).
    :param ADASrate bremsRate: Bremsstrahlung rate interpolator (or list thereof) (optional).
    :param bool solveInLogScale: Parameter that deterimnes whether to solve for T in log-log space (default True).
    :param str solver: Parameters that determines which solver is used. Possible values are 'brentq' (recommended) and 'fsolve' (default 'brentq')
    """

    if not isinstance(n_Z,list):
        n_Z = [np.array(n_Z)]
        ionRate = [ionRate]
        recombRate = [recombRate]
        lineRadRate = [lineRadRate]
        if bremsRate is not None:
            bremsRate = [bremsRate]

    # Preallocate array of solutions
    Tsol = np.full(T_guess.size,np.nan)
    a = 2e-1 # Lower temperature bound
    b0 = 1e4 # Upper temperature bound

    for i in range(0,T_guess.size):
        # TODO: Fix for arbitrary number of species
        n_Z_temp = []
        for j in range(0,len(ionRate)):
            n_Z_temp.append(n_Z[j][i])
        if solver.lower() == 'fsolve': # Not recommended
            if solveInLogScale:
                Tsol[i] = sp.optimize.fsolve(lambda T: powerBalance(n_Z_temp,10**T,j0,ionRate,recombRate,lineRadRate,bremsRate,solveInLogScale = solveInLogScale),np.log10(T_guess[i]))
            else:
                Tsol[i] = sp.optimize.fsolve(lambda T: powerBalance(n_Z_temp,T,j0,ionRate,recombRate,lineRadRate,bremsRate,solveInLogScale = solveInLogScale),T_guess[i])
        elif solver.lower() == 'brentq':
            # The brentq solver is based on the bisection method, and needs an interval where the function changes sign.
            # P_ohm typically decreases monotonically with T, whereas P_rad has a number of peaks. It is reasonable to assume
            # that between two critical points in P_rad there is at most one solution to P_ohm = P_rad. We are looking for the 
            # smallest solution, so we find all critical points and look for the solution between a and the first critical point
            # where P_rad > P_ohm.
            Tvec = np.geomspace(a,b0,100)
            PradVec = radiatedPower(n_Z_temp,Tvec,ionRate,recombRate,lineRadRate,bremsRate)
            bndPts = np.array(a)
            bndPts = np.append(bndPts,findCriticalPoints(Tvec,PradVec))
            bndPts = np.append(bndPts,b0)
  
            pwrBalanceAtBndPts = powerBalance(n_Z_temp,bndPts,j0,ionRate,recombRate,lineRadRate,bremsRate)
            signChangeInd = np.flatnonzero(pwrBalanceAtBndPts[0]*pwrBalanceAtBndPts < 0)
            if signChangeInd.size == 0:
                b = b0
            else:
                b = bndPts[signChangeInd[0]]
            if solveInLogScale:
                Tsol[i] = sp.optimize.brentq(lambda T: powerBalance(n_Z_temp,10**T,j0,ionRate,recombRate,lineRadRate,bremsRate,solveInLogScale = solveInLogScale),np.log10(a),np.log10(b),rtol = 1e-6)
            else:
                Tsol[i] = sp.optimize.brentq(lambda T: powerBalance(n_Z_temp,T,j0,ionRate,recombRate,lineRadRate,bremsRate,solveInLogScale = solveInLogScale),a,b,rtol = 1e-6)
        else: # TODO: implement vectorized bisection method, should be faster during parameter scans
            print(f'Unknown solver: {solver}')
    if solveInLogScale:
        return 10**Tsol
    else:
        return Tsol
    
def findCriticalPoints(x,f,axis = -1):
    """
    Finds the critical points of f(x) along an axis

    :param array_like x: Independent coordinate
    :param array_like f: The function for which to find the critical point
    :param int axis: Along which axis to find critcal points (default -1)
    """
    fprime = np.gradient(f,x,axis = axis)
    fprimeSign = np.sign(fprime) # Sign of derivative
    zeroInd = fprimeSign == 0 # Find if there is an exact critical point
    # Remove index from list and replace with value on the right
    while zeroInd.any():
        fprimeSign[zeroInd] = np.roll(fprimeSign, -1,axis = axis)[zeroInd]
        zeroInd = fprimeSign == 0
    # Find sign change
    critInd = (np.roll(fprimeSign,1,axis = axis) - fprimeSign) != 0
    # No sign change in the first index
    critInd[...,0] = False
    return x[critInd]