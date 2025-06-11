import numpy as np
import math
import random as r
from numba import jit
import itertools
import matplotlib.pyplot as plt
import csv
import os
import itertools

np.random.seed(42)
r.seed(42)


# --- Lattice and energy functions ---
sign = lambda x : math.copysign(1,x)

@jit(nopython=True)
def CEnergy(latt, N):
    Ene = 0
    for i in range(N):
        for j in range(N):
            S = latt[i,j]
            WF = latt[i, (j+1)%N] + latt[i, (j-1)%N] + latt[(i+1)%N, j] + latt[(i-1)%N, j]
            Ene += -S*WF
    return int(Ene/2)

def RandomL(N):
    latt = np.zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            latt[i,j] = sign(2*r.random() - 1)
    return latt

def metropolis_energy_samples(N, T, latt, nsamples=10000, therm_steps=10000, sweep_steps=100, sample_gap=10):
    """
    Generate thermalized energy samples using Metropolis MC for Ising model.
    Parameters:
      N           : lattice size (N x N)
      T           : temperature
      nsamples    : number of energy samples to collect
      therm_steps : equilibration sweeps before sampling
      sweep_steps : MC sweeps between each sample (decorrelation)
      sample_gap  : How many MC sweeps between storing samples (controls autocorrelation)
    Returns:
      energies    : array of energies
    """
    # Thermalize
    for step in range(therm_steps):
        for _ in range(N*N): # one sweep = N^2 single-spin attempts
            i, j = np.random.randint(0, N, size=2)
            S = latt[i, j]
            WF = latt[i, (j+1)%N] + latt[i, (j-1)%N] + latt[(i+1)%N, j] + latt[(i-1)%N, j]
            dE = 2 * S * WF
            if dE <= 0 or np.random.rand() < math.exp(-dE/T):
                latt[i, j] = -S
    # Sampling
    energies = []
    for sample in range(nsamples):
        for _ in range(sample_gap*N*N): # decorrelate
            i, j = np.random.randint(0, N, size=2)
            S = latt[i, j]
            WF = latt[i, (j+1)%N] + latt[i, (j-1)%N] + latt[(i+1)%N, j] + latt[(i-1)%N, j]
            dE = 2 * S * WF
            if dE <= 0 or np.random.rand() < math.exp(-dE/T):
                latt[i, j] = -S
        energies.append(CEnergy(latt, N))
    return np.array(energies)



def enumerate_partition_function(N, T):
    n_sites = N*N
    Z = 0.0
    for spins in itertools.product([-1, 1], repeat=n_sites):
        latt = np.array(spins).reshape((N, N))
        E = CEnergy(latt, N)
        Z += math.exp(-E / T)
    return Z

def mhr_partition_function(kT, E, fe, kT_new):
    """
    Estimate (relative) partition function at kT_new using MHR.
    """
    nrun = len(kT)
    N = [len(e) for e in E]
    Z = 0.0
    for n in range(nrun):
        for i in range(N[n]):
            E_ni = E[n][i]
            num = np.exp(-E_ni/kT_new - fe[n])
            denom = 0.0
            for m in range(nrun):
                denom += N[m] * np.exp(-E_ni/kT[m] + fe[m])
            Z += num / denom
    return Z


def free_energies(b, X, nmaxit=20, weights=None, argoffset=True):
    """
    Calculate the relative free energies for MHR
        b (2 dimensional array)
        X (list of 3-dimensional arrays)
        nmaxit (int) : Number of iterations
        weights (list of arrays): Weights to be applied to each data point in `X`;
        argoffset (bool) : If True then a constant is added to the arguments to the exponential functions in
          the equation given above for evaluating :math:`F^{(1)},F^{(2)}\dotsc` in order to prevent overflows
          and underflows (as much as is possible).

    Returns:
        array: Free energies,\dotsc` corresponding to :math:`\mathbf{b}^{(1)},\mathbf{b}^{(1)},\dotsc`. 
        Equivalently, `F[n]` is the free energy corresopnding to `b[n]`

    """

    # Relevant equation:
    # \exp(-F^{(n)}) = \sum_{p=1}^{R}\sum_{i=1}^{N_p}\frac{ \exp(\mathbf{b}^{(n)}\cdot\mathbf{X}_{pi}+D) }{\sum_{q=1}^{R}\exp(\mathbf{b}^{(q)}\cdot\mathbf{X}_{pi}+F^{(q)}+D)N_q}
    
    nrun = len(b)        
    assert len(X) == nrun, "Check 'b' and 'X' have matching lengths"
                      
    # Free energies for each data set
    fold = np.ones(nrun)
    f = np.zeros(nrun)

    # Number of data points in each data set
    ndata = np.zeros(nrun, dtype = 'int')
    for irun1 in range(nrun):
        ndata[irun1] = X[irun1].shape[0]

    D = 0.0
    Dcount = 0 
    if argoffset:
        for irun1 in range(nrun):
            for irun2 in range(nrun):
                for idata in range(ndata[irun2]):
                    D += np.dot(b[irun1,:],X[irun2][idata,:])
                    Dcount += 1
        D = - D / Dcount
       
        
    for n in range(nmaxit):
        for i in range(nrun):
            f[i] = 0.0
        
            # f = free_energy(...)
            for irun1 in range(nrun):
                for idata in range(ndata[irun1]):
                    # Accumulate the denominator
                    sum = 0.0
                    for irun2 in range(nrun):
                        arg = np.dot(b[irun2,:],X[irun1][idata,:]) + fold[irun2] + D
                        sum += 1.0*ndata[irun2]*np.exp(arg)

                    # 'sum' now the denominator in the equation given above
                    # Calculate the numerator and add the term to the sum
                    arg = np.dot(b[irun1,:],X[irun1][idata,:]) + D
                    if weights == None:
                        f[i] += (1.0/sum)*np.exp(arg)
                    else:
                        f[i] += (weights[irun1][idata]/sum)*np.exp(arg)

        fold[:] = -np.log(f[:])
        fold = fold - fold.min()
        
    return fold


def reweight_observable(b, X, obs, bnew, fe=None, weights=None, argoffset=True):

    """

    Arguments:
        b (2 dimensional array)
        X (list of 3-dimensional arrays)
        obs (list of 2-dimensional arrays): Values of the observables
        fe (array) (optional): Free energies
        weights (list of arrays): Weights to be applied to each data point in `X`;
        argoffset (bool) : If True then a constant is added to the arguments to the exponential functions
    Returns:
        float: The value of the observable, :math:`\langle O'\rangle`

    """

    # Relevant equation:
    # `\langle O'\rangle = \frac{ \sum_{n=1}^{R}\sum_{i=1}^{N_n}O_{ni}\exp( (\mathbf{b}'-\mathbf{b}^{(n)})\cdot\mathbf{X}_{ni}-F^{(n)}+C ) }{ \sum_{n=1}^{R}\sum_{i=1}^{N_n}\exp( (\mathbf{b}'-\mathbf{b}^{(n)})\cdot\mathbf{X}_{ni}-F^{(n)}+C )}`,

    
    nrun = len(b)
    assert len(X) == nrun, "Check 'X' and 'b' have matching lengths"
    assert len(obs) == nrun, "Check 'obs' and 'b' have matching lengths"

    ndata = np.zeros(nrun, dtype = 'int')

    # If free energies are specified then use them; if not then calculate them from scratch
    # using default parameters
    if fe == None:
        fe = free_energies(b, X, weights=weights)
    
    
    for irun1 in range(nrun):
        assert len(X[irun1]) == len(obs[irun1]), "Check 'X' and 'obs' have same shape"
        # Set ndata to the number of energy data points in 'e[i,:]' - for run 'i'
        ndata[irun1] = X[irun1].shape[0]

    # Determine 'C' if needed
    C = 0.0
    Ccount = 0
    if argoffset:
        for irun1 in range(nrun):
            for idata in range(ndata[irun1]):
                C += np.dot((bnew[:]-b[irun1,:]),X[irun1][idata,:])
                Ccount += 1
        C = - C / Ccount


    # Calculate denominator in the equation
    denom = 0.0
    for irun1 in range(nrun):
        for idata in range(ndata[irun1]):
            arg = np.dot((bnew[:]-b[irun1,:]),X[irun1][idata,:]) - fe[irun1]
            if weights == None:
                denom += np.exp(arg)
            else:
                denom += weights[irun1][idata]*np.exp(arg)
            
    # Calculate the value
    robs = 0.0
    for irun1 in range(nrun):
        for idata in range(ndata[irun1]):
            arg = np.dot((bnew[:]-b[irun1,:]),X[irun1][idata,:]) - fe[irun1]
            if weights == None:
                robs += (1.0/denom)*np.exp(arg)*obs[irun1][idata]
            else:
                robs += (weights[irun1][idata]/denom)*np.exp(arg)*obs[irun1][idata]
            
    return robs


def reweight_observable_nvt(kT, E, obs, kT_new, weights=None):

    r"""
    Calculate an observable in the NVT ensemble at a new temperature using MHR.
    Arguments:
        kT (array): Values of :math:`kT` for the various simulations, where :math:`k` is the Boltzmann
          constant and :math:`T` is the temperature.
        E (list of arrays): `E[n]` is an array containing the energies for the `n` th simulation;
          `E[n][i]` is the energy for the `i` th data point.
        obs (list of arrays): `obs[n][i]` is the observable corresponding to the `i` th data point
           in the `n` th simulation.
        kT_new (array): The :math:`kT` to be reweighted to
        weights (list of arrays): Weights to be applied to each data point in `X`;
          

    Returns:
        float: The value of the observable at `kT_new` calculated using MHR.

    """
    
    nrun = len(kT)
    
    b = []
    for n in range(nrun):
        b.append([ -1.0/kT[n] ])
    b = np.asarray(b)
    
    X = []
    for n in range(nrun):
        Xn = np.zeros( (len(E[n]),1) )
        Xn[:,0] = E[n]
        X.append(Xn)

    obs2 = []
    for n in range(nrun):
        obs2n = np.zeros( (len(obs[n]),1) )
        obs2n[:,0] = obs[n]
        obs2.append(Xn)

    bnew = np.asarray([-1.0/kT_new])
    
    return reweight_observable(b, X, obs, bnew, weights=weights)

def save_csv(filename, xvals, yvals, xlabel='x', ylabel='y'):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([xlabel, ylabel])
        for x, y in zip(xvals, yvals):
            writer.writerow([x, y])


def main():
    N = 5 
    Tc = 2.269
    delta = 1.5
    npoints = 30
    kT = np.linspace(Tc - delta, Tc + delta, npoints)
    nsamples = 5000
    therm_steps = 2000
    sample_gap = 10

    os.makedirs("results", exist_ok=True)

    latt = RandomL(N)
    E = []
    for idx, T in enumerate(kT):
        print(f"Sampling energies at T={T} ...")
        energies = metropolis_energy_samples(N, T, np.copy(latt), nsamples=nsamples, therm_steps=therm_steps, sample_gap=sample_gap)
        E.append(energies)
        # Plot and save histogram for each temperature
        plt.figure()
        plt.hist(energies, bins=40, alpha=0.7, color='b')
        plt.xlabel('Energy')
        plt.ylabel('Counts')
        plt.title(f'Ising Model Energy Histogram (T={T})')
        plt.tight_layout()
        plt.savefig(f"results/hist_T{T}.png")
        plt.close()

    # Prepare b and X for MHR
    print("Preparing data for MHR...")
    b = np.array([[-1.0/T] for T in kT])
    X = [energies[:, np.newaxis] for energies in E]
    print(f"Done preparing data for MHR with {len(kT)} temperatures.")
    
    # Compute relative free energies
    print("Calculating free energies...")
    fe = free_energies(b, X)
    print("Done calculating free energies.")

    # Estimate partition function at temperatures
    Te = np.linspace(Tc - delta, Tc + delta, 5)
    Z_mhr = []
    print("Estimating partition function Z(T) with MHR...")
    for T in Te:
        Z_mhr.append(mhr_partition_function(kT, E, fe, T))
    Z_mhr = np.array(Z_mhr)
    save_csv("results/partition_mhr.csv", Te, Z_mhr, xlabel='Temperature', ylabel='Partition_MHR')

    # Brute-force enumerate partition function (Warning: exponential cost!)
    print(f"Enumerating all {2**(N*N):,} configurations for N={N} ...")
    Z_exact = []
    for T in Te:
        Z_exact.append(enumerate_partition_function(N, T))
    Z_exact = np.array(Z_exact)
    save_csv("results/partition_exact.csv", Te, Z_exact, xlabel='Temperature', ylabel='Partition_Exact')

    mse_pt = (Z_mhr - Z_exact)**2
    save_csv("results/mse.csv", Te, mse_pt, xlabel='Temperature', ylabel='MSE')
    
    #Plot and save MHR and exact partition function
    plt.figure(figsize=(8,5))
    plt.plot(Te, Z_mhr, 'r-', lw=2, label="MHR (relative)")
    plt.plot(Te, Z_exact, 'k--', lw=1.8, label=f'Exact (N={N})')
    plt.plot(Te, mse_pt, 'b-.', label='Pointwise Squared Error')
    plt.xlabel("Temperature $T$")
    plt.ylabel("Partition Function $Z(T)$ (relative)")
    plt.title(f"Partition Function: MHR vs Exact (N={N})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/partition_MHR_vs_Exact.png")
    plt.close()

if __name__ == "__main__":
    main()