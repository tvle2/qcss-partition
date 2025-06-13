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
    """
    Compute the total energy of a 2D Ising lattice with periodic boundary conditions.

    Parameters
    ----------
    latt : np.ndarray, shape (N, N)
        2D array of spins (+1/-1).

    Returns
    -------
    float
        The total energy of the lattice configuration.
    """
    Ene = 0
    for i in range(N):
        for j in range(N):
            S = latt[i,j]
            WF = latt[i, (j+1)%N] + latt[i, (j-1)%N] + latt[(i+1)%N, j] + latt[(i-1)%N, j]
            Ene += -S*WF
    return int(Ene/2)

def RandomL(N):
    """
    Generate a random N x N Ising spin lattice with spins (+1 / -1).

    Parameters
    ----------
    N : int
        Size of the lattice.

    Returns
    -------
    np.ndarray, shape (N, N)
        Randomly initialized lattice of spins (+1 or -1).
    """
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
    """
    Compute the exact partition function by brute-force enumeration of all possible spin configurations.

    Parameters
    ----------
    N : int
        Lattice size (N x N).
    T : float
        Temperature.

    Returns
    -------
    float
        Partition function Z(T) for the given temperature.
    """
    n_sites = N*N
    Z = 0.0
    for spins in itertools.product([-1, 1], repeat=n_sites):
        latt = np.array(spins).reshape((N, N))
        E = CEnergy(latt, N)
        Z += math.exp(-E / T)
    return Z

def mhr_partition_function(kT, E, fe, kT_new):
    """
    Estimate the partition function at a new temperature using the Multiple Histogram Reweighting (MHR) method.

    Parameters
    ----------
    kT : Temperatures used for sampling (array of floats).
    E : list of np.ndarray
        List of arrays of sampled energies for each temperature in kT.
    fe : np.ndarray
        Relative free energies for each temperature in kT.
    kT_new : float
        The target temperature at which to estimate the partition function.

    Returns
    -------
    float
        Estimated partition function at temperature kT_new.
    """
    nrun = len(kT)
    N = [len(e) for e in E]
    Z = 0.0
    for n in range(nrun):
        for i in range(N[n]):
            E_ni = E[n][i]
            # num = np.exp(-E_ni/kT_new - fe[n])
            num = np.exp(-E_ni/kT_new)
            denom = 0.0
            for m in range(nrun):
                denom += N[m] * np.exp(-E_ni/kT[m] + fe[m])
            Z += num / denom
    return Z

def free_energies(b, X, nmaxit=20, weights=None, argoffset=True):
    """
    Relation with partition function: $F = -k_b T \ln Z$, where $Z$ is the partition function.
    Calculate the relative free energies for MHR
        b (2 dimensional array)
        X (list of 3-dimensional arrays)
        nmaxit (int) : Number of iterations
        weights (list of arrays): Weights to be applied to each data point in `X`;

    Returns:
        array: Free energies
    """
    nrun = len(b)        
    assert len(X) == nrun, "Check 'b' and 'X' have matching lengths"
                      
    # Free energies for each data set
    fold = np.ones(nrun)
    f = np.zeros(nrun)

    # Number of data points in each data set
    ndata = np.zeros(nrun, dtype='int')
    for irun1 in range(nrun):
        ndata[irun1] = X[irun1].shape[0]

    D = 0.0
    Dcount = 0 
    if argoffset:
        for irun1 in range(nrun):
            for irun2 in range(nrun):
                for idata in range(ndata[irun2]):
                    D += np.dot(b[irun1,:], X[irun2][idata,:])
                    Dcount += 1
        D = -D / Dcount
       
    for n in range(nmaxit):
        for i in range(nrun):
            f[i] = 0.0
        
            for irun1 in range(nrun):
                for idata in range(ndata[irun1]):
                    # Find maximum argument for stability
                    max_val = -np.inf
                    for irun2 in range(nrun):
                        arg_val = np.dot(b[irun2,:], X[irun1][idata,:]) + fold[irun2] + D
                        if arg_val > max_val:
                            max_val = arg_val
                    
                    # Compute denominator with stabilization
                    denom_val = 0.0
                    for irun2 in range(nrun):
                        arg_val = np.dot(b[irun2,:], X[irun1][idata,:]) + fold[irun2] + D
                        denom_val += ndata[irun2] * np.exp(arg_val - max_val)
                    
                    # Compute numerator argument
                    num_arg = np.dot(b[i,:], X[irun1][idata,:]) + D
                    
                    # Compute stable term: exp(num_arg - max_val) / denom_val
                    term = np.exp(num_arg - max_val) / denom_val
                    
                    if weights is None:
                        f[i] += term
                    else:
                        f[i] += weights[irun1][idata] * term

        fold[:] = -np.log(f[:])
        fold = fold - fold.min()
        
    return fold

def save_csv(filename, xvals, yvals, xlabel='x', ylabel='y'):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([xlabel, ylabel])
        for x, y in zip(xvals, yvals):
            writer.writerow([x, y])


def main():
    N = 4 
    Tc = 2.269
    delta = 1.5
    npoints = 20
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

    # Brute-force enumerate partition function 
    print(f"Enumerating all {2**(N*N):,} configurations for N={N} ...")
    Z_exact = []
    for T in Te:
        Z_exact.append(enumerate_partition_function(N, T))
    Z_exact = np.array(Z_exact)
    save_csv("results/partition_exact.csv", Te, Z_exact, xlabel='Temperature', ylabel='Partition_Exact')

    Z_mhr = []
    print("Estimating partition function Z(T) with MHR...")
    for T in Te:
        Z_mhr.append(mhr_partition_function(kT, E, fe, T))
    Z_mhr = np.array(Z_mhr)
    scale = Z_exact[0] / Z_mhr[0]
    Z_mhr = Z_mhr * scale
    save_csv("results/partition_mhr.csv", Te, Z_mhr, xlabel='Temperature', ylabel='Partition_MHR')
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