import numpy as np
from numpy import random
from numba import jit
import matplotlib.pyplot as plt
import itertools
import math
import os

# ----- Wang-Landau Core Functions -----
@jit(nopython=True)
def CEnergy(latt):
    N = np.shape(latt)[0]
    Ene = 0
    for i in range(N):
        for j in range(N):
            S = latt[i,j]
            WF = latt[(i+1)%N,j]+latt[i,(j+1)%N]+latt[(i-1)%N,j]+latt[i,(j-1)%N]
            Ene += -S * WF
    return Ene/2.

def RandomL(N):
    return np.array(np.sign(2*random.random((N,N))-1),dtype=int) 

def PrepareEnergies(N):
    Energies = (np.array(4*np.arange(-int(N*N/2),int(N*N/2)+1),dtype=int)).tolist()
    Energies.pop(1)  
    Energies.pop(-2) 
    Energies = np.array(Energies) 
    Emin, Emax = Energies[0],Energies[-1]
    indE = -np.ones(Emax+1-Emin, dtype=int)
    for i,E in enumerate(Energies):
        indE[E-Emin]=i
    return (Energies, indE, Emin)

def WangLandau(Nitt, N, flatness):
    (Energies, indE, Emin) = PrepareEnergies(N)
    latt = RandomL(N)
    lngE, Hist = RunWangLandau(Nitt,Energies,latt,indE,flatness)
    return (Energies,lngE, Hist)

@jit(nopython=True)
def RunWangLandau(Nitt,Energies,latt,indE,flatness):
    N   = len(latt)
    Ene = int(CEnergy(latt))
    Emin, Emax = Energies[0],Energies[-1]
    lngE = np.zeros(len(Energies))
    Hist = np.zeros(len(Energies))
    lnf = 1.0
    N2 = N*N
    for itt in range(Nitt):
        t = int(random.rand()*N2)
        (i, j) = (int(t/N), t%N)
        S = latt[i,j]
        WF = latt[(i+1)%N,j]+latt[i,(j+1)%N]+latt[(i-1)%N,j]+latt[i,(j-1)%N]
        Enew = Ene + int(2*S*WF)
        lgnew = lngE[indE[Enew-Emin]]
        lgold = lngE[indE[Ene-Emin]]
        P = 1.0
        if lgold-lgnew < 0 : P=np.exp(lgold-lgnew)
        if P > random.rand():
            latt[i,j] = -S
            Ene = Enew
        Hist[indE[Ene-Emin]] += 1
        lngE[indE[Ene-Emin]] += lnf
        if (itt+1) % 1000 == 0:
            aH = np.sum(Hist)/N2
            mH = np.min(Hist)
            if mH > aH*flatness:
                Hist[:] = 0
                lnf /= 2.
    return (lngE, Hist)

def partition_function(Energies, lngE, T):
    Energies = np.array(Energies)
    lngE = np.array(lngE)
    if np.isscalar(T):
        Z = np.sum(np.exp(lngE - Energies / T))
        return Z
    else:
        Zs = []
        for temp in T:
            Zs.append(np.sum(np.exp(lngE - Energies / temp)))
        return np.array(Zs)

def enumerate_partition_function(N, T):
    n_sites = N*N
    Z = 0.0
    for spins in itertools.product([-1, 1], repeat=n_sites):
        latt = np.array(spins).reshape((N, N))
        E = CEnergy(latt)
        Z += math.exp(-E / T)
    return Z

def normalize_lngE(lngE):
    # For N=4, ground state degeneracy = 4
    if lngE[-1]>lngE[0]:
        lgC = np.log(4)-lngE[-1]-np.log(1+np.exp(lngE[0]-lngE[-1]))
    else:
        lgC = np.log(4)-lngE[0]-np.log(1+np.exp(lngE[-1]-lngE[0]))
    return lngE + lgC

def main():
    N = 4
    Nitt = int(1e9)  # Number of Wang-Landau iterations 
    # Flatness values to sweep
    flatness_list = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

    Tc = 2.269
    delta = 1.5
    Te = np.linspace(Tc - delta, Tc + delta, 5)

    os.makedirs("results", exist_ok=True)

    # Compute exact
    print(f"Enumerating all {2**(N*N):,} configs for N={N} ...")
    Z_exact = []
    for T in Te:
        Z_exact.append(enumerate_partition_function(N, T))
    Z_exact = np.array(Z_exact)

    all_Z = []
    for flatness in flatness_list:
        print(f"\nRunning Wang-Landau for flatness={flatness} ...")
        Energies, lngE, Hist = WangLandau(Nitt, N, flatness)
        lngE = normalize_lngE(lngE)
        Z_wl = partition_function(Energies, lngE, Te)
        all_Z.append(Z_wl)
        mse = np.mean((Z_wl - Z_exact) ** 2)
        print(f"  MSE vs exact: {mse:.2e}")

        data = np.column_stack((Te, Z_wl, Z_exact))
        header = "Temperature,Z_WL,Z_Exact"
        fname = f"results/wl_partition_flatness_{flatness:.2f}.csv"
        np.savetxt(fname, data, delimiter=",", header=header, comments='')
        print(f"  Saved data for flatness={flatness} to {fname}")

    plt.figure(figsize=(12,4))
    bar_width = 0.8 / (len(flatness_list) + 1)
    x = np.arange(len(Te))
    colors = plt.cm.viridis(np.linspace(0, 1, len(flatness_list)))

    # Plot all bars
    for i, flatness in enumerate(flatness_list):
        values = np.array(all_Z[i]) / Z_exact
        plt.bar(x + (i - len(flatness_list)/2)*bar_width, values,
                width=bar_width, color=colors[i],
                label=f'WL flatness={flatness:.2f}', alpha=0.9)

    # Plot exact bars at 1
    plt.bar(x + (len(flatness_list)/2)*bar_width, np.ones_like(Z_exact), 
            width=bar_width, color='k', label='Exact Enumeration', alpha=0.7)

    # the closest bar for each temperature
    for idx_T, T in enumerate(Te):
        norm_values = [np.array(all_Z[i])[idx_T] / Z_exact[idx_T] for i in range(len(flatness_list))]
        min_idx = np.argmin(np.abs(np.array(norm_values) - 1))
        highlight_x = x[idx_T] + (min_idx - len(flatness_list)/2)*bar_width
        highlight_val = norm_values[min_idx]
        plt.bar(highlight_x, highlight_val, width=bar_width, 
                edgecolor='red', linewidth=2.5, fill=False, zorder=5)  # highlight with red edge

    plt.xticks(x, [f"{t:.2f}" for t in Te])
    plt.xlabel("Temperature $T$")
    plt.ylabel(r"Normalized $Z_{WL}/Z_{exact}$")
    plt.title("WL Partition Function")
    plt.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("results/wl_flatness_sweep.png")
    plt.close()




if __name__ == "__main__":
    main()