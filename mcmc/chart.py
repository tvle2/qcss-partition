import numpy as np
import matplotlib.pyplot as plt
import csv

def read_csv(filename):
    T = []
    Z = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            T.append(float(row[0]))
            Z.append(float(row[1]))
    return np.array(T), np.array(Z)

Te_mhr, Z_mhr = read_csv("results/partition_mhr.csv")
Te_exact, Z_exact = read_csv("results/partition_exact.csv")

if not np.allclose(Te_mhr, Te_exact):
    raise ValueError("Temperature arrays do not match between MHR and exact data.")

# Compute root square error (absolute difference) at each T
rse = np.abs(Z_mhr - Z_exact)

plt.figure(figsize=(8,5))
plt.plot(Te_mhr, rse, 'o-', lw=2)
plt.xlabel("Temperature $T$")
plt.ylabel("Root Square Error $|Z_{\mathrm{MHR}} - Z_{\mathrm{Exact}}|$")
plt.yscale('symlog', linthresh=1e-2)
plt.title("Root Square Error of Partition Function: MHR vs. Exact")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/mhr_exact.png")
plt.close()
