"""
The key thing is that for all simulations, we need to keep track of all experimental parameters used,
and to save all samples produced by the D-Wave QPU. 
This saving of data means we can go back and analyze data as needed. 
"""
import networkx as nx
from dwave.cloud import Client
import zlib
import json
import math
import os
import minorminer.subgraph


def Start_DWave_connection(device):
    client = Client.from_config()
    DWave_solver = client.get_solver(device)
    A = DWave_solver.undirected_edges
    connectivity_graph = nx.Graph(list(A))
    return connectivity_graph, DWave_solver
def run_DWave(params, path, h, J, solver):
	"""
	Runs ising model on D-Wave. 
	Saves data to compressed file on disk. 
	
	params: the D-Wave parameter dictionary
	path: a string which is a relative path from the python script. 
	h: h dictionary
	J: J dictionary
	solver: D-Wavw solver object
	"""
	while (True):
		try:
			sampleset = solver.sample_ising(h, J, answer_mode='raw', **params)
			vectors = sampleset.samples
			QPU_time = sampleset['timing']['qpu_access_time']/float(1000000)
			break
		except Exception as e:
			print(e)
			print("fail", flush=True)
			time.sleep(2)
			continue
	file = open(path+"_QPU_time.txt", "w")
	file.write(str(QPU_time))
	file.close()
	print("QPU_time", QPU_time)
	###############################################
	file = open(path+"_solutions.json", "w")
	json.dump(vectors, file)
	file.close()

	filename_in = path+"_solutions.json"
	filename_out = path+"_solutions.json.zip"
	with open(filename_in, mode="rb") as fin, open(filename_out, mode="wb") as fout:
		data = fin.read()
		compressed_data = zlib.compress(data, zlib.Z_BEST_COMPRESSION)
		fout.write(compressed_data)
	os.remove(filename_in)
def read_results(path):
	"""
	reads in all of the D-Wave samples from the compressed zip file that we have written to previously
	"""
	with open(path+"_solutions.json.zip", mode="rb") as fin:
		data = fin.read()
		decompressed = zlib.decompress(data)
		data = json.loads(decompressed)
	return data

#Rename data directory as needed
HEADER = "DWave_data/your_local_directory/"
#Note: make sure to not over-write existing data on disk
#Use the filenames to encode the simulation parameters, so that the data can be uniquely tracked


graph, solver = Start_DWave_connection("Avdantage_system4.1")
params = {"num_reads": 1000, "auto_scale": False, "annealing_time": 1}

#h = {}
#J = {}
#run_DWave(params, HEADER, h, J, solver)