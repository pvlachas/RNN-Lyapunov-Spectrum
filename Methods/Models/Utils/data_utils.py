#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import pickle
import time
from time_utils import *

##############################################################
### SAVING AND LOADING FILES PROTOCOLS
##############################################################

def loadDataTorch():
	return torch.load("loadDataTorch")
def saveData(data, data_path, protocol):
	assert(protocol in ["pickle"])
	if protocol=="pickle":
		saveDataPickle(data, data_path)
	else:
		raise ValueError("Invalid protocol.")
	return 0

def loadData(data_path, protocol):
	assert(protocol in ["pickle"])
	if protocol=="pickle":
		return loadDataPickle(data_path)
	else:
		raise ValueError("Invalid protocol.")

def saveDataPickle( data, data_path):
	data_path += ".pickle"
	with open(data_path, "wb") as file:
		# Pickle the "data" dictionary using the highest protocol available.
		pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
		del data
	return 0

def loadDataPickle(data_path):
	data_path += ".pickle"
	print("Loading Data...")
	time_load = time.time()
	try:
		with open(data_path, "rb") as file:
			data = pickle.load(file)
	except Exception as inst:
		print("Datafile\n {:s}\nNOT FOUND.".format(data_path))
		raise ValueError(inst)
	time_load_end = time.time()
	printTime(time_load_end - time_load)
	return data

def printTime(seconds):
	time_str = secondsToTimeStr(seconds)
	print("Time passed: {:}".format(time_str))
	return 0


