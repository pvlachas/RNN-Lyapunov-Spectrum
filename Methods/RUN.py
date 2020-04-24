#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import sys
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from plotting_utils import *
from global_utils import *

import argparse


def getModel(params):
	sys.path.insert(0, global_params.py_models_path.format(params["model_name"]))
	if params["model_name"] == "rnn":
		import rnn as model
		return model.rnn(params)
	else:
		raise ValueError("model not found.")

def runModel(params_dict):
	if params_dict["mode"] in ["train", "all"]:
		trainModel(params_dict)
	if params_dict["mode"] in ["test", "all"]:
		testModel(params_dict)
	if params_dict["mode"] in ["test", "postprocess", "all"]:
		postprocessModel(params_dict)
	if params_dict["mode"] in ["lyapunov"]:
		calculateLyapunovSpectrum(params_dict)
	if params_dict["mode"] in ["lyapunov", "plotLyapunov"]:
		plotLyapunovSpectrum(params_dict)
	return 0

def plotLyapunovSpectrum(params_dict):
	model = getModel(params_dict)
	model.plotLyapunovSpectrum()
	model.delete()
	del model
	return 0
	
def calculateLyapunovSpectrum(params_dict):
	model = getModel(params_dict)
	model.calculateLyapunovSpectrum()
	model.delete()
	del model
	return 0

def trainModel(params_dict):
	model = getModel(params_dict)
	model.train()
	model.delete()
	del model
	return 0

def testModel(params_dict):
	model = getModel(params_dict)
	model.testing()
	model.delete()
	del model
	return 0

def postprocessModel(params_dict):
	model = getModel(params_dict)
	model.postprocess()
	model.delete()
	del model
	return 0
	
def defineParser():
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(help='Selection of the model.', dest='model_name')
	rnn = subparsers.add_parser("rnn")
	rnn = getMDARNNParser(rnn)
	return parser

def main():
	parser = defineParser()
	args = parser.parse_args()
	print(args.model_name)
	args_dict = args.__dict__

	# for key in args_dict:
		# print(key)

	# DEFINE PATHS AND DIRECTORIES
	args_dict["saving_path"] = global_params.saving_path.format(args_dict["system_name"])
	args_dict["model_dir"] = global_params.model_dir
	args_dict["fig_dir"] = global_params.fig_dir
	args_dict["results_dir"] = global_params.results_dir
	args_dict["logfile_dir"] = global_params.logfile_dir
	args_dict["train_data_path"] = global_params.training_data_path.format(args.system_name, args.N)
	args_dict["test_data_path"] = global_params.testing_data_path.format(args.system_name, args.N)
	args_dict["worker_id"] = 0

	runModel(args_dict)

if __name__ == '__main__':
	main()

