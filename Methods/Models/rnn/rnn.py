 #!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

# TORCH
import torch
import sys
print("-V- Python Version = {:}".format(sys.version))
print("-V- Torch Version = {:}".format(torch.__version__))
from torch.autograd import Variable

# LIBRARIES
import numpy as np
import socket
import os

from time_utils import *
from data_utils import *
from global_utils import *
from plotting_utils import *

import random
import cmath
import time
from tqdm import tqdm

# MEMORY TRACKING
import psutil
import subprocess

from rnn_model import *
from lyapunov_spectrum_wrapper import *

# PRINTING
from functools import partial
print = partial(print, flush=True)

SAVE_FORMAT = "pickle"

def get_gpu_memory_map():
	"""Get the current gpu usage.

	Returns
	-------
	usage: dict
		Keys are device ids as integers.
		Values are memory usage as integers in MB.
	"""
	result = subprocess.check_output(
		[
			'nvidia-smi', '--query-gpu=memory.used',
			'--format=csv,nounits,noheader'
		], encoding='utf-8')
	gpu_memory = [int(x) for x in result.strip().split('\n')]
	gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))

	result_ = subprocess.check_output(
		[
			'nvidia-smi'
		], encoding='utf-8')
	# Convert lines into a dictionary
	# print(result_)
	return gpu_memory_map


class rnn():
	def __init__(self, params):
		super(rnn, self).__init__()
		self.start_time = time.time()
		self.params = params.copy()

		self.system_name = params["system_name"]

		self.reference_train_time = getReferenceTrainingTime(params["reference_train_time"], params["buffer_train_time"])
		self.gpu = torch.cuda.is_available()

		# SETTING DEFAULT DATATYPE:
		if self.gpu:
			self.torch_dtype = torch.cuda.DoubleTensor
			torch.set_default_tensor_type(torch.cuda.DoubleTensor)
			if self.params["cudnn_benchmark"]: torch.backends.cudnn.benchmark = True
		else:
			self.torch_dtype = torch.DoubleTensor
			torch.set_default_tensor_type(torch.DoubleTensor)

		self.random_seed = params["random_seed"]

		# FIXING THE RANDOM SEED
		np.random.seed(self.random_seed)
		random.seed(self.random_seed)
		torch.manual_seed(self.random_seed)
		if self.gpu: torch.cuda.manual_seed(self.random_seed)

		self.optimizer_str = params["optimizer_str"]

		# SETTING THE PATHS...
		self.train_data_path = params['train_data_path']
		self.test_data_path = params['test_data_path']
		self.saving_path = params['saving_path']
		self.model_dir = params['model_dir']
		self.fig_dir = params['fig_dir']
		self.results_dir = params['results_dir']
		self.logfile_dir = params["logfile_dir"]
		self.write_to_log = params["write_to_log"]

		self.display_output = params["display_output"]
		self.num_test_ICS = params["num_test_ICS"]
		self.iterative_prediction_length = params["iterative_prediction_length"]
		self.hidden_state_propagation_length = params["hidden_state_propagation_length"]
		self.input_dim = params['input_dim']
		self.rnn_state_dim = self.input_dim
		self.N_used = params["N_used"]

		self.sequence_length = params['sequence_length']
		self.rnn_cell_type = params['rnn_cell_type']
		self.skip = params["skip"]

		# TESTING MODES
		self.teacher_forcing_forecasting=params["teacher_forcing_forecasting"]
		self.iterative_state_forecasting=params["iterative_state_forecasting"]

		self.multiple_ics_in_data = bool(self.params["multiple_ics_in_data"])

		self.scaler_tt = params["scaler"]
		self.scaler = scaler(self.scaler_tt, multiple_ics_in_data=self.multiple_ics_in_data)

		self.noise_level = params["noise_level"]

		self.retrain =  params['retrain']
		self.train_val_ratio = params['train_val_ratio']

		self.batch_size =  params['batch_size']
		self.overfitting_patience =  params['overfitting_patience']
		self.max_epochs =  params['max_epochs']
		self.max_rounds = params['max_rounds']
		self.learning_rate =  params['learning_rate']
		self.weight_decay =  params['weight_decay']

		self.layers_rnn = [self.params["rnn_layers_size"]] * self.params["rnn_layers_num"]

		self.model_name = self.createModelName()
		print("## Model name: \n{:}".format(self.model_name))
		self.saving_model_path = self.getModelDir() + "/model"


		self.makeDirectories()

		self.model = rnn_model(params, self)
		self.model.printModuleList()

		# PRINT PARAMS BEFORE PARALLELIZATION
		self.printParams()

		#TODO: load model when retrain
		self.model_parameters = self.model.getParams()

		# Initialize model parameters
		self.model.initializeWeights()
		self.device_count = torch.cuda.device_count()
		if self.gpu:
			print("USING CUDA -> SENDING THE MODEL TO THE GPU.")
			self.model.sendModelToCuda()
			if self.device_count > 1:
				print("# Using {:} GPUs! -> PARALLEL Distributing the batch.".format(self.device_count))
				device_ids = list(range(self.device_count))
				print("# device_ids = {:}".format(device_ids))
				self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
			print("# GPU MEMORY nvidia-smi in MB={:}".format(get_gpu_memory_map()))
		self.printGPUInfo()
		# Saving some info file for the model
		data = {"model":self, "params":params, "selfParams":self.params, "name":self.model_name}
		data_path = self.getModelDir() + "/info"
		saveData(data, data_path, "pickle")


	def printGPUInfo(self):
		print("CUDA Device available? {:}".format(torch.cuda.is_available()))
		print("Number of devices: {:}".format(self.device_count))
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print('Using device:', device)
		#Additional Info when using cuda
		if device.type == 'cuda':
			for i in range(self.device_count):
				print("DEVICE NAME {:}: {:}".format(i, torch.cuda.get_device_name(i)))
			print('MEMORY USAGE:')
			print('Memory allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
			print('Max memory allocated:', round(torch.cuda.max_memory_allocated(0)/1024**3,1), 'GB')
			print('Memory cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
			print('MAX memory cached:   ', round(torch.cuda.max_memory_cached(0)/1024**3,1), 'GB')

	def getKeysInModelName(self, with_rnn=True):

		keys = {
		'N_used':'-N_used_', 
		'N':'-N_', 
		'scaler':'-scaler_', 
		# 'skip':'-skip_', 
		'noise_level':'-NL_',
		# 'optimizer_str':'-OPT_', 
		'learning_rate':'-LR_',
		'weight_decay':'-L2_',
		'rnn_cell_type':'-C_',
		'rnn_layers_num':'-RNN_',
		'rnn_layers_size':'x',
		'sequence_length':'-SL_',
		}
		return keys

	def createModelName(self):
		keys = self.getKeysInModelName()
		str_ = "GPU-" * self.gpu + "ARNN"
		for key in keys:
			str_ += keys[key] + "{:}".format(self.params[key])
		return str_

	def makeDirectories(self):
		os.makedirs(self.getModelDir(), exist_ok=True)
		os.makedirs(self.getFigureDir(), exist_ok=True)
		os.makedirs(self.getResultsDir(), exist_ok=True)
		os.makedirs(self.getLogFileDir(), exist_ok=True)

	def getModelDir(self):
		model_dir = self.saving_path + self.model_dir + self.model_name
		return model_dir

	def getFigureDir(self, unformatted=False):
		fig_dir = self.saving_path + self.fig_dir + self.model_name
		return fig_dir

	def getResultsDir(self, unformatted=False):
		results_dir = self.saving_path + self.results_dir + self.model_name
		return results_dir

	def getLogFileDir(self, unformatted=False):
		logfile_dir = self.saving_path + self.logfile_dir + self.model_name
		return logfile_dir

	def printParams(self):
		self.n_trainable_parameters = self.model.countTrainableParams()
		self.n_model_parameters = self.model.countParams()
		# Print parameter information:
		print("# Trainable params {:}/{:}".format(self.n_trainable_parameters, self.n_model_parameters))
		return 0

	def declareOptimizer(self, lr):
		# print("LEARNING RATE: {}".format(lr))
		params = self.model_parameters
		weight_decay = 0.0
		if self.weight_decay > 0: print("No weight decay in RNN training.")

		print("LEARNING RATE: {:}, WEIGHT DECAY: {:}".format(lr, self.weight_decay))

		if self.optimizer_str == "adam":
			self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
			# self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=self.params["overfitting_patience"], verbose=True, threshold=1e-5, threshold_mode='rel')
		elif self.optimizer_str == "sgd":
			self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
			# self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr/10.0, max_lr=lr)
		elif self.optimizer_str == "rmsprop":
			self.optimizer = torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
		else:
			raise ValueError("Optimizer {:} not recognized.".format(self.optimizer_str))

	def getZeroRnnHiddenState(self, batch_size):
		hidden_state = []
		for ln in self.layers_rnn:
			hidden_state.append(self.getZeroRnnHiddenStateLayer(batch_size, ln))
		hidden_state = torch.stack(hidden_state)
		hidden_state = self.getModel().transposeHiddenState(hidden_state)
		return hidden_state

	def getZeroRnnHiddenStateLayer(self, batch_size, hidden_units):
		hx = Variable(torch.zeros(batch_size, hidden_units))
		if self.params["rnn_cell_type"]=="lstm":
			cx = Variable(torch.zeros(batch_size, hidden_units))
			hidden_state = torch.stack([hx, cx])
			return hidden_state
		elif self.params["rnn_cell_type"]=="gru":
			return hx
		else:
			raise ValueError("Unknown cell type {}.".format(self.params["rnn_cell_type"]))

	def plotBatchNumber(self, i, n_batches, is_train):
		if self.display_output:
			str_ = "\n"+ is_train * "TRAINING: " + (not is_train)* "EVALUATION"
			print("{:s} batch {:d}/{:d},  {:f}%".format(str_, int(i+1), int(n_batches), (i+1)/n_batches*100.))
			sys.stdout.write("\033[F")

	def getBatch(self, sequence, batch_idx):
		if not self.multiple_ics_in_data:
			input_batch = []
			target_batch = []
			for predict_on in batch_idx:
				input = sequence[predict_on-self.sequence_length:predict_on]
				target = sequence[predict_on-self.sequence_length+1:predict_on+1]
				input_batch.append(input)
				target_batch.append(target)
			input_batch = np.array(input_batch)
			target_batch = np.array(target_batch)
		else:
			input_batch = []
			target_batch = []
			for predict_on in batch_idx:
				ic, T = predict_on
				input = sequence[ic, T-self.sequence_length:T]
				target = sequence[ic, T-self.sequence_length+1:T+1]
				input_batch.append(input)
				target_batch.append(target)
			input_batch = np.array(input_batch)
			target_batch = np.array(target_batch)
		return input_batch, target_batch

	def sendHiddenStateToGPU(self, h_state):
		return h_state.cuda()

	def detachHiddenState(self, h_state):
		return h_state.detach()

	def getLoss(self, output, target):
		loss = output-target
		loss = loss.pow(2.0)
		# Mean over all dimensions
		loss = loss.mean(2)
		# Mean over all batches
		loss = loss.mean(0)
		loss = loss.mean()
		return loss

	def repeatAlongDim(self, var, axis, repeat_times, interleave=False):
		if not interleave:
			repeat_idx = len(var.size())*[1]
			repeat_idx[axis] = repeat_times
			var = var.repeat(*repeat_idx)
		else:
			var = var.repeat_interleave(repeat_times, dim=axis)
		return var

	def trainOnBatch(self, idx_train_on_epoch, batch_idx, input_sequence, is_train=False):
		# if self.gpu: print("0 # GPU MEMORY nvidia-smi in MB={:}".format(get_gpu_memory_map()))

		if not self.multiple_ics_in_data:
			idx_train_on_epoch -= set(batch_idx)
			times_to_remove = np.array([range(j-self.sequence_length, j) for j in batch_idx]).flatten()
			idx_train_on_epoch -= set(times_to_remove)
			times_to_remove = np.array([range(j, j+self.hidden_state_propagation_length) for j in batch_idx]).flatten()
			idx_train_on_epoch -= set(times_to_remove)
			initial_hidden_states = self.getZeroRnnHiddenState(np.shape(batch_idx)[0])
		else:
			# print(batch_idx)
			idx_train_on_epoch -= set(batch_idx)
			idx_to_remove = []
			ics_in_batch = [temp[0] for temp in batch_idx]
			# print(idx_train_on_epoch)
			for ic_num in range(len(ics_in_batch)):
				ic = ics_in_batch[ic_num]
				T = batch_idx[ic_num][1]
				times_to_remove = np.array(range(T-self.sequence_length, T)).flatten()
				# print(times_to_remove)
				idx_to_remove = set(tuple((ic, t)) for t in times_to_remove)
				# print(idx_to_remove)
				idx_train_on_epoch  -= idx_to_remove
				times_to_remove = np.array(range(T, T+self.hidden_state_propagation_length)).flatten()
				idx_to_remove = set(tuple((ic, t)) for t in times_to_remove)
				idx_train_on_epoch  -= idx_to_remove
				initial_hidden_states = self.getZeroRnnHiddenState(len(batch_idx))
				# print(len(batch_idx))

		losses_vec = []
		num_propagations = int(self.hidden_state_propagation_length//self.sequence_length)
		assert num_propagations>0, "Number of propagations (P/S) in sequence cannot be 0. Increase hidden_state_propagation_length (P={:}) or reduce the sequence length (S={:}).".format(self.hidden_state_propagation_length, self.sequence_length) 
		for p in range(int(self.hidden_state_propagation_length//self.sequence_length)):
			# Setting the optimizer to zero grad
			self.optimizer.zero_grad()
			# Getting the batch
			input_batch, target_batch = self.getBatch(input_sequence, batch_idx)

			# Transform to pytorch and forward the network
			input_batch = self.torch_dtype(input_batch)
			target_batch = self.torch_dtype(target_batch)
			if self.gpu:
				# SENDING THE TENSORS TO CUDA
				input_batch = input_batch.cuda()
				target_batch = target_batch.cuda()
				initial_hidden_states = self.sendHiddenStateToGPU(initial_hidden_states)

			output_batch, last_hidden_state, rnn_outputs = self.model.forward(input_batch, initial_hidden_states, is_train=is_train)

			loss_fwd = self.getLoss(
				output_batch,
				target_batch,
				)

			if is_train:
				loss_fwd.backward()
				self.optimizer.step()


			loss_fwd = loss_fwd.cpu().detach().numpy()

			losses_batch = np.array([loss_fwd])

			last_hidden_state = self.detachHiddenState(last_hidden_state)

			# APPENDING LOSSES
			losses_vec.append(losses_batch)
			initial_hidden_states = last_hidden_state

			#################################
			### UPDATING BATCH IDX
			#################################
			if not self.multiple_ics_in_data:
				batch_idx = np.array(batch_idx) + self.sequence_length
			else:
				temp = []
				for batch_member in batch_idx:
					ic, idx = batch_member
					idx = idx + self.sequence_length
					temp.append(tuple((ic, idx)))
				batch_idx = temp
		losses = np.mean(np.array(losses_vec))
		losses = [losses]
		return idx_train_on_epoch, losses

	def torchZero(self):
		return self.torch_dtype([0.0])[0]

	def trainEpoch(self, idx_on, n_samples, input_sequence, is_train=False):
		if self.gpu: print("# GPU MEMORY nvidia-smi in MB={:}".format(get_gpu_memory_map()))

		idx_on_epoch = idx_on.copy()
		epoch_losses_vec = []
		stop_limit = np.max([self.hidden_state_propagation_length, self.batch_size])
		stop_limit = np.min([stop_limit, len(idx_on)])
		while len(idx_on_epoch) >= stop_limit:
			batch_idx = random.sample(idx_on_epoch, self.batch_size)
			idx_on_epoch, losses = self.trainOnBatch(idx_on_epoch, batch_idx, input_sequence, is_train=is_train)
			epoch_losses_vec.append(losses)
		epoch_losses = np.mean(np.array(epoch_losses_vec), axis=0)
		time_ = time.time() - self.start_time
		return epoch_losses, time_

	def getStartingPoints(self, input_sequence):
		if not self.multiple_ics_in_data:
			NN = np.shape(input_sequence)[0]
			if NN - self.hidden_state_propagation_length - self.sequence_length <= 0:
				raise ValueError("The hidden_state_propagation_length is too big. Reduce it. N_data !> H + SL, {:} !> {:} + {:} = {:}".format(NN, self.hidden_state_propagation_length, self.sequence_length, self.sequence_length+self.hidden_state_propagation_length))
			idx_on = set(np.arange(self.sequence_length, NN - self.hidden_state_propagation_length))
			n_samples = len(idx_on)
			assert(n_samples>0)
		else:

			NICS = np.shape(input_sequence)[0]
			NN = np.shape(input_sequence)[1]
			if NN - self.hidden_state_propagation_length - self.sequence_length <= 0:
				raise ValueError("The hidden_state_propagation_length is too big. Reduce it. N_data !> H + SL, {:} !> {:} + {:} = {:}".format(NN, self.hidden_state_propagation_length, self.sequence_length, self.sequence_length+self.hidden_state_propagation_length))
			idx_on_n = np.arange(self.sequence_length, NN - self.hidden_state_propagation_length)
			idx_on_ics = np.arange(NICS)
			idx_on  = [tuple((ic, t)) for t in idx_on_n for ic in idx_on_ics]
			idx_on = set(idx_on)
			n_samples = len(idx_on)
		# Checking idx_on compared to the batch size
		if len(idx_on) < self.batch_size:
			raise ValueError("The size of the indexes to train on ({:}) is smaller than the batch size ({:}).\n".format(self.batch_size, len(idx_on)))
		return idx_on, n_samples

	def getTrainingData(self):
		data = loadData(self.train_data_path, "pickle")
		input_sequence = data["train_input_sequence"]
		print("Adding noise to the training data. {:} per mille ".format(self.noise_level))
		if not self.params["multiple_ics_in_data"]:
			input_sequence = addNoise(input_sequence, self.noise_level)
		else:
			for ic_num in range(len(input_sequence)):
				input_sequence[ic_num] = addNoise(input_sequence[ic_num], self.noise_level)

		shapes = np.shape(input_sequence)
		if not self.params["multiple_ics_in_data"]:
			N_all, first_dim = shapes
			print("##Using {:}/{:} dimensions and {:}/{:} samples ##".format(self.input_dim, first_dim, self.N_used, shapes[0]))
			if self.N_used > shapes[0]: raise ValueError("Not enough samples in the training data.")
		else:
			N_ics, N_all, first_dim = shapes
			print("##Using {:}/{:} dimensions  and {:}/{:} samples from {:} initial conditions. ##".format(self.input_dim, first_dim, self.N_used, N_all, N_ics))
			if self.N_used > N_all: raise ValueError("Not enough samples in the training data.")

		if self.input_dim > first_dim: raise ValueError("Requested input dimension is wrong.")

		if not self.params["multiple_ics_in_data"]:
			input_sequence = input_sequence[:self.N_used, :self.input_dim]
			input_sequence = input_sequence[::self.params["skip"]]
		else:
			input_sequence = input_sequence[:, :self.N_used, :self.input_dim]
			input_sequence = input_sequence[:, ::self.params["skip"]]
		dt = data["dt"]
		del data
		return input_sequence

	def train(self):
		input_sequence = self.getTrainingData()

		print("SCALING")
		input_sequence = self.scaler.scaleData(input_sequence)

		print(input_sequence.dtype)
		print(np.shape(input_sequence))

		train_input_sequence, val_input_sequence = divideData(input_sequence, self.train_val_ratio, self.multiple_ics_in_data)

		idx_train_on, n_train_samples = self.getStartingPoints(train_input_sequence)
		idx_val_on, n_val_samples = self.getStartingPoints(val_input_sequence)

		print("NUMBER OF TRAINING SAMPLES: {:d}".format(n_train_samples))
		print("NUMBER OF VALIDATION SAMPLES: {:d}".format(n_val_samples))

		self.loss_total_train_vec = []
		self.loss_total_val_vec = []
		self.losses_train_vec = []
		self.losses_time_train_vec = []
		self.losses_val_vec = []
		self.losses_time_val_vec = []
		isWallTimeLimit = False

		# Termination criterion:
		# If the training procedure completed the maximum number of epochs

		# Learning rate decrease criterion:
		# If the validation loss does not improve for some epochs (patience)
		# the round is terminated, the learning rate decreased and training 
		# proceeds in the next round.

		self.epochs_iter = 0
		self.epochs_iter_global = self.epochs_iter
		self.rounds_iter = 0
		# TRACKING
		self.tqdm = tqdm(total=self.max_epochs)
		while self.epochs_iter < self.max_epochs and self.rounds_iter < self.max_rounds:
			isWallTimeLimit = self.trainRound(idx_train_on, idx_val_on, n_train_samples, n_val_samples, train_input_sequence, val_input_sequence)
			# INCREMENTING THE ROUND NUMBER
			if isWallTimeLimit: break

		# If the training time limit was not reached, save the model...
		if not isWallTimeLimit:
			if self.epochs_iter == self.max_epochs:
				print("## Training finished. Maximum number of epochs reached.")
			elif self.rounds_iter == self.max_rounds:
				print("## Training finished. Maximum number of rounds reached.")
			else:
				print(self.rounds_iter)
				print(self.epochs_iter)
				raise ValueError("## Training finished. I do not know why!")
			self.saveModel()
			plotTrainingLosses(self, self.loss_total_train_vec, self.loss_total_val_vec, self.min_val_total_loss)
			plotAllLosses(self, self.losses_train_vec, self.losses_time_train_vec, self.losses_val_vec, self.losses_time_val_vec, self.min_val_total_loss)

	def printLosses(self, label, losses):
		# PRINT ALL THE NON-ZERO LOSSES
		losses_labels = ["TOTAL"]
		to_print="# {:s}-losses: ".format(label)
		to_print+="{:}={:1.2E} |".format(losses_labels[0], losses[0])
		print(to_print)

	def printEpochStats(self, epoch_time_start, epochs_iter, epochs_in_round, losses_train, losses_val):
		epoch_duration = time.time() - epoch_time_start
		time_covered = epoch_duration*epochs_iter
		time_total = epoch_duration*self.max_epochs
		percent = time_covered/time_total*100
		print("###################################################")
		label = "EP={:} - R={:} - ER={:} - [ TIME= {:}, {:} / {:} - {:.2f} %] - LR={:1.2E}".format(epochs_iter, self.rounds_iter, epochs_in_round, secondsToTimeStr(epoch_duration), secondsToTimeStr(time_covered), secondsToTimeStr(time_total), percent, self.learning_rate_round)
		print(label)
		self.printLosses("TRAIN", losses_train)
		self.printLosses("VAL  ", losses_val)


	def printLearningRate(self):
		for param_group in self.optimizer.param_groups:
			print("Current learning rate = {:}".format(param_group["lr"]))
		return 0

	def getModel(self):
		if (not self.gpu) or (self.device_count <=1):
			return self.model
		elif self.gpu and self.device_count>1:
			return self.model.module
		else:
			raise ValueError("Value of self.gpu {:} not recognized.".format(self.gpu))

	def trainRound(self, idx_train_on, idx_val_on, n_train_samples, n_val_samples, train_input_sequence, val_input_sequence):
		# Check if retraining of a model is requested else random initialization of the weights
		isWallTimeLimit = False

		# SETTING THE INITIAL LEARNING RATE
		if self.rounds_iter==0:
			self.learning_rate_round = self.learning_rate
			self.previous_round_converged = 0
		elif self.previous_round_converged == 0:
			self.learning_rate_round = self.learning_rate_round
			self.previous_round_converged = 0
		elif self.previous_round_converged == 1:
			self.previous_round_converged = 0
			self.learning_rate_round = self.learning_rate_round/2

		# OPTIMIZER HAS TO BE DECLARED
		self.declareOptimizer(self.learning_rate_round)

		if self.retrain == 1:
			print("RESTORING MODEL")
			self.loadModel()
		else:
			# SAVING THE INITIAL MODEL
			print("SAVING THE INITIAL MODEL")
			torch.save(self.getModel().state_dict(), self.saving_model_path)
			pass

		print("##### ROUND: {:}, LEARNING RATE={:} #####".format(self.rounds_iter, self.learning_rate_round))

		losses_train, time_train = self.trainEpoch(idx_train_on, n_train_samples, train_input_sequence, is_train=False)
		losses_val, time_val = self.trainEpoch(idx_val_on, n_val_samples, val_input_sequence, is_train=False)

		label = "INITIAL (NEW ROUND):  EP{:} - R{:}".format(self.epochs_iter, self.rounds_iter)
		print(label)

		self.printLosses("TRAIN", losses_train)
		self.printLosses("VAL  ", losses_val)

		self.min_val_total_loss = losses_val[0]
		self.loss_total_train = losses_train[0]

		rnn_loss_round_train_vec = []
		rnn_loss_round_val_vec = []

		rnn_loss_round_train_vec.append(losses_train[0])
		rnn_loss_round_val_vec.append(losses_val[0])

		self.loss_total_train_vec.append(losses_train[0])
		self.loss_total_val_vec.append(losses_val[0])

		self.losses_train_vec.append(losses_train)
		self.losses_time_train_vec.append(time_train)
		self.losses_val_vec.append(losses_val)
		self.losses_time_val_vec.append(time_val)

		for epochs_iter in range(self.epochs_iter, self.max_epochs+1):
			epoch_time_start = time.time()
			epochs_in_round = epochs_iter - self.epochs_iter
			self.epochs_iter_global = epochs_iter

			losses_train, time_train = self.trainEpoch(idx_train_on, n_train_samples, train_input_sequence, is_train=True)
			losses_val, time_val = self.trainEpoch(idx_val_on, n_val_samples, val_input_sequence, is_train=False)
			rnn_loss_round_train_vec.append(losses_train[0])
			rnn_loss_round_val_vec.append(losses_val[0])
			self.loss_total_train_vec.append(losses_train[0])
			self.loss_total_val_vec.append(losses_val[0])

			self.losses_train_vec.append(losses_train)
			self.losses_time_train_vec.append(time_train)
			self.losses_val_vec.append(losses_val)
			self.losses_time_val_vec.append(time_val)

			self.printEpochStats(epoch_time_start, epochs_iter, epochs_in_round, losses_train, losses_val)

			if losses_val[0] < self.min_val_total_loss:
				print("SAVING MODEL!!!")
				self.min_val_total_loss = losses_val[0]
				self.loss_total_train = losses_train[0]
				torch.save(self.getModel().state_dict(), self.saving_model_path)

			if epochs_in_round > self.overfitting_patience:
				if all(self.min_val_total_loss < rnn_loss_round_val_vec[-self.overfitting_patience:]):
					self.previous_round_converged = True
					break

			# # LEARNING RATE SCHEDULER (PLATEU ON VALIDATION LOSS)
			# if self.optimizer_str == "adam": self.scheduler.step(losses_val[0])
			self.tqdm.update(1)
			isWallTimeLimit = self.isWallTimeLimit()
			if isWallTimeLimit:
				break

		self.rounds_iter += 1
		self.epochs_iter = epochs_iter
		return isWallTimeLimit

	def isWallTimeLimit(self):
		training_time = time.time() - self.start_time
		if training_time > self.reference_train_time:
			print("## Maximum train time reached: saving model... ##")
			self.tqdm.close()
			self.saveModel()
			plotTrainingLosses(self, self.loss_total_train_vec, self.loss_total_val_vec, self.min_val_total_loss)
			return True
		else:
			return False

	def delete(self):
		pass

	def saveModel(self):
		print("Recording time...")
		self.total_training_time = time.time() - self.start_time
		if hasattr(self, 'loss_total_train_vec'):
			if len(self.loss_total_train_vec)!=0:
				self.training_time = self.total_training_time/len(self.loss_total_train_vec)
			else:
				self.training_time = self.total_training_time
		else:
			self.training_time = self.total_training_time

		print("Total training time per epoch is {:}".format(secondsToTimeStr(self.training_time)))
		print("Total training time is {:}".format(secondsToTimeStr(self.total_training_time)))

		print("MEMORY TRACKING IN MB...")
		process = psutil.Process(os.getpid())
		memory = process.memory_info().rss/1024/1024
		self.memory = memory
		print("Script used {:} MB".format(self.memory))

		data = {
		"params":self.params,
		"model_name":self.model_name,
		"memory":self.memory,
		"total_training_time":self.total_training_time,
		"training_time":self.training_time,
		"n_trainable_parameters":self.n_trainable_parameters,
		"n_model_parameters":self.n_model_parameters,
		"rnn_loss_train_vec":self.loss_total_train_vec,
		"rnn_loss_val_vec":self.loss_total_val_vec,
		"rnn_min_val_error":self.min_val_total_loss,
		"rnn_train_error":self.loss_total_train,
		"losses_train_vec":self.losses_train_vec,
		"losses_time_train_vec":self.losses_time_train_vec,
		"losses_val_vec":self.losses_val_vec,
		"losses_time_val_vec":self.losses_time_val_vec,
		"scaler":self.scaler,
		}
		fields_to_write = [
		"memory",
		"total_training_time",
		"n_model_parameters",
		"n_trainable_parameters"
		]
		if self.write_to_log == 1:
			logfile_train = self.getLogFileDir()  + "/train.txt"
			print("Writing to log-file in path {:}".format(logfile_train))
			writeToLogFile(self, logfile_train, data, fields_to_write)

		data_path = self.getModelDir() + "/data"
		saveData(data, data_path, "pickle")
		self.printGPUInfo()

	def loadModel(self, in_cpu=False):
		try:
			if not in_cpu and self.gpu:
				print("# LOADING model in GPU.")
				self.getModel().load_state_dict(torch.load(self.saving_model_path))
			else:
				print("# LOADING model in CPU...")
				self.getModel().load_state_dict(torch.load(self.saving_model_path, map_location=torch.device('cpu')))
		except Exception as inst:
			print("MODEL {:s} NOT FOUND. Is hippo mounted? Are you testing ? Did you already train the model?".format(self.saving_model_path))
			raise ValueError(inst)
		data_path = self.getModelDir() + "/data"
		data = loadData(data_path, "pickle")
		self.scaler = data["scaler"]
		# TODO: REMOVE
		self.scaler.multiple_ics_in_data = self.params["multiple_ics_in_data"]
		self.loss_total_train_vec = data["rnn_loss_train_vec"]
		self.loss_total_val_vec = data["rnn_loss_val_vec"]
		self.min_val_total_loss = data["rnn_min_val_error"]
		self.losses_time_train_vec = data["losses_time_train_vec"]
		self.losses_time_val_vec = data["losses_time_val_vec"]
		self.losses_val_vec = data["losses_val_vec"]
		self.losses_train_vec = data["losses_train_vec"]
		del data
		return 0

	def testing(self):
		if self.loadModel()==0:
			# MODEL LOADED IN EVALUATION MODE
			with torch.no_grad():
				# self.n_warmup = int(self.hidden_state_propagation_length/self.sequence_length//4)
				self.n_warmup = int(self.hidden_state_propagation_length//4)
				if self.n_warmup <=1: self.n_warmup = 2
				print("WARMING UP STEPS (for statefull RNNs): {:d}".format(self.n_warmup))

				for set_ in ["train", "test"]:
					self.testingOnSet(set_)
		return 0

	def testingOnSet(self, set_="train"):
		print("#####	 Testing on set: {:}	 ######".format(set_))
		data = loadData(self.test_data_path, "pickle")
		testing_ic_indexes = data["testing_ic_indexes"]
		dt = data["dt"]
		if set_ == "test":
			if not self.multiple_ics_in_data:
				input_sequence = data["test_input_sequence"][::self.params["skip"], :self.input_dim]
			else:
				input_sequence = data["test_input_sequence"][:, ::self.params["skip"], :self.input_dim]
			print("Input sequence shape {:}".format(np.shape(input_sequence)))
			del data
		elif set_ == "train":
			data = loadData(self.train_data_path, "pickle")
			if not self.multiple_ics_in_data:
				input_sequence = data["train_input_sequence"][::self.params["skip"], :self.input_dim]
			else:
				input_sequence = data["train_input_sequence"][:, ::self.params["skip"], :self.input_dim]
			print("Input sequence shape {:}".format(np.shape(input_sequence)))
			del data
		else:
			raise ValueError("Invalid set {:}.".format(set_))

		for testing_mode in self.getRNNTestingModes():
			# Check invalid combinations
			self.checkTestingMode(testing_mode)
			self.testOnMode(input_sequence, testing_ic_indexes, dt, set_, testing_mode)


	def checkTestingMode(self, testing_mode):
		if "iterative" in testing_mode and not self.iterative_state_forecasting: raise ValueError
		if "teacher" in testing_mode and not self.teacher_forcing_forecasting: raise ValueError
		return 0

	def testOnMode(self, input_sequence, testing_ic_indexes, dt, set_, testing_mode=None):
		assert (testing_mode in self.getTestingModes())
		assert (set_=="train" or set_=="test")
		print("---- Testing on Mode {:} ----".format(testing_mode))
		if self.num_test_ICS>0:
			results = self.predictIndexes(input_sequence, testing_ic_indexes, dt, set_, testing_mode=testing_mode)
			data_path = self.getResultsDir() + "/results_{:}_{:}".format(testing_mode, set_)
			saveData(results, data_path, SAVE_FORMAT)
		else:
			print("Model has RNN/PREDICTOR but no initial conditions set to test num_test_ICS={:}.".format(self.num_test_ICS))
		return 0

	def getRNNDeterministicTestingModes(self):
		modes = []
		if self.iterative_state_forecasting: modes.append("iterative_state_forecasting")
		if self.teacher_forcing_forecasting: modes.append("teacher_forcing_forecasting")
		return modes

	def getRNNTestingModes(self):
		testing_modes = self.getRNNDeterministicTestingModes()
		return testing_modes

	def getTestingModes(self):
		modes = self.getRNNDeterministicTestingModes()
		return modes

	def predictIndexes(self, input_sequence, ic_indexes, dt, set_name, testing_mode):
		assert (testing_mode in self.getTestingModes())
		if testing_mode in self.getRNNDeterministicTestingModes():
			return self.predictIndexesDeterministic(input_sequence, ic_indexes, dt, set_name, testing_mode)
		else:
			raise ValueError("Mode {:} is invalid".format(testing_mode))

	def predictIndexesDeterministic(self, input_sequence, ic_indexes, dt, set_name, testing_mode=None):
		num_test_ICS = self.num_test_ICS
		input_sequence = self.scaler.scaleData(input_sequence, reuse=1)
		predictions_all = []
		targets_all = []

		predictions_augmented_all = []
		targets_augmented_all = []

		rmse_all = []
		rmnse_all = []
		mnad_all = []
		num_accurate_pred_005_all = []
		num_accurate_pred_050_all = []
		if num_test_ICS > len(ic_indexes): raise ValueError("Not enough ICs in the dataset {:}.".format(set_name))
		for ic_num in range(num_test_ICS):
			if self.display_output:
				print("IC {:}/{:}, {:2.3f}%".format(ic_num, num_test_ICS, ic_num/num_test_ICS*100))
			ic_idx = ic_indexes[ic_num]
			if not self.multiple_ics_in_data:
				input_sequence_ic = input_sequence[ic_idx-self.n_warmup:ic_idx+self.iterative_prediction_length]
			else:
				T = self.n_warmup
				input_sequence_ic = input_sequence[ic_idx, T-self.n_warmup:T+self.iterative_prediction_length]

			prediction, target, prediction_augment, target_augment, time_total_per_iter = self.predictSequence(input_sequence_ic, testing_mode, dt, ic_idx)

			without_first_dim = True if self.multiple_ics_in_data else False
			prediction = self.scaler.descaleData(prediction, without_first_dim)
			target = self.scaler.descaleData(target, without_first_dim)

			prediction_augment = self.scaler.descaleData(prediction_augment, without_first_dim)
			target_augment = self.scaler.descaleData(target_augment, without_first_dim)

			rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, mnad = computeErrors(target, prediction, self.scaler.data_std)

			predictions_all.append(prediction)
			targets_all.append(target)

			predictions_augmented_all.append(prediction_augment)
			targets_augmented_all.append(target_augment)

			rmse_all.append(rmse)
			rmnse_all.append(rmnse)
			mnad_all.append(mnad)
			num_accurate_pred_005_all.append(num_accurate_pred_005)
			num_accurate_pred_050_all.append(num_accurate_pred_050)

		predictions_all = np.array(predictions_all)
		targets_all = np.array(targets_all)

		predictions_augmented_all = np.array(predictions_augmented_all)
		targets_augmented_all = np.array(targets_augmented_all)

		rmse_all = np.array(rmse_all)
		rmnse_all = np.array(rmnse_all)
		mnad_all = np.array(mnad_all)
		num_accurate_pred_005_all = np.array(num_accurate_pred_005_all)
		num_accurate_pred_050_all = np.array(num_accurate_pred_050_all)

		print("TRAJECTORIES SHAPES:")
		print(np.shape(targets_all))
		print(np.shape(predictions_all))

		mnad_avg = np.mean(mnad_all)
		print("AVERAGE MNAD ERROR: {:}".format(mnad_avg))
		mnad_avg_over_ics = np.mean(mnad_all, axis=0)

		rmnse_avg = np.mean(rmnse_all)
		print("AVERAGE RMNSE ERROR: {:}".format(rmnse_avg))
		rmnse_avg_over_ics = np.mean(rmnse_all, axis=0)

		num_accurate_pred_005_avg = np.mean(num_accurate_pred_005_all)
		print("AVG NUMBER OF ACCURATE 0.05 PREDICTIONS: {:}".format(num_accurate_pred_005_avg))
		num_accurate_pred_050_avg = np.mean(num_accurate_pred_050_all)
		print("AVG NUMBER OF ACCURATE 0.5 PREDICTIONS: {:}".format(num_accurate_pred_050_avg))

		freq_pred, freq_true, sp_true, sp_pred, error_freq = computeFrequencyError(predictions_all, targets_all, dt)
		print("FREQUENCY ERROR: {:}".format(error_freq))

		fields_2_save_2_logfile = [
		"rmnse_avg",
		"mnad_avg",
		"num_accurate_pred_005_avg",
		"num_accurate_pred_050_avg",
		"error_freq",
		"state_dist_L1_hist_error",
		"state_dist_wasserstein_distance",
		"state_dist_KS_error",
		"time_total_per_iter",
		]

		results = {
		"fields_2_save_2_logfile":fields_2_save_2_logfile,
		"rmnse_avg":rmnse_avg,
		"rmse_all":rmse_all,
		"rmnse_all":rmnse_all,
		"rmnse_avg_over_ics":rmnse_avg_over_ics,
		"mnad_avg":mnad_avg,
		"mnad_avg_over_ics":mnad_avg_over_ics,
		"num_accurate_pred_005_avg":num_accurate_pred_005_avg,
		"num_accurate_pred_050_avg":num_accurate_pred_050_avg,
		"error_freq":error_freq,
		"predictions_all":predictions_all,
		"targets_all":targets_all,
		"predictions_augmented_all":predictions_augmented_all,
		"targets_augmented_all":targets_augmented_all,
		"freq_pred":freq_pred,
		"freq_true":freq_true,
		"sp_true":sp_true,
		"sp_pred":sp_pred,
		"n_warmup":self.n_warmup,
		"ic_indexes":ic_indexes,
		"testing_mode":testing_mode,
		"dt":dt,
		"time_total_per_iter":time_total_per_iter,
		}
		return results

	def predictSequence(self, input_sequence, testing_mode=None, dt=1, ic_idx=0):
		N = np.shape(input_sequence)[0]
		# PREDICTION LENGTH
		if N - self.n_warmup != self.iterative_prediction_length: raise ValueError("Error! N ({:}) - self.n_warmup ({:}) != iterative_prediction_length ({:})".format(N, self.n_warmup, self.iterative_prediction_length))
		# PREPARING THE HIDDEN STATES
		initial_hidden_states = self.getZeroRnnHiddenState(1)
		assert self.n_warmup > 1, "Warm up steps cannot be <= 1. Increase the iterative prediction length."

		warmup_data_input = input_sequence[:self.n_warmup-1]
		warmup_data_input = warmup_data_input[np.newaxis, :]

		warmup_data_target = input_sequence[1:self.n_warmup]
		warmup_data_target = warmup_data_target[np.newaxis, :]

		if testing_mode in self.getRNNDeterministicTestingModes():
			target = input_sequence[self.n_warmup:self.n_warmup+self.iterative_prediction_length]
		else:
			raise ValueError("Testing mode {:} not recognized.".format(testing_mode))

		warmup_data_input = self.torch_dtype(warmup_data_input)

		if self.gpu:
			# SENDING THE TENSORS TO CUDA
			warmup_data_input = warmup_data_input.cuda()
			initial_hidden_states = self.sendHiddenStateToGPU(initial_hidden_states)

		warmup_data_output, last_hidden_state, _ = self.model.forward(warmup_data_input, initial_hidden_states, is_train=False)

		prediction = []

		if ("iterative_state" in testing_mode):
			# LATTENT PROPAGATION
			input_t = input_sequence[self.n_warmup-1]
			input_t = input_t[np.newaxis,np.newaxis,:]
		elif "teacher_forcing" in testing_mode:
			input_t = np.reshape(input_sequence[self.n_warmup-1:-1], (1, -1, self.input_dim))
		else:
			raise ValueError("I do not know how to initialize the state for {:}.".format(testing_mode))
			
		input_t = self.torch_dtype(input_t)

		if self.gpu:
			input_t = input_t.cuda()
			last_hidden_state = self.sendHiddenStateToGPU(last_hidden_state)

		time_start = time.time()
		if "teacher_forcing" in testing_mode:
			input_t = self.torch_dtype(input_t)
			prediction, last_hidden_state, rnn_outputs = self.model.forward(input_t, last_hidden_state, is_iterative_forecasting=False, horizon=self.iterative_prediction_length, is_train=False)

		elif "iterative_state" in testing_mode:
			prediction, last_hidden_state, rnn_outputs = self.model.forward(input_t, last_hidden_state, is_iterative_forecasting=True, horizon=self.iterative_prediction_length, is_train=False)

		time_end = time.time()
		time_total = time_end - time_start

		time_total_per_iter = time_total/self.iterative_prediction_length

		prediction = prediction[0]
		rnn_outputs = rnn_outputs[0]

		prediction = prediction.cpu().detach().numpy()
		rnn_outputs = rnn_outputs.cpu().detach().numpy()

		prediction=np.array(prediction)
		rnn_outputs=np.array(rnn_outputs)
		target=np.array(target)

		target_augment = np.concatenate((warmup_data_target[0], target), axis=0)
		warmup_data_output = warmup_data_output.cpu().detach().numpy()
		prediction_augment = np.concatenate((warmup_data_output[0], prediction), axis=0)
		return prediction, target, prediction_augment, target_augment,time_total_per_iter


	def postprocess(self):
		for set_name in ["train", "test"]:
			fields_to_compare = [
			"time_total_per_iter",
			"rmnse_avg_over_ics",
			"rmnse_avg", 
			"num_accurate_pred_050_avg",
			"error_freq",
			"mnad_avg",
			"mnad_avg_over_ics",
			]

			dicts_to_compare = {}
			for testing_mode in self.getRNNTestingModes():
				data_path = self.getResultsDir() + "/results_{:}_{:}".format(testing_mode, set_name)
				results = loadData(data_path, SAVE_FORMAT)
				# plotSpectrum(self, results, set_name, testing_mode)
				logfile = self.getLogFileDir()  + "/results_{:}_{:}.txt".format(testing_mode, set_name)
				if self.write_to_log: writeToLogFile(self, logfile, results, results["fields_2_save_2_logfile"])

				ic_indexes = results["ic_indexes"]
				dt = results["dt"]
				predictions_augmented_all = results["predictions_augmented_all"]
				targets_augmented_all = results["targets_augmented_all"]
				n_warmup = results["n_warmup"]
				rmse_all = results["rmse_all"]
				rmnse_all = results["rmnse_all"]
				predictions_all = results["predictions_all"]
				targets_all = results["targets_all"]

				results_dict = {}
				for field in fields_to_compare:
					results_dict[field] = results[field]
				dicts_to_compare[testing_mode] = results_dict

				max_index = np.min([3, np.shape(results["targets_all"])[0]])
				# max_index = np.min([1, np.shape(results["targets_all"])[0]])
				for idx in range(max_index):
					createIterativePredictionPlots(self, \
						targets_all[idx], \
						predictions_all[idx], \
						dt, ic_indexes[idx], set_name, \
						testing_mode=testing_mode, \
						warm_up=n_warmup, \
						error=rmse_all[idx], nerror=rmnse_all[idx], \
						target_augment=targets_augmented_all[idx], \
						prediction_augment=predictions_augmented_all[idx], \
						)


	def calculateLyapunovSpectrum(self):
		self.model.sendModelToCPU()
		if self.loadModel(in_cpu=True)==0:
			# self.n_warmup = int(self.hidden_state_propagation_length/self.sequence_length//4)
			self.n_warmup = int(self.hidden_state_propagation_length//4)
			if self.n_warmup <=1: self.n_warmup = 2
			print("WARMING UP STEPS (for statefull RNNs): {:d}".format(self.n_warmup))

			data = loadData(self.test_data_path, "pickle")
			testing_ic_indexes = data["testing_ic_indexes"]
			dt				 = data["dt"]
			input_sequence	 = data["test_input_sequence"]
			del data
			ic_idx = testing_ic_indexes[0]
			input_sequence = input_sequence[ic_idx-self.n_warmup:ic_idx+self.iterative_prediction_length]
			input_sequence = self.scaler.scaleData(input_sequence, reuse=1)

			print("Input sequence shape {:}".format(np.shape(input_sequence)))

			if not isinstance(self.params["num_lyaps"], int) or self.params["num_lyaps"]==0:
				raise ValueError("Number of lyapunov exponents to be calculated has to be positive. Might not given as input argument.")

			print("calculateLyapunovSpectrum() checking where model is (CUDA or CPU):")
			print("model.model.RNN[0]	 = {:}".format(self.checkIfModelOnCuda(self.model.RNN[0])))
			print("model.model.RNN_OUTPUT[0] = {:}".format(self.checkIfModelOnCuda(self.model.RNN_OUTPUT[0])))

			num_lyaps = self.params["num_lyaps"]
			print("Building wrapper...")
			lyapunov_wrapper = lyapunov_spectrum_wrapper(self)
			print("Calculating Lyapunov Spectrum...")
			lyapunov_wrapper.calculate(input_sequence, dt, num_lyaps, self)

	def plotLyapunovSpectrum(self):
			plotLyapunovSpectrumResults(self)

	def checkIfModelOnCuda(self, model):
		is_cuda = next(model.parameters()).is_cuda
		return is_cuda


			
