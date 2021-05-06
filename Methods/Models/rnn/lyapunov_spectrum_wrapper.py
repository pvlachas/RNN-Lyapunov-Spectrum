#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

# TORCH
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import socket
import os
from tqdm import tqdm

# Needed for Lyapunov exponent calculation
import scipy
import cmath
import time
from data_utils import *

def log(x):
	return np.array([cmath.log(xx) for xx in x])

class lyapunov_spectrum_wrapper(nn.Module):
	def __init__(self, model):
		super(lyapunov_spectrum_wrapper, self).__init__()
		print(model.model)

		self.rnn_cell = model.model.RNN[0]

		self.cell_str = model.rnn_cell_type
		assert(len(model.layers_rnn)==1)
		self.NUM_HIDDEN_UNITS = model.layers_rnn[0]
		print(self.rnn_cell)
		self.rnn_mlp = model.model.RNN_OUTPUT[0]
		print(self.rnn_mlp)
		parameters = list(self.named_parameters())
		print("Parameters:")
		for param in parameters:
			name, data = param
			print(name)
		self.n_warmup = model.n_warmup
		self.iterative_prediction_length = model.iterative_prediction_length
		self.results_dir = model.getResultsDir()
		self.figures_dir = model.getFigureDir()
		self.cudnn_benchmark = int(model.params["cudnn_benchmark"])

		# Using cuda ?
		# self.gpu = torch.cuda.is_available()
		self.gpu = False

		if self.gpu:
			print("Model in Lyapunov exponent calculation running on GPU.")
			torch.set_default_tensor_type(torch.cuda.DoubleTensor)
			self.torch_dtype = torch.cuda.DoubleTensor
			if self.cudnn_benchmark: torch.backends.cudnn.benchmark = True
		else:
			print("Model in Lyapunov exponent calculation running on CPU.")
			self.torch_dtype = torch.DoubleTensor
			torch.set_default_tensor_type(torch.DoubleTensor)
		self.checkVariables(model)


		print("CHECKING IF CUDA:")
		print("self.rnn_cell		  = {:}".format(self.checkIfModelOnCuda(self.rnn_cell)))
		print("self.rnn_mlp		   = {:}".format(self.checkIfModelOnCuda(self.rnn_mlp)))
		print("model.model.RNN[0]	 = {:}".format(self.checkIfModelOnCuda(model.model.RNN[0])))
		print("model.model.RNN_OUTPUT[0] = {:}".format(self.checkIfModelOnCuda(model.model.RNN_OUTPUT[0])))

	def checkIfModelOnCuda(self, temp):
		is_cuda = next(temp.parameters()).is_cuda
		return is_cuda

	def checkVariables(self, model_temp):
		########################################################
		## Checking that the RNN parameters match
		########################################################
		ditc_model = {}
		ditc_le	= {}
		for name, param in model_temp.model.RNN[0].named_parameters():
			ditc_model[name] = param.data
		for name, param in self.rnn_cell.named_parameters():
			ditc_le[name] = param.data
		for name in ditc_model.keys():
			diff_ = np.linalg.norm(ditc_model[name] - ditc_le[name])
			assert(diff_<1e-8)

		########################################################
		## Checking that the MLP parameters match
		########################################################
		ditc_model = {}
		ditc_le	= {}
		for name, param in model_temp.model.RNN_OUTPUT[0].named_parameters():
			ditc_model[name] = param.data
		for name, param in self.rnn_mlp.named_parameters():
			ditc_le[name] = param.data
		for name in ditc_model.keys():
			diff_ = np.linalg.norm(ditc_model[name] - ditc_le[name])
			assert(diff_<1e-8)
		print("Parameters match test passed.")
		return 0


	def forward(self, input_sequence, initial_hidden_state):
		state = initial_hidden_state
		outputs = []
		hidden = []
		for input_t in input_sequence:
			inputs = (input_t, state)
			output_t, state = self.rnnStep(inputs)
			outputs.append(output_t)
			hidden.append(state)
		last_hidden = state
		outputs = torch.stack(outputs)
		return outputs, hidden, last_hidden

	def forecast(self, input_t, initial_hidden_state, horizon):
		state = initial_hidden_state
		outputs = []
		hidden = []
		for i in range(horizon):
			inputs = (input_t, state)
			output_t, state = self.rnnStep(inputs)
			outputs.append(output_t)
			hidden.append(state)
			input_t = output_t
		last_hidden = state
		outputs = torch.stack(outputs)
		return outputs, hidden, last_hidden

	def getPartialDerivative(self, var1, var2):
		partial = []
		for r in range(var1.size()[1]):
			temp = torch.zeros(var1.size())
			temp[0, r] = 1.0
			temp = self.torch_dtype(temp)
			var1.backward(temp, retain_graph=True)
			grads = var2.grad.data.cpu().numpy().copy()
			partial.append(grads[0].copy())
			var2.grad.data.zero_()
		partial = np.array(partial)
		return partial

	def getRNNJacobian(self, h_t, h_t_1):
		jacobian = self.getPartialDerivative(h_t, h_t_1)
		self.zero_grad()
		return jacobian

	def outputMapping(self, h_t_1):
		if self.cell_str == "gru":
			o_t = self.rnn_mlp(h_t_1)
		else:
			raise ValueError("Not implemented.")
		return o_t

	def reccurentMapping(self, o_t, h_t_1):
		# Ignore is_train (only relevant for Zoneout)
		h_t = self.rnn_cell(o_t, h_t_1)
		return h_t

	def lyapunovPropagation(self, h_t_1):
		# print("INSIDE lyapunovPropagation(self, h_t_1)")
		# time.sleep(2)
		if self.cell_str == "gru":
			h_t_1 = Variable(self.torch_dtype(h_t_1), requires_grad=True)
			o_t = self.outputMapping(h_t_1)
			h_t = self.reccurentMapping(o_t, h_t_1)
		else:
			raise ValueError("Not implemented.")
		return h_t, o_t, h_t_1

	def rnnStep(self, inputs):
		# print("rnnStep()")
		# x [batch_size, RDIM]
		# hx [batch_size, NUM_HIDDEN_UNITS]
		x, state = inputs
		state = self.rnn_cell(x, state)
		if self.cell_str == "gru":
			x = self.rnn_mlp(state)
		else:
			raise ValueError("Not implemented.")
		return x, state

	def detachState(self, state):
		if self.cell_str == "gru":
			state = state.detach()
			return state
		else:
			raise ValueError("Not implemented.")

	def sendModelToCuda(self):
		print("Lyapunov exponent calculation: Sending model to GPU.")
		self.rnn_cell.cuda()
		self.rnn_mlp.cuda()
		return 0

	def calculate(self, input_sequence, dt, num_lyaps, model):
		print("Calculating lypunov spectrum (entering calculate()).")

		print("## Checking if CUDA: ##")
		print("self.rnn_cell		  = {:}".format(self.checkIfModelOnCuda(self.rnn_cell)))
		print("self.rnn_mlp		   = {:}".format(self.checkIfModelOnCuda(self.rnn_mlp)))
		self.gpu = torch.cuda.is_available()

		if self.gpu:
			print("Model in Lyapunov exponent calculation running on GPU.")
			torch.set_default_tensor_type(torch.cuda.DoubleTensor)
			self.torch_dtype = torch.cuda.DoubleTensor
			if self.cudnn_benchmark: torch.backends.cudnn.benchmark = True
		else:
			print("Model in Lyapunov exponent calculation running on CPU.")
			self.torch_dtype = torch.DoubleTensor
			torch.set_default_tensor_type(torch.DoubleTensor)


		if self.gpu:
			self.sendModelToCuda()

		print("CHECKING IF CUDA:")
		print("self.rnn_cell		  = {:}".format(self.checkIfModelOnCuda(self.rnn_cell)))
		print("self.rnn_mlp		   = {:}".format(self.checkIfModelOnCuda(self.rnn_mlp)))
		print("model.model.RNN[0]	 = {:}".format(self.checkIfModelOnCuda(model.model.RNN[0])))
		print("model.model.RNN_OUTPUT[0] = {:}".format(self.checkIfModelOnCuda(model.model.RNN_OUTPUT[0])))


		print("Inside calculation.")
		n_warmup = self.n_warmup
		iterative_prediction_length = self.iterative_prediction_length
		print("Number of warm-up steps={:}.".format(n_warmup))

		N = np.shape(input_sequence)[0]

		# PREDICTION LENGTH
		if N - n_warmup != iterative_prediction_length: raise ValueError("Error! N ({:}) - self.n_warmup ({:}) != iterative_prediction_length ({:})".format(N, n_warmup, iterative_prediction_length))
		# PREPARING THE HIDDEN STATES
		assert n_warmup > 1, "Warm up steps cannot be <= 1. Increase the iterative prediction length."

		warmup_data_input = input_sequence[:n_warmup-1]
		warmup_data_input = warmup_data_input[:, np.newaxis, :]
		warmup_data_input = self.torch_dtype(warmup_data_input)

		batch_size = 1
		initial_hidden_states = Variable(torch.zeros(batch_size, self.NUM_HIDDEN_UNITS))


		print("Forwarding the warm-up period.")
		warmup_data_output, _, last_hidden = self.forward(warmup_data_input, initial_hidden_states)
		# Removing last point, as it will be given by the last_hidden state in the LS calculation
		warmup_data_output = warmup_data_output[:-1]
		warmup_data_output = warmup_data_output.cpu().detach().numpy().copy()
		warmup_data_output = warmup_data_output[:,0,:]


		################################################
		## DEFAULT PARAMETERS
		################################################
		print("Using default parameters.")
		norm_time = 10
		# TT = 1000
		TT = int(np.floor(iterative_prediction_length/norm_time))
		assert TT>0, "Error, TT = int(np.floor(iterative_prediction_length/norm_time)) is not positive."
		print("TT={:}".format(TT))

		TT_every = 100
		TT_save = 100
		TRESH = 1e-4

		pl = TT*norm_time
		RDIM = input_sequence.shape[1]
		print(np.shape(input_sequence))
		pl_data = np.shape(input_sequence)[0]

		# RDIM=N is the dimension of the system's hidden state
		target_sequence = input_sequence[:n_warmup+pl]
		# test_data_input = test_data_input[:TT*norm_time]
		print("Target shape:")
		print(np.shape(target_sequence))

		# time.sleep(5)

		delta_dim = self.NUM_HIDDEN_UNITS

		print("Orthonormal delta of dimensions {:}x{:}.".format(delta_dim,num_lyaps))
		delta = scipy.linalg.orth(np.random.rand(delta_dim,num_lyaps))

		prediction = np.zeros((pl, RDIM))
		R_ii = np.zeros((num_lyaps,int(pl/norm_time)), dtype=np.complex64)

		prediction = []
		exponents_vec = []
		diff_vec = []
		exponents_temp_prev = np.zeros((num_lyaps))
		ITER = 0
		# time.sleep(5)

		self.tqdm = tqdm(total=pl+2)

		h_t_1 = last_hidden
		for t in range(-1, pl+1):
			# print("Time {:}/{:}".format(t,pl))
			# time.sleep(2)

			# print("h_t, o_t, h_t_1 = self.lyapunovPropagation(h_t_1)")
			h_t, o_t, h_t_1 = self.lyapunovPropagation(h_t_1)
			# time.sleep(2)

			if t >= 0:

				jacobian = self.getRNNJacobian(h_t, h_t_1)
				self.zero_grad()

				delta = np.matmul(jacobian, delta)
				if t % norm_time == 0:
					QQ, RR = np.linalg.qr(delta)
					delta = QQ[:,:num_lyaps]
					R_ii[:,int(t/norm_time)-1] = log(np.diag(RR[:num_lyaps,:num_lyaps]))

					if (t/norm_time) % TT_every == 0:
						exponents_temp = np.real(np.sum(R_ii,1))/((t-1)*dt)
						exponents_temp = np.sort(exponents_temp)
						diff = np.linalg.norm(exponents_temp - exponents_temp_prev)
						ITER = ITER + 1
						exponents_temp_prev = exponents_temp
						print("Time {:}/{:}, {:3.2f}%".format(t,pl,t/pl*100.0))
						print("LE: {:}".format(exponents_temp))
						print("Difference {:.4f}".format(diff))
						diff_vec.append(diff)
						exponents_vec.append(exponents_temp)
						if diff < TRESH:
							print("TERMINATED AFTER {:} ITERATIONS".format(ITER))
							break;

					if (t/norm_time) % TT_save == 0:
						exponents = exponents_vec[-1]
						# exponents = np.real(np.sum(R_ii,1))/(pl*dt)
						exponents_sum = np.sum(exponents)
						results = {
						"TT_every":TT_every,
						"TRESH":TRESH,
						"ITER":ITER,
						"diff_vec":diff_vec,
						"exponents_vec":exponents_vec,
						"exponents_sum":exponents_sum,
						"exponents":exponents_vec[-1],
						"norm_time":norm_time,
						"num_lyaps":num_lyaps,
						"dt":dt,
						"TT":TT,
						}
						data_path = self.results_dir + '/le_results_N{:}_ITER{:}'.format(iterative_prediction_length, ITER)
						saveDataPickle(results, data_path)

			# GRADIENTS_PREV = GRADIENTS
			prediction.append(o_t.detach().cpu().numpy().copy())
			o_t = self.detachState(o_t)
			h_t = self.detachState(h_t)
			h_t_1 = self.detachState(h_t)
			self.tqdm.update(1)
		self.tqdm.close()

		exponents = exponents_vec[-1]
		# exponents = np.real(np.sum(R_ii,1))/(pl*dt)
		exponents_sum = np.sum(exponents)

		print("Sum of exponents is:")
		print(exponents_sum)

		print("Exponents are:")
		print(exponents)

		prediction = np.array(prediction)
		prediction = prediction[:,0,:]

		output_sequence = np.concatenate((warmup_data_output, prediction), axis=0)
		print(np.shape(output_sequence))

		results = {
		"n_warmup":n_warmup,
		"TT_every":TT_every,
		"TRESH":TRESH,
		"ITER":ITER,
		"diff_vec":diff_vec,
		"exponents_vec":exponents_vec,
		"exponents_sum":exponents_sum,
		"exponents":exponents,
		"norm_time":norm_time,
		"num_lyaps":num_lyaps,
		"output_sequence":output_sequence,
		"target_sequence":target_sequence,
		"dt":dt,
		"TT":TT,
		}
		data_path = self.results_dir + '/le_results_N{:}'.format(iterative_prediction_length)
		saveDataPickle(results, data_path)



