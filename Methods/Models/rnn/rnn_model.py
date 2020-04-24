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

# PRINTING
from functools import partial
print = partial(print, flush=True)


class rnn_model(nn.Module):
	def __init__(self, params, model):
		super(rnn_model, self).__init__()
		self.parent = model

		# Determining cell type
		if self.parent.params["rnn_cell_type"]=="lstm":
			self.rnn_cell = nn.LSTMCell
		elif self.parent.params["rnn_cell_type"]=="gru":
			self.rnn_cell = nn.GRUCell
		else:
			raise ValueError("Invalid rnn_cell_type {:}".format(params["rnn_cell_type"]))

		self.buildNetwork()

	def ifAnyIn(self, list_, name):
		for element in list_:
			if element in name:
				return True
		return False

	def initializeWeights(self):
		print("INITIALIZING PARAMETERS...\n")
		for modules in self.module_list:
			for module in modules:
				for name, param in module.named_parameters():
					print(name)
					# INITIALIZING RNN, GRU CELLS
					if 'weight_ih' in name:
						torch.nn.init.xavier_uniform_(param.data)

					elif 'weight_hh' in name:
						torch.nn.init.orthogonal_(param.data)

					elif self.ifAnyIn(["Wxi.weight", "Wxf.weight", "Wxc.weight", "Wxo.weight"], name):
						torch.nn.init.xavier_uniform_(param.data)

					elif self.ifAnyIn(["Wco", "Wcf", "Wci", "Whi.weight", "Whf.weight", "Whc.weight", "Who.weight"], name):
						torch.nn.init.orthogonal_(param.data)

					elif self.ifAnyIn(["Whi.bias", "Wxi.bias", "Wxf.bias", "Whf.bias", "Wxc.bias", "Whc.weight", "Wxo.bias", "Who.bias"], name):
						param.data.fill_(0)

					elif 'weight' in name:
						torch.nn.init.xavier_uniform_(param.data)

					elif 'bias' in name:
						param.data.fill_(0)
					else:
						raise ValueError("NAME {:} NOT FOUND!".format(name))
						# print("NAME {:} NOT FOUND!".format(name))
		print("PARAMETERS INITIALISED!")
		return 0

	def sendModelToCPU(self):
		print("SENDING MODEL TO CPU")
		for modules in self.module_list:
			for model in modules:
				model.cpu()
		return 0

	def sendModelToCuda(self):
		print("SENDING MODEL TO CUDA")
		for modules in self.module_list:
			for model in modules:
				model.cuda()
		return 0

	def buildNetwork(self):
		# Parsing the layers of the RNN
		input_size = self.parent.rnn_state_dim
		self.RNN = nn.ModuleList()
		for ln in range(len(self.parent.layers_rnn)):
			self.RNN.append(self.rnn_cell(input_size, self.parent.layers_rnn[ln]))
			input_size = self.parent.layers_rnn[ln]

		# Output MLP of the RNN
		self.RNN_OUTPUT = nn.ModuleList()
		self.RNN_OUTPUT.extend([nn.Linear(self.parent.layers_rnn[-1], self.parent.rnn_state_dim, bias=True)])

		self.module_list = [self.RNN, self.RNN_OUTPUT]
		return 0

	def countTrainableParams(self):
		temp = 0
		for layers in self.module_list:
			for layer in layers:
				temp+= sum(p.numel() for p in layer.parameters() if p.requires_grad)
		return temp

	def countParams(self):
		temp = 0
		for layers in self.module_list:
			for layer in layers:
				temp+= sum(p.numel() for p in layer.parameters())
		return temp

	def getParams(self):
		params = list()
		for layers in self.module_list:
			for layer in layers:
				params += layer.parameters()
		return params

	def getRNNParams(self):
		params = list()
		for layers in [self.RNN, self.RNN_OUTPUT, self.PREDICTOR]:
			for layer in layers:
				params += layer.parameters()
		return params

	def printModuleList(self):
		print(self.module_list)
		return 0

	def eval(self):
		for modules in [self.RNN, self.RNN_OUTPUT]:
			for layer in modules:
				layer.eval()
		return 0

	def train(self):
		for modules in [self.RNN, self.RNN_OUTPUT]:
			for layer in modules:
				layer.train()
		return 0

	def forecast(self, inputs, init_hidden_state, horizon=None, is_train=False, teacher_forcing_forecasting=0):
		if is_train:
			self.train()
		else:
			self.eval()

		with torch.set_grad_enabled(is_train):
			outputs = []
			rnn_internal_states = []
			rnn_outputs = []

			K, T, D = inputs.size()


			if (horizon is not None):
				if (not (K==1)) or (not (T==1)):
					raise ValueError("Forward iterative called with K!=1 or T!=1 and a horizon. This is not allowed! K={:}, T={:}, D={:}".format(K,T,D))
				else:
					# Horizon is not None and T=1, so forecast called in the testing phase
					pass
			else:
				horizon = T
				# If teacher forcing equals the horizon (input time dimension T), the network never propagates its predictions
				assert teacher_forcing_forecasting <= horizon, "Teacher forcing {:} cannot be bigger than the horizon {:}.".format(teacher_forcing_forecasting, horizon)
				assert teacher_forcing_forecasting >= 0, "Teacher forcing {:} cannot be negative.".format(teacher_forcing_forecasting)

			# When T>1, only inputs[:,0,:] is taken into account. The network is propagating its own predictions.
			input_t = inputs[:,0].view(K, 1, *inputs.size()[2:])

			assert(T>0)
			assert(horizon>0)
			for t in range(horizon):

				output, next_hidden_state, rnn_output = self.forward_(input_t, init_hidden_state, is_train=is_train)
				if t >= teacher_forcing_forecasting:
					# Iterative prediction:
					assert teacher_forcing_forecasting != self.parent.sequence_length
					input_t = output
				else:
					input_t = inputs[:,t].view(K, 1, *inputs.size()[2:])

				outputs.append(output[:, 0])

				rnn_internal_states.append(next_hidden_state)
				rnn_outputs.append(rnn_output[:,0])
				init_hidden_state = next_hidden_state

			outputs = torch.stack(outputs)
			outputs = outputs.transpose(1,0)
			rnn_outputs = torch.stack(rnn_outputs)
			rnn_outputs = rnn_outputs.transpose(1,0)
		return outputs, next_hidden_state, rnn_outputs


	def transposeHiddenState(self, hidden_state):
		# Transpose hidden state from batch_first to Layer first
		# (gru)  [K, L, H]	-> [L, K, H] 
		# (lstm) [K, 2, L, H] -> [L, 2, K, H]
		if self.parent.params["rnn_cell_type"]=="gru": 
			hidden_state = hidden_state.transpose(0, 1) #
		elif self.parent.params["rnn_cell_type"]=="lstm": 
			hidden_state = hidden_state.transpose(0, 2) # (lstm)
		else:
			raise ValueError("rnn_cell_type {:} not recognized".format(self.parent.params["rnn_cell_type"]))
		return hidden_state

	def forward(self, inputs, init_hidden_state, is_train=False, is_iterative_forecasting=False, horizon=None, teacher_forcing_forecasting=0):

		if is_iterative_forecasting:
			return self.forecast(inputs, init_hidden_state, horizon, is_train=is_train, teacher_forcing_forecasting=teacher_forcing_forecasting)
		else:
			return self.forward_(inputs, init_hidden_state, is_train=is_train)

	def forward_(self, inputs, init_hidden_state, is_train=True):

		# TRANSPOSE FROM BATCH FIRST TO LAYER FIRST
		init_hidden_state = self.transposeHiddenState(init_hidden_state)

		if is_train:
			self.train()
		else:
			self.eval()

		with torch.set_grad_enabled(is_train):
			K, T, D = inputs.size()
			inputs = inputs.transpose(1,0)

			if (K != self.parent.batch_size and is_train==True and (not self.parent.device_count>1)): raise ValueError("Batch size {:d} does not match {:d} and model not in multiple GPUs.".format(K, self.parent.batch_size))

			rnn_outputs, next_hidden_state = self.forwardRNN(inputs, init_hidden_state, is_train)

			outputs = self.forwardRNNOutput(rnn_outputs)

			# TRANSPOSING BATCH_SIZE WITH TIME
			rnn_outputs = rnn_outputs.transpose(1,0).contiguous()
			outputs = outputs.transpose(1,0).contiguous()
			next_hidden_state = self.transposeHiddenState(next_hidden_state)

		return outputs, next_hidden_state, rnn_outputs

	def forwardRNN(self, inputs, init_hidden_state, is_train):
		T = inputs.size()[0]
		rnn_outputs = []
		for t in range(T):
			input_t = inputs[t]
			next_hidden_state = []
			for ln in range(len(self.RNN)):
				hidden_state = init_hidden_state[ln]

				# TRANSFORMING THE HIDEN STATE TO TUPLE FOR LSTM
				# if not isinstance(hidden_state, tuple):
				# if len(hidden_state.size())==3: # [2, K, H]
				if self.parent.params["rnn_cell_type"]=="lstm":
					hx, cx = init_hidden_state[ln]
					hidden_state = tuple([hx, cx])

				rnn_output = self.RNN[ln].forward(input_t, hidden_state)
				next_hidden_state_layer = rnn_output

				if self.parent.params["rnn_cell_type"]=="lstm":
					hx, cx = next_hidden_state_layer
					next_hidden_state_layer = torch.stack([hx, cx])

				next_hidden_state.append(next_hidden_state_layer)
				input_t = rnn_output

			init_hidden_state = next_hidden_state
			rnn_outputs.append(rnn_output)

		rnn_outputs = torch.stack(rnn_outputs)
		next_hidden_state = torch.stack(next_hidden_state)

		return rnn_outputs, next_hidden_state

	def forwardRNNOutput(self, inputs):
		outputs = self.RNN_OUTPUT[0](inputs)
		return outputs



