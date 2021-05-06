#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import pickle
import io
import os
import time

######################################################
## UTILITIES
######################################################
from time_utils import *
from data_utils import *


def writeToLogFile(model, logfile, data, fields_to_write):
	with io.open(logfile, 'a+') as f:
		f.write("model_name:" + str(model.model_name))
		for field in fields_to_write:
			assert field in data , "field {:} not in data".format(field)
			f.write(":{:}:{:}".format(field, data[field]))
		f.write("\n")
	return 0

def getReferenceTrainingTime(rtt, btt):
	reference_train_time = 60*60*(rtt-btt)
	print("Reference train time {:}".format(secondsToTimeStr(reference_train_time)))
	return reference_train_time

def countTrainableParams(layers):
	temp = 0
	for layer in layers:
		temp+= sum(p.numel() for p in layer.parameters() if p.requires_grad)
	return temp

def countParams(layers):
	temp = 0
	for layer in layers:
		temp+= sum(p.numel() for p in layer.parameters())
	return temp

def replaceNaN(data):
	data[np.isnan(data)]=float('Inf')
	return data

def computeErrors(target, prediction, std):
	prediction = replaceNaN(prediction)
	spatial_dims = tuple([*range(len(np.shape(target)))[1:]])
	# ABSOLUTE ERROR
	abserror = np.abs(target-prediction)
	# NORMALIZED ABSOLUTE DIFFERENCE
	nad = abserror /(np.max(target) - np.min(target))
	mnad = np.mean(nad, axis=spatial_dims)
	# SQUARE ERROR
	serror = np.square(target-prediction)
	# MEAN (over-space) SQUARE ERROR
	mse = np.mean(serror, axis=spatial_dims)
	# ROOT MEAN SQUARE ERROR
	rmse = np.sqrt(mse)
	# NORMALIZED SQUARE ERROR
	nserror = serror/np.square(std)
	# MEAN (over-space) NORMALIZED SQUARE ERROR
	mnse = np.mean(nserror, axis=spatial_dims)
	# ROOT MEAN NORMALIZED SQUARE ERROR
	rmnse = np.sqrt(mnse)
	num_accurate_pred_005 = getNumberOfAccuratePredictions(rmnse, 0.05)
	num_accurate_pred_050 = getNumberOfAccuratePredictions(rmnse, 0.5)
	return rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, mnad


def computeFrequencyError(predictions_all, targets_all, dt):
	spatial_dims = len(np.shape(predictions_all)[2:])
	# print(spatial_dims)
	if spatial_dims == 1:
		sp_pred, freq_pred = computeSpectrum(predictions_all, dt)
		sp_true, freq_true = computeSpectrum(targets_all, dt)
		# s_dbfs = 20 * np.log10(s_mag)
		# TRANSFORM TO AMPLITUDE FROM DB
		sp_pred = np.exp(sp_pred/20.0)
		sp_true = np.exp(sp_true/20.0)
		error_freq = np.mean(np.abs(sp_pred - sp_true))
		return freq_pred, freq_true, sp_true, sp_pred, error_freq
	elif spatial_dims == 3:
		# RGB Image channells (Dz) of Dy x Dx
		# Applying two dimensional FFT
		sp_true = computeSpectrum2D(targets_all)
		sp_pred = computeSpectrum2D(predictions_all)
		error_freq = np.mean(np.abs(sp_pred - sp_true))
		return None, None, sp_pred, sp_true, error_freq
	elif spatial_dims == 2:
		nics, T, n_o, Dx = np.shape(predictions_all)
		predictions_all = np.reshape(predictions_all, (nics, T, n_o*Dx))
		targets_all = np.reshape(targets_all, (nics, T, n_o*Dx))
		sp_pred, freq_pred = computeSpectrum(predictions_all, dt)
		sp_true, freq_true = computeSpectrum(targets_all, dt)
		# s_dbfs = 20 * np.log10(s_mag)
		# TRANSFORM TO AMPLITUDE FROM DB
		sp_pred = np.exp(sp_pred/20.0)
		sp_true = np.exp(sp_true/20.0)
		error_freq = np.mean(np.abs(sp_pred - sp_true))
		return freq_pred, freq_true, sp_true, sp_pred, error_freq
	else:
		raise ValueError("Not implemented.")

def addNoise(data, percent):
	std_data = np.std(data, axis=0)
	std_data = np.reshape(std_data, (1, *data.shape[1:]))
	std_data = np.repeat(std_data, np.shape(data)[0], axis=0)
	noise = np.multiply(np.random.randn(*np.shape(data)), percent/1000.0*std_data)
	data += noise
	return data

class scaler(object):
	def __init__(self, tt, multiple_ics_in_data=False):
		self.tt = tt
		self.data_min = 0
		self.data_max = 0
		self.data_mean = 0
		self.data_std = 0
		self.multiple_ics_in_data = multiple_ics_in_data
		self.slack = 1e-6

	def scaleData(self, input_sequence, reuse=None, without_first_dim=False):
		if reuse is not None:
			shape_length = len(np.shape(input_sequence))
			if not without_first_dim:
				pass
			elif without_first_dim:
				input_sequence = input_sequence[np.newaxis]
		else:
			self.data_shape_length = len(np.shape(input_sequence))

		if not self.multiple_ics_in_data:
			axes_ = 0
		elif self.multiple_ics_in_data:
			axes_ = (0,1)

		if reuse == None:
			self.data_mean = np.mean(input_sequence,axis=axes_)
			self.data_std = np.std(input_sequence,axis=axes_)
			self.data_min = np.min(input_sequence,axis=axes_) - self.slack
			self.data_max = np.max(input_sequence,axis=axes_) + self.slack

		D1 = np.shape(input_sequence)[0] # Number of ICs or number of timesteps
		D2 = np.shape(input_sequence)[1] # Number of timesteps of number of particles in permutation invariant data
		D3 = None

		if self.tt == "MinMaxZeroOne":
			data_min = self.repeatScalerParam(self.data_min, D1, D2, D3)
			data_max = self.repeatScalerParam(self.data_max, D1, D2, D3)
			input_sequence = np.array((input_sequence-data_min)/(data_max-data_min))
		elif self.tt == "Standard" or self.tt == "standard":
			data_mean = self.repeatScalerParam(self.data_mean, D1, D2, D3)
			data_std = self.repeatScalerParam(self.data_std, D1, D2, D3)
			input_sequence = np.array((input_sequence-data_mean)/data_std)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		if without_first_dim: input_sequence = input_sequence[0]
		return input_sequence

	def repeatScalerParam(self, data, D1, D2, D3):
		if not self.multiple_ics_in_data:
			data = np.repeat(data[np.newaxis], D1, 0)
		elif self.multiple_ics_in_data:
			data = np.repeat(data[np.newaxis], D2, 0)
			data = np.repeat(data[np.newaxis], D1, 0)
		return data

	def descaleData(self, input_sequence, without_first_dim=False):
		shape_length = len(np.shape(input_sequence))
		if not without_first_dim:
			# assert(self.data_shape_length == shape_length)
			pass
		elif without_first_dim:
			input_sequence = input_sequence[np.newaxis]

		D1 = np.shape(input_sequence)[0]
		D2 = np.shape(input_sequence)[1]
		D3 = None

		# NORMAL RNN
		if self.tt == "MinMaxZeroOne":
			data_min = self.repeatScalerParam(self.data_min, D1, D2, D3)
			data_max = self.repeatScalerParam(self.data_max, D1, D2, D3)
			input_sequence = np.array(input_sequence*(data_max - data_min) + data_min)
		elif self.tt == "Standard" or self.tt == "standard":
			data_mean = self.repeatScalerParam(self.data_mean, D1, D2, D3)
			data_std = self.repeatScalerParam(self.data_std, D1, D2, D3)
			input_sequence = np.array(input_sequence*data_std + data_mean)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		if without_first_dim: input_sequence = input_sequence[0]
		return np.array(input_sequence)

def dbfft2D(x):
	# !! SPATIAL FFT !!
	N = len(x)  # Length of input sequence
	if N % 2 != 0:
		x = x[:-1, :-1]
		N = len(x)
	x = np.reshape(x, (N,N))
	# Calculate real FFT and frequency vector
	sp = np.fft.fft2(x)
	sp = np.fft.fftshift(sp)
	s_mag = np.abs(sp)
	# Convert to dBFS
	s_dbfs = 20 * np.log10(s_mag)
	return s_dbfs


def computeSpectrum(data_all, dt):
	# Of the form [n_ics, T, n_dim]
	spectrum_db = []
	for data in data_all:
		data = np.transpose(data)
		for d in data:
			freq, s_dbfs = dbfft(d, 1/dt)
			spectrum_db.append(s_dbfs)
	spectrum_db = np.array(spectrum_db).mean(axis=0)
	return spectrum_db, freq

def dbfft(x, fs):
	# !!! TIME DOMAIN FFT !!!
	"""
	Calculate spectrum in dB scale
	Args:
		x: input signal
		fs: sampling frequency
	Returns:
		freq: frequency vector
		s_db: spectrum in dB scale
	"""
	N = len(x)  # Length of input sequence
	if N % 2 != 0:
		x = x[:-1]
		N = len(x)
	x = np.reshape(x, (1,N))
	# Calculate real FFT and frequency vector
	sp = np.fft.rfft(x)
	freq = np.arange((N / 2) + 1) / (float(N) / fs)
	# Scale the magnitude of FFT by window and factor of 2,
	# because we are using half of FFT spectrum.
	s_mag = np.abs(sp) * 2 / N
	# Convert to dBFS
	s_dbfs = 20 * np.log10(s_mag)
	s_dbfs = s_dbfs[0]
	return freq, s_dbfs


def getNumberOfAccuratePredictions(nerror, tresh=0.05):
	nerror_bool = nerror < tresh
	n_max = np.shape(nerror)[0]
	n = 0
	while nerror_bool[n] == True:
		n += 1
		if n == n_max: break
	return n
	
def getFirstDataDimension(var):
	if isinstance(var, (list,)):
		dim = len(var)
	elif type(var) == np.ndarray:
		dim = np.shape(var)[0]
	elif  type(var) == np.matrix:
		raise ValueError("Variable is a matrix. NOT ALLOWED!")
	else:
		raise ValueError("Variable not a list or a numpy array. No dimension to compute!")
	return dim

def divideData(data, train_val_ratio, multiple_ics=False):
	str_ = "samples" if not multiple_ics else "ICs"
	print("Dividing data.")
	n_samples = getFirstDataDimension(data)
	print("Total number of {:} {:}".format(str_, n_samples))
	n_train = int(n_samples*train_val_ratio)
	print("Training {:} {:}".format(str_, n_train))
	data_train = data[:n_train]
	data_val = data[n_train:]
	print(np.shape(data_train))
	print(np.shape(data_val))
	return data_train, data_val


def getMDARNNParser(parser):
	parser.add_argument("--mode", help="train, test, all", type=str, required=True)
	parser.add_argument("--system_name", help="system_name", type=str, required=True)
	parser.add_argument("--write_to_log", help="write_to_log", type=int, required=False, default=0)
	parser.add_argument("--N", help="N", type=int, required=True)
	parser.add_argument("--N_used", help="N_used", type=int, required=True)

	parser.add_argument("--input_dim", help="input_dim", type=int, required=True)
	parser.add_argument("--train_val_ratio", help="train_val_ratio", type=float, required=True)

	parser.add_argument("--rnn_cell_type", help="type of the rnn cell", type=str, required=False, default="gru")
	parser.add_argument('--rnn_layers_size', type=int, help='size of the RNN layers', required=False, default=0)
	parser.add_argument('--rnn_layers_num', type=int, help='number of the RNN layers', required=False, default=0)

	parser.add_argument("--sequence_length", help="sequence_length", type=int, required=True)
	parser.add_argument("--hidden_state_propagation_length", help="hidden_state_propagation_length", type=int, required=True)
	parser.add_argument("--scaler", help="scaler", type=str, required=True)
	parser.add_argument("--noise_level", help="noise level per mille in the training data", type=int, default=0, required=True)
	parser.add_argument("--skip", help="skip", type=int, required=False, default=1)

	parser.add_argument("--learning_rate", help="learning_rate", type=float, required=True)
	parser.add_argument("--weight_decay", help="weight_decay", type=float, required=False, default=0.0)
	parser.add_argument("--batch_size", help="batch_size", type=int, required=True)
	parser.add_argument("--overfitting_patience", help="overfitting_patience", type=int, required=True)
	parser.add_argument("--max_epochs", help="max_epochs", type=int, required=True)
	parser.add_argument("--max_rounds", help="max_rounds", type=int, required=True)
	parser.add_argument("--retrain", help="retrain", type=int, required=True)
	parser.add_argument("--num_test_ICS", help="num_test_ICS", type=int, required=False, default=0)
	
	parser.add_argument("--iterative_prediction_length", help="iterative_prediction_length", type=int, required=False, default=0)
	parser.add_argument("--display_output", help="control the verbosity level of output , default True", type=int, required=False, default=1)
	parser.add_argument("--reference_train_time", help="The reference train time in hours", type=float, default=24, required=False)
	parser.add_argument("--buffer_train_time", help="The buffer train time to save the model in hours", type=float, default=0.5, required=False)
	parser.add_argument("--random_seed", help="random_seed", type=int, default=1, required=False)
	parser.add_argument("--optimizer_str", help="adam or sgd with cyclical learning rate", type=str, default="adam", required=False)

	parser.add_argument("--teacher_forcing_forecasting", help="to test the the model in teacher forcing.", type=int, default=0, required=False)
	parser.add_argument("--iterative_state_forecasting", help="to test the model in iterative forecasting, propagating the output state of the model.", type=int, default=0, required=False)
	parser.add_argument("--cudnn_benchmark", help="cudnn_benchmark", type=int, default=0, required=False)


	parser.add_argument("--multiple_ics_in_data", help="if the training data are comming from multiple initial conditions.", type=int, default=0, required=False)
	parser.add_argument("--num_lyaps", help="Number of lyapunov exponents to be calculated.", type=int, default=0, required=False)

	return parser

