#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import socket
import os
import subprocess

from scipy.signal import savgol_filter

# Plotting parameters
import matplotlib

hostname = socket.gethostname()
print("PLOTTING HOSTNAME: {:}".format(hostname))
CLUSTER = True if ((hostname[:2]=='eu')  or (hostname[:5]=='daint') or (hostname[:3]=='nid') or (hostname[:14]=='barrycontainer')) else False

if (hostname[:2]=='eu'):
	CLUSTER_NAME = "euler" 
elif (hostname[:5]=='daint'):
	CLUSTER_NAME = "daint"
elif (hostname[:3]=='nid'):
	CLUSTER_NAME = "daint"
elif (hostname[:14]=='barrycontainer'):
	CLUSTER_NAME = "barry"
else:
	CLUSTER_NAME = "local"

print("CLUSTER={:}, CLUSTER_NAME={:}".format(CLUSTER, CLUSTER_NAME))

if CLUSTER: matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm
from mpl_toolkits import mplot3d
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

print("-V- Matplotlib Version = {:}".format(matplotlib.__version__))

from matplotlib import colors
import six
color_dict = dict(six.iteritems(colors.cnames))
# GENERATING FIGURES
# color_labels = ['blue', 'green', 'brown', 'darkcyan', 'purple', 'orange', 'darkorange']
# color_labels = ['blue', 'green', 'orange', 'cadetblue', 'maroon', 'brown', 'springgreen', 'deepskyblue', 'lightgrey', 'rebeccapurple', 'darkcyan', 'purple', 'darkorange' , 'cadetblue', 'honeydew', 'lightskyblue', 'greenyellow', 'paleturquoise', 'darkmagenta', 'darkturquoise', 'palegoldenrod', 'lightpink']

color_labels = ['tab:red', 'tab:blue', 'tab:green', 'tab:brown', 'tab:orange', 'tab:cyan', 'tab:olive', 'tab:pink', 'tab:gray', 'tab:purple']

linestyles = ['-','--','-.',':','-','--','-.',':']
linemarkers = ["s","d", "o",">","*","x","<",">"]
linemarkerswidth = [3,2,2,2,4,2,2,2]


FONTSIZE=18
font = {'size':FONTSIZE, 'family':'Times New Roman'}
matplotlib.rc('xtick', labelsize=FONTSIZE) 
matplotlib.rc('ytick', labelsize=FONTSIZE) 
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

if CLUSTER_NAME in ["local", "barry"]:
	# Plotting parameters
	rc('text', usetex=True)
	plt.rcParams["text.usetex"] = True
	plt.rcParams['xtick.major.pad']='10'
	plt.rcParams['ytick.major.pad']='10'

FIGTYPE="png"
# FIGTYPE="pdf"

from scipy.stats.stats import pearsonr

from global_utils import *

def plotMeanStd(data, ax, label, color, dt=1, with_std=True, with_samples=True, filtered=False, markeredgewidth=1, linestyle="-", marker=None):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	T = np.shape(data)[1]
	time_vector = np.arange(T) * dt

	if filtered:
		window_size = int(np.min([51, len(time_vector)]))
		if window_size % 2 == 0: window_size = window_size -1
		meanhat = savgol_filter(mean, window_size, 3) # window size 51, polynomial order 3
		mean = meanhat

	ax.plot(time_vector, mean, label=label, color=color, linestyle=linestyle, marker=marker, markeredgewidth=markeredgewidth, markersize=10, markevery=int(T/12), linewidth=2)
	if with_std: ax.fill_between(np.arange(np.shape(mean)[0]), mean+2*std, mean-2*std, facecolor=color, alpha=0.6)
	if with_samples: ax.plot(time_vector, np.array(data).T, lw=1, color=color, alpha=0.3)
	ax.set_xlim([np.min(time_vector), np.max(time_vector)])
	ax.set_xlabel(r"$t$")
	return ax


def plotAllLosses(model, losses_train, time_train, losses_val, time_val, min_val_error, name_str=""):
	loss_labels = ["TOTAL"]
	idx1 = np.nonzero(losses_train[0])[0]
	idx2 = np.nonzero(losses_train[-1])[0]
	idx = np.union1d(idx1, idx2)
	losses_val = np.array(losses_val)
	losses_train = np.array(losses_train)
	color_labels = ['blue', 'green', 'brown', 'darkcyan', 'purple']
	min_val_epoch = np.argmin(np.abs(np.array(losses_val[:,0])-min_val_error))
	min_val_time = time_val[min_val_epoch]

	losses_train=losses_train[:,idx]
	losses_val=losses_val[:,idx]
	loss_labels=[loss_labels[i] for i in idx]
	color_labels=[color_labels[i] for i in idx]

	time_train = np.array(time_train)
	time_val = np.array(time_val)

	if np.all(np.array(losses_train)>0.0) and np.all(np.array(losses_val)>0.0):
		losses_train = np.log10(losses_train)
		losses_val = np.log10(losses_val)
		min_val_error_log = np.log10(min_val_error)
		if len(time_train)>1:
			for time_str in ["", "_time"]:
				fig_path = model.getFigureDir() + "/losses_all_log"+ time_str + name_str + "." + FIGTYPE
				fig, ax = plt.subplots(figsize=(20,10))
				plt.title("MIN LOSS-VAL={:.4f}".format(min_val_error))
				max_i = np.min([np.shape(losses_train)[1], len(loss_labels)])
				for i in range(max_i):
					if time_str != "_time":
						x_axis_train = np.arange(np.shape(losses_train[:,i])[0])
						x_axis_val = np.arange(np.shape(losses_val[:,i])[0])
						min_val_axis = min_val_epoch
						ax.set_xlabel(r"Epoch")
					else:
						dt = time_train[1]-time_train[0]
						x_axis_train = time_train+i*dt
						x_axis_val = time_val+i*dt
						min_val_axis = min_val_time
						ax.set_xlabel(r"Time")
					plt.plot(x_axis_train, losses_train[:,i], color=color_dict[color_labels[i]], label=loss_labels[i] + " Train")
					plt.plot(x_axis_val, losses_val[:,i], color=color_dict[color_labels[i]], label=loss_labels[i] + " Val", linestyle="--")
				plt.plot(min_val_axis, min_val_error_log, "o", color=color_dict['red'], label="optimal")
				ax.set_ylabel(r"Log${}_{10}$(Loss)")
				plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
				plt.tight_layout()
				plt.savefig(fig_path)
				plt.close()
	else:
		if len(time_train)>1:
			for time_str in ["", "_time"]:
				fig_path = model.getFigureDir() + "/losses_all"+ time_str + name_str + "." + FIGTYPE
				fig, ax = plt.subplots(figsize=(20,10))
				plt.title("MIN LOSS-VAL={:.4f}".format(min_val_error))
				max_i = np.min([np.shape(losses_train)[1], len(loss_labels)])
				for i in range(max_i):
					if time_str != "_time":
						x_axis_train = np.arange(np.shape(losses_train[:,i])[0])
						x_axis_val = np.arange(np.shape(losses_val[:,i])[0])
						min_val_axis = min_val_epoch
						ax.set_xlabel(r"Epoch")
					else:
						dt = time_train[1]-time_train[0]
						x_axis_train = time_train+i*dt
						x_axis_val = time_val+i*dt
						min_val_axis = min_val_time
						ax.set_xlabel(r"Time")
					plt.plot(x_axis_train, losses_train[:,i], color=color_dict[color_labels[i]], label=loss_labels[i] + " Train")
					plt.plot(x_axis_val, losses_val[:,i], color=color_dict[color_labels[i]], label=loss_labels[i] + " Val", linestyle="--")
				plt.plot(min_val_axis, min_val_error, "o", color=color_dict['red'], label="optimal")
				ax.set_ylabel(r"Loss")
				plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
				plt.tight_layout()
				plt.savefig(fig_path)
				plt.close()


def plotTrainingLosses(model, loss_train, loss_val, min_val_error, name_str=""):
	if (len(loss_train) != 0) and (len(loss_val) != 0):
		min_val_epoch = np.argmin(np.abs(np.array(loss_val)-min_val_error))
		fig_path = model.getFigureDir() + "/loss_total"+ name_str + "." + FIGTYPE
		fig, ax = plt.subplots()
		plt.title("Validation error {:.10f}".format(min_val_error))
		plt.plot(np.arange(np.shape(loss_train)[0]), loss_train, color=color_dict['green'], label="Train RMSE")
		plt.plot(np.arange(np.shape(loss_val)[0]), loss_val, color=color_dict['blue'], label="Validation RMSE")
		plt.plot(min_val_epoch, min_val_error, "o", color=color_dict['red'], label="optimal")
		ax.set_xlabel(r"Epoch")
		ax.set_ylabel(r"Loss")
		plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
		plt.tight_layout()
		plt.savefig(fig_path)
		plt.close()

		
		loss_train = np.array(loss_train)
		loss_val = np.array(loss_val)
		if (np.all(loss_train[~np.isnan(loss_train)]>0.0) and np.all(loss_val[~np.isnan(loss_val)]>0.0)):
			fig_path = model.getFigureDir() + "/loss_total_log"+ name_str + "." + FIGTYPE
			fig, ax = plt.subplots()
			plt.title("Validation error {:.10f}".format(min_val_error))
			plt.plot(np.arange(np.shape(loss_train)[0]), np.log10(loss_train), color=color_dict['green'], label="Train RMSE")
			plt.plot(np.arange(np.shape(loss_val)[0]), np.log10(loss_val), color=color_dict['blue'], label="Validation RMSE")
			plt.plot(min_val_epoch, np.log10(min_val_error), "o", color=color_dict['red'], label="optimal")
			ax.set_xlabel(r"Epoch")
			ax.set_ylabel(r"Log${}_{10}$(Loss)")
			plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
			plt.tight_layout()
			plt.savefig(fig_path)
			plt.close()
	else:
		print("## Empty losses. Not printing... ##")

def createIterativePredictionPlots(model, target, prediction, dt, ic_idx, set_name,  testing_mode="", warm_up=None, error=None, nerror=None, target_augment=None, prediction_augment=None):

	# if error is not None:
	#	 fig_path = model.getFigureDir() + "/{:}_{:}_{:}_error.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
	#	 plt.plot(error, label='error')
	#	 plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
	#	 plt.tight_layout()
	#	 plt.savefig(fig_path)
	#	 plt.close()

	#	 fig_path = model.getFigureDir() + "/{:}_{:}_{:}_log_error.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
	#	 plt.plot(np.log10(np.arange(np.shape(error)[0])), np.log10(error), label='Log${}_{10}$(Loss)')
	#	 plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
	#	 plt.tight_layout()
	#	 plt.savefig(fig_path)
	#	 plt.close()

	# if nerror is not None:
	#	 fig_path = model.getFigureDir() + "/{:}_{:}_{:}_nerror.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
	#	 plt.plot(nerror, label='nerror')
	#	 plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
	#	 plt.tight_layout()
	#	 plt.savefig(fig_path)
	#	 plt.close()

	if len(np.shape(prediction)) == 2 or len(np.shape(prediction)) == 3:
		if ((target_augment is not None) and (prediction_augment is not None)):
			prediction_augment_plot = prediction_augment[:,0] if len(np.shape(prediction_augment)) == 2 else prediction_augment[:,0,0] if len(np.shape(prediction_augment)) == 3 else None
			target_augment_plot = target_augment[:,0] if len(np.shape(target_augment)) == 2 else target_augment[:,0,0] if len(np.shape(target_augment)) == 3 else None

			fig_path = model.getFigureDir() + "/{:}_augmented_{:}_{:}.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
			plt.plot(np.arange(np.shape(prediction_augment_plot)[0]), prediction_augment_plot, 'b', linewidth = 2.0, label='output')
			plt.plot(np.arange(np.shape(target_augment_plot)[0]), target_augment_plot, 'r', linewidth = 2.0, label='target')
			plt.plot(np.ones((100,1))*warm_up, np.linspace(np.min(target_augment_plot), np.max(target_augment_plot), 100), 'g--', linewidth = 2.0, label='warm-up')
			plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
			plt.tight_layout()
			plt.savefig(fig_path)
			plt.close()

			prediction_plot = prediction[:,0] if len(np.shape(prediction)) == 2 else prediction[:,0,0] if len(np.shape(prediction)) == 3 else None
			target_plot = target[:,0] if len(np.shape(target)) == 2 else target[:,0,0] if len(np.shape(target)) == 3 else None

			fig_path = model.getFigureDir() + "/{:}_{:}_{:}.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
			plt.plot(prediction_plot, 'r--', label='prediction')
			plt.plot(target_plot, 'g--', label='target')
			plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
			plt.tight_layout()
			plt.savefig(fig_path)
			plt.close()

		if model.input_dim >=3:
			plotTestingContours(model, target, prediction, dt, ic_idx, set_name, testing_mode=testing_mode)

def plotTestingContourEvolution(model, target, output, dt, ic_idx, set_name, testing_mode=""):
	error = np.abs(target-output)
	vmin = target.min()
	vmax = target.max()
	vmin_error = 0.0
	vmax_error = target.max()
	fig, axes = plt.subplots(nrows=1, ncols=4,figsize=(14, 6), sharey=True)
	fig.subplots_adjust(hspace=0.4, wspace = 0.4)
	axes[0].set_ylabel(r"Time $t$")
	contours_vec = []
	mp = createContour_(fig, axes[0], target, "Target", vmin, vmax, plt.get_cmap("seismic"), dt, xlabel="State")
	contours_vec.append(mp)
	mp = createContour_(fig, axes[1], output, "Output", vmin, vmax, plt.get_cmap("seismic"), dt, xlabel="State")
	contours_vec.append(mp)
	mp = createContour_(fig, axes[2], error, "Error", vmin_error, vmax_error, plt.get_cmap("Reds"), dt, xlabel="State")
	contours_vec.append(mp)
	corr = [pearsonr(target[i],output[i])[0] for i in range(len(target))]
	time_vector = np.arange(target.shape[0])*dt
	axes[3].plot(corr, time_vector)
	axes[3].set_title("Correlation")
	axes[3].set_xlabel(r"Correlation")
	axes[3].set_xlim((-1, 1))
	axes[3].set_ylim((time_vector.min(), time_vector.max()))
	for contours in contours_vec:
		for pathcoll in contours.collections:
			pathcoll.set_rasterized(True)
	fig_path = model.getFigureDir() + "/{:}_{:}_{:}_contour.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
	plt.savefig(fig_path)
	plt.close()

def plotTestingContours(model, target, output, dt, ic_idx, set_name,testing_mode=""):

	print("# plotTestingContours() # - {:}, {:}".format(testing_mode, set_name))

	plotTestingContourEvolution(model, target, output, dt, ic_idx, set_name,  testing_mode=testing_mode)

	N_PLOT_MAX = 1000
	target = target[:N_PLOT_MAX]
	output = output[:N_PLOT_MAX]
	# PLOTTING 10 SIGNALS FOR REFERENCE
	plot_max = np.min([np.shape(target)[1], 10])
	fig_path = model.getFigureDir() + "/{:}_{:}_{:}_signals.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
	for idx in range(plot_max):
		plt.plot(np.arange(np.shape(output)[0]), output[:,idx], color='blue', linewidth = 1.0, label='Output' if idx==0 else None)
		plt.plot(np.arange(np.shape(target)[0]), target[:,idx], color='red', linewidth = 1.0, label='Target' if idx==0 else None)	 
	plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
	plt.tight_layout()
	plt.savefig(fig_path)
	plt.close()

	fig_path = model.getFigureDir() + "/{:}_{:}_{:}_signals_input.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
	for idx in range(plot_max):
		plt.plot(np.arange(np.shape(target)[0]), target[:,idx], linewidth = 1.0, label='Target' if idx==0 else None)
	plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
	plt.tight_layout()
	plt.savefig(fig_path)
	plt.close()

	fig_path = model.getFigureDir() + "/{:}_{:}_{:}_signals_target.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
	for idx in range(plot_max):
		plt.plot(np.arange(np.shape(output)[0]), output[:,idx], linewidth = 1.0, label='Output' if idx==0 else None)
	plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
	plt.tight_layout()
	plt.savefig(fig_path)
	plt.close()


def createDensityContour_(fig, ax, density, bins, title, vmin, vmax, cmap, dt, xlabel="Value"):
	ax.set_title(title)
	t, s = np.meshgrid(np.arange(density.shape[0])*dt, bins)
	mp = ax.contourf(s, t, np.transpose(density), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both")
	fig.colorbar(mp, ax=ax)
	ax.set_xlabel(r"{:}".format(xlabel))
	return mp

def createContour_(fig, ax, data, title, vmin, vmax, cmap, dt, mask_where=None, xlabel=None):
	ax.set_title(title)
	time_vec = np.arange(data.shape[0])*dt
	state_vec = np.arange(data.shape[1])
	if mask_where is not None:
		# print(mask_where)
		mask = [mask_where[i]*np.ones(data.shape[1]) for i in range(np.shape(mask_where)[0])]
		mask = np.array(mask)
		data = np.ma.array(data, mask=mask)

	t, s = np.meshgrid(time_vec, state_vec)
	mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both")
	fig.colorbar(mp, ax=ax)
	ax.set_xlabel(r"{:}".format(xlabel))
	return mp



def createTestingSurfaces(model, target, output, dt, ic_idx, set_name, testing_mode=""):
	error = np.abs(target-output)
	# vmin = np.array([target.min(), output.min()]).min()
	# vmax = np.array([target.max(), output.max()]).max()
	vmin = target.min()
	vmax = target.max()
	vmin_error = 0.0
	vmax_error = target.max()

	print("VMIN: {:} \nVMAX: {:} \n".format(vmin, vmax))
	# Plotting the contour plot
	# fig, axes = plt.subplots(nrows=1, ncols=4,figsize=(12, 6), sharey=True)

	fig = plt.figure(figsize=(12, 6))
	fig.subplots_adjust(hspace=0.4, wspace = 0.4)
	# fig.suptitle('A tale of 2 subplots')

	ax = fig.add_subplot(1, 4, 1, projection='3d')
	createSurface_(fig, ax, target, "Target", vmin, vmax, plt.get_cmap("seismic"), dt)
	ax = fig.add_subplot(1, 4, 2, projection='3d')
	createSurface_(fig, ax, output, "Output", vmin, vmax, plt.get_cmap("seismic"), dt)
	ax = fig.add_subplot(1, 4, 3, projection='3d')
	createSurface_(fig, ax, error, "Error", vmin_error, vmax_error, plt.get_cmap("Reds"), dt)
	ax = fig.add_subplot(1, 4, 4)

	# ax.set_ylabel(r"Time $t$")

	corr = [pearsonr(target[i],output[i])[0] for i in range(len(target))]
	time_vector = np.arange(target.shape[0])*dt
	ax.plot(corr, time_vector)
	ax.set_xlabel(r"Correlation")
	ax.set_xlim((-1, 1))
	ax.set_ylim((time_vector.min(), time_vector.max()))
	fig_path = model.getFigureDir() + "/{:}{:}_{:}_surfaces.{:}".format(testing_mode, set_name, ic_idx, FIGTYPE)
	plt.savefig(fig_path)
	plt.close()

def createSurface_(fig, ax, data, title, vmin, vmax, cmap, dt):
	ax.set_title(title)
	t, s = np.meshgrid(np.arange(data.shape[0])*dt, np.arange(data.shape[1]))
	surf = ax.plot_surface(s, t, np.transpose(data), rstride=1, cstride=1, linewidth=0, antialiased=False)
	# fig.colorbar(mp, ax=ax)
	# ax.set_xlabel(r"$State$")
	# return mp


def plotSpectrum(model, results, set_name, testing_mode=""):
	assert("sp_true" in results)
	assert("sp_pred" in results)
	assert("freq_true" in results)
	assert("freq_pred" in results)
	sp_true = results["sp_true"]
	sp_pred = results["sp_pred"]
	freq_true = results["freq_true"]
	freq_pred = results["freq_pred"]
	fig_path = model.getFigureDir() + "/{:}_frequencies_{:}.{:}".format(testing_mode,set_name, FIGTYPE)
	spatial_dims = len(np.shape(sp_pred))
	if spatial_dims==1:
		# plt.title("Frequency error={:.4f}".format(np.mean(np.abs(sp_true-sp_pred))))
		plt.plot(freq_pred, sp_pred, '--', color="tab:red", label="prediction")
		plt.plot(freq_true, sp_true, '--', color="tab:green", label="target")
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Power Spectrum [dB]')
		# plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
		plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), borderaxespad=0., ncol=2, frameon=False)
	elif spatial_dims==2:
		fig, axes = plt.subplots(figsize=(8, 8), ncols=2)
		plt.suptitle("Frequency error = {:.4f}".format(np.mean(np.abs(sp_true-sp_pred))))
		mp1 = axes[0].imshow(sp_true, cmap=plt.get_cmap("plasma"), aspect=1.0, interpolation='lanczos')
		axes[0].set_title("True Spatial FFT2D")
		mp2 = axes[1].imshow(sp_pred, cmap=plt.get_cmap("plasma"), aspect=1.0, interpolation='lanczos')
		axes[1].set_title("Predicted Spatial FFT2D")
		fig.colorbar(mp1, ax=axes[0])
		fig.colorbar(mp2, ax=axes[1])
	else:
		raise ValueError("Not implemented.")
	plt.tight_layout()
	plt.savefig(fig_path)
	plt.close()



def plotLyapunovSpectrumResults(model):
	data = loadDataPickle(model.getResultsDir() + '/le_results_N{:}.pickle'.format(model.iterative_prediction_length))
	num_lyaps = data["num_lyaps"]
	exponents = data["exponents"]
	output_sequence = data["output_sequence"]
	target_sequence = data["target_sequence"]
	n_warmup = data["n_warmup"]
	target_sequence = data["target_sequence"]
	target_sequence = data["target_sequence"]

	K = np.arange(1,num_lyaps+1)
	SPECTRUM = exponents[::-1]
	marker_str = "x"
	color_str = "green"

	plt.figure()
	plt.plot(np.arange(1,num_lyaps+1), np.zeros(*np.arange(0,num_lyaps).shape), "k--")
	plt.plot(K, SPECTRUM, marker=marker_str, color=color_str, markeredgewidth=2, linestyle='None', markerfacecolor='none', markersize=12)

	plt.xlabel(r"$k$", labelpad=20)
	plt.ylabel(r"$\Lambda_k$", rotation='horizontal', labelpad=20)
	fig_path = model.getFigureDir() + '/Lyapunov_Spectrum.png'
	plt.savefig(fig_path, bbox_inches = 'tight')
	plt.close()



	# draw the result
	plt.figure(figsize=(30,10))
	plt.xlabel('x', fontsize=20)
	plt.ylabel('y', fontsize=20)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.plot(np.arange(np.shape(output_sequence)[0]), output_sequence[:,0], 'r', linewidth = 2.0, label='output')
	plt.plot(np.arange(np.shape(target_sequence)[0]), target_sequence[:,0], 'b', linewidth = 2.0, label='target')
	plt.plot(np.ones((100,1))*n_warmup, np.linspace(np.min(target_sequence[:,0]), np.max(target_sequence[:,0]), 100), 'g--', linewidth = 4.0, label='warm-up')
	plt.legend()
	plt.savefig(model.getFigureDir() + '/Lyapunov_Spectrum_Calc_Prediction.png')
	plt.close()


	error = np.abs(target_sequence-output_sequence)
	vmin = target_sequence.min()
	vmax = target_sequence.max()
	vmin_error = 0.0
	vmax_error = error.max()
	fontsize = 14
	# draw the result   
	fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12, 6), sharey=True)
	fig.subplots_adjust(hspace=0.4, wspace = 0.4)
	axes[0].set_ylabel(r"Time-step", fontsize=fontsize)

	t, s = np.meshgrid(np.arange(target_sequence.shape[0]), np.arange(target_sequence.shape[1]))
	mp = axes[0].contourf(s, t, np.transpose(target_sequence), 15, cmap=plt.get_cmap("seismic"), levels=np.linspace(vmin, vmax, 60), extend="both")
	fig.colorbar(mp, ax=axes[0])

	mp = axes[1].contourf(s, t, np.transpose(output_sequence), 15, cmap=plt.get_cmap("seismic"), levels=np.linspace(vmin, vmax, 60), extend="both")
	fig.colorbar(mp, ax=axes[1])
	mp = axes[2].contourf(s, t, np.transpose(error), 15, cmap=plt.get_cmap("Reds"), levels=np.linspace(vmin_error, vmax_error, 60), extend="both")
	fig.colorbar(mp, ax=axes[2])
	axes[0].set_title("Target", fontsize=fontsize)
	axes[1].set_title("Prediction", fontsize=fontsize)
	axes[2].set_title("Error", fontsize=fontsize)
	plt.savefig(model.getFigureDir() + '/Lyapunov_Spectrum_Calc_Contour.png')
	plt.close()




