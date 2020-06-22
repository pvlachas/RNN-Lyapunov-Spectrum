#!/bin/bash


cd ../../Methods

mode=lyapunov

python3 RUN.py rnn \
--mode $mode \
--system_name Lorenz3D \
--write_to_log 0 \
--N 100000 \
--N_used 20000 \
--input_dim 3 \
--skip 1 \
--train_val_ratio 0.75 \
--rnn_cell_type gru \
--rnn_layers_num 1  \
--rnn_layers_size 40  \
--sequence_length 30 \
--hidden_state_propagation_length 1000 \
--scaler Standard \
--noise_level 1 \
--learning_rate 0.001 \
--batch_size 32 \
--overfitting_patience 5 \
--max_epochs 200 \
--max_rounds 10 \
--random_seed 7 \
--display_output 1 \
--retrain 0 \
--num_test_ICS 1 \
--iterative_prediction_length 20000 \
--teacher_forcing_forecasting 1 \
--iterative_state_forecasting 1 \
--num_lyaps 3


