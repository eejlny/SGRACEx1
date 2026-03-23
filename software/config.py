import numpy as np

#device = "cuda:1"
device = "cpu"
hidden_channels = 16
write_file = 0 #write file for debugging
instant_layer_count = 1 #how many layers to process in each SGRACE (core) hardware call.
total_layer_count = 3 #how many total SGRACE layers.

core_count = 1 #how mny cores in dataflow mode avaialble in the hardware configuration
load_weights = 1 #load weights into hardware (needed for training, inference can just load once and then reuse)

accb = 0 #use accelerator in backward path
acc = 1 #use accelerator in forward path
show_max_min = 0
show_sparsity = 0
min_output = 1
profiling = 0
fake_quantization = 1 #for software only execution simulates effects of quantization
hardware_quantize = 1 #should be always one
stream_mode = 0 #read from memory input and write to memory output (normal with layer_count=1)
head_count = 1 #not in use


#global profiling
N_adj = 20480 # max number of nodes
M_adj = 20480 # max number of nodes
M_fea = 4096 #max number of input features
P_w =  hidden_channels #hid number hidden chnnels
NNZ_adj = 1000000 # max number of non-zero values of adjacency
NNZ_fea = 4000000 # max number of non-zero values of feature
w_qbits = 8#1
w_qbitsl = 8#flaot


float_type = np.float32
model_type = np.uint8 


