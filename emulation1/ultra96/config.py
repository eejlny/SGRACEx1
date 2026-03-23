import numpy as np


hidden_channels = 16
layer_count = 1 #how many layers to process in one hardware call.
load_weights = 1 #load weights into hardware (needed for training, inference can just load once and then reuse)

accb = 0 #use accelerator in backward path
acc = 1 #use accelerator in forward path
show_max_min = 0
min_output = 1
profiling = 0
fake_quantization = 1 #for software only execution simulates effects of quantization
hardware_quantize = 1 #should be always one
compute_attention = 0 #compute GAT at 1 or GCN at 0
stream_mode = 0 #read from memory input and write to memory output (normal with layer_count=1)
head_count = 1 #not in use


#global profiling
N_adj = 20480 # max number of nodes
M_adj = 20480 # max number of nodes
M_fea = 2048 #max number of input features
P_w =  hidden_channels #hid number hidden chnnels
NNZ_adj = 1000000 # max number of non-zero values of adjacency
NNZ_fea = 4000000 # max number of non-zero values of feature
w_qbits = 8

#global hard_type
#hard_type = np.int8
#global out_type
#out_type = np.int32
#global float_type
float_type = np.float32

#global layern

#global rowPtr_adj_buffer
#rowPtr_adj_buffer = []
#global values_adj_buffer
#values_adj_buffer = []
#global rowPtr_fea_buffer
#rowPtr_fea_buffer = []
#global columnIndex_fea_buffer
#columnIndex_fea_buffer = []
#global values_fea_buffer
#values_fea_buffer = []

