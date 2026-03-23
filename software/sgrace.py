import torch
import torch_geometric.transforms as T
import pandas as pd
import sys
#from torch_geometric.nn import GATConv, GATv2Conv


import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.utils import add_remaining_self_loops,add_self_loops,sort_edge_index,degree
from torch_scatter import scatter_add


import config
if (config.acc==1):
 from pynq import allocate
 from pynq import Overlay
 from pynq import get_rails, DataRecorder

def isSparse(array, m, n): 
 counter = 0
 # Count number of zeros
 # in the matrix
 for i in range(0, m):
    for j in range(0, n):
       if (array[i][j] == 0):
           counter = counter + 1
 print("total values ",m*n)
 print("zero values ",counter)
 return (counter > ((m * n) // 2))

def sym_norm2(edge_index, num_nodes, edge_weight=None, fill=0, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

    
    # Calculate node degrees
    node_degrees = degree(edge_index[1], num_nodes=num_nodes)

    #print('max_degree')
    #print(torch.max(node_degrees))

    
    #fill_value = math.trunc(math.log2(average_node_degree)) if not improved else 2
    
    #print("fill value")
    #print(fill_value)
    #fill_value = torch.max(node_degrees) if not improved else 2
    #fill_value = 0 if not improved else 2 #32

    
    #edge_weight = torch.zeros((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
    
    #print("edge index")
    #print(edge_index)
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill, num_nodes)
    
    edge_index, edge_weight = sort_edge_index(edge_index, edge_weight) #make sure that self loops are in order
    
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def quantization(x, s, z, alpha_q, beta_q):

    x_q = np.round(1 / s * x + z, decimals=0)
    x_q = np.clip(x_q, a_min=alpha_q, a_max=beta_q)
    
  
    return x_q


def quantization_b(x, s, z, alpha_q, beta_q):

    x_q = (1 / s * x + z)
    x_q[x_q < 0] = -1
    x_q[x_q >= 0] = 1
    return x_q


def quantization_uqbits(x, s, z, qbits):

    alpha_q = 0
    beta_q = (2**(qbits) - 1)
    x_q = quantization(x, s, z, alpha_q, beta_q)
    #x_q = x_q.astype(np.int8)

    #print(x_q.shape)
    return x_q

def quantization_qbits(x, s, z, qbits):

    if (qbits==1):
     alpha_q = -1
     beta_q = 1
     x_q = quantization_b(x, s, z, alpha_q, beta_q)
    else:
     alpha_q = (-2**(qbits - 1) + 1)
     beta_q = (2**(qbits - 1) - 1)
     x_q = quantization(x, s, z, alpha_q, beta_q)

    #print(x_q.shape)
    return x_q


def generate_quantization_constants(alpha, beta, alpha_q, beta_q, qbits):

    # Affine quantization mapping
    #this beta_o and alpha_o take into account that during training the integer values are inserted in a fractional pipeline
    #This pipeline has 7 bit integer and 25 bit fractional (total 32). If the values are inserted with an alignment of 18 and have 8 bits width
    # then they become x.xxxxxxx like if they are divided by (2**(8-1) and this effect must be removed during dequant.
    

    #beta_o = beta_q/(2**(frac_bits-f_align))
    #alpha_o = alpha_q/(2**(frac_bits-f_align))

    #beta_o = beta_q/(2**(config.w_qbits-f_align-1)) #ojo perhaps the one on top....
    #alpha_o = alpha_q/(2**(config.w_qbits-f_align-1))


    #if(config.w_qbits == 1):
    if(qbits == 1):
     beta_o = beta_q/(2**2) #ojo perhaps the one on top....
     alpha_o = alpha_q/(2**2)
    else: 
     beta_o = beta_q/(2**(qbits)) #ojo perhaps the one on top....
     alpha_o = alpha_q/(2**(qbits))

    #beta_o = beta_q/(2**(config.w_qbits-1)) #ojo perhaps the one on top....
    #alpha_o = alpha_q/(2**(config.w_qbits-1))
   
 
    s_o = (beta - alpha) / (beta_o - alpha_o)

    #print('quantization Scale output',s_o)
    
    s = (beta - alpha) / (beta_q - alpha_q)
    
    
    z = int((beta * alpha_q - alpha * beta_q) / (beta - alpha))
    #print('quantization Scale ',s)
    #print('Zero point ',z)

    return s_o,s, z


def generate_quantization_uqbits_constants(alpha, beta,qbits):

    alpha_q = 0
    beta_q = (2**(qbits) - 1)
    
    #print(alpha_q)
    #print(beta_q)

    s_o,s, z = generate_quantization_constants(alpha=alpha,
                                           beta=beta,
                                           alpha_q=alpha_q,
                                           beta_q=beta_q,
                                           qbits=qbits)
    

    return s_o,s, z


def generate_quantization_qbits_constants(alpha, beta,qbits):

    if(qbits==1): 
     alpha_q = -1
     beta_q = 1
    else:
     alpha_q = ((-2**(qbits - 1) + 1))
     beta_q = (2**(qbits - 1) - 1) 
        
    #print(alpha_q)
    #print(beta_q)
    

    s_o,s, z = generate_quantization_constants(alpha=alpha,
                                           beta=beta,
                                           alpha_q=alpha_q,
                                           beta_q=beta_q,
                                           qbits=qbits)
   
    
    #print(s)
    #print(z)

    return s_o,s, z


def fake_quantization_b(x, s, z, alpha_q, beta_q):

    x_q = (1 / s * x + z)
    x_q[x_q < 0] = -0.5
    x_q[x_q >= 0] = 0.5
    return x_q

def fake_quantization_b2(x, s, z, alpha_q, beta_q):

    x_r = torch.round(1 / s * x + z, decimals=0)
    x_q = torch.clip(x_r, min=alpha_q, max=beta_q)
    x_q = x_q/2 #good for 1 bit
    return x_q

def fake_quantization(x, s, z, alpha_q, beta_q):

    #print(x)
    x_r = torch.round(1 / s * x + z, decimals=0)
    x_q = torch.clip(x_r, min=alpha_q, max=beta_q)
    #x_q = x_r
    #print("s")
    #print(s)
    #print("x_r")
    #print(x_r)
    #print("alpha_q")
    #print(alpha_q)
    #print("beta_q")
    #print(beta_q)
    #print("x_q")
    #print(x_q)

    #emulate hardware effects
    #x_q = x_q << f_align
    #x_q = x_q / (pow(2,(config.w_qbits-1)))
     
    #print(x_q) 

    #x_q = x_q/(2**(frac_bits-f_align)) #back to float

    #scale = config.w_qbits-f_align-1

    #print(scale)

    x_q = x_q/(2**(config.w_qbits-1)) #back to float
    #x_q = x_q/(2**(config.w_qbits)) #good for 1 bit

    #print("x_q")
    #print(x_q)

    #print(config.w_qbits)
    #print(f_align)

    #x_q = x_q/(config.w_qbits-f_align-1) #back to float

    #x_q = x_q*s #back to float

    #print(x_q) 
    
    return x_q


def quantization_fbits(x, s, z, qbits):

    if (qbits==1):
     alpha_q = -1
     beta_q = 1
     x_q = fake_quantization_b(x, s, z, alpha_q, beta_q)
    else:
     alpha_q = (-2**(qbits - 1) + 1)
     #alpha_q = (-2**(qbits - 1)) #use the lowest possible negative value as well
     beta_q = (2**(qbits - 1) - 1)
     x_q = fake_quantization(x, s, z, alpha_q, beta_q)

    #print(x_q.shape)
    return x_q

def quantization_ufbits(x, s, z, qbits):

    alpha_q = 0
    beta_q = (2**(qbits) - 1)
    if (qbits==1):
     x_q = fake_quantization_b2(x, s, z, alpha_q, beta_q)
    else:
     x_q = fake_quantization(x, s, z, alpha_q, beta_q)
    #x_q = x_q.astype(np.int8)
    #print("ufbits")
    #print(x_q)
    #print(x_q.shape)
    return x_q


def float_to_fix(f_in,n_bits):
    f = (1 << n_bits)
    i_out = np.round(f_in*f)*(1.0/f) 
    return i_out  

class RPYNQ(torch.autograd.Function):
    """Both forward and backward are static methods."""

    @staticmethod
    def forward(ctx,input,srelu):
    #def forward(ctx,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.srelu = srelu
        ctx.save_for_backward(input)
        output = input.clone() #this clone is important to make it work
        return output

    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the inputs: here input and weights
        """
        srelu = ctx.srelu
        input, = ctx.saved_tensors

        grad_srelu = None
 
        grad_input = grad_output.clone() #this clone is iportatant
        grad_input[input == 0] = 0 #hardware style that takes into account the hardware integrated relu 
     
        #grad_input[input <= srelu] = 0 #hardware style that takes into account the hardware integrated relu 
     
        grad_srelu =  -grad_output.clone()

        grad_srelu[input == 0] = 0

        #print("grad srelu")

        #print(grad_srelu)

        return grad_input,grad_srelu
        #return grad_input
   


class FPYNQ_GAT(torch.autograd.Function):
    """Both forward and backward are static methods."""



    @staticmethod
    def forward(ctx,my_ip, self,adj,nnz_adj,input, weights,weights_linear,weights_l2,weights_l3,attention,out_features,dropout,relu):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        def prepare_attentional_mechanism_input(Wh,attention,out_features):
          Wh1 = torch.matmul(Wh, attention[:out_features, :])
          Wh2 = torch.matmul(Wh, attention[out_features:, :])
          # broadcast add
          e = Wh1 + Wh2.T
          return e
    
     
        if (config.profiling == 1):
          fmult = time.time()

    
        if (config.acc==1):
            
         #print("acc ON") 
         if (config.profiling == 1):
          rmult = time.time()

         global layern
         #print("deq_o and active layer")
         #print(deq_o)
         #print(layern)



         #if (layern == 1):
          #my_ip.register_map.scale_fea = scale_fea #2 #scale fea

          #int32bits = np.asarray(deq_o, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.deq_factor = int32bits
          #deq_factor = int32bits

          #qsf = 1/f_s
          #int32bits = np.asarray(qsf, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_fea = int32bits
        
        
          #qsw = 1/w_s
          #int32bits = np.asarray(qsw, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_w = int32bits

   

         #elif (layern == 2):
          #my_ip.register_map.scale_fea = scale_fea2 #2 #scale fea
          #int32bits = np.asarray(deq_o2, dtype=np.float32).view(np.int32).item() 
          #deq_factor = int32bits
          #my_ip.register_map.deq_factor = int32bits
          #qsf = 1/f_s2
          #int32bits = np.asarray(qsf, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_fea = int32bits
          #qsw = 1/w_s2
          #int32bits = np.asarray(qsw, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_w = int32bits   
    
         #else:
          #int32bits = np.asarray(deq_ol, dtype=np.float32).view(np.int32).item() 
          #deq_factor = int32bits
          #my_ip.register_map.deq_factor = int32bits
          #qsf = 1/f_sl
          #int32bits = np.asarray(qsf, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_fea = int32bits
          #qsw = 1/w_sl
          #int32bits = np.asarray(qsw, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_w = int32bits 

         #print("deq_factor")
         #print(deq_factor) 
         #print("qsf")
         #print(qsf) 
         #print("qsw")
         #print(qsw)  
            
         qsa = 1/a_s

         #print("quantization_scale_adj")
         #print(qsa)
         #print("a_s")
         #print(a_s)
         int32bits = np.asarray(qsa, dtype=np.float32).view(np.int32).item() 
         my_ip.register_map.quantization_scale_adj = int32bits

        
         
         if(config.instant_layer_count==1):

          my_ip.register_map.rowPtr_adj1_offset_1 = rowPtr_adj_buffer.physical_address 
          my_ip.register_map.rowPtr_adj2_offset_1 = rowPtr_adj_buffer.physical_address 
          my_ip.register_map.rowPtr_adj3_offset_1 = rowPtr_adj_buffer.physical_address
          my_ip.register_map.rowPtr_adj4_offset_1 = rowPtr_adj_buffer.physical_address

          my_ip.register_map.columnIndex_adj1_offset_1 = columnIndex_adj_buffer.physical_address 
          my_ip.register_map.columnIndex_adj2_offset_1 = columnIndex_adj_buffer.physical_address
          my_ip.register_map.columnIndex_adj3_offset_1 = columnIndex_adj_buffer.physical_address 
          my_ip.register_map.columnIndex_adj4_offset_1 = columnIndex_adj_buffer.physical_address 
          my_ip.register_map.values_adj1_offset_1 = values_adj_buffer.physical_address
          my_ip.register_map.values_adj2_offset_1 = values_adj_buffer.physical_address
          my_ip.register_map.values_adj3_offset_1 = values_adj_buffer.physical_address
          my_ip.register_map.values_adj4_offset_1 = values_adj_buffer.physical_address


          my_ip.register_map.N_adj=input.shape[0]
          my_ip.register_map.M_adj=input.shape[0]
          my_ip.register_map.M_fea=input.shape[1]
          my_ip.register_map.P_w=weights.shape[1]

          #print('use my_ip.register_map.N_adj')
          #print("P_w")
          #print(weights.shape[1])

        
        
       
          my_ip.register_map.E1_offset_1 = E_buffer.physical_address
          my_ip.register_map.S1_offset_1 = S_buffer.physical_address
   
          my_ip.register_map.D1_offset_1 = D_buffer.physical_address
          my_ip.register_map.D2_offset_1 = D_buffer.physical_address
          my_ip.register_map.D3_offset_1 = D_buffer.physical_address
          my_ip.register_map.D4_offset_1 = D_buffer.physical_address

          #print('use rowPtr_fea_buffer.physical_address')
          #print(rowPtr_fea_buffer.physical_address)
  
          my_ip.register_map.rowPtr_fea1_offset_1 = rowPtr_fea_buffer.physical_address
          my_ip.register_map.rowPtr_fea2_offset_1 = rowPtr_fea_buffer.physical_address
          my_ip.register_map.rowPtr_fea3_offset_1 = rowPtr_fea_buffer.physical_address
          my_ip.register_map.rowPtr_fea4_offset_1 = rowPtr_fea_buffer.physical_address

          my_ip.register_map.columnIndex_fea1_offset_1 =columnIndex_fea_buffer.physical_address
          my_ip.register_map.columnIndex_fea2_offset_1 =columnIndex_fea_buffer.physical_address 
          my_ip.register_map.columnIndex_fea3_offset_1 =columnIndex_fea_buffer.physical_address 
          my_ip.register_map.columnIndex_fea4_offset_1 =columnIndex_fea_buffer.physical_address 
          my_ip.register_map.values_fea1_offset_1 = values_fea_buffer.physical_address
          my_ip.register_map.values_fea2_offset_1 = values_fea_buffer.physical_address
          my_ip.register_map.values_fea3_offset_1 = values_fea_buffer.physical_address
          my_ip.register_map.values_fea4_offset_1 = values_fea_buffer.physical_address
         else:
          #print('layern')
          #print(layern)
          if(layern==1):
           my_ip.register_map.rowPtr_adj1_offset_1 = rowPtr_adj_buffer.physical_address 
           my_ip.register_map.rowPtr_adj2_offset_1 = rowPtr_adj_buffer.physical_address 
           my_ip.register_map.rowPtr_adj3_offset_1 = rowPtr_adj_buffer.physical_address
           my_ip.register_map.rowPtr_adj4_offset_1 = rowPtr_adj_buffer.physical_address

           my_ip.register_map.columnIndex_adj1_offset_1 = columnIndex_adj_buffer.physical_address 
           my_ip.register_map.columnIndex_adj2_offset_1 = columnIndex_adj_buffer.physical_address
           my_ip.register_map.columnIndex_adj3_offset_1 = columnIndex_adj_buffer.physical_address 
           my_ip.register_map.columnIndex_adj4_offset_1 = columnIndex_adj_buffer.physical_address 
           my_ip.register_map.values_adj1_offset_1 = values_adj_buffer.physical_address
           my_ip.register_map.values_adj2_offset_1 = values_adj_buffer.physical_address
           my_ip.register_map.values_adj3_offset_1 = values_adj_buffer.physical_address
           my_ip.register_map.values_adj4_offset_1 = values_adj_buffer.physical_address


           my_ip.register_map.N_adj=input.shape[0]
           my_ip.register_map.M_adj=input.shape[0]
           my_ip.register_map.M_fea=input.shape[1]
           #my_ip.register_map.P_w=weights.shape[1]

           #print('input shape 1')
           #print(input.shape[1])
           #print(weights.shape[1])

        
        
       
           my_ip.register_map.E1_offset_1 = E_buffer.physical_address
           my_ip.register_map.S1_offset_1 = S_buffer.physical_address
   
           my_ip.register_map.D1_offset_1 = D_buffer.physical_address
           my_ip.register_map.D2_offset_1 = D_buffer.physical_address
           my_ip.register_map.D3_offset_1 = D_buffer.physical_address
           my_ip.register_map.D4_offset_1 = D_buffer.physical_address

           #print('use rowPtr_fea_buffer.physical_address')
           #print(rowPtr_fea_buffer.physical_address)
  
           my_ip.register_map.rowPtr_fea1_offset_1 = rowPtr_fea_buffer.physical_address
           my_ip.register_map.rowPtr_fea2_offset_1 = rowPtr_fea_buffer.physical_address
           my_ip.register_map.rowPtr_fea3_offset_1 = rowPtr_fea_buffer.physical_address
           my_ip.register_map.rowPtr_fea4_offset_1 = rowPtr_fea_buffer.physical_address

           my_ip.register_map.columnIndex_fea1_offset_1 =columnIndex_fea_buffer.physical_address
           my_ip.register_map.columnIndex_fea2_offset_1 =columnIndex_fea_buffer.physical_address 
           my_ip.register_map.columnIndex_fea3_offset_1 =columnIndex_fea_buffer.physical_address 
           my_ip.register_map.columnIndex_fea4_offset_1 =columnIndex_fea_buffer.physical_address 
           my_ip.register_map.values_fea1_offset_1 = values_fea_buffer.physical_address
           my_ip.register_map.values_fea2_offset_1 = values_fea_buffer.physical_address
           my_ip.register_map.values_fea3_offset_1 = values_fea_buffer.physical_address
           my_ip.register_map.values_fea4_offset_1 = values_fea_buffer.physical_address


           if (config.profiling == 1):   
            print('Register time: {:.5f}ms'.format(1000/1*(time.time() - rmult)))
        
           my_ip.register_map.B_offset_1 = B_buffer.physical_address
           my_ip.register_map.B2_offset_1 = B2_buffer.physical_address
         
         if (config.profiling == 1):
          amult = time.time()
         
         support_linear = torch.transpose(weights_linear,0,1)
         support = torch.transpose(weights,0,1)
         support_l2 = torch.transpose(weights_l2,0,1)
         support_l3 = torch.transpose(weights_l3,0,1)
         #support_l4 = torch.transpose(weights_l4,0,1)
         #print(weights_l2.shape)
         #B_buffer[0:(weights.shape[0]*weights.shape[1])] = torch.transpose(weights,0,1).reshape(1, (weights.shape[0]*weights.shape[1]))
         if (config.profiling == 1):   
          print('Transpose time: {:.5f}ms'.format(1000/1*(time.time() - amult)))
  
        
         if(config.min_output == 0):
          print('values_fea_buffer')
          print(values_fea_buffer[0:100])
          print('columnIndex_fea_buffer')
          print(columnIndex_fea_buffer[0:100])
          print('rowPtr_fea_buffer')
          print(rowPtr_fea_buffer[0:100])
         
         attention_q=attention.reshape(1,(attention.shape[0]*attention.shape[1]))
         support_pynq = support.data.numpy() #OJO USE TRANSPOSE
         support_pynq_linear = support_linear.data.numpy()
         support_pynq_l2 = support_l2.data.numpy() #OJO USE TRANSPOSE
         support_pynq_l3 = support_l3.data.numpy() #OJO USE TRANSPOSE
         #support_pynq_l4 = support_l4.data.numpy() #OJO USE TRANSPOSE
       
  
    
          
         if(config.show_max_min==1):
          print("active layer: ",layern)   
          print("max/min graph weights")
          print(np.max(support_pynq))
          print(np.min(support_pynq))
          print("max/min linear weights")
          print(np.max(support_pynq_linear))
          print(np.min(support_pynq_linear))
         support_pynq_q = support_pynq
         support_pynq_linear_q = support_pynq_linear
         support_pynq_q_l2 = support_pynq_l2
         support_pynq_q_l3 = support_pynq_l3
         #support_pynq_q_l4 = support_pynq_l4

         write_weights = 0
         if(write_weights == 1):   
           #support_pynq_q_t = np.transpose(support_pynq_q)
           df = pd.DataFrame(support_pynq_q) #convert to a dataframe
           df.to_csv("./weights.txt",index=False,header=False) #save to file    
           #sys.exit()
            
         support_pynq_q = support_pynq_q.reshape(1, (weights.shape[0]*weights.shape[1]))

         #print("weights linear shape")
         #print(weights_linear.shape)

         #if(support_pynq_linear_q.shape[0]!=16):
         # zero_mat = np.ones(((16-support_pynq_linear_q.shape[0]),support_pynq_linear_q.shape[1]),dtype=np.float32)
         # #print("support_pynq_linear_q 1")
         # #print(support_pynq_linear_q.shape)
         # #print(zero_mat.shape)
         # support_pynq_linear_q = np.concatenate((support_pynq_linear_q, zero_mat))
         # #support_pynq_linear_q = np.concatenate((zero_mat,support_pynq_linear_q))
         # #print("support_pynq_linear_q")
         # #print(support_pynq_linear_q)
         # support_pynq_linear_q = support_pynq_linear_q.reshape(1, (support_pynq_linear_q.shape[0]*support_pynq_linear_q.shape[1]))
         #else:



         if(config.write_file == 1 and layern == 3):   
              support_pynq_linear_q_t = np.transpose(support_pynq_linear_q)
              df = pd.DataFrame(support_pynq_linear_q_t) #convert to a dataframe
              df.to_csv("./linear_weights.txt",index=False,header=False) #save to file    
              sys.exit()

         support_pynq_linear_q = support_pynq_linear_q.reshape(1, (support_pynq_linear_q.shape[0]*support_pynq_linear_q.shape[1]))


         #print("weights l2")
         #print(support_pynq_q_l2)
    
         if(config.core_count == 1):
           if(config.instant_layer_count == 1):
             B_buffer[0:(weights.shape[0]*weights.shape[1])] = support_pynq_q.astype(config.float_type)
             B2_buffer[0:(support_pynq_linear_q.shape[0]*support_pynq_linear_q.shape[1])] = support_pynq_linear_q.astype(config.float_type)
              #print("B2_buffer")
              #print(B2_buffer[0:10])
           else:
            if(layern==1):
             B_buffer[0:(weights.shape[0]*weights.shape[1])] = support_pynq_q.astype(config.float_type)
             global weight_shift
             support_pynq_q_l2 = support_pynq_q_l2.reshape(1, (weights_l2.shape[0]*weights_l2.shape[1]))
             weight_shift = (weights.shape[0]*weights.shape[1])
             B_buffer[weight_shift:(weight_shift+weights_l2.shape[0]*weights_l2.shape[1])] = support_pynq_q_l2.astype(config.float_type)
             support_pynq_q_l3 = support_pynq_q_l3.reshape(1, (weights_l3.shape[0]*weights_l3.shape[1]))
             weight_shift = weight_shift+(weights_l2.shape[0]*weights_l2.shape[1])
             #B_buffer[weight_shift:(weight_shift+weights_l3.shape[0]*weights_l3.shape[1])] = support_pynq_q_l3.astype(config.float_type)
             #print("pynq l3")
             B2_buffer[weight_shift:(weight_shift+weights_l3.shape[0]*weights_l3.shape[1])] = support_pynq_q_l3.astype(config.float_type)
             #support_pynq_q_l4 = support_pynq_q_l4.reshape(1, (weights_l4.shape[0]*weights_l4.shape[1]))
             #weight_shift = weight_shift+(weights_l3.shape[0]*weights_l3.shape[1])
             #print("pynq l4")
             #print(support_pynq_q_l4[0:10])
             #B2_buffer[weight_shift:(weight_shift+weights_l4.shape[0]*weights_l4.shape[1])] = support_pynq_q_l4.astype(config.float_type)
             #B2_buffer[0:(weights_l3.shape[0]*weights_l3.shape[1])] = support_pynq_q_l3.astype(config.float_type)

         if (config.min_output == 0):
         #if (self.linear == 1):
           print("B_Buffer")
           print(B_buffer[0:10])
         #print("B2_Buffer linear")
         #print(B2_buffer[0:32])

         if(self.compute_attention == 1):
          attention_q = attention_q.numpy()
          attention_buffer[0:(attention.shape[0]*attention.shape[1])] = attention_q.astype(config.float_type)

         #if(config.show_max_min==1):
         # print("max/min quantized weights")
         # print(np.max(support_pynq_q))
         # print(np.min(support_pynq_q))
        
         #global B_size
         #B_size = (weights.shape[0]*weights.shape[1])
          
         my_ip.register_map.quantized_multiplier = internal_quantization #apply internal quantization
         
         #print("layern")
         #print(layern)

         if (config.profiling == 2):
          amult = time.time()
          for _ in range(1):
           my_ip.register_map.CTRL.AP_START=1
           kernel_done = my_ip.register_map.CTRL.AP_DONE
           while kernel_done == 0:
            kernel_done = my_ip.register_map.CTRL.AP_DONE
          dmult =  time.time()
         else:
          #print('config.instant_layer_count ',config.instant_layer_count)
          if(config.core_count == 1):
           if(config.instant_layer_count == 1):
             #if (config.min_output == 0):
             #print('start core 1 to process 1 layer')
             #print('model buffer is')
             #print(model_buffer[0])
             #print(model_buffer[1])
             amult = time.time()
             next_inst_addr = model_buffer.physical_address+(layern-1) 
             next_P_w_addr = P_w_buffer.physical_address+(layern-1) 
             next_quantization_scale_fea_addr = quantization_scale_fea_buffer.physical_address+4*(layern-1) 
             next_quantization_scale_w_addr = quantization_scale_w_buffer.physical_address+4*(layern-1)
             next_quantization_scale_l_addr = quantization_scale_l_buffer.physical_address+4*(layern-1)
             next_scale_fea_addr = scale_fea_buffer.physical_address+(layern-1)
             next_deq_factor_addr = deq_factor_buffer.physical_address+4*(layern-1)


             #print("instruction address")
             #print(next_inst_addr) 
             my_ip.register_map.P_w_offset_1 = next_P_w_addr  #point to next instruction
             my_ip.register_map.model_offset_1 = next_inst_addr  #point to next instruction
             my_ip.register_map.scale_fea_offset_1 = next_scale_fea_addr
             my_ip.register_map.quantization_scale_fea_offset_1 = next_quantization_scale_fea_addr
             my_ip.register_map.quantization_scale_w_offset_1 = next_quantization_scale_w_addr
             my_ip.register_map.deq_factor_offset_1 = next_deq_factor_addr

             #print("next_deq_factor_addr")
             #print(next_deq_factor_addr)

             my_ip.register_map.CTRL.AP_START=1
             kernel_done = my_ip.register_map.CTRL.AP_DONE
             while kernel_done == 0:
               kernel_done = my_ip.register_map.CTRL.AP_DONE  
             dmult =  time.time()
             if (config.profiling == 1):   
              print('Accelerator forward kernel 1-layer time: {:.5f}ms'.format(1000/1*(dmult - amult)))
             if (config.min_output == 0):
              print('done core 1')  
           else:  
            if(layern == 1):
             if (config.min_output == 0):
              print('start core 1 to process layer count: ', config.instant_layer_count)
             amult =  time.time()
             my_ip.register_map.CTRL.AP_START=1
             kernel_done = my_ip.register_map.CTRL.AP_DONE
             while kernel_done == 0:
               kernel_done = my_ip.register_map.CTRL.AP_DONE  
             dmult =  time.time()
             if (config.profiling == 1):   
              print('Accelerator forward kernel n-layer time: {:.5f}ms'.format(1000/1*(dmult - amult)))
             if (config.min_output == 0):
              print('done core 1') 
            else:  
             if (config.min_output == 0): 
              print('Nothing to do in layer>1')

      

         if(config.instant_layer_count == 1): 
          #print("active layer: ", layern)  

     
          
          if(self.compute_attention==1):
           output_e_val = E_buffer[0:nnz_adj].astype(config.float_type)
           output_s_val = S_buffer[0:nnz_adj].astype(config.float_type) #you should use this
        


          max_fea = my_ip.register_map.max_fea
          if(config.min_output == 0):
           print("MAX FEA INT GAT")
           print(max_fea)
           print(float(max_fea)/(2**frac_bits_o))

          max_fea_float = float(max_fea)/(2**frac_bits_o)
          if (layern == 1):
           global cur_max_fea
           if(max_fea_float > cur_max_fea):
            cur_max_fea = max_fea_float
          else:
           global cur_max_fea2
           if(max_fea_float > cur_max_fea2):
            cur_max_fea2 = max_fea_float
        
          #if(weights.shape[1]!=16):
          # output_acc = D_buffer[0:input.shape[0]*16]
          # output_acc = output_acc.reshape(input.shape[0],16)
          # output_acc = output_acc[0:input.shape[0],0:weights.shape[1]] 
          # #output_acc = output_acc[0:input.shape[0],weights.shape[1]:16] 
          # print("Output shape")
          # print(output_acc.shape)
          
          #print("Output_acc")
          #print("shapes")
          #print(input.shape[0])
          #print(weights.shape[1])

          ## print(output_acc)
          #else:
          output_acc = D_buffer[0:input.shape[0]*weights.shape[1]]
          output_acc = output_acc.reshape(input.shape[0],weights.shape[1]) 


          write_output = 0
          if(write_output == 1):   
           #support_pynq_q_t = np.transpose(support_pynq_q)
           df = pd.DataFrame(output_acc) #convert to a dataframe
           df.to_csv("./output.txt",index=False,header=False) #save to file    
           sys.exit()
          if(config.show_sparsity==1):
           zeros = np.count_nonzero(output_acc==0)
           print("sparsity from layer:",layern,"zeros:",zeros/(input.shape[0]*weights.shape[1]))  
          #print("Sample Output")
          #print(output_acc[0])
    
          if(layern<config.total_layer_count):
           layern+=1
          else:
           layern=1 

          #get sparse matrix for e and softmax
          if(self.compute_attention==1):
           rindex = rowPtr_adj_buffer[0:nnz_adj]
           cindex =  columnIndex_adj_buffer[0:nnz_adj] 
           output_s = coo_matrix((output_s_val,(rindex ,cindex)), shape=(input.shape[0], input.shape[0]))
           output_s = output_s.todense()
           output_e = coo_matrix((output_e_val,(rindex, cindex)), shape=(input.shape[0], input.shape[0]))
           output_e = output_e.todense()

           output_s = torch.from_numpy(output_s) 
           output_e = torch.from_numpy(output_e) 
           output_s = output_s.float()
           #print("output_s")
           #print(output_s)
           output_e = output_e.float()
         
          if (config.profiling == 1): 
            bmult = time.time()
          output_acc = torch.from_numpy(output_acc)       
          output_acc = output_acc.float()

          #print("output_acc")
          #print(output_acc)
          if (config.profiling == 1):   
           print('output_acc time: {:.5f}ms'.format(1000/1*(time.time() - bmult)))
         
          ctx.nheads = self.nheads
          ctx.alpha = self.alpha

          ctx.linear = self.linear
          ctx.compute_attention = self.compute_attention
          ctx.sage = self.sage

          #if(self.linear==1):
          # print("output is linear")
          # print(output_acc[0:10])

          if(self.compute_attention == 1):
           ctx.save_for_backward(adj,input, weights,weights_linear,output_e,output_s,output_acc)
          else:
           ctx.save_for_backward(adj,input, weights,weights_linear,adj,adj,output_acc) 
        
          if (config.profiling == 1):   
           print('Forward function time: {:.5f}ms'.format(1000/1*(time.time() - fmult)))
          return output_acc
         else: #n layer processing 
          #print("input shape")
          #print(input.shape) 
          #print("weight shape")
          #print(weights.shape)  
          if(layern==1): #if layer 1 we return the input so layer 2 gets the right dimensions.

           output_acc = D_buffer[0:input.shape[0]*weights.shape[1]]
           #zeros = np.count_nonzero(output_acc==0)
           #print("sparsity into layer 2 zeros ", zeros/(input.shape[0]*weights.shape[1]))
           #print("output_acc")
           #print(output_acc)
          elif(layern==2):
           output_acc = D_buffer[0:input.shape[0]*weights.shape[1]]
           #output_acc = D_buffer[input.shape[0]*weights.shape[1]:2*input.shape[0]*weights.shape[1]]
           #zeros = np.count_nonzero(output_acc==0)
           #print("sparsity into layer 3 zeros ", zeros/(input.shape[0]*weights.shape[1]))
           #print(output_acc)
          #elif(layern==3):
          # output_acc = D_buffer[2*input.shape[0]*weights.shape[2]:3*input.shape[0]*weights.shape[2]]
          # zeros = np.count_nonzero(output_acc==0)
          # print("sparsity into layer 4 zeros ", zeros/(input.shape[0]*weights.shape[1]))
          # #print(output_acc)
          else:
           output_acc = D_buffer[0:input.shape[0]*weights.shape[1]]
           #output_acc = D_buffer[2*input.shape[0]*config.hidden_channels:2*input.shape[0]*config.hidden_channels+input.shape[0]*weights.shape[1]]
           #print(output_acc)
           
          # output_acc = output_acc.reshape(input.shape[0],weights.shape[1]) 
          # output_acc = torch.from_numpy(output_acc)       
          # output_acc = output_acc.float()
          # layern+=1
           
          #print("output acc 1")
          #print(output_acc)
          #return output_acc
          #elif(layern==2):
          # layern=1

          #output_acc = D_buffer[(layern-1)*input.shape[0]*weights.shape[1]:layern*input.shape[0]*weights.shape[1]]

          if(layern<config.total_layer_count):
           layern+=1
          else:
           layern=1
           #print("output acc")
          #print("layern here")
          #print(layern)
          
          if(self.compute_attention==1):
           output_e_val = E_buffer[0:nnz_adj].astype(config.float_type)
           output_s_val = S_buffer[0:nnz_adj].astype(config.float_type) #you should use this
        


          max_fea = my_ip.register_map.max_fea
          if(config.min_output == 0):
            print("MAX FEA INT GAT")
            print(max_fea)
            print(float(max_fea)/(2**frac_bits_o))

          max_fea_float = float(max_fea)/(2**frac_bits_o)
          if (layern == 1):
            if(max_fea_float > cur_max_fea):
             cur_max_fea = max_fea_float
          else:
            if(max_fea_float > cur_max_fea2):
             cur_max_fea2 = max_fea_float
        

          #print("Output_acc")
          #print("shapes")
          #print(input.shape[0])
          #print(weights.shape[1])

          output_acc = output_acc.reshape(input.shape[0],weights.shape[1]) 
          #get sparse matrix for e and softmax
          if(self.compute_attention==1):
            rindex = rowPtr_adj_buffer[0:nnz_adj]
            cindex =  columnIndex_adj_buffer[0:nnz_adj] 
            output_s = coo_matrix((output_s_val,(rindex ,cindex)), shape=(input.shape[0], input.shape[0]))
            output_s = output_s.todense()
            output_e = coo_matrix((output_e_val,(rindex, cindex)), shape=(input.shape[0], input.shape[0]))
            output_e = output_e.todense()

            output_s = torch.from_numpy(output_s) 
            output_e = torch.from_numpy(output_e) 
            output_s = output_s.float()
            #print("output_s")
            #print(output_s)
            output_e = output_e.float()
         
          if (config.profiling == 1): 
            bmult = time.time()
          output_acc = torch.from_numpy(output_acc)       
          output_acc = output_acc.float()

          #print("output_acc")
          #print(output_acc)
          if (config.profiling == 1):   
            print('output_acc time: {:.5f}ms'.format(1000/1*(time.time() - bmult)))
         
          #else: #layer 3
          # layern=1
          # output_acc = D_buffer[2*input.shape[0]*weights.shape[1]:3*input.shape[0]*weights.shape[1]]
          # output_acc = output_acc.reshape(input.shape[0],weights.shape[1]) 
          # output_acc = torch.from_numpy(output_acc)       
          # output_acc = output_acc.float()
           
           #print("output acc 1")
           #print(output_acc)
           #return output_acc 

          ctx.nheads = self.nheads
          ctx.alpha = self.alpha
          ctx.linear = self.linear
          ctx.compute_attention = self.compute_attention
          ctx.sage = self.sage



          if(self.compute_attention == 1):
           ctx.save_for_backward(adj,input, weights,weights_linear,output_e,output_s,output_acc)
          else:
           ctx.save_for_backward(adj,input, weights,weights_linear,adj,adj,output_acc) 
        
          if (config.profiling == 1):   
           print('Forward function time: {:.5f}ms'.format(1000/1*(time.time() - fmult)))
          return output_acc

        else: #no accelerator

          #print("No accelerator active layer: ",layern)

          input = input.float()
          torch.max(input)

          #monitor input values to calibrate thresholds
          if (layern==3): 
           max_input = torch.max(input)
           min_input = torch.min(input)
           global global_max_input
           if (max_input > global_max_input):
            global_max_input = max_input
            global global_min_input
           if (min_input < global_min_input):
            global_min_input = min_input

          if(config.fake_quantization==1):
           #print("input")
           print("active layer: ",layern) 
           if (layern==1): 
            input_q = quantization_ufbits(input, f_s, f_z, config.w_qbits)
            if(layern<config.total_layer_count):
             layern=2
            else:
             layern=1
           elif(layern==2): 
            input_q = quantization_ufbits(input, f_s2, f_z2, config.w_qbits)
            if(layern<config.total_layer_count):
             layern=3
            else:
             layern=1 
           else: 
            input_q = quantization_ufbits(input, f_sl, f_zl, config.w_qbitsl)
            layern=1
          else:
           input_q = input

          #print(adj.shape)
          #print(input.shape)
          #print(weights.shape)
          if(config.show_max_min==1):
           print("max/min weights")
           print(torch.max(weights))
           print(torch.min(weights))
        
        
    
         
          if(config.fake_quantization==1):
           if(config.show_max_min==1):
            print("max input", torch.max(input))
            print("min input", torch.min(input))
            print("max weight", torch.max(weights)) 
           #print("weights_linear") 
           #print(weights_linear) 
           weights_q = quantization_fbits(weights, w_s, w_z,  config.w_qbits) 
           weights_linear_q = quantization_fbits(weights_linear, w_sl, w_zl,  config.w_qbitsl) 
           #print("weights linear after quant")  
           #print(weights_linear_q)        
          else:
           weights_q = weights  
           weights_linear_q = weights_linear      
        
          #Wh = torch.mm(input, weights[i]) # h.shape: (N, in_features), Wh.shape: (N, out_features)
          #e = prepare_attentional_mechanism_input(Wh,attention[i],out_features)


          if(self.linear==0):
           linear_layer = torch.mm(input_q, weights_linear_q) # h.shape: (N, in_features), Wh.shape: (N, out_features)
           Wh = torch.mm(input_q, weights_q) # h.shape: (N, in_features), Wh.shape: (N, out_features)
           #need to emulate the hardware effects of the QTYPE quantization after input*weights
           if (config.fake_quantization == 1):
            #print(Wh)
            if(config.show_max_min==1):
             print("Max fea value", torch.max(Wh))
            #Wh = float_to_fix(Wh,(internal_quantization-1))
            Wh = Wh/(2**scale_fea) 
            Wh = float_to_fix(Wh,(internal_quantization-1))
            #print("Internal GNN FIFO data")
            #print(Wh[0:10])
            #Wh = torch.round(Wh, decimals = (internal_quantization-1))
            linear_layer = linear_layer/(2**scale_fea) 
            linear_layer = float_to_fix(linear_layer,(internal_quantization-1))
            a_min = -(2**internal_quantization-1)/(2**internal_quantization)
            a_max = (2**internal_quantization-1)/(2**internal_quantization)
            #print(a_min)
            #print(a_max)
            #Wh = np.clip(Wh, a_min=-0.9921875, a_max=0.9921875)
            #Wh = np.clip(Wh, a_min=-0.875, a_max=0.875)
            #Wh = np.clip(Wh, a_min=-0.99999999, a_max=0.99999999)
            Wh = torch.clip(Wh, min=a_min, max=a_max)
            #Wh = torch.round(Wh, decimals = (internal_quantization-1))
            linear_layer = torch.clip(linear_layer, min=a_min, max=a_max)
            #linear_layer = torch.round(linear_layer, decimals = (internal_quantization-1))

            #print(Wh)
           #print("attention")
           #print(attention[i])
     
           adj_d = adj.to_dense()  
           if(config.fake_quantization==1):
            attention = quantization_fbits(attention, w_s, w_z,  config.w_qbits) 
            #print("quantize adj")
            adj_d = quantization_ufbits(adj_d, a_s, a_z,  config.w_qbits) 
            if(config.show_max_min==1):
             print("max adj", torch.max(adj_d))
            adj_q = adj_d.to_sparse()
           else:
            adj_q = adj_d.to_sparse()
            

           e = prepare_attentional_mechanism_input(Wh,attention,out_features)
           e = self.leakyrelu(e)
           #print("size of e[i]")
           #print(e[i].size())
           zero_vec = -9e15*torch.ones_like(e)
          
           attention1 = torch.where(adj_d > 0, e, zero_vec)
           attentions = F.softmax(attention1, dim=1)
           #print('attentions') 
           #print(attentions[0])  
           #attention2[i] = F.dropout(attentions, dropout, True) #training set to True
           #print("attention2 shape")s
           #print(attention2[i].size())
           attention2 = attentions
          
           if(self.compute_attention==1):
            output_cpu = torch.matmul(attention2, Wh)
        
            #if (config.profiling == 1):
            # print('cpu GAT kernel time: {:.5f}s'.format(time.time() - tmult))
             #print(output_cpu)
           else:
            output_cpu = torch.matmul(adj_q, Wh)

            #if (config.profiling == 1):
            # print('cpu GCN kernel time: {:.5f}s'.format(time.time() - tmult))
            attention2 = adj_q


        #SAGE_RESIDUAL
           if(self.sage==1):
            output_cpu = output_cpu + linear_layer 


           if (config.fake_quantization==1):
            output_cpu = output_cpu*deq_o
            #linear_layer = linear_layer*deq_o

           #RELU
           if(relu==1):
            output_cpu = torch.where(output_cpu > 0, output_cpu, 0)
           
          #LINEAR ONLY
          else:
           print("output is linear")
           #print(linear_layer[0:10])
           #linear_layer = float_to_fix(linear_layer,(internal_quantization-1))

           tmult = time.time() 

           linear_layer = torch.mm(input_q, weights_linear_q) # h.shape: (N, in_features), Wh.shape: (N, out_features)

           #print(input_q[0:10]) 
           #print(weights_linear_q[0:10]) 
           #print(linear_layer[0:10]) 
           if (config.profiling == 1):   
            print('cpu linear kernel time: {:.5f}ms'.format(1000/1*(time.time() - tmult)))
           if (config.fake_quantization == 1):
            linear_layer = linear_layer/(2**scale_feal)
            #torch.set_printoptions(precision=8)
            #print(linear_layer[0:10])
            linear_layer = float_to_fix(linear_layer,(internal_quantization-1))
            #print("Internal LINEAR FIFO data")
            #print(linear_layer[0:10])
            #print(linear_layer[0:10])
            #print("test")
            #test_data = float_to_fix(0.0494,7)
            #print(test_data)
            #linear_layer = torch.round(linear_layer, decimals = (internal_quantization-1))
            a_min = -(2**internal_quantization-1)/(2**internal_quantization)
            a_max = (2**internal_quantization-1)/(2**internal_quantization)
            linear_layer = torch.clip(linear_layer, min=a_min, max=a_max)
            #linear_layer = torch.round(linear_layer, decimals = (internal_quantization-1))
            #linear_layer = torch.round(linear_layer, decimals = 3)
            #print(linear_layer[0:10])
           
           output_cpu = linear_layer 
           if (config.fake_quantization==1):
            output_cpu = output_cpu*deq_ol




          #print("output_cpu")
          #print(output_cpu[0:10])
          #print("deq_o")
          #print(deq_o)

          ctx.nheads = self.nheads
          ctx.alpha = self.alpha
          ctx.linear = self.linear
          ctx.compute_attention = self.compute_attention
          ctx.sage = self.sage
          if(self.linear==0):
           if(self.compute_attention == 1):
            ctx.save_for_backward(adj,input, weights,weights_linear,e,attention2,output_cpu)
           else:
            ctx.save_for_backward(adj,input, weights,weights_linear,adj,adj,output_cpu) 
          else:
           ctx.save_for_backward(adj,input, weights,weights_linear,adj,adj,output_cpu)  
          return output_cpu

    
  
    @staticmethod
    def backward(ctx, grad_output):
        
        def isSparse(array, m, n): 
         counter = 0
         # Count number of zeros
         # in the matrix
         for i in range(0, m):
            for j in range(0, n):
               if (array[i][j] == 0):
                   counter = counter + 1
         print("total values ",m*n)
         print("zero values ",counter)
         return (counter > ((m * n) // 2))
 
        
        if (config.accb==1):
                
         #grad_weights = input.t()@adj@grad_output
        
         print("ACCB ON")
        
         #we set the gemm_mode to 2 so dense, sparse  (adj if sparse)
                    
         adj,input, weights,e,attentions,output = ctx.saved_tensors
         nheads = ctx.nheads
         alpha = ctx.alpha
         linear = ctx.linear
         sage = ctx.sage
         compute_attention = ctx.compute_attention
        
         something = grad_adj = grad_input = grad_weights = grad_attention = None
            
      
         my_ip.register_map.gemm_mode = 2 
         my_ip.register_map.relu = 0
        
         #we set the adj_loop to point to in transpose      
            
         input_t = input.t()
        
      
         my_ip.register_map.N_adj=input_t.shape[0]
         my_ip.register_map.M_adj=input_t.shape[1]
         support_pynq_b = input_t.data.numpy()
 
         support_pynq_b = support_pynq_b.reshape(1, (input_t.shape[0]*input_t.shape[1]))


         #grad_weights = input.t()@adj@grad_output
         if(config.show_max_min==1):
          print("max/min input_t accb")
          print(np.max(support_pynq_b))
          print(np.min(support_pynq_b))
         support_pynq_q = quantization_uqbits(support_pynq_b,f_s,f_z,f_qbits)
         values_fea_buffer[0:(input_t.shape[0]*input_t.shape[1])] = (support_pynq_q * (2**0))      
             
         my_ip.register_map.values_adj1_offset_1 = values_fea_buffer.physical_address 
         my_ip.register_map.values_adj2_offset_1 = values_fea_buffer.physical_address 
         my_ip.register_map.values_adj3_offset_1 = values_fea_buffer.physical_address 
         my_ip.register_map.values_adj4_offset_1 = values_fea_buffer.physical_address 
            
         #we set the fea_loop to point to adj              

         my_ip.register_map.M_fea=adj.shape[1]

         my_ip.register_map.values_fea1_offset_1 = values_adj_buffer.physical_address 
         my_ip.register_map.values_fea2_offset_1 = values_adj_buffer.physical_address 
         my_ip.register_map.values_fea3_offset_1 = values_adj_buffer.physical_address 
         my_ip.register_map.values_fea4_offset_1 = values_adj_buffer.physical_address 
            
         my_ip.register_map.rowPtr_fea1_offset_1 = rowPtr_adj_buffer.physical_address
         my_ip.register_map.rowPtr_fea2_offset_1 = rowPtr_adj_buffer.physical_address
         my_ip.register_map.rowPtr_fea3_offset_1 = rowPtr_adj_buffer.physical_address
         my_ip.register_map.rowPtr_fea4_offset_1 = rowPtr_adj_buffer.physical_address

         my_ip.register_map.columnIndex_fea1_offset_1 =columnIndex_adj_buffer.physical_address
         my_ip.register_map.columnIndex_fea2_offset_1 =columnIndex_adj_buffer.physical_address 
         my_ip.register_map.columnIndex_fea3_offset_1 =columnIndex_adj_buffer.physical_address 
         my_ip.register_map.columnIndex_fea4_offset_1 =columnIndex_adj_buffer.physical_address
        
         support_g = torch.transpose(grad_output,0,1)
     
         my_ip.register_map.P_w=grad_output.shape[1]
        
         support_pynq_g = support_g.data.numpy()
            
         support_pynq_g = support_pynq_g.reshape(1, (grad_output.shape[0]*grad_output.shape[1]))
            
         #ojo B_buffer[0:(grad_output.shape[0]*grad_output.shape[1])] = (support_pynq_g * (1<<w_align))
         #grad_weights = input.t()@adj@grad_output
         if(config.show_max_min==1):
          print("max/min grad_output accb")
          print(np.max(support_pynq_g))
          print(np.min(support_pynq_g))
         support_pynq_q = quantization_qbits(support_pynq_g,go_s,go_z,go_qbits)
         B_buffer[0:(grad_output.shape[0]*grad_output.shape[1])] = (support_pynq_q * (2**0))
            
         amult = time.time()
         my_ip.register_map.CTRL.AP_START=1
         kernel_done = my_ip.register_map.CTRL.AP_DONE
         while kernel_done == 0:
          kernel_done = my_ip.register_map.CTRL.AP_DONE
         if (config.profiling == 1):
          print('acc backward grad_weights kernel mult: {:.5f}s'.format(time.time() - amult))
        
    
         output_acc = D_buffer[0:input_t.shape[0]*grad_output.shape[1]]*deq_gw/(2**frac_bits_o) #you should use this
      
         max_fea = my_ip.register_map.max_fea

         output_acc = output_acc.reshape(input_t.shape[0],grad_output.shape[1])    
       
         grad_weights = torch.from_numpy(output_acc).clone()  
  
         grad_weights = grad_weights.float()
                       
         my_ip.register_map.gemm_mode = 1 
         my_ip.register_map.relu = 0
         my_ip.register_map.gat_mode=compute_attention
        
         my_ip.register_map.N_adj=adj.shape[0]
         my_ip.register_map.M_adj=adj.shape[1]
         my_ip.register_map.M_fea=grad_output.shape[1]
         my_ip.register_map.P_w=weights.shape[0]
    
         my_ip.register_map.values_adj1_offset_1 = values_adj_buffer.physical_address 
         my_ip.register_map.values_adj2_offset_1 = values_adj_buffer.physical_address 
         my_ip.register_map.values_adj3_offset_1 = values_adj_buffer.physical_address 
         my_ip.register_map.values_adj4_offset_1 = values_adj_buffer.physical_address 
            
         my_ip.register_map.rowPtr_adj1_offset_1 = rowPtr_adj_buffer.physical_address
         my_ip.register_map.rowPtr_adj2_offset_1 = rowPtr_adj_buffer.physical_address
         my_ip.register_map.rowPtr_adj3_offset_1 = rowPtr_adj_buffer.physical_address
         my_ip.register_map.rowPtr_adj4_offset_1 = rowPtr_adj_buffer.physical_address

         my_ip.register_map.columnIndex_adj1_offset_1 =columnIndex_adj_buffer.physical_address
         my_ip.register_map.columnIndex_adj2_offset_1 =columnIndex_adj_buffer.physical_address 
         my_ip.register_map.columnIndex_adj3_offset_1 =columnIndex_adj_buffer.physical_address 
         my_ip.register_map.columnIndex_adj4_offset_1 =columnIndex_adj_buffer.physical_address
            
         #we set the fea_loop to point to grad_output
                  
         support_pynq_b = grad_output.data.numpy()
         #print(support_pynq_b)

         support_pynq_b = support_pynq_b.reshape(1, (grad_output.shape[0]*grad_output.shape[1]))
         #print(input_t.shape[0]*input_t.shape[1])
               
         #grad_input = adj@grad_output@weights.t()
         if(config.show_max_min==1):
          print("max/min grad_output accb")
          print(np.max(support_pynq_b))
          print(np.min(support_pynq_b))
         support_pynq_q = quantization_uqbits(support_pynq_b,go_s,go_z,go_qbits)
         values_fea_buffer[0:(grad_output.shape[0]*grad_output.shape[1])] = (support_pynq_q * (2**0))
            
         my_ip.register_map.values_fea1_offset_1 = values_fea_buffer.physical_address 
         my_ip.register_map.values_fea2_offset_1 = values_fea_buffer.physical_address 
         my_ip.register_map.values_fea3_offset_1 = values_fea_buffer.physical_address 
         my_ip.register_map.values_fea4_offset_1 = values_fea_buffer.physical_address 
                
         #we set the weights. transpose of transpose so nothing to transpose.
        
         support_g = weights
     
         support_pynq_g = support_g.data.numpy()
            
  
         support_pynq_g = support_pynq_g.reshape(1, (weights.shape[0]*weights.shape[1]))
         if(config.show_max_min==1):
          print("max/min weights_t accb")
          print(np.max(support_pynq_g))
          print(np.min(support_pynq_g))
         support_pynq_q = quantization_qbits(support_pynq_g,w_s,w_z,w_qbits)
         B_buffer[0:(weights.shape[0]*weights.shape[1])] = (support_pynq_q * (2**0))
  
         amult = time.time()
         my_ip.register_map.CTRL.AP_START=1
         kernel_done = my_ip.register_map.CTRL.AP_DONE
         while kernel_done == 0:
          kernel_done = my_ip.register_map.CTRL.AP_DONE
         print('acc backward grad_input kernel mult: {:.5f}s'.format(time.time() - amult))

         

         #grad_input = adj@grad_output@weights.t()
         output_acc2 = D_buffer[0:adj.shape[0]*weights.shape[0]]*deq_gi/(2**frac_bits_o) #you should use this
 
         max_fea = my_ip.register_map.max_fea
        
         output_acc2 = output_acc2.reshape(adj.shape[0],weights.shape[0]) 

         grad_input = torch.from_numpy(output_acc2).clone()  

         grad_input = grad_input.float()

         return something,something, something, grad_adj,something, grad_input, grad_weights, grad_attention,something,something,something,something
        
            
        else: 
        
  
         adj,input, weights,weights_linear,e,attentions,output = ctx.saved_tensors
         nheads = ctx.nheads
         alpha = ctx.alpha
         linear = ctx.linear
         sage = ctx.sage
         compute_attention = ctx.compute_attention

         #adj, input, weights,output = ctx.saved_tensors

         something = grad_adj = grad_input = grad_weights = grad_weights_linear = grad_attention = None

    
        
           
         #input = input.float()
            
        
            
    
         #print(grad_output.shape)
         #grad_input = grad_output.clone() #this to merge relub. Note that conv2 layer has no relu so this should not run for conv2
         #grad_input[input < 0] = 0
         #grad_input[output == 0] = 0 #this to merge relub
    
         ##########support = adj@input;
         ##########support2 = grad_output@weights.t()
         ##########grad_weights = support.t()@grad_output #this to unmerge relub
         #grad_weights = support.t()@grad_input #this to merge relub
    
         ##########grad_input = adj@support2
        
         
        
         tmult = time.time()
         input_t = input.t()
         #print(input)
         #print(adj)
         #print(grad_output)   
         
         #print("in")
         #print(adj)
         #print(grad_output)
         #print(input_t)

     
         if (config.profiling == 1):
          print('CPU backward grad_weights: {:.5f}s'.format(time.time() - tmult))  
        
         #print("out grad_weights")
         #print(grad_weights)
        
         # compute attention
         weights_t = weights.t()
         weights_linear_t = weights_linear.t()
  
            
         if(compute_attention==1):
         

            
          #attention matrix
          #support = torch.mm(weights_t,input_t)
          #softmax_out = torch.mm(grad_output[i], support)
          #Joe
          #print("grad_output") 
          #print(grad_output[i])
          #print(output[i].size())
          #print(grad_output[i].size())
          #delta_k_prime = output[i]*grad_output[i] 
          support = torch.mm(weights_t,input_t)
          softmax_out = torch.mm(grad_output, support)
        
          #softmax derivate

          soft_gradient = torch.empty(input.shape[0],input.shape[0])
        
  
          #d_softmax = (attentions*np.identity(2708) - attentions.t() * attentions)
          #soft_gradient = softmax_out.float() @ d_softmax.float()
        
          row_identity = np.identity(input.shape[0])
  
          #joe 
          #for j in range(data.num_nodes):
          # d_softmax = (attentions[i][j]*row_identity - attentions[i][j].t() @ attentions[i][j])
          # layer =  softmax_out[j]  
          # layer = layer.unsqueeze(0)
          # soft_gradient[j] = layer.float() @ d_softmax.float()
            
          #simpler 
          #for j in range(data.num_nodes):
          # d_softmax = (attentions[i][j]*row_identity[j] - attentions[i][j])
          # layer =  softmax_out[j]  
          # layer = layer.unsqueeze(0)
          # soft_gradient[j] = layer.float() * d_softmax.float()
        
          #magic
          dx = attentions*softmax_out
          s = dx.sum(axis=dx.ndim-1,keepdims=True)
          soft_gradient = dx - attentions*s
        
          #check_this
          #for j in range(data.num_nodes):
          # attentionv = attentions[i][j].unsqueeze(0)
          # diagonal = attentions[i][j]*row_identity
          # attentionv_t = attentionv.t()
          # outer_product  = torch.mm(attentionv_t,attentionv)
            
          # d_softmax = diagonal - outer_product
          # layer =  softmax_out[j]  
          # layer = layer.unsqueeze(0)
          # soft_gradient[j] = layer.float() @ d_softmax.float()
          #print(soft_gradient[j])
           
          #layer =  softmax_out[i]      
          #soft_gradient = layer.float()
        
          #zero_vec = -9e15*torch.ones_like(soft_gradient)
          zero_vec = torch.zeros_like(soft_gradient)
          adj_d = adj.to_dense()
          soft_gradient = torch.where(adj_d > 0, soft_gradient, zero_vec) # not sure about this but it works better with zero_vec
            
          #print('soft gradient')
          #print(soft_gradient)
          #print(soft_gradient[0])
    
          #soft_gradient[e[i] < 0] = 0.1 #normal software inplementation leaky relu 
          #dx = torch.ones_like(e[i])
          #dx[e[i] < 0] = 0.1
          dx = ((e > 0) + alpha*(e<=0)) 
            
          #with sparse e
           
        
          #print('e shapes')
          #print(soft_gradient.shape)
          #print(dx.shape)
            
          #joe c_prime*dLdL 
          soft_gradient = dx*soft_gradient
            
          #input gradient calculation

            
          #layer =  softmax_out[0]  
        
          #layer = layer.unsqueeze(0)
        
          #soft_gradient = layer.float() @ d_softmax.float()

          #soft_gradient = softmax_backward(softmax_out)
         
          #for i in range(len(softmax_out)):
          #soft_gradient = softmax_out * (1-softmax_out)
       
        
 
        
          #print('soft gradient2')
          #print(d_softmax.shape)
          #print(soft_gradient[0])
        
          #X in Joe
          #support = torch.mm(input,weights[i])   
         
          #Dlda = X @ sigma in Joe 
          #support1 = torch.mm(soft_gradient,support) 
            
          #print('support1')
          #print(d_softmax.shape)
          #print(support1[0])
          #torch_ones = torch.ones(data.num_nodes)
          #torch_ones_t = torch_ones.t()
            
          #print("grad preattention")   
          #print(support1)
            
          support = torch.mm(weights_t,input_t)
          torch_ones = torch.ones(input.shape[0])
          torch_ones_t =  torch_ones.t()
          #print(support.type())
          #print(soft_gradient.type())
          support1 = torch.mm(support,soft_gradient)
          grad_attention1 = torch.matmul(support1,torch_ones_t)
            
          support = torch.mm(input,weights)
          support2 =  torch.mm(soft_gradient,support)
          grad_attention2 = torch.matmul(torch_ones,support2)
          grad_attention2 = grad_attention2.t()
        
          #print(grad_attention1)
          
          #print(grad_attention2)
        
          tuple = (grad_attention1,grad_attention2)
        
          grad_attention = torch.cat(tuple)
            
          #grad_attention = grad_attention.unsqueeze(1)
          output_attention = grad_attention.unsqueeze(1)
         
         else:
          #output_attention = torch.zeros(size=(config.hidden_channels*2, 1)).cuda()
          #output_attention = torch.zeros(size=(config.hidden_channels*2, 1))
          output_attention = torch.zeros(size=(weights.shape[1]*2, 1))
          output_attention = output_attention.to(config.device)
         
         tmult = time.time()
         #grad_weights = input.t()@adj@grad_output
         #print("in")
         #print(weights_t)
         #print(grad_output)
        
         support = torch.mm(grad_output,weights_t)
         #print(grad_output)
         if(compute_attention==1):
          output_input = torch.mm(attentions, support)
          support = torch.mm(attentions,grad_output)
         else:
          if(sage==1):  
           output_input = torch.mm(adj, support) + torch.mm(grad_output,weights_linear_t)
          elif(linear==1):
           output_input = torch.mm(grad_output,weights_linear_t)
           #output_input = torch.zeros(size=(grad_output.shape[0],weights_linear_t.shape[1])) 
           #print("weights_linear_t")
           #print(weights_linear_t.shape)
          else:
           output_input = torch.mm(adj, support)
          support = torch.mm(adj,grad_output) 
            
         output_weights = torch.mm(input_t, support)
        
         if(sage==1):
          grad_weights_linear = torch.mm(input_t,grad_output)
         elif(linear==1):
          grad_weights_linear = torch.mm(input_t,grad_output) 
          #print('grad weights linear')
          #print(grad_weights_linear)
          #print('grad output')
          #print(grad_output)
          #print('grad input')
          #print(output_input)
         else:
          grad_weights_linear = torch.zeros(size=(input_t.shape[0],grad_output.shape[1])) 
    
         #print("out")
         #print(grad_input)
         #grad_input = adj@grad_output@weights.t()
         if (config.profiling == 1):
          print('CPU backward grad_input: {:.5f}s'.format(time.time() - tmult))
 
         #print(grad_weights)
         #print(grad_weights)
         #print(grad_input)
         #in forward: my_ip, self,adj,input, weights,attention,out_features,dropout):
        grad_weights = output_weights
         #print("grad weights")
         #print(grad_weights)
        grad_attention = output_attention

        grad_input = output_input
        #print("grad_input")
        #print(grad_input.shape)

        #print("device")
        #print(grad_attention.device)
        #return something, something, something,something, grad_input, grad_weights, grad_linear_weights, something, grad_attention, something, something, something,something

        return something, something, something,something, grad_input, grad_weights, grad_weights_linear, something, something, grad_attention, something, something,something

  

import math
import time
import sys

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init
from torch.nn import LeakyReLU
from scipy.sparse import csr_matrix
import math
import time
from pynq import allocate
import torch.nn.functional as F
import numpy as np


class Relu_SGRACE(Module):
    """
    Relu activation.

    The forward pass receives the input data (array) and exchanges any negative
    entry for zero.

    The backward pass should calculate the gradient of the maximum function in
    the forward pass and return it
    """
    def __init__(self):
        super(Relu_SGRACE, self).__init__()
        self.fn = RPYNQ.apply

        self.srelu = Parameter(torch.FloatTensor(1))
        self.srelu.data.fill_(0.0)
        #init.ones_(self.srelu.data)

     
    def forward(self, x):
        if (self.srelu.data < 0.0):
         self.srelu.data.fill_(0.0)
         #init.zeros_(self.srelu.data) #do not let relu to be negative

        #print('srelu in Relu_SGRACE')
        #print(self.srelu.data)
        output = self.fn(x,self.srelu)
        #output = self.fn(x)
        return output
    
class GATConv_SGRACE(Module):
    """
    GAT layer 
    """
    def __init__(self, in_features, out_features, nheads=1, bias=True, dropout=0.2, alpha=0.2,concat=False):
        super(GATConv_SGRACE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.dropout = dropout
        #torch.manual_seed(12345)

        self.compute_attention = 1
        self.sage=0
        self.linear=0
        
        self.weight_linear = Parameter(torch.FloatTensor(in_features, out_features))      
        
        #print('weight linear')
        #print(self.weight_linear.shape)

        self.weight = Parameter(torch.FloatTensor(in_features, out_features*nheads))
        
        init.xavier_uniform_(self.weight.data, gain=1.414)
        init.xavier_uniform_(self.weight_linear.data, gain=1.414)
        
        self.attention = Parameter(torch.empty(size=(2*out_features*nheads, 1)))
        init.xavier_uniform_(self.attention.data, gain=1.414)
        #print('first attention')
        #print(self.attention)
        self.leakyrelu = LeakyReLU(self.alpha)


        self.nheads = nheads
        self.concat = concat
        self.fn = FPYNQ_GAT.apply
        if(config.acc==1):
         self.my_ip = my_ip
        else:
         self.my_ip = None   
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def run_kernel(self):
        self.my_ip.register_map.CTRL.AP_START=1
        kernel_done = self.my_ip.register_map.CTRL.AP_DONE
        while kernel_done == 0:
            kernel_done = self.my_ip.register_map.CTRL.AP_DONE
   


    def forward(self, stream,dense, relu,input, edge_index,norm, adj,weights_l2):
        

        if(config.acc==1):
         self.my_ip.stream_mode = stream 
         self.my_ip.register_map.relu = relu
         self.my_ip.register_map.gemm_mode = dense
         self.my_ip.register_map.gat_mode = self.compute_attention
         self.my_ip.register_map.linear_mode = self.linear
         self.my_ip.register_map.sage_mode = self.sage

        if(config.instant_layer_count==1):
         if(dense==0):
          pynq_features = input.to_sparse() #coo
          nnz_fea = len(pynq_features.values())
          if(config.acc==1):
           self.my_ip.register_map.nnz_fea1 = nnz_fea
          rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
          columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
          #values_np = pynq_features.values().data.numpy() 
          values_np = pynq_features.values() 
          #values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
          values_fea_buffer[0:nnz_fea] = values_np

         else:
          if(config.device=="cpu"):   
           xaux = input.detach().numpy()
          else:
           xaux = input
          #xaux = input
          xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
          #values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
          values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
          #print('layern') 
          #print(layern)
          if(config.write_file == 1 and layern == 3):  
           df = pd.DataFrame(xaux) #convert to a dataframe
           df.to_csv("./linear_in.txt",index=False,header=False) #save to file   
          #print('edge_index shape') 
          #print(edge_index.shape)
          #print('input size') 
          #print(input.size(0))
         nnz_adj = len(norm)

         rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
         values_adj_buffer[0:nnz_adj] = norm
         columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
         if(config.acc==1):
          self.my_ip.register_map.nnz_adj1 = nnz_adj

         output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3, self.attention,self.out_features,self.dropout,relu)
 
        else: #2 layer
         if(layern==1):  
          if(dense==0):
           pynq_features = input.to_sparse() #coo
           nnz_fea = len(pynq_features.values())
           if(config.acc==1):
            self.my_ip.register_map.nnz_fea1 = nnz_fea
           rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
           columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
           #values_np = pynq_features.values().data.numpy() 
           values_np = pynq_features.values() 
           #values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
           values_fea_buffer[0:nnz_fea] = values_np

          else:
           if(config.device=="cpu"):   
            xaux = input.detach().numpy()
           else:
            xaux = input
           #xaux = input
           xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
           #values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
           values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
           #print('edge_index shape') 
           #print(edge_index.shape)
           #print('input size') 
           #print(input.size(0))

  
          nnz_adj = len(norm)
          rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
          values_adj_buffer[0:nnz_adj] = norm
          columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
          if(config.acc==1):
           self.my_ip.register_map.nnz_adj1 = nnz_adj
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, weights_l2,weights_l3,self.attention,self.out_features,self.dropout,relu)
         else: # second layer
          nnz_adj = len(norm)
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, weights_l2,weights_l3, self.attention,self.out_features,self.dropout,relu)        
       
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class GCNConv_SGRACE(Module):
    """
    GAT layer 
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv_SGRACE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #torch.manual_seed(12345)

        self.compute_attention = 0    
        self.sage = 0
        self.linear = 0
        self.dropout = 0
        self.alpha = 0
        self.nheads = 1
        self.concat = 0
        
        self.weight_linear = torch.FloatTensor(in_features, out_features)      
        
        #print('weight linear')
        #print(self.weight_linear.shape)

        self.weight = Parameter(torch.FloatTensor(in_features, out_features*self.nheads))

        init.xavier_uniform_(self.weight.data, gain=1.414)

   
        #init.xavier_uniform_(self.weight_linear.data, gain=1.414)
        
        self.attention = torch.empty(size=(2*out_features*self.nheads, 1))
        #init.xavier_uniform_(self.attention.data, gain=1.414)
        #print('first attention')
        #print(self.attention)
        self.leakyrelu = LeakyReLU(self.alpha)
    
        self.fn = FPYNQ_GAT.apply
        if(config.acc==1):
         self.my_ip = my_ip
        else:
         self.my_ip = None   
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def run_kernel(self):
        self.my_ip.register_map.CTRL.AP_START=1
        kernel_done = self.my_ip.register_map.CTRL.AP_DONE
        while kernel_done == 0:
            kernel_done = self.my_ip.register_map.CTRL.AP_DONE
   


    def forward(self, stream,dense, relu,input, edge_index,norm, adj,srelu,weights_l2,weights_l3,weights_l4):
  
        
        if(config.acc==1):
         self.my_ip.register_map.load_weights = config.load_weights
         #self.my_ip.stream_mode = stream 
         #self.my_ip.register_map.relu = relu
         #self.my_ip.register_map.gemm_mode = dense
         #self.my_ip.register_map.gat_mode = self.compute_attention
         #self.my_ip.register_map.linear_mode = self.linear
         #self.my_ip.register_map.sage_mode = self.sage
         #print("srelu")
         #print(srelu)
         srelu_buffer[0] = srelu


        if(config.instant_layer_count==1):
         if(dense==0):
          pynq_features = input.to_sparse() #coo
          nnz_fea = len(pynq_features.values())
          #print("nnz_fea")
          #print(nnz_fea)
          if(config.acc==1):
           self.my_ip.register_map.nnz_fea1 = nnz_fea
          rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
          columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
          #values_np = pynq_features.values().data.numpy() 
          values_np = pynq_features.values()\
          #values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
          values_fea_buffer[0:nnz_fea] = values_np

          write_fea = 0
          if (write_fea == 1):
           fea_np = pynq_features.indices()[0].numpy() #convert to Numpy array
           df = pd.DataFrame(fea_np) #convert to a dataframe
           df.to_csv("./fea_in.txt",sep=',',line_terminator=',',index=False,header=False) #save to file
           with open('./fea_in.txt', 'a') as f:
            f.write('\n') 
           fea_np = pynq_features.indices()[1].numpy() #convert to Numpy array  
           df = pd.DataFrame(fea_np) #convert to a dataframe
           df.to_csv("./fea_in.txt",sep=',',line_terminator=',',mode='a',index=False,header=False) #save to file
           with open('./fea_in.txt', 'a') as f:
            f.write('\n') 
           fea_np = pynq_features.values().numpy() #convert to Numpy array  
           df = pd.DataFrame(fea_np) #convert to a dataframe
           df.to_csv("./fea_in.txt",sep=',',line_terminator=',',mode='a',index=False,header=False) #save to file
           #sys.exit()

         else:
          if(config.device=="cpu"):   
           xaux = input.detach().numpy()
          else:
           xaux = input
          #xaux = input
          xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
          #values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
          values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
  
          #print('edge_index shape') 
          #print(edge_index.shape)
          #print('input size') 
          #print(input.size(0))
          #print('edge_index shape') 
          #print(edge_index.shape)
          #print('input size') 
          #print(input.size(0))

  
         nnz_adj = len(norm)
         #print('nnz_adj')
         #print(nnz_adj)
         rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
         values_adj_buffer[0:nnz_adj] = norm
         columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]

         write_adj = 0
         if (write_adj == 1):
          adj_np = edge_index[0].numpy() #convert to Numpy array
          df = pd.DataFrame(adj_np) #convert to a dataframe
          df.to_csv("./adj_in.txt",sep=',',line_terminator=',',index=False,header=False) #save to file
          adj_np = edge_index[1].numpy() #convert to Numpy array  
          with open('./adj_in.txt', 'a') as f:
            f.write('\n') 
          df = pd.DataFrame(adj_np) #convert to a dataframe
          df.to_csv("./adj_in.txt",sep=',',line_terminator=',',mode='a',index=False,header=False) #save to file
          adj_np = norm.numpy() #convert to Numpy array  
          df = pd.DataFrame(adj_np) #convert to a dataframe
          with open('./adj_in.txt', 'a') as f:
            f.write('\n')
          df.to_csv("./adj_in.txt",sep=',',line_terminator=',',mode='a',index=False,header=False) #save to file
          #sys.exit()

         if(config.acc==1):
          self.my_ip.register_map.nnz_adj1 = nnz_adj

         output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3, self.attention,self.out_features,self.dropout,relu)
 
        else: #2 layer
         if(layern==1):  
          if(dense==0):
           pynq_features = input.to_sparse() #coo
           #print("pynq features length")
           nnz_fea = len(pynq_features.values())
           #print(nnz_fea)
           #new_input = torch.rand(input.shape[0],input.shape[1])
           #new_input[new_input < 0.99] = 0.0 #1% non-zeros
           #new_input[new_input != 0.0] = 1.0 #10% non-zeros
           #new_input[::, 0] = 1.0
           #pynq_features = new_input.to_sparse() #coo
           #print("new pynq features length")
           #nnz_fea = len(pynq_features.values())
           #print(nnz_fea)
           if(config.acc==1):
            self.my_ip.register_map.nnz_fea1 = nnz_fea
           rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
           columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
           #values_np = pynq_features.values().data.numpy() 
           values_np = pynq_features.values() 
           #values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
           values_fea_buffer[0:nnz_fea] = values_np

          else:
           if(config.device=="cpu"):   
            xaux = input.detach().numpy()
           else:
            xaux = input
           #xaux = input
           xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
           #values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
           values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
           #print('edge_index shape') 
           #print(edge_index.shape)
           #print('input size') 
           #print(input.size(0))

  
          nnz_adj = len(norm)
          rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
          values_adj_buffer[0:nnz_adj] = norm
          columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
          if(config.acc==1):
           self.my_ip.register_map.nnz_adj1 = nnz_adj
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2,weights_l3,self.attention,self.out_features,self.dropout,relu)
         else: # second layer
          nnz_adj = len(norm)
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3,self.attention,self.out_features,self.dropout,relu)        
       
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SAGEConv_SGRACE(Module):
    """
    GAT layer 
    """
    def __init__(self, in_features, out_features, bias=True):
        super(SAGEConv_SGRACE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0
        self.dropout = 0
        self.concat = 0
        #torch.manual_seed(12345)

        #relu = 0 #when merging two layers OJO 
        self.compute_attention = 0
        self.nheads = 1
        self.sage=1
        self.linear=0
        
        self.weight_linear = Parameter(torch.FloatTensor(in_features, out_features))      
        
        #print('weight linear')
        #print(self.weight_linear.shape)

        self.weight = Parameter(torch.FloatTensor(in_features, out_features*self.nheads))
        
        self.attention = torch.empty(size=(2*out_features*self.nheads, 1))
        #init.xavier_uniform_(self.attention.data, gain=1.414)
        #print('first attention')
        #print(self.attention)
        self.leakyrelu = LeakyReLU(self.alpha)

        init.xavier_uniform_(self.weight.data, gain=1.414)
        init.xavier_uniform_(self.weight_linear.data, gain=1.414)
        
        self.fn = FPYNQ_GAT.apply
        if(config.acc==1):
         self.my_ip = my_ip
        else:
         self.my_ip = None   
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, stream,dense, relu,input, edge_index,norm, adj,weights_l2):
  

        if(config.acc==1):
         self.my_ip.stream_mode = stream 
         self.my_ip.register_map.relu = relu
         self.my_ip.register_map.gemm_mode = dense
         self.my_ip.register_map.gat_mode = self.compute_attention
         self.my_ip.register_map.linear_mode = self.linear
         self.my_ip.register_map.sage_mode = self.sage

        if(config.instant_layer_count==1):
         if(dense==0):
          pynq_features = input.to_sparse() #coo
          nnz_fea = len(pynq_features.values())
          if(config.acc==1):
           self.my_ip.register_map.nnz_fea1 = nnz_fea
          rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
          columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
          #values_np = pynq_features.values().data.numpy() 
          values_np = pynq_features.values() 
          #values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
          values_fea_buffer[0:nnz_fea] = values_np

         else:
          if(config.device=="cpu"):   
           xaux = input.detach().numpy()
          else:
           xaux = input
          #xaux = input
          xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
          #values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
          values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
          #print('edge_index shape') 
          #print(edge_index.shape)
          #print('input size') 
          #print(input.size(0))

  
         nnz_adj = len(norm)
         rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
         values_adj_buffer[0:nnz_adj] = norm
         columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
         if(config.acc==1):
          self.my_ip.register_map.nnz_adj1 = nnz_adj

         output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, self.attention,self.out_features,self.dropout,relu)
 
        else: #2 layer
         if(layern==1):  
          if(dense==0):
           pynq_features = input.to_sparse() #coo
           nnz_fea = len(pynq_features.values())
           if(config.acc==1):
            self.my_ip.register_map.nnz_fea1 = nnz_fea
           rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
           columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
           #values_np = pynq_features.values().data.numpy() 
           values_np = pynq_features.values() 
           #values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
           values_fea_buffer[0:nnz_fea] = values_np

          else:
           if(config.device=="cpu"):   
            xaux = input.detach().numpy()
           else:
            xaux = input
           #xaux = input
           xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
           #values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
           values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
           #print('edge_index shape') 
           #print(edge_index.shape)
           #print('input size') 
           #print(input.size(0))

  
          nnz_adj = len(norm)
          rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
          values_adj_buffer[0:nnz_adj] = norm
          columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
          if(config.acc==1):
           self.my_ip.register_map.nnz_adj1 = nnz_adj
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight,  self.weight_linear,weights_l2,weights_l3,self.attention,self.out_features,self.dropout,relu)
         else: # second layer
          nnz_adj = len(norm)
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3,self.attention,self.out_features,self.dropout,relu)        
       
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SAGEGAT_SGRACE(Module):
    """
    GAT layer 
    """
    def __init__(self, in_features, out_features, nheads=1, bias=True, dropout=0.2, alpha=0.2,concat=False):
        super(SAGEGAT_SGRACE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.dropout = dropout
        #torch.manual_seed(12345)

        #relu = 0 #when merging two layers OJO 
        self.compute_attention = 1
        self.sage=1
        self.linear=0
        
        self.weight_linear = Parameter(torch.FloatTensor(in_features, out_features))      
        
        #print('weight linear')
        #print(self.weight_linear.shape)

        self.weight = Parameter(torch.FloatTensor(in_features, out_features*nheads))
        
        init.xavier_uniform_(self.weight.data, gain=1.414)
        init.xavier_uniform_(self.weight_linear.data, gain=1.414)

        self.attention = Parameter(torch.empty(size=(2*out_features*nheads, 1)))
        init.xavier_uniform_(self.attention.data, gain=1.414)
        #print('first attention')
        #print(self.attention)

        self.nheads = nheads
        self.concat = concat

        self.leakyrelu = LeakyReLU(self.alpha)
        
        self.fn = FPYNQ_GAT.apply
        if(config.acc==1):
         self.my_ip = my_ip
        else:
         self.my_ip = None   
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, stream,dense, relu,input, edge_index,norm, adj,weights_l2):
  

        if(config.acc==1):
         self.my_ip.stream_mode = stream 
         self.my_ip.register_map.relu = relu
         self.my_ip.register_map.gemm_mode = dense
         self.my_ip.register_map.gat_mode = self.compute_attention
         self.my_ip.register_map.linear_mode = self.linear
         self.my_ip.register_map.sage_mode = self.sage

        if(config.instant_layer_count==1):
         if(dense==0):
          pynq_features = input.to_sparse() #coo
          nnz_fea = len(pynq_features.values())
          if(config.acc==1):
           self.my_ip.register_map.nnz_fea1 = nnz_fea
          rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
          columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
          #values_np = pynq_features.values().data.numpy() 
          values_np = pynq_features.values() 
          #values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
          values_fea_buffer[0:nnz_fea] = values_np

         else:
          if(config.device=="cpu"):   
           xaux = input.detach().numpy()
          else:
           xaux = input
          #xaux = input
          xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
          #values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
          values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
          #print('edge_index shape') 
          #print(edge_index.shape)
          #print('input size') 
          #print(input.size(0))

  
         nnz_adj = len(norm)
         rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
         values_adj_buffer[0:nnz_adj] = norm
         columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
         if(config.acc==1):
          self.my_ip.register_map.nnz_adj1 = nnz_adj

         output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3,self.attention,self.out_features,self.dropout,relu)
 
        else: #2 layer
         if(layern==1):  
          if(dense==0):
           pynq_features = input.to_sparse() #coo
           nnz_fea = len(pynq_features.values())
           if(config.acc==1):
            self.my_ip.register_map.nnz_fea1 = nnz_fea
           rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
           columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
           #values_np = pynq_features.values().data.numpy() 
           values_np = pynq_features.values() 
           #values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
           values_fea_buffer[0:nnz_fea] = values_np

          else:
           if(config.device=="cpu"):   
            xaux = input.detach().numpy()
           else:
            xaux = input
           #xaux = input
           xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
           #values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
           values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
           #print('edge_index shape') 
           #print(edge_index.shape)
           #print('input size') 
           #print(input.size(0))

  
          nnz_adj = len(norm)
          rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
          values_adj_buffer[0:nnz_adj] = norm
          columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
          if(config.acc==1):
           self.my_ip.register_map.nnz_adj1 = nnz_adj
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3, self.attention,self.out_features,self.dropout,relu)
         else: # second layer
          nnz_adj = len(norm)
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight,  self.weight_linear, weights_l2, weights_l3, self.attention,self.out_features,self.dropout,relu)        
       
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    

class Linear_SGRACE(Module):
    """
    Linear layer 
    """
    def __init__(self, in_features, out_features,bias=False):
        super(Linear_SGRACE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.compute_attention = 0    
        self.sage = 0
        self.linear = 1
        self.dropout = 0
        self.nheads = 1
        self.alpha = 0

        self.attention = torch.empty(size=(2*out_features, 1))
        self.weight = torch.empty(in_features, out_features) #torch.FloatTensor(in_features, out_features)
        
        self.weight_linear = Parameter(torch.FloatTensor(in_features, out_features))      
        
       
        #init.xavier_uniform_(self.weight_linear.data, gain=1.414)
        init.xavier_uniform_(self.weight_linear.data)
     
        self.fn = FPYNQ_GAT.apply
        if(config.acc==1):
         self.my_ip = my_ip
        else:
         self.my_ip = None   
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

 

    def forward(self, stream, dense, relu, input, edge_index,norm, adj,weights_l2,weights_l3,weights_l4):
    
        

        if(config.acc==1):
         self.my_ip.stream_mode = stream 
         self.my_ip.register_map.relu = relu
         self.my_ip.register_map.gemm_mode = dense
         self.my_ip.register_map.gat_mode = self.compute_attention
         self.my_ip.register_map.linear_mode = self.linear
         self.my_ip.register_map.sage_mode = self.sage

        if(config.instant_layer_count==1):
         if(dense==0):
          pynq_features = input.to_sparse() #coo
          nnz_fea = len(pynq_features.values())
          if(config.acc==1):
           self.my_ip.register_map.nnz_fea1 = nnz_fea
          if(config.show_max_min==1):
           print("active layer: ",layern)   
           print('max/min features')   
           print(np.max(input))
           print(np.min(input))
          rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
          columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
          #values_np = pynq_features.values().data.numpy() 
          values_np = pynq_features.values() 
          #values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
          values_fea_buffer[0:nnz_fea] = values_np

         else:
          if(config.device=="cpu"):   
           xaux = input.detach().numpy()
          else:
           xaux = input
          #xaux = input
          if(config.show_max_min==1):
           print("active layer: ",layern)   
           print('max/min features')   
           print(np.max(xaux))
           print(np.min(xaux))
          xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
          #values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
          values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
          #print('layern') 
          #print(layern)
          if(config.write_file == 1 and layern == 3):  
           df = pd.DataFrame(xaux) #convert to a dataframe
           df.to_csv("./linear_in.txt",index=False,header=False) #save to file  
          #print('values fea buffer')
          #print(values_fea_buffer[0:10])
          #print('edge_index shape') 
          #print(edge_index.shape)
          #print('input size') 
          #print(input.size(0))

  
         nnz_adj = len(norm)
         #rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
         #values_adj_buffer[0:nnz_adj] = norm
         #columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
         if(config.acc==1):
          self.my_ip.register_map.nnz_adj1 = nnz_adj

         output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3, self.attention,self.out_features,self.dropout,relu)
 
        else: #n layer
         if(layern==1):  
          if(dense==0):
           pynq_features = input.to_sparse() #coo
           nnz_fea = len(pynq_features.values())
           if(config.acc==1):
            self.my_ip.register_map.nnz_fea1 = nnz_fea
           rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
           columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
           #values_np = pynq_features.values().data.numpy() 
           values_np = pynq_features.values() 
           #values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
           values_fea_buffer[0:nnz_fea] = values_np

          else:
           if(config.device=="cpu"):   
            xaux = input.detach().numpy()
           else:
            xaux = input
           #xaux = input
           xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
           #values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
           values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
           #print('edge_index shape') 
           #print(edge_index.shape)
           #print('input size') 
           #print(input.size(0))

  
          nnz_adj = len(norm)
          rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
          values_adj_buffer[0:nnz_adj] = norm
          columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
          if(config.acc==1):
           self.my_ip.register_map.nnz_adj1 = nnz_adj 
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear,weights_l2,weights_l3,self.attention,self.out_features,self.dropout,relu)
         else: # second layer
          nnz_adj = len(norm)
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear,weights_l2, weights_l3,self.attention,self.out_features,self.dropout,relu)        
       
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

def init_SGRACE(number_of_classes):

 if(config.acc==1):
  ol = Overlay("gat_all_unsigned.bit")
 #else:
 # ol = Overlay("gat_all_unsigned.bit",download=False)
  global my_ip
  my_ip = ol.mmult_top_0
 
 #print("my ip")
 #print(ol.ip_dict)

 global global_max_input
 global_max_input = 0
 global global_min_input
 global_min_input = 0 
 global frac_bits_o
 frac_bits_o = 16
 global frac_bits
 frac_bits = 8
 global f_align
 global beta_qu
 global scale_fea
 global layern 
 global deq_o
 global scale_fea2
 global deq_o2
 global scale_feal
 global deq_ol
 #remember layer number active to adjust the parameters 

 layern = 1

 if(config.w_qbitsl == 32): #float,need this for calculations but then ignored.

  f_alignl = 0 #8
  beta_qul = 255
  #w_maxl = 0.5 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_minl = -0.5 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_maxl = 0.5 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_minl = -0.5 #citeseer/cora 4-bit/8-bit gcn/gat
  w_maxl = 0.5 #foto
  w_minl = -0.5 #foto
  f_maxl = 8.0 #cora
  #l_maxl = 4.0 #4.0 #2 cora
  #l_minl = -4.0 #-4.0 #-2 cora
  l_maxl = 4.0 #4.0 #2 foto
  l_minl = -4.0 #-4.0 #-2 foto
  #w_maxl = 1.0 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_minl = -1.0 #citeseer/cora 4-bit/8-bit gcn/gat
  #f_maxl = 4.0 #cora
  #f_maxl = 4.0 #cora
  f_minl = 0.0

 if(config.w_qbitsl == 8):

  #cora/photo
  #f_alignl = 0 #8
  #beta_qul = 255
  #w_maxl = 1.0#foto
  #w_minl = -1.0 #foto
  #f_maxl = 8.0 #cora
  #l_maxl = 4.0 #4.0 #2 foto
  #l_minl = -4.0 #-4.0 #-2 foto
  #f_minl = 0.0
  #computers
  f_alignl = 0 #8
  beta_qul = 255
  w_maxl = 1.0#foto
  w_minl = -1.0 #foto
  f_maxl = 8.0 #cora
  l_maxl = 4.0 #4.0 #2 foto
  l_minl = -4.0 #-4.0 #-2 foto
  f_minl = 0.0

 elif(config.w_qbitsl == 4):
  f_alignl = 4 #8
  beta_qul = 15
  w_maxl = 0.5 #citeseer/cora 4-bit/8-bit gcn/gat
  w_minl = -0.5 #citeseer/cora 4-bit/8-bit gcn/gat
  #f_maxl = 16.0 #cora  
  f_maxl = 2.0 #cora
  #f_maxl = 1.0 #cora
  f_minl = 0
  l_maxl = 1.0 #4.0 #2
  l_minl = -1.0 #-4.0 #-2

 elif(config.w_qbitsl==1):
  f_alignl = 6 #8
  beta_qul = 1
  w_maxl = 0.1 #citeseer/cora 4-bit/8-bit gcn/gat
  w_minl = -0.1 #citeseer/cora 4-bit/8-bit gcn/gat  
  f_maxl = 0.1 #cora
  f_minl = 0.0  
  l_maxl = 0.1 #2
  l_minl = -0.1 #-2

 if(config.w_qbits == 8):

  f_align = 0 #8
  beta_qu = 255

  #photo
  global w_max
  w_max = 1.0
  global w_min
  w_min = -1.0 
  global a_max
  a_max = 1.0 
  global w_max2
  w_max2 = 1.0
  global w_min2
  w_min2 = -1.0
  global f_max2   
  f_max2 = 1.0 
  global f_max
  f_max = 1.0 
  global a_min
  a_min = 0
  global f_min
  f_min = 0
  global f_min2
  f_min2 = 0
  go_max = 0.10
  go_min = -0.10

  #cora
  #global w_max
  #w_max = 1.0 #8 #citeseer/cora
  #global w_min
  #w_min = -1.0 #8 #citeseer/cora
  #global a_max
  #a_max = 1.0 #cora gcn/gat 8-bit
  #w_max2 = 1.0 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_min2 = -1.0 #citeseer/cora 4-bit/8-bit gcn/gat  
  #f_max2 = 1.0 #cora
  #global f_max
  #f_max = 1.0 #cox/ermd/dd/mutag
  #global a_min
  #a_min = 0
  #global f_min
  #f_min = 0
  #f_min2 = 0
  #go_max = 0.10
  #go_min = -0.10

  #weibo
  #global w_max
  #w_max = 1.0 #8 #citeseer/cora
  #global w_min
  #w_min = -1.0 #8 #citeseer/cora
  #global a_max
  #a_max = 1.0 #cora gcn/gat 8-bit
  #w_max2 = 1.0 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_min2 = -1.0 #citeseer/cora 4-bit/8-bit gcn/gat  
  #f_max2 = 1.0 #cora
  #global f_max
  #f_max = 1.0 #cox/ermd/dd/mutag
  #global a_min
  #a_min = 0
  #global f_min
  #f_min = 0
  #f_min2 = 0
  #go_max = 0.10
  #go_min = -0.10

  #enrom
  #global w_max
  #w_max = 1.0 #8 #citeseer/cora
  #global w_min
  #w_min = -1.0 #8 #citeseer/cora
  #global a_max
  #a_max = 2.0 #cora gcn/gat 8-bit
  #w_max2 = 1.0 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_min2 = -1.0 #citeseer/cora 4-bit/8-bit gcn/gat  
  #f_max2 = 1.0 #cora
  #global f_max
  #f_max = 1.0 #cox/ermd/dd/mutag
  #global a_min
  #a_min = 0
  #global f_min
  #f_min = 0
  #f_min2 = 0
  #go_max = 0.10
  #go_min = -0.10

  #cora
  #global w_max
  #w_max = 0.3 #8 #citeseer/cora
  #global w_min
  #w_min = -0.3 #8 #citeseer/cora
  #global a_max
  #a_max = 0.5 #cora gcn/gat 8-bit
  #w_max2 = 0.6 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_min2 = -0.6 #citeseer/cora 4-bit/8-bit gcn/gat  
  #f_max2 = 1.0 #cora
  #global f_max
  #f_max = 1.0 #cox/ermd/dd/mutag
  #global a_min
  #a_min = 0
  #global f_min
  #f_min = 0
  #f_min2 = 0
  #go_max = 0.10
  #go_min = -0.10

  #transformer gat
  #w_max = 1.0 #0.3 #8 #citeseer/cora
  #w_min = -1.0 #-0.3 #8 #citeseer/cora
  #a_max = 1.0 #cora gcn/gat 8-bit
  #w_max2 = 1.0 # 0.6 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_min2 = -1.0 #-0.6 #citeseer/cora 4-bit/8-bit gcn/gat  
  #f_max2 = 1.0 #cora
  #f_max = 1.0 #cox/ermd/dd/mutag
  #a_min = 0
  #f_min = 0
  #f_min2 = 0
  #go_max = 0.10
  #go_min = -0.10

  #cora gat
  #w_max = 0.3 #0.3 #8 #citeseer/cora
  #w_min = -0.3 #-0.3 #8 #citeseer/cora
  #a_max = 1.0 #cora gcn/gat 8-bit
  #w_max2 = 0.6 # 0.6 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_min2 = -0.6 #-0.6 #citeseer/cora 4-bit/8-bit gcn/gat  
  #f_max2 = 1.0 #cora
  #f_max = 1.0 #cox/ermd/dd/mutag
  #a_min = 0
  #f_min = 0
  #f_min2 = 0
  #go_max = 0.10
  #go_min = -0.10


    
 elif(config.w_qbits == 4):

  f_align = 4 
  #w_max = 0.5 #citeseer 4-bit/8-bit gcn/gat
  #w_min = -0.5 #citeseer 4-bit/8-bit gcn/ga
  #a_max = 1.0 #cora gcn/gat 8-bit
  #a_min = 0.0 #in training the first tensor of the matrix could be negative. In inference is always positive.  
  #f_max = 1.0 #cox/ermd/dd/mutag
  #f_min = 0.0
  #go_max = 0.10
  #go_min = -0.10

  #enrom
  
  #w_max = 1.0 #8 #citeseer/cora
  
  #w_min = -1.0 #8 #citeseer/cora
  
  #a_max = 1.0 #cora gcn/gat 8-bit
  #w_max2 = 1.0 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_min2 = -1.0 #citeseer/cora 4-bit/8-bit gcn/gat  
  #f_max2 = 1.0 #cora
  
  #f_max = 1.0 #cox/ermd/dd/mutag
  #a_min = 0

  #f_min = 0
  #f_min2 = 0
  #go_max = 0.10
  #go_min = -0.10


  #cora

  w_max = 0.3 #8 #citeseer/cora

  w_min = -0.3 #8 #citeseer/cora

  a_max = 0.5 #cora gcn/gat 8-bit
  w_max2 = 0.6 #citeseer/cora 4-bit/8-bit gcn/gat
  w_min2 = -0.6 #citeseer/cora 4-bit/8-bit gcn/gat  
  f_max2 = 1.0 #cora

  
  f_max = 1.0 #cox/ermd/dd/mutag

  a_min = 0

  f_min = 0
  f_min2 = 0
  go_max = 0.10
  go_min = -0.10

  #photo  
  #w_max = 1.0 #photo
  #w_min = -1.0 #photo
  #a_max = 1.0 #computers/photo
  #f_max = 1.0 
  #f_min = 0.0 
  #a_min = 0.0
  #f_max2 = 2.0 #photo
  #f_min2 = 0.0
  #w_max2 = 1.0 #5 #computers/photo
  #w_min2 = -1.0 #computers/photo

 elif(config.w_qbits == 2):

  f_align = 6


  #cora
  w_max = 0.1 #citeseer 4-bit/8-bit gcn/gat
  w_min = -0.1 #citeseer 4-bit/8-bit gcn/gat
  w_max2 = 0.1 #citeseer 4-bit/8-bit gcn/gat
  w_min2 = -0.1 #citeseer 4-bit/8-bit gcn/gat
  a_max = 0.1 #cora gcn/gat 8-bit
  a_min = 0.0 #in training the first tensor of the matrix could be negative. In inference is always positive.  
  f_max = 1.0 #cox/ermd/dd/mutag citeseer/cora
  f_min = 0.0 
  f_max2 = 1.0 #cox/ermd/dd/mutag citeseer/cora
  f_min2 = 0.0 
  go_max = 0.10
  go_min = -0.10



 elif(config.w_qbits == 1):
    
  f_align = 6 #note that we have 6 here because this is needed for the quantization constants. Hardware receives 7.
  w_max = 0.1 #cora 4-bit/8-bit gcn/gat
  w_min = -0.1 #cora 4-bit/8-bit gcn/gat
  w_max2 = 0.1 #cora 4-bit/8-bit gcn/gat
  w_min2 = -0.1 #cora 4-bit/8-bit gcn/gat
  a_max = 0.1 #cora gcn/gat 8-bit
  a_min = 0.0 #in training the first tensor of the matrix could be negative. In inference is always positive.  
  f_max = 1.0 #cox/ermd/dd/mutag
  f_min = 0.0 
  f_max2 = 1.0 #cox/ermd/dd/mutag
  f_min2 = 0.0 

  go_max = 0.10
  go_min = -0.10


 # do no touch
 hard_type = np.int8
 frac_bits_o = 16
 frac_bits = 8
 out_type = np.int32
 config.float_type = np.float32
 global cur_max_fea
 cur_max_fea=0.0 #keep track of max internal value seen during training
 global cur_max_fea2
 cur_max_fea2=0.0 #keep track of max internal value seen during training
 global model_buffer
 global P_w_buffer
 global srelu_buffer
 global quantization_scale_fea_buffer
 global quantization_scale_w_buffer
 global quantization_scale_l_buffer
 global deq_factor_buffer
 global scale_fea_buffer 
 if (config.acc == 1):
  model_buffer  = allocate(5, dtype=config.model_type)
  srelu_buffer = allocate(5, dtype=config.float_type)
  quantization_scale_fea_buffer = allocate(5, dtype=config.float_type)
  quantization_scale_w_buffer = allocate(5, dtype=config.float_type)
  quantization_scale_l_buffer = allocate(5, dtype=config.float_type)
  deq_factor_buffer = allocate(5, dtype=config.float_type)
  scale_fea_buffer = allocate(5, dtype=config.model_type)
  P_w_buffer = allocate(5, dtype=config.model_type) 
 else:
  model_buffer  = []
  srelu_buffer = []
  quantization_scale_fea_buffer = []
  quantization_scale_w_buffer = []
  quantization_scale_l_buffer = []
  deq_factor_buffer = []
  scale_fea_buffer = []
  P_w_buffer = []
 
 #models 
 if (config.acc == 1):

  #program
  #1 layer execution
  if(config.total_layer_count == 2):
   if(config.instant_layer_count == 1): 
    model_buffer[0] = 0x10
    model_buffer[1] = 0x02 #12 relu 02 no relu important for low bit precision
   else:
    #fuse two layer
    model_buffer[0] =0x14#sparse first input
    model_buffer[1] =0x1a#sparse
    #model_buffer[1] =0x1e#dense
  elif(config.total_layer_count == 3):
   if(config.instant_layer_count == 1): 
    model_buffer[0] = 0x10
    model_buffer[1] = 0x02
    model_buffer[2] = 0x42
   else:
    #fuse three layer
    model_buffer[0] =0x14#sparse first input
    model_buffer[1] =0x0c#sparse  no relu
    #model_buffer[1] =0x1c#sparse  relu
    #model_buffer[1] =0x0e#dense no relu
    #model_buffer[1] =0x1e#dense  relu
    model_buffer[2] =0x48 #sparse
    #model_buffer[2] =0x4a #dense
  elif(config.total_layer_count == 4):
   if(config.instant_layer_count == 1): 
    model_buffer[0] = 0x10
    model_buffer[1] = 0x12
    model_buffer[2] = 0x52
    model_buffer[3] = 0x42
   else:
    #fuse four layer
    model_buffer[0] =0x14
    model_buffer[1] =0x1e
    model_buffer[2] =0x5c #sparse
    #model_buffer[2] =0x5e #dense
    model_buffer[3] =0x4a #dense
    #model_buffer[3] =0x48 #sparse


  #fuse two layer
  #model_buffer[0] = 0x14
  #model_buffer[1] = 0x1a






 global attention_buffer
 if (config.acc == 1):
  attention_buffer  = allocate(config.P_w*2, dtype=config.float_type)
 else:
  attention_buffer  = []

 global bias_buffer
 if (config.acc == 1):
  bias_buffer  = allocate(1024, dtype=np.int32)
 else:
  bias_buffer  = []

 global profiling_buffer
 if (config.acc == 1):
  profiling_buffer  = allocate(16, dtype=np.int64)
 else:
  profiling_buffer  = []

 global rowPtr_fea_buffer
 if (config.acc == 1):
  rowPtr_fea_buffer = allocate(config.NNZ_fea, dtype=np.int32)
 else:
  rowPtr_fea_buffer = []


 #print('allocate rowPtr_fea_buffer.physical_address')
 #print(rowPtr_fea_buffer.physical_address)
 #print(config.rowPtr_fea_buffer)
 
 global columnIndex_fea_buffer
 if (config.acc == 1):
  columnIndex_fea_buffer = allocate(config.NNZ_fea, dtype=np.int32)
 else:
  columnIndex_fea_buffer = []

 
 global values_fea_buffer
 #if (config.hardware_quantize == 0):
 # config.values_fea_buffer = allocate(config.NNZ_fea, dtype=config.hard_type)
 #else:    
 #values_fea_buffer = allocate(config.N_adj*64, dtype=config.float_type)
 if (config.acc == 1):
  values_fea_buffer = allocate(config.NNZ_fea, dtype=config.float_type)  
 else:
  values_fea_buffer = []
    

 global rowPtr_adj_buffer
 if (config.acc == 1):
  rowPtr_adj_buffer = allocate(config.NNZ_adj, dtype=np.int32)
 else:
  rowPtr_adj_buffer = []


 global columnIndex_adj_buffer
 if (config.acc == 1):
  columnIndex_adj_buffer = allocate(config.NNZ_adj, dtype=np.int32)
 else:
  columnIndex_adj_buffer = []



 global values_adj_buffer
 if (config.acc == 1):
  values_adj_buffer = allocate(config.NNZ_adj, dtype=config.float_type)
 else:
  values_adj_buffer = []
 
 global D_buffer
 global B_buffer  
 global B2_buffer
 if (config.acc == 1):
  #B_buffer = allocate((config.N_adj*config.P_w*config.head_count), dtype=config.float_type)
  B2_buffer = allocate((config.N_adj*config.P_w+2*config.P_w*config.P_w), dtype=config.float_type)
  B_buffer = allocate((config.N_adj*config.P_w+2*config.P_w*config.P_w), dtype=config.float_type) #two layers
  D_buffer = allocate((config.N_adj*config.P_w+2*config.P_w*config.P_w), dtype=config.float_type) 
 else:
  B2_buffer = []
  B_buffer = []
  D_buffer = [] 
 #small buffer to store the E sparse information for backward.
 
 global E_buffer
 if (config.acc == 1):
  E_buffer = allocate(config.NNZ_adj,dtype=config.float_type)
 else:
  E_buffer = []
   
 #small buffer to store the result of softmax with lots of zero probabilities
 
 global S_buffer
 if (config.acc == 1):
  S_buffer = allocate(config.NNZ_adj,dtype=config.float_type)
 else:
  S_buffer = []


 a_qbits = config.w_qbits
 f_qbits = config.w_qbits
 f_qbitsl = config.w_qbitsl
 l_qbitsl = config.w_qbitsl

 go_qbits = 8

 #generate constants

 if(config.min_output == 0):
  print("generating qbits w constants with bits: ",config.w_qbits)
 


 #signed w
 global w_s_o,w_s,w_z
 w_s_o,w_s,w_z=generate_quantization_qbits_constants(w_min, w_max,config.w_qbits)
 global w_s_o2,w_s2,w_z2
 w_s_o2,w_s2,w_z2=generate_quantization_qbits_constants(w_min2, w_max2,config.w_qbits) #forward

 global w_s_ol,w_sl,w_zl
 w_s_ol,w_sl,w_zl=generate_quantization_qbits_constants(w_minl, w_maxl,config.w_qbitsl) #forward


 
 if(config.min_output == 0):
  print(w_s)
 #unsigned a and f
 if(config.min_output == 0):
  print("generating qbits a constants")
 global a_s_o,a_s,a_z
 a_s_o,a_s,a_z=generate_quantization_uqbits_constants(a_min, a_max,a_qbits)
 
 if(config.min_output == 0):
  print(a_s)
 if(config.min_output == 0):
  print("generating qbits f constants")
 global f_s_o,f_s,f_z
 f_s_o,f_s,f_z=generate_quantization_uqbits_constants(f_min, f_max,f_qbits)
 global f_s_o2,f_s2,f_z2
 f_s_o2,f_s2,f_z2=generate_quantization_uqbits_constants(f_min2, f_max2,f_qbits)
 global f_s_ol,f_sl,f_zl
 f_s_ol,f_sl,f_zl=generate_quantization_uqbits_constants(f_minl, f_maxl,f_qbitsl)

  #sign linear layer
 global l_s_ol,l_sl,l_zl
 l_s_ol,l_sl,l_zl=generate_quantization_qbits_constants(l_minl, l_maxl,l_qbitsl)

 quantization_scale_fea_buffer[0] = (1/f_s)
 quantization_scale_fea_buffer[1] = (1/f_s2)
 quantization_scale_fea_buffer[2] = (1/f_sl)
 quantization_scale_fea_buffer[3] = (1/f_sl)

 quantization_scale_w_buffer[0] = (1/w_s)
 quantization_scale_w_buffer[1] = (1/w_s2)
 quantization_scale_w_buffer[2] = (1/w_sl)
 quantization_scale_w_buffer[3] = (1/w_sl)

 quantization_scale_l_buffer[0] = (1/l_sl)
 quantization_scale_l_buffer[1] = (1/l_sl)
 quantization_scale_l_buffer[2] = (1/l_sl)
 quantization_scale_l_buffer[3] = (1/l_sl)

 srelu_buffer[0] = 0.0
 srelu_buffer[1] = 0.0
 srelu_buffer[2] = 0.0
 srelu_buffer[3] = 0.0
 srelu_buffer[4] = 0.0

 if(config.total_layer_count==2):
  P_w_buffer[0] = config.hidden_channels
  P_w_buffer[1] = config.hidden_channels
 if(config.total_layer_count==3):
  P_w_buffer[0] = config.hidden_channels
  P_w_buffer[1] = config.hidden_channels
  P_w_buffer[2] = number_of_classes
 if(config.total_layer_count==4):
  P_w_buffer[0] = config.hidden_channels
  P_w_buffer[1] = config.hidden_channels
  P_w_buffer[2] = config.hidden_channels
  P_w_buffer[3] = number_of_classes

 #quantization_scale_fea_buffer[0] = np.asarray((1/f_s), dtype=np.float32)#1/f_s
 #quantization_scale_fea_buffer[1] = np.asarray((1/f_s2), dtype=np.float32)#1/f_s2
 #quantization_scale_fea_buffer[2] = np.asarray((1/f_sl), dtype=np.float32)#1/f_sl

 #quantization_scale_w_buffer[0] = np.asarray((1/w_s), dtype=np.float32)#1/w_s
 #quantization_scale_w_buffer[1] = np.asarray((1/w_s2), dtype=np.float32)#1/w_s2
 #quantization_scale_w_buffer[2] = np.asarray((1/w_sl), dtype=np.float32)#1/w_sl


 
 if(config.min_output == 0):
  print(f_s)
 if(config.min_output == 0):
  print("generating qbits gi gradient input constants")
 go_s_o,go_s,go_z=generate_quantization_uqbits_constants(go_min, go_max,go_qbits)

 deq_o = w_s_o*f_s_o*a_s_o

 #deq_ol = w_s_ol*f_s_ol #linear layer
 deq_ol = w_s_ol*l_s_ol #linear layer



 deq = w_s*f_s*a_s
 #print("DEQ")
 #print(deq)

 deq_o2 = w_s_o2*f_s_o2*a_s_o
 deq_gw = f_s_o*a_s_o*go_s_o
 deq_gi = a_s_o*go_s_o*w_s_o
    

 #adjust internal quantization

    #(4/3)

 if (config.w_qbitsl == 32): #float  
    scale_feal = 0
    deq_ol=1.0
    #scale_feal = 1 cora
    #deq_ol=deq_ol*pow(2, 1) cora
    my_ip.register_map.f_alignl = 0
    my_ip.register_map.beta_qul = 255

 if (config.w_qbitsl == 8):  
    #scale_feal = 2 #2 photo
    #deq_ol=deq_ol*pow(2, 2) #2 photo
    #scale_feal = 1 #cora
    #deq_ol=deq_ol*pow(2, 1) #cora
    scale_feal = 2 #2 computers
    deq_ol=deq_ol*pow(2, 2) #2 computers
    my_ip.register_map.f_alignl = 0
    my_ip.register_map.beta_qul = 255

 if (config.w_qbitsl == 4):  
    #scale_feal = 2
    #deq_ol=deq_ol*pow(2, 2)
    scale_feal = 1
    deq_ol=deq_ol*pow(2, 1)
    my_ip.register_map.f_alignl = 4
    my_ip.register_map.beta_qul = 15

 if (config.w_qbitsl == 1):  
    scale_feal = 1
    deq_ol=deq_ol*pow(2, 1)
    my_ip.register_map.f_alignl = 7
    my_ip.register_map.beta_qul = 1


    #scale_feal = 1
    #deq_ol=deq_ol*pow(2, 1)

 global internal_quantization 
 #8-bit 
 if (config.w_qbits == 8):   
    #weibo
    #scale_fea = 4 #scale fea
    #scale_fea2 = 4
    #deq_o=deq_o*pow(2, 1) #cora 8-bit gcn/gat
    #deq_o2=deq_o2*pow(2, 1) #2c#cora 8-bit gcn/gat

    #computers
    scale_fea = 3
    scale_fea2 = 3
    deq_o=deq_o*pow(2, 1) 
    deq_o2=deq_o2*pow(2, 1)

    #photo
    #scale_fea = 3 
    #scale_fea2 = 3
    #deq_o=deq_o*pow(2, 1) 
    #deq_o2=deq_o2*pow(2, 1) 
   
    #test sage
    #scale_fea = 3
    #scale_fea2 = 3
    #deq_o=deq_o*pow(2, 1) 
    #deq_o2=deq_o2*pow(2, 1)

    #cora with final SGRACE linear 0.84
    #scale_fea = 3
    #scale_fea2 = 3 #4
    #deq_o=deq_o*pow(2, 1) 
    #deq_o2=deq_o2*pow(2, 1) 

    #amazon/cora gae
    #scale_fea = 3
    #scale_fea2 = 3
    #deq_o=deq_o*pow(2, 1) 
    #deq_o2=deq_o2*pow(2, 1) 

    #enron
    #scale_fea = 6
    #scale_fea2 = 6
    #deq_o=deq_o*pow(2, 1) 
    #deq_o2=deq_o2*pow(2, 1) 


    if(config.min_output == 0):
     print("Deq factor ",deq_o)
    
    #int32bits = np.asarray(deq_o, dtype=np.float32).view(np.int32).item() 
    #my_ip.register_map.deq_factor = int32bits
    #qsf = 1/f_s
    #int32bits = np.asarray(qsf, dtype=np.float32).view(np.int32).item() 
    #my_ip.register_map.quantization_scale_fea = int32bits
    #if(config.min_output == 0):
    # print("qsf ",qsf)
    #qsw = 1/w_s
    #int32bits = np.asarray(qsw, dtype=np.float32).view(np.int32).item() 
    #my_ip.register_map.quantization_scale_w = int32bits
    #if(config.min_output == 0):
    # print("qsw ",qsw)
    #qsa = 1/a_s
    #int32bits = np.asarray(qsa, dtype=np.float32).view(np.int32).item() 
    #my_ip.register_map.quantization_scale_adj = int32bits
    #print("f align is ",f_align)
    if(config.acc==1):
     my_ip.register_map.f_align = 0
     my_ip.register_map.beta_qu = 255
    internal_quantization =  16 # 0x0000FFFF#bit QTYPE 32
  
 #4-bit 
 if (config.w_qbits == 4):
       
    #cora
    #my_ip.register_map.scale_fea = 3 #scale fea
    #deq_o=deq_o*pow(2, 2)
    if(config.acc==1):
     my_ip.register_map.f_align = 4
     my_ip.register_map.beta_qu = 15
    internal_quantization =  8 #bit QTYPE 4

    #cora
    scale_fea = 3
    scale_fea2 = 3
    deq_o=deq_o*pow(2, 3)
    deq_o2=deq_o2*pow(2,3)

    #amazom
    #scale_fea = 3
    #scale_fea2 = 3
    #deq_o=deq_o*pow(2, 3)
    #deq_o2=deq_o2*pow(2,3)

    
    #weibo
    #scale_fea = 3
    #scale_fea2 = 3
    #deq_o=deq_o*pow(2, 1)
    #deq_o2=deq_o2*pow(2,1)

    #emron
    #scale_fea = 4
    #scale_fea2 = 4
    #deq_o=deq_o*pow(2, 1)
    #deq_o2=deq_o2*pow(2,1)

    #photo
    #scale_fea = 6
    #scale_fea2 = 1
    #deq_o=deq_o*pow(2, 1)
    #deq_o2=deq_o2*pow(2, 1)


 #2-bit
 if (config.w_qbits == 2):
       
  
    if(config.min_output == 0):
     print("Deq factor ",deq_o)
    
    #amazom
    #scale_fea = 3
    #scale_fea2 = 3
    #deq_o=deq_o*pow(2, 3)
    #deq_o2=deq_o2*pow(2, 3)

    #cora
    scale_fea = 1
    scale_fea2 = 1
    deq_o=deq_o*pow(2,1)
    deq_o2=deq_o2*pow(2,1)

    #weibo
    #scale_fea = 3
    #scale_fea2 = 3
    #deq_o=deq_o*pow(2, 1)
    #deq_o2=deq_o2*pow(2,1)
    
    #emron
    #scale_fea = 2
    #scale_fea2 = 2
    #deq_o=deq_o*pow(2, 1)
    #deq_o2=deq_o2*pow(2,1)

    if(config.acc==1):
     my_ip.register_map.beta_qu = 2
     my_ip.register_map.f_align = 6
    internal_quantization =  4
    
 if (config.w_qbits == 1):



    #amazon cora
    scale_fea = 1
    scale_fea2 = 1
    deq_o=deq_o*pow(2, 1)
    deq_o2=deq_o2*pow(2, 1)

    #emron
    #scale_fea = 2
    #scale_fea2 = 2
    #deq_o=deq_o*pow(2, 1)
    #deq_o2=deq_o2*pow(2, 1)

    if(config.acc==1):
     my_ip.register_map.beta_qu = 1
     my_ip.register_map.f_align = 7
    internal_quantization =  4
    f_align = 7
     
 
    
    if(config.min_output == 0):
     print("Deq factor ",deq_o)


 deq_factor_buffer[0] = deq_o
 deq_factor_buffer[1] = deq_o2
 deq_factor_buffer[2] = deq_ol  

 scale_fea_buffer[0] = scale_fea
 scale_fea_buffer[1] = scale_fea2
 scale_fea_buffer[2] = scale_feal

 #deq_factor_buffer[0] = np.asarray(deq_o, dtype=np.float32)#deq_o
 #deq_factor_buffer[1] = np.asarray(deq_o2, dtype=np.float32)#deq_o2
 #deq_factor_buffer[2] = np.asarray(deq_ol, dtype=np.float32) 

 print("instant_layer_count")
 print(config.instant_layer_count)

 print("internal scale")
 print(scale_fea_buffer[0])
 print(scale_fea_buffer[1])
 print(scale_fea_buffer[2])

 print("deq_factor")
 print(deq_factor_buffer[0])
 print(deq_factor_buffer[1])
 print(deq_factor_buffer[2])

 print("quantization_scale_w")
 print(quantization_scale_w_buffer[0])
 print(quantization_scale_w_buffer[1])
 print(quantization_scale_w_buffer[2])

 print("quantization_scale_lin")
 print(quantization_scale_l_buffer[0])
 print(quantization_scale_l_buffer[1])
 print(quantization_scale_l_buffer[2])

 print("quantization_scale_fea")
 print(quantization_scale_fea_buffer[0])
 print(quantization_scale_fea_buffer[1])
 print(quantization_scale_fea_buffer[2])
 
 #terminate IP configuration
 if(config.acc==1):
  my_ip.register_map.load_weights = config.load_weights #load the weights first before execution (needed for training)
  my_ip.register_map.gat_mode= 0
  my_ip.register_map.model_offset_1 = model_buffer.physical_address
  my_ip.register_map.scale_fea_offset_1 = scale_fea_buffer.physical_address
  my_ip.register_map.quantization_scale_fea_offset_1 = quantization_scale_fea_buffer.physical_address
  my_ip.register_map.quantization_scale_w_offset_1 = quantization_scale_w_buffer.physical_address
  my_ip.register_map.quantization_scale_lin_offset_1 = quantization_scale_l_buffer.physical_address
  my_ip.register_map.srelu_offset_1 = srelu_buffer.physical_address
  my_ip.register_map.deq_factor_offset_1 = deq_factor_buffer.physical_address
  my_ip.register_map.P_w_offset_1 = P_w_buffer.physical_address
  my_ip.register_map.E1_offset_1 = E_buffer.physical_address
  my_ip.register_map.E2_offset_1 = E_buffer.physical_address
  my_ip.register_map.E3_offset_1 = E_buffer.physical_address
  my_ip.register_map.E4_offset_1 = E_buffer.physical_address
  my_ip.register_map.S1_offset_1 = S_buffer.physical_address
  my_ip.register_map.S2_offset_1 = S_buffer.physical_address
  my_ip.register_map.S3_offset_1 = S_buffer.physical_address
  my_ip.register_map.S4_offset_1 = S_buffer.physical_address
  my_ip.register_map.layer_count=config.instant_layer_count
  my_ip.register_map.ate_m_offset_1 = attention_buffer.physical_address
  my_ip.register_map.B_offset_1 = B_buffer.physical_address
  my_ip.register_map.B2_offset_1 = B2_buffer.physical_address
  my_ip.register_map.rowPtr_fea1_offset_1 = rowPtr_fea_buffer.physical_address
  my_ip.register_map.rowPtr_fea2_offset_1 = rowPtr_fea_buffer.physical_address
  my_ip.register_map.rowPtr_fea3_offset_1 = rowPtr_fea_buffer.physical_address
  my_ip.register_map.rowPtr_fea4_offset_1 = rowPtr_fea_buffer.physical_address
  my_ip.register_map.columnIndex_fea1_offset_1 =columnIndex_fea_buffer.physical_address
  my_ip.register_map.columnIndex_fea2_offset_1 =columnIndex_fea_buffer.physical_address 
  my_ip.register_map.columnIndex_fea3_offset_1 =columnIndex_fea_buffer.physical_address 
  my_ip.register_map.columnIndex_fea4_offset_1 =columnIndex_fea_buffer.physical_address 
  my_ip.register_map.values_fea1_offset_1 = values_fea_buffer.physical_address 
  my_ip.register_map.values_fea2_offset_1 = values_fea_buffer.physical_address 
  my_ip.register_map.values_fea3_offset_1 = values_fea_buffer.physical_address 
  my_ip.register_map.values_fea4_offset_1 = values_fea_buffer.physical_address 
  my_ip.register_map.rowPtr_adj1_offset_1 = rowPtr_adj_buffer.physical_address 
  my_ip.register_map.rowPtr_adj2_offset_1 = rowPtr_adj_buffer.physical_address 
  my_ip.register_map.rowPtr_adj3_offset_1 = rowPtr_adj_buffer.physical_address
  my_ip.register_map.rowPtr_adj4_offset_1 = rowPtr_adj_buffer.physical_address
  my_ip.register_map.columnIndex_adj1_offset_1 = columnIndex_adj_buffer.physical_address 
  my_ip.register_map.columnIndex_adj2_offset_1 = columnIndex_adj_buffer.physical_address
  my_ip.register_map.columnIndex_adj3_offset_1 = columnIndex_adj_buffer.physical_address 
  my_ip.register_map.columnIndex_adj4_offset_1 = columnIndex_adj_buffer.physical_address 
  my_ip.register_map.values_adj1_offset_1 = values_adj_buffer.physical_address
  my_ip.register_map.values_adj2_offset_1 = values_adj_buffer.physical_address
  my_ip.register_map.values_adj3_offset_1 = values_adj_buffer.physical_address
  my_ip.register_map.values_adj4_offset_1 = values_adj_buffer.physical_address
  my_ip.register_map.quantized_multiplier = internal_quantization
  my_ip.register_map.bias_offset_1  = bias_buffer.physical_address
  my_ip.register_map.profiling_offset_1  = profiling_buffer.physical_address
  #load model program....(instructions that define the model layer sequence)
  #NA, terminate, stream, dense, relu, compute_attention, sage, linear

  #gcnconv  0,0,1,0,1,0,0,0
  #gcnconv  0,0,1,1,1,0,0,0
  #linear   0,1,0,1,0,0,0,1  

 if(config.acc==1):
  print("SGRACE hardware loaded and ready!") 
 else:
  print("SGRACE emulation ready!") 
   