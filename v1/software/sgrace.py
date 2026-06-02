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
    def forward(ctx,my_ip, self,adj,nnz_adj,input, weights,weights_linear,weights_l2,weights_l3,attention,attention_l2,attention_l3,out_features,dropout,relu):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        st = self.state
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

         #print("st.deq_o and active layer")
         #print(st.deq_o)
         #print(st.layern)



         #if (st.layern == 1):
          #my_ip.register_map.scale_fea = st.scale_fea #2 #scale fea

          #int32bits = np.asarray(st.deq_o, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.deq_factor = int32bits
          #deq_factor = int32bits

          #qsf = 1/st.f_s
          #int32bits = np.asarray(qsf, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_fea = int32bits
        
        
          #qsw = 1/st.w_s
          #int32bits = np.asarray(qsw, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_w = int32bits

   

         #elif (st.layern == 2):
          #my_ip.register_map.scale_fea = st.scale_fea2 #2 #scale fea
          #int32bits = np.asarray(st.deq_o2, dtype=np.float32).view(np.int32).item() 
          #deq_factor = int32bits
          #my_ip.register_map.deq_factor = int32bits
          #qsf = 1/st.f_s2
          #int32bits = np.asarray(qsf, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_fea = int32bits
          #qsw = 1/st.w_s2
          #int32bits = np.asarray(qsw, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_w = int32bits   
    
         #else:
          #int32bits = np.asarray(st.deq_ol, dtype=np.float32).view(np.int32).item() 
          #deq_factor = int32bits
          #my_ip.register_map.deq_factor = int32bits
          #qsf = 1/st.f_sl
          #int32bits = np.asarray(qsf, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_fea = int32bits
          #qsw = 1/st.w_sl
          #int32bits = np.asarray(qsw, dtype=np.float32).view(np.int32).item() 
          #my_ip.register_map.quantization_scale_w = int32bits 

         #print("deq_factor")
         #print(deq_factor) 
         #print("qsf")
         #print(qsf) 
         #print("qsw")
         #print(qsw)  
            
         qsa = 1/st.a_s

         #print("quantization_scale_adj")
         #print(qsa)
         #print("st.a_s")
         #print(st.a_s)
         int32bits = np.asarray(qsa, dtype=np.float32).view(np.int32).item() 
         my_ip.register_map.quantization_scale_adj = int32bits

        
         
         if(config.instant_layer_count==1):

          my_ip.register_map.rowPtr_adj1_offset_1 = st.rowPtr_adj_buffer.physical_address 
          my_ip.register_map.rowPtr_adj2_offset_1 = st.rowPtr_adj_buffer.physical_address 
          my_ip.register_map.rowPtr_adj3_offset_1 = st.rowPtr_adj_buffer.physical_address
          my_ip.register_map.rowPtr_adj4_offset_1 = st.rowPtr_adj_buffer.physical_address

          my_ip.register_map.columnIndex_adj1_offset_1 = st.columnIndex_adj_buffer.physical_address 
          my_ip.register_map.columnIndex_adj2_offset_1 = st.columnIndex_adj_buffer.physical_address
          my_ip.register_map.columnIndex_adj3_offset_1 = st.columnIndex_adj_buffer.physical_address 
          my_ip.register_map.columnIndex_adj4_offset_1 = st.columnIndex_adj_buffer.physical_address 
          my_ip.register_map.values_adj1_offset_1 = st.values_adj_buffer.physical_address
          my_ip.register_map.values_adj2_offset_1 = st.values_adj_buffer.physical_address
          my_ip.register_map.values_adj3_offset_1 = st.values_adj_buffer.physical_address
          my_ip.register_map.values_adj4_offset_1 = st.values_adj_buffer.physical_address


          my_ip.register_map.N_adj=input.shape[0]
          my_ip.register_map.M_adj=input.shape[0]
          my_ip.register_map.M_fea=input.shape[1]
          my_ip.register_map.P_w=weights.shape[1]

          #print('use my_ip.register_map.N_adj')
          #print("P_w")
          #print(weights.shape[1])

        
        
       
          my_ip.register_map.E1_offset_1 = st.E_buffer.physical_address
          my_ip.register_map.S1_offset_1 = st.S_buffer.physical_address
   
          my_ip.register_map.D1_offset_1 = st.D_buffer.physical_address
          my_ip.register_map.D2_offset_1 = st.D_buffer.physical_address
          my_ip.register_map.D3_offset_1 = st.D_buffer.physical_address
          my_ip.register_map.D4_offset_1 = st.D_buffer.physical_address

          #print('use st.rowPtr_fea_buffer.physical_address')
          #print(st.rowPtr_fea_buffer.physical_address)
  
          my_ip.register_map.rowPtr_fea1_offset_1 = st.rowPtr_fea_buffer.physical_address
          my_ip.register_map.rowPtr_fea2_offset_1 = st.rowPtr_fea_buffer.physical_address
          my_ip.register_map.rowPtr_fea3_offset_1 = st.rowPtr_fea_buffer.physical_address
          my_ip.register_map.rowPtr_fea4_offset_1 = st.rowPtr_fea_buffer.physical_address

          my_ip.register_map.columnIndex_fea1_offset_1 =st.columnIndex_fea_buffer.physical_address
          my_ip.register_map.columnIndex_fea2_offset_1 =st.columnIndex_fea_buffer.physical_address 
          my_ip.register_map.columnIndex_fea3_offset_1 =st.columnIndex_fea_buffer.physical_address 
          my_ip.register_map.columnIndex_fea4_offset_1 =st.columnIndex_fea_buffer.physical_address 
          my_ip.register_map.values_fea1_offset_1 = st.values_fea_buffer.physical_address
          my_ip.register_map.values_fea2_offset_1 = st.values_fea_buffer.physical_address
          my_ip.register_map.values_fea3_offset_1 = st.values_fea_buffer.physical_address
          my_ip.register_map.values_fea4_offset_1 = st.values_fea_buffer.physical_address
         else:
          #print('st.layern')
          #print(st.layern)
          if(st.layern==1):
           my_ip.register_map.rowPtr_adj1_offset_1 = st.rowPtr_adj_buffer.physical_address 
           my_ip.register_map.rowPtr_adj2_offset_1 = st.rowPtr_adj_buffer.physical_address 
           my_ip.register_map.rowPtr_adj3_offset_1 = st.rowPtr_adj_buffer.physical_address
           my_ip.register_map.rowPtr_adj4_offset_1 = st.rowPtr_adj_buffer.physical_address

           my_ip.register_map.columnIndex_adj1_offset_1 = st.columnIndex_adj_buffer.physical_address 
           my_ip.register_map.columnIndex_adj2_offset_1 = st.columnIndex_adj_buffer.physical_address
           my_ip.register_map.columnIndex_adj3_offset_1 = st.columnIndex_adj_buffer.physical_address 
           my_ip.register_map.columnIndex_adj4_offset_1 = st.columnIndex_adj_buffer.physical_address 
           my_ip.register_map.values_adj1_offset_1 = st.values_adj_buffer.physical_address
           my_ip.register_map.values_adj2_offset_1 = st.values_adj_buffer.physical_address
           my_ip.register_map.values_adj3_offset_1 = st.values_adj_buffer.physical_address
           my_ip.register_map.values_adj4_offset_1 = st.values_adj_buffer.physical_address


           my_ip.register_map.N_adj=input.shape[0]
           my_ip.register_map.M_adj=input.shape[0]
           my_ip.register_map.M_fea=input.shape[1]
           #my_ip.register_map.P_w=weights.shape[1]

           #print('input shape 1')
           #print(input.shape[1])
           #print(weights.shape[1])

        
        
       
           my_ip.register_map.E1_offset_1 = st.E_buffer.physical_address
           my_ip.register_map.S1_offset_1 = st.S_buffer.physical_address
   
           my_ip.register_map.D1_offset_1 = st.D_buffer.physical_address
           my_ip.register_map.D2_offset_1 = st.D_buffer.physical_address
           my_ip.register_map.D3_offset_1 = st.D_buffer.physical_address
           my_ip.register_map.D4_offset_1 = st.D_buffer.physical_address

           #print('use st.rowPtr_fea_buffer.physical_address')
           #print(st.rowPtr_fea_buffer.physical_address)
  
           my_ip.register_map.rowPtr_fea1_offset_1 = st.rowPtr_fea_buffer.physical_address
           my_ip.register_map.rowPtr_fea2_offset_1 = st.rowPtr_fea_buffer.physical_address
           my_ip.register_map.rowPtr_fea3_offset_1 = st.rowPtr_fea_buffer.physical_address
           my_ip.register_map.rowPtr_fea4_offset_1 = st.rowPtr_fea_buffer.physical_address

           my_ip.register_map.columnIndex_fea1_offset_1 =st.columnIndex_fea_buffer.physical_address
           my_ip.register_map.columnIndex_fea2_offset_1 =st.columnIndex_fea_buffer.physical_address 
           my_ip.register_map.columnIndex_fea3_offset_1 =st.columnIndex_fea_buffer.physical_address 
           my_ip.register_map.columnIndex_fea4_offset_1 =st.columnIndex_fea_buffer.physical_address 
           my_ip.register_map.values_fea1_offset_1 = st.values_fea_buffer.physical_address
           my_ip.register_map.values_fea2_offset_1 = st.values_fea_buffer.physical_address
           my_ip.register_map.values_fea3_offset_1 = st.values_fea_buffer.physical_address
           my_ip.register_map.values_fea4_offset_1 = st.values_fea_buffer.physical_address


           if (config.profiling == 1):   
            print('Register time: {:.5f}ms'.format(1000/1*(time.time() - rmult)))
        
           my_ip.register_map.B_offset_1 = st.B_buffer.physical_address
           my_ip.register_map.B2_offset_1 = st.B2_buffer.physical_address
         
         if (config.profiling == 1):
          amult = time.time()
         
         support_linear = torch.transpose(weights_linear,0,1)
         support = torch.transpose(weights,0,1)
         support_l2 = torch.transpose(weights_l2,0,1)
         support_l3 = torch.transpose(weights_l3,0,1)
         #support_l4 = torch.transpose(weights_l4,0,1)
         #print(weights_l2.shape)
         #st.B_buffer[0:(weights.shape[0]*weights.shape[1])] = torch.transpose(weights,0,1).reshape(1, (weights.shape[0]*weights.shape[1]))
         if (config.profiling == 1):   
          print('Transpose time: {:.5f}ms'.format(1000/1*(time.time() - amult)))
  
        
         if(config.min_output == 0):
          print('st.values_fea_buffer')
          print(st.values_fea_buffer[0:100])
          print('st.columnIndex_fea_buffer')
          print(st.columnIndex_fea_buffer[0:100])
          print('st.rowPtr_fea_buffer')
          print(st.rowPtr_fea_buffer[0:100])
         
         attention_q=attention.reshape(1,(attention.shape[0]*attention.shape[1]))
         #attention_q_l2=attention_l2.reshape(1,(attention.shape[0]*attention.shape[1]))
         #attention_q_l3=attention_l3.reshape(1,(attention.shape[0]*attention.shape[1]))

         support_pynq = support.data.numpy() #OJO USE TRANSPOSE
         support_pynq_linear = support_linear.data.numpy()
         support_pynq_l2 = support_l2.data.numpy() #OJO USE TRANSPOSE
         support_pynq_l3 = support_l3.data.numpy() #OJO USE TRANSPOSE
         #support_pynq_l4 = support_l4.data.numpy() #OJO USE TRANSPOSE
       
  
    
          
         if(config.show_max_min==1):
          print("active layer: ",st.layern)   
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



         if(config.write_file == 1 and st.layern == 3):   
              support_pynq_linear_q_t = np.transpose(support_pynq_linear_q)
              df = pd.DataFrame(support_pynq_linear_q_t) #convert to a dataframe
              df.to_csv("./linear_weights.txt",index=False,header=False) #save to file    
              sys.exit()

         support_pynq_linear_q = support_pynq_linear_q.reshape(1, (support_pynq_linear_q.shape[0]*support_pynq_linear_q.shape[1]))


         #print("weights l2")
         #print(support_pynq_q_l2)
    
         if(config.core_count == 1):
           if(config.instant_layer_count == 1):
             st.B_buffer[0:(weights.shape[0]*weights.shape[1])] = support_pynq_q.astype(config.float_type)
             st.B2_buffer[0:(support_pynq_linear_q.shape[0]*support_pynq_linear_q.shape[1])] = support_pynq_linear_q.astype(config.float_type)
              #print("st.B2_buffer")
              #print(st.B2_buffer[0:10])
           else:
            if(st.layern==1):
             st.B_buffer[0:(weights.shape[0]*weights.shape[1])] = support_pynq_q.astype(config.float_type)
             support_pynq_q_l2 = support_pynq_q_l2.reshape(1, (weights_l2.shape[0]*weights_l2.shape[1]))
             st.weight_shift = (weights.shape[0]*weights.shape[1])
             st.B_buffer[st.weight_shift:(st.weight_shift+weights_l2.shape[0]*weights_l2.shape[1])] = support_pynq_q_l2.astype(config.float_type)
             support_pynq_q_l3 = support_pynq_q_l3.reshape(1, (weights_l3.shape[0]*weights_l3.shape[1]))
             st.weight_shift = st.weight_shift+(weights_l2.shape[0]*weights_l2.shape[1])
             #st.B_buffer[st.weight_shift:(st.weight_shift+weights_l3.shape[0]*weights_l3.shape[1])] = support_pynq_q_l3.astype(config.float_type)
             #print("pynq l3")
             st.B2_buffer[st.weight_shift:(st.weight_shift+weights_l3.shape[0]*weights_l3.shape[1])] = support_pynq_q_l3.astype(config.float_type)
             #support_pynq_q_l4 = support_pynq_q_l4.reshape(1, (weights_l4.shape[0]*weights_l4.shape[1]))
             #st.weight_shift = st.weight_shift+(weights_l3.shape[0]*weights_l3.shape[1])
             #print("pynq l4")
             #print(support_pynq_q_l4[0:10])
             #st.B2_buffer[st.weight_shift:(st.weight_shift+weights_l4.shape[0]*weights_l4.shape[1])] = support_pynq_q_l4.astype(config.float_type)
             #st.B2_buffer[0:(weights_l3.shape[0]*weights_l3.shape[1])] = support_pynq_q_l3.astype(config.float_type)

         if (config.min_output == 0):
         #if (self.linear == 1):
           print("B_Buffer")
           print(st.B_buffer[0:10])
         #print("B2_Buffer linear")
         #print(st.B2_buffer[0:32])

         if(self.compute_attention == 1):
          if(config.instant_layer_count == 1):
           attention_q = attention_q.numpy()
           st.attention_buffer[0:(attention.shape[0]*attention.shape[1])] = attention_q.astype(config.float_type)
          else:
           if(st.layern==1):
             attention_q = attention_q.numpy()
             st.attention_buffer[0:(attention.shape[0]*attention.shape[1])] = attention_q.astype(config.float_type)
             #global attention_shift
             #attention_q_l2 = attention_q_l2.numpy();
             #attention_shift = (attention.shape[0]*attention.shape[1])
             #st.attention_buffer[attention_shift:(attention_shift+attention.shape[0]*weights_l2.shape[1])] = attention_q_l2.astype(config.float_type)
           
         #if(config.show_max_min==1):
         # print("max/min quantized weights")
         # print(np.max(support_pynq_q))
         # print(np.min(support_pynq_q))
        
         #global B_size
         #B_size = (weights.shape[0]*weights.shape[1])
          
         my_ip.register_map.quantized_multiplier = st.internal_quantization #apply internal quantization
         
         #print("st.layern")
         #print(st.layern)

         if (config.profiling == 2):
          amult = time.time()
          for _ in range(1):
           my_ip.register_map.CTRL.AP_START = 1
           while my_ip.register_map.CTRL.AP_IDLE == 0:
            pass
          dmult =  time.time()
         else:
          #print('config.instant_layer_count ',config.instant_layer_count)
          if(config.core_count == 1):
           if(config.instant_layer_count == 1):
             #if (config.min_output == 0):
             #print('start core 1 to process 1 layer')
             #print('model buffer is')
             #print(st.model_buffer[0])
             #print(st.model_buffer[1])
             amult = time.time()
             next_inst_addr = st.model_buffer.physical_address+(st.layern-1) 
             next_P_w_addr = st.P_w_buffer.physical_address+(st.layern-1) 
             next_quantization_scale_fea_addr = st.quantization_scale_fea_buffer.physical_address+4*(st.layern-1) 
             next_quantization_scale_w_addr = st.quantization_scale_w_buffer.physical_address+4*(st.layern-1)
             next_quantization_scale_l_addr = st.quantization_scale_l_buffer.physical_address+4*(st.layern-1)
             next_scale_fea_addr = st.scale_fea_buffer.physical_address+(st.layern-1)
             next_deq_factor_addr = st.deq_factor_buffer.physical_address+4*(st.layern-1)


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

             my_ip.register_map.CTRL.AP_START = 1
             while my_ip.register_map.CTRL.AP_IDLE == 0:
              pass
             dmult =  time.time()
             if (config.profiling == 1):   
              print('Accelerator forward kernel 1-layer time: {:.5f}ms'.format(1000/1*(dmult - amult)))
             if (config.min_output == 0):
              print('done core 1')  
           else:  
            if(st.layern == 1):
             if (config.min_output == 0):
              print('start core 1 to process layer count: ', config.instant_layer_count)
             amult =  time.time()
             my_ip.register_map.CTRL.AP_START = 1
             while my_ip.register_map.CTRL.AP_IDLE == 0:
              pass
             dmult =  time.time()
             if (config.profiling == 1):   
              print('Accelerator forward kernel n-layer time: {:.5f}ms'.format(1000/1*(dmult - amult)))
             if (config.min_output == 0):
              print('done core 1') 
            else:  
             if (config.min_output == 0): 
              print('Nothing to do in layer>1')

      

         if(config.instant_layer_count == 1): 
          #print("active layer: ", st.layern)  

     
          
          if(self.compute_attention==1):
           output_e_val = st.E_buffer[0:nnz_adj].astype(config.float_type)
           output_s_val = st.S_buffer[0:nnz_adj].astype(config.float_type) #you should use this
        


          max_fea = my_ip.register_map.max_fea
          if(config.min_output == 0):
           print("MAX FEA INT GAT")
           print(max_fea)
           print(float(max_fea)/(2**st.frac_bits_o))

          max_fea_float = float(max_fea)/(2**st.frac_bits_o)
          if (st.layern == 1):
           if(max_fea_float > st.cur_max_fea):
            st.cur_max_fea = max_fea_float
          else:
           if(max_fea_float > st.cur_max_fea2):
            st.cur_max_fea2 = max_fea_float
        
          #if(weights.shape[1]!=16):
          # output_acc = st.D_buffer[0:input.shape[0]*16]
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
          output_acc = st.D_buffer[0:input.shape[0]*weights.shape[1]]
          output_acc = output_acc.reshape(input.shape[0],weights.shape[1]) 


          write_output = 0
 
          if(config.show_sparsity==1):
           zeros = np.count_nonzero(output_acc==0)
           print("sparsity from layer:",st.layern,"zeros:",zeros/(input.shape[0]*weights.shape[1]))  
           sparsity_is = zeros/(input.shape[0]*weights.shape[1])

          if(write_output == 1):   
           #support_pynq_q_t = np.transpose(support_pynq_q)
           df = pd.DataFrame(output_acc) #convert to a dataframe
           df.to_csv("./output.txt",index=False,header=False) #save to file    
           #if(st.layern==2):
           sys.exit()
          #print("Sample Output")
          #print(output_acc[0])
    
          if(st.layern<config.total_layer_count):
           st.layern+=1
          else:
           st.layern=1 

          #get sparse matrix for e and softmax
          if(self.compute_attention==1):
           rindex = st.rowPtr_adj_buffer[0:nnz_adj]
           cindex =  st.columnIndex_adj_buffer[0:nnz_adj] 
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
          if(st.layern==1): #if layer 1 we return the input so layer 2 gets the right dimensions.

           output_acc = st.D_buffer[0:input.shape[0]*weights.shape[1]]
           #zeros = np.count_nonzero(output_acc==0)
           #print("sparsity into layer 2 zeros ", zeros/(input.shape[0]*weights.shape[1]))
           #print("output_acc")
           #print(output_acc)
          elif(st.layern==2):
           output_acc = st.D_buffer[0:input.shape[0]*weights.shape[1]]
           #output_acc = st.D_buffer[input.shape[0]*weights.shape[1]:2*input.shape[0]*weights.shape[1]]
           #zeros = np.count_nonzero(output_acc==0)
           #print("sparsity into layer 3 zeros ", zeros/(input.shape[0]*weights.shape[1]))
           #print(output_acc)
          #elif(st.layern==3):
          # output_acc = st.D_buffer[2*input.shape[0]*weights.shape[2]:3*input.shape[0]*weights.shape[2]]
          # zeros = np.count_nonzero(output_acc==0)
          # print("sparsity into layer 4 zeros ", zeros/(input.shape[0]*weights.shape[1]))
          # #print(output_acc)
          else:
           output_acc = st.D_buffer[0:input.shape[0]*weights.shape[1]]
           #output_acc = st.D_buffer[2*input.shape[0]*config.hidden_channels:2*input.shape[0]*config.hidden_channels+input.shape[0]*weights.shape[1]]
           #print(output_acc)
           
          # output_acc = output_acc.reshape(input.shape[0],weights.shape[1]) 
          # output_acc = torch.from_numpy(output_acc)       
          # output_acc = output_acc.float()
          # st.layern+=1
           
          #print("output acc 1")
          #print(output_acc)
          #return output_acc
          #elif(st.layern==2):
          # st.layern=1

          #output_acc = st.D_buffer[(st.layern-1)*input.shape[0]*weights.shape[1]:st.layern*input.shape[0]*weights.shape[1]]

          if(st.layern<config.total_layer_count):
           st.layern+=1
          else:
           st.layern=1
           #print("output acc")
          #print("st.layern here")
          #print(st.layern)
          
          if(self.compute_attention==1):
           output_e_val = st.E_buffer[0:nnz_adj].astype(config.float_type)
           output_s_val = st.S_buffer[0:nnz_adj].astype(config.float_type) #you should use this
        


          max_fea = my_ip.register_map.max_fea
          if(config.min_output == 0):
            print("MAX FEA INT GAT")
            print(max_fea)
            print(float(max_fea)/(2**st.frac_bits_o))

          max_fea_float = float(max_fea)/(2**st.frac_bits_o)
          if (st.layern == 1):
            if(max_fea_float > st.cur_max_fea):
             st.cur_max_fea = max_fea_float
          else:
            if(max_fea_float > st.cur_max_fea2):
             st.cur_max_fea2 = max_fea_float
        

          #print("Output_acc")
          #print("shapes")
          #print(input.shape[0])
          #print(weights.shape[1])

          output_acc = output_acc.reshape(input.shape[0],weights.shape[1]) 
          #get sparse matrix for e and softmax
          if(self.compute_attention==1):
            rindex = st.rowPtr_adj_buffer[0:nnz_adj]
            cindex =  st.columnIndex_adj_buffer[0:nnz_adj] 
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
          # st.layern=1
          # output_acc = st.D_buffer[2*input.shape[0]*weights.shape[1]:3*input.shape[0]*weights.shape[1]]
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

          #print("No accelerator active layer: ",st.layern)

          input = input.float()
          torch.max(input)

          #monitor input values to calibrate thresholds
          if (st.layern==3): 
           max_input = torch.max(input)
           min_input = torch.min(input)
           if (max_input > st.global_max_input):
            st.global_max_input = max_input
           if (min_input < st.global_min_input):
            st.global_min_input = min_input

          if(config.fake_quantization==1):
           #print("input")
           print("active layer: ",st.layern) 
           if (st.layern==1): 
            input_q = quantization_ufbits(input, st.f_s, st.f_z, config.w_qbits)
            if(st.layern<config.total_layer_count):
             st.layern=2
            else:
             st.layern=1
           elif(st.layern==2): 
            input_q = quantization_ufbits(input, st.f_s2, st.f_z2, config.w_qbits)
            if(st.layern<config.total_layer_count):
             st.layern=3
            else:
             st.layern=1 
           else: 
            input_q = quantization_ufbits(input, st.f_sl, st.f_zl, config.w_qbitsl)
            st.layern=1
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
           weights_q = quantization_fbits(weights, st.w_s, st.w_z,  config.w_qbits) 
           weights_linear_q = quantization_fbits(weights_linear, st.w_sl, st.w_zl,  config.w_qbitsl) 
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
            #Wh = float_to_fix(Wh,(st.internal_quantization-1))
            Wh = Wh/(2**st.scale_fea) 
            Wh = float_to_fix(Wh,(st.internal_quantization-1))
            #print("Internal GNN FIFO data")
            #print(Wh[0:10])
            #Wh = torch.round(Wh, decimals = (st.internal_quantization-1))
            linear_layer = linear_layer/(2**st.scale_fea) 
            linear_layer = float_to_fix(linear_layer,(st.internal_quantization-1))
            st.a_min = -(2**st.internal_quantization-1)/(2**st.internal_quantization)
            st.a_max = (2**st.internal_quantization-1)/(2**st.internal_quantization)
            #print(st.a_min)
            #print(st.a_max)
            #Wh = np.clip(Wh, st.a_min=-0.9921875, st.a_max=0.9921875)
            #Wh = np.clip(Wh, st.a_min=-0.875, st.a_max=0.875)
            #Wh = np.clip(Wh, st.a_min=-0.99999999, st.a_max=0.99999999)
            Wh = torch.clip(Wh, min=st.a_min, max=st.a_max)
            #Wh = torch.round(Wh, decimals = (st.internal_quantization-1))
            linear_layer = torch.clip(linear_layer, min=st.a_min, max=st.a_max)
            #linear_layer = torch.round(linear_layer, decimals = (st.internal_quantization-1))

            #print(Wh)
           #print("attention")
           #print(attention[i])
     
           adj_d = adj.to_dense()  
           if(config.fake_quantization==1):
            attention = quantization_fbits(attention, st.w_s, st.w_z,  config.w_qbits) 
            #print("quantize adj")
            adj_d = quantization_ufbits(adj_d, st.a_s, st.a_z,  config.w_qbits) 
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
            output_cpu = output_cpu*st.deq_o
            #linear_layer = linear_layer*st.deq_o

           #RELU
           if(relu==1):
            output_cpu = torch.where(output_cpu > 0, output_cpu, 0)
           
          #LINEAR ONLY
          else:
           print("output is linear")
           #print(linear_layer[0:10])
           #linear_layer = float_to_fix(linear_layer,(st.internal_quantization-1))

           tmult = time.time() 

           linear_layer = torch.mm(input_q, weights_linear_q) # h.shape: (N, in_features), Wh.shape: (N, out_features)

           #print(input_q[0:10]) 
           #print(weights_linear_q[0:10]) 
           #print(linear_layer[0:10]) 
           if (config.profiling == 1):   
            print('cpu linear kernel time: {:.5f}ms'.format(1000/1*(time.time() - tmult)))
           if (config.fake_quantization == 1):
            linear_layer = linear_layer/(2**st.scale_feal)
            #torch.set_printoptions(precision=8)
            #print(linear_layer[0:10])
            linear_layer = float_to_fix(linear_layer,(st.internal_quantization-1))
            #print("Internal LINEAR FIFO data")
            #print(linear_layer[0:10])
            #print(linear_layer[0:10])
            #print("test")
            #test_data = float_to_fix(0.0494,7)
            #print(test_data)
            #linear_layer = torch.round(linear_layer, decimals = (st.internal_quantization-1))
            st.a_min = -(2**st.internal_quantization-1)/(2**st.internal_quantization)
            st.a_max = (2**st.internal_quantization-1)/(2**st.internal_quantization)
            linear_layer = torch.clip(linear_layer, min=st.a_min, max=st.a_max)
            #linear_layer = torch.round(linear_layer, decimals = (st.internal_quantization-1))
            #linear_layer = torch.round(linear_layer, decimals = 3)
            #print(linear_layer[0:10])
           
           output_cpu = linear_layer 
           if (config.fake_quantization==1):
            output_cpu = output_cpu*st.deq_ol




          #print("output_cpu")
          #print(output_cpu[0:10])
          #print("st.deq_o")
          #print(st.deq_o)

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
        st  = SGRACEState.get()
        my_ip = st.my_ip

        # ── hardware backward (config.accb == 1) ─────────────────────────────
        if config.accb == 1:

            adj, input, weights, e, attentions, output = ctx.saved_tensors
            compute_attention = ctx.compute_attention
            something = grad_adj = grad_attention = None
            go_qbits  = 8
            w_qbits   = config.w_qbits

            # --- grad_weights = input.T @ adj @ grad_output ---
            # Set up gemm_mode=2 (dense×sparse): adj_loop→input.T, fea_loop→adj
            my_ip.register_map.gemm_mode = 2
            my_ip.register_map.relu      = 0

            input_np = input.numpy()                              # avoid repeated .t() alloc
            input_t  = input_np.T
            N_in, M_in = input_t.shape
            my_ip.register_map.N_adj = N_in
            my_ip.register_map.M_adj = M_in
            my_ip.register_map.M_fea = adj.shape[1]
            my_ip.register_map.P_w   = grad_output.shape[1]

            # Quantise input.T → values_fea_buffer (reused as adj_loop source)
            inp_flat = input_t.reshape(1, N_in * M_in)
            st.values_fea_buffer[:N_in * M_in] = (
                quantization_uqbits(inp_flat, st.f_s, st.f_z, w_qbits))
            my_ip.register_map.values_adj1_offset_1 = st.values_fea_buffer.physical_address
            my_ip.register_map.values_adj2_offset_1 = st.values_fea_buffer.physical_address
            my_ip.register_map.values_adj3_offset_1 = st.values_fea_buffer.physical_address
            my_ip.register_map.values_adj4_offset_1 = st.values_fea_buffer.physical_address

            # Point fea_loop at the adj COO data already in adj buffers
            my_ip.register_map.values_fea1_offset_1      = st.values_adj_buffer.physical_address
            my_ip.register_map.values_fea2_offset_1      = st.values_adj_buffer.physical_address
            my_ip.register_map.values_fea3_offset_1      = st.values_adj_buffer.physical_address
            my_ip.register_map.values_fea4_offset_1      = st.values_adj_buffer.physical_address
            my_ip.register_map.rowPtr_fea1_offset_1      = st.rowPtr_adj_buffer.physical_address
            my_ip.register_map.rowPtr_fea2_offset_1      = st.rowPtr_adj_buffer.physical_address
            my_ip.register_map.rowPtr_fea3_offset_1      = st.rowPtr_adj_buffer.physical_address
            my_ip.register_map.rowPtr_fea4_offset_1      = st.rowPtr_adj_buffer.physical_address
            my_ip.register_map.columnIndex_fea1_offset_1 = st.columnIndex_adj_buffer.physical_address
            my_ip.register_map.columnIndex_fea2_offset_1 = st.columnIndex_adj_buffer.physical_address
            my_ip.register_map.columnIndex_fea3_offset_1 = st.columnIndex_adj_buffer.physical_address
            my_ip.register_map.columnIndex_fea4_offset_1 = st.columnIndex_adj_buffer.physical_address

            # Quantise grad_output.T → B_buffer (weight input for grad_weights kernel)
            go_np  = grad_output.numpy()
            go_t   = go_np.T
            go_flat = go_t.reshape(1, go_np.shape[0] * go_np.shape[1])
            st.B_buffer[:go_flat.size] = quantization_qbits(go_flat, st.go_s, st.go_z, go_qbits)

            if config.profiling == 1:
                amult = time.time()
            my_ip.register_map.CTRL.AP_START = 1
            while my_ip.register_map.CTRL.AP_IDLE == 0:
                pass
            if config.profiling == 1:
                print("acc backward grad_weights: {:.5f}s".format(time.time() - amult))

            n_gw = N_in * grad_output.shape[1]
            grad_weights = torch.from_numpy(
                (st.D_buffer[:n_gw] * st.deq_gw / (2 ** st.frac_bits_o))
                .reshape(N_in, grad_output.shape[1]).copy()).float()

            # --- grad_input = adj @ grad_output @ weights.T ---
            # Set up gemm_mode=1 (sparse×dense): adj_loop→adj, fea_loop→grad_output
            my_ip.register_map.gemm_mode     = 1
            my_ip.register_map.relu          = 0
            my_ip.register_map.gat_mode      = compute_attention
            my_ip.register_map.N_adj         = adj.shape[0]
            my_ip.register_map.M_adj         = adj.shape[1]
            my_ip.register_map.M_fea         = grad_output.shape[1]
            my_ip.register_map.P_w           = weights.shape[0]

            my_ip.register_map.values_adj1_offset_1      = st.values_adj_buffer.physical_address
            my_ip.register_map.values_adj2_offset_1      = st.values_adj_buffer.physical_address
            my_ip.register_map.values_adj3_offset_1      = st.values_adj_buffer.physical_address
            my_ip.register_map.values_adj4_offset_1      = st.values_adj_buffer.physical_address
            my_ip.register_map.rowPtr_adj1_offset_1      = st.rowPtr_adj_buffer.physical_address
            my_ip.register_map.rowPtr_adj2_offset_1      = st.rowPtr_adj_buffer.physical_address
            my_ip.register_map.rowPtr_adj3_offset_1      = st.rowPtr_adj_buffer.physical_address
            my_ip.register_map.rowPtr_adj4_offset_1      = st.rowPtr_adj_buffer.physical_address
            my_ip.register_map.columnIndex_adj1_offset_1 = st.columnIndex_adj_buffer.physical_address
            my_ip.register_map.columnIndex_adj2_offset_1 = st.columnIndex_adj_buffer.physical_address
            my_ip.register_map.columnIndex_adj3_offset_1 = st.columnIndex_adj_buffer.physical_address
            my_ip.register_map.columnIndex_adj4_offset_1 = st.columnIndex_adj_buffer.physical_address

            # Quantise grad_output → values_fea_buffer
            go_flat2 = go_np.reshape(1, go_np.shape[0] * go_np.shape[1])
            st.values_fea_buffer[:go_flat2.size] = (
                quantization_uqbits(go_flat2, st.go_s, st.go_z, go_qbits))
            my_ip.register_map.values_fea1_offset_1 = st.values_fea_buffer.physical_address
            my_ip.register_map.values_fea2_offset_1 = st.values_fea_buffer.physical_address
            my_ip.register_map.values_fea3_offset_1 = st.values_fea_buffer.physical_address
            my_ip.register_map.values_fea4_offset_1 = st.values_fea_buffer.physical_address

            # Quantise weights → B_buffer (no transpose needed: transpose of transpose = identity)
            w_np    = weights.numpy()
            w_flat  = w_np.reshape(1, w_np.shape[0] * w_np.shape[1])
            st.B_buffer[:w_flat.size] = quantization_qbits(w_flat, st.w_s, st.w_z, w_qbits)

            if config.profiling == 1:
                amult = time.time()
            my_ip.register_map.CTRL.AP_START = 1
            while my_ip.register_map.CTRL.AP_IDLE == 0:
                pass
            if config.profiling == 1:
                print("acc backward grad_input: {:.5f}s".format(time.time() - amult))

            n_gi = adj.shape[0] * weights.shape[0]
            grad_input = torch.from_numpy(
                (st.D_buffer[:n_gi] * st.deq_gi / (2 ** st.frac_bits_o))
                .reshape(adj.shape[0], weights.shape[0]).copy()).float()

            return (something, something, something, grad_adj, something,
                    grad_input, grad_weights, grad_attention,
                    something, something, something, something)

        # ── CPU backward (config.accb == 0) ──────────────────────────────────
        else:
            adj, input, weights, weights_linear, e, attentions, output = ctx.saved_tensors
            alpha             = ctx.alpha
            linear            = ctx.linear
            sage              = ctx.sage
            compute_attention = ctx.compute_attention

            something = grad_adj = grad_input = grad_weights = grad_weights_linear = grad_attention = None

            input_t          = input.t()
            weights_t        = weights.t()
            weights_linear_t = weights_linear.t()

            if config.profiling == 1:
                tmult = time.time()

            # ── attention gradient ────────────────────────────────────────────
            if compute_attention == 1:
                # support shape: (out, N) — projected features transposed.
                # For GAT:         weights=(in,out), input=(N,in)  → W^T·X^T = (out,in)·(in,N) = (out,N)
                # For Transformer: weights=W_V=(in,out), input=X_V=(N,out) → use input_t=(out,N) directly
                if weights.shape[0] == input.shape[1]:
                    # Standard GAT: weights is (in, out), input is (N, in)
                    support = torch.mm(weights_t, input_t)           # (out, N)
                else:
                    # Transformer: input is already projected X_V=(N,out), use its transpose
                    support = input_t                                 # (out, N)
                softmax_out = torch.mm(grad_output, support)          # (N, N)

                # Vectorised softmax Jacobian: s_ij * (delta_ij - s_ik) * g_kj
                dx  = (e > 0).float() + alpha * (e <= 0).float()      # LeakyReLU mask
                # dx * (diag(a) - a aᵀ) @ softmax_out, sparse-masked
                s   = (attentions * softmax_out).sum(dim=-1, keepdim=True)
                soft_gradient = dx * attentions * (softmax_out - s)

                # Mask to graph edges only
                adj_d         = adj.to_dense()
                soft_gradient = torch.where(adj_d > 0, soft_gradient,
                                            torch.zeros_like(soft_gradient))

                # grad_attention = [W^T X^T soft_grad 1 ; 1^T soft_grad^T X W]
                support1        = torch.mm(support, soft_gradient)          # (out, N)
                grad_attention1 = support1.sum(dim=-1)                      # (out,)
                # For GAT: input=(N,in), weights=(in,out) → input@weights=(N,out)
                # For Transformer: input=X_V=(N,out) already projected — use directly
                xw = torch.mm(input, weights) if weights.shape[0] == input.shape[1] else input
                support2        = torch.mm(soft_gradient, xw)               # (N, out)
                grad_attention2 = support2.sum(dim=0)                       # (out,)
                output_attention = torch.cat([grad_attention1,
                                              grad_attention2]).unsqueeze(1)
            else:
                output_attention = torch.zeros(
                    weights.shape[1] * 2, 1, device=config.device)

            # ── grad_input and grad_weights ───────────────────────────────────
            # For GAT:         weights=(in,out), grad w.r.t. input=(N,in) → grad_output @ W^T
            # For Transformer: input=X_V=(N,out), grad w.r.t. X_V=(N,out) → no W multiplication
            is_transformer = (weights.shape[0] != input.shape[1])

            if not is_transformer:
                support = torch.mm(grad_output, weights_t)           # (N, in)
            else:
                support = grad_output                                 # (N, out)

            if compute_attention == 1:
                output_input = torch.mm(attentions, support)
                support      = torch.mm(attentions, grad_output)
            else:
                if sage == 1:
                    output_input = (torch.mm(adj, support)
                                    + torch.mm(grad_output, weights_linear_t))
                elif linear == 1:
                    output_input = torch.mm(grad_output, weights_linear_t)
                else:
                    output_input = torch.mm(adj, support)
                support = torch.mm(adj, grad_output)

            if not is_transformer:
                output_weights = torch.mm(input_t, support)          # (in, out)
            else:
                # W_V gradient is computed outside FPYNQ_GAT via autograd on X_V=input@W_V
                output_weights = torch.zeros_like(weights)

            if sage == 1 or linear == 1:
                grad_weights_linear = torch.mm(input_t, grad_output)
            else:
                grad_weights_linear = torch.zeros(
                    input_t.shape[0], grad_output.shape[1])

            if config.profiling == 1:
                print("CPU backward: {:.5f}s".format(time.time() - tmult))

            grad_weights  = output_weights
            grad_attention = output_attention
            grad_input     = output_input

            return (something, something, something, something,
                    grad_input, grad_weights, grad_weights_linear,
                    something, something, grad_attention,
                    something, something, something, something, something)


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
    


class GCNConv_SGRACE(Module):
    """
    GAT layer 
    """
    def __init__(self, state, in_features, out_features, bias=True):
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
        self.state = state
        if(config.acc==1):
         self.my_ip = state.my_ip
        else:
         self.my_ip = None   
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def run_kernel(self):
        self.my_ip.register_map.CTRL.AP_START = 1
        while self.my_ip.register_map.CTRL.AP_IDLE == 0:
            pass
   


    def forward(self, stream,dense, relu,input, edge_index,norm, adj,srelu,weights_l2,weights_l3,weights_l4,attention_l2, attention_l3):
  
        
        st = self.state
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
         st.srelu_buffer[0] = srelu


        if(config.instant_layer_count==1):
         if(dense==0):
          if(config.show_sparsity==1):
           zeros = np.count_nonzero(input==0)
           print("sparsity in the input:",st.layern,"zeros:",zeros/(input.shape[0]*input.shape[1]))   
          pynq_features = input.to_sparse() #coo
          nnz_fea = len(pynq_features.values())
          #print("nnz_fea")
          #print(nnz_fea)
          if(config.acc==1):
           self.my_ip.register_map.nnz_fea1 = nnz_fea
          st.rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
          st.columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
          #values_np = pynq_features.values().data.numpy() 
          values_np = pynq_features.values()\
          #st.values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
          st.values_fea_buffer[0:nnz_fea] = values_np

          write_fea = 0
          if (write_fea == 1):
           pynq_features_dense = input.detach().numpy()
           df = pd.DataFrame(pynq_features_dense) #convert to a dataframe
           df.to_csv("./fea_dense.txt",index=False,header=False) #save to file  
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
           print('nnz fea') 
           print(nnz_fea)
           #sys.exit()

         else:
          if(config.device=="cpu"):   
           xaux = input.detach().numpy()
          else:
           xaux = input
          #xaux = input
            
          #support_pynq_q_t = np.transpose(support_pynq_q)
  
          xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
          #st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
          st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
  
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
         st.rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
         st.values_adj_buffer[0:nnz_adj] = norm
         st.columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]

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

         output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3, self.attention,attention_l2, attention_l3,self.out_features,self.dropout,relu)
 
        else: #2 layer
         if(st.layern==1):  
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
           st.rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
           st.columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
           #values_np = pynq_features.values().data.numpy() 
           values_np = pynq_features.values() 
           #st.values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
           st.values_fea_buffer[0:nnz_fea] = values_np

          else:
           if(config.device=="cpu"):   
            xaux = input.detach().numpy()
           else:
            xaux = input
           #xaux = input
           xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
           #st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
           st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
           #print('edge_index shape') 
           #print(edge_index.shape)
           #print('input size') 
           #print(input.size(0))

  
          nnz_adj = len(norm)
          st.rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
          st.values_adj_buffer[0:nnz_adj] = norm
          st.columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
          if(config.acc==1):
           self.my_ip.register_map.nnz_adj1 = nnz_adj
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2,weights_l3,self.attention,attention_l2, attention_l3,self.out_features,self.dropout,relu)
         else: # second layer
          nnz_adj = len(norm)
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3,self.attention,attention_l2, attention_l3,self.out_features,self.dropout,relu)        
       
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GATConv_SGRACE(Module):
    """
    GAT layer 
    """
    def __init__(self, state, in_features, out_features, nheads=1, bias=True, dropout=0.2, alpha=0.2, concat=False):
        super(GATConv_SGRACE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #torch.manual_seed(12345)

        self.compute_attention = 1    
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
        
        self.attention = Parameter(torch.empty(size=(2*out_features*nheads, 1)))
        init.xavier_uniform_(self.attention.data, gain=1.414)

        #init.xavier_uniform_(self.attention.data, gain=1.414)
        #print('first attention')
        #print(self.attention)
        self.leakyrelu = LeakyReLU(self.alpha)
    
        self.fn = FPYNQ_GAT.apply
        if(config.acc==1):
         self.state = state
         self.my_ip = state.my_ip
        else:
         self.my_ip = None   
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def run_kernel(self):
        self.my_ip.register_map.CTRL.AP_START = 1
        while self.my_ip.register_map.CTRL.AP_IDLE == 0:
            pass
   


    def forward(self, stream,dense, relu,input, edge_index,norm, adj,srelu,weights_l2,weights_l3,weights_l4,attention_l2, attention_l3):
  
        
        st = self.state
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
         st.srelu_buffer[0] = srelu


        if(config.instant_layer_count==1):
         if(dense==0):
          if(config.show_sparsity==1):
           zeros = np.count_nonzero(input==0)
           print("sparsity in the input:",st.layern,"zeros:",zeros/(input.shape[0]*input.shape[1]))   
          pynq_features = input.to_sparse() #coo
          nnz_fea = len(pynq_features.values())
          #print("nnz_fea")
          #print(nnz_fea)
          if(config.acc==1):
           self.my_ip.register_map.nnz_fea1 = nnz_fea
          st.rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
          st.columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
          #values_np = pynq_features.values().data.numpy() 
          values_np = pynq_features.values()\
          #st.values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
          st.values_fea_buffer[0:nnz_fea] = values_np

          write_fea = 0
          if (write_fea == 1):
           pynq_features_dense = input.detach().numpy()
           df = pd.DataFrame(pynq_features_dense) #convert to a dataframe
           df.to_csv("./fea_dense.txt",index=False,header=False) #save to file  
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
           print('nnz fea') 
           print(nnz_fea)
           #sys.exit()

         else:
          if(config.device=="cpu"):   
           xaux = input.detach().numpy()
          else:
           xaux = input
          #xaux = input
            
          #support_pynq_q_t = np.transpose(support_pynq_q)
  
          xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
          #st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
          st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
  
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
         st.rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
         st.values_adj_buffer[0:nnz_adj] = norm
         st.columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]

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

         output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3, self.attention,attention_l2, attention_l3,self.out_features,self.dropout,relu)
 
        else: #2 layer
         if(st.layern==1):  
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
           st.rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
           st.columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
           #values_np = pynq_features.values().data.numpy() 
           values_np = pynq_features.values() 
           #st.values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
           st.values_fea_buffer[0:nnz_fea] = values_np

          else:
           if(config.device=="cpu"):   
            xaux = input.detach().numpy()
           else:
            xaux = input
           #xaux = input
           xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
           #st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
           st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
           #print('edge_index shape') 
           #print(edge_index.shape)
           #print('input size') 
           #print(input.size(0))

  
          nnz_adj = len(norm)
          st.rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
          st.values_adj_buffer[0:nnz_adj] = norm
          st.columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
          if(config.acc==1):
           self.my_ip.register_map.nnz_adj1 = nnz_adj
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2,weights_l3,self.attention,attention_l2, attention_l3,self.out_features,self.dropout,relu)
         else: # second layer
          nnz_adj = len(norm)
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3,self.attention,attention_l2, attention_l3,self.out_features,self.dropout,relu)        
       
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SAGEConv_SGRACE(Module):
    """
    GAT layer 
    """
    def __init__(self, state, in_features, out_features, bias=True):
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
        self.linear=1
        
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
         self.state = state
         self.my_ip = state.my_ip
        else:
         self.my_ip = None   
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, stream,dense, relu,input, edge_index,norm, adj,srelu,weights_l2,weights_l3,weights_l4,attention_l2, attention_l3):
  

        st = self.state
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
          st.rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
          st.columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
          #values_np = pynq_features.values().data.numpy() 
          values_np = pynq_features.values() 
          #st.values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
          st.values_fea_buffer[0:nnz_fea] = values_np

         else:
          if(config.device=="cpu"):   
           xaux = input.detach().numpy()
          else:
           xaux = input
          #xaux = input
          xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
          #st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
          st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
          #print('edge_index shape') 
          #print(edge_index.shape)
          #print('input size') 
          #print(input.size(0))

  
         nnz_adj = len(norm)
         st.rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
         st.values_adj_buffer[0:nnz_adj] = norm
         st.columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
         if(config.acc==1):
          self.my_ip.register_map.nnz_adj1 = nnz_adj

         output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3, self.attention,attention_l2,attention_l3,self.out_features,self.dropout,relu)
 
        else: #2 layer
         if(st.layern==1):  
          if(dense==0):
           pynq_features = input.to_sparse() #coo
           nnz_fea = len(pynq_features.values())
           if(config.acc==1):
            self.my_ip.register_map.nnz_fea1 = nnz_fea
           st.rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
           st.columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
           #values_np = pynq_features.values().data.numpy() 
           values_np = pynq_features.values() 
           #st.values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
           st.values_fea_buffer[0:nnz_fea] = values_np

          else:
           if(config.device=="cpu"):   
            xaux = input.detach().numpy()
           else:
            xaux = input
           #xaux = input
           xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
           #st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
           st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
           #print('edge_index shape') 
           #print(edge_index.shape)
           #print('input size') 
           #print(input.size(0))

  
          nnz_adj = len(norm)
          st.rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
          st.values_adj_buffer[0:nnz_adj] = norm
          st.columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
          if(config.acc==1):
           self.my_ip.register_map.nnz_adj1 = nnz_adj
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight,  self.weight_linear,weights_l2,weights_l3,self.attention,attention_l2,attention_l3,self.out_features,self.dropout,relu)
         else: # second layer
          nnz_adj = len(norm)
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3,self.attention,attention_l2,attention_l3,self.out_features,self.dropout,relu)        
       
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SAGEGAT_SGRACE(Module):
    """
    GAT layer 
    """
    def __init__(self, state, in_features, out_features, nheads=1, bias=True, dropout=0.2, alpha=0.2, concat=False):
        super(SAGEGAT_SGRACE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.dropout = dropout
        #torch.manual_seed(12345)

        #relu = 0 #when merging two layers OJO 
        self.compute_attention = 1
        self.sage=1
        self.linear=1
        
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
         self.state = state
         self.my_ip = state.my_ip
        else:
         self.my_ip = None   
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, stream,dense, relu,input, edge_index,norm, adj,srelu,weights_l2,weights_l3,weights_l4,attention_l2, attention_l3):
  

        st = self.state
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
          st.rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
          st.columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
          #values_np = pynq_features.values().data.numpy() 
          values_np = pynq_features.values() 
          #st.values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
          st.values_fea_buffer[0:nnz_fea] = values_np

         else:
          if(config.device=="cpu"):   
           xaux = input.detach().numpy()
          else:
           xaux = input
          #xaux = input
          xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
          #st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
          st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
          #print('edge_index shape') 
          #print(edge_index.shape)
          #print('input size') 
          #print(input.size(0))

  
         nnz_adj = len(norm)
         st.rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
         st.values_adj_buffer[0:nnz_adj] = norm
         st.columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
         if(config.acc==1):
          self.my_ip.register_map.nnz_adj1 = nnz_adj

         output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3,self.attention,attention_l2,attention_l3,self.out_features,self.dropout,relu)
 
        else: #2 layer
         if(st.layern==1):  
          if(dense==0):
           pynq_features = input.to_sparse() #coo
           nnz_fea = len(pynq_features.values())
           if(config.acc==1):
            self.my_ip.register_map.nnz_fea1 = nnz_fea
           st.rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
           st.columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
           #values_np = pynq_features.values().data.numpy() 
           values_np = pynq_features.values() 
           #st.values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
           st.values_fea_buffer[0:nnz_fea] = values_np

          else:
           if(config.device=="cpu"):   
            xaux = input.detach().numpy()
           else:
            xaux = input
           #xaux = input
           xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
           #st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
           st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
           #print('edge_index shape') 
           #print(edge_index.shape)
           #print('input size') 
           #print(input.size(0))

  
          nnz_adj = len(norm)
          st.rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
          st.values_adj_buffer[0:nnz_adj] = norm
          st.columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
          if(config.acc==1):
           self.my_ip.register_map.nnz_adj1 = nnz_adj
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3, self.attention,attention_l2,attention_l3,self.out_features,self.dropout,relu)
         else: # second layer
          nnz_adj = len(norm)
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight,  self.weight_linear, weights_l2, weights_l3, self.attention,attention_l2,attention_l3,self.out_features,self.dropout,relu)        
       
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    

class Linear_SGRACE(Module):
    """
    Linear layer 
    """
    def __init__(self, state, in_features, out_features, bias=False):
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
         self.state = state
         self.my_ip = state.my_ip
        else:
         self.my_ip = None   
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

 

    def forward(self, stream,dense, relu,input, edge_index,norm, adj,srelu,weights_l2,weights_l3,weights_l4,attention_l2, attention_l3):
    
        

        st = self.state
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
           print("active layer: ",st.layern)   
           print('max/min features')   
           print(np.max(input))
           print(np.min(input))
          st.rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
          st.columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
          #values_np = pynq_features.values().data.numpy() 
          values_np = pynq_features.values() 
          #st.values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
          st.values_fea_buffer[0:nnz_fea] = values_np

         else:
          if(config.device=="cpu"):   
           xaux = input.detach().numpy()
          else:
           xaux = input
          #xaux = input
          if(config.show_max_min==1):
           print("active layer: ",st.layern)   
           print('max/min features')   
           print(np.max(xaux))
           print(np.min(xaux))
          xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
          #st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
          st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
          #print('st.layern') 
          #print(st.layern)
          if(config.write_file == 1 and st.layern == 3):  
           df = pd.DataFrame(xaux) #convert to a dataframe
           df.to_csv("./linear_in.txt",index=False,header=False) #save to file  
          #print('values fea buffer')
          #print(st.values_fea_buffer[0:10])
          #print('edge_index shape') 
          #print(edge_index.shape)
          #print('input size') 
          #print(input.size(0))

  
         nnz_adj = len(norm)
         #st.rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
         #st.values_adj_buffer[0:nnz_adj] = norm
         #st.columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
         if(config.acc==1):
          self.my_ip.register_map.nnz_adj1 = nnz_adj

         output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear, weights_l2, weights_l3, self.attention,attention_l2,attention_l3,self.out_features,self.dropout,relu)
 
        else: #n layer
         if(st.layern==1):  
          if(dense==0):
           pynq_features = input.to_sparse() #coo
           nnz_fea = len(pynq_features.values())
           if(config.acc==1):
            self.my_ip.register_map.nnz_fea1 = nnz_fea
           st.rowPtr_fea_buffer[0:nnz_fea] = pynq_features.indices()[0]
           st.columnIndex_fea_buffer[0:nnz_fea] =  pynq_features.indices()[1]
           #values_np = pynq_features.values().data.numpy() 
           values_np = pynq_features.values() 
           #st.values_fea_buffer[0:nnz_fea] = values_np.astype(config.float_type)# *  (2**f_align)
           st.values_fea_buffer[0:nnz_fea] = values_np

          else:
           if(config.device=="cpu"):   
            xaux = input.detach().numpy()
           else:
            xaux = input
           #xaux = input
           xaux = xaux.reshape(1,xaux.shape[0]*xaux.shape[1])
           #st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux.astype(config.float_type)# *  (2**f_align)
           st.values_fea_buffer[0:xaux.shape[0]*xaux.shape[1]] = xaux
           #print('edge_index shape') 
           #print(edge_index.shape)
           #print('input size') 
           #print(input.size(0))

  
          nnz_adj = len(norm)
          st.rowPtr_adj_buffer[0:nnz_adj]=edge_index[0]
          st.values_adj_buffer[0:nnz_adj] = norm
          st.columnIndex_adj_buffer[0:nnz_adj]=edge_index[1]
          if(config.acc==1):
           self.my_ip.register_map.nnz_adj1 = nnz_adj 
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear,weights_l2,weights_l3,self.attention,attention_l2,attention_l3,self.out_features,self.dropout,relu)
         else: # second layer
          nnz_adj = len(norm)
          output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.weight_linear,weights_l2, weights_l3,self.attention,attention_l2,attention_l3,self.out_features,self.dropout,relu)        
       
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
 
 # model_buffer contents are set by SGRACEState._write_model_program()






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
  #NA, terminate, stream, dense_fea, relu, compute_attention, sage, linear

  #gcnconv  0,0,1,0,1,0,0,0
  #gcnconv  0,0,1,1,1,0,0,0
  #linear   0,1,0,1,0,0,0,1  

 if(config.acc==1):
  print("SGRACE hardware loaded and ready!") 
 else:
  print("SGRACE emulation ready!") 
   

# =============================================================================
# SGRACE EXTENSIONS
# =============================================================================
# Contents added below:
#   1. SGRACEState  — centralised dataclass replacing the ~40 bare module-level
#                     globals produced by init_SGRACE / consumed by FPYNQ_GAT.
#   2. _SGRACELayerBase — shared DMA boilerplate for new layer classes.
#   3. GINConv_SGRACE   — Graph Isomorphism Network convolution.
#   4. TransformerConv_SGRACE — Graph Transformer convolution.
#
# Instruction byte layout (8-bit ISA):
#   [7] sage     — residual self-loop branch (weight_linear path)
#   [6] linear   — skip aggregation, run weight_linear only
#   [5] gat      — learnable edge-attention scoring
#   [4] relu     — fused ReLU on output
#   [3] s_in     — feature input is streamed
#   [2] s_out    — feature output is streamed to next layer
#   [1:0] gemm   — matrix-multiply mode
#
# GINConv        → sage=1, linear=1, gat=0
# TransformerConv → sage=0/1, linear=0, gat=1
# =============================================================================

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1.  SGRACEState
# ---------------------------------------------------------------------------

@dataclass
class SGRACEState:
    """
    Centralised hardware state object — replaces the ~40 bare ``global``
    variables that ``init_SGRACE`` scattered across the module namespace.

    Usage
    -----
    Call ``SGRACEState.init(number_of_classes)`` **once** at startup in place
    of the old ``init_SGRACE(number_of_classes)`` call.  Every layer object
    receives ``state=SGRACEState.get()`` (or the return value of ``.init()``
    directly).

    Example
    -------
    ::

        state = SGRACEState.init(number_of_classes=7)
        conv  = GINConv_SGRACE(64, 32, state=state)
        tconv = TransformerConv_SGRACE(32, 16, state=state)
    """

    # ── layer counter ────────────────────────────────────────────────────────
    layern:               int   = 1
    total_layer_count:    int   = 2
    instant_layer_count:  int   = 1

    # ── fixed-point pipeline ─────────────────────────────────────────────────
    frac_bits_o:          int   = 16
    frac_bits:            int   = 8
    internal_quantization: int  = 16

    # ── quantisation scale / zero-point (layer 1, 2, last-linear) ───────────
    w_s:   float = 1.0;  w_z:   int = 0;  w_s_o:   float = 1.0
    w_s2:  float = 1.0;  w_z2:  int = 0;  w_s_o2:  float = 1.0
    w_sl:  float = 1.0;  w_zl:  int = 0;  w_s_ol:  float = 1.0
    a_s:   float = 1.0;  a_z:   int = 0;  a_s_o:   float = 1.0
    f_s:   float = 1.0;  f_z:   int = 0;  f_s_o:   float = 1.0
    f_s2:  float = 1.0;  f_z2:  int = 0;  f_s_o2:  float = 1.0
    f_sl:  float = 1.0;  f_zl:  int = 0;  f_s_ol:  float = 1.0
    l_sl:  float = 1.0;  l_zl:  int = 0;  l_s_ol:  float = 1.0
    go_s:  float = 1.0;  go_z:  int = 0;  go_s_o:  float = 1.0

    # ── dequantisation factors ───────────────────────────────────────────────
    deq_o:  float = 1.0
    deq_o2: float = 1.0
    deq_ol: float = 1.0
    deq_gw: float = 1.0
    deq_gi: float = 1.0

    # ── internal pipeline scale shifts ───────────────────────────────────────
    scale_fea:  int = 3
    scale_fea2: int = 3
    scale_feal: int = 2

    # ── running calibration trackers ─────────────────────────────────────────
    cur_max_fea:      float = 0.0
    cur_max_fea2:     float = 0.0
    global_max_input: float = 0.0
    global_min_input: float = 0.0

    # ── PYNQ DMA buffers (populated by init) ─────────────────────────────────
    model_buffer:                   object = None
    P_w_buffer:                     object = None
    srelu_buffer:                   object = None
    quantization_scale_fea_buffer:  object = None
    quantization_scale_w_buffer:    object = None
    quantization_scale_l_buffer:    object = None
    deq_factor_buffer:              object = None
    scale_fea_buffer:               object = None
    attention_buffer:               object = None
    bias_buffer:                    object = None
    profiling_buffer:               object = None
    rowPtr_fea_buffer:              object = None
    columnIndex_fea_buffer:         object = None
    values_fea_buffer:              object = None
    rowPtr_adj_buffer:              object = None
    columnIndex_adj_buffer:         object = None
    values_adj_buffer:              object = None
    B_buffer:                       object = None
    B2_buffer:                      object = None
    D_buffer:                       object = None
    E_buffer:                       object = None
    S_buffer:                       object = None
    my_ip:                          object = None

    # ── singleton bookkeeping ─────────────────────────────────────────────────
    _instance: object = field(
        default=None, repr=False, compare=False)

    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def get(cls):
        """Return the singleton created by :meth:`init`."""
        if cls._instance is None:
            raise RuntimeError(
                "SGRACEState.init() has not been called. "
                "Call it once before constructing any SGRACE layer."
            )
        return cls._instance

    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def init(cls, number_of_classes: int, op_type: str = "GCN"):
        """
        Allocate PYNQ buffers and compute all quantisation constants.

        This is the drop-in replacement for ``init_SGRACE(number_of_classes)``.
        It returns a state object instead of polluting the module namespace.
        The legacy ``init_SGRACE`` function is still present in this file for
        backward compatibility with existing model scripts.
        """
        st = cls()

        # ── PYNQ overlay ──────────────────────────────────────────────────────
        if config.acc == 1:
            ol = Overlay("gat_all_unsigned.bit")
            st.my_ip = ol.mmult_top_0

        # ── quantisation range setup ──────────────────────────────────────────
        st._setup_quant_ranges()

        # ── quantisation constants ────────────────────────────────────────────
        st.w_s_o,  st.w_s,  st.w_z  = generate_quantization_qbits_constants(
            st._w_min,  st._w_max,  config.w_qbits)
        st.w_s_o2, st.w_s2, st.w_z2 = generate_quantization_qbits_constants(
            st._w_min2, st._w_max2, config.w_qbits)
        st.w_s_ol, st.w_sl, st.w_zl = generate_quantization_qbits_constants(
            st._w_minl, st._w_maxl, config.w_qbitsl)
        st.a_s_o,  st.a_s,  st.a_z  = generate_quantization_uqbits_constants(
            st._a_min,  st._a_max,  config.w_qbits)
        st.f_s_o,  st.f_s,  st.f_z  = generate_quantization_uqbits_constants(
            st._f_min,  st._f_max,  config.w_qbits)
        st.f_s_o2, st.f_s2, st.f_z2 = generate_quantization_uqbits_constants(
            st._f_min2, st._f_max2, config.w_qbits)
        st.f_s_ol, st.f_sl, st.f_zl = generate_quantization_uqbits_constants(
            st._f_minl, st._f_maxl, config.w_qbitsl)
        st.l_s_ol, st.l_sl, st.l_zl = generate_quantization_qbits_constants(
            st._l_minl, st._l_maxl, config.w_qbitsl)

        go_min, go_max = -0.10, 0.10
        st.go_s_o, st.go_s, st.go_z = generate_quantization_uqbits_constants(
            go_min, go_max, 8)

        # ── dequantisation factors ────────────────────────────────────────────
        st.deq_o  = st.w_s_o  * st.f_s_o  * st.a_s_o
        st.deq_o2 = st.w_s_o2 * st.f_s_o2 * st.a_s_o
        st.deq_ol = st.w_s_ol * st.l_s_ol
        st.deq_gw = st.f_s_o  * st.a_s_o  * st.go_s_o
        st.deq_gi = st.a_s_o  * st.go_s_o * st.w_s_o
        st._apply_qbit_deq_corrections()

        # ── PYNQ buffer allocation ────────────────────────────────────────────
        st._allocate_buffers(number_of_classes)

        # ── model-program buffer ──────────────────────────────────────────────
        if config.acc == 1:
            st._write_model_program(op_type)

        # ── write all static buffer addresses to hardware register map ────────
        if config.acc == 1:
            st._write_register_map()

        cls._instance = st
        return st

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_quant_ranges(self):
        """Per-bit-width quantisation range lookup — replaces the scattered
        if/elif blocks in ``init_SGRACE``."""
        qb  = config.w_qbits
        qbl = config.w_qbitsl

        _ranges = {
            8: dict(w=(-1.0, 1.0), w2=(-1.0, 1.0), a=(0.0, 1.0), f=(0.0, 1.0),
                    f2=(0.0, 1.0), scale_fea=3, scale_fea2=3, deq_shift=1),
            4: dict(w=(-0.3, 0.3), w2=(-0.6, 0.6), a=(0.0, 0.5), f=(0.0, 1.0),
                    f2=(0.0, 1.0), scale_fea=3, scale_fea2=3, deq_shift=3),
            2: dict(w=(-0.1, 0.1), w2=(-0.1, 0.1), a=(0.0, 0.1), f=(0.0, 1.0),
                    f2=(0.0, 1.0), scale_fea=1, scale_fea2=1, deq_shift=1),
            1: dict(w=(-0.1, 0.1), w2=(-0.1, 0.1), a=(0.0, 0.1), f=(0.0, 1.0),
                    f2=(0.0, 1.0), scale_fea=1, scale_fea2=1, deq_shift=1),
        }
        _ranges_l = {
            32: dict(wl=(-0.5, 0.5), fl=(0.0, 8.0), ll=(-4.0, 4.0),
                     scale_feal=0, deq_shift_l=0),
            8:  dict(wl=(-1.0, 1.0), fl=(0.0, 8.0), ll=(-4.0, 4.0),
                     scale_feal=2, deq_shift_l=2),
            4:  dict(wl=(-0.5, 0.5), fl=(0.0, 2.0), ll=(-1.0, 1.0),
                     scale_feal=1, deq_shift_l=1),
            1:  dict(wl=(-0.1, 0.1), fl=(0.0, 0.1), ll=(-0.1, 0.1),
                     scale_feal=1, deq_shift_l=1),
        }

        r  = _ranges.get(qb,  _ranges[8])
        rl = _ranges_l.get(qbl, _ranges_l[8])

        self._w_min,  self._w_max  = r['w']
        self._w_min2, self._w_max2 = r['w2']
        self._a_min,  self._a_max  = r['a']
        self._f_min,  self._f_max  = r['f']
        self._f_min2, self._f_max2 = r['f2']
        self.scale_fea              = r['scale_fea']
        self.scale_fea2             = r['scale_fea2']
        self._deq_shift             = r['deq_shift']

        self._w_minl, self._w_maxl = rl['wl']
        self._f_minl, self._f_maxl = rl['fl']
        self._l_minl, self._l_maxl = rl['ll']
        self.scale_feal             = rl['scale_feal']
        self._deq_shift_l           = rl['deq_shift_l']

        _falign = {8: 0, 4: 4, 2: 6, 1: 7}
        _betaqu = {8: 255, 4: 15, 2: 2, 1: 1}
        self._f_align  = _falign.get(qb,  0)
        self._beta_qu  = _betaqu.get(qb,  255)
        self._f_alignl = _falign.get(qbl, 0)
        self._beta_qul = _betaqu.get(qbl, 255)

    def _apply_qbit_deq_corrections(self):
        """Scale deq factors by hardware internal pipeline shift."""
        self.deq_o  *= pow(2, self._deq_shift)
        self.deq_o2 *= pow(2, self._deq_shift)
        if self._deq_shift_l > 0:
            self.deq_ol *= pow(2, self._deq_shift_l)
        _iq = {8: 16, 4: 8, 2: 4, 1: 4}
        self.internal_quantization = _iq.get(config.w_qbits, 16)

    def _allocate_buffers(self, number_of_classes: int):
        """Allocate all PYNQ DMA buffers (acc=1) or set to empty lists (acc=0)."""
        if config.acc != 1:
            for attr in (
                "model_buffer", "P_w_buffer", "srelu_buffer",
                "quantization_scale_fea_buffer", "quantization_scale_w_buffer",
                "quantization_scale_l_buffer", "deq_factor_buffer",
                "scale_fea_buffer", "attention_buffer", "bias_buffer",
                "profiling_buffer", "rowPtr_fea_buffer",
                "columnIndex_fea_buffer", "values_fea_buffer",
                "rowPtr_adj_buffer", "columnIndex_adj_buffer",
                "values_adj_buffer", "B_buffer", "B2_buffer",
                "D_buffer", "E_buffer", "S_buffer",
            ):
                setattr(self, attr, [])
            return

        ft = config.float_type    # np.float32
        mt = config.model_type    # np.uint8

        self.model_buffer                  = allocate(5,               dtype=mt)
        self.P_w_buffer                    = allocate(5,               dtype=mt)
        self.srelu_buffer                  = allocate(5,               dtype=ft)
        self.quantization_scale_fea_buffer = allocate(5,               dtype=ft)
        self.quantization_scale_w_buffer   = allocate(5,               dtype=ft)
        self.quantization_scale_l_buffer   = allocate(5,               dtype=ft)
        self.deq_factor_buffer             = allocate(5,               dtype=ft)
        self.scale_fea_buffer              = allocate(5,               dtype=mt)
        self.attention_buffer              = allocate(config.P_w * 2,  dtype=ft)
        self.bias_buffer                   = allocate(1024,            dtype=np.int32)
        self.profiling_buffer              = allocate(16,              dtype=np.int64)
        self.rowPtr_fea_buffer             = allocate(config.NNZ_fea,  dtype=np.int32)
        self.columnIndex_fea_buffer        = allocate(config.NNZ_fea,  dtype=np.int32)
        self.values_fea_buffer             = allocate(config.NNZ_fea,  dtype=ft)
        self.rowPtr_adj_buffer             = allocate(config.NNZ_adj,  dtype=np.int32)
        self.columnIndex_adj_buffer        = allocate(config.NNZ_adj,  dtype=np.int32)
        self.values_adj_buffer             = allocate(config.NNZ_adj,  dtype=ft)
        sz = config.N_adj * config.P_w + 2 * config.P_w * config.P_w
        self.B_buffer  = allocate(sz,             dtype=ft)
        self.B2_buffer = allocate(sz,             dtype=ft)
        self.D_buffer  = allocate(sz,             dtype=ft)
        self.E_buffer  = allocate(config.NNZ_adj, dtype=ft)
        self.S_buffer  = allocate(config.NNZ_adj, dtype=ft)

        # Populate scale / deq / srelu buffers
        self.quantization_scale_fea_buffer[0] = 1.0 / self.f_s
        self.quantization_scale_fea_buffer[1] = 1.0 / self.f_s2
        self.quantization_scale_fea_buffer[2] = 1.0 / self.f_sl
        self.quantization_scale_fea_buffer[3] = 1.0 / self.f_sl

        self.quantization_scale_w_buffer[0] = 1.0 / self.w_s
        self.quantization_scale_w_buffer[1] = 1.0 / self.w_s2
        self.quantization_scale_w_buffer[2] = 1.0 / self.w_sl
        self.quantization_scale_w_buffer[3] = 1.0 / self.w_sl

        self.quantization_scale_l_buffer[0] = 1.0 / self.l_sl
        self.quantization_scale_l_buffer[1] = 1.0 / self.l_sl
        self.quantization_scale_l_buffer[2] = 1.0 / self.l_sl
        self.quantization_scale_l_buffer[3] = 1.0 / self.l_sl

        self.deq_factor_buffer[0] = self.deq_o
        self.deq_factor_buffer[1] = self.deq_o2
        self.deq_factor_buffer[2] = self.deq_ol

        self.scale_fea_buffer[0] = self.scale_fea
        self.scale_fea_buffer[1] = self.scale_fea2
        self.scale_fea_buffer[2] = self.scale_feal

        for i in range(5):
            self.srelu_buffer[i] = 0.0

        lc = config.total_layer_count
        hc = config.hidden_channels
        for i in range(lc - 1):
            self.P_w_buffer[i] = hc
        self.P_w_buffer[lc - 1] = number_of_classes

        ip = self.my_ip
        ip.register_map.f_align              = self._f_align
        ip.register_map.beta_qu              = self._beta_qu
        ip.register_map.f_alignl             = self._f_alignl
        ip.register_map.beta_qul             = self._beta_qul
        ip.register_map.quantized_multiplier = self.internal_quantization

    @staticmethod
    def _enc(sage=0, linear=0, gat=0, relu=0, s_in=0, s_out=0,
             dense_fea=0, dense_adj=0):
        """
        Encode one ISA instruction byte from its semantic field values.

        Bitfield layout (bit 0 = LSB):
            [7] sage      — residual self-loop branch (GraphSAGE / GIN)
            [6] linear    — linear-only layer (no graph aggregation)
            [5] gat       — learnable attention scoring
            [4] relu      — fused ReLU on output
            [3] s_in      — feature input streamed from previous layer
            [2] s_out     — feature output streamed to next layer
            [1] dense_fea — combination step uses dense input (XW dense)
            [0] dense_adj — aggregation step uses dense mode (AXW dense)
        """
        return (sage    << 7 | linear   << 6 | gat     << 5 |
                relu    << 4 | s_in     << 3 | s_out   << 2 |
                dense_fea << 1 | dense_adj << 0)

    @staticmethod
    def _op_flags(op):
        """
        Return (sage, linear, gat) flag triple for a canonical operator name.

        Recognised names (case-insensitive):
            "GCN"         sage=0, linear=0, gat=0
            "GAT"         sage=0, linear=0, gat=1
            "SAGE"        sage=1, linear=1, gat=0
            "SAGEGAT"     sage=1, linear=1, gat=1
            "GIN"         sage=0, linear=0, gat=0  (same ISA path as GCN; differs in adjacency)
            "TRANSFORMER" sage=0, linear=0, gat=1  (same ISA path as GAT)
            "LINEAR"      sage=0, linear=1, gat=0  (classifier linear layer)

        Note: SAGE requires both sage=1 and linear=1 because the residual
        self-loop branch (weight_linear) must be enabled alongside aggregation.
        """
        op = op.upper()
        aliases = {"TRANSFORMER": "GAT"}
        op = aliases.get(op, op)
        #              sage  linear  gat
        table = {
            "GCN":     (0,   0,      0),
            "GAT":     (0,   0,      1),
            "SAGE":    (1,   1,      0),
            "SAGEGAT": (1,   1,      1),
            "GIN":     (0,   0,      0),
            "LINEAR":  (0,   1,      0),
        }
        if op not in table:
            raise ValueError(
                f"Unknown operator {op!r}. "
                "Use: GCN, GAT, SAGE, SAGEGAT, GIN, Transformer, or Linear.")
        return table[op]

    def _write_model_program(self, program):
        """
        Encode a layer sequence into ``model_buffer``.

        ``program`` can be:

        A) A shorthand string — uniform operator for all graph layers,
           followed by a Linear classifier.  The streaming flags
           (s_in / s_out) and dense/sparse input flags are derived
           automatically from layer position and instant_layer_count:

               state.set_model("GAT")
               state.set_model("GCN")

        B) A list of layer descriptor dicts, one per layer including
           the final classifier.  Each dict may contain:

               op        : str — operator name (required)
                                 "GCN" | "GAT" | "SAGE" | "SAGEGAT" |
                                 "GIN" | "Transformer" | "Linear"
               relu      : int — 1 = fused ReLU on output  (default 1
                                 for graph layers, 0 for Linear)
               dense_fea : int — 1 = feature matrix (XW) is dense
                                 (default 0 for layer 0, 1 for subsequent layers)
               dense_adj : int — 1 = adjacency matrix (A) is dense
                                 (default 0 for all layers)

           Streaming flags (s_in / s_out) are always set automatically
           from instant_layer_count — do not set them manually.

           Example — two GCN layers then a Linear classifier:
               state.set_model([
                   {"op": "GCN",    "relu": 1, "dense_fea": 0, "dense_adj": 0},
                   {"op": "GCN",    "relu": 0, "dense_fea": 1, "dense_adj": 0},
                   {"op": "Linear", "relu": 0, "dense_fea": 1, "dense_adj": 0},
               ])

           Example — mixed GCN + GAT:
               state.set_model([
                   {"op": "GCN",    "relu": 1, "dense_fea": 0, "dense_adj": 0},
                   {"op": "GAT",    "relu": 0, "dense_fea": 1, "dense_adj": 0},
                   {"op": "Linear", "relu": 0, "dense_fea": 1, "dense_adj": 0},
               ])

           Example — three graph layers + Linear (4-layer model):
               state.set_model([
                   {"op": "GAT",    "relu": 1, "dense_fea": 0, "dense_adj": 0},
                   {"op": "GAT",    "relu": 1, "dense_fea": 1, "dense_adj": 0},
                   {"op": "GCN",    "relu": 0, "dense_fea": 1, "dense_adj": 0},
                   {"op": "Linear", "relu": 0, "dense_fea": 1, "dense_adj": 0},
               ])
        """
        enc = SGRACEState._enc
        ilc = config.instant_layer_count
        mb  = self.model_buffer

        # ── A: shorthand string → expand to descriptor list ───────────────────
        if isinstance(program, str):
            op  = program.upper()
            if op in ("GIN",): op = "SAGE"
            if op in ("TRANSFORMER",): op = "GAT"
            lc  = config.total_layer_count
            # Graph layers: layer 0 sparse (dense_fea=0), remaining dense (dense_fea=1);
            # dense_adj=0 for all layers by default; relu=1 for all except last graph layer
            # Except the last graph layer before Linear has relu=0
            descs = []
            for i in range(lc - 1):          # graph conv layers
                descs.append({
                    "op":    op,
                    "relu":  1 if i < lc - 2 else 0,
                    "dense_fea": 0 if i == 0 else 1,
                    "dense_adj": 0,
                })
            descs.append({"op": "Linear", "relu": 0, "dense_fea": 1, "dense_adj": 0})
            program = descs

        # ── B: descriptor list → encode each layer byte ───────────────────────
        n = len(program)
        if n > len(mb):
            raise ValueError(
                f"Program has {n} layers but model_buffer only holds "
                f"{len(mb)}. Increase config.total_layer_count.")

        chained = (ilc > 1)

        for i, desc in enumerate(program):
            op_name = desc["op"].upper()
            is_last = (i == n - 1)
            is_first = (i == 0)
            is_linear = (op_name in ("LINEAR",))

            sage, linear, gat = SGRACEState._op_flags(op_name)
            relu       = desc.get("relu",  0 if is_linear else 1)
            dense_fea  = desc.get("dense_fea", 0 if is_first else 1)
            dense_adj  = desc.get("dense_adj", 0)

            # Streaming flags derived automatically from chained mode
            if chained:
                s_out = 0 if is_last  else 1
                s_in  = 0 if is_first else 1
                # In chained mode dense flags are carried via stream
                # for intermediate layers
                if not is_last and not is_first:
                    dense_fea = 0
                    dense_adj = 0
            else:
                s_in  = 0
                s_out = 0

            mb[i] = enc(sage=sage, linear=linear, gat=gat,
                        relu=relu, s_in=s_in, s_out=s_out,
                        dense_fea=dense_fea, dense_adj=dense_adj)

    def _write_register_map(self):
        """
        Write all static buffer physical addresses to the hardware register map.
        This mirrors lines 3551-3600 of the original init_SGRACE and MUST be
        called after buffers are allocated, otherwise the FPGA has no valid
        DMA addresses and will hang on AP_START.
        """
        ip = self.my_ip
        ip.register_map.load_weights                    = config.load_weights
        ip.register_map.gat_mode                        = 0
        ip.register_map.model_offset_1                  = self.model_buffer.physical_address
        ip.register_map.scale_fea_offset_1              = self.scale_fea_buffer.physical_address
        ip.register_map.quantization_scale_fea_offset_1 = self.quantization_scale_fea_buffer.physical_address
        ip.register_map.quantization_scale_w_offset_1   = self.quantization_scale_w_buffer.physical_address
        ip.register_map.quantization_scale_lin_offset_1 = self.quantization_scale_l_buffer.physical_address
        ip.register_map.srelu_offset_1                  = self.srelu_buffer.physical_address
        ip.register_map.deq_factor_offset_1             = self.deq_factor_buffer.physical_address
        ip.register_map.P_w_offset_1                    = self.P_w_buffer.physical_address
        ip.register_map.E1_offset_1                     = self.E_buffer.physical_address
        ip.register_map.E2_offset_1                     = self.E_buffer.physical_address
        ip.register_map.E3_offset_1                     = self.E_buffer.physical_address
        ip.register_map.E4_offset_1                     = self.E_buffer.physical_address
        ip.register_map.S1_offset_1                     = self.S_buffer.physical_address
        ip.register_map.S2_offset_1                     = self.S_buffer.physical_address
        ip.register_map.S3_offset_1                     = self.S_buffer.physical_address
        ip.register_map.S4_offset_1                     = self.S_buffer.physical_address
        ip.register_map.layer_count                     = config.instant_layer_count
        ip.register_map.ate_m_offset_1                  = self.attention_buffer.physical_address
        ip.register_map.B_offset_1                      = self.B_buffer.physical_address
        ip.register_map.B2_offset_1                     = self.B2_buffer.physical_address
        ip.register_map.rowPtr_fea1_offset_1            = self.rowPtr_fea_buffer.physical_address
        ip.register_map.rowPtr_fea2_offset_1            = self.rowPtr_fea_buffer.physical_address
        ip.register_map.rowPtr_fea3_offset_1            = self.rowPtr_fea_buffer.physical_address
        ip.register_map.rowPtr_fea4_offset_1            = self.rowPtr_fea_buffer.physical_address
        ip.register_map.columnIndex_fea1_offset_1       = self.columnIndex_fea_buffer.physical_address
        ip.register_map.columnIndex_fea2_offset_1       = self.columnIndex_fea_buffer.physical_address
        ip.register_map.columnIndex_fea3_offset_1       = self.columnIndex_fea_buffer.physical_address
        ip.register_map.columnIndex_fea4_offset_1       = self.columnIndex_fea_buffer.physical_address
        ip.register_map.values_fea1_offset_1            = self.values_fea_buffer.physical_address
        ip.register_map.values_fea2_offset_1            = self.values_fea_buffer.physical_address
        ip.register_map.values_fea3_offset_1            = self.values_fea_buffer.physical_address
        ip.register_map.values_fea4_offset_1            = self.values_fea_buffer.physical_address
        ip.register_map.rowPtr_adj1_offset_1            = self.rowPtr_adj_buffer.physical_address
        ip.register_map.rowPtr_adj2_offset_1            = self.rowPtr_adj_buffer.physical_address
        ip.register_map.rowPtr_adj3_offset_1            = self.rowPtr_adj_buffer.physical_address
        ip.register_map.rowPtr_adj4_offset_1            = self.rowPtr_adj_buffer.physical_address
        ip.register_map.columnIndex_adj1_offset_1       = self.columnIndex_adj_buffer.physical_address
        ip.register_map.columnIndex_adj2_offset_1       = self.columnIndex_adj_buffer.physical_address
        ip.register_map.columnIndex_adj3_offset_1       = self.columnIndex_adj_buffer.physical_address
        ip.register_map.columnIndex_adj4_offset_1       = self.columnIndex_adj_buffer.physical_address
        ip.register_map.values_adj1_offset_1            = self.values_adj_buffer.physical_address
        ip.register_map.values_adj2_offset_1            = self.values_adj_buffer.physical_address
        ip.register_map.values_adj3_offset_1            = self.values_adj_buffer.physical_address
        ip.register_map.values_adj4_offset_1            = self.values_adj_buffer.physical_address
        ip.register_map.quantized_multiplier            = self.internal_quantization
        ip.register_map.bias_offset_1                   = self.bias_buffer.physical_address
        ip.register_map.profiling_offset_1              = self.profiling_buffer.physical_address

    def print_model_program(self):
        """
        Print the contents of model_buffer with a human-readable decode of
        every instruction byte.

        Instruction byte bitfield (bit 0 = LSB):
            [7] sage    — residual self-loop branch (GraphSAGE / GIN)
            [6] linear  — linear-only layer (no graph aggregation)
            [5] gat     — learnable attention scoring
            [4] relu    — fused ReLU on output
            [3] s_in    — feature input streamed from previous layer
            [2] s_out   — feature output streamed to next layer
            [1] dense_fea — combination step (XW) uses dense input
            [0] dense_adj — aggregation step (AXW) uses dense mode
        """
        lc = config.total_layer_count

        def decode(byte):
            sage   = (byte >> 7) & 1
            linear = (byte >> 6) & 1
            gat    = (byte >> 5) & 1
            relu   = (byte >> 4) & 1
            s_in   = (byte >> 3) & 1
            s_out  = (byte >> 2) & 1
            dense_fea = (byte >> 1) & 1
            dense_adj = (byte >> 0) & 1

            # Derive operator name from ISA bits.
            # GCN and GIN share the same opcode (sage=0, linear=0, gat=0);
            # they differ only in the adjacency matrix passed at runtime.
            if sage and gat and linear:
                op = "SAGEGAT"
            elif sage and linear:
                op = "SAGE"
            elif sage:
                op = "SAGE(no-linear)"
            elif gat:
                op = "GAT/Transformer"
            elif linear:
                op = "Linear"
            else:
                op = "GCN/GIN"

            flags = []
            if relu:      flags.append("relu")
            if s_in:      flags.append("s_in")
            if s_out:     flags.append("s_out")
            if dense_fea: flags.append("dense_fea")
            if dense_adj: flags.append("dense_adj")
            flag_str = ", ".join(flags) if flags else "—"

            return op, flag_str

        print(f"model_buffer  (total_layer_count={lc}, "
              f"instant_layer_count={config.instant_layer_count})")
        print(f"  {'Slot':<6} {'Byte':<8} {'Operator':<18} {'Flags'}")
        print(f"  {'-'*6} {'-'*8} {'-'*18} {'-'*30}")
        for i in range(lc):
            byte = int(self.model_buffer[i])
            op, flags = decode(byte)
            print(f"  [{i}]    0x{byte:02X}     {op:<18} {flags}")

    def set_model(self, program):
        """
        Write a new instruction program into the hardware model buffer.

        ``program`` is either:

        - A shorthand string for a uniform model, e.g. ``"GAT"``.
        - A list of layer descriptor dicts for a custom model, e.g.::

            state.set_model([
                {"op": "GCN",    "relu": 1, "dense": 0},
                {"op": "GAT",    "relu": 0, "dense": 1},
                {"op": "Linear", "relu": 0, "dense": 1},
            ])

        See ``_write_model_program`` for full descriptor syntax.
        """
        if config.acc == 1:
            self._write_model_program(program)
            self.my_ip.register_map.model_offset_1 = self.model_buffer.physical_address
        label = program if isinstance(program, str) else "custom"
        print(f"Model program updated: {label}")
        self.print_model_program()

    def advance_layer(self):
        """Increment the active-layer counter, wrapping at total_layer_count."""
        if self.layern < self.total_layer_count:
            self.layern += 1
        else:
            self.layern = 1

    def update_max_fea(self, raw_int_val: int):
        """Update running feature-max trackers from hardware register readout."""
        fval = float(raw_int_val) / (2 ** self.frac_bits_o)
        if self.layern == 1:
            if fval > self.cur_max_fea:
                self.cur_max_fea = fval
        else:
            if fval > self.cur_max_fea2:
                self.cur_max_fea2 = fval


# ---------------------------------------------------------------------------
# 2.  _SGRACELayerBase — shared DMA boilerplate
# ---------------------------------------------------------------------------

class _SGRACELayerBase(Module):
    """
    Internal base class shared by GINConv_SGRACE and TransformerConv_SGRACE.

    Encapsulates the repeated DMA fill / kernel-launch / readback pattern
    that is duplicated verbatim across the original GCNConv_SGRACE,
    GATConv_SGRACE, SAGEConv_SGRACE, and Linear_SGRACE classes.
    """

    def __init__(self, in_features: int, out_features: int,
                 state: SGRACEState, bias: bool = False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.state        = state
        self.nheads       = 1
        self.alpha        = 0.0
        self.dropout      = 0.0
        self.leakyrelu    = LeakyReLU(self.alpha)
        self.my_ip        = state.my_ip if config.acc == 1 else None

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

    def _fill_feature_buffer(self, input: torch.Tensor, dense: int):
        st = self.state
        if dense == 0:
            coo = input.detach().to_sparse()
            nnz = len(coo.values())
            self.my_ip.register_map.nnz_fea1 = nnz
            st.rowPtr_fea_buffer[0:nnz]      = coo.indices()[0]
            st.columnIndex_fea_buffer[0:nnz] = coo.indices()[1]
            st.values_fea_buffer[0:nnz]      = coo.values()
        else:
            xaux = (input.detach().numpy()
                    if config.device == "cpu" else input.detach())
            flat = xaux.reshape(1, xaux.shape[0] * xaux.shape[1])
            st.values_fea_buffer[0:flat.shape[1]] = flat

    def _fill_adj_buffer(self, edge_index, norm) -> int:
        st  = self.state
        nnz = len(norm)
        norm_d       = norm.detach() if isinstance(norm, torch.Tensor) else norm
        edge_index_d = edge_index.detach() if isinstance(edge_index, torch.Tensor) else edge_index
        st.rowPtr_adj_buffer[0:nnz]      = edge_index_d[0]
        st.values_adj_buffer[0:nnz]      = norm_d
        st.columnIndex_adj_buffer[0:nnz] = edge_index_d[1]
        self.my_ip.register_map.nnz_adj1 = nnz
        return nnz

    def _run_kernel(self) -> float:
        t0 = time.time()
        self.my_ip.register_map.CTRL.AP_START = 1
        while self.my_ip.register_map.CTRL.AP_IDLE == 0:
            pass
        return 1000.0 * (time.time() - t0)

    def _read_output(self, n_rows: int, n_cols: int) -> torch.Tensor:
        raw = self.state.D_buffer[0:n_rows * n_cols].reshape(n_rows, n_cols)
        return torch.from_numpy(np.array(raw)).float()

    def _set_layer_registers(self, nnz_adj: int,
                              input: torch.Tensor,
                              weights: torch.Tensor):
        ip = self.my_ip
        ip.register_map.N_adj    = input.shape[0]
        ip.register_map.M_adj    = input.shape[0]
        ip.register_map.M_fea    = input.shape[1]
        ip.register_map.P_w      = weights.shape[1]
        ip.register_map.nnz_adj1 = nnz_adj

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"({self.in_features} → {self.out_features})")


# ---------------------------------------------------------------------------
# 3.  GINConv_SGRACE
# ---------------------------------------------------------------------------

class GINConv_SGRACE(_SGRACELayerBase):
    """
    Graph Isomorphism Network convolution (Xu et al., 2019).

    Forward computation
    -------------------
    .. math::
        h_v^{(k)} = \\mathrm{MLP}\\!\\left(
            (1+\\varepsilon)\\,h_v^{(k-1)} W_{\\mathrm{self}}
            + \\sum_{u \\in \\mathcal{N}(v)} h_u^{(k-1)} W_{\\mathrm{agg}}
        \\right)

    Hardware mapping
    ----------------
    The SGRACE ``sage=1`` instruction computes::

        output = A · (X · W_agg) + X · W_self

    Setting ``W_self = (1 + ε) · weight_linear`` at forward time recovers
    GIN semantics on the unmodified FPYNQ_GAT kernel (no new ISA entry).

    Instruction byte: ``sage=1, gat=0, linear=1, relu=<arg>``.

    Parameters
    ----------
    in_features : int
    out_features : int
    state       : SGRACEState
    eps         : float   Initial ε value (default 0.0).
    train_eps   : bool    Make ε a learnable parameter (default True).
    bias        : bool
    """

    def __init__(self, in_features: int, out_features: int,
                 state: SGRACEState,
                 eps: float = 0.0,
                 train_eps: bool = True,
                 bias: bool = False):

        self.compute_attention = 0
        self.sage              = 0
        self.linear            = 0

        super().__init__(in_features, out_features, state, bias)

        # Single weight matrix shared by aggregation and self-loop.
        # GIN uses the same GCN hardware path (sage=0, linear=0).
        # The difference from GCN is purely in the adjacency matrix:
        # raw un-normalised A is passed instead of D^{-1/2} A D^{-1/2}.
        # The (1+ε)I self-loop is handled by the ε-scaled weight.
        #   H = σ(((1+ε)I + A) H W)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        init.xavier_uniform_(self.weight.data, gain=1.414)
        self.weight_linear = self.weight   # shared — not a separate parameter

        self.eps = (Parameter(torch.tensor([eps]))
                    if train_eps
                    else self.register_buffer("eps", torch.tensor([eps])))

        # Dummy attention — not used by sage path, but FPYNQ_GAT expects it
        self.attention = torch.empty(size=(2 * out_features, 1))

    def forward(self, input: torch.Tensor,
                edge_index: torch.Tensor,
                norm: torch.Tensor,
                adj,
                relu: int = 1,
                dense: int = 0,
                stream: int = 0,
                weights_l2=None,
                weights_l3=None,
                attention_l2=None,
                attention_l3=None) -> torch.Tensor:
        """
        Parameters
        ----------
        input      : (N, F_in) node feature matrix.
        edge_index : (2, E) COO adjacency indices.
        norm       : (E,) edge normalisation weights.
        adj        : dense/sparse adjacency (used by FPYNQ_GAT backward).
        relu       : 1 to enable fused hardware ReLU.
        dense      : 1 if features are dense (skips COO packing).
        weights_l2, weights_l3, attention_l2, attention_l3 :
                     next-layer weights for chained (instant_layer_count > 1)
                     execution.
        """
        st = self.state
        ip = self.my_ip

        # Fold ε into the self-loop weight for this forward pass
        weight_self = (1.0 + self.eps) * self.weight_linear

        if config.acc == 1:
            ip.register_map.load_weights = config.load_weights
            st.srelu_buffer[0] = 0.0

        _wd = torch.zeros(self.out_features, self.out_features)
        _ad = torch.zeros(2 * self.out_features, 1)
        weights_l2   = weights_l2   if weights_l2   is not None else _wd
        weights_l3   = weights_l3   if weights_l3   is not None else _wd
        attention_l2 = attention_l2 if attention_l2 is not None else _ad
        attention_l3 = attention_l3 if attention_l3 is not None else _ad

        if config.instant_layer_count == 1:
            self._fill_feature_buffer(input, dense)
            nnz_adj = self._fill_adj_buffer(edge_index, norm)
            self._set_layer_registers(nnz_adj, input, self.weight)
            output = FPYNQ_GAT.apply(
                ip, self, adj, nnz_adj, input,
                self.weight, weight_self,
                weights_l2, weights_l3,
                self.attention, attention_l2, attention_l3,
                self.out_features, self.dropout, relu,
            )
        else:
            if st.layern == 1:
                self._fill_feature_buffer(input, dense)
                nnz_adj = self._fill_adj_buffer(edge_index, norm)
                self._set_layer_registers(nnz_adj, input, self.weight)
            output = FPYNQ_GAT.apply(
                ip, self, adj, len(norm), input,
                self.weight, weight_self,
                weights_l2, weights_l3,
                self.attention, attention_l2, attention_l3,
                self.out_features, self.dropout, relu,
            )

        return output


# ---------------------------------------------------------------------------
# 4.  TransformerConv_SGRACE
# ---------------------------------------------------------------------------

class TransformerConv_SGRACE(_SGRACELayerBase):
    """
    Graph Transformer convolution (Shi et al., 2021).

    Forward computation
    -------------------
    For each edge (i, j):

    .. math::
        e_{ij} = \\frac{(x_i W_Q)(x_j W_K)^\\top}{\\sqrt{d_k}}

    Then row-wise softmax and value aggregation:

    .. math::
        h_i = \\sum_j \\alpha_{ij}\\,(x_j W_V)

    Hardware mapping
    ----------------
    The SGRACE ``compute_attention=1`` path computes the LeakyReLU-GAT score
    ``e = Wh · a`` where ``a ∈ ℝ^{2d×1}``.  Scaled dot-product attention is
    approximated by initialising ``a = 1/√d · ones(2d, 1)`` so that::

        aᵀ [Whᵢ ‖ Whⱼ] ≈ Whᵢ · Whⱼᵀ / √d

    This runs without any new hardware instructions.  For exact attention,
    set ``exact_attn=True`` — the CPU precomputes the full attention matrix
    and the hardware reduces to a plain SpMM (``gat=0``).

    Instruction byte: ``sage=0/1, gat=1, linear=0, relu=<arg>``.

    Parameters
    ----------
    in_features  : int
    out_features : int
    state        : SGRACEState
    nheads       : int   Number of attention heads (default 1).
    beta         : bool  Add a skip-connection via weight_linear (default False).
    concat       : bool  Concatenate multi-head outputs; else average.
    exact_attn   : bool  Use exact CPU dot-product attention (default False).
    bias         : bool
    """

    def __init__(self, in_features: int, out_features: int,
                 state: SGRACEState,
                 nheads: int = 1,
                 beta: bool = False,
                 concat: bool = False,
                 exact_attn: bool = False,
                 bias: bool = False):

        self.compute_attention = 1
        self.sage              = 1 if beta else 0
        self.linear            = 0

        super().__init__(in_features, out_features, state, bias)

        self.nheads     = nheads
        self.concat     = concat
        self.exact_attn = exact_attn
        self.beta       = beta

        head_dim = out_features

        self.W_Q = Parameter(torch.FloatTensor(in_features, head_dim * nheads))
        self.W_K = Parameter(torch.FloatTensor(in_features, head_dim * nheads))
        self.W_V = Parameter(torch.FloatTensor(in_features, head_dim * nheads))
        init.xavier_uniform_(self.W_Q.data, gain=1.414)
        init.xavier_uniform_(self.W_K.data, gain=1.414)
        init.xavier_uniform_(self.W_V.data, gain=1.414)

        self.weight_linear = (
            Parameter(torch.FloatTensor(in_features, out_features))
            if beta else torch.zeros(in_features, out_features)
        )
        if beta:
            init.xavier_uniform_(self.weight_linear.data, gain=1.414)

        # Hardware aggregation weight = value projection
        self.weight = self.W_V

        # Attention vector initialised to approximate scaled dot-product
        d = head_dim * nheads
        self.attention = Parameter(
            torch.full((2 * d, 1), fill_value=1.0 / math.sqrt(d)))

        self._scale = math.sqrt(head_dim)

    def _exact_attention(self, input: torch.Tensor,
                         adj_dense: torch.Tensor) -> torch.Tensor:
        """Compute exact scaled dot-product attention on CPU."""
        Q      = input @ self.W_Q                    # (N, H·d)
        K      = input @ self.W_K                    # (N, H·d)
        scores = (Q @ K.T) / self._scale             # (N, N)
        mask   = torch.full_like(scores, -9e15)
        masked = torch.where(adj_dense > 0, scores, mask)
        return F.softmax(masked, dim=1)              # (N, N)

    def forward(self, input: torch.Tensor,
                edge_index: torch.Tensor,
                norm: torch.Tensor,
                adj,
                relu: int = 0,
                dense: int = 1,
                stream: int = 0,
                weights_l2=None,
                weights_l3=None,
                attention_l2=None,
                attention_l3=None) -> torch.Tensor:
        """
        Parameters
        ----------
        input      : (N, F_in) node feature matrix.
        edge_index : (2, E) COO adjacency indices.
        norm       : (E,) normalised edge weights.
        adj        : dense/sparse adjacency tensor.
        relu       : fused ReLU flag (default 0).
        dense      : feature-input mode (default 1 for transformer layers).
        weights_l2, weights_l3, attention_l2, attention_l3 :
                     chained-mode next-layer weights.
        """
        st = self.state
        ip = self.my_ip

        _wd = torch.zeros(self.out_features, self.out_features)
        _ad = torch.zeros(2 * self.out_features * self.nheads, 1)
        weights_l2   = weights_l2   if weights_l2   is not None else _wd
        weights_l3   = weights_l3   if weights_l3   is not None else _wd
        attention_l2 = attention_l2 if attention_l2 is not None else _ad
        attention_l3 = attention_l3 if attention_l3 is not None else _ad

        if config.acc == 1:
            ip.register_map.load_weights = config.load_weights
            st.srelu_buffer[0] = 0.0

        if self.exact_attn:
            adj_dense  = adj.to_dense() if adj.is_sparse else adj
            attn       = self._exact_attention(input, adj_dense)
            attn_sp    = attn.to_sparse()
            edge_index = attn_sp.indices().detach()
            norm       = attn_sp.values().detach()
            adj        = attn.to_sparse()

        X_V = input @ self.W_V    # project to value space before DMA fill

        if config.instant_layer_count == 1:
            self._fill_feature_buffer(X_V, dense)
            nnz_adj = self._fill_adj_buffer(edge_index, norm)
            self._set_layer_registers(nnz_adj, X_V, self.W_V)
            output = FPYNQ_GAT.apply(
                ip, self, adj, nnz_adj, X_V,
                self.W_V, self.weight_linear,
                weights_l2, weights_l3,
                self.attention, attention_l2, attention_l3,
                self.out_features, self.dropout, relu,
            )
        else:
            if st.layern == 1:
                self._fill_feature_buffer(X_V, dense)
                nnz_adj = self._fill_adj_buffer(edge_index, norm)
                self._set_layer_registers(nnz_adj, X_V, self.W_V)
            output = FPYNQ_GAT.apply(
                ip, self, adj, len(norm), X_V,
                self.W_V, self.weight_linear,
                weights_l2, weights_l3,
                self.attention, attention_l2, attention_l3,
                self.out_features, self.dropout, relu,
            )

        if self.nheads > 1:
            N = output.shape[0]
            output = output.view(N, self.nheads, self.out_features)
            output = (output.reshape(N, self.nheads * self.out_features)
                      if self.concat else output.mean(dim=1))

        return output


# ---------------------------------------------------------------------------
# k-th order adjacency utility
# ---------------------------------------------------------------------------

def kth_order_adj(edge_index, num_nodes, k, norm_each=True,
                  exclusive=True, max_nnz=None):
    """
    Compute the k-th order adjacency matrix in COO sparse format.

    Parameters
    ----------
    edge_index : (2, E) LongTensor — COO indices of the 1-hop adjacency.
    num_nodes  : int              — number of nodes N.
    k          : int              — order (1=standard, 2=2-hop, 3=3-hop).
    norm_each  : bool             — re-normalise after each multiplication.
    exclusive  : bool             — if True (default), return ONLY edges
                                    between nodes exactly k hops apart,
                                    removing any edge that also appears in
                                    Aʲ for j < k.  This avoids redundancy
                                    when stacking layers with different k.
                                    If False, return all edges reachable
                                    within k hops (cumulative).
    max_nnz    : int or None      — raise an error if the result exceeds
                                    this many non-zeros.  Use config.NNZ_adj
                                    to enforce DMA buffer limits.

    Returns
    -------
    edge_index_k : (2, E_k) LongTensor
    norm_k       : (E_k,)   FloatTensor — normalised edge weights
    """
    import scipy.sparse as sp

    # Build scipy CSR from edge_index
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    val = np.ones(len(row), dtype=np.float32)
    A   = sp.csr_matrix((val, (row, col)), shape=(num_nodes, num_nodes))

    def _sym_norm(M):
        """Symmetric normalisation D^{-1/2} M D^{-1/2}."""
        d     = np.array(M.sum(1)).flatten()
        d_inv = np.where(d > 0, d ** -0.5, 0.0)
        D_inv = sp.diags(d_inv)
        return D_inv @ M @ D_inv

    # Build binary (un-normalised) powers to track reachability
    A_bin = A.astype(np.float32)
    A_bin.data[:] = 1.0

    # Cumulative reachability mask: union of A¹ … A^(k-1)
    # used to subtract lower-order edges when exclusive=True
    lower_mask = sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)

    Ak_bin = A_bin.copy()
    for hop in range(1, k):
        lower_mask = lower_mask + Ak_bin
        Ak_bin = (Ak_bin @ A_bin)
        Ak_bin.data[:] = 1.0        # keep binary
        Ak_bin.eliminate_zeros()

    # Ak_bin now contains 1s wherever two nodes are reachable in exactly k steps
    # (including shorter paths); lower_mask has 1s for all hops < k

    # Now compute the normalised k-hop matrix
    A_norm = _sym_norm(A)
    Ak = A_norm.copy() if norm_each else A.astype(np.float32)
    for _ in range(k - 1):
        Ak = Ak @ (A_norm if norm_each else A_bin)
        Ak.eliminate_zeros()

    if not norm_each:
        Ak = _sym_norm(Ak)

    if exclusive and k > 1:
        # Zero out entries that exist in any lower-order adjacency
        lower_mask.eliminate_zeros()
        # Convert lower_mask to boolean mask and apply
        lower_bool = lower_mask.astype(bool)
        Ak = Ak - Ak.multiply(lower_bool)
        Ak.eliminate_zeros()

    Ak = Ak.tocoo()

    if max_nnz is not None and Ak.nnz > max_nnz:
        raise ValueError(
            f"k={k} order adjacency ({'exclusive' if exclusive else 'cumulative'}) "
            f"has {Ak.nnz} non-zeros which exceeds max_nnz={max_nnz} "
            "(config.NNZ_adj). Reduce k or increase NNZ_adj in config.py.")

    edge_index_k = torch.tensor(
        np.vstack([Ak.row, Ak.col]), dtype=torch.long)
    norm_k = torch.tensor(Ak.data, dtype=torch.float32)
    return edge_index_k, norm_k


# ---------------------------------------------------------------------------
# HighOrderGATConv_SGRACE  — GAT/Transformer with k-th order adjacency
# ---------------------------------------------------------------------------

class HighOrderGATConv_SGRACE(GATConv_SGRACE):
    """
    GAT convolution using the k-th order adjacency matrix.

    This layer precomputes Aᵏ (k-hop neighbourhood) once on the first
    forward call and caches it.  The hardware sees a standard (denser)
    sparse graph and runs the same GAT kernel — no new hardware instructions
    are needed.

    For k=1 this is identical to GATConv_SGRACE.
    For k=2 each node attends to all 2-hop neighbours.
    For k=3 each node attends to all 3-hop neighbours.

    Parameters
    ----------
    in_features, out_features, state : same as GATConv_SGRACE
    k          : int  — adjacency order (default 2)
    norm_each  : bool — re-normalise after each hop (default True)
    max_nnz    : int  — buffer overflow guard (default config.NNZ_adj)
    All other kwargs forwarded to GATConv_SGRACE.
    """

    def __init__(self, in_features: int, out_features: int,
                 state: SGRACEState,
                 k: int = 2,
                 norm_each: bool = True,
                 exclusive: bool = True,
                 max_nnz: int = None,
                 **kwargs):
        super().__init__(state, in_features, out_features, **kwargs)
        self.k          = k
        self.norm_each  = norm_each
        self.exclusive  = exclusive
        self.max_nnz    = max_nnz if max_nnz is not None else config.NNZ_adj
        # Cache: keyed by (num_nodes, edge_index hash) → (edge_index_k, norm_k)
        self._adj_cache = {}

    def _get_kth_adj(self, edge_index, num_nodes):
        """Return cached or freshly computed k-th order adjacency."""
        key = (num_nodes, edge_index.data_ptr(), self.exclusive)
        if key not in self._adj_cache:
            mode = "exclusive" if self.exclusive else "cumulative"
            print(f"HighOrderGATConv_SGRACE: computing A^{self.k} ({mode}) "
                  f"(N={num_nodes}, 1-hop E={edge_index.shape[1]})...")
            ei_k, norm_k = kth_order_adj(
                edge_index, num_nodes, self.k,
                norm_each=self.norm_each,
                exclusive=self.exclusive,
                max_nnz=self.max_nnz)
            self._adj_cache[key] = (ei_k, norm_k)
            print(f"  Done: {self.k}-hop E={ei_k.shape[1]}")
        return self._adj_cache[key]

    def forward(self, stream, dense, relu, input,
                edge_index, norm, adj,
                srelu, weights_l2, weights_l3, weights_l4,
                attention_l2, attention_l3):
        # Replace edge_index/norm/adj with k-th order versions
        num_nodes       = input.size(0)
        ei_k, norm_k    = self._get_kth_adj(edge_index, num_nodes)
        adj_k           = torch.sparse_coo_tensor(ei_k, norm_k)
        return super().forward(
            stream, dense, relu, input,
            ei_k, norm_k, adj_k,
            srelu, weights_l2, weights_l3, weights_l4,
            attention_l2, attention_l3)
