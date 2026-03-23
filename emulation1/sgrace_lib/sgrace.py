import torch
import torch_geometric.transforms as T
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


def generate_quantization_constants(alpha, beta, alpha_q, beta_q):

    # Affine quantization mapping
    #this beta_o and alpha_o take into account that during training the integer values are inserted in a fractional pipeline
    #This pipeline has 7 bit integer and 25 bit fractional (total 32). If the values are inserted with an alignment of 18 and have 8 bits width
    # then they become x.xxxxxxx like if they are divided by (2**(8-1) and this effect must be removed during dequant.
    

    #beta_o = beta_q/(2**(frac_bits-f_align))
    #alpha_o = alpha_q/(2**(frac_bits-f_align))

    #beta_o = beta_q/(2**(config.w_qbits-f_align-1)) #ojo perhaps the one on top....
    #alpha_o = alpha_q/(2**(config.w_qbits-f_align-1))


    if(config.w_qbits == 1):
     beta_o = beta_q/(2**2) #ojo perhaps the one on top....
     alpha_o = alpha_q/(2**2)
    else: 
     beta_o = beta_q/(2**(config.w_qbits)) #ojo perhaps the one on top....
     alpha_o = alpha_q/(2**(config.w_qbits))

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
                                           beta_q=beta_q)
    

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
                                           beta_q=beta_q)
   
    
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

class RPYNQ(torch.autograd.Function):
    """Both forward and backward are static methods."""

    @staticmethod
    def forward(ctx,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        output = input.clone() #this clone is important to make it work
        return output

    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the inputs: here input and weights
        """
        input, = ctx.saved_tensors
 
        grad_input = grad_output.clone() #this clone is iportatant
        grad_input[input == 0] = 0 #hardware style that takes into account the hardware integrated relu 
     

        return grad_input
   


class FPYNQ_GAT(torch.autograd.Function):
    """Both forward and backward are static methods."""
    
    @staticmethod
    def forward(ctx,my_ip, self,adj,nnz_adj,input, weights,attention,out_features,dropout,relu):
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
    
     
        #if (config.profiling == 1):
        #  fmult = time.time()

    
        if (config.acc==1):
            
         #print("acc ON") 
         #if (config.profiling == 1):
         # rmult = time.time()

         global layern
         #print("deq_o and active layer")
         #print(deq_o)
         #print(layern)



         if (layern == 1):
          my_ip.register_map.scale_fea = scale_fea #2 #scale fea
          int32bits = np.asarray(deq_o, dtype=np.float32).view(np.int32).item() 
          my_ip.register_map.deq_factor = int32bits
    
          qsf = 1/f_s
          int32bits = np.asarray(qsf, dtype=np.float32).view(np.int32).item() 
          my_ip.register_map.quantization_scale_fea = int32bits
        
        
          qsw = 1/w_s
          int32bits = np.asarray(qsw, dtype=np.float32).view(np.int32).item() 
          my_ip.register_map.quantization_scale_w = int32bits
          layern = 2
         else:
          my_ip.register_map.scale_fea = scale_fea2 #2 #scale fea
          int32bits = np.asarray(deq_o2, dtype=np.float32).view(np.int32).item() 
          my_ip.register_map.deq_factor = int32bits
          qsf = 1/f_s2
          int32bits = np.asarray(qsf, dtype=np.float32).view(np.int32).item() 
          my_ip.register_map.quantization_scale_fea = int32bits
        
          qsw = 1/w_s2
          int32bits = np.asarray(qsw, dtype=np.float32).view(np.int32).item() 
          my_ip.register_map.quantization_scale_w = int32bits   
          layern = 1      
            
         qsa = 1/a_s
         #print("a_s")
         #print(a_s)
         int32bits = np.asarray(qsa, dtype=np.float32).view(np.int32).item() 
         my_ip.register_map.quantization_scale_adj = int32bits

        
         
       
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
         #print(input.shape[0])
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
         
         if (config.profiling == 1):
          amult = time.time()
         support = torch.transpose(weights,0,1)
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
       
  
    
          
         if(config.show_max_min==1):
          print("max/min weights")
          print(np.max(support_pynq))
          print(np.min(support_pynq))

         support_pynq_q = support_pynq
            
         support_pynq_q = support_pynq_q.reshape(1, (weights.shape[0]*weights.shape[1]))
    
         B_buffer[0:(weights.shape[0]*weights.shape[1])] = support_pynq_q.astype(config.float_type)
         if (config.min_output == 0):
           print("B_Buffer")
           print(B_buffer[0:10])

         if(config.compute_attention == 1):
          attention_q = attention_q.numpy()
          attention_buffer[0:(attention.shape[0]*attention.shape[1])] = attention_q.astype(config.float_type)

         if(config.show_max_min==1):
          print("max/min quantized weights")
          print(np.max(support_pynq_q))
          print(np.min(support_pynq_q))
        
         global B_size
         B_size = (weights.shape[0]*weights.shape[1])
          
         my_ip.register_map.quantized_multiplier = internal_quantization #apply internal quantization
         
         
         if (config.profiling == 1):
          amult = time.time()
          for _ in range(1):
           my_ip.register_map.CTRL.AP_START=1
           kernel_done = my_ip.register_map.CTRL.AP_DONE
           while kernel_done == 0:
            kernel_done = my_ip.register_map.CTRL.AP_DONE
          dmult =  time.time()
         else:
          my_ip.register_map.CTRL.AP_START=1
          kernel_done = my_ip.register_map.CTRL.AP_DONE
          while kernel_done == 0:
           kernel_done = my_ip.register_map.CTRL.AP_DONE  

         if (config.profiling == 1):   
          print('Accelerator forward kernel mult time: {:.5f}ms'.format(1000/1*(dmult - amult)))
   
       

         output_acc = D_buffer[0:input.shape[0]*weights.shape[1]]
          
         if(config.compute_attention==1):
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
        

         output_acc = output_acc.reshape(input.shape[0],weights.shape[1]) 
         #get sparse matrix for e and softmax
         if(config.compute_attention==1):
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
         if(config.compute_attention == 1):
          ctx.save_for_backward(adj,input, weights,output_e,output_s,output_acc)
         else:
          ctx.save_for_backward(adj,input, weights,adj,adj,output_acc) 
        
         if (config.profiling == 1):   
          print('Forward function time: {:.5f}ms'.format(1000/1*(time.time() - fmult)))
         return output_acc



        else: #no accelerator

          input = input.float()



        
          if(config.fake_quantization==1):
           #print("input")
           if(config.show_max_min==1): 
            print("max input", torch.max(input))
           input_q = quantization_ufbits(input, f_s, f_z, config.w_qbits)
          else:
           input_q = input

          #print(adj.shape)
          #print(input.shape)
          #print(weights.shape)
          if(config.show_max_min==1):
           print("max/min weights")
           print(torch.max(weights))
           print(torch.min(weights))
        
        
          tmult = time.time() 
         
          if(config.fake_quantization==1):
           if(config.show_max_min==1):
            print("max weight", torch.max(weights)) 
           #print("weights") 
           weights_q = quantization_fbits(weights, w_s, w_z,  config.w_qbits) 
           #print("weights after quant")  
           #print(weights)        
          else:
           weights_q = weights      
        
          #Wh = torch.mm(input, weights[i]) # h.shape: (N, in_features), Wh.shape: (N, out_features)
          #e = prepare_attentional_mechanism_input(Wh,attention[i],out_features)
          Wh = torch.mm(input_q, weights_q) # h.shape: (N, in_features), Wh.shape: (N, out_features)
          #need to emulate the hardware effects of the QTYPE quantization after input*weights
          if (config.fake_quantization == 1):
            #print(Wh)
            if(config.show_max_min==1):
             print("Max fea value", torch.max(Wh))
            Wh = Wh/(2**scale_fea) 
            a_min = -(2**internal_quantization-1)/(2**internal_quantization)
            a_max = (2**internal_quantization-1)/(2**internal_quantization)
            #print(a_min)
            #print(a_max)
            #Wh = np.clip(Wh, a_min=-0.9921875, a_max=0.9921875)
            #Wh = np.clip(Wh, a_min=-0.875, a_max=0.875)
            #Wh = np.clip(Wh, a_min=-0.99999999, a_max=0.99999999)
            Wh = torch.clip(Wh, min=a_min, max=a_max)
            Wh = torch.round(Wh, decimals = (internal_quantization-1))

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
          
          if(config.compute_attention==1):
           output_cpu = torch.matmul(attention2, Wh)
        
           if (config.profiling == 1):
            print('cpu forward mult: {:.5f}s'.format(time.time() - tmult))
            #print(output_cpu)
          else:
           output_cpu = torch.matmul(adj_q, Wh)
           attention2 = adj_q
 
          #RELU
          if(relu==1):
            output_cpu = torch.where(output_cpu > 0, output_cpu, 0)

     
            

          if (config.fake_quantization==1):
           output_cpu = output_cpu*deq_o


          #print("output_cpu")
          #print(output_cpu)
          #print("deq_o")
          #print(deq_o)

          ctx.nheads = self.nheads
          ctx.alpha = self.alpha
          if(config.compute_attention == 1):
           ctx.save_for_backward(adj,input, weights,e,attention2,output_cpu)
          else:
           ctx.save_for_backward(adj,input, weights,adj,adj,output_cpu) 
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
         my_ip.register_map.gat_mode=config.compute_attention
        
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

         return something,something, something, grad_adj,something, grad_input, grad_weights, grad_attention, something, something, something
        
            
        else: 
        
  
         adj,input, weights,e,attentions,output = ctx.saved_tensors
         nheads = ctx.nheads
         alpha = ctx.alpha
        
         #adj, input, weights,output = ctx.saved_tensors

         something = grad_adj = grad_input = grad_weights = grad_attention = None

    
        
           
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
  
            
         if(config.compute_attention==1):
         

            
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
          output_attention = torch.zeros(size=(config.hidden_channels*2, 1))
          output_attention = output_attention.to(config.device)
         
         tmult = time.time()
         #grad_weights = input.t()@adj@grad_output
         #print("in")
         #print(weights_t)
         #print(grad_output)
        
         support = torch.mm(grad_output,weights_t)
         #print(grad_output)
         if(config.compute_attention==1):
          output_input = torch.mm(attentions, support)
          support = torch.mm(attentions,grad_output)
         else:
          output_input = torch.mm(adj, support)
          support = torch.mm(adj,grad_output) 
            
         output_weights = torch.mm(input_t, support)
        
    
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
         #print("grad_attention")
         #print(grad_attention)
        grad_input = output_input

        #print("device")
        #print(grad_attention.device)
        return something, something, something,something, grad_input, grad_weights, grad_attention, something, something, something

  

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
     
    def forward(self, x):
        output = self.fn(x)
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
      
        self.weight = Parameter(torch.FloatTensor(in_features, out_features*nheads))
        init.xavier_uniform_(self.weight.data, gain=1.414)
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
   


    def forward(self, compute_attention,dense, relu,input, edge_index,norm, adj):
  

       
        

        if(config.acc==1):
         self.my_ip.register_map.relu = relu
         self.my_ip.register_map.gemm_mode = dense
         self.my_ip.register_map.gat_mode = compute_attention


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
 
        
        
 
     

     
        output = self.fn(self.my_ip,self,adj,nnz_adj,input, self.weight, self.attention,self.out_features,self.dropout,relu)
 
       
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    




def init_SGRACE():

 if(config.acc==1):
  ol = Overlay("gat_all_unsigned.bit")
 #else:
 # ol = Overlay("gat_all_unsigned.bit",download=False)
  global my_ip
  my_ip = ol.mmult_top_0
 
 #print("my ip")
 #print(ol.ip_dict)

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
 #remember layer number active to adjust the parameters 

 layern = 1

 if(config.w_qbits == 8):

  f_align = 0 #8
  beta_qu = 255

  #photo
  #global w_max
  #w_max = 1.0 
  #global w_min
  #w_min = -1.0 
  #global a_max
  #a_max = 1.0 
  #global w_max2
  #w_max2 = 1.0 
  #global w_min2
  #w_min2 = -1.0
  #global f_max2   
  #f_max2 = 4.0 
  #global f_max
  #f_max = 1.0 
  #global a_min
  #a_min = 0
  #global f_min
  #f_min = 0
  #global f_min2
  #f_min2 = 0
  #go_max = 0.10
  #go_min = -0.10

  #cora
  global w_max
  w_max = 1.0 #8 #citeseer/cora
  global w_min
  w_min = -1.0 #8 #citeseer/cora
  global a_max
  a_max = 1.0 #cora gcn/gat 8-bit
  w_max2 = 1.0 #citeseer/cora 4-bit/8-bit gcn/gat
  w_min2 = -1.0 #citeseer/cora 4-bit/8-bit gcn/gat  
  f_max2 = 1.0 #cora
  global f_max
  f_max = 1.0 #cox/ermd/dd/mutag
  global a_min
  a_min = 0
  global f_min
  f_min = 0
  f_min2 = 0
  go_max = 0.10
  go_min = -0.10

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
  
  w_max = 1.0 #8 #citeseer/cora
  
  w_min = -1.0 #8 #citeseer/cora
  
  a_max = 1.0 #cora gcn/gat 8-bit
  w_max2 = 1.0 #citeseer/cora 4-bit/8-bit gcn/gat
  w_min2 = -1.0 #citeseer/cora 4-bit/8-bit gcn/gat  
  f_max2 = 1.0 #cora
  
  f_max = 1.0 #cox/ermd/dd/mutag
  a_min = 0

  f_min = 0
  f_min2 = 0
  go_max = 0.10
  go_min = -0.10


  #cora

  #w_max = 0.3 #8 #citeseer/cora

  #w_min = -0.3 #8 #citeseer/cora

  #a_max = 0.5 #cora gcn/gat 8-bit
  #w_max2 = 0.6 #citeseer/cora 4-bit/8-bit gcn/gat
  #w_min2 = -0.6 #citeseer/cora 4-bit/8-bit gcn/gat  
  #f_max2 = 1.0 #cora

  
  #f_max = 1.0 #cox/ermd/dd/mutag

  #a_min = 0

  #f_min = 0
  #f_min2 = 0
  #go_max = 0.10
  #go_min = -0.10


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
 if (config.acc == 1):
  B_buffer = allocate((config.N_adj*config.P_w*config.head_count), dtype=config.float_type)
  D_buffer = allocate((config.N_adj*config.P_w*config.head_count), dtype=config.float_type) 
 else:
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
 go_qbits = 8

 #generate constants

 if(config.min_output == 0):
  print("generating qbits w constants with bits: ",config.w_qbits)
 #signed w
 global w_s_o,w_s,w_z
 w_s_o,w_s,w_z=generate_quantization_qbits_constants(w_min, w_max,config.w_qbits)
 global w_s_o2,w_s2,w_z2
 w_s_o2,w_s2,w_z2=generate_quantization_qbits_constants(w_min2, w_max2,config.w_qbits) #forward

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
 if(config.min_output == 0):
  print(f_s)
 if(config.min_output == 0):
  print("generating qbits gi gradient input constants")
 go_s_o,go_s,go_z=generate_quantization_uqbits_constants(go_min, go_max,go_qbits)

 deq_o = w_s_o*f_s_o*a_s_o
 #print("DEQ_O")
 #print(deq_o)

 deq = w_s*f_s*a_s
 #print("DEQ")
 #print(deq)

 deq_o2 = w_s_o2*f_s_o2*a_s_o
 deq_gw = f_s_o*a_s_o*go_s_o
 deq_gi = a_s_o*go_s_o*w_s_o
    

 #adjust internal quantization

 global internal_quantization 
 #8-bit 
 if (config.w_qbits == 8):   
    #weibo
    scale_fea = 4 #scale fea
    scale_fea2 = 4
    deq_o=deq_o*pow(2, 1) #cora 8-bit gcn/gat
    deq_o2=deq_o2*pow(2, 1) #2c#cora 8-bit gcn/gat

    #photo
    #scale_fea = 3
    #scale_fea2 = 3
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
    internal_quantization =  16 #16 # 0x0000FFFF#bit QTYPE 32
  
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
    #scale_fea = 3
    #scale_fea2 = 3
    #deq_o=deq_o*pow(2, 3)
    #deq_o2=deq_o2*pow(2,3)

    #amazom
    #scale_fea = 3
    #scale_fea2 = 3
    #deq_o=deq_o*pow(2, 3)
    #deq_o2=deq_o2*pow(2,3)

    
    #weibo
    scale_fea = 3
    scale_fea2 = 3
    deq_o=deq_o*pow(2, 1)
    deq_o2=deq_o2*pow(2,1)

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
    
    #amazom/cora
    #scale_fea = 3
    #scale_fea2 = 3
    #deq_o=deq_o*pow(2, 3)
    #deq_o2=deq_o2*pow(2, 3)

    #weibo
    scale_fea = 3
    scale_fea2 = 3
    deq_o=deq_o*pow(2, 1)
    deq_o2=deq_o2*pow(2,1)
    
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
    #scale_fea = 1
    #scale_fea2 = 1
    #deq_o=deq_o*pow(2, 1)
    #deq_o2=deq_o2*pow(2, 1)

    #emron
    scale_fea = 2
    scale_fea2 = 2
    deq_o=deq_o*pow(2, 1)
    deq_o2=deq_o2*pow(2, 1)

    if(config.acc==1):
     my_ip.register_map.beta_qu = 1
     my_ip.register_map.f_align = 7
    internal_quantization =  4
    f_align = 7
     
 
    
    if(config.min_output == 0):
     print("Deq factor ",deq_o)

    
 

 #terminate IP configuration
 if(config.acc==1):
  my_ip.register_map.load_weights = config.load_weights #load the weights first before execution (needed for training)
  my_ip.register_map.gat_mode= config.compute_attention
  my_ip.register_map.E1_offset_1 = E_buffer.physical_address
  my_ip.register_map.E2_offset_1 = E_buffer.physical_address
  my_ip.register_map.E3_offset_1 = E_buffer.physical_address
  my_ip.register_map.E4_offset_1 = E_buffer.physical_address
  my_ip.register_map.S1_offset_1 = S_buffer.physical_address
  my_ip.register_map.S2_offset_1 = S_buffer.physical_address
  my_ip.register_map.S3_offset_1 = S_buffer.physical_address
  my_ip.register_map.S4_offset_1 = S_buffer.physical_address
  my_ip.register_map.layer_count=config.layer_count
  my_ip.register_map.ate_m_offset_1 = attention_buffer.physical_address
  my_ip.register_map.B_offset_1 = B_buffer.physical_address
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

 if(config.acc==1):
  print("SGRACE hardware loaded and ready!") 
 else:
  print("SGRACE emulation ready!") 
   