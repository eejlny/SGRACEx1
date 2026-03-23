#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
import torch_geometric.transforms as T
#from torch_geometric.nn import GATConv, GATv2Conv
from decimal import Decimal
import os
import numpy as np
import time
import pandas as pd
import networkx as nx
#%load_ext autoreload
#%autoreload 2

import sys
sys.path.insert(1, '/home/xilinx/jupyter_notebooks/sgrace_lib')

#GATRes_SGRACE
#GCNConv_SGRACE
#GCNRes_SGRACE
#GATConv_SGRACE
#Linear_SGRACE
#Relu_SGRACE

import config
import sgrace
from sgrace import init_SGRACE,GATConv_SGRACE, SAGEGAT_SGRACE, GCNConv_SGRACE, SAGEConv_SGRACE, Relu_SGRACE, Linear_SGRACE

#torch.manual_seed(12345)
torch.manual_seed(3407)


# the node degree is calculate in adj and if a row has a node degree of zero then the features of the node are set to zero.
# I thought that for deep quantization there will be more rows at zero but this is not the case. The normalization
# seems to make the adj values higher and then after quantization there are still not zero. This is problaby not a bad thing since 
# nodes that are initially connected and then remove will hurt accuracy. 
# In summary quantization reduces the number of connections for a node but nodes with just a single connection remain connected. 
norm_adj = 1 #use normalize adjacency
custom = 1 #at 0 use CPU pytorch implementation
full_graph = 0
training = 1

if(training==1):
 config.instant_layer_count=1
else:
 config.instant_layer_count=3


preload = 0 # preload a trained model. Use a low learning rate to just tune the weights.

batch_value = 256 #128 #not relevant in planetoid that is a single graph, relevant for Amazon
num_epochs = 200

#gnn max size







# In[2]:


import math

from torch_geometric.transforms import RemoveIsolatedNodes

from torch_geometric.utils import subgraph

def remove_zero_feature_nodes(data):
    # Mask: True for nodes where ANY feature is non-zero
    mask = (data.x != 0).any(dim=1)
    
    # Get new edge_index, remapped to new node indices
    new_edge_index, new_edge_attr = subgraph(
        mask, data.edge_index, data.edge_attr, relabel_nodes=True
    )
    
    data.x = data.x[mask]
    data.y = data.y[mask]
    #data.train_mask = data.train_mask[mask]
    #data.val_mask = data.val_mask[mask]
    #data.test_mask = data.test_mask[mask]

    data.edge_index = new_edge_index
    data.edge_attr = new_edge_attr
    
    return data

transform = None
#dataset_sel = 'Computers'
#dataset_sel = "Cora"
dataset_sel = "Citeseer"
dataset = Planetoid(root="data/Planetoid", name=dataset_sel, split="full", transform=transform) #split = "full"
#dataset_sel = 'Photo'
#dataset = Amazon(root="data/Amazon", name=dataset_sel, transform=transform)



    


print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

#transform = RemoveIsolatedNodes()
#data = transform(data)
#data = remove_zero_feature_nodes(data)



print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

print("average node degree")
average_node_degree = data.num_edges / data.num_nodes
print(average_node_degree)
print("Fill value")
print(math.log2(average_node_degree))


init_SGRACE(dataset.num_classes)

# In[3]:


if (full_graph==1):
 from torch_geometric.loader import DataLoader



 train_loader = DataLoader(dataset, batch_size=batch_value, shuffle=True)
 test_loader = DataLoader(dataset, batch_size=batch_value, shuffle=False)



else:
 graph_list = []
 #from torch_geometric.loader import NeighborLoader
 from torch_geometric.loader import DataLoader

 data = dataset[0]
    
 transform = RemoveIsolatedNodes()
 data = transform(data)
 data = remove_zero_feature_nodes(data)

 print("Clean data")
 print(data)  
    
 #standard
 data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
 data.train_mask[:data.num_nodes - 1000] = 1

 data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
 data.test_mask[data.num_nodes - 1000:data.num_nodes - 500] = 1

 graph_list.append(data)

    
 train_loader = DataLoader(graph_list, batch_size=batch_value, shuffle=True)
 test_loader = DataLoader(graph_list, batch_size=batch_value, shuffle=False)
    
 

 #train_loader = NeighborLoader(data, batch_size=batch_value, num_neighbors=[10] *1,input_nodes= data.train_mask,shuffle=False)
 #test_loader = NeighborLoader(data, batch_size=batch_value, num_neighbors=[10] * 1,input_nodes= data.test_mask,shuffle=False)

   

# In[4]:


from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import LeakyReLU
from torch_geometric.nn import GATConv,GCNConv
from torch_geometric.nn import global_mean_pool
#from ogb.graphproppred.mol_encoder import AtomEncoder

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        #torch.manual_seed(12345)
        #self.emb = AtomEncoder(dataset.num_node_features)
        self.att = GCNConv(dataset.num_node_features, hidden_channels,bias=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels,bias=False)
        #self.conv3 = GCNConv(hidden_channels, 16)
        self.lin = Linear(hidden_channels, dataset.num_classes,bias=False)
        #self.lin2 = Linear(16, 16)
        #self.lin3 = Linear(16, dataset.num_classes)

    def forward(self, x, edge_index):
        # 1. Obtain node embeddings 
        x = x.float()
        #x = self.emb(x)
        rmult = time.time()
        
        #recorder = DataRecorder(rails['0V85'].power)
        #print('CPU forward kernel on')
        #with recorder.record(0.2): # Sample every 500 ms
        #  amult = time.time()
        #  for _ in range(10):
        x = self.att(x, edge_index)

        x = x.relu() 
        #dmult =  time.time()   
         #if (config.profiling == 1):
        #print(recorder.frame)
        #x = self.att(x, edge_index)
        #x = x.relu()        
        #lrelu = LeakyReLU(0.1)
        #x = lrelu(x)
        if (config.profiling == 1):
         print('conv1 layer timing : {:.5f}ms'.format(1000/1*(time.time() - rmult)))
        rmult = time.time()
        x = self.conv2(x, edge_index)
        if (config.profiling == 1):
         print('conv2 layer timing : {:.5f}ms'.format(1000/1*(time.time() - rmult)))
        
        #x = x.relu()
        #x = self.conv3(x, edge_index)
        
        # 2. Readout layer
        #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        #x = x.relu()
        #x = lrelu(x)

        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.dropout(x, p=0.5, training=self.training)
        rmult = time.time()
        x = self.lin(x)
        if (config.profiling == 1):
         print('Linear layer timing : {:.5f}ms'.format(1000/1*(time.time() - rmult)))
        #x = self.lin2(x)
        #x = self.lin3(x)
        
 
        #return F.log_softmax(x, dim=1)
        
        return x


model = GAT(hidden_channels=config.hidden_channels)
print(model)

# In[5]:


from torch.nn import Linear
from torch.nn import LeakyReLU
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops,add_self_loops,sort_edge_index,degree


import pandas as pd

def sym_norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def sym_norm2(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

    
    # Calculate node degrees
    node_degrees = degree(edge_index[1], num_nodes=num_nodes)

    #print('max_degree')
    #print(torch.max(node_degrees))

  
    
    fill_value = math.trunc(math.log2(average_node_degree)) if not improved else 2
    
    #print("fill value")
    #print(fill_value)
    #fill_value = torch.max(node_degrees) if not improved else 2
    #fill_value = 2 if not improved else 2 #32

    
    #edge_weight = torch.zeros((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
    
    #print("edge index")
    #print(edge_index)
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
    
    edge_index, edge_weight = sort_edge_index(edge_index, edge_weight) #make sure that self loops are in order
    
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]



class GAT_PYNQ(torch.nn.Module):
    

    def __init__(self, hidden_channels,head_count):
        super(GAT_PYNQ, self).__init__()
        print("GAT_PYNQ INIT")
        #torch.manual_seed(12345)
        

        #self.att1 = GATConv(dataset.num_node_features, hidden_channels)
        self.att2 = GCNConv_SGRACE(dataset.num_node_features, hidden_channels)
        #self.att2 = GCNConv(dataset.num_node_features, hidden_channels)
        #self.att2 = SAGEConv_SGRACE(dataset.num_node_features, hidden_channels)
        #self.att2 = GATConv_SGRACE(dataset.num_node_features, hidden_channels,head_count,dropout=0.1, alpha=0.2, concat=False)
        #self.att2 = SAGEGAT_SGRACE(dataset.num_node_features, hidden_channels,head_count,dropout=0.1, alpha=0.2, concat=False)

        #self.conv21 = GATConv(hidden_channels, hidden_channels)
        self.conv22 = GCNConv_SGRACE(hidden_channels*head_count, hidden_channels)
        #self.conv22 = GCNConv(hidden_channels*head_count, hidden_channels)
        #self.conv22 = SAGEConv_SGRACE(hidden_channels*head_count, hidden_channels)
        #self.conv22 = GATConv_SGRACE(hidden_channels*head_count, hidden_channels,1)
        #self.conv22 = SAGEGAT_SGRACE(hidden_channels*head_count, hidden_channels,1)
        
 
        
        self.reluh = Relu_SGRACE()
        
        #self.lin2 = Linear(hidden_channels, dataset.num_classes, bias=False)
        
        #self.lin = Linear_SGRACE(hidden_channels, dataset.num_classes)
        
        #self.lin1 = Linear_SGRACE(hidden_channels, hidden_channels)
    
        self.lin2 = Linear_SGRACE(hidden_channels, dataset.num_classes)
        
        #self.lin = GATConv_SGRACE(hidden_channels, dataset.num_classes,1)
        
   
        
    def forward(self, x, edge_index):
        if (config.profiling==1):
         ptime = time.time()
            
        if(config.profiling==1):
         vtime = time.time();  
        
       
         #print("Normalizing adjacency")
      #global adj
        #adj = to_dense_adj(edge_index, edge_attr=norm)
        #adj=torch.squeeze(adj)

        #quantize adj

         
        #global pynq_adj 
        #pynq_adj = adj._to_sparse_csr()
        

        
        edge_index, norm = sym_norm2(edge_index,x.size(0),improved=False)
        
        #global adj
        adj = torch.sparse_coo_tensor(edge_index, norm) 
        
        if(config.min_output==0):  
         print("Quantize adjacency sparsity")
         #xaux = norm.detach().numpy()
         #y = sgrace.quantization_uqbits(xaux,sgrace.a_s,sgrace.a_z,config.w_qbits)
         #y = np.expand_dims(y, 1) 
         #sgrace.isSparse(y, y.shape[0],y.shape[1])
        
        dense = 0
        relu = 1
        #relu = 0 #when merging two layers OJO 
        #compute_attention = 0
        #sage=0
        #linear=0
        
        stream = 0
        if (config.profiling==1):
         fmult = time.time()
        
        #if (config.acc_deep==0):
        # x = self.att1(relu,x,edge_index)
        #else:
        
        #extract weights for the accelerator
        if(config.acc==1):
         srelul_l1 = self.reluh.srelu.data
         if(srelul_l1 < 0.0):
          srelul_l1 = 0.0    
         #srelul_l1 = 10.0
         #print("srelul_l1")
         #print(srelul_l1)   
         weights_l2 = self.conv22.weight.data
         if(config.total_layer_count>2):
          weights_l3 = self.lin2.weight_linear.data
          weights_l4 = self.lin2.weight_linear.data
         else:
          weights_l3 = self.lin2.weight.data  
          weights_l4 = None 
        else:    
         weights_l2 = None
         weights_l3 = None
         weights_l4 = None
        
        #print(weights_l2.shape)
        
        #print("in first layer")
        
        if(config.min_output==0):  
         print("First layer quantize features sparsity")
         #xaux = x.detach().numpy()
         #print("input features shape ",x.shape)   
         #y = sgrace.quantization_uqbits(xaux,sgrace.f_s,sgrace.f_z,config.w_qbits)
         #sgrace.isSparse(y, y.shape[0],y.shape[1])
        
        x = self.att2(stream,dense,relu,x,edge_index,norm,adj,srelul_l1,weights_l2,weights_l3,weights_l4)
        #x = self.att2(x,edge_index)
        
        #x = self.att2(compute_attention,stream,dense,relu,sage,linear,x,edge_index,norm,adj,weights_l2)
        #x = self.att2(config.compute_attention,stream,dense,relu,x,edge_index,norm,adj)
        
        #print("out form first layer")
        #print(x)
        
        if (config.profiling == 1):
         print('L1 layer time: {:.5f}ms'.format(1000*(time.time() - fmult)))
        
        if (config.profiling==1):
         fmult = time.time()
        #if (config.layer_count == 1): #if more than 1 layer then inference mode and x not available until layer 2 completes
        x = self.reluh(x) #enable this to unmerge relu and take into account that relu is done in hardware 
        #x = x.relu()
        dense = 1 #hardwware execution mode for layer 2. 1 => fea dense

        if(config.min_output==0):
         print("SECOND LAYER ON")

        ######dense X
        #xaux = x.detach().numpy()

        #if(config.hardware_quantize == 0):
        # support_xaux = quantization_uqbits(xaux,f_s2,f_z2,f_qbits) * (2**f_align)
        #else:
        # support_xaux = xaux
        
        if(config.min_output==0):  
         print("Second layer quantize features sparsity")
         xaux = x.detach().numpy()
         y = sgrace.quantization_uqbits(xaux,sgrace.f_s,sgrace.f_z,config.w_qbits)
         sgrace.isSparse(y, y.shape[0],y.shape[1])
        #print(xaux)

        #print(support_xaux)
        #values_fea_buffer[0:(x.shape[0]*x.shape[1])] = (support_xaux.reshape(1,x.shape[0]*x.shape[1])) * (1<<f_align)
        #config.values_fea_buffer[0:(x.shape[0]*x.shape[1])] = (support_xaux.reshape(1,x.shape[0]*x.shape[1]))# * (2**f_align) #cuidado    
      
   
        relu = 1
        stream = 0
        #compute_attention = 0
        #sage = 0
     
        if (config.profiling == 1):
         print('Relu time: {:.5f}ms'.format(1000*(time.time() - fmult)))


        if (config.profiling == 1):
         fmult = time.time()
 
        #if (config.acc_deep==0):
        # x = self.conv21(x,edge_index)
        #else:
        #x = self.conv22(x,edge_index)
        x = self.conv22(stream,dense,relu,x,edge_index,norm,adj,srelul_l1,weights_l2,weights_l3,weights_l4)

        
        
        #x = self.reluh(x) #enable this to unmerge relu and take into account that relu is done in hardware 
            
        #x = self.conv22(config.compute_attention,stream,dense,relu,x,edge_index,norm,adj)
        

        if (config.profiling == 1):
         print('L2 layer time: {:.5f}ms'.format(1000*(time.time() - fmult)))


        # 2. Readout layer
        if (config.profiling == 1):
         fmult = time.time()
        x = x.float()

        # 3. Apply a final classifier
  
        x = F.dropout(x, p=0.5, training=self.training)
        #print(x.shape)
        
        if(config.min_output==0):  
         print("Linear layer quantize features sparsity")
         xaux = x.detach().numpy()
         y = sgrace.quantization_uqbits(xaux,sgrace.f_sl,sgrace.f_zl,config.w_qbitsl)
         sgrace.isSparse(y, y.shape[0],y.shape[1])
        
        if(config.min_output==0):
         print("LINEAR LAYER ON")
        
        
        dense = 1
        relu = 0  
        stream = 0
        #compute_attention = 0
        
 
        
        if (config.profiling == 1):
         fmult = time.time()
        #x = self.lin(compute_attention,stream,dense,relu,sage,linear,x,edge_index,norm,adj,weights_l2
        #x.data.fill_(1.0)
        write_linear = 0
        if(write_linear==1):
         df = pd.DataFrame(x.detach().numpy()) #convert to a dataframe
         df.to_csv("./linear_in.txt",index=False,header=False) #save to file 
         weights_linear = self.lin2.weight.data 
         #weights_linear = self.lin2.weight_linear.data
         #weights_linear = np.transpose(weights_linear)
         #weights_linear.data.fill_(1.0)
         df = pd.DataFrame(weights_linear.detach().numpy()) #convert to a dataframe
         df.to_csv("./linear_weights.txt",index=False,header=False) #save to file  
        #x = self.lin2(x)

        #x = self.lin1(stream,dense,relu,x,edge_index,norm,adj,weights_l2,weights_l3,weights_l4)
        
        dense = 1
        relu = 0  
        stream = 0
        
        
        x = self.lin2(stream,dense,relu,x,edge_index,norm,adj,weights_l2,weights_l3,weights_l4)
        
        if(write_linear==1):
         df = pd.DataFrame(x.detach().numpy()) #convert to a dataframe
         df.to_csv("./linear_out.txt",index=False,header=False) #save to file  
         sys.exit()
        
        if (config.profiling == 1):
         print('Linear layer time: {:.5f}ms'.format(1000*(time.time() - fmult)))


        if (config.profiling == 1):
         print('Readout time: {:.5f}ms'.format(1000*(time.time() - fmult)))
        #print(x)
        
        if (config.profiling == 1):
          print('Model time {:.5f}ms'.format(1000*(time.time() - ptime)))

        return x

model = GAT_PYNQ(config.hidden_channels,config.head_count)
print(model)

# In[6]:


#from IPython.display import Javascript
#display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from scipy import sparse
from torch_geometric.utils import to_dense_adj


if (custom==0):
  model = GAT(config.hidden_channels)
else:        
  model = GAT_PYNQ(config.hidden_channels,config.head_count)
  #model_path = "models/model_" + dataset_sel + "_fp.ptx" #load best model
if (preload == 1):  
  #model_path = "models/model_" + dataset_sel + "_fake_1bit.ptx" #load best model
  model_path = "models/model_" + dataset_sel + "_2l_2bit.ptx" #load best model
  #model_path = "/media/josnu02/hd1/josnu02/cuda_performance/vision_transformer_cifar10_fake_8bit.ptx" #load best model
  model.load_state_dict(torch.load(model_path),strict=True)
    
  #Final results. The different learning rates are very important for different quantizations.  
  #for 8-4 bit 
  #optimizer = torch.optim.Adam(model.parameters(),  lr=0.005)
  #for 2-1 bit optimizer 
  #optimizer = torch.optim.Adam(model.parameters(),  lr=0.05)
  #optimizer = torch.optim.Adam(model.parameters(),  lr=0.1)
    
if (preload == 1):
  print("Using very low learning rate")
  lr = 0.0001
#elif (config.w_qbits>2 or config.acc==0):
elif (config.w_qbits>0):    
  print("Using low learning rate")
  lr=0.01
  #lr=0.005
else:
  print("Using high learning rate")
  lr=0.1

# Freeze all layers
#for param in model.parameters():
#    param.requires_grad = False    
    
#model.att2.weight.requires_grad = False
#model.conv22.weight.requires_grad = False

#for name, layer in model.named_children():
#    if name in ['lin']:
#        for param in layer.parameters():
#            param.requires_grad = True


#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)


#optimizer = torch.optim.RAdam(model.parameters(), lr=lr)


optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.001)

#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

#optimizer = torch.optim.Adam([{'params': model.lin.parameters(),'lr': 0.01}],lr=lr)

#optimizer = torch.optim.Adam([{'params': model.lin.parameters(),'lr': 0.01},
#                              {'params': model.att2.parameters(), 'lr': lr},
#                              {'params': model.conv22.parameters(), 'lr': lr}])

#optimizer = torch.optim.Adam([{'params': model.lin.parameters(),'lr': 0.01}],lr=0.01)





#optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
#optimizer = torch.optim.Adam(model.parameters(),  lr=0.1) #GAT benefits from this ?
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
       
    
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.NLLLoss()

def zero_count(array, n): 
 counter = 0
 # Count number of zeros
 # in the matrix
 for i in range(0, n):
  if (array[i] == 0):
   counter = counter + 1
 print("total values ",n)
 print("zero values ",counter)
 return (counter > ((n) // 2))


def accuracy(x, labels, dataset_sel):

 x1 = np.equal(x, labels)
 x2 = np.sum(x1)

 if isinstance(x, list):
     acc = x2 / len(x)
 else:
     acc = x2 / x.size
 return acc


def train():
  model.train()
  #for bid, batch in enumerate(train_loader):
  #     batchsize = batch.x.shape[0]
  #     print("graph size is ", batch.x.shape[0]) 
  for data in train_loader:  # Iterate in batches over the training dataset.
       tmult = time.time()
       #global num_nodes_h
       #num_nodes_h = batchsize
       if (custom==0):
        #print("Running TRAIN with full precision")
        out = model(data.x, data.edge_index)  # Perform a single forward pass. 
        #out = model(batch.x, batch.edge_index)  # Perform a single forward pass. 
       else:
        out = model(data.x, data.edge_index)
        #out = model(batch.x, batch.edge_index)
        if (config.profiling == 1):
         print('Forward train time: {:.5f}s'.format(time.time() - tmult))
       #print(data.y.shape)
       loss = criterion(out[data.train_mask], data.y[data.train_mask])
       #loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])

       tmult = time.time()
       loss.backward()  # Derive gradients.
       #for param_group in optimizer.param_groups:
           # print("current learning rate: ", param_group['lr'])
       if (config.profiling == 1):
        print('backward time: {:.5f}s'.format(time.time() - tmult))
       optimizer.step()  # Update parameters based on gradients.
       current_lr = optimizer.param_groups[0]["lr"]
       #print("current learning rate: ", current_lr)
       #scheduler.step()
       optimizer.zero_grad()  # Clear gradients.

def test(loader,split):
   model.eval()
     
   preds_l = []
   labels_l = []

   for bid, batch in enumerate(loader):  # Iterate in batches over the training/test dataset.
   #    batchsize = batch.x.shape[0]
   #    global num_nodes_h 
   #    num_nodes_h = batchsize
       #print("graph size is ", batch.x.shape[0]) 
   #for data in loader:  # Iterate in batches over the training/test dataset.
       if (config.profiling==1):
        tmult = time.time()
       if (custom==0):
        #out = model(data.x, data.edge_index) 
        out = model(batch.x, batch.edge_index) 
       #print("Test")
       else:
        #out = model(data.x, data.edge_index)
        #print("input data is")
        #print(batch.x)
        out = model(batch.x, batch.edge_index) 
       #print(out)
       if (config.profiling == 1):
        print('Forward test time: {:.5f}s'.format(time.time() - tmult))
            
            
       if (split == "train"):
         preds_l.append(out[batch.train_mask].detach().numpy())
         labels_l.append(batch.y[batch.train_mask].detach().numpy())
       elif (split == "test"):
         preds_l.append(out[batch.test_mask].detach().numpy())
         labels_l.append(batch.y[batch.test_mask].detach().numpy()) 
       preds = np.argmax(np.concatenate(preds_l), axis=1)
        
       #if (split == "train"):
       # preds_l.append(out[data.train_mask].detach().numpy())
       # labels_l.append(data.y[data.train_mask].detach().numpy())
       #elif (split == "test"):
       # preds_l.append(out[data.test_mask].detach().numpy())
       # labels_l.append(data.y[data.test_mask].detach().numpy()) 
       #preds = np.argmax(np.concatenate(preds_l), axis=1)

    
   pred_acc = accuracy(preds, np.concatenate(labels_l), dataset_sel)
            
   return pred_acc  # Derive ratio of correct predictions.


#print('Running inference only with train and test data sets')

#if (training==0):

# for epoch in range(20):
    
#   amult = time.time()
#   test_acc = test(test_loader,"test") 
#   train_acc = test(train_loader,"train")
#   best_acc = test_acc
#   print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {(time.time() - amult):.4f}')

#quit()
#exit()
#raise SystemExit("Stop right there!")
    



if (training==1):
    
 print("TRAINING")
    
 best_acc = 0
 best_epoch = 0
 for epoch in range(num_epochs):
       
  amult = time.time()
    #print("Running TRAIN")
  train()
  #print("Running TEST")
  #train_acc = test(train_loader,"train") #remove to speed up
  #train_acc = 0
  test_acc = test(test_loader,"test") 
    
  if (test_acc > best_acc):
   best_acc = test_acc
   best_epoch = epoch
   print("best model saved") 
   model_path = "models/model_" + dataset_sel + ".ptx" #save best model
   torch.save(model.state_dict(), model_path)

  print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}, Time: {(time.time() - amult):.4f}')
   
 print(' ')
  
 print('Best training accuracy: ', best_acc)
 print('Best training epoch: ', best_epoch)
    
 #test result of training



# In[7]:


print("TESTING only")

#model_path = "models/model_" + dataset_sel + "_089.ptx" #load best model

#model_path = "models/model_" + dataset_sel + "_8b_8b_16_sparse.ptx" #load best model
#model_path = "models/model_" + dataset_sel + ".ptx" #load best model
#model_path = "models/model_" + dataset_sel + "_8b_16_no_relu.ptx" #load best model
#model_path = "models/model_" + dataset_sel + "_8b_64_relu.ptx" #load best model
#model_path = "models/model_" + dataset_sel + "_1b_16_no_relu.ptx" #load best model
model_path = "models/model_" + dataset_sel + ".ptx" #load best model
model.load_state_dict(torch.load(model_path),strict=False)
sgrace.my_ip.register_map.load_weights = 1

for epoch in range(5):
    
  amult = time.time()
  test_acc = test(test_loader,"test") 
  #train_acc = test(train_loader,"train") 
  print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}, Time: {(time.time() - amult):.4f}')
  #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {(time.time() - amult):.4f}')
  best_acc = test_acc

#sgrace.my_ip.register_map.load_weights = 0

#for epoch in range(1):
    
#  amult = time.time()
#  test_acc = test(test_loader,"test") 
#  #train_acc = test(train_loader,"train") 
#  print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}, Time: {(time.time() - amult):.4f}')
# #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {(time.time() - amult):.4f}')
#  best_acc = test_acc




from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx



#print("Number of values in weight matrix")
#print(dataset.num_features*16)
#print(B_buffer[0:(dataset.num_features*16)])


    

##dense_adj = adj.to_dense()
#dense_adj[dense_adj > 0] = 1
##plt.imshow(dense_adj, cmap='binary')
##plt.colorbar()
##plt.title("Adjacency Matrix")
##plt.show()

##print("positive, zero and negative adj counts")
##pos_a = (dense_adj > 0).sum()
##zero_a = (dense_adj == 0).sum()
##neg_a = (dense_adj < 0).sum()
##print(pos_a," ",zero_a," ",neg_a) 

##print("positive, zero and negative feature counts")
##pos_f = (support_x > 0).sum()
##zero_f = (support_x == 0).sum()
##neg_f = (support_x < 0).sum()
##print(pos_f," ",zero_f," ",neg_f) 
    

##mybins = []
##for k in range(0,(2**(f_qbits))+2):
#for k in range(-128,128):    
#for k in range(-8,7):
    #print(k) 
##    mybins += [k]
#mybins = [-2, -1, 0, 1]
#y = B_buffer[0:(dataset.num_features*16)]
##y = quantization_uqbits(norm,a_s,a_z,a_qbits)
#print('max_degree')
#print(torch.max(node_degrees))
#y = model.att2.weight.data*128
#y = y.reshape(1,adj_size)
##print("Adjacency values")
##print(y)
##y_q = y #quantization_qbits(y,w_s,w_z,w_qbits)

#y = B_buffer[0:(dataset.num_features*16)]


#print("Number of selected value")
#print(np.count_nonzero(y == -8))

##plt.figure(figsize=(10, 10))
##plt.xlabel('Adjacency')
##plt.ylabel('Frequency')
#counts, bins, bars = plt.hist(y, bins=256)
##counts, bins, bars = plt.hist(y_q,mybins)
#plt.yticks(np.arange(0, 12000, step=500))
#plt.xticks(mybins,horizontalalignment='center',fontsize=12,rotation=90)
#plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=32))

#plt.xticks(mybins,horizontalalignment='right',fontsize=12,rotation=90)
##plt.xticks(mybins[::8],horizontalalignment='right',fontsize=12,rotation=90)
#plt.xticks(range(0,2**(a_qbits)+2,1)[::8],rotation=90)
##plt.show()




mybins = []
for k in range((-2**(config.w_qbits-1)),(2**(config.w_qbits-1))+2):
#for k in range(-128,128):    
#for k in range(-8,7):
    #print(k) 
    mybins += [k]
#mybins = [-2, -1, 0, 1]
#y = B_buffer[0:(dataset.num_features*16)]
y = sgrace.quantization_qbits(model.att2.weight.data,sgrace.w_s,sgrace.w_z,config.w_qbits)
#y = model.att2.weight.data
#y = model.att2.weight.data*128
y = y.reshape(1,dataset.num_features*config.hidden_channels)
#torch.set_printoptions(threshold=np.inf)
#print("Weights L1 bins")
#print(mybins)
#print("Weights L1 float values before q")
#print(model.att2.weight.data)
#print("Weights L1 float values after q")
#y_d = y*w_s_o/(2**frac_bits_o)
#y_d = y*w_s
#print(y_d)
y_q = y #quantization_qbits(y,w_s,w_z,w_qbits)

#y = B_buffer[0:(dataset.num_features*16)]


print("First layer quantize weights sparsity")
sgrace.isSparse(y_q, y_q.shape[0],y_q.shape[1])
print('max/min weight')
print(torch.max(model.att2.weight.data))
print(torch.min(model.att2.weight.data))

#print("Number of selected value")
#print(np.count_nonzero(y == -8))

plt.figure(figsize=(10, 10))
plt.xlabel('Weights L1')
plt.ylabel('Frequency')
#counts, bins, bars = plt.hist(y, bins=256)
counts, bins, bars = plt.hist(y_q,mybins)
#plt.yticks(np.arange(0, 12000, step=500))
#plt.xticks(mybins,horizontalalignment='center',fontsize=12,rotation=90)
plt.xticks(mybins[::8],horizontalalignment='right',fontsize=12,rotation=90)
plt.savefig('fig1.png')
#plt.show()

#plt.bar(y_q,mybins, align='center')
#plt.gca().set_xticks(labels)
#plt.show()





mybins = []
for k in range((-2**(config.w_qbits-1)),(2**(config.w_qbits-1))+2):
#for k in range(-128,128):
#for k in range(-8,7):
    #print(k)  
    mybins += [k]
#mybins = [-2, -1, 0, 1]
#y = D_buffer[0:(num_nodes_h*16)]
y = sgrace.quantization_qbits(model.conv22.weight.data,sgrace.w_s2,sgrace.w_z2,config.w_qbits)
#y = model.conv22.weight.data
y = y.reshape(1,config.hidden_channels*config.hidden_channels)
#print("Weights L2 values")
#print(y)
#print(mybins)
y_q = y #quantization_qbits(y,f_s,f_z,f_qbits)

 
print("Second layer quantize weights sparsity")
sgrace.isSparse(y_q, y_q.shape[0],y_q.shape[1])

print('max/min weight')
print(torch.max(model.conv22.weight.data))
print(torch.min(model.conv22.weight.data))

#y = B_buffer[0:(dataset.num_features*16)]


#print("Number of selected value")
#print(np.count_nonzero(y == -8))

plt.figure(figsize=(10, 10))
#plt.xlabel('Output  values')
plt.xlabel('Weights L2')
plt.ylabel('Frequency')
#counts, bins, bars = plt.hist(y, bins=256)
counts, bins, bars = plt.hist(y_q,mybins)
#plt.yticks(np.arange(0, 12000, step=500))
#plt.xticks(mybins,horizontalalignment='center',fontsize=12,rotation=90)
plt.xticks(mybins[::8],horizontalalignment='center',fontsize=12,rotation=90)
#plt.show()
plt.savefig('fig2.png')


mybins = []
for k in range((-2**(config.w_qbitsl-1)),(2**(config.w_qbitsl-1))+2):
#for k in range(-128,128):
#for k in range(-8,7):
    #print(k)  
    mybins += [k]
#mybins = [-2, -1, 0, 1]
#y = D_buffer[0:(num_nodes_h*16)]
if(config.total_layer_count==3):
 y = sgrace.quantization_qbits(model.lin2.weight_linear.data,sgrace.w_sl,sgrace.w_zl,config.w_qbitsl)
else:
 y = model.lin.weight.data
#y = model.conv22.weight.data
y = y.reshape(1,config.hidden_channels*dataset.num_classes)
#print("Weights L2 values")
#print(y)
#print(mybins)
y_q = y #quantization_qbits(y,f_s,f_z,f_qbits)



print("Linear layer quantize weights sparsity")
sgrace.isSparse(y_q, y_q.shape[0],y_q.shape[1])

print('max/min weight')
print(torch.max(model.lin2.weight_linear.data))
print(torch.min(model.lin2.weight_linear.data))

#y = B_buffer[0:(dataset.num_features*16)]


#print("Number of selected value")
#print(np.count_nonzero(y == -8))

plt.figure(figsize=(10, 10))
#plt.xlabel('Output  values')
plt.xlabel('Weights Linear')
plt.ylabel('Frequency')
#counts, bins, bars = plt.hist(y, bins=256)
counts, bins, bars = plt.hist(y_q,mybins)
#plt.yticks(np.arange(0, 12000, step=500))
#plt.xticks(mybins,horizontalalignment='center',fontsize=12,rotation=90)
plt.xticks(mybins[::8],horizontalalignment='center',fontsize=12,rotation=90)
plt.savefig('fig3.png')
#plt.show()


#print("MAX FEA INTERNAL VALUE linear no acc", sgrace.global_max_input)
#print("MIN FEA INTERNAL VALUE linear no acc", sgrace.global_min_input)

#print("MAX FEA INTERNAL VALUE layer 1", sgrace.cur_max_fea)
#print("MAX FEA INTERNAL VALUE layer 2", sgrace.cur_max_fea2)

#print("MAX FEA INTERNAL VALUE layer 1", cur_max_fea)
#print("MAX FEA INTERNAL VALUE layer 2", cur_max_fea2)
#print("Use this to adjust your hardware ITYPE width")
#print("Current attention is:")
#print(attention_buffer)
#print(bins)
#print(counts)


# ## (Optional) Exercise
# 
# Can we do better than this?
# As multiple papers pointed out ([Xu et al. (2018)](https://arxiv.org/abs/1810.00826), [Morris et al. (2018)](https://arxiv.org/abs/1810.02244)), applying **neighborhood normalization decreases the expressivity of GNNs in distinguishing certain graph structures**.
# An alternative formulation ([Morris et al. (2018)](https://arxiv.org/abs/1810.02244)) omits neighborhood normalization completely and adds a simple skip-connection to the GNN layer in order to preserve central node information:
# 
# $$
# \mathbf{x}_v^{(\ell+1)} = \mathbf{W}^{(\ell + 1)}_1 \mathbf{x}_v^{(\ell)} + \mathbf{W}^{(\ell + 1)}_2 \sum_{w \in \mathcal{N}(v)} \mathbf{x}_w^{(\ell)}
# $$
# 
# This layer is implemented under the name [`GraphConv`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv) in PyTorch Geometric.
# 
# As an exercise, you are invited to complete the following code to the extent that it makes use of PyG's `GraphConv` rather than `GCNConv`.
# This should bring you close to **82% test accuracy**.

# from torch_geometric.nn import GraphConv
# 
# 
# class GNN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super(GNN, self).__init__()
#         torch.manual_seed(12345)
#         self.conv1 = ...  # TODO
#         self.conv2 = ...  # TODO
#         self.conv3 = ...  # TODO
#         self.lin = Linear(hidden_channels, dataset.num_classes)
# 
#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.conv3(x, edge_index)
# 
#         x = global_mean_pool(x, batch)
# 
#         
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)
#         
#         return x
# 
# model = GNN(hidden_channels=64)
# print(model)

# from IPython.display import Javascript
# display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))
# 
# model = GNN(hidden_channels=64)
# print(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 
# for epoch in range(1, 201):
#     train()
#     train_acc = test(train_loader)
#     test_acc = test(test_loader)
#     print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# ## Conclusion
# 
# In this chapter, you have learned how to apply GNNs to the task of graph classification.
# You have learned how graphs can be batched together for better GPU utilization, and how to apply readout layers for obtaining graph embeddings rather than node embeddings.
# 
# In the next session, you will learn how you can utilize PyTorch Geometric to let Graph Neural Networks scale to single large graphs.
# 
# [Next: Scaling Graph Neural Networks](https://colab.research.google.com/drive/1XAjcjRHrSR_ypCk_feIWFbcBKyT4Lirs)
