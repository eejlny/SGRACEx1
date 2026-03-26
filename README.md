<img width="360" height="140" alt="image" class="center-img" src="https://github.com/user-attachments/assets/984c9357-5526-429a-af1c-ce79fb895bbd" />


SGRACE is a high-performance dataflow architecture for graph convolutional and attention networks that supports adaptive quantization and sparsity. Input tensors for adjacency and features are presented to SGRACE in COO format and SGRACE uses sparse operations (i.e. SPMM) to process them. SGRACE performs all quantization/dequantization processes on-device and adaptively quantizes data from an input precision (i.e. float)  to 1/2/4/8-bit depending on data complexity and systems requirements.  The input precision depends on the data source and could be 12-bit for a ADC sensor or floats for a standard graph dataset.

Layers supported in SGRACE include GCNConv, GATconv, SAGEConv and Linear. 

SGRACE offers two main modes of operation training and inference. The hardware operates end-to-end in inference mode so with a single invocation the whole model is executed while in training mode each layer is executed by the hardware independently so activations can be sent to the backpropagation loop on a per layer basis.  

In training mode the accelerator operates with 8-bit precision that is used to emulate a target precision from 8 to 1 bit for features, adjacency and weights. The hardware operates within the backpropagation loop and implements a form of hardware-aware quantized training. Then, the resulting trained model can be used in inference with a customized and efficient pipeline for the precision selected during training. The FPGA logic is reconfigured to switch between the 8-bit training mode to downto a 1-bit inference mode, for example.

SGRACE is integrated with Pytorch and PYNQ and can be used to replaced pytorch geometric layers such as GCNConv with their sgrace equivalent GCNConv_sgrace. In order to use SGRACE you need Pytorch and other libraries installed in your PYNQ image. These are the frameworks and libraries that have been used:

pynq 3.0.1  
numpy 1.24.4  
torch 1.12.1  
torch-geometric 2.6.1  
torch_scatter 2.1.2  
torch_sparse 0.6.18  

After connection your to FPGA board running PYNQ you can install them running these commands in the FPGA board:
pip install torch==1.12.1

For torch-geometric, the main issues are with torch_sparse and torch_scatter. You should install them first using the following commands:

pip install --no-use-pep517 torch-sparse
pip install --no-use-pep517 torch-scatter

And then:
pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-1.12.1.html

If you encounter problems with NumPy, you can reinstall it with this command:
pip install "numpy<2" --force-reinstall

You can find more information on SGRACE capabilities and performance here:


SGRACE: Scalable Architecture for On-Device Inference and Training of Graph Attention and Convolutional Networks
https://ieeexplore.ieee.org/document/11108959

<img width="2469" height="1022" alt="image" src="https://github.com/user-attachments/assets/2603d852-0c80-464d-a5b8-29a1f976f685" />



This release directory includes the HLS source code of a base SGRACE configuration.


Steps to run full implementation with a base design with one thread done with Linux Ubuntu.

Clone the contents of this repository in a sgracex1 directory.

Setup path to the Vitis FPGA tools, for example:

**module load vitis/2022.1**

or

**source <path to tools>/Xilinx/Vitis/2022.1/settings64-Vitis.sh**

edit matrix.h located the in the src directory and modify the following lines if needed:

#define MAX_N    4096 // max number of input nodes in the graph. For example the cora dataset has 2708 nodes
#define MAX_M    2048 // max number of features in each input node. For example the cora dataset has 1433 features per node;
#define MAX_P    16  // number of hidden channels. 

Now go to the hardware directory and perform simulation, C synthesis with this command:

**vitis_hls -f ./script.tcl**

Check script.tcl to make sure that the set_part command matches your device is correct or modify as needed.
The default part is xczu27dr-ffve1156-2-i that is the FPGA available in an RFSOC2x2 board.


HLS simulation should report that the results match. 

Once HLS synthesis and IP export has completed launch implementation and bitstream generation with this command:


**vivado -mode batch -source ./project_1.tcl**

Optionally modify this line as needed in project_1.tcl to set a new project name/directory 

**set _xil_proj_name_ "vivado"**

After completion all results are available under the new project name directory. 

The software directory contains python scripts that can be used to test the design in the PYNQ FPGA board
and measure performance. 

Open demo_sgrace.py, make sure that dataset is cora.
dataset_sel = "Cora"
dataset = Planetoid(root="data/Planetoid", name=dataset_sel, split="full", transform=transform) 


The software directory also contains the sgrace.py file that initializes and controls the accelerator.


The demo_sgrace script uses this library to offload the GNN execution and its location is indicated in the notebook with:

sys.path.insert(1, '/home/xilinx/jupyter_notebooks/sgrace_lib')

Make sure this location matches your PYNQ system and store sgrace.py and config.py in that location.

Make sure training = 1 in demo_sgrace.py so the hardware will be used in training mode.


On the board prepare the PYNQ environment with:

sudo su
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh

Launch execution with:

python3 demo_sgrace.py

After the training run the accuracy reached will the around 0.852 with the cora dataset with 16 hidden channels. You can experiment with wider configurations by simply replacing #define MAX_P    16  with 64, for example. You can also support larger graphs by modifying MAX_N and MAX_M. As expected more channels and larger graphs have a significant impact on complexity specially on BRAM usage. You can reduce the impact on complexity by reducing the number of bits used to store the different parameters.  

Now you can open demo_sgrace.py and set training = 0 and run the script again.

In this inference only mode the model will the executed end-to-end on the hardware fully using the streaming dataflow with a single invocation.
The model saved from training will be loaded and used and the accuracy should be the same.

To obtain performance profiling data use profiling = 1 in config.py. The model execution is shown as:

Accelerator forward kernel n-layer time: 1.81174ms

This represents the execution time of the model in hardware from inputs to final classification. The other times refer to python execution time that are not hardware accelerated. 
 










 



