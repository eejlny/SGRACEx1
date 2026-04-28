<img width="360" height="140" alt="image" class="center-img" src="https://github.com/user-attachments/assets/984c9357-5526-429a-af1c-ce79fb895bbd" /> 

SGRACE (Scalable GRaph Attention and Convolutional Engine) is a high-performance dataflow architecture for graph convolutional and attention networks that supports adaptive quantization and sparsity. Input tensors for adjacency and features are presented to SGRACE in COO format and SGRACE uses sparse data operations (i.e. SPMM) to process them. SGRACE performs all quantization/dequantization processes on-device and adaptively quantizes data from an input precision (i.e. float)  to 1/2/4/8-bit depending on data complexity and systems requirements.  The input precision depends on the data source and could be 12-bit for a ADC sensor or floats for a standard graph dataset.

Layers supported in SGRACE include GCNConv, GATconv, SAGEConv and Linear but the flexibility of the hardware accelerator means that other configurations are possible for GINConv, SAGE-GAT etc. SGRACE is built around 4 compute engines: agregation engine, combination engine, attention engine and linear engine arranged in a parallel and dataflow configurations as shown in the figure below. SGRACE uses a global formulation to maximize performance and the flexibility of the hardware means that multiple message passing layers can be implemented. 

SGRACE offers two main modes of operation training and inference. The hardware operates end-to-end in inference mode so with a single invocation the whole model is executed while in training mode each layer is executed by the hardware independently so activations can be sent to the backpropagation loop on a per layer basis.  

In training mode the accelerator operates with 8-bit precision that is used to emulate a target precision from 8 to 1 bit for features, adjacency and weights. The hardware operates within the backpropagation loop and implements a form of hardware-aware quantized training. Then, the resulting trained model can be used in inference with a customized and efficient pipeline for the precision selected during training. The FPGA logic is reconfigured to switch between the 8-bit training mode downto a 1-bit inference mode, for example. 

SGRACE has been targeted to Zynq FPGA boards running AMD PYNQ. In the Zynq FPGA board SGRACE is integrated with Pytorch and AMD PYNQ  and can be used to replaced pytorch geometric layers such as GCNConv with their sgrace equivalent GCNConv_sgrace. In order to use SGRACE you need Pytorch and other libraries installed in your PYNQ image. These are the frameworks and libraries that have been used:

pynq 3.0.1  
numpy 1.24.4  
torch 1.12.1  
torch-geometric 2.6.1  
torch_scatter 2.1.2  
torch_sparse 0.6.18  

After connection your to FPGA board running PYNQ and with internet available you can install them running these commands in the FPGA board:

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

Most of the tests have been done with Vitis 2022.1 version and Pynq 2.7.0.

Note that Vitis version 2024 seems to have issues with the dataflow compilation although version 2025 does not report errors during compilation. 

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


To enable the attention engine set GAT_ENABLE to 1 
#define GAT_ENABLE 1 //implement support for GAT

Now go to the hardware directory and perform simulation, C synthesis (optional cosimulation check the script cosim command) with this command:

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

Open notebook demo_sgrace.pynb, make sure that dataset is cora.
dataset_sel = "Cora"
dataset = Planetoid(root="data/Planetoid", name=dataset_sel, split="full", transform=transform) 


The software directory also contains the sgrace.py/config.py library files that compiles GNN models for the accelerator, initializes and controls the accelerator. 

Verify that config.py uses a standard 8-bit quantization mode for linear and graph layers:

w_qbits = 8
w_qbitsl = 8

sgrace.py will use these values to set up all the quantization constants.

Other important parameters present in config.py include:

hidden_channels = 64 #how many hidden channels. 
instant_layer_count = 1 #how many layers to process in each SGRACE (core) hardware call.
total_layer_count = 3 #how many total SGRACE layers.

The demo_sgrace notebook uses sgrace.py/config.py files to offload the GNN execution and its location is indicated in the notebook with:

sys.path.insert(1, '/home/xilinx/jupyter_notebooks/sgrace_lib')

Make sure this location matches your PYNQ system and store sgrace.py and config.py in that location.

Make sure training = 1 in demo_sgrace.pynb so the hardware will be used in training mode.

If you are using a py file instead of the notebook pynb on the board prepare the PYNQ environment with:

sudo su
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh

In demo_sgrace.pynb select your predefined model in cell 1. For example pynq_class = "GCN" will instantiate a GCN model with two gcn layers and one fully connected layer.

<img width="896" height="509" alt="{A64FC7F9-1D1A-405D-BE86-BEFFA7197328}" src="https://github.com/user-attachments/assets/f7d1d327-7fb7-4b1d-b9d5-e9a6e327c180" />
<img width="832" height="502" alt="{26091111-457D-444B-BD5E-D49EDC318142}" src="https://github.com/user-attachments/assets/9805ec65-6cea-4a1e-bb30-72a0657437f0" />


Run the notebook, the pynq_class is compiled into a SGRACE opbytes and the following information is printed:

<img width="680" height="315" alt="{251958C9-DDBD-434C-9360-A867F731E3A1}" src="https://github.com/user-attachments/assets/a41ec5f0-7acd-40c4-ab29-4ae1cd6a3d59" />


model_buffer contains the SGRACE dataflow configuration descriptors and it is similar to the instructions of a CPU. The sgrace compiler derives automatically these descriptors from the contents of the model described in pynq_class. It is possible to create custom pynq classes for the compiler with more layers or combining different layer types, for example. 

A number of predefined pynq_classes are available such as GAT, SAGE etc that build models using _sgrace layers. You can select them simply by changing pynq_class = "GAT", for example.

Note SGRACE performs all quantization/dequantization on device and inputs and outputs floating point numbers. Quantization parameters are written to the accelerator and for quantization to be effective quantization paramereters must be optimized depending on the data set and quantization target. You can observe the max and min values that control quantization for adjacency, weights and features for tested data sets in sgrace.py. For example for 8-bit search for "if(config.w_qbits == 8):". 

After training  the accuracy reaches around 0.852 with the cora dataset with 16 hidden channels. You can experiment with wider configurations by simply replacing #define MAX_P    16  with 64, for example. You can also support larger graphs by modifying MAX_N and MAX_M. As expected more channels and larger graphs have a significant impact on complexity specially on BRAM usage. You can reduce the impact on complexity by reducing the number of bits used to store the different parameters.  

Now you can open demo_sgrace.pynb and set training = 0 and run the script again. The SGRACE compiler generates new instructions to change the dataflow configuration:

<img width="699" height="311" alt="{A8A270CA-322D-4D5F-9610-69EC0F11E194}" src="https://github.com/user-attachments/assets/4fbfa86e-37d5-4b19-b9b2-3bfcdfe19c83" />

Note that this only changes the contents of model_buffer and the same FPGA bit file is used. In this case the changes in the program affect how streaming of data from one layer output to the next layer input is done to avoid interactions with DDR memory. In this inference only mode the model will be executed end-to-end on the FPGA fully using the streaming dataflow with a single invocation. The model weights saved from training will be loaded and the accuracy should be equivalent.

The general SGRACE design principle is that the same hardware can execute any model independently of the layer types or the number of layers. For example, the 8-bit configuration can execute any quantization target from 1 to 8 bit but more optimized hardware can be obtained if the implementation is done for lower quantization target. 

To obtain performance profiling data use profiling = 1 in config.py. The model execution is shown as:

Accelerator forward kernel n-layer time: ~1.6 ms (end-to-end performance with 2 GCN layers and 1 Linear layer) an 16 channels.

This represents the execution time of the model in hardware from inputs to final classification. The other times reported refer to python execution time that are not hardware accelerated. 

Higher performance is possible with multithreaded configurations with up to 4 threads possible in a Zynq ultrascale device. The design is compatible with Versal/Alveo boards although HBM/BRAM optimizations could be needed for these devices. 

The sgrace compiler is critical to facilitate the implementation of more complex models such as graph-transformers etc with the accelerator. Current work is targeting how to extend the framework to these layer types and how to obtain quantization parameters automatically. 

Contact as if you want to know more and explore possible collaborations (jose.nunez.yanez@upm.es).  

If you find this work useful please check these papers:

J. Nunez-Yanez and H. Mousanejad Jeddi, "SGRACE: Scalable Architecture for On-Device Inference and Training of Graph Attention and Convolutional Networks," in IEEE Transactions on Very Large Scale Integration (VLSI) Systems, vol. 33, no. 11, pp. 2929-2939, Nov. 2025, doi: 10.1109/TVLSI.2025.3591522. https://ieeexplore.ieee.org/document/11108959

Nunez-Yanez, Jose and Olle Hansson. “On-Device Inference and Training Acceleration of Graph Neural Networks with Quantized Arithmetic.” 2025 International Conference on Field Programmable Technology (ICFPT) (2025): 1-9. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11363780


 










 



