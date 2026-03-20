1.SGRACE is a high-performance dataflow architecture for graph convolutional and attention networks that supports adaptive quantization and sparsity/pruning.

Layers supported in SGRACE include GCNConv, GATconv, SAGEConv and Linear. 

You can find more information on SGRACE capabilities and performance here:


SGRACE: Scalable Architecture for On-Device Inference and Training of Graph Attention and Convolutional Networks
https://ieeexplore.ieee.org/document/11108959


2.This release directory includes the HLS source code of a base SGRACE configuration.


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

Now go to the sgracex1 directory and perform simulation, C synthesis with this command:

**vitis_hls -f ./hls/gat/solution1/script.tcl**

Check script.tcl to make sure that the set_part command matches your device is correct or modify as needed.
The default part is xczu27dr-ffve1156-2-i that is the FPGA available in an RFSOC2x2 board.


HLS simulation should report that the results match. 

Once HLS synthesis and IP export has completed launch implementation and bitstream generation with this command:


**vivado -mode batch -source project_1.tcl**

Optionally modify this line as needed in project_1.tcl to set a new project name/directory 

**set _xil_proj_name_ "vivado"**

After completion all results are available under the new project name directory. 

The software directory contains a test jupyter notebook that can be used to test the design in the PYNQ FPGA board
and measure performance. The software directory also contains the sgrace_lib.py file that initializes and controls the accelerator.

The jupyter notebook uses this library to offload the GNN execution and its location is indicated in the notebook with:

sys.path.insert(1, '/home/xilinx/jupyter_notebooks/sgrace_lib')



 



