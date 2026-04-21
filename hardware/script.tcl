############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
############################################################
open_project gnn
set_top mmult_top
add_files ./src/kernelMatrixmult.h
add_files ./src/kernelMatrixmult_all.cpp
add_files ./src/matrix_mult.h
add_files -tb ./src/main_float_stream_test.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xczu28dr-ffve1156-1-e}
create_clock -period 3 -name default
config_interface -s_axilite_auto_restart_counter 1
config_export -format ip_catalog -rtl verilog
source "./directives.tcl"
csim_design -O
#csynth_design
#cosim_design -O -disable_deadlock_detection
#export_design -rtl verilog -format ip_catalog
exit
