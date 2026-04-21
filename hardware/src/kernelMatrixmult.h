/*===============================================================================
* This file is part of the SGRACE GNN accelerator
* has been written at Linkoping/UPM University
* Author : Jose Nunez-Yanez
*Copyright (C) 2026 Jose Nunez-Yanez
*Licensed under the MIT license. See LICENSE file in the project root for details
===============================================================================
*/

#ifndef KERNELMATRIXMULT_H_
#define KERNELMATRIXMULT_H_




void kernelmult1(
bool load_weights,
int	beta_qu,
int f_align,
float quantization_scale_adj,
float quantization_scale_fea[5],
float quantization_scale_w[5],
float quantization_scale_lin[5],
float deq_factor[5],
int layer_count,
ap_uint<8> model[5],
STYPE scale_fea[5],
ITYPE* max_fea,
int quantized_multiplier,
ap_int<32> *shift,
ap_int<32> *bias,
ap_int<32> bias_count,
ap_int<64> *profiling,
ap_int<8> zero_point_lhs,
ap_int<8> zero_point_rhs,
ap_int<8> zero_point_dst,
ap_int<8> clamp_max,
ap_int<8> clamp_min,
INTYPES *array_b,
INTYPES *array_b2,
OUTTYPE *array_d1,
OUTTYPE *array_d2,
OUTTYPE *array_d3,
OUTTYPE *array_d4,
hls::stream<ASTYPE>& stream_d1,
hls::stream<ASTYPE>& stream_d1r,
hls::stream<ASTYPE>& stream_d1c,
hls::stream<ASTYPE>& stream_d2,
hls::stream<ASTYPE>& stream_d3,
hls::stream<ASTYPE>& stream_d4,
OUTTYPE *array_e1,
OUTTYPE *array_s1,
INTYPE *ate_m,
INTYPE *values_fea1,
INTYPE *values_fea2,
INTYPE *values_fea3,
INTYPE *values_fea4,
hls::stream<ASTYPE>& values_feas1,
hls::stream<ASTYPE>& values_feas2,
hls::stream<ASTYPE>& values_feas3,
hls::stream<ASTYPE>& values_feas4,
int *colIndices_fea1,
int *colIndices_fea2,
int *colIndices_fea3,
int *colIndices_fea4,
hls::stream<ASTYPE>& columnIndex_feas1,
hls::stream<ASTYPE>& columnIndex_feas2,
hls::stream<ASTYPE>& columnIndex_feas3,
hls::stream<ASTYPE>& columnIndex_feas4,
int nnz_fea1,
int nnz_fea2,
int nnz_fea3,
int nnz_fea4,
int *rowPtr_fea1,
int *rowPtr_fea2,
int *rowPtr_fea3,
int *rowPtr_fea4,
hls::stream<ASTYPE>& rowPtr_feas1,
hls::stream<ASTYPE>& rowPtr_feas2,
hls::stream<ASTYPE>& rowPtr_feas3,
hls::stream<ASTYPE>& rowPtr_feas4,
INTYPE *values_adj1,
INTYPE *values_adj2,
INTYPE *values_adj3,
INTYPE *values_adj4,
int *colIndices_adj1,
int *colIndices_adj2,
int *colIndices_adj3,
int *colIndices_adj4,
int nnz_adj1,
int nnz_adj2,
int nnz_adj3,
int nnz_adj4,
int *rowPtr_adj1,
int *rowPtr_adj2,
int *rowPtr_adj3,
int *rowPtr_adj4,
int N_adj,
int M_adj,
int M_fea,
int P_w
);


#endif 
