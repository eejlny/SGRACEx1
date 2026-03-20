/*===============================================================================
* This file is part of the SGRACE GNN accelerator
* has been written at UPM/Linkoping University
*  Copyright (c) [2026] [Jose Nunez-Yanez]
* All rights reserved.
*
* This software is licensed under the GPL-3.0 License.
* You may obtain a copy of the License at: https://www.gnu.org/licenses/gpl-3.0.en.html#license-text
*
===============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <hls_math.h>

#include <string>
#include <fstream>
#include <sstream> // //std::stringstream

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"

#include "matrix_mult.h"

#include "hls_streamofblocks.h"

typedef QTYPE buf[F_HEIGHT/FEA_THREADS][B_WIDTH_BLOCK];

const int BLOCK=B_WIDTH_BLOCK;   //BLOCK SHOULD be less than B_WIDTH_BLOCK
const int SBLOCK=SPMM_BLOCK;   //BLOCK SHOULD be less than B_WIDTH_BLOCK

const int PARALLEL_ROW = B_BLOCK_PARALLEL;
const int FIFO_DEPTH = MAX_FIFO;

#if (LINEAR_ONLY==1)
 const int LINEAR_DEPTH=16;
#else
 const int LINEAR_DEPTH=B_WIDTH_BLOCK*B_HEIGHT;
#endif

const int FIFO_DEPTH_ATTN = A_HEIGHT/OPT_ATTN;
const int FIFO_DEPTH_ATTN2 = A_HEIGHT*ATEN_BLOCK/OPT_ATTN;

const int FADD_LATENCY_ADJ = FTYPE_LATENCY_ADJ;
const int FADD_LATENCY_FEA = FTYPE_LATENCY_FEA;

static ap_int<64> fifo_full_0;
static ap_int<64> fifo_full_1;
static ap_int<64> fifo_full_2;
static ap_int<64> fifo_empty_0;
static ap_int<64> fifo_empty_1;
static ap_int<64> fifo_empty_2;
static ap_int<64> fifo_read_0;
static ap_int<64> fifo_read_1;
static ap_int<64> fifo_read_2;
static ap_int<64> fifo_write_0;
static ap_int<64> fifo_write_1;
static ap_int<64> fifo_write_2;
static ap_int<64> fifo_cycle_0;
static ap_int<64> fifo_cycle_1;
static ap_int<64> fifo_cycle_2;

#ifdef simulation
extern float max_adj;
extern float min_adj;
extern float max_fea;
extern float min_fea;
extern float acc2_fea_min;
extern float acc2_fea_max;
extern float acc2_adj_min;
extern float acc2_adj_max;
#endif

void quanta(ATYPE &BW,float B,float quantization_scale,int f_align, int beta_qu)
{

    float vfloat = quantization_scale*B+zero_point;
    float vround = hls::round(vfloat);

    ITYPE vquant = ITYPE(vround);

    #if (SIGNED_MODE==0)
    ITYPE ibeta_q = (ITYPE)beta_qu;
    ITYPE ialpha_q = (ITYPE)(0.0);
    #else
    ITYPE beta_q = ITYPE(beta_qu>>1);
    ITYPE ibeta_q = (ITYPE)beta_q;
    ITYPE ialpha_q = -(ITYPE)beta_q;
    #endif

    if (vquant>ibeta_q)
        vquant = ibeta_q;
    else if (vquant<ialpha_q)
        vquant = ialpha_q;

    if(f_align==7) //BINARY MODE
        f_align = 6;
    ITYPE vnorm = vquant >> (qbits-f_align-1);
    ATYPE fval = ATYPE(vnorm);

    BW = fval;

}

void quantf(FTYPE &BW,float B,float quantization_scale[5],int f_align, int beta_qu, int B_index)
{

    float vfloat = quantization_scale[B_index]*B+zero_point;
    float vround = hls::round(vfloat);

    ITYPE vquant = ITYPE(vround);

    #if (SIGNED_MODE==0)
    ITYPE ibeta_q = (ITYPE)beta_qu;
    ITYPE ialpha_q = (ITYPE)(0.0);
    #else
    ITYPE beta_q = ITYPE(beta_qu>>1);
    ITYPE ibeta_q = (ITYPE)beta_q;
    ITYPE ialpha_q = -(ITYPE)beta_q;
    #endif

    if (vquant>ibeta_q)
        vquant = ibeta_q;
    else if (vquant<ialpha_q)
        vquant = ialpha_q;

    if(f_align==7) //BINARY MODE
        f_align = 6;
    ITYPE vnorm = vquant >> (qbits-f_align-1);
    FTYPE fval = FTYPE(vnorm);

    BW = fval;

}

void quantl(LTYPE &BW,float B,float quantization_scale[5],int f_align, int beta_qu, int B_index)
{

    float vfloat = quantization_scale[B_index]*B+zero_point;

    float vround;

    ITYPE ibeta_q,ialpha_q,beta_q;

    if(f_align==7)
    {
        ibeta_q = 1;
        ialpha_q = -1;
        if(vfloat < 0.0) //BINARY MODE
         vround = -1.0;
        else
         vround = 1.0;
    }
    else
    {
        beta_q = ITYPE(beta_qu>>1);
        ibeta_q = (ITYPE)beta_q;
        ialpha_q = -(ITYPE)beta_q;
        vround = hls::round(vfloat);
    }

    ITYPE vquant = ITYPE(vround);

    if (vquant>ibeta_q)
        vquant = ibeta_q;
    else if (vquant<ialpha_q)
        vquant = ialpha_q;

    if(f_align==7) //BINARY MODE
        f_align = 6;
    ITYPE vnorm = vquant >> (qbitsl-f_align-1);
    LTYPE lval = LTYPE(vnorm);

    BW = lval;

}

void quantw(BTYPE &BW,float B,float quantization_scale[5],int f_align, int beta_qu, int B_index)
{

    float vfloat = quantization_scale[B_index]*B+zero_point;

    float vround;

    ITYPE ibeta_q,ialpha_q,beta_q;

    if(f_align==7)
    {
        ibeta_q = 1;
        ialpha_q = -1;
        if(vfloat < 0.0) //BINARY MODE
         vround = -1.0;
        else
         vround = 1.0;
    }
    else
    {
        beta_q = ITYPE(beta_qu>>1);
        ibeta_q = (ITYPE)beta_q;
        ialpha_q = -(ITYPE)beta_q;
        vround = hls::round(vfloat);
    }

    ITYPE vquant = ITYPE(vround);

    if (vquant>ibeta_q)
        vquant = ibeta_q;
    else if (vquant<ialpha_q)
        vquant = ialpha_q;

    if(f_align==7) //BINARY MODE
        f_align = 6;
    ITYPE vnorm = vquant >> (qbits-f_align-1);
    BTYPE fval = BTYPE(vnorm);

    BW = fval;

}

void dsp_kernel_float_adj_1(ATYPE a_value,BTYPE b_block[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
    #pragma HLS INLINE

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            BTYPE b_val = b_block[b_row][j];
            ATYPE a_val = a_value;
            acc[j] = (ITYPE)a_val*(ITYPE)b_val;

    } // j loop

}
void dsp_kernel_float_adj_2(int block_size,ATYPE a_value,BTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block2[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
    #pragma HLS INLINE

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {

            ATYPE a_val = a_value;
            BTYPE b_val;

            int sel_block; // = (b_row>>(log2N-2))&0x3;
            int b_row_block;

            if (b_row < block_size)
            {
                b_row_block = b_row;
                sel_block = 0;
            }
            if (b_row > (block_size-1))
            {
                b_row_block = b_row-block_size;
                sel_block = 1;
            }

            BTYPE b_val1 = b_block1[b_row_block][j];
            BTYPE b_val2 = b_block2[b_row_block][j];

            switch(sel_block)
            {
                case 0:
                    b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
                    break;
                case 1:
                    b_val = b_val2;
                break;
            }
            acc[j] = (ITYPE)a_val*(ITYPE)b_val;

    } // j loop

}

void dsp_kernel_float_adj_4(int block_size,ATYPE a_value,BTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block2[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block3[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block4[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
    #pragma HLS INLINE

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {

            ATYPE a_val = a_value;
            BTYPE b_val;

            int sel_block; // = (b_row>>(log2N-2))&0x3;
            int b_row_block;

            if (b_row < block_size)
            {
                b_row_block = b_row;
                sel_block = 0;
            }
            if (b_row > (block_size-1))
            {
                b_row_block = b_row-block_size;
                sel_block = 1;
            }

            if (b_row > (2*block_size-1) && b_row < 3*block_size)
            {
                b_row_block = b_row-2*block_size;
                sel_block = 2;
            }
            if (b_row > 3*block_size-1)
            {
                b_row_block = b_row-3*block_size;
                sel_block = 3;
            }

            BTYPE b_val1 = b_block1[b_row_block][j];
            BTYPE b_val2 = b_block2[b_row_block][j];
            BTYPE b_val3 = b_block3[b_row_block][j];
            BTYPE b_val4 = b_block4[b_row_block][j];

            switch(sel_block)
            {
                case 0:
                    b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
                    break;
                case 1:
                    b_val = b_val2;
                break;
                case 2:
                    b_val = b_val3;
                break;
                case 3:
                    b_val = b_val4;
                break;
            }
            acc[j] = (ITYPE)a_val*(ITYPE)b_val;

    } // j loop

}

void dsp_kernel_float_fea(ATYPE a_value,BTYPE b_block[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
    #pragma HLS INLINE

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            BTYPE b_val = b_block[b_row][j];
            ATYPE a_val = a_value;
            acc[j] = (ITYPE)a_val*(ITYPE)b_val;

    } // j loop

}

void dsp_kernel_int_adj_1(int block_size,TTYPE a_value,QTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],
        ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            TTYPE a_val = a_value;
            QTYPE b_val;

            int sel_block; // = (b_row>>(log2N-2))&0x3;
            int b_row_block;

            if (b_row < block_size)
            {
                b_row_block = b_row;
                sel_block = 0;
            }

            QTYPE b_val1 = b_block1[b_row_block][j];

            switch(sel_block)
            {
                case 0:
                    b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
                    break;
            }

            ITYPE a_val_i = (ITYPE)a_val;
            ITYPE b_val_i = (ITYPE)b_val;

            ITYPE acc_i = a_val_i*b_val_i;
            acc[j] = acc_i;
    } // j loop

}

void dsp_kernel_int_adj_2(int block_size,ITYPE a_value,QTYPE b_block1[B_HEIGHT/2][B_WIDTH_BLOCK],
        QTYPE b_block2[B_HEIGHT/2][B_WIDTH_BLOCK],
        ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {

            #pragma HLS UNROLL

                acc[j] = 0;
        }

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            ATYPE a_val = a_value;
            BTYPE b_val;

            int sel_block; // = (b_row>>(log2N-2))&0x3;
            int b_row_block;

            if (b_row < block_size)
            {
                b_row_block = b_row;
                sel_block = 0;
            }
            if (b_row > (block_size-1))
            {
                b_row_block = b_row-block_size;
                sel_block = 1;
            }

            BTYPE b_val1 = b_block1[b_row_block][j];
            BTYPE b_val2 = b_block2[b_row_block][j];

            switch(sel_block)
            {
                case 0:
                    b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
                    break;
                case 1:
                    b_val = b_val2;
                    break;
            }

            ITYPE a_val_i = (ITYPE)a_val;
            ITYPE b_val_i = (ITYPE)b_val;

            ITYPE acc_i = a_val_i*b_val_i;
            acc[j] += acc_i;
    } // j loop

}

void dsp_kernel_int_adj_4(int block_size,TTYPE a_value,QTYPE b_block1[B_HEIGHT/4][B_WIDTH_BLOCK],
        QTYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],QTYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],
        QTYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {

            #pragma HLS UNROLL

                acc[j] = 0;
        }

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            TTYPE a_val = a_value;
            QTYPE b_val;

            int sel_block; // = (b_row>>(log2N-2))&0x3;
            int b_row_block;

            if (b_row < block_size)
            {
                b_row_block = b_row;
                sel_block = 0;
            }
            if (b_row > (block_size-1) && b_row < 2*block_size)
            {
                b_row_block = b_row-block_size;
                sel_block = 1;
            }
            if (b_row > (2*block_size-1) && b_row < 3*block_size)
            {
                b_row_block = b_row-2*block_size;
                sel_block = 2;
            }
            if (b_row > 3*block_size-1)
            {
                b_row_block = b_row-3*block_size;
                sel_block = 3;
            }

            QTYPE b_val1 = b_block1[b_row_block][j];
            QTYPE b_val2 = b_block2[b_row_block][j];
            QTYPE b_val3 = b_block3[b_row_block][j];
            QTYPE b_val4 = b_block4[b_row_block][j];

            switch(sel_block)
            {
                case 0:
                    b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
                    break;
                case 1:
                    b_val = b_val2;
                    break;
                case 2:
                    b_val = b_val3;
                    break;
                case 3:
                    b_val = b_val4;
                    break;
            }

            ITYPE a_val_i = (ITYPE)a_val;
            ITYPE b_val_i = (ITYPE)b_val;

            ITYPE acc_i = a_val_i*b_val_i;
            acc[j] += acc_i;
    } // j loop

}

void dsp_kernel_int_fea(FTYPE a_value,BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            FTYPE a_val = a_value;
            BTYPE b_val = b_block[b_row][j]; //only one value of B in each row. This is the result of the first matrix mult.

            ITYPE b_val_i;
            ITYPE a_val_i;
               b_val_i = (ITYPE)b_val;
               a_val_i = (ITYPE)a_val;

            ITYPE acc_i = a_val_i*b_val_i;
            acc[j] = acc_i;

    } // j loop

}

void dsp_kernel_int_lin(LTYPE a_value,BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            LTYPE a_val = a_value;
            BTYPE b_val = b_block[b_row][j]; //only one value of B in each row. This is the result of the first matrix mult.

            ITYPE b_val_i;
            ITYPE a_val_i;
               b_val_i = (ITYPE)b_val;
               a_val_i = (ITYPE)a_val;

            ITYPE acc_i = a_val_i*b_val_i;
            acc[j] = acc_i;

    } // j loop

}

void writec(float deq_factor[5],ap_uint<1> model[5][8],int first_row, int row_count,int N_adj,ap_uint<8> P[5], hls::stream<ITYPE> write_fifo[B_WIDTH_BLOCK],QTYPE linear_pipo[B_HEIGHT][B_WIDTH_BLOCK],hls::stream<OUTTYPE>& CS, int B_index, int layer_loop)
{
        int B_WIDTH_INT;

        bool linear_mode;
        bool sage_mode;

        int WL;

        #if defined FLOAT
            WL = row_count;
        #endif

        #if defined HALF
            WL = row_count;
        #endif

        #ifdef EIGHTBIT
            WL = row_count;

        #endif

        linear_mode = model[B_index][6];
        sage_mode = model[B_index][7];

        DTYPE C_out = DTYPE(0.0);
        DTYPE residual;

        LOOP_WRITE42:    for (int i = 0; i < WL; i++) {
            LOOP_WRITE52: for (int j = 0; j <  B_WIDTH_BLOCK; j++) {
                     #pragma HLS PIPELINE II=1
                        #if LINEAR_ONLY == 0
                        if (linear_mode==0)
                         C_out =  DTYPE(write_fifo[j].read());
                        else
                         C_out = 0;
                        #endif
                        #if GNN_ONLY == 0
                        residual = DTYPE(linear_pipo[i][j]);
                        #else
                        residual = DTYPE(0.0);
                        #endif

                        #if (INT_DEQUANT == 1)
                             OUTTYPE C_float = (OUTTYPE)C_out*deq_factor[B_index]+(OUTTYPE)residual*deq_factor[B_index]*(sage_mode || linear_mode);
                        #else
                             OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif

                        if(j<P[B_index]) //when relu only write non-zeros for sparse computation
                         CS.write(C_float);

                    }
         }

}

void writeout(ap_uint<1> model[5][8],int first_row, int row_count,int N_adj,ap_uint<8> P[5], hls::stream<OUTTYPE>& write_fifo,OUTTYPE* C,hls::stream<ASTYPE>& CS, hls::stream<ASTYPE>& CSR, hls::stream<ASTYPE>& CSC, int B_index, int layer_loop)
{
        int B_WIDTH_INT;

        int WL;

        WL = row_count;

        B_WIDTH_INT = P[B_index];
        ap_uint<1> stream_mode = model[B_index][2];
        ap_uint<1> gemm_mode = model[B_index+1][1]; //check if next layer wants sparse features

        std::cout << "stream mode in adj is " << stream_mode << std::endl;

        if (stream_mode==1) //write to stream
        {

            bool last=0;
            std::cout << "write to stream count " << WL*B_WIDTH_BLOCK << std::endl;

                 LOOP_WRITE42:    for (int i = 0; i < WL; i++) {
                   LOOP_WRITE52: for (int j = 0; j <  B_WIDTH_INT; j++) {
                     #pragma HLS PIPELINE II=1

                     if(i*j==(WL-1)*(B_WIDTH_INT-1))
                       last = 1;

                     OUTTYPE C_float =  OUTTYPE(write_fifo.read());
                     ASTYPE temp;
                     fp_int C_float_int;
                     C_float_int.f = C_float;
                     temp.data = C_float_int.i;

                     if(gemm_mode==1)
                     {
                      temp.last = last;
                      CS.write(temp);
                     }
                     else
                     {
                      if(j==0 or C_float!=0 or last==1) //do not write zero but always write if last=1 or if first element of row
                      {
                       temp.last = last;
                       CS.write(temp);
                       temp.data = i;
                       temp.last = last;
                       CSR.write(temp);
                       temp.data = j;
                       temp.last = last;
                       CSC.write(temp);
                      }
                     }

                   }
                  }

        }
        else  // write to memory
        {
                LOOP_WRITE45:    for (int i = 0; i < WL; i++) {
                 LOOP_WRITE55: for (int j = 0; j <  B_WIDTH_INT; j++) {
                    #pragma HLS PIPELINE II=1
                    OUTTYPE C_float =  OUTTYPE(write_fifo.read());
                    C[i*B_WIDTH_INT+j+first_row*B_WIDTH_BLOCK] = C_float;
                    }
                   }

        }

}

void writec_transpose(float deq_factor,bool stream_mode,int first_row, int row_count,int N_adj,int P, hls::stream<ITYPE> write_fifo[B_WIDTH_BLOCK], OUTTYPE* C,hls::stream<ASTYPE>& CS, int B_index)
{
        int B_WIDTH_INT;

        int WL;

        #if defined FLOAT
            WL = row_count;
        #endif

        #if defined HALF
            WL = row_count;
        #endif

        #ifdef EIGHTBIT
            WL = row_count;
        #endif

            B_WIDTH_INT = B_WIDTH_BLOCK;

        LOOP_WRITE4:    for (int i = 0; i < WL; i+=FIFO_DEPTH) {

                 LOOP_WRITE5: for (int j = 0; j <  B_WIDTH_BLOCK; j++) {
                        DTYPE C_out;

                     LOOP_WRITE6: for (int z = 0; z <  FIFO_DEPTH; z++) {
                        #pragma HLS PIPELINE II=1
                        if (i+z < WL)
                            C_out =  DTYPE(write_fifo[j].read());
                        else
                            C_out = 0.0;
                        #if (INT_DEQUANT == 1)
                         OUTTYPE C_float = (OUTTYPE)C_out*deq_factor;
                        #else
                         OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif

                        #if (STREAM_MODE_OUT == 1)  //if (stream_mode == 1)
                        {
                           ASTYPE temp;
                           temp.data = C_float;
                           CS.write(temp);
                        }
                        #else
                           C[i*FIFO_DEPTH+j*WL+z+first_row*B_WIDTH_BLOCK+B_index*N_adj*B_WIDTH_BLOCK] = C_float;
                        #endif
                    }

                 }

        }

}

void writes(float deq_factor[5],ap_uint<1> model[5][8], int first_row, int row_count,int N_adj,ap_uint<8> P[5], hls::stream<TTYPE> &write_fifo, hls::stream<int> &rnnz_fifo,  OUTTYPE* C,int B_index)
{
        int B_WIDTH_INT;

        int WL;

        WL = row_count;

        B_WIDTH_INT = B_WIDTH_BLOCK;

        int rnnz = rnnz_fifo.read();

        bool gat_mode = model[B_index][5];

        if (gat_mode == 1)
        {

                 DTYPE C_out;

                 LOOP_WRITE5: for (int i = 0; i <  rnnz; i++) {
                    #pragma HLS PIPELINE
                        C_out =  write_fifo.read();
                        #if (INT_DEQUANT == 1)
                           OUTTYPE C_float = (OUTTYPE)C_out*deq_factor[B_index];
                        #else
                           OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif
                        C[i] = C_float;
                    }

        }

}



void readptr_csr_fea(bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

    #pragma HLS inline off
    int rnnz,current_index,next_index;

    current_index= rowPtr[0];

    if (gemm_mode==0)
    {
        LOOP_A_INDEX_SPMM1 : for (int A_index = 0; A_index < N; A_index++) {
            #pragma HLS PIPELINE
                    next_index=rowPtr[A_index+1];
                    rnnz = next_index-current_index;
                    current_index = next_index;
                    rnnz_fifo << rnnz;
            }
    }
    else
    {
        LOOP_A_INDEX_SPMM2 : for (int A_index = 0; A_index < N; A_index++) {
            #pragma HLS PIPELINE
                rnnz = M;
                rnnz_fifo << rnnz;
         }
    } //end else

}

void read_ptr2(int nnz_fea,int *rowPtr, hls::stream<int> &index_fifo)
{
    int next_index1;
    LOOP_A_INDEX0 : for (int A_index = 0; A_index <nnz_fea+1 ; A_index++)
    {
        #pragma HLS PIPELINE
        next_index1=rowPtr[A_index];
        index_fifo << next_index1;
    }

}

void read_ptr(bool stream_mode,int nnz_fea,int *rowPtr,  hls::stream<int> &index_fifo)
{
    int next_index1;

    if (stream_mode == 0)
    {
      LOOP_A_INDEX0 : for (int A_index = 0; A_index <nnz_fea+1 ; A_index++)
      {
        #pragma HLS PIPELINE
        next_index1=rowPtr[A_index];
        index_fifo << next_index1;
      }
    }

}

void proc_ptr(int nnz_fea,hls::stream<int> &index_fifo,hls::stream<int> &rnnz_fifo)
{
    int next_index2;
    int rnnz = 0;
    int current_index;
    int B_index = 0;
    int first_read = 1;

    current_index =index_fifo.read();
    rnnz++;

    LOOP_A_INDEX1 : while(B_index < nnz_fea-1) {
    #pragma HLS PIPELINE
    next_index2=index_fifo.read();
    B_index++;
    if(next_index2 == current_index)
    {
       rnnz++;

    }
    else
    {
#if (LINEAR_ONLY == 0)
      rnnz_fifo << rnnz;
#endif
      current_index=next_index2;
      rnnz = 1;
    }

   }

#if (LINEAR_ONLY == 0)
    rnnz_fifo << rnnz;
#endif

    next_index2=index_fifo.read();
}

void proc_ptr2(bool linear_mode,bool stream_mode,int nnz_fea,hls::stream<int> &index_fifo,hls::stream<ASTYPE>&  rowPtrs,hls::stream<int> &rnnz_fifo,hls::stream<int> &rnnz_fifo_sage)
{
    int next_index2;
    int rnnz = 0;
    int current_index;
    ASTYPE  temp;
    int B_index = 0;
    int first_read = 1;

    if(stream_mode==0)
    {

     current_index = index_fifo.read();
     rnnz++;
     LOOP_A_INDEX1 : while(B_index < nnz_fea-1) {

     #pragma HLS PIPELINE
     next_index2 =index_fifo.read();
     B_index++;
     if(next_index2 == current_index)
     {
       rnnz++;

     }
     else
     {
       #if (LINEAR_ONLY == 0)
       if(linear_mode==0)
        rnnz_fifo << rnnz;
       #endif
       #if (GNN_ONLY == 0)
       rnnz_fifo_sage << rnnz;
       #endif

       current_index=next_index2;
       rnnz = 1;
     }

   }

   #if (LINEAR_ONLY == 0)
   if(linear_mode==0)
    rnnz_fifo << rnnz;
   #endif
   #if (GNN_ONLY == 0)
   rnnz_fifo_sage << rnnz;
   #endif
   next_index2=index_fifo.read();
   }
   else //stream mode on
   {

     temp=rowPtrs.read();
     rnnz=1;
     current_index= temp.data;

     if(temp.last!=1)
     {
      LOOP_A_INDEX2 : do {
         #pragma HLS PIPELINE
         temp=rowPtrs.read();
         next_index2= temp.data;
         if(next_index2 == current_index)
         {
           rnnz++;

         }
         else
         {
           #if (LINEAR_ONLY == 0)
           if(linear_mode==0)
            rnnz_fifo << rnnz;
           #endif
           #if (GNN_ONLY == 0)
           rnnz_fifo_sage << rnnz;
           #endif

           current_index=next_index2;
           rnnz = 1;
         }
       }while(temp.last!=1);
     }

   #if (LINEAR_ONLY == 0)
   if(linear_mode==0)
    rnnz_fifo << rnnz;
   #endif
   #if (GNN_ONLY == 0)
   rnnz_fifo_sage << rnnz;
   #endif

   }

}

void read_dataflow2(bool linear_mode,bool stream_mode,int nnz_fea,int *rowPtr,hls::stream<ASTYPE>&  rowPtrs,hls::stream<int> &rnnz_fifo,hls::stream<int> &rnnz_fifo_sage)
{

    hls::stream<int>  index_fifo("index fifo");
    #pragma HLS STREAM variable= index_fifo depth=FIFO_DEPTH

    #pragma HLS DATAFLOW
    read_ptr(stream_mode,nnz_fea,rowPtr,index_fifo);
    proc_ptr2(linear_mode,stream_mode,nnz_fea,index_fifo,rowPtrs,rnnz_fifo,rnnz_fifo_sage);

}

void read_dataflow(int nnz_fea,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

    hls::stream<int>  index_fifo("index fifo");
    #pragma HLS STREAM variable= index_fifo depth=FIFO_DEPTH

    #pragma HLS DATAFLOW
    read_ptr2(nnz_fea,rowPtr,index_fifo);
    proc_ptr(nnz_fea,index_fifo,rnnz_fifo);

}

void readptr_coo_fea(int nnz_fea,bool linear_mode,bool stream_mode,bool gemm_mode,int N,int M,int *rowPtr,hls::stream<ASTYPE>&  rowPtrs,hls::stream<int> &rnnz_fifo,hls::stream<int> &rnnz_fifo_sage)
{

    #pragma HLS inline off

    if (gemm_mode==0)
    {

        read_dataflow2(linear_mode,stream_mode,nnz_fea,rowPtr,rowPtrs,rnnz_fifo,rnnz_fifo_sage);

    }
    else
    {
        int rnnz;
        LOOP_A_INDEX2 : for (int A_index = 0; A_index <N ; A_index++) {
            #pragma HLS PIPELINE
                rnnz = M;
                #if (LINEAR_ONLY == 0)
                if(linear_mode==0)
                 rnnz_fifo << rnnz;
                #endif
                #if (GNN_ONLY == 0)
                rnnz_fifo_sage << rnnz;
                #endif
         }
    } //end else

}

void readptr_csr_adj(bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{
    #pragma HLS inline off
    int rnnz,current_index,next_index;

    current_index= rowPtr[0];

    if (gemm_mode==0)
    {
        LOOP_A_INDEX_SPMM1 : for (int A_index = 0; A_index < N; A_index++) {
                next_index=rowPtr[A_index+1];
                rnnz = next_index-current_index;
                current_index = next_index;
                rnnz_fifo << rnnz;

        }
    }
    else
    {
        LOOP_A_INDEX_SPMM2 : for (int A_index = 0; A_index < N; A_index++) {
            #pragma HLS PIPELINE
            rnnz = M;
            rnnz_fifo << rnnz;

       }
    } //end else

}

void readptr_coo_adj(int nnz_adj,bool linear_mode,bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

    #pragma HLS inline off

    if(linear_mode==0)
    {
     if (gemm_mode==0)
     {

        read_dataflow(nnz_adj,rowPtr,rnnz_fifo);

     }
     else
     {
        int rnnz;
        LOOP_A_INDEX2 : for (int A_index = 0; A_index <N ; A_index++) {
            #pragma HLS PIPELINE
                rnnz = M;
      #if (LINEAR_ONLY == 0)
                rnnz_fifo << rnnz;
      #endif
         }
     } //end else
    }//end linear
}

void readval_csr_adj(int beta_qu,int f_align,float quantization_scale_fea,bool gemm_mode,int ccount,int last_index,hls::stream<ATYPE> &A_fifo,hls::stream<int> &col_indices_fifo,INTYPE *values,int *columnIndex)
{

        #pragma HLS inline off
        if (gemm_mode==0)
        {
          LOOP_J_SPMM : for (int j = 0; j < last_index; j++) {
            #pragma HLS PIPELINE

            INTYPE value_temp;
            ATYPE value_temp2;
            value_temp = (INTYPE)values[j];

            #if (INT_QUANT == 1)
                quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
            #else
                value_temp2 = value_temp;
            #endif

            A_fifo <<  value_temp2;
            col_indices_fifo << columnIndex[j];
          }
        }
        else
        {
                int c=0;
                LOOP_J_SPMM2 : for (int j = 0; j < last_index; j++) {
                #pragma HLS PIPELINE

                   INTYPE value_temp;
                   ATYPE value_temp2;

                   value_temp = (INTYPE)values[j];

                   #if (INT_QUANT == 1)
                     quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
                   #else
                     value_temp2 = value_temp;
                   #endif

                    A_fifo <<  value_temp2;
                    col_indices_fifo << c;
                    if (c == (ccount-1)) //column count
                        c=0;
                    else
                        c++;
                }
        }

}

void readval_coo_adj(int beta_qu,int f_align,float quantization_scale_fea,bool gemm_mode,int ccount,int last_index,hls::stream<ATYPE> &A_fifo,hls::stream<int> &col_indices_fifo,INTYPE *values,int *columnIndex)
{

        #pragma HLS inline off
        if (gemm_mode==0)
        {
          LOOP_J_SPMM : for (int j = 0; j < last_index; j++) {
            #pragma HLS PIPELINE

            INTYPE value_temp;
            ATYPE value_temp2;
            value_temp = (INTYPE)values[j];

            #if (INT_QUANT == 1)
                quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
            #else
                value_temp2 = value_temp;
            #endif

            A_fifo <<  value_temp2;
            col_indices_fifo << columnIndex[j];
          }
        }
        else
        {
                int c=0;
                LOOP_J_SPMM2 : for (int j = 0; j < last_index; j++) {
                #pragma HLS PIPELINE

                   INTYPE value_temp;
                   ATYPE value_temp2;

                   value_temp = (INTYPE)values[j];

                   #if (INT_QUANT == 1)
                     quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
                   #else
                     value_temp2 = value_temp;
                   #endif

                    A_fifo <<  value_temp2;
                    col_indices_fifo << c;
                    if (c == (ccount-1)) //column count
                        c=0;
                    else
                        c++;
                }
        }

}

void readval_csr_adj2(int beta_qu,int f_align,float quantization_scale_fea,bool gemm_mode,int ccount,int last_index,hls::stream<ITYPE> &A_fifo,hls::stream<int> &col_indices_fifo,INTYPE *values,int *columnIndex)
{

        #pragma HLS inline off
        if (gemm_mode==0)
        {
          LOOP_J_SPMM : for (int j = 0; j < last_index; j++) {
            #pragma HLS PIPELINE

            INTYPE value_temp;
            ATYPE value_temp2;
            value_temp = (INTYPE)values[j];

            #if (INT_QUANT == 1)
                quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
            #else
                value_temp2 = value_temp;
            #endif

            A_fifo <<   (ITYPE)value_temp2;
            col_indices_fifo << columnIndex[j];
         }
       }
       else
       {
                int c=0;
                LOOP_J_SPMM2 : for (int j = 0; j < last_index; j++) {
                #pragma HLS PIPELINE

                    INTYPE value_temp;
                    ATYPE value_temp2;
                    value_temp = (INTYPE)values[j];

                    #if (INT_QUANT == 1)
                        quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
                    #else
                        value_temp2 = value_temp;
                    #endif

                    A_fifo <<   (ITYPE)value_temp2;
                    col_indices_fifo << c;
                    if (c == (ccount-1)) //column count
                        c=0;
                    else
                        c++;
                }
       }

}

void readval_coo_adj2(int beta_qu,int f_align,float quantization_scale_fea,bool linear_mode,bool gemm_mode,int ccount,int last_index,hls::stream<ITYPE> &A_fifo,hls::stream<int> &col_indices_fifo,INTYPE *values,int *columnIndex)
{

        #pragma HLS inline off
       if(linear_mode==0)
       {
        if (gemm_mode==0)
        {
          LOOP_J_SPMM : for (int j = 0; j < last_index; j++) {
            #pragma HLS PIPELINE

            INTYPE value_temp;
            ATYPE value_temp2;
            value_temp = (INTYPE)values[j];

            #if (INT_QUANT == 1)
                quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
            #else
                value_temp2 = value_temp;
            #endif

            #if(LINEAR_ONLY==0)

            A_fifo <<   (ITYPE)value_temp2;
            col_indices_fifo << columnIndex[j];
            #endif
         }
       }
       else
       {
                int c=0;
                LOOP_J_SPMM2 : for (int j = 0; j < last_index; j++) {
                #pragma HLS PIPELINE

                    INTYPE value_temp;
                    ATYPE value_temp2;
                    value_temp = (INTYPE)values[j];

                    #if (INT_QUANT == 1)
                        quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
                    #else
                        value_temp2 = value_temp;
                    #endif

                    #if (LINEAR_ONLY == 0)
                    A_fifo <<   (ITYPE)value_temp2;

                    col_indices_fifo << c;
                    #endif
                    if (c == (ccount-1)) //column count
                        c=0;
                    else
                        c++;
                }
          }
       }

}

void readval_coo_fea(int beta_qu,int f_align,int beta_qul,int f_alignl,
        float quantization_scale_fea[5],float quantization_scale_lin[5],bool linear_mode,bool stream_mode,bool gemm_mode,int ccount,int last_index,
        hls::stream<FTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,
        hls::stream<LTYPE> &A_fifo_sage,hls::stream<int> &col_indices_fifo_sage,
        INTYPE *values,hls::stream<ASTYPE>&  valuess,int *columnIndex,hls::stream<ASTYPE>&  columnIndex_feas, int B_index)
{

    #pragma HLS inline off

    std::cout << "gemm mode " <<  gemm_mode << std::endl;
    std::cout << "stream mode " <<  stream_mode << std::endl;
    std::cout << "read count " <<  last_index << std::endl;

    if (gemm_mode==0)
    {

        fp_int C_float_int;

        if(stream_mode==1)
        {

            bool last_index1;
            LOOP_J_SPMM11 : do{
            #pragma HLS PIPELINE
            INTYPE value_temp;
            FTYPE value_temp2;
            LTYPE value_temp3;

             ASTYPE temp = valuess.read();

             C_float_int.i = temp.data;

             value_temp = (INTYPE)C_float_int.f;

             temp = columnIndex_feas.read();

             last_index1=temp.last;

             int c = temp.data;

            #if (INT_QUANT == 1)
               quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
               quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
            #else
               value_temp2 = value_temp;
               value_temp3 = value_temp;
            #endif

            #if (LINEAR_ONLY == 0)

              if(linear_mode==0)
              {

                A_fifo << value_temp2;

                col_indices_fifo << c;
              }
           #endif

              #if (GNN_ONLY == 0)

               col_indices_fifo_sage << c;
               A_fifo_sage << value_temp3;

              #endif

          }while(last_index1==0);

        }
        else
        {
          LOOP_J_SPMM12 : for (int j = 0; j < last_index; j++) {
            #pragma HLS PIPELINE
            INTYPE value_temp;
            FTYPE value_temp2;
            LTYPE value_temp3;
            int col_temp;

              value_temp = (INTYPE)values[j];
              col_temp = columnIndex[j];

            #if (INT_QUANT == 1)
              quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
              quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
            #else
              value_temp2 = value_temp;
              value_temp3 = value_temp;
            #endif

           #if (LINEAR_ONLY == 0)
            if(linear_mode==0)
            {
              A_fifo << value_temp2;
              col_indices_fifo << col_temp;
            }
            #endif

            #if (GNN_ONLY == 0)

            col_indices_fifo_sage << col_temp;
            A_fifo_sage << value_temp3;

            #endif
         }
        }
    }
    else//gemm mode
    {

            int c=0;
            fp_int C_float_int;

            bool last_index1=0;

            if(stream_mode==1) {

            LOOP_J_SPMM21 : for (int j = 0; j < last_index; j++) {

            #pragma HLS PIPELINE
                   INTYPE value_temp;
                   FTYPE value_temp2;
                   LTYPE value_temp3;

                    ASTYPE temp = valuess.read();

                    C_float_int.i = temp.data;

                    value_temp = (INTYPE)C_float_int.f;

                    last_index1=temp.last;

                   #if (INT_QUANT == 1)
                     quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
                     quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
                   #else
                     value_temp2 = value_temp;
                     value_temp3 = value_temp;
                    #endif

                #if (LINEAR_ONLY == 0)
                 if(linear_mode==0)
                 {
                   A_fifo <<  value_temp2;
                   col_indices_fifo << c;
                 }
                #endif
                #if (GNN_ONLY == 0)
                col_indices_fifo_sage << c;
                A_fifo_sage <<  value_temp3;
                #endif
                if (c == (ccount-1)) //column count
                    c=0;
                else
                    c++;
              }//while(last_index1==0);
            }
            else
            {

            LOOP_J_SPMM22 : for (int j = 0; j < last_index; j++) {

            #pragma HLS PIPELINE
                   INTYPE value_temp;
                   FTYPE value_temp2;
                   LTYPE value_temp3;

                     value_temp = (INTYPE)values[j];

                   #if (INT_QUANT == 1)
                     quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
                     quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
                   #else
                     value_temp2 = value_temp;
                    #endif

                #if (LINEAR_ONLY == 0)
                 if(linear_mode==0)
                 {
                  A_fifo <<  value_temp2;
                  col_indices_fifo << c;
                 }
                #endif
                #if (GNN_ONLY == 0)
                col_indices_fifo_sage << c;
                A_fifo_sage <<  value_temp3;
                #endif

                if (c == (ccount-1)) //column count
                    c=0;
                else
                    c++;
            }
        }
    }
}

void check_fifo_0(int a_values, hls::stream<ITYPE> &A_fifo, hls::stream<ITYPE> &A_fifo_out)
{
    ITYPE data_buffer;
    int data_count=0;
    bool loop_done = 0;
    bool data_in_buffer = 0; //data exits in data_buffer
    while((data_count < a_values) || data_in_buffer == 0)
    {
        #pragma HLS PIPELINE
            fifo_cycle_0++;
            if (data_in_buffer == 0) //data_buffer empty
            {
                if(A_fifo.read_nb(data_buffer) == 1)
                {
                    fifo_read_0++;
                    data_count++;
                    if(A_fifo_out.write_nb(data_buffer) == 0)
                    {
                        fifo_full_0++;
                        data_in_buffer = 1; //fifo full and data stored in data_in_buffer
                    }
                    else
                    {
                        fifo_write_0++;
                    }
                }
                else
                {
                }
            }
            else //data_buffer not empty
            {
                if (A_fifo_out.write_nb(data_buffer) == 1)
                {
                    fifo_write_0++;
                    if(A_fifo.read_nb(data_buffer) == 0)
                    {
                        data_in_buffer = 0; //data_buffer empty
                    }
                    else
                    {
                        fifo_read_0++;
                        data_count++;
                    }

                }
                else
                {
                    fifo_full_0++;
                }
            }
    }

}

void check_fifo_2(int N, hls::stream<ITYPE> &C_fifo, hls::stream<ITYPE> &C_fifo_out)
{
    ITYPE data_buffer;
    int data_count=0;
    bool data_in_buffer= 0; //data exits in data_buffer

    while(data_count < N)
    {
        #pragma HLS PIPELINE
            fifo_cycle_2++;
                if (data_in_buffer == 0) //data_buffer empty
            {
                if(C_fifo.read_nb(data_buffer) == 1)
                {

                    fifo_read_2++;
                    if(C_fifo_out.write_nb(data_buffer) == 0)
                    {
                        fifo_full_2++;
                        data_in_buffer = 1; //fifo full and data stored in data_in_buffer
                    }
                    else
                    {
                        data_count++;
                        fifo_write_2++;
                    }
                }
                else
                {
                    fifo_empty_2++;
                }

            }
            else //data_buffer not empty
            {
                if (C_fifo_out.write_nb(data_buffer) == 1)
                {
                    fifo_write_2++;
                    if(C_fifo.read_nb(data_buffer) == 0)
                    {
                        fifo_empty_2++;
                        data_in_buffer = 0; //data_buffer empty
                    }
                    else
                    {
                        fifo_read_2++;
                    }
                    data_count++;

                }
                else
                {
                    fifo_full_2++;
                }
            }
    } //while

}

void check_fifo_1(int N, int B_index, int B_index_loop, int tail, hls::stream<ITYPE> &C_fifo, hls::stream<ITYPE> &C_fifo_out)
{
    ITYPE data_buffer;
    int data_count=0;
    bool data_in_buffer= 0; //data exits in data_buffer

    while(data_count < N)
    {
        #pragma HLS PIPELINE
            fifo_cycle_1++;
                if (data_in_buffer == 0) //data_buffer empty
            {
                if(C_fifo.read_nb(data_buffer) == 1)
                {

                    fifo_read_1++;
                    if(C_fifo_out.write_nb(data_buffer) == 0)
                    {
                        fifo_full_1++;
                        data_in_buffer = 1; //fifo full and data stored in data_in_buffer
                    }
                    else
                    {
                        data_count++;
                        fifo_write_1++;
                    }
                }
                else
                {
                }

            }
            else //data_buffer not empty
            {
                if (C_fifo_out.write_nb(data_buffer) == 1)
                {
                    fifo_write_1++;
                    if(C_fifo.read_nb(data_buffer) == 0)
                    {
                        data_in_buffer = 0; //data_buffer empty
                    }
                    else
                    {
                        fifo_read_1++;
                    }
                    data_count++;

                }
                else
                {
                    fifo_full_1++;
                }
            }
    } //while

}

void reada1_coo(int nnz_fea,int beta_qu,int f_align,int beta_qul,int f_alignl,float quantization_scale_fea[5],float quantization_scale_lin[5],
        int &last_index,ap_uint<1> model[5][8],int M, int first_row, int row_count,
        hls::stream<FTYPE> &A_fifo_fea,hls::stream<int> &col_indices_fifo_fea, hls::stream<int> &rnnz_fifo_fea,
        hls::stream<LTYPE> &A_fifo_fea_sage,hls::stream<int> &col_indices_fifo_fea_sage, hls::stream<int> &rnnz_fifo_fea_sage,
int *rowPtr_fea,int *columnIndex_fea,INTYPE *values_fea,
hls::stream<ASTYPE>&  rowPtr_feas,hls::stream<ASTYPE>&  columnIndex_feas,hls::stream<ASTYPE>&  values_feas,
int B_index, int layer_loop)
{

    int last_index_fea;
    bool gemm_mode,stream_mode,linear_mode;
    int M_int;

    gemm_mode = model[B_index][1];
    stream_mode = model[B_index][3];
    linear_mode = model[B_index][6];

    if (B_index == 0) //first layer
          M_int = M;
    else
          M_int = B_WIDTH_BLOCK; //in hidden layers the input width (number of features) is the B_WDITH BLOCK

    if (gemm_mode==0)
    {
        columnIndex_fea += first_row;
        values_fea += first_row;
        rowPtr_fea += first_row;
        last_index_fea = nnz_fea;
    }
    else
    {
        values_fea+=first_row*M_int;
        last_index_fea=row_count*M_int;
    }

    std::cout << "Last_index_fea " << last_index_fea << std::endl;

    readptr_coo_fea(nnz_fea,linear_mode,stream_mode,gemm_mode,row_count,M_int,rowPtr_fea,rowPtr_feas,rnnz_fifo_fea,rnnz_fifo_fea_sage);
    readval_coo_fea(beta_qu,f_align,beta_qul,f_alignl,quantization_scale_fea,quantization_scale_lin,linear_mode,stream_mode,gemm_mode,M_int,last_index_fea,
            A_fifo_fea,col_indices_fifo_fea,
            A_fifo_fea_sage,col_indices_fifo_fea_sage,
            values_fea,values_feas,columnIndex_fea,columnIndex_feas,B_index);

}

void reada2_csr(int beta_qu,int f_align,float quantization_scale_adj,bool gemm_mode,int M,int first_row, int row_count,  hls::stream<ATYPE> &A_fifo_adj,hls::stream<int> &col_indices_fifo_adj, hls::stream<int> &rnnz_fifo_adj_total_e, hls::stream<int> &rnnz_fifo_adj_total_s,hls::stream<int> &rnnz_fifo_adj,
int *rowPtr_adj,int *columnIndex_adj,INTYPE *values_adj)
{

    int last_index_adj;

    if (gemm_mode==0)
    {

     last_index_adj=rowPtr_adj[first_row+row_count]-rowPtr_adj[first_row];

     columnIndex_adj += rowPtr_adj[first_row];
     values_adj += rowPtr_adj[first_row];
     rowPtr_adj += first_row;
    }
    else
    {

        last_index_adj=row_count*M;
        values_adj+=first_row*M;
    }

    rnnz_fifo_adj_total_e << last_index_adj;
    rnnz_fifo_adj_total_s << last_index_adj;

    readptr_csr_adj(gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);

    readval_csr_adj(beta_qu,f_align,quantization_scale_adj,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);

}

void reada2_coo(int nnz_adj,int beta_qu,int f_align,float quantization_scale_adj,ap_uint<1> model[5][8],int M,int first_row, int row_count,  hls::stream<ATYPE> &A_fifo_adj,hls::stream<int> &col_indices_fifo_adj, hls::stream<int> &rnnz_fifo_adj_total_e, hls::stream<int> &rnnz_fifo_adj_total_s,hls::stream<int> &rnnz_fifo_adj,
int *rowPtr_adj,int *columnIndex_adj,INTYPE *values_adj,int B_index)
{

    bool gemm_mode;

    gemm_mode = model[B_index][0];

    bool linear_mode;

    linear_mode =  model[B_index][6];

    int last_index_adj;

    if (gemm_mode==0)
    {

     columnIndex_adj += rowPtr_adj[first_row];
     values_adj += rowPtr_adj[first_row];
     rowPtr_adj += first_row;
     last_index_adj = nnz_adj;
    }
    else
    {

        values_adj+=first_row*M;
        last_index_adj = row_count*M;
    }

    rnnz_fifo_adj_total_e << nnz_adj;
    rnnz_fifo_adj_total_s << nnz_adj;

    readptr_coo_adj(nnz_adj,linear_mode,gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);

    readval_coo_adj(beta_qu,f_align,quantization_scale_adj,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);

}

void reada22_coo(int nnz_adj,int beta_qu,int f_align,float quantization_scale_adj,ap_uint<1> model[5][8],int M,int first_row, int row_count, hls::stream<ITYPE> &A_fifo_adj,hls::stream<int> &col_indices_fifo_adj, hls::stream<int> &rnnz_fifo_adj,
int *rowPtr_adj,int *columnIndex_adj,INTYPE *values_adj, int B_index)
{

    int last_index_adj;
    bool gemm_mode;

    gemm_mode = model[B_index][0];

    bool linear_mode;

    linear_mode =  model[B_index][6];

    if (gemm_mode==0)
    {

       columnIndex_adj += rowPtr_adj[first_row];
       values_adj += rowPtr_adj[first_row];
       rowPtr_adj += first_row;
       last_index_adj = nnz_adj;
    }
    else
    {
        values_adj+=first_row*M;
        last_index_adj = row_count*M;
    }

    readptr_coo_adj(nnz_adj,linear_mode,gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);
    readval_coo_adj2(beta_qu,f_align,quantization_scale_adj,linear_mode,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);

}

void reada22_csr(int beta_qu,int f_align,float quantization_scale_adj,bool gemm_mode,int M,int first_row, int row_count, hls::stream<ITYPE> &A_fifo_adj,hls::stream<int> &col_indices_fifo_adj, hls::stream<int> &rnnz_fifo_adj,
int *rowPtr_adj,int *columnIndex_adj,INTYPE *values_adj)
{

    int last_index_adj;

    if (gemm_mode==0)
    {

       last_index_adj=rowPtr_adj[first_row+row_count]-rowPtr_adj[first_row];
       columnIndex_adj += rowPtr_adj[first_row];
       values_adj += rowPtr_adj[first_row];
       rowPtr_adj += first_row;
    }
    else
    {
        last_index_adj=row_count*M;
        values_adj+=first_row*M;
    }

    readptr_csr_adj(gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);
    readval_csr_adj2(beta_qu,f_align,quantization_scale_adj,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);

}

void dsp_kernel_wrapper_adj_4(int block_size,int M,hls::stream<ITYPE> &A_fifo,hls::stream<int> &col_indices_fifo,QTYPE b_block1[B_HEIGHT/4][B_WIDTH_BLOCK],
        QTYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],QTYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],QTYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],
        ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK])
{

#if defined FLOAT || defined HALF

FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
#pragma HLS ARRAY_PARTITION variable=acc_part complete

FTYPE acc_float[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_float complete

for (int j = 0; j < B_WIDTH_BLOCK; j++) {

    #pragma HLS UNROLL

        acc_float[j] = 0;
}

        RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
        #pragma HLS UNROLL
            for (int l = 0; l < FADD_LATENCY_ADJ; l++) {
            #pragma HLS UNROLL
                for (int z = 0; z < SPMM_BLOCK; z++){
                    #pragma HLS UNROLL
                        acc_part[l][j][z] = 0;
                }
        }
    }

          int BM = M[SPMM_BLOCK-1];

          int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
          M_aux[0] = 0;
          for (int j = 1; j < SPMM_BLOCK+1; j++)
          {
             #pragma HLS UNROLL
             M_aux[j] = M[j-1];
          }

        DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_ADJ) {
        #pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

        DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_ADJ; i++) {

            DTYPE v;
            int ci;
            if ((k+i) < BM) //avoid trying to read empty FIFO that only contains M elements
            {
                v = A_fifo.read();
                ci = col_indices_fifo.read();
            }
                else
            {
                v=0;
                ci=0;
            }

            dsp_kernel_float_adj_4(block_size,v,b_block1,b_block2,b_block3,b_block4,ci,zero_point_lhs,zero_point_rhs,acc_float);

                for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                    for (int z = 0; z < SPMM_BLOCK; z++)
                    {
                            #pragma HLS UNROLL
                            if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
                                                    acc_part[i][j][z] += acc_float[j];
                    }
                #ifdef simulation
                if (acc_part[i][j] > max_adj)
                            max_adj = acc_part[i][j];
                        if (acc_part[i][j] < min_adj)
                            min_adj = acc_part[i][j];
                #endif
            }

                  } //i loop

    } // k loop

for (int j = 0; j < B_WIDTH_BLOCK; j++) {
        #pragma HLS UNROLL
        for (int l = 1; l < FADD_LATENCY_ADJ; l++) {
            for (int z = 0; z < SPMM_BLOCK; z++)
            {
                 acc_part[0][j][z] += acc_part[l][j][z];
            }
        }
    }

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
            for (int z = 0; z < SPMM_BLOCK; z++)
            {
              #pragma HLS UNROLL
              FTYPE acc_part_float = acc_part[0][j][z];
              acc2[j][z] = acc_part_float;
            }
    }

#endif

    #ifdef EIGHTBIT

    ITYPE acc[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc complete

     DSP_LOOP_SPMM: for (int i = 0; i < M; i++) {
             #pragma HLS PIPELINE
            DTYPE v = A_fifo.read();

            int ci = col_indices_fifo.read();

            dsp_kernel_int_adj_4(block_size,v,b_block1,b_block2,
                    b_block3,b_block4,
                    ci,zero_point_lhs,zero_point_rhs,acc);

            for (int j = 0; j < B_WIDTH_BLOCK; j++) {

                #pragma HLS UNROLL
                acc2[j] += acc[j];
            }//j loop

            } //i loop

    #endif

}

void dsp_kernel_wrapper_adj_2(int block_size,int M[SPMM_BLOCK],hls::stream<ITYPE> &A_fifo,hls::stream<int> &col_indices_fifo,QTYPE b_block1[B_HEIGHT/4][B_WIDTH_BLOCK],
        QTYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],
        ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK])
{

        #if defined FLOAT || defined HALF

    FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_part complete

    FTYPE acc_float[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc_float complete

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {

        #pragma HLS UNROLL

            acc_float[j] = 0;
    }

        RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
                for (int l = 0; l < FADD_LATENCY_ADJ; l++) {
                #pragma HLS UNROLL
                    for (int z = 0; z < SPMM_BLOCK; z++){
                                    acc_part[l][j][z] = 0;
                }
            }
        }

        int BM = M[SPMM_BLOCK-1];

         int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
         M_aux[0] = 0;
         for (int j = 1; j < SPMM_BLOCK+1; j++)
         {
            #pragma HLS UNROLL
            M_aux[j] = M[j-1];
         }

        DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_ADJ) {
            #pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

            DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_ADJ; i++) {

                DTYPE v;
                int ci;
                if ((k+i) < BM) //avoid trying to read empty FIFO that only contains M elements
                {
                    v = A_fifo.read();
                    ci = col_indices_fifo.read();
                }
                    else
                {
                    v=0;
                    ci=0;
                }

                dsp_kernel_float_adj_2(block_size,v,b_block1,b_block2,ci,zero_point_lhs,zero_point_rhs,acc_float);

                    for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                        for (int z = 0; z < SPMM_BLOCK; z++)
                        {
                            #pragma HLS UNROLL
                            if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
                                        acc_part[i][j][z] += acc_float[j];
                            }//#pragma HLS UNROLL
                    #ifdef simulation
                    if (acc_part[i][j] > max_adj)
                                max_adj = acc_part[i][j];
                            if (acc_part[i][j] < min_adj)
                                min_adj = acc_part[i][j];
                    #endif
                }

                      } //i loop

        } // k loop

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
            for (int l = 1; l < FADD_LATENCY_ADJ; l++) {
                #pragma HLS unroll
                  for (int z = 0; z < SPMM_BLOCK; z++)
                  {
                    acc_part[0][j][z] += acc_part[l][j][z];
                  }

            }
        }

        for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
                for (int z = 0; z < SPMM_BLOCK; z++)
                {
                   FTYPE acc_part_float = acc_part[0][j][z];
                   acc2[j][z] = acc_part_float;
                }
        }

    #endif

        #ifdef EIGHTBIT

                ITYPE acc[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        int BM = M[SPMM_BLOCK-1];

        int M_aux[SPMM_BLOCK+1];
        M_aux[0] = 0;
        for (int j = 1; j < SPMM_BLOCK+1; j++)
        {
                #pragma HLS UNROLL
                M_aux[j] = M[j-1];
        }

         DSP_LOOP_SPMM: for (int i = 0; i < BM; i+=1) {
             #pragma HLS PIPELINE
                    DTYPE v = A_fifo.read();

                int ci = col_indices_fifo.read();

                dsp_kernel_int_adj_2(block_size,v,b_block1,b_block2,
                        ci,zero_point_lhs,zero_point_rhs,acc);

                for (int j = 0; j < B_WIDTH_BLOCK; j++) {

                    #pragma HLS UNROLL
                    for (int z = 0; z < SPMM_BLOCK; z++)
                    {
                            #pragma HLS UNROLL
                            if (i>=M_aux[z]&&i<M_aux[z+1])
                                    acc2[j][z] += acc[j];
                    }//z loop
                }//j loop

                } //i loop

        #endif

}

void dsp_kernel_wrapper_adj_1(int block_size,int M,hls::stream<TTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,QTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],
        ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK])
{

        #if defined FLOAT || defined HALF

    FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0 //partition all dimensions

    FTYPE acc_float[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc_float complete

    for (int j = 0; j < B_WIDTH_BLOCK; j++) {

        #pragma HLS UNROLL

            acc_float[j] = 0;
    }

        RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
                for (int l = 0; l < FADD_LATENCY_ADJ; l++) {
                #pragma HLS UNROLL
                    for (int z = 0; z < SPMM_BLOCK; z++){
                    acc_part[l][j][z] = 0;
                }
            }
        }

        int BM = M[SPMM_BLOCK-1];

         int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
         M_aux[0] = 0;
         for (int j = 1; j < SPMM_BLOCK+1; j++)
         {
            #pragma HLS UNROLL
            M_aux[j] = M[j-1];
         }

        DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_ADJ) {
            #pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

            DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_ADJ; i++) {

                DTYPE v;
                int ci;
                if ((k+i) < BM) //avoid trying to read empty FIFO that only contains M elements
                {

                    v = A_fifo.read();
                    ci = col_indices_fifo.read();
                }
                    else
                {
                    v=0;
                    ci=0;
                }

                dsp_kernel_float_adj_1(v,b_block1,ci,zero_point_lhs,zero_point_rhs,acc_float);

                    for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                        for (int z = 0; z < SPMM_BLOCK; z++)
                        {
                            #pragma HLS UNROLL
                            if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
                                acc_part[i][j][z] += acc_float[j];
                        }
                    #ifdef simulation
                    if (acc_part[i][j] > max_adj)
                                max_adj = acc_part[i][j];
                            if (acc_part[i][j] < min_adj)
                                min_adj = acc_part[i][j];
                    #endif
                }

                      } //i loop

        } // k loop

    ACC_PART1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
        ACC_PART2 : for (int z = 0; z < SPMM_BLOCK; z++)
            {
                #pragma HLS UNROLL
                ACC_PART3 : for (int l = 1; l < FADD_LATENCY_ADJ; l++) {
                    #pragma HLS PIPELINE=1
                     acc_part[0][j][z] += acc_part[l][j][z];
                }
            }
        }

    FLOAT_PART1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
        FLOAT_PART2 : for (int z = 0; z < SPMM_BLOCK; z++)
                {
                    #pragma HLS UNROLL
                   FTYPE acc_part_float = acc_part[0][j][z];
                   acc2[j][z] = acc_part_float;
                }
        }

    #endif

        #ifdef EIGHTBIT

            ITYPE acc[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc complete

         DSP_LOOP_SPMM: for (int i = 0; i < M; i++) {
             #pragma HLS PIPELINE
                    TTYPE v = A_fifo.read();

                int ci = col_indices_fifo.read();

                dsp_kernel_int_adj_1(block_size,v,b_block1,//b_block2,
                        ci,zero_point_lhs,zero_point_rhs,acc);

                for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                        acc2[j] += acc[j];
                }//j loop

          } //i loop

        #endif

}

void dsp_kernel_wrapper_fea(bool gemm_mode,int M,hls::stream<FTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK])
{

#if defined FLOAT || defined HALF

        ITYPE acc_part[FADD_LATENCY_FEA][B_WIDTH_BLOCK][SPMM_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0 //partition all dimensions

        ITYPE acc_float[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc_float complete

        for (int j = 0; j < B_WIDTH_BLOCK; j++) {

            #pragma HLS UNROLL

                acc_float[j] = 0;
        }

            RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
                     for (int l = 0; l < FADD_LATENCY_FEA; l++) {
                     #pragma HLS UNROLL
                        for (int z = 0; z < SPMM_BLOCK; z++){
                            #pragma HLS UNROLL
                            acc_part[l][j][z] = 0;
                }
              }
            }

             int BM = M[SPMM_BLOCK-1];

             int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
             M_aux[0] = 0;
             for (int j = 1; j < SPMM_BLOCK+1; j++)
             {
                #pragma HLS UNROLL
                M_aux[j] = M[j-1];
             }

            DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_FEA) {
                #pragma HLS PIPELINE II=FADD_LATENCY_FEA

                DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_FEA; i++) {
                    DTYPE v;
                    int ci;
                    if ((k+i) < BM) //avoid trying to read empty FIFO that only contains BM elements
                    {
                        v = A_fifo.read();
                            ci = col_indices_fifo.read();
                    }
                        else
                    {
                        v=0;
                        ci=0;
                    }

                    dsp_kernel_float_fea(v,b_block,ci,zero_point_lhs,zero_point_rhs,acc_float);

                    SPMM_BLOCK_LOOP1 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                        #pragma HLS UNROLL
                            SPMM_BLOCK_LOOP2 : for (int z = 0; z < SPMM_BLOCK; z++)
                            {
                            #pragma HLS PIPELINE II=1
                            if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
                                    acc_part[i][j][z] += acc_float[j];
                            }//z loop
                    } //j loop

                } //i loop

            } // k loop

            ACC_PART1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
                ACC_PART2: for (int z = 0; z < SPMM_BLOCK; z++)
                {
                    #pragma HLS UNROLL
                    ACC_PART3 : for (int l = 1; l < FADD_LATENCY_FEA; l++)
                    {
                        #pragma HLS PIPELINE II=1
                         acc_part[0][j][z] += acc_part[l][j][z];
                    }
                }
            }

            ACC_PART_FLOAT1 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
                ACC_PART_FLOAT2 : for (int z = 0; z < SPMM_BLOCK; z++)
                {
                #pragma HLS UNROLL
                FTYPE acc_part_float = acc_part[0][j][z];
                    acc2[j][z] = acc_part_float;
                }

            }

        #endif

#ifdef EIGHTBIT

        ITYPE acc[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS bind_op variable=acc op=add impl=dsp

         DSP_LOOP_SPMM: for (int i = 0; i < M; i++) {
             #pragma HLS PIPELINE

                FTYPE v = A_fifo.read();

                int ci;
                    ci = col_indices_fifo.read();

                dsp_kernel_int_fea(v,b_block,ci,zero_point_lhs,zero_point_rhs,acc);

                for (int j = 0; j < B_WIDTH_BLOCK; j++) {

                    #pragma HLS UNROLL
                    acc2[j] += acc[j];

                 }//j loop

                } //i loop

        #endif

}

void dsp_kernel_wrapper_lin(bool gemm_mode,int M,hls::stream<LTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK])
{

#if defined FLOAT || defined HALF

        ITYPE acc_part[FADD_LATENCY_FEA][B_WIDTH_BLOCK][SPMM_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0 //partition all dimensions

        ITYPE acc_float[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc_float complete

        for (int j = 0; j < B_WIDTH_BLOCK; j++) {

            #pragma HLS UNROLL

                acc_float[j] = 0;
        }

            RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
                     for (int l = 0; l < FADD_LATENCY_FEA; l++) {
                     #pragma HLS UNROLL
                        for (int z = 0; z < SPMM_BLOCK; z++){
                            #pragma HLS UNROLL
                            acc_part[l][j][z] = 0;
                }
              }
            }

             int BM = M[SPMM_BLOCK-1];

             int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
             M_aux[0] = 0;
             for (int j = 1; j < SPMM_BLOCK+1; j++)
             {
                #pragma HLS UNROLL
                M_aux[j] = M[j-1];
             }

            DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_FEA) {
                #pragma HLS PIPELINE II=FADD_LATENCY_FEA

                DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_FEA; i++) {
                    DTYPE v;
                    int ci;
                    if ((k+i) < BM) //avoid trying to read empty FIFO that only contains BM elements
                    {
                        v = A_fifo.read();
                            ci = col_indices_fifo.read();
                    }
                        else
                    {
                        v=0;
                        ci=0;
                    }

                    dsp_kernel_float_fea(v,b_block,ci,zero_point_lhs,zero_point_rhs,acc_float);

                    SPMM_BLOCK_LOOP1 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                        #pragma HLS UNROLL
                            SPMM_BLOCK_LOOP2 : for (int z = 0; z < SPMM_BLOCK; z++)
                            {
                            #pragma HLS PIPELINE II=1
                            if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
                                    acc_part[i][j][z] += acc_float[j];
                            }//z loop
                    } //j loop

                } //i loop

            } // k loop

            ACC_PART1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
                ACC_PART2: for (int z = 0; z < SPMM_BLOCK; z++)
                {
                    #pragma HLS UNROLL
                    ACC_PART3 : for (int l = 1; l < FADD_LATENCY_FEA; l++)
                    {
                        #pragma HLS PIPELINE II=1
                         acc_part[0][j][z] += acc_part[l][j][z];
                    }
                }
            }

            ACC_PART_FLOAT1 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
                ACC_PART_FLOAT2 : for (int z = 0; z < SPMM_BLOCK; z++)
                {
                #pragma HLS UNROLL
                FTYPE acc_part_float = acc_part[0][j][z];
                    acc2[j][z] = acc_part_float;
                }

            }

        #endif

#ifdef EIGHTBIT

        ITYPE acc[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS bind_op variable=acc op=add impl=dsp

         DSP_LOOP_SPMM: for (int i = 0; i < M; i++) {
             #pragma HLS PIPELINE

                LTYPE v = A_fifo.read();

                int ci;
                    ci = col_indices_fifo.read();

                dsp_kernel_int_lin(v,b_block,ci,zero_point_lhs,zero_point_rhs,acc);

                for (int j = 0; j < B_WIDTH_BLOCK; j++) {

                    #pragma HLS UNROLL
                    acc2[j] += acc[j];

                 }//j loop

                } //i loop

        #endif

}

void scale(ap_int<32> *quantized_multiplier, ap_int<32> *shift, ap_int<32> *bias, ap_int<8> zero_point_dst, ap_int<8> clamp_max,ap_int<8> clamp_min,int N, int M, int P, hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK],int B_index, int B_index_loop,int tail,hls::stream<ITYPE> write_fifo[B_WIDTH_BLOCK])
{

            int B_WIDTH_INT;
            if (B_index < (B_index_loop-1))
                B_WIDTH_INT = B_WIDTH_BLOCK;
            else
                B_WIDTH_INT = tail;

            #if defined FLOAT || defined HALF
                LOOP_CH1f:    for (int i = 0; i < N; i++) {
                 LOOP_CW1f: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                    #pragma HLS PIPELINE II=1
                    if (j<B_WIDTH_INT)
                    {
                        #ifdef ENABLE_SCALING
                            int bias_int = bias[i];
                            FTYPE bias_val_float=*(FTYPE*)&(bias_int);
                            DTYPE C_fifo_int = C_fifo[j].read();
                                FTYPE C_fifo_float=*(FTYPE*)&C_fifo_int;
                            FTYPE zero_point_dst_float=(FTYPE)zero_point_dst; //simply cast to float
                            FTYPE clamp_min_float=(FTYPE)clamp_min; //simply cast to float
                            FTYPE clamp_max_float=(FTYPE)clamp_max; //simply cast to float
                            FTYPE C_temp_float = C_fifo_float + bias_val_float + zero_point_dst_float;
                                        if (C_temp_float < clamp_min_float) C_temp_float = clamp_min_float;
                            if (C_temp_float > clamp_max_float) C_temp_float = clamp_max_float;
                            DTYPE C_out = *(int*)&C_temp_float;
                            write_fifo[j] << C_out;
                        #else
                            DTYPE C_fifo_int = C_fifo[j].read();
                                write_fifo[j] << C_fifo_int;
                        #endif

                    }

                 }
                 }
            #endif

            #if defined EIGHTBIT
                LOOP_CH1:    for (int i = 0; i < N; i+=4) {
                ap_int<32> bias_val[4];
                ap_int<32> shift_val[4];
                ap_int<32> mult_val[4];
                bias_val[0] =  bias[i];
                bias_val[1] =  bias[i+1];
                bias_val[2] =  bias[i+2];
                bias_val[3] =  bias[i+3];
                shift_val[0] = shift[i];
                shift_val[1] = shift[i+1];
                shift_val[2] = shift[i+2];
                shift_val[3] = shift[i+3];
                mult_val[0] = quantized_multiplier[i];
                mult_val[1] = quantized_multiplier[i+1];
                mult_val[2] = quantized_multiplier[i+2];
                mult_val[3] = quantized_multiplier[i+3];
                LOOP_CW1: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                    #pragma HLS PIPELINE II=4
                    DTYPE C_out;
                    LOOP_CH3:    for (int z = 0; z < 4; z++) {
                     #pragma HLS loop_tripcount min=1 max=1 avg=1
                        if (j<B_WIDTH_INT)
                        {
                            #ifdef ENABLE_SCALING
                            ap_int<64> C_temp1;
                            C_temp1 =  C_fifo[j].read() + bias_val[z];
                            ap_int<32> total_shift1 = 31 - shift_val[z];
                                    ap_int<64> round1 = (ap_int<64>)1 << (total_shift1 - 1);
                            C_temp1 = C_temp1*mult_val[z] + round1;
                            C_temp1 = (C_temp1 >> total_shift1) + zero_point_dst;
                            ap_int<8> C_temp5 = C_temp1;
                                        if (C_temp1 < clamp_min) C_temp5 = clamp_min;
                            if (C_temp1 > clamp_max) C_temp5 = clamp_max;
                            C_out = ((C_out >> 8) | ((int)C_temp5 << 24));

                            if (z==3)
                            {
                                write_fifo[j].write(C_out);

                            }
                            #else
                                C_out =  C_fifo[j].read();
                                write_fifo[j].write(C_out);
                            #endif

                        }
                    }

                    }
            }
        #endif
}

void compute2_4(bool relu, float relu_t, int block_size,ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<ITYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo, QTYPE B_accel1[B_HEIGHT/2][B_WIDTH_BLOCK],QTYPE B_accel2[B_HEIGHT/2][B_WIDTH_BLOCK],
QTYPE B_accel3[B_HEIGHT/4][B_WIDTH_BLOCK],QTYPE B_accel4[B_HEIGHT/4][B_WIDTH_BLOCK],
hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK])
{

        ITYPE acc[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        ITYPE acc2[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0

          int B_WIDTH_INT;
          ITYPE C_fifo_val;

          B_WIDTH_INT = B_WIDTH_BLOCK;

        for (int A_index = 0; A_index < row_count; A_index++) {

            LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
                    acc2[j] = 0;
                }

            int rnnz;
            rnnz = rnnz_fifo.read();

            dsp_kernel_wrapper_adj_4(block_size,rnnz,A_fifo,col_indices_fifo,B_accel1,
                    B_accel2,
                    B_accel3,B_accel4,
                    zero_point_lhs,zero_point_rhs,acc2);

            LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                    #pragma HLS UNROLL
                    if (j < B_WIDTH_INT)
                    {

                        if (relu == 0)
                            C_fifo_val = acc2[j];
                        else
                            if (acc2[j] < (ITYPE)relu_t)
                               C_fifo_val = 0.0;
                            else
                               C_fifo_val = acc2[j];

                        C_fifo[j].write(C_fifo_val);

                    }
               }

              } // A_index loop

}

void compute2_2(int block_size,ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<ITYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> rnnz_fifo[SPMM_BLOCK], QTYPE B_accel1[B_HEIGHT/2][B_WIDTH_BLOCK],QTYPE B_accel2[B_HEIGHT/2][B_WIDTH_BLOCK],
hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK][SPMM_BLOCK],int B_index, int B_index_loop, int tail)
{

        ITYPE acc[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc complete

        ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0

          int B_WIDTH_INT;

          if (B_index < (B_index_loop-1))
            B_WIDTH_INT = B_WIDTH_BLOCK;
          else
            B_WIDTH_INT = tail;

        for (int A_index = 0; A_index < row_count; A_index+=SPMM_BLOCK) {

            LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
                LOOP_ACC22 : for (int i = 0; i < SPMM_BLOCK; i++) {
                    #pragma HLS UNROLL
                    acc2[j][i] = 0;
                }
            }

            int rnnz[SPMM_BLOCK];
            int crows = 0;
            LOOP_RNNZ :for (int i = 0; i < SPMM_BLOCK; i++) {
                #pragma HLS UNROLL
                rnnz[i] = rnnz_fifo[i].read();
                if ((A_index+i)<row_count)
                    crows++;

            }

            dsp_kernel_wrapper_adj_2(block_size,rnnz,A_fifo,col_indices_fifo,B_accel1,
                    B_accel2,
                    zero_point_lhs,zero_point_rhs,acc2);

            LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                    #pragma HLS UNROLL
                    if (j < B_WIDTH_INT)
                    {
                        #ifdef simulation
                        if (acc2[j] < acc2_adj_min)
                            acc2_adj_min = acc2[j];
                        else if (acc2[j] > acc2_adj_max)
                            acc2_adj_max = acc2[j];
                        #endif
                        LOOP_C_BUF2 : for (int i = 0; i < SPMM_BLOCK; i++) {
                            #pragma HLS UNROLL
                            if (i<crows)
                                C_fifo[j][i].write(acc2[j][i]);
                        }

                    }
            }

              } // A_index loop

}

void compute2_1(ap_uint<1> model[5][8],float srelu[5],int block_size,ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<TTYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo, QTYPE B_accel1[B_HEIGHT/2][B_WIDTH_BLOCK],
hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK],int B_index)
{

    ITYPE acc[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    ITYPE acc2[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0

      int B_WIDTH_INT;
      ITYPE C_fifo_val;

        B_WIDTH_INT = B_WIDTH_BLOCK;

    bool relu;

    relu = model[B_index][4];

    float relu_t = srelu[B_index];

    bool linear_mode;

    linear_mode = model[B_index][6];

    if (linear_mode==0)
    {
     for (int A_index = 0; A_index < row_count; A_index+=SPMM_BLOCK) {

        LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
                acc2[j] = 0;
        }

        int rnnz;
        rnnz = rnnz_fifo.read();

        dsp_kernel_wrapper_adj_1(block_size,rnnz,A_fifo,col_indices_fifo,B_accel1,
                zero_point_lhs,zero_point_rhs,acc2);

        LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
                #pragma HLS UNROLL
                if (j < B_WIDTH_INT)
                {

                    if(relu==0)
                            C_fifo_val = acc2[j];
                    else
                        if(acc2[j] < (ITYPE)relu_t)
                            C_fifo_val = 0.0;
                        else
                            C_fifo_val = acc2[j];

                    C_fifo[j].write(C_fifo_val);
                }
        }

          } // A_index loop
    }

}

QTYPE8 float_to_fix(float f_in,int n_bits)
{
    float f=(1<<n_bits);
    QTYPE8 i_out = (f_in*f)*(1.0/f);
    return i_out;
}

void compute1_1(STYPE scale_fea[5],ITYPE* max_fea,int quantized_multiplier,ap_uint<1> model[5][8],ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<FTYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo,BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],QTYPE C_buf1[B_HEIGHT][B_WIDTH_BLOCK],
        QTYPE A_buf1[B_HEIGHT][B_WIDTH_BLOCK], int B_index)
{

    ITYPE acc[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    ITYPE acc2[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc2 complete //all dimensions are partitioned

      int B_WIDTH_INT;

            B_WIDTH_INT = B_WIDTH_BLOCK;

     bool gemm_mode;
     gemm_mode = model[B_index][1];

     bool linear_mode;

     linear_mode = model[B_index][6];

     if(linear_mode==0)
     {

      for (int A_index = 0; A_index < row_count; A_index++) {

        LOOP_ACC21 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
                acc2[j] = 0;
        }

        int rnnz;

        rnnz = rnnz_fifo.read();

        dsp_kernel_wrapper_fea(gemm_mode,rnnz,A_fifo,col_indices_fifo,B_accel,zero_point_lhs,zero_point_rhs,acc2);

        LOOP_C_BUF1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
            if (j < B_WIDTH_INT)
            {
                #ifdef simulation
                if (acc2[j] < acc2_fea_min)
                    acc2_fea_min = acc2[j];
                else if (acc2[j] > acc2_fea_max)
                    acc2_fea_max = acc2[j];
                #endif

                    ITYPE cur_val = ITYPE(acc2[j]);

                    if (cur_val > *max_fea)
                    {
                        *max_fea= cur_val;
                    }

                    ap_fixed<32, 16>  acc2_temp_1 = acc2[j];

                    QTYPE2 acc2_temp_1_2 = QTYPE2(acc2_temp_1 >> scale_fea[B_index]);
                    QTYPE4 acc2_temp_1_4 = QTYPE4(acc2_temp_1 >> scale_fea[B_index]);
                    QTYPE8 acc2_temp_1_8 = QTYPE8(acc2_temp_1 >> scale_fea[B_index]);
                    QTYPE acc2_temp_1_16 = QTYPE(acc2_temp_1 >> scale_fea[B_index]);

                   #if GAT_ENABLE == 1

                     if(quantized_multiplier==2)
                     {
                             C_buf1[A_index][j]=acc2_temp_1_2;
                             A_buf1[A_index][j]=acc2_temp_1_2;
                     }
                     else if(quantized_multiplier==4)
                     {
                             C_buf1[A_index][j]=acc2_temp_1_4;
                             A_buf1[A_index][j]=acc2_temp_1_4;
                     }
                     else if(quantized_multiplier==8)
                     {
                             C_buf1[A_index][j]=acc2_temp_1_8;
                             A_buf1[A_index][j]=acc2_temp_1_8;
                     }
                     else
                     {
                            C_buf1[A_index][j]=acc2_temp_1_16;
                            A_buf1[A_index][j]=acc2_temp_1_16;
                     }

                   #else

                     if(quantized_multiplier==2)
                         C_buf1[A_index][j]=acc2_temp_1_2;
                     else if(quantized_multiplier==4)
                         C_buf1[A_index][j]=acc2_temp_1_4;
                     else if(quantized_multiplier==8)
                         C_buf1[A_index][j]=acc2_temp_1_8;

                     else
                     {
                            C_buf1[A_index][j]=acc2_temp_1_16;

                     }

                    #endif

                } //c_buf loop

           } // j < B_WIDTH_BLOCK

          } // A_index loop
     } // linear_mode
}

void compute1_12(STYPE scale_fea[5],ITYPE* max_fea,int quantized_multiplier,bool gemm_mode,ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<LTYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo,BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK], QTYPE linear_pipo[B_HEIGHT][B_WIDTH_BLOCK], int B_index)
{

    ITYPE acc[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    ITYPE acc2[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc2 complete //all dimensions are partitioned

      int B_WIDTH_INT;

            B_WIDTH_INT = B_WIDTH_BLOCK;

    for (int A_index = 0; A_index < row_count; A_index++) {

        LOOP_ACC21 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
                acc2[j] = 0;
        }

        int rnnz;

        rnnz = rnnz_fifo.read();

        dsp_kernel_wrapper_lin(gemm_mode,rnnz,A_fifo,col_indices_fifo,B_accel,zero_point_lhs,zero_point_rhs,acc2);

        LOOP_C_BUF1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
            if (j < B_WIDTH_INT)
            {
                #ifdef simulation
                if (acc2[j] < acc2_fea_min)
                    acc2_fea_min = acc2[j];
                else if (acc2[j] > acc2_fea_max)
                    acc2_fea_max = acc2[j];
                #endif

                    ITYPE cur_val = ITYPE(acc2[j]);

                    if (cur_val > *max_fea)
                    {
                        *max_fea= cur_val;
                    }

                    ap_fixed<32, 16>  acc2_temp_1 = acc2[j];

                    QTYPE2 acc2_temp_1_2 = QTYPE2(acc2_temp_1 >> scale_fea[B_index]);
                    QTYPE4 acc2_temp_1_4 = QTYPE4(acc2_temp_1 >> scale_fea[B_index]);
                    QTYPE8 acc2_temp_1_8 = QTYPE8(acc2_temp_1 >> scale_fea[B_index]);
                    QTYPE acc2_temp_1_16 = QTYPE(acc2_temp_1 >> scale_fea[B_index]);

                     if(quantized_multiplier==2)
                         linear_pipo[A_index][j]=acc2_temp_1_2;
                     else if(quantized_multiplier==4)
                         linear_pipo[A_index][j]=acc2_temp_1_4;
                     else if(quantized_multiplier==8)
                         linear_pipo[A_index][j]=acc2_temp_1_8;
                     else
                         linear_pipo[A_index][j]=acc2_temp_1_16;

                } //c_buf loop

           } // j < B_WIDTH_BLOCK

          } // A_index loop

}

void compute1_4(STYPE scale_fea,ITYPE* max_fea,int quantized_multiplier,bool gemm_mode,ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<FTYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo,BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],QTYPE C_buf1[B_HEIGHT/4][B_WIDTH_BLOCK],
        QTYPE C_buf2[B_HEIGHT/4][B_WIDTH_BLOCK],QTYPE C_buf3[B_HEIGHT/4][B_WIDTH_BLOCK],QTYPE C_buf4[B_HEIGHT/4][B_WIDTH_BLOCK],
        QTYPE A_buf1[B_HEIGHT/4][B_WIDTH_BLOCK])
{

    ITYPE acc[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    ITYPE acc2[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc2 complete  //all dimensions are partitioned

     int B_WIDTH_INT;

            B_WIDTH_INT = B_WIDTH_BLOCK;

    for (int A_index = 0; A_index < row_count; A_index++) {

        LOOP_ACC21 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
           #pragma HLS UNROLL
                acc2[j] = 0;
        }

        int rnnz;

                rnnz = rnnz_fifo.read();

        dsp_kernel_wrapper_fea(gemm_mode,rnnz,A_fifo,col_indices_fifo,B_accel,zero_point_lhs,zero_point_rhs,acc2);

        LOOP_C_BUF1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
            if (j < B_WIDTH_INT)
            {
                #ifdef simulation
                if (acc2[j] < acc2_fea_min)
                    acc2_fea_min = acc2[j];
                else if (acc2[j] > acc2_fea_max)
                    acc2_fea_max = acc2[j];
                #endif

                    int cur_val = abs(int(acc2[j]));

                    if (cur_val > *max_fea)
                    {
                        *max_fea= cur_val;
                    }

                    ap_fixed<32, 16>  acc2_temp_1 = acc2[j];

                    QTYPE2 acc2_temp_1_2 = QTYPE2(acc2_temp_1 >> scale_fea);
                    QTYPE4 acc2_temp_1_4 = QTYPE4(acc2_temp_1 >> scale_fea);
                    QTYPE8 acc2_temp_1_8 = QTYPE8(acc2_temp_1 >> scale_fea);
                    QTYPE acc2_temp_1_16 = QTYPE(acc2_temp_1 >> scale_fea);

                   #if GAT_ENABLE == 1

                     if(quantized_multiplier==2)
                     {
                             C_buf1[A_index][j]=acc2_temp_1_2;
                             C_buf2[A_index][j]=acc2_temp_1_2;
                             C_buf3[A_index][j]=acc2_temp_1_2;
                             C_buf4[A_index][j]=acc2_temp_1_2;
                             A_buf1[A_index][j]=acc2_temp_1_2;
                     }
                     else if(quantized_multiplier==4)
                     {
                             C_buf1[A_index][j]=acc2_temp_1_4;
                             C_buf2[A_index][j]=acc2_temp_1_4;
                             C_buf3[A_index][j]=acc2_temp_1_4;
                             C_buf4[A_index][j]=acc2_temp_1_4;
                             A_buf1[A_index][j]=acc2_temp_1_4;
                     }
                     else if(quantized_multiplier==8)
                     {
                             C_buf1[A_index][j]=acc2_temp_1_8;
                             C_buf2[A_index][j]=acc2_temp_1_8;
                             C_buf3[A_index][j]=acc2_temp_1_8;
                             C_buf4[A_index][j]=acc2_temp_1_8;
                             A_buf1[A_index][j]=acc2_temp_1_8;
                     }
                     else
                     {
                            C_buf1[A_index][j]=acc2_temp_1_16;
                            C_buf2[A_index][j]=acc2_temp_1_16;
                            C_buf3[A_index][j]=acc2_temp_1_16;
                            C_buf4[A_index][j]=acc2_temp_1_16;
                            A_buf1[A_index][j]=acc2_temp_1_16;
                     }

                   #else

                     if(quantized_multiplier==2)
                     {
                         C_buf1[A_index][j]=acc2_temp_1_2;
                         C_buf2[A_index][j]=acc2_temp_1_2;
                         C_buf3[A_index][j]=acc2_temp_1_2;
                         C_buf4[A_index][j]=acc2_temp_1_2;
                     }
                     else if(quantized_multiplier==4)
                     {
                         C_buf1[A_index][j]=acc2_temp_1_4;
                         C_buf2[A_index][j]=acc2_temp_1_4;
                         C_buf3[A_index][j]=acc2_temp_1_4;
                         C_buf4[A_index][j]=acc2_temp_1_4;

                     }
                     else if(quantized_multiplier==8)
                     {
                         C_buf1[A_index][j]=acc2_temp_1_8;
                         C_buf2[A_index][j]=acc2_temp_1_8;
                         C_buf3[A_index][j]=acc2_temp_1_8;
                         C_buf4[A_index][j]=acc2_temp_1_8;

                     }

                     else
                     {
                            C_buf1[A_index][j]=acc2_temp_1_16;
                            C_buf2[A_index][j]=acc2_temp_1_16;
                            C_buf3[A_index][j]=acc2_temp_1_16;
                            C_buf4[A_index][j]=acc2_temp_1_16;

                     }

                    #endif

                } //c_buf loop

           } // j < B_WIDTH_BLOCK

         } // A_index loop

}

void loop_attention(float deq_factor[5],int beta_qu,int f_align,float quantization_scale_adj,float quantization_scale_w[5],
        ap_uint<1> model[5][8],
        int nnz_adj1,int nnz_adj2,int nnz_adj3,int nnz_adj4,
        int * rowPtr_adj1,int * rowPtr_adj2,int * rowPtr_adj3,int * rowPtr_adj4,
        int *columnIndex_adj1, int *columnIndex_adj2, int *columnIndex_adj3, int *columnIndex_adj4,
        INTYPE *values_adj1,    INTYPE *values_adj2,    INTYPE *values_adj3,    INTYPE *values_adj4,
        int N_adj, int M_adj, ap_uint<8> P_w[5],
        INTYPE *A,
        #if(PIPO_BLOCKS>=2)
         hls::stream_of_blocks<buf> &A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
        #else
         buf A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
        #endif
        hls::stream_of_blocks<buf> &A_buffer31,hls::stream_of_blocks<buf> &A_buffer41,
        OUTTYPE* E1,
        OUTTYPE* S1,
        hls::stream<int> &rnnz_att_fifo1,hls::stream<int> &col_att_fifo1,hls::stream<TTYPE> &val_att_fifo1,
        hls::stream<int> &rnnz_att_fifo2,hls::stream<int> &col_att_fifo2,hls::stream<TTYPE> &val_att_fifo2,
        hls::stream<int> &rnnz_att_fifo3,hls::stream<int> &col_att_fifo3,hls::stream<TTYPE> &val_att_fifo3,
        hls::stream<int> &rnnz_att_fifo4,hls::stream<int> &col_att_fifo4,hls::stream<TTYPE> &val_att_fifo4,
        int layer_loop)

{

         hls::stream<TTYPE>   EO_fifo1("EO fifo1");
         #pragma HLS STREAM variable=EO_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_fifo1 type=FIFO impl=URAM
         hls::stream<int>   EO_rnnz_fifo1("EO rnnz fifo1");;
         #pragma HLS STREAM variable=EO_rnnz_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_rnnz_fifo1 type=FIFO impl=URAM
         hls::stream<TTYPE>   SO_fifo1("SO fifo1");
         #pragma HLS STREAM variable=SO_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_fifo1 type=FIFO impl=URAM
         hls::stream<int>   SO_rnnz_fifo1("SO rnnz fifo1");;
         #pragma HLS STREAM variable=SO_rnnz_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_rnnz_fifo1 type=FIFO impl=URAM

         hls::stream<TTYPE>   EO_fifo2("EO fifo2");
         #pragma HLS STREAM variable=EO_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_fifo2 type=FIFO impl=URAM
         hls::stream<int>   EO_rnnz_fifo2("EO rnnz fifo2");;
         #pragma HLS STREAM variable=EO_rnnz_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_rnnz_fifo2 type=FIFO impl=URAM
         hls::stream<TTYPE>   SO_fifo2("SO fifo2");
         #pragma HLS STREAM variable=SO_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_fifo2 type=FIFO impl=URAM
         hls::stream<int>   SO_rnnz_fifo2("SO rnnz fifo2");;
         #pragma HLS STREAM variable=SO_rnnz_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_rnnz_fifo2 type=FIFO impl=URAM

         hls::stream<TTYPE>   EO_fifo3("EO fifo3");
         #pragma HLS STREAM variable=EO_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_fifo3 type=FIFO impl=URAM
         hls::stream<int>   EO_rnnz_fifo3("EO rnnz fifo3");;
         #pragma HLS STREAM variable=EO_rnnz_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_rnnz_fifo3 type=FIFO impl=URAM
         hls::stream<TTYPE>   SO_fifo3("SO fifo3");
         #pragma HLS STREAM variable=SO_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_fifo3 type=FIFO impl=URAM
         hls::stream<int>   SO_rnnz_fifo3("SO rnnz fifo3");;
         #pragma HLS STREAM variable=SO_rnnz_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_rnnz_fifo3 type=FIFO impl=URAM

         hls::stream<TTYPE>   EO_fifo4("EO fifo4");
         #pragma HLS STREAM variable=EO_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_fifo4 type=FIFO impl=URAM
         hls::stream<int>   EO_rnnz_fifo4("EO rnnz fifo4");;
         #pragma HLS STREAM variable=EO_rnnz_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_rnnz_fifo4 type=FIFO impl=URAM
         hls::stream<TTYPE>   SO_fifo4("SO fifo4");
         #pragma HLS STREAM variable=SO_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_fifo4 type=FIFO impl=URAM
         hls::stream<int>   SO_rnnz_fifo4("SO rnnz fifo4");;
         #pragma HLS STREAM variable=SO_rnnz_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_rnnz_fifo4 type=FIFO impl=URAM

         hls::stream<TTYPE>   max_fifo1("max_fifo1");
         #pragma HLS STREAM variable=max_fifo1 depth=FIFO_DEPTH
         hls::stream<TTYPE>   E_fifo1("E fifo1");
         #pragma HLS STREAM variable=E_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_fifo1 type=FIFO impl=URAM
         hls::stream<ATYPE>   A_fifo1("A fifo1");
         #pragma HLS STREAM variable=A_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = A_fifo1 type=FIFO impl=URAM
         hls::stream<int>  E_col_indices_fifo1("E col fifo1");
         #pragma HLS STREAM variable=E_col_indices_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_col_indices_fifo1 type=FIFO impl=URAM
         hls::stream<int> E_rnnz_fifo1;
         #pragma HLS STREAM variable=E_rnnz_fifo1 depth=FIFO_DEPTH

         hls::stream<TTYPE>   max_fifo2("max_fifo2");
         #pragma HLS STREAM variable=max_fifo2 depth=FIFO_DEPTH
         hls::stream<TTYPE>   E_fifo2("E fifo2");
         #pragma HLS STREAM variable=E_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_fifo2 type=FIFO impl=URAM
         hls::stream<ATYPE>   A_fifo2("A fifo2");
         #pragma HLS STREAM variable=A_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = A_fifo2 type=FIFO impl=URAM
         hls::stream<int>  E_col_indices_fifo2("E col fifo2");
         #pragma HLS STREAM variable=E_col_indices_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_col_indices_fifo2 type=FIFO impl=URAM
         hls::stream<int> E_rnnz_fifo2;
         #pragma HLS STREAM variable=E_rnnz_fifo2 depth=FIFO_DEPTH

         hls::stream<TTYPE>   max_fifo3("max_fifo3");
         #pragma HLS STREAM variable=max_fifo3 depth=FIFO_DEPTH
         hls::stream<TTYPE>   E_fifo3("E fifo3");
         #pragma HLS STREAM variable=E_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_fifo3 type=FIFO impl=URAM
         hls::stream<ATYPE>   A_fifo3("A fifo3");
         #pragma HLS STREAM variable=A_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = A_fifo3 type=FIFO impl=URAM
         hls::stream<int>  E_col_indices_fifo3("E col fifo3");
         #pragma HLS STREAM variable=E_col_indices_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_col_indices_fifo3 type=FIFO impl=URAM
         hls::stream<int> E_rnnz_fifo3;
         #pragma HLS STREAM variable=E_rnnz_fifo3 depth=FIFO_DEPTH

         hls::stream<TTYPE>   max_fifo4("max_fifo4");
         #pragma HLS STREAM variable=max_fifo4 depth=FIFO_DEPTH
         hls::stream<TTYPE>   E_fifo4("E fifo4");
         #pragma HLS STREAM variable=E_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_fifo4 type=FIFO impl=URAM
         hls::stream<ATYPE>   A_fifo4("A fifo4");
         #pragma HLS STREAM variable=A_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = A_fifo4 type=FIFO impl=URAM
         hls::stream<int>  E_col_indices_fifo4("E col fifo4");
         #pragma HLS STREAM variable=E_col_indices_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_col_indices_fifo4 type=FIFO impl=URAM
         hls::stream<int> E_rnnz_fifo4;
         #pragma HLS STREAM variable=E_rnnz_fifo4 depth=FIFO_DEPTH

         hls::stream<int> rnnz_fifo_adj1;
         #pragma HLS STREAM variable=rnnz_fifo_adj1 depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj1_total_e;
         #pragma HLS STREAM variable=rnnz_fifo_adj1_total_e depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj1_total_s;
         #pragma HLS STREAM variable=rnnz_fifo_adj1_total_s depth=FIFO_DEPTH
         hls::stream<ATYPE> A_fifo_adj1("A fifo adj1");
         #pragma HLS STREAM variable=A_fifo_adj1 depth=FIFO_DEPTH
         hls::stream<int>  col_indices_fifo_adj1("col fifo1");
         #pragma HLS STREAM variable=col_indices_fifo_adj1 depth=FIFO_DEPTH

         hls::stream<int> rnnz_fifo_adj2;
         #pragma HLS STREAM variable=rnnz_fifo_adj2 depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj2_total_e;
         #pragma HLS STREAM variable=rnnz_fifo_adj2_total_e depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj2_total_s;
         #pragma HLS STREAM variable=rnnz_fifo_adj2_total_s depth=FIFO_DEPTH
         hls::stream<ATYPE> A_fifo_adj2("A fifo adj2");
         #pragma HLS STREAM variable=A_fifo_adj2 depth=FIFO_DEPTH
         hls::stream<int>  col_indices_fifo_adj2("col fifo2");
         #pragma HLS STREAM variable=col_indices_fifo_adj2 depth=FIFO_DEPTH

         hls::stream<int> rnnz_fifo_adj3;
         #pragma HLS STREAM variable=rnnz_fifo_adj3 depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj3_total_e;
         #pragma HLS STREAM variable=rnnz_fifo_adj3_total_e depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj3_total_s;
         #pragma HLS STREAM variable=rnnz_fifo_adj3_total_s depth=FIFO_DEPTH
         hls::stream<ATYPE> A_fifo_adj3("A fifo adj3");
         #pragma HLS STREAM variable=A_fifo_adj3 depth=FIFO_DEPTH
         hls::stream<int>  col_indices_fifo_adj3("col fifo3");
         #pragma HLS STREAM variable=col_indices_fifo_adj3 depth=FIFO_DEPTH

         hls::stream<int> rnnz_fifo_adj4;
         #pragma HLS STREAM variable=rnnz_fifo_adj4 depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj4_total_e;
         #pragma HLS STREAM variable=rnnz_fifo_adj4_total_e depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj4_total_s;
         #pragma HLS STREAM variable=rnnz_fifo_adj4_total_s depth=FIFO_DEPTH
         hls::stream<ATYPE> A_fifo_adj4("A fifo adj4");
         #pragma HLS STREAM variable=A_fifo_adj4 depth=FIFO_DEPTH
         hls::stream<int>  col_indices_fifo_adj4("col fifo4");
         #pragma HLS STREAM variable=col_indices_fifo_adj4 depth=FIFO_DEPTH

         BTYPE ate_m1[2*C_WIDTH];

         #if (PIPO_BLOCKS>=2)

           LOOP_ATTN : for (int B_index = 0; B_index < layer_loop; B_index++) {
         #else
           int B_index = 0;
         #endif

        std::cout << "attention layer " << B_index << std::endl;

        #pragma HLS DATAFLOW

        #if ADJ_THREADS == 1

          for (int j = 0; j < 2*B_WIDTH_BLOCK; j++) {
                                #pragma HLS PIPELINE
                                BTYPE ate_temp;
                                INTYPE AF = A[j];
                                #if (INT_QUANT == 1)
                                   quantw(ate_temp,AF,quantization_scale_w,f_align,beta_qu,B_index);
                                #else
                                   ate_temp = AF;
                                #endif
                                ate_m1[j] = ate_temp;
         }

        int first_row1;//,first_row2;//,first_row3,first_row4;
        int row_count1;//,row_count2;//,row_count3,row_count4;

        int N_adj_block = N_adj/ADJ_THREADS;
        int N_adj_block_compute = N_adj/FEA_THREADS; // in compute2 each block only contains  N_adj/FEA_THREADS elements
        row_count1 = N_adj_block;
        first_row1 = 0;

       #if GAT_ENABLE == 1

       #else //GAT DISABLE
           std::cout << "Read ADJ data" << std::endl;
           hls::stream<ATYPE> val_att_fifo1_int;
           #if (COO_MODE == 0)
             reada22_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,val_att_fifo1,col_att_fifo1,rnnz_att_fifo1,rowPtr_adj1,columnIndex_adj1,values_adj1);
           #else
             reada22_coo(nnz_adj1,beta_qu,f_align,quantization_scale_adj,model,M_adj,first_row1,row_count1,val_att_fifo1,col_att_fifo1,rnnz_att_fifo1,rowPtr_adj1,columnIndex_adj1,values_adj1,B_index);
           #endif

        #endif

       #endif

         #if (PIPO_BLOCKS>=2)
           }
        #endif

}

void readb(bool load_weights,int beta_qu,int f_align,float quantization_scale_w[5],int M_fea,ap_uint<8> P_w[5],int B_index,BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],INTYPES* B)
{

    int B_shift;
    int M_fea_current;

    if (B_index == 0)
    {
        B_shift = 0;
        M_fea_current = M_fea;
    }
    else
    {
        B_shift = B_WIDTH_BLOCK*M_fea+(B_index-1)*B_WIDTH_BLOCK*B_WIDTH_BLOCK; //shift the weight loading
        M_fea_current = B_WIDTH_BLOCK;
    }

    if(load_weights==1)
      {

         //LOOP_BLOCKB1 : for (int j = 0; j < P_w[B_index]; j++) {
         LOOP_BLOCKB1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                    LOOP_BLOCKB2 : for (int i = 0; i < M_fea_current; i++) {
                            #pragma HLS PIPELINE
		            //if(j < P_w[B_index]) {			
                             INTYPE BF = (INTYPE)B[i+j*M_fea_current+B_shift];
                             BTYPE B_accel_temp;
                             #if (INT_QUANT == 1)
                              quantw(B_accel_temp,BF,quantization_scale_w,f_align,beta_qu,B_index);
                             #else
                              B_accel_temp = BF;
                             #endif

                             B_accel[i][j] = B_accel_temp;
                             //}

                      }
          }

      }
}

void readbl(bool load_weights,int beta_qu,int f_align,float quantization_scale_w[5],int M_fea,ap_uint<8> P_w[5],int B_index,BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],INTYPES* B)
{

    int B_shift;
    int M_fea_current;

    if (B_index == 0)
    {
        B_shift = 0;
        M_fea_current = M_fea;
    }
    else
    {
        B_shift = B_WIDTH_BLOCK*M_fea+(B_index-1)*B_WIDTH_BLOCK*B_WIDTH_BLOCK; //shift the weight loading
        M_fea_current = B_WIDTH_BLOCK;
    }

    if(load_weights==1)
      {

         LOOP_BLOCKB1 : for (int j = 0; j < P_w[B_index]; j++) {
         //LOOP_BLOCKB1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                    LOOP_BLOCKB2 : for (int i = 0; i < M_fea_current; i++) {
                            #pragma HLS PIPELINE
                            //if(j < P_w[B_index]) {	
                             INTYPE BF = (INTYPE)B[i+j*M_fea_current+B_shift];
                             BTYPE B_accel_temp;
                             #if (INT_QUANT == 1)
                              quantw(B_accel_temp,BF,quantization_scale_w,f_align,beta_qu,B_index);
                             #else
                              B_accel_temp = BF;
                             #endif

                             B_accel[i][j] = B_accel_temp;
                            //}
                      }
          }

      }
}

void loop_fea(bool load_weights,int beta_qu,int f_align,int beta_qul,int f_alignl,float quantization_scale_fea[5],float quantization_scale_w[5],float quantization_scale_lin[5],
    ap_uint<1> model[5][8],
    STYPE scale_fea[5],ITYPE* max_fea,int quantized_multiplier,
    int nnz_fea1,int nnz_fea2,int nnz_fea3,int nnz_fea4,
    int *rowPtr_fea1,int *rowPtr_fea2,int *rowPtr_fea3,int *rowPtr_fea4,
    int *columnIndex_fea1,int *columnIndex_fea2,int *columnIndex_fea3,int *columnIndex_fea4,
    INTYPE *values_fea1,INTYPE *values_fea2,INTYPE *values_fea3,INTYPE *values_fea4,
    hls::stream<ASTYPE>&  rowPtr_feas1,hls::stream<ASTYPE>& rowPtr_feas2,hls::stream<ASTYPE>&  rowPtr_feas3,hls::stream<ASTYPE>& rowPtr_feas4,
    hls::stream<ASTYPE>&  columnIndex_feas1,hls::stream<ASTYPE>& columnIndex_feas2,hls::stream<ASTYPE>&  columnIndex_feas3,hls::stream<ASTYPE>& columnIndex_feas4,
    hls::stream<ASTYPE>&  values_feas1,hls::stream<ASTYPE>& values_feas2,hls::stream<ASTYPE>&  values_feas3,hls::stream<ASTYPE>& values_feas4,
    INTYPES* B,INTYPES* B2,
    int N_fea, int M_fea,ap_uint<8> P_w[5],
    ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs,
    #if (PIPO_BLOCKS>=2)
      hls::stream_of_blocks<buf> &C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
    #else
      buf C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
    #endif
    hls::stream_of_blocks<buf> &C_buffer13,hls::stream_of_blocks<buf> &C_buffer14,
    hls::stream_of_blocks<buf> &C_buffer21,hls::stream_of_blocks<buf> &C_buffer22,
    hls::stream_of_blocks<buf> &C_buffer23,hls::stream_of_blocks<buf> &C_buffer24,
    hls::stream_of_blocks<buf> &C_buffer31,hls::stream_of_blocks<buf> &C_buffer32,
    hls::stream_of_blocks<buf> &C_buffer33,hls::stream_of_blocks<buf> &C_buffer34,
    hls::stream_of_blocks<buf> &C_buffer41,hls::stream_of_blocks<buf> &C_buffer42,
    hls::stream_of_blocks<buf> &C_buffer43,hls::stream_of_blocks<buf> &C_buffer44,
    #if (PIPO_BLOCKS>=2)
    hls::stream_of_blocks<buf> &A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
    #else
    buf A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
    #endif
    hls::stream_of_blocks<buf> &A_buffer31,hls::stream_of_blocks<buf> &A_buffer41,
    #if (PIPO_BLOCKS>=2)
     hls::stream_of_blocks<buf> &linear_pipo,
    #else
     buf linear_pipo,
    #endif
    int layer_loop)
{

     BTYPE B_accel1[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel1 block factor= BLOCK/2 dim=2
     BTYPE B_accel2[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel2 block factor= BLOCK/2 dim=2
     BTYPE B_accel3[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel3 block factor= BLOCK/2 dim=2
     BTYPE B_accel4[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel4 block factor= BLOCK/2 dim=2

     BTYPE B_accel12[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel12 block factor= BLOCK/2 dim=2
     BTYPE B_accel22[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel22 block factor= BLOCK/2 dim=2
     BTYPE B_accel32[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel32 block factor= BLOCK/2 dim=2
     BTYPE B_accel42[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel42 block factor= BLOCK/2 dim=2

     hls::stream<int> rnnz_fifo_fea1;
     #pragma HLS STREAM variable=rnnz_fifo_fea1 depth=FIFO_DEPTH
     hls::stream<int> rnnz_fifo_fea2;
     #pragma HLS STREAM variable=rnnz_fifo_fea2 depth=FIFO_DEPTH
     hls::stream<int> rnnz_fifo_fea3;
     #pragma HLS STREAM variable=rnnz_fifo_fea3 depth=FIFO_DEPTH
     hls::stream<int> rnnz_fifo_fea4;
     #pragma HLS STREAM variable=rnnz_fifo_fea4 depth=FIFO_DEPTH

     hls::stream<int> rnnz_fifo_fea12;
     #pragma HLS STREAM variable=rnnz_fifo_fea12 depth=FIFO_DEPTH
     hls::stream<int> rnnz_fifo_fea22;
     #pragma HLS STREAM variable=rnnz_fifo_fea22 depth=FIFO_DEPTH
     hls::stream<int> rnnz_fifo_fea32;
     #pragma HLS STREAM variable=rnnz_fifo_fea32 depth=FIFO_DEPTH
     hls::stream<int> rnnz_fifo_fea42;
     #pragma HLS STREAM variable=rnnz_fifo_fea42 depth=FIFO_DEPTH

     hls::stream<FTYPE> A_fifo_fea1;
     #pragma HLS STREAM variable=A_fifo_fea1 depth=FIFO_DEPTH
     hls::stream<FTYPE> A_fifo_fea2;
     #pragma HLS STREAM variable=A_fifo_fea2 depth=FIFO_DEPTH
     hls::stream<FTYPE> A_fifo_fea3;
     #pragma HLS STREAM variable=A_fifo_fea3 depth=FIFO_DEPTH
     hls::stream<FTYPE> A_fifo_fea4;
     #pragma HLS STREAM variable=A_fifo_fea4 depth=FIFO_DEPTH

     hls::stream<LTYPE> A_fifo_fea12;
     #pragma HLS STREAM variable=A_fifo_fea12 depth=FIFO_DEPTH
     hls::stream<LTYPE> A_fifo_fea22;
     #pragma HLS STREAM variable=A_fifo_fea22 depth=FIFO_DEPTH
     hls::stream<LTYPE> A_fifo_fea32;
     #pragma HLS STREAM variable=A_fifo_fea32 depth=FIFO_DEPTH
     hls::stream<LTYPE> A_fifo_fea42;
     #pragma HLS STREAM variable=A_fifo_fea42 depth=FIFO_DEPTH

     hls::stream<bool> exit_loop;
     #pragma HLS STREAM variable=exit_loop depth=FIFO_DEPTH

     hls::stream<int>  col_indices_fifo_fea1;
     #pragma HLS STREAM variable=col_indices_fifo_fea1 depth=FIFO_DEPTH
     hls::stream<int>  col_indices_fifo_fea2;
     #pragma HLS STREAM variable=col_indices_fifo_fea2 depth=FIFO_DEPTH
     hls::stream<int>  col_indices_fifo_fea3;
     #pragma HLS STREAM variable=col_indices_fifo_fea3 depth=FIFO_DEPTH
     hls::stream<int>  col_indices_fifo_fea4;
     #pragma HLS STREAM variable=col_indices_fifo_fea4 depth=FIFO_DEPTH

     hls::stream<int>  col_indices_fifo_fea12;
     #pragma HLS STREAM variable=col_indices_fifo_fea12 depth=FIFO_DEPTH
     hls::stream<int>  col_indices_fifo_fea22;
     #pragma HLS STREAM variable=col_indices_fifo_fea22 depth=FIFO_DEPTH
     hls::stream<int>  col_indices_fifo_fea32;
     #pragma HLS STREAM variable=col_indices_fifo_fea32 depth=FIFO_DEPTH
     hls::stream<int>  col_indices_fifo_fea42;
     #pragma HLS STREAM variable=col_indices_fifo_fea42 depth=FIFO_DEPTH

     int B_WIDTH_INT;

    #if (PIPO_BLOCKS>=2)
     LOOP_FEA : for (int B_index = 0; B_index < layer_loop; B_index++) {
    #else
      int B_index = 0;
     #endif
        #pragma HLS DATAFLOW

        B_WIDTH_INT = B_WIDTH_BLOCK;

        std::cout << "fea layer " << B_index << std::endl;

        #if FEA_THREADS == 1

             std::cout << "load weights " << std::endl;

             #if LINEAR_ONLY == 0

             readb(load_weights,beta_qu,f_align,quantization_scale_w,M_fea,P_w,B_index,B_accel1,B); //gnn weights

             #endif

             #if GNN_ONLY == 0

             readbl(load_weights,beta_qu,f_align,quantization_scale_w,M_fea,P_w,B_index,B_accel12,B2); //linear weights

             #endif

             #if (PIPO_BLOCKS>=2)
                 hls::write_lock<buf> C_fea11(C_buffer11); // one output for the ADJ_LOOP and one for attention
                 hls::write_lock<buf> linear_fea(linear_pipo); //

             #if GAT_ENABLE == 1
                 hls::write_lock<buf> A_fea11(A_buffer11); //we write the same output to two buffers
              #else
                 QTYPE A_fea11[B_HEIGHT][B_WIDTH_BLOCK];
             #endif
             #else
            #if GAT_ENABLE == 1
            #else
               QTYPE A_fea11[B_HEIGHT][B_WIDTH_BLOCK];
            #endif

            #endif

                  int first_row1,first_row2,first_row3,first_row4;
                  int row_count1,row_count2,row_count3,row_count4;

                  int N_fea_block = N_fea;
                  int N_fea_rest = 0;
                  row_count1 = N_fea_block;
                  first_row1 = 0;

                  int last_index1;

                  #if (COO_MODE == 0)
                    reada1_csr(beta_qu,f_align,quantization_scale_fea,last_index1,stream_mode_int,gemm_mode_int,M_fea_int,first_row1,row_count1,A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,
                    rowPtr_fea1,columnIndex_fea1,values_fea1,values_feas1);
                  #else
                    reada1_coo(nnz_fea1,beta_qu,f_align,beta_qul,f_alignl,quantization_scale_fea,quantization_scale_lin,last_index1,model,M_fea,first_row1,row_count1,
                            A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,
                            A_fifo_fea12,col_indices_fifo_fea12,rnnz_fifo_fea12,
                     rowPtr_fea1,columnIndex_fea1,values_fea1,
                     rowPtr_feas1,columnIndex_feas1,values_feas1,
                     B_index,layer_loop);
                   #endif

                std::cout << "COMPUTE1 " << std::endl;

              ITYPE max_fea1,max_fea2;

             #if (PIPO_BLOCKS>=2)
              #if LINEAR_ONLY == 0
              compute1_1(scale_fea,&max_fea1,quantized_multiplier,model,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_fea11,
              A_fea11,B_index);
              #endif
              #if GNN_ONLY == 0
              compute1_12(scale_fea,&max_fea2,quantized_multiplier,model,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea12, col_indices_fifo_fea12, rnnz_fifo_fea12,B_accel12,linear_fea,B_index);
              #endif
             #else
              #if LINEAR_ONLY == 0
              compute1_1(scale_fea,&max_fea1,quantized_multiplier,gemm_mode_int,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_buffer11,
              A_buffer11);
              #endif
              #if GNN_ONLY == 0
              compute1_12(scale_fea,&max_fea2,quantized_multiplier,gemm_mode_int,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea12, col_indices_fifo_fea12, rnnz_fifo_fea12,B_accel12,linear_pipo);
              #endif
              #endif
              *max_fea = max_fea1;

        #endif

       #if (PIPO_BLOCKS>=2)
           }
        #endif

}

void loop_adj(float deq_factor[5],ap_uint<1> model[5][8],float srelu[5],hls::stream<ITYPE> &A_fifo_adj1,hls::stream<int> &col_indices_fifo_adj1,hls::stream<int> &rnnz_fifo_adj1,
    hls::stream<TTYPE> &A_fifo_adj2,hls::stream<int> &col_indices_fifo_adj2,hls::stream<int> &rnnz_fifo_adj2,
    hls::stream<TTYPE> &A_fifo_adj3,hls::stream<int> &col_indices_fifo_adj3,hls::stream<int> &rnnz_fifo_adj3,
    hls::stream<TTYPE> &A_fifo_adj4,hls::stream<int> &col_indices_fifo_adj4,hls::stream<int> &rnnz_fifo_adj4,
    int N_adj, int M_adj,ap_uint<8> P_w[5], ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs,
    #if (PIPO_BLOCKS>=2)
      hls::stream_of_blocks<buf> &C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
    #else
      buf C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
    #endif
    hls::stream_of_blocks<buf> &C_buffer13,hls::stream_of_blocks<buf> &C_buffer14,
    hls::stream_of_blocks<buf> &C_buffer21,hls::stream_of_blocks<buf> &C_buffer22,
    hls::stream_of_blocks<buf> &C_buffer23,hls::stream_of_blocks<buf> &C_buffer24,
    hls::stream_of_blocks<buf> &C_buffer31,hls::stream_of_blocks<buf> &C_buffer32,
    hls::stream_of_blocks<buf> &C_buffer33,hls::stream_of_blocks<buf> &C_buffer34,
    hls::stream_of_blocks<buf> &C_buffer41,hls::stream_of_blocks<buf> &C_buffer42,
    hls::stream_of_blocks<buf> &C_buffer43,hls::stream_of_blocks<buf> &C_buffer44,
    #if (PIPO_BLOCKS>=2)
     hls::stream_of_blocks<buf> &linear_pipo,
    #else
     buf linear_pipo,
    #endif
    int layer_loop,OUTTYPE* D1,OUTTYPE* D2,OUTTYPE* D3,OUTTYPE* D4,hls::stream<ASTYPE>& DS1,hls::stream<ASTYPE>& DS1R, hls::stream<ASTYPE>& DS1C,
    hls::stream<ASTYPE>&  DS2, hls::stream<ASTYPE>& DS3,hls::stream<ASTYPE>&  DS4)
{

       hls::stream<ITYPE>       D_fifo1[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=D_fifo1 depth=FIFO_DEPTH
       hls::stream<ITYPE>       D_fifo2[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=D_fifo2 depth=FIFO_DEPTH
       hls::stream<ITYPE>       D_fifo3[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=D_fifo3 depth=FIFO_DEPTH
       hls::stream<ITYPE>       D_fifo4[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=D_fifo4 depth=FIFO_DEPTH

       hls::stream<OUTTYPE>   out_fifo1;
       #pragma HLS STREAM variable=out_fifo1 depth=FIFO_DEPTH

       hls::stream<ITYPE>       write_fifo1[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=write_fifo1 depth=FIFO_DEPTH
       hls::stream<ITYPE>       write_fifo2[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=write_fifo2 depth=FIFO_DEPTH
       hls::stream<ITYPE>       write_fifo3[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=write_fifo3 depth=FIFO_DEPTH
       hls::stream<ITYPE>       write_fifo4[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=write_fifo4 depth=FIFO_DEPTH

    #if (PIPO_BLOCKS>=2)

       LOOP_ADJ : for (int B_index = 0; B_index < layer_loop; B_index++) {

    #else

           int B_index = 0;

    #endif

        #pragma HLS DATAFLOW

              std::cout << "adj layer " << B_index << std::endl;

    #if ADJ_THREADS == 1

         #if (PIPO_BLOCKS>=2)
             hls::read_lock<buf> C_adj11(C_buffer11);
             hls::read_lock<buf> linear_adj(linear_pipo);
         #endif
                    
                int first_row1;//,first_row2;//,first_row3,first_row4;
                int row_count1;//,row_count2;//,row_count3,row_count4;

                 int N_adj_block = N_adj/ADJ_THREADS;
                 int N_adj_block_compute = N_adj/FEA_THREADS; // in compute2 each block only contains  N_adj/FEA_THREADS elements
                row_count1 = N_adj_block;
                first_row1 = 0;

                #if FEA_THREADS == 1
                   #if(PIPO_BLOCKS>=2)
                    #if LINEAR_ONLY == 0
                     compute2_1(model,srelu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_adj11,
                        D_fifo1,B_index);
                    #endif
                   #else
                     #if LINEAR_ONLY == 0
                     compute2_1(relu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_buffer11,
                            D_fifo1);
                       #endif
                   #endif
                #endif

                      #if(PIPO_BLOCKS>=2)
                       writec(deq_factor,model,first_row1,row_count1,N_adj,P_w, D_fifo1,linear_adj,out_fifo1,B_index,layer_loop);
                      #else
                       writec(deq_factor,model,first_row1,row_count1,N_adj,P_w, D_fifo1,linear_pipo,D1,DS1,B_index,layer_loop);
                      #endif

                      writeout(model,first_row1,row_count1,N_adj,P_w, out_fifo1,D1,DS1,DS1R, DS1C,B_index,layer_loop);

    #endif

     #if (PIPO_BLOCKS>=2)

         }

     #endif

}

void loop_adj2(
int nnz_adj1,int nnz_adj2,int nnz_adj3,int nnz_adj4,
int beta_qu,int f_align,float quantization_scale_adj,float quantization_scale_w[5],
float deq_factor[5],
ap_uint<1> model[5][8],float srelu[5],
int * rowPtr_adj1,int * rowPtr_adj2,int * rowPtr_adj3,int * rowPtr_adj4,
int *columnIndex_adj1, int *columnIndex_adj2, int *columnIndex_adj3, int *columnIndex_adj4,
INTYPE *values_adj1,    INTYPE *values_adj2,    INTYPE *values_adj3,    INTYPE *values_adj4,
int N_adj, int M_adj, ap_uint<8> P_w[5], ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs,
INTYPE *A,
#if (PIPO_BLOCKS>=2)
hls::stream_of_blocks<buf> &A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
#else
buf A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
#endif
hls::stream_of_blocks<buf> &A_buffer31,hls::stream_of_blocks<buf> &A_buffer41,
OUTTYPE* E1,
OUTTYPE* S1,
#if (PIPO_BLOCKS>=2)
hls::stream_of_blocks<buf> &C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
#else
buf C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
#endif
hls::stream_of_blocks<buf> &C_buffer13,hls::stream_of_blocks<buf> &C_buffer14,
hls::stream_of_blocks<buf> &C_buffer21,hls::stream_of_blocks<buf> &C_buffer22,
hls::stream_of_blocks<buf> &C_buffer23,hls::stream_of_blocks<buf> &C_buffer24,
hls::stream_of_blocks<buf> &C_buffer31,hls::stream_of_blocks<buf> &C_buffer32,
hls::stream_of_blocks<buf> &C_buffer33,hls::stream_of_blocks<buf> &C_buffer34,
hls::stream_of_blocks<buf> &C_buffer41,hls::stream_of_blocks<buf> &C_buffer42,
hls::stream_of_blocks<buf> &C_buffer43,hls::stream_of_blocks<buf> &C_buffer44,
#if (PIPO_BLOCKS>=2)
 hls::stream_of_blocks<buf> &linear_pipo,
#else
 buf linear_pipo,
#endif
int layer_loop,OUTTYPE* D1,OUTTYPE* D2,OUTTYPE* D3,OUTTYPE* D4,hls::stream<ASTYPE>& DS1,
hls::stream<ASTYPE>& DS1R, hls::stream<ASTYPE>& DS1C,
hls::stream<ASTYPE>&  DS2, hls::stream<ASTYPE>& DS3,hls::stream<ASTYPE>&  DS4)
{

    hls::stream<int>  rnnz_att1("rnnz_att1 stream");
    #pragma HLS STREAM variable= rnnz_att1 depth=FIFO_DEPTH
    hls::stream<ITYPE>  values_att1("values_att1 stream");
    #pragma HLS STREAM variable= values_att1 depth=FIFO_DEPTH
    hls::stream<int>  columnIndex_att1("columnIndex_att1 stream");
    #pragma HLS STREAM variable= columnIndex_att1 depth=FIFO_DEPTH

    hls::stream<int>  rnnz_att2;
    #pragma HLS STREAM variable= rnnz_att2 depth=FIFO_DEPTH
    hls::stream<ITYPE>  values_att2;
    #pragma HLS STREAM variable= values_att2 depth=FIFO_DEPTH
    hls::stream<int>  columnIndex_att2;
    #pragma HLS STREAM variable= columnIndex_att2 depth=FIFO_DEPTH

    hls::stream<int>  rnnz_att3;
    #pragma HLS STREAM variable= rnnz_att3 depth=FIFO_DEPTH
    hls::stream<ITYPE>  values_att3;
    #pragma HLS STREAM variable= values_att3 depth=FIFO_DEPTH
    hls::stream<int>  columnIndex_att3;
    #pragma HLS STREAM variable= columnIndex_att3 depth=FIFO_DEPTH

    hls::stream<int>  rnnz_att4;
    #pragma HLS STREAM variable= rnnz_att4 depth=FIFO_DEPTH
    hls::stream<ITYPE>  values_att4;
    #pragma HLS STREAM variable= values_att4 depth=FIFO_DEPTH
    hls::stream<int>  columnIndex_att4;
    #pragma HLS STREAM variable= columnIndex_att4 depth=FIFO_DEPTH

   #pragma HLS DATAFLOW

    loop_attention(deq_factor,beta_qu,f_align,quantization_scale_adj,quantization_scale_w,
    model,
    nnz_adj1,nnz_adj2,nnz_adj3,nnz_adj4,
    rowPtr_adj1,rowPtr_adj2,rowPtr_adj3,rowPtr_adj4,
    columnIndex_adj1,columnIndex_adj2,columnIndex_adj3,columnIndex_adj4,
    values_adj1,values_adj2,values_adj3,values_adj4,
    N_adj,M_adj,P_w,A,A_buffer11,A_buffer21,A_buffer31,A_buffer41,
    E1,
    S1,
    rnnz_att1,columnIndex_att1,values_att1,
    rnnz_att2,columnIndex_att2,values_att2,
    rnnz_att3,columnIndex_att3,values_att3,
    rnnz_att4,columnIndex_att4,values_att4,
    layer_loop);

    std::cout << "Done loop attention" << std::endl;

    loop_adj(deq_factor,model,srelu,
    values_att1,columnIndex_att1,rnnz_att1,
    values_att2,columnIndex_att2,rnnz_att2,
    values_att3,columnIndex_att3,rnnz_att3,
    values_att4,columnIndex_att4,rnnz_att4,
    N_adj, M_adj,P_w,zero_point_lhs,zero_point_rhs,
    C_buffer11,C_buffer12,C_buffer13,C_buffer14,
    C_buffer21,C_buffer22,C_buffer23,C_buffer24,
    C_buffer31,C_buffer32,C_buffer33,C_buffer34,
    C_buffer41,C_buffer42,C_buffer43,C_buffer44,
        linear_pipo,
    layer_loop,D1,D2,D3,D4,DS1,DS1R,DS1C,DS2,DS3,DS4);

      std::cout << "Done loop adj" << std::endl;

}

void mmult_wrapper(bool load_weights,int beta_qu,int f_align,int beta_qul,int f_alignl,float quantization_scale_adj,float quantization_scale_fea[5],float quantization_scale_w[5],float quantization_scale_lin[5],
    float deq_factor[5],
    ap_uint<1> model[5][8],float srelu[5],
    STYPE scale_fea[5],ITYPE* max_fea,int quantized_multiplier, ap_int<32> *shift, ap_int<32> *bias,
    ap_int<32> bias_count, ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs,
    ap_int<8> zero_point_dst, ap_int<8> clamp_max,ap_int<8> clamp_min,int N_adj, int M_adj, int M_fea,
    ap_uint<8> P_w[5], INTYPES* B, INTYPES* B2,
     OUTTYPE* D1, OUTTYPE* D2, OUTTYPE* D3,OUTTYPE* D4,
     hls::stream<ASTYPE>& DS1, hls::stream<ASTYPE>& DS1R, hls::stream<ASTYPE>& DS1C,
     hls::stream<ASTYPE>&  DS2, hls::stream<ASTYPE>& DS3,hls::stream<ASTYPE>&  DS4,
     OUTTYPE* E1,
     OUTTYPE* S1,
     INTYPE *ate_m,
     int array_c_adjust, ap_int<32>  layer_loop,
     int nnz_fea1,int nnz_fea2,int nnz_fea3,int nnz_fea4,
     int *rowPtr_fea1,int *rowPtr_fea2,int *rowPtr_fea3,int *rowPtr_fea4,
     int *columnIndex_fea1, int *columnIndex_fea2, int *columnIndex_fea3, int *columnIndex_fea4,
     INTYPE *values_fea1,INTYPE *values_fea2,INTYPE *values_fea3,INTYPE *values_fea4,
     hls::stream<ASTYPE>&  rowPtr_feas1,hls::stream<ASTYPE>& rowPtr_feas2,hls::stream<ASTYPE>&  rowPtr_feas3,hls::stream<ASTYPE>& rowPtr_feas4,
     hls::stream<ASTYPE>&  columnIndex_feas1,hls::stream<ASTYPE>& columnIndex_feas2,hls::stream<ASTYPE>&  columnIndex_feas3,hls::stream<ASTYPE>& columnIndex_feas4,
     hls::stream<ASTYPE>&  values_feas1,hls::stream<ASTYPE>& values_feas2,hls::stream<ASTYPE>&  values_feas3,hls::stream<ASTYPE>& values_feas4,
     int nnz_adj1, int nnz_adj2, int nnz_adj3, int nnz_adj4,
     int *rowPtr_adj1,int *rowPtr_adj2,int *rowPtr_adj3,int *rowPtr_adj4,
     int *columnIndex_adj1,int *columnIndex_adj2,int *columnIndex_adj3,int *columnIndex_adj4,
     INTYPE *values_adj1,INTYPE *values_adj2,INTYPE *values_adj3,INTYPE *values_adj4)
{

      #if (PIPO_BLOCKS>=2)
        hls::stream_of_blocks<buf,PIPO_BLOCKS> linear_pipo;
      #else
        buf linear_pipo;
      #endif
      #pragma HLS array_partition variable=linear_pipo block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=linear_pipo cyclic factor= SBLOCK dim=1

      #if (PIPO_BLOCKS>=2)
       hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer11;
      #else
       buf C_buffer11;
      #endif
      #pragma HLS array_partition variable=C_buffer11 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer11 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer12;
      #pragma HLS array_partition variable=C_buffer12 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer12 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer13;
      #pragma HLS array_partition variable=C_buffer13 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer13 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer14;
      #pragma HLS array_partition variable=C_buffer14 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer14 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer21;
      #pragma HLS array_partition variable=C_buffer21 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer21 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer22;
      #pragma HLS array_partition variable=C_buffer22 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer22 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer23;
      #pragma HLS array_partition variable=C_buffer23 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer23 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer24;
      #pragma HLS array_partition variable=C_buffer24 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer24 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer31;
      #pragma HLS array_partition variable=C_buffer31 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer31 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer32;
      #pragma HLS array_partition variable=C_buffer32 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer32 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer33;
      #pragma HLS array_partition variable=C_buffer33 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer33 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer34;
      #pragma HLS array_partition variable=C_buffer34 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer34 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer41;
      #pragma HLS array_partition variable=C_buffer41 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer41 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer42;
      #pragma HLS array_partition variable=C_buffer42 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer42 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer43;
      #pragma HLS array_partition variable=C_buffer43 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer43 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer44;
      #pragma HLS array_partition variable=C_buffer44 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer44 cyclic factor= SBLOCK dim=1

      #if (PIPO_BLOCKS>=2)
      hls::stream_of_blocks<buf,PIPO_BLOCKS> A_buffer11;
      #else
      buf A_buffer11;
      #endif

      #pragma HLS array_partition variable=A_buffer11 block factor= BLOCK/2 dim=2

      hls::stream_of_blocks<buf,PIPO_BLOCKS> A_buffer21;
      #pragma HLS array_partition variable=A_buffer21 block factor= BLOCK/2 dim=2

      hls::stream_of_blocks<buf,PIPO_BLOCKS> A_buffer31;
      #pragma HLS array_partition variable=A_buffer31 block factor= BLOCK/2 dim=2

      hls::stream_of_blocks<buf,PIPO_BLOCKS> A_buffer41;
      #pragma HLS array_partition variable=A_buffer41 block factor= BLOCK/2 dim=2

      int B_WIDTH_INT,a_values;

      #if (PIPO_BLOCKS>=2)
        #pragma HLS DATAFLOW
      #endif

      loop_fea(load_weights,beta_qu,f_align,beta_qul,f_alignl,quantization_scale_fea,quantization_scale_w,quantization_scale_lin,
      model,
      scale_fea,max_fea,quantized_multiplier,
      nnz_fea1,nnz_fea2,nnz_fea3,nnz_fea4,
      rowPtr_fea1,rowPtr_fea2,rowPtr_fea3,rowPtr_fea4,
      columnIndex_fea1,columnIndex_fea2,columnIndex_fea3,columnIndex_fea4,
      values_fea1,values_fea2,values_fea3,values_fea4,
      rowPtr_feas1,rowPtr_feas2,rowPtr_feas3,rowPtr_feas4,
      columnIndex_feas1,columnIndex_feas2,columnIndex_feas3,columnIndex_feas4,
      values_feas1,values_feas2,values_feas3,values_feas4,
      B,B2,
      M_adj, M_fea, P_w,
      zero_point_lhs, zero_point_rhs,
      C_buffer11,C_buffer12,C_buffer13,C_buffer14,
      C_buffer21,C_buffer22,C_buffer23,C_buffer24,
      C_buffer31,C_buffer32,C_buffer33,C_buffer34,
      C_buffer41,C_buffer42,C_buffer43,C_buffer44,
      A_buffer11,A_buffer21,A_buffer31,A_buffer41,
      linear_pipo,
      layer_loop);

      std::cout << "Done loop fea" << std::endl;

      loop_adj2(nnz_adj1,nnz_adj2,nnz_adj3,nnz_adj4,
      beta_qu,f_align,quantization_scale_adj,quantization_scale_w,
      deq_factor,
      model,srelu,
      rowPtr_adj1,rowPtr_adj2,rowPtr_adj3,rowPtr_adj4,
      columnIndex_adj1,columnIndex_adj2,columnIndex_adj3,columnIndex_adj4,
      values_adj1,values_adj2,values_adj3,values_adj4,
      N_adj,M_adj, P_w,zero_point_lhs,zero_point_rhs,
      ate_m,
      A_buffer11,A_buffer21,A_buffer31,A_buffer41,
      E1,S1,
      C_buffer11,C_buffer12,C_buffer13,C_buffer14,
      C_buffer21,C_buffer22,C_buffer23,C_buffer24,
      C_buffer31,C_buffer32,C_buffer33,C_buffer34,
      C_buffer41,C_buffer42,C_buffer43,C_buffer44,
      linear_pipo,
      layer_loop,D1,D2,D3,D4,DS1,DS1R,DS1C,DS2,DS3,DS4);

}

typedef unsigned long u32;

void mmult_top(bool load_weights,int beta_qu,int f_align,int beta_qul,int f_alignl,float quantization_scale_adj,
float quantization_scale_fea[5],float quantization_scale_w[5],float quantization_scale_lin[5],
float deq_factor[5],
ap_uint<8> model[5],float srelu[5],
STYPE scale_fea[5], ITYPE* max_fea,ap_int<32>  layer_count,int quantized_multiplier, ap_int<32> *shift, ap_int<32> *bias,
ap_int<32> bias_count,ap_int<64> *profiling, ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs,
ap_int<8> zero_point_dst,
ap_int<8> clamp_max,ap_int<8> clamp_min,int N_adj, int M_adj, int M_fea, ap_uint<8> P_w[5],
INTYPES* B,INTYPES* B2,
OUTTYPE* D1, OUTTYPE* D2, OUTTYPE* D3,OUTTYPE* D4,
hls::stream<ASTYPE>& DS1, hls::stream<ASTYPE>& DS1R,hls::stream<ASTYPE>& DS1C,
hls::stream<ASTYPE>&  DS2, hls::stream<ASTYPE>& DS3,hls::stream<ASTYPE>&  DS4,
OUTTYPE* E1,
OUTTYPE* S1,
INTYPE *ate_m,
int array_c_adjust,
int nnz_fea1,int nnz_fea2,int nnz_fea3,int nnz_fea4,
int *rowPtr_fea1,int *rowPtr_fea2,int *rowPtr_fea3,int *rowPtr_fea4,
int *columnIndex_fea1, int *columnIndex_fea2, int *columnIndex_fea3, int *columnIndex_fea4,
INTYPE *values_fea1,INTYPE *values_fea2,INTYPE *values_fea3,INTYPE *values_fea4,
hls::stream<ASTYPE>&  rowPtr_feas1,hls::stream<ASTYPE>& rowPtr_feas2,hls::stream<ASTYPE>&  rowPtr_feas3,hls::stream<ASTYPE>& rowPtr_feas4,
hls::stream<ASTYPE>&  columnIndex_feas1,hls::stream<ASTYPE>& columnIndex_feas2,hls::stream<ASTYPE>&  columnIndex_feas3,hls::stream<ASTYPE>& columnIndex_feas4,
hls::stream<ASTYPE>&  values_feas1,hls::stream<ASTYPE>& values_feas2,hls::stream<ASTYPE>&  values_feas3,hls::stream<ASTYPE>& values_feas4,
int nnz_adj1,int nnz_adj2,int nnz_adj3,int nnz_adj4,
int *rowPtr_adj1,int *rowPtr_adj2,int *rowPtr_adj3,int *rowPtr_adj4,
int *columnIndex_adj1,int *columnIndex_adj2,int *columnIndex_adj3,int *columnIndex_adj4,
INTYPE *values_adj1,INTYPE *values_adj2,INTYPE *values_adj3,INTYPE *values_adj4)
{

     #pragma HLS INTERFACE s_axilite port = return bundle = control
     #pragma HLS INTERFACE s_axilite port = load_weights bundle = control
     #pragma HLS INTERFACE s_axilite port = beta_qu bundle = control
     #pragma HLS INTERFACE s_axilite port = f_align bundle = control
     #pragma HLS INTERFACE s_axilite port = beta_qul bundle = control
     #pragma HLS INTERFACE s_axilite port = f_alignl bundle = control
     #pragma HLS INTERFACE s_axilite port = deq_factor bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_fea1 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_fea2 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_fea3 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_fea4 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_adj1 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_adj2 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_adj3 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_adj4 bundle = control
     #pragma HLS INTERFACE s_axilite port = quantization_scale_adj bundle = control
     #pragma HLS INTERFACE s_axilite port = quantization_scale_fea bundle = control
     #pragma HLS INTERFACE s_axilite port = quantization_scale_w bundle = control
     #pragma HLS INTERFACE s_axilite port = quantization_scale_lin bundle = control
     #pragma HLS INTERFACE s_axilite port = bias_count bundle = control
     #pragma HLS INTERFACE s_axilite port = zero_point_lhs bundle = control
     #pragma HLS INTERFACE s_axilite port = zero_point_rhs bundle = control
     #pragma HLS INTERFACE s_axilite port = zero_point_dst bundle = control
     #pragma HLS INTERFACE s_axilite port = clamp_max bundle = control
     #pragma HLS INTERFACE s_axilite port = clamp_min bundle = control
     #pragma HLS INTERFACE s_axilite port = N_adj bundle = control
     #pragma HLS INTERFACE s_axilite port = M_adj bundle = control
     #pragma HLS INTERFACE s_axilite port = M_fea bundle = control
     #pragma HLS INTERFACE s_axilite port = P_w bundle = control
     #pragma HLS INTERFACE s_axilite port = array_c_adjust bundle = control
     #pragma HLS INTERFACE s_axilite port = model bundle = control

     #pragma HLS INTERFACE s_axilite port = layer_count bundle = control
     #pragma HLS INTERFACE s_axilite port = quantized_multiplier bundle = control

     #pragma HLS INTERFACE axis port = DS1 depth=64000
     #pragma HLS INTERFACE axis port = DS1R depth=64000
     #pragma HLS INTERFACE axis port = DS1C depth=64000
     #pragma HLS INTERFACE axis port = DS2 depth=4096
     #pragma HLS INTERFACE axis port = DS3 depth=4096
     #pragma HLS INTERFACE axis port = DS4 depth=4096

     #pragma HLS INTERFACE axis port = columnIndex_feas1 depth=4096
     #pragma HLS INTERFACE axis port = columnIndex_feas2 depth=4096
     #pragma HLS INTERFACE axis port = columnIndex_feas3 depth=4096
     #pragma HLS INTERFACE axis port = columnIndex_feas4 depth=4096

     #pragma HLS INTERFACE axis port = rowPtr_feas1 depth=4096
     #pragma HLS INTERFACE axis port = rowPtr_feas2 depth=4096
     #pragma HLS INTERFACE axis port = rowPtr_feas3 depth=4096
     #pragma HLS INTERFACE axis port = rowPtr_feas4 depth=4096

     #pragma HLS INTERFACE axis port = values_feas1 depth=64000
     #pragma HLS INTERFACE axis port = values_feas2 depth=4096
     #pragma HLS INTERFACE axis port = values_feas3 depth=4096
     #pragma HLS INTERFACE axis port = values_feas4 depth=4096

     #pragma HLS INTERFACE m_axi port = profiling depth=16 offset=slave bundle = profiling
     #pragma HLS INTERFACE m_axi port=rowPtr_fea1 depth=64000 offset=slave bundle = rowPtr_fea1
     #pragma HLS INTERFACE m_axi port=rowPtr_fea2 depth=4096 offset=slave bundle = rowPtr_fea2
     #pragma HLS INTERFACE m_axi port=rowPtr_fea3 depth=4096 offset=slave bundle = rowPtr_fea3
     #pragma HLS INTERFACE m_axi port=rowPtr_fea4 depth=4096 offset=slave bundle = rowPtr_fea4
     #pragma HLS INTERFACE m_axi port=columnIndex_fea1 depth=64000 offset=slave bundle = columnIndex_fea1
     #pragma HLS INTERFACE m_axi port=columnIndex_fea2 depth=4096 offset=slave bundle = columnIndex_fea2
     #pragma HLS INTERFACE m_axi port=columnIndex_fea3 depth=4096 offset=slave bundle = columnIndex_fea3
     #pragma HLS INTERFACE m_axi port=columnIndex_fea4 depth=4096 offset=slave bundle = columnIndex_fea4
     #pragma HLS INTERFACE m_axi port=values_fea1 depth=64000 offset=slave bundle = values_fea1
     #pragma HLS INTERFACE m_axi port=values_fea2 depth=4096 offset=slave bundle = values_fea2
     #pragma HLS INTERFACE m_axi port=values_fea3 depth=4096 offset=slave bundle = values_fea3
     #pragma HLS INTERFACE m_axi port=values_fea4 depth=4096 offset=slave bundle = values_fea4
     #pragma HLS INTERFACE m_axi port=rowPtr_adj1 depth=64000 offset=slave bundle = rowPtr_adj1
     #pragma HLS INTERFACE m_axi port=rowPtr_adj2 depth=4096 offset=slave bundle = rowPtr_adj2
     #pragma HLS INTERFACE m_axi port=rowPtr_adj3 depth=4096 offset=slave bundle = rowPtr_adj3
     #pragma HLS INTERFACE m_axi port=rowPtr_adj4 depth=4096 offset=slave bundle = rowPtr_adj4
     #pragma HLS INTERFACE m_axi port=columnIndex_adj1 depth=64000 offset=slave bundle = columnIndex_adj1
     #pragma HLS INTERFACE m_axi port=columnIndex_adj2 depth=4096 offset=slave bundle = columnIndex_adj2
     #pragma HLS INTERFACE m_axi port=columnIndex_adj3 depth=4096 offset=slave bundle = columnIndex_adj3
     #pragma HLS INTERFACE m_axi port=columnIndex_adj4 depth=4096 offset=slave bundle = columnIndex_adj4
     #pragma HLS INTERFACE m_axi port=values_adj1 depth=64000 offset=slave bundle = values_adj1
     #pragma HLS INTERFACE m_axi port=values_adj2 depth=4096 offset=slave bundle = values_adj2
     #pragma HLS INTERFACE m_axi port=values_adj3 depth=4096 offset=slave bundle = values_adj3
     #pragma HLS INTERFACE m_axi port=values_adj4 depth=4096 offset=slave bundle = values_adj4
     #pragma HLS INTERFACE m_axi port=B depth=32000 offset=slave bundle=B
     #pragma HLS INTERFACE m_axi port=B2 depth=32000 offset=slave bundle=B2
     #pragma HLS INTERFACE m_axi port=D1 depth=64000 offset=slave  bundle=D1
     #pragma HLS INTERFACE m_axi port=D2 depth=1000 offset=slave bundle=D2
     #pragma HLS INTERFACE m_axi port=D3 depth=1000 offset=slave bundle=D3
     #pragma HLS INTERFACE m_axi port=D4 depth=1000 offset=slave bundle=D4
     #pragma HLS INTERFACE m_axi port=E1 depth=64000 offset=slave bundle=E1
     #pragma HLS INTERFACE m_axi port=S1 depth=64000 offset=slave bundle=S1
     #pragma HLS INTERFACE m_axi port=ate_m depth=1000 offset=slave bundle=ate_m
     #pragma HLS INTERFACE m_axi port=shift offset=slave depth=1024 bundle=shift
     #pragma HLS INTERFACE m_axi port=bias offset=slave depth=1024 bundle=bias
     #pragma HLS INTERFACE m_axi port=model offset=slave depth=1024 bundle=model
     #pragma HLS INTERFACE m_axi port=quantization_scale_fea offset=slave bundle=quantization_scale_fea
     #pragma HLS INTERFACE m_axi port=quantization_scale_w offset=slave bundle=quantization_scale_w
     #pragma HLS INTERFACE m_axi port=quantization_scale_lin offset=slave bundle=quantization_scale_lin
     #pragma HLS INTERFACE m_axi port=deq_factor offset=slave bundle=deq_factor
     #pragma HLS INTERFACE m_axi port=scale_fea offset=slave bundle=scale_fea
     #pragma HLS INTERFACE m_axi port=P_w offset=slave bundle=P_w
     #pragma HLS INTERFACE m_axi port=srelu offset=slave bundle=srelu

     #pragma HLS INTERFACE s_axilite port=columnIndex_fea1 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_fea2 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_fea3 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_fea4 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_fea1 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_fea2 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_fea3 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_fea4 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_adj1 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_adj2 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_adj3 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_adj4 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_adj1 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_adj2 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_adj3 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_adj4 bundle = control
     #pragma HLS INTERFACE s_axilite port=values_adj1  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_adj2  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_adj3  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_adj4  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_fea1  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_fea2  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_fea3  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_fea4  bundle = control
     #pragma HLS INTERFACE s_axilite port=B  bundle = control
     #pragma HLS INTERFACE s_axilite port=B2  bundle = control
     #pragma HLS INTERFACE s_axilite port=D1  bundle = control
     #pragma HLS INTERFACE s_axilite port=D2  bundle = control
     #pragma HLS INTERFACE s_axilite port=D3  bundle = control
     #pragma HLS INTERFACE s_axilite port=D4  bundle = control
     #pragma HLS INTERFACE s_axilite port=E1  bundle = control
     #pragma HLS INTERFACE s_axilite port=S1  bundle = control
     #pragma HLS INTERFACE s_axilite port=profiling  bundle = control
     #pragma HLS INTERFACE s_axilite port=quantized_multiplier  bundle = control
     #pragma HLS INTERFACE s_axilite port=shift  bundle = control
     #pragma HLS INTERFACE s_axilite port=bias  bundle = control
     #pragma HLS INTERFACE s_axilite port=ate_m  bundle = control
     #pragma HLS INTERFACE s_axilite port=scale_fea  bundle = control
     #pragma HLS INTERFACE s_axilite port=max_fea  bundle = control
     #pragma HLS INTERFACE s_axilite port=srelu  bundle = control

     ap_int<32> bias_data[1024];
     ap_int<32> shift_data[1024];

     float srelu_int[5];

     ap_uint<8> P_w_int[5];
     #pragma HLS ARRAY_PARTITION variable=P_w_int complete

     ap_uint<1> model_int[5][8];
     #pragma HLS ARRAY_PARTITION variable=model_int complete

     float quantization_scale_lin_int[5];
     #pragma HLS ARRAY_PARTITION variable=quantization_scale_lin_int complete

     float quantization_scale_w_int[5];
     #pragma HLS ARRAY_PARTITION variable=quantization_scale_w_int complete

     float quantization_scale_fea_int[5];
     #pragma HLS ARRAY_PARTITION variable=quantization_scale_fea_int complete

     float deq_factor_int[5];
     #pragma HLS ARRAY_PARTITION variable=deq_factor_int complete

     STYPE scale_fea_int[5];
     #pragma HLS ARRAY_PARTITION variable=scale_fea_int complete

     {

     fifo_empty_0 = 0;
     fifo_empty_1 = 0;
     fifo_empty_2 = 0;
     fifo_full_0 = 0;
     fifo_full_1 = 0;
     fifo_full_2 = 0;
     fifo_read_0 = 0;
     fifo_read_1 = 0;
     fifo_read_2 = 0;
     fifo_write_0 = 0;
     fifo_write_1 = 0;
     fifo_write_2 = 0;
     fifo_cycle_0 = 0;
     fifo_cycle_1 = 0;
     fifo_cycle_2 = 0;

     ap_int<32> layer_loop = layer_count;

     for(int i=0;i<layer_loop;i++)
     {
         model_int[i][0] = model[i][0];
         model_int[i][1] = model[i][1];
         model_int[i][2] = model[i][2];
         model_int[i][3] = model[i][3];
         model_int[i][4] = model[i][4];
         model_int[i][5] = model[i][5];
         model_int[i][6] = model[i][6];
         model_int[i][7] = model[i][7];
         srelu_int[i] = srelu[i],
         quantization_scale_lin_int[i] = quantization_scale_lin[i];
         quantization_scale_w_int[i] = quantization_scale_w[i];
         quantization_scale_fea_int[i] = quantization_scale_fea[i];
         deq_factor_int[i] = deq_factor[i];
         scale_fea_int[i] = scale_fea[i];
         P_w_int[i] = P_w[i];

         std::cout << " Instruction is "<< model_int[i][7] <<  model_int[i][6] <<  model_int[i][5] <<  model_int[i][4] <<
         model_int[i][3] <<  model_int[i][1] <<  model_int[i][1] <<  model_int[i][0] << std::endl;
     }

      ITYPE max_fea_val = 0;

      mmult_wrapper(load_weights,beta_qu,f_align,beta_qul,f_alignl,quantization_scale_adj,quantization_scale_fea_int,quantization_scale_w_int,quantization_scale_lin_int,
      deq_factor_int,
      model_int,srelu_int,
      scale_fea_int,&max_fea_val,quantized_multiplier, shift_data, bias_data, bias_count, zero_point_lhs, zero_point_rhs, zero_point_dst, clamp_max,clamp_min,N_adj, M_adj, M_fea, P_w_int,
      B,B2,
      D1, D2, D3,D4,
      DS1, DS1R, DS1C,
      DS2, DS3,DS4,
      E1,
      S1,
      ate_m,
      array_c_adjust, layer_loop,
      nnz_fea1,nnz_fea2,nnz_fea3,nnz_fea4,
      rowPtr_fea1,rowPtr_fea2,rowPtr_fea3,rowPtr_fea4,
      columnIndex_fea1, columnIndex_fea2, columnIndex_fea3,columnIndex_fea4,
      values_fea1,values_fea2,values_fea3,values_fea4,
      rowPtr_feas1,rowPtr_feas2,rowPtr_feas3,rowPtr_feas4,
      columnIndex_feas1, columnIndex_feas2, columnIndex_feas3,columnIndex_feas4,
      values_feas1,values_feas2,values_feas3,values_feas4,
      nnz_adj1,nnz_adj2,nnz_adj3,nnz_adj4,
      rowPtr_adj1,rowPtr_adj2,rowPtr_adj3,rowPtr_adj4,
      columnIndex_adj1,columnIndex_adj2,columnIndex_adj3,columnIndex_adj4,
      values_adj1,values_adj2,values_adj3,values_adj4);

     *max_fea = max_fea_val;

      std::cout << "Done mmult wrapper" << std::endl;

     }
}

void kernelmult1(
bool load_weights,
int beta_qu,
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
)
{

    int array_c_adjust=N_adj;
    ap_uint<8> P_w_int[5];

    P_w_int[0]=P_w;
    float srelu[5];
    srelu[0]=0.0;

    std::cout << " kernel starting " << std::endl;

    mmult_top(load_weights,beta_qu,f_align,beta_qu,f_align,quantization_scale_adj,quantization_scale_fea,quantization_scale_w,quantization_scale_lin,deq_factor,
    model,srelu,scale_fea,max_fea,layer_count,quantized_multiplier,shift,bias,bias_count,profiling,zero_point_lhs,zero_point_rhs, zero_point_dst,clamp_max,clamp_min,
    N_adj, M_adj, M_fea, P_w_int,
    array_b, array_b2,
    array_d1,array_d2,array_d3,array_d4,
    stream_d1,stream_d1r,stream_d1c,
    stream_d2,stream_d3,stream_d4,
    array_e1,
    array_s1,
    ate_m,
    array_c_adjust,
    nnz_fea1,nnz_fea2,nnz_fea3,nnz_fea4,
    rowPtr_fea1,rowPtr_fea2,rowPtr_fea3,rowPtr_fea4,
    colIndices_fea1,colIndices_fea2,colIndices_fea3,colIndices_fea4,
    values_fea1,values_fea2,values_fea3,values_fea4,
    rowPtr_feas1,rowPtr_feas2,rowPtr_feas3,rowPtr_feas4,
    columnIndex_feas1, columnIndex_feas2, columnIndex_feas3,columnIndex_feas4,
    values_feas1,values_feas2,values_feas3,values_feas4,
    nnz_adj1,nnz_adj2,nnz_adj3,nnz_adj4,
    rowPtr_adj1,rowPtr_adj2,rowPtr_adj3,rowPtr_adj4,
    colIndices_adj1,colIndices_adj2,colIndices_adj3,colIndices_adj4,
    values_adj1,values_adj2,values_adj3,values_adj4);

    std::cout << " 0 output " << array_d1[0] << std::endl;

    std::cout << " 3 output " << array_d1[3] << std::endl;

    std::cout << " 7 output " << array_d1[7] << std::endl;

    std::cout << " kernel done " << std::endl;
}
