/*===============================================================================
* This file is part of the SGRACE GNN accelerator
* has been written at Linkoping/UPM University
* Author : Jose Nunez-Yanez
*Copyright (C) 2026 Jose Nunez-Yanez
*Licensed under the MIT license. See LICENSE file in the project root for details
===============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <hls_math.h>

#include <string>
#include <fstream>
#include <sstream> // ////std::stringstream

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"


#include "matrix_mult.h"

#include "hls_streamofblocks.h"

//typedef ITYPE buf[B_HEIGHT/4][B_WIDTH_BLOCK];

typedef QTYPE buf[F_HEIGHT/FEA_THREADS][B_WIDTH_BLOCK];

// note that  BLOCK shoudl be less than B_WIDTH_gmmBLOCK

//#if (B_WIDTH_BLOCK == 1)
//	const int BLOCK=2;
//#else
	//const int BLOCK=B_WIDTH_BLOCK;   //BLOCK should be less than B_WIDTH_BLOCK
//#endif

const int BLOCK=B_WIDTH_BLOCK;   //BLOCK should be less than B_WIDTH_BLOCK
const int SBLOCK=SPMM_BLOCK;   //BLOCK should be less than B_WIDTH_BLOCK
const int SBLOCK_LIN=1;   //BLOCK should be less than B_WIDTH_BLOCK

const int PARALLEL_ROW = B_BLOCK_PARALLEL;
//const int A_WIDTH_FIFO =  A_WIDTH;
//const int UNROLL_ADJ = PES_ADJ;
//const int UNROLL_FEA = PES_FEA;
const int FIFO_DEPTH = MAX_FIFO;

#if (LINEAR_ONLY==1)
 const int LINEAR_DEPTH=16;
#else
 const int LINEAR_DEPTH=B_WIDTH_BLOCK*B_HEIGHT;
#endif

//worst case attention fifo
//const int FIFO_DEPTH_ATTN = A_HEIGHT;
//const int FIFO_DEPTH_ATTN2 = A_HEIGHT*ATEN_BLOCK;
//attention FIFO with adj up to 12.5% nonzeros
//const int FIFO_DEPTH_ATTN = A_HEIGHT/8;
//const int FIFO_DEPTH_ATTN2 = A_HEIGHT*ATEN_BLOCK/8;
//attention FIFO with adj up to 12.5% nonzeros
//const int FIFO_DEPTH_ATTN = A_HEIGHT/8;
//const int FIFO_DEPTH_ATTN2 = A_HEIGHT*ATEN_BLOCK/8;

//OPT_ATTN control how much sparsity expected in adj (e.g. OPT_ATTN worse case with fully dense adjacency)
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

//extern "C" {
//     void *__dso_handle = NULL;
//}

void quanta(ATYPE &BW,float B,float quantization_scale,int f_align, int beta_qu)
{

    //std::cout << "adj in " << B << std::endl;
	float vfloat = quantization_scale*B+zero_point;
	float vround = hls::round(vfloat);
	//float vround = vfloat;

	ITYPE vquant = ITYPE(vround);

    //std::cout << "VROUND " << vround << std::endl;


    #if (SIGNED_MODE==0)
	ITYPE ibeta_q = (ITYPE)beta_qu;
	ITYPE ialpha_q = (ITYPE)(0.0);
    #else
	ITYPE beta_q = ITYPE(beta_qu>>1);
	ITYPE ibeta_q = (ITYPE)beta_q;
	ITYPE ialpha_q = -(ITYPE)beta_q;
    #endif

    //clippping
	if (vquant>ibeta_q)
		vquant = ibeta_q;
	else if (vquant<ialpha_q)
		vquant = ialpha_q;



    //std::cout << "VQUANT " << vquant << std::endl;
	if(f_align==7) //BINARY MODE
		f_align = 6;
	ITYPE vnorm = vquant >> (qbits-f_align-1);
 	ATYPE fval = ATYPE(vnorm);

    //std::cout << "FVAL " << fval << std::endl;

	BW = fval;


}


void quantf(FTYPE &BW,float B,float quantization_scale[5],int f_align, int beta_qu, int B_index)
{

    //std::cout << "fea in " << B << std::endl;
	float vfloat = quantization_scale[B_index]*B+zero_point;
	float vround = hls::round(vfloat);

	ITYPE vquant = ITYPE(vround);

    //std::cout << "VROUND " << vround << std::endl;

	//std::cout << "VQUANT " << vquant << std::endl;


    #if (SIGNED_MODE==0)
    ITYPE ibeta_q = (ITYPE)beta_qu;
    ITYPE ialpha_q = (ITYPE)(0.0);
    #else
    ITYPE beta_q = ITYPE(beta_qu>>1);
    ITYPE ibeta_q = (ITYPE)beta_q;
    ITYPE ialpha_q = -(ITYPE)beta_q;
    #endif

	//clippping
	if (vquant>ibeta_q)
		vquant = ibeta_q;
	else if (vquant<ialpha_q)
		vquant = ialpha_q;



    //std::cout << "FQUANT " << vquant << std::endl;
	if(f_align==7) //BINARY MODE
		f_align = 6;
	ITYPE vnorm = vquant >> (qbits-f_align-1);
 	FTYPE fval = FTYPE(vnorm);

    //std::cout << "FNORM " << fval << std::endl;

    //if(fval != 0.5)
    //    std::cout << "Error " << fval << std::endl;

 	//std::cout <<  fval << std::endl;

    //std::cout << "Quantize Feature out " << fval << std::endl;

	BW = fval;


}





void quantl(LTYPE &BW,float B,float quantization_scale[5],int f_align, int beta_qu, int B_index)
{

    //std::cout << "float Feature in " << B << std::endl;
	float vfloat = quantization_scale[B_index]*B+zero_point;

	float vround;

	//float vround = vfloat;

    //std::cout <<  "float in " << B << " ";

    //std::cout << vround << " ";

	ITYPE ibeta_q,ialpha_q,beta_q;

    if(f_align==7)
	{
	    ibeta_q = 1;
	    ialpha_q = -1;
        //vround = hls::round(vfloat*10);
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
	    //ialpha_q = -(ITYPE)beta_q-1; //use lowest negative number
    	vround = hls::round(vfloat);
    }

    //std::cout << "VROUND " << vround << std::endl;


	//float vround = hls::round(vfloat);


    //std::cout << vround << " ";


	ITYPE vquant = ITYPE(vround);

	//std::cout << "WQUANT " << vquant << std::endl;







	//clippping
	if (vquant>ibeta_q)
		vquant = ibeta_q;
	else if (vquant<ialpha_q)
		vquant = ialpha_q;

    //std::cout << "WQUANT " << vquant << std::endl;
	if(f_align==7) //BINARY MODE
		f_align = 6;
	ITYPE vnorm = vquant >> (qbitsl-f_align-1);
 	LTYPE lval = LTYPE(vnorm);

    //std::cout <<  fval << std::endl;


    //std::cout << "Quantize Feature out " << lval << std::endl;

	BW = lval;


}


void quantw(BTYPE &BW,float B,float quantization_scale[5],int f_align, int beta_qu, int B_index)
{

    //std::cout << "W in " << B << std::endl;
	float vfloat = quantization_scale[B_index]*B+zero_point;

	float vround;

	//float vround = vfloat;

	//std::cout <<  B << " " ;

    //std::cout << vround << " ";

	ITYPE ibeta_q,ialpha_q,beta_q;

    if(f_align==7)
	{
	    ibeta_q = 1;
	    ialpha_q = -1;
        //vround = hls::round(vfloat*10);
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
	    //ialpha_q = -(ITYPE)beta_q-1; //use lowest negative number
    	vround = hls::round(vfloat);
    }

    //std::cout << "VROUND " << vround << std::endl;


	//float vround = hls::round(vfloat);


    //std::cout << vround << " ";


	ITYPE vquant = ITYPE(vround);

	//std::cout << "WQUANT " << vquant << std::endl;







	//clippping
	if (vquant>ibeta_q)
		vquant = ibeta_q;
	else if (vquant<ialpha_q)
		vquant = ialpha_q;

    //std::cout << "WQUANT " << vquant << std::endl;
	if(f_align==7) //BINARY MODE
		f_align = 6;
	ITYPE vnorm = vquant >> (qbits-f_align-1);
 	BTYPE fval = BTYPE(vnorm);

    //std::cout <<  fval << std::endl;

	BW = fval;


}


void dsp_kernel_float_adj_1(ATYPE a_value,BTYPE b_block[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	//#pragma HLS ALLOCATION instances=mul limit=64 operation
	#pragma HLS INLINE

	//ITYPE acc[B_WIDTH_BLOCK];
	//#pragma HLS ARRAY_PARTITION variable=acc complete


	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			//#pragma HLS UNROLL factor=UNROLL_ADJ
			BTYPE b_val = b_block[b_row][j];
 	  		ATYPE a_val = a_value;
			//acc[j] += A_val*(b_value-zero_point_rhs);
 			//acc[j] = (a_val_float)*(b_val_float-rhs_float);
			acc[j] = (ITYPE)a_val*(ITYPE)b_val;
			//if (b_val > 1)
			//{
			//	//std::cout << " b_val " << b_val << std::endl;
			//	exit(0);
			//}
			////std::cout << "a_val " << a_val << " b_val " << b_val << std::endl;
			////std::cout << "acc " << acc[j] << std::endl;

	} // j loop


}
void dsp_kernel_float_adj_2(int block_size,ATYPE a_value,BTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block2[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	//#pragma HLS ALLOCATION instances=mul limit=64 operation
	#pragma HLS INLINE

	//ITYPE acc[B_WIDTH_BLOCK];
	//#pragma HLS ARRAY_PARTITION variable=acc complete


	for (int j = 0; j < B_WIDTH_BLOCK; j++) {

	  		ATYPE a_val = a_value;
	  		BTYPE b_val;


	  		int sel_block; // = (b_row>>(log2N-2))&0x3;
	  		int b_row_block;

	  		//std::cout << "b_row " << b_row << std::endl;

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

  		//if (b_row > (2*block_size-1) && b_row < 3*block_size)
	  		//{
  		//	b_row_block = b_row-2*block_size;
	  		//	sel_block = 2;
	  		//
  		//if (b_row > 3*block_size-1)
	  		//{
  		//	b_row_block = b_row-3*block_size;
	  		//	sel_block = 3;
	  		//}
  		//std::cout << "sel_block "  << sel_block << "b_row_block " << b_row_block << std::endl;

	  		//b_row = b_row&mask;



	  		BTYPE b_val1 = b_block1[b_row_block][j];
			BTYPE b_val2 = b_block2[b_row_block][j];
			//BTYPE b_val3 = b_block3[b_row_block][j];
			//BTYPE b_val4 = b_block4[b_row_block][j];


	  		switch(sel_block)
	  		{
	  			case 0:
	  				b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
	  				break;
	  			case 1:
	  				b_val = b_val2;
  				break;
	  			//case 2:
	  			//	b_val = b_val3;
  			//	break;
	  			//case 3:
	  			//	b_val = b_val4;
  			//	break;
	  		}
			//#pragma HLS UNROLL factor=UNROLL_ADJ
			//acc[j] += A_val*(b_value-zero_point_rhs);
 			//acc[j] = (a_val_float)*(b_val_float-rhs_float);
			acc[j] = (ITYPE)a_val*(ITYPE)b_val;
			//if (b_val > 1)
			//{
			//	//std::cout << " b_val " << b_val << std::endl;
			//	exit(0);
			//}
			////std::cout << "a_val " << a_val << " b_val " << b_val << std::endl;
			////std::cout << "acc " << acc[j] << std::endl;

	} // j loop


}


void dsp_kernel_float_adj_4(int block_size,ATYPE a_value,BTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block2[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block3[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block4[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	//#pragma HLS ALLOCATION instances=mul limit=64 operation
	#pragma HLS INLINE

	//ITYPE acc[B_WIDTH_BLOCK];
	//#pragma HLS ARRAY_PARTITION variable=acc complete


	for (int j = 0; j < B_WIDTH_BLOCK; j++) {

	  		ATYPE a_val = a_value;
	  		BTYPE b_val;


	  		int sel_block; // = (b_row>>(log2N-2))&0x3;
	  		int b_row_block;

	  		//std::cout << "b_row " << b_row << std::endl;

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
  		//std::cout << "sel_block "  << sel_block << "b_row_block " << b_row_block << std::endl;

	  		//b_row = b_row&mask;



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
			//#pragma HLS UNROLL factor=UNROLL_ADJ
			//acc[j] += A_val*(b_value-zero_point_rhs);
 			//acc[j] = (a_val_float)*(b_val_float-rhs_float);
			acc[j] = (ITYPE)a_val*(ITYPE)b_val;
			//if (b_val > 1)
			//{
			//	//std::cout << " b_val " << b_val << std::endl;
			//	exit(0);
			//}
			////std::cout << "a_val " << a_val << " b_val " << b_val << std::endl;
			////std::cout << "acc " << acc[j] << std::endl;

	} // j loop


}



void dsp_kernel_float_fea(ATYPE a_value,BTYPE b_block[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	//#pragma HLS ALLOCATION instances=mul limit=64 operation
	#pragma HLS INLINE

    //ITYPE acc_int[B_WIDTH_BLOCK];
	//#pragma HLS ARRAY_PARTITION variable=acc complete


	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			//#pragma HLS UNROLL factor=UNROLL_ADJ
			BTYPE b_val = b_block[b_row][j];
 	  		ATYPE a_val = a_value;
			//acc[j] += A_val*(b_value-zero_point_rhs);
 			//acc[j] = (a_val_float)*(b_val_float-rhs_float);
 	  		acc[j] = (ITYPE)a_val*(ITYPE)b_val;
			//if(b_val > 1)
			//{
			//	//std::cout << " b_val " << b_val << std::endl;
			//	exit(0);
			//}
			//std::cout << "a_val " << a_val << " b_val " << b_val << std::endl;
			//std::cout << "acc " << acc[j] << std::endl;

	} // j loop


}

void dsp_kernel_int_adj_1(int block_size,TTYPE a_value,QTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],
		//ITYPE b_block2[B_HEIGHT/2][B_WIDTH_BLOCK],
		//ITYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],
		//ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],
		ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	//#pragma HLS ALLOCATION instances=mul limit=64 operation
	//#pragma HLS INLINE

	//ITYPE acc[B_WIDTH_BLOCK];
	//#pragma HLS ARRAY_PARTITION variable=acc complete

	//for (int j = 0; j < B_WIDTH_BLOCK; j++) {

		//	#pragma HLS UNROLL

			//	acc[j] = 0;
        //}

	//int mask = (1 << (log2N-2)) - 1;


	//////////std::cout << "a_value " << a_value << ////std::endl;
	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
		//#pragma HLS UNROLL factor=UNROLL_ADJ
	    //#pragma HLS PIPELINE
 	  		TTYPE a_val = a_value;
 	  		QTYPE b_val;


 	  		int sel_block; // = (b_row>>(log2N-2))&0x3;
 	  		int b_row_block;

 	  		//std::cout << "b_row " << b_row << std::endl;

 	  		if (b_row < block_size)
 	  		{
 	  			b_row_block = b_row;
 	  			sel_block = 0;
 	  		}
 	  		//if (b_row > (block_size-1))
 	  		//{
 	  		//	b_row_block = b_row-block_size;
 	  		//	sel_block = 1;
 	  		//}

	  		//if (b_row > (2*block_size-1) && b_row < 3*block_size)
 	  		//{
	  		//	b_row_block = b_row-2*block_size;
 	  		//	sel_block = 2;
 	  		//}
	  		//if (b_row > 3*block_size-1)
 	  		//{
	  		//	b_row_block = b_row-3*block_size;
 	  		//	sel_block = 3;
 	  		//}
	  		//std::cout << "sel_block "  << sel_block << "b_row_block " << b_row_block << std::endl;

 	  		//b_row = b_row&mask;



 	  		QTYPE b_val1 = b_block1[b_row_block][j];
 	  		//std::cout << "b_val1 " << b_val1 << " " << "b_block1 "<< b_block1[b_row_block][j] << std::endl;
 			//BTYPE b_val2 = b_block2[b_row_block][j];
 			//BTYPE b_val3 = b_block3[b_row_block][j];
 			//BTYPE b_val4 = b_block4[b_row_block][j];


 	  		switch(sel_block)
 	  		{
 	  			case 0:
 	  				b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
 	  				break;
 	  			//case 1:
 	  			//	b_val = b_val2;
	  			//	break;
 	  			//case 2:
 	  			//	b_val = b_val3;
	  			//	break;
 	  			//case 3:
 	  			//	b_val = b_val4;
	  			//	break;
 	  		}

 	  		////std::cout << "b_val "  << b_val << std::endl;

 	  		//acc[j] += (A_val-zero_point_lhs)*(b_value-zero_point_rhs);
			//acc[j] += A_val*(b_value-zero_point_rhs);
			ITYPE a_val_i = (ITYPE)a_val;
            ITYPE b_val_i = (ITYPE)b_val;


			ITYPE acc_i = a_val_i*b_val_i;
			acc[j] = acc_i;
            //std::cout << "A val " << a_val_i << " " << " B val " << b_val_i << " " <<  acc_i << std::endl;
            //std::cout << "dsp kernel adj for j " << j << " is " << acc[j] << std::endl;
	} // j loop




}

void dsp_kernel_int_adj_2(int block_size,ITYPE a_value,QTYPE b_block1[B_HEIGHT/2][B_WIDTH_BLOCK],
		QTYPE b_block2[B_HEIGHT/2][B_WIDTH_BLOCK],
		//ITYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],
		//ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],
		ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	//#pragma HLS ALLOCATION instances=mul limit=64 operation
	//#pragma HLS INLINE

	//ITYPE acc[B_WIDTH_BLOCK];
	//#pragma HLS ARRAY_PARTITION variable=acc complete

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {

			#pragma HLS UNROLL

				acc[j] = 0;
        }

	//int mask = (1 << (log2N-2)) - 1;


	//////////std::cout << "a_value " << a_value << ////std::endl;
	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
		//#pragma HLS UNROLL factor=UNROLL_ADJ
	    //#pragma HLS PIPELINE
 	  		ATYPE a_val = a_value;
 	  		BTYPE b_val;


 	  		int sel_block; // = (b_row>>(log2N-2))&0x3;
 	  		int b_row_block;

 	  		//std::cout << "b_row " << b_row << std::endl;

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

	  		//if (b_row > (2*block_size-1) && b_row < 3*block_size)
 	  		//{
	  		//	b_row_block = b_row-2*block_size;
 	  		//	sel_block = 2;
 	  		//}
	  		//if (b_row > 3*block_size-1)
 	  		//{
	  		//	b_row_block = b_row-3*block_size;
 	  		//	sel_block = 3;
 	  		//}
	  		//std::cout << "sel_block "  << sel_block << "b_row_block " << b_row_block << std::endl;

 	  		//b_row = b_row&mask;



 	  		BTYPE b_val1 = b_block1[b_row_block][j];
 			BTYPE b_val2 = b_block2[b_row_block][j];
 			//BTYPE b_val3 = b_block3[b_row_block][j];
 			//BTYPE b_val4 = b_block4[b_row_block][j];


 	  		switch(sel_block)
 	  		{
 	  			case 0:
 	  				b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
 	  				break;
 	  			case 1:
 	  				b_val = b_val2;
	  				break;
 	  			//case 2:
 	  			//	b_val = b_val3;
	  			//	break;
 	  			//case 3:
 	  			//	b_val = b_val4;
	  			//	break;
 	  		}

 	  		////std::cout << "b_val "  << b_val << std::endl;

 	  		//acc[j] += (A_val-zero_point_lhs)*(b_value-zero_point_rhs);
			//acc[j] += A_val*(b_value-zero_point_rhs);
			ITYPE a_val_i = (ITYPE)a_val;
			ITYPE b_val_i = (ITYPE)b_val;

			ITYPE acc_i = a_val_i*b_val_i;
			acc[j] += acc_i;
			////std::cout << "A val " << a_val_i << " " << a_val << " B val " << b_val_i << //std::endl;
			////std::cout << "dsp kernel for j " << j << " is " << acc_i << //std::endl;
	} // j loop




}

void dsp_kernel_int_adj_4(int block_size,TTYPE a_value,QTYPE b_block1[B_HEIGHT/4][B_WIDTH_BLOCK],
		QTYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],QTYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],
		QTYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	//#pragma HLS ALLOCATION instances=mul limit=64 operation
	//#pragma HLS INLINE

	//ITYPE acc[B_WIDTH_BLOCK];
	//#pragma HLS ARRAY_PARTITION variable=acc complete

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {

			#pragma HLS UNROLL

				acc[j] = 0;
        }

	//int mask = (1 << (log2N-2)) - 1;


	//////////std::cout << "a_value " << a_value << ////std::endl;
	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
		//#pragma HLS UNROLL factor=UNROLL_ADJ
	    //#pragma HLS PIPELINE
 	  		TTYPE a_val = a_value;
 	  		QTYPE b_val;


 	  		int sel_block; // = (b_row>>(log2N-2))&0x3;
 	  		int b_row_block;

 	  		//std::cout << "b_row " << b_row << std::endl;

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
	  		//std::cout << "sel_block "  << sel_block << "b_row_block " << b_row_block << std::endl;

 	  		//b_row = b_row&mask;



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

 	  		////std::cout << "b_val "  << b_val << std::endl;

 	  		//acc[j] += (A_val-zero_point_lhs)*(b_value-zero_point_rhs);
			//acc[j] += A_val*(b_value-zero_point_rhs);
			ITYPE a_val_i = (ITYPE)a_val;
			ITYPE b_val_i = (ITYPE)b_val;

			ITYPE acc_i = a_val_i*b_val_i;
			acc[j] += acc_i;
			////std::cout << "A val " << a_val_i << " " << a_val << " B val " << b_val_i << //std::endl;
			////std::cout << "dsp kernel for j " << j << " is " << acc_i << //std::endl;
	} // j loop




}


void dsp_kernel_int_fea(FTYPE a_value,BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	//#pragma HLS ALLOCATION instances=mul limit=64 operation
	//#pragma HLS INLINE

	//ITYPE acc[B_WIDTH_BLOCK];
	//#pragma HLS ARRAY_PARTITION variable=acc complete

	//for (int j = 0; j < B_WIDTH_BLOCK; j++) {

	//		#pragma HLS UNROLL

	//			acc[j] = 0;
    //    }

	//////////std::cout << "a_value " << a_value << ////std::endl;
	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
		//#pragma HLS UNROLL factor=UNROLL_FEA
	    //#pragma HLS PIPELINE
 	  		FTYPE a_val = a_value;
			BTYPE b_val = b_block[b_row][j]; //only one value of B in each row. This is the result of the first matrix mult.

			//std::cout << "b bal " << b_val << std::endl;
			//acc[j] += (A_val-zero_point_lhs)*(b_value-zero_point_rhs);
			//acc[j] += A_val*(b_value-zero_point_rhs);

            //#if (BINARY_MODE == 1)
	        ITYPE b_val_i;
	        ITYPE a_val_i;
            ///if (b_val.length() == 1) //fix for binary mode
            //{
 //
 //              a_val_i = (ITYPE)a_val;
//	           if(b_val==0)
//	               b_val_i = (ITYPE)("0b0.1"); //we transform 0 into ...00001 (so +1)and 1 into ...111111 so (-1) for weights
//	               //b_val_i = (ITYPE)(1); //we transform 0 into ...00001 (so +1)and 1 into ...111111 so (-1) for weights
//	           else
//	               //b_val_i = (ITYPE)(b_val);
//	        	   b_val_i = -(ITYPE)("0b0.1");
//
//	           //if(a_val==0)
//	           //    a_val_i = (ITYPE)("0b0.0"); //we transform 0 into ...00001 (so +1)and 1 into ...111111 so (-1) for weights
//	               //b_val_i = (ITYPE)(1); //we transform 0 into ...00001 (so +1)and 1 into ...111111 so (-1) for weights
//	           //else
//	               //b_val_i = (ITYPE)(b_val);
//	        	//   a_val_i = (ITYPE)("0b0.1");
//	           //std::cout << "b_val "  << b_val << std::endl;
//	           //std::cout << "b_val_i "  << b_val_i << std::endl;
//	           //std::cout << "a_val "  << a_val << std::endl;
//	           //std::cout << "a_val_i "  << a_val_i << std::endl;
//            }
//	        else
//	        {
			   b_val_i = (ITYPE)b_val;
			   a_val_i = (ITYPE)a_val;
//	        }
			  //#endif

		    ITYPE acc_i = a_val_i*b_val_i;
			acc[j] = acc_i;

            //std::cout << "A val " << a_val_i <<  " B val " << b_val_i << std::endl;
			//std::cout << "dsp kernel fea for j " << j << " is " << acc_i << std::endl;

			//std::cout << "fea val " << a_val_i << " weight val " << b_val_i << std::endl;
			//std::cout << "acumm dsp kernel fea for j " << j << " is " << acc[j] << std::endl;
	} // j loop
	//std::cout << "dsp kernel fea is " << acc_i << std::endl;

   //exit(0);

}


void dsp_kernel_int_lin(LTYPE a_value,BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	//#pragma HLS ALLOCATION instances=mul limit=64 operation
	//#pragma HLS INLINE

	//ITYPE acc[B_WIDTH_BLOCK];
	//#pragma HLS ARRAY_PARTITION variable=acc complete

	//for (int j = 0; j < B_WIDTH_BLOCK; j++) {

	//		#pragma HLS UNROLL

	//			acc[j] = 0;
    //    }

	//////////std::cout << "a_value " << a_value << ////std::endl;
	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
		//#pragma HLS UNROLL factor=UNROLL_FEA
	    //#pragma HLS PIPELINE
 	  		LTYPE a_val = a_value;
			BTYPE b_val = b_block[b_row][j]; //only one value of B in each row. This is the result of the first matrix mult.

			//std::cout << "b bal " << b_val << std::endl;
			//acc[j] += (A_val-zero_point_lhs)*(b_value-zero_point_rhs);
			//acc[j] += A_val*(b_value-zero_point_rhs);

            //#if (BINARY_MODE == 1)
	        ITYPE b_val_i;
	        ITYPE a_val_i;
            ///if (b_val.length() == 1) //fix for binary mode
            //{
 //
 //              a_val_i = (ITYPE)a_val;
//	           if(b_val==0)
//	               b_val_i = (ITYPE)("0b0.1"); //we transform 0 into ...00001 (so +1)and 1 into ...111111 so (-1) for weights
//	               //b_val_i = (ITYPE)(1); //we transform 0 into ...00001 (so +1)and 1 into ...111111 so (-1) for weights
//	           else
//	               //b_val_i = (ITYPE)(b_val);
//	        	   b_val_i = -(ITYPE)("0b0.1");
//
//	           //if(a_val==0)
//	           //    a_val_i = (ITYPE)("0b0.0"); //we transform 0 into ...00001 (so +1)and 1 into ...111111 so (-1) for weights
//	               //b_val_i = (ITYPE)(1); //we transform 0 into ...00001 (so +1)and 1 into ...111111 so (-1) for weights
//	           //else
//	               //b_val_i = (ITYPE)(b_val);
//	        	//   a_val_i = (ITYPE)("0b0.1");
//	           //std::cout << "b_val "  << b_val << std::endl;
//	           //std::cout << "b_val_i "  << b_val_i << std::endl;
//	           //std::cout << "a_val "  << a_val << std::endl;
//	           //std::cout << "a_val_i "  << a_val_i << std::endl;
//            }
//	        else
//	        {
			   b_val_i = (ITYPE)b_val;
			   a_val_i = (ITYPE)a_val;
//	        }
			  //#endif

		    ITYPE acc_i = a_val_i*b_val_i;
			acc[j] = acc_i;

            //if (acc_i < 0)
			//	std::cout << " here " << std::endl;

			//if(acc_i!=0)
			//{
            // std::cout << "A val " << a_val_i <<  " B val " << b_val_i << std::endl;
			// std::cout << "dsp kernel lin for j " << j << " is " << acc_i << std::endl;
			//}
			//std::cout << "fea val " << a_val_i << " weight val " << b_val_i << std::endl;
			//std::cout << "acumm dsp kernel fea for j " << j << " is " << acc[j] << std::endl;
	} // j loop
	//std::cout << "dsp kernel fea is " << acc_i << std::endl;

   //exit(0);

}

/*void writec(int first_row, int row_count,int P, hls::stream<ITYPE> write_fifo[B_WIDTH_BLOCK][SPMM_BLOCK], DTYPE* C,int B_index,int B_index_loop,int tail)
{
		int B_WIDTH_INT;

		int WL;

		#if defined FLOAT
			WL = row_count;
			//array_c_adjust = array_c_adjust;
		#endif

		#if defined HALF
			WL = row_count;
			//array_c_adjust = array_c_adjust;
		#endif

		#ifdef EIGHTBIT
			//WL = (N>>2);
			//array_c_adjust = (array_c_adjust>>2);
			WL = row_count;
			//array_c_adjust = array_c_adjust;

		#endif



		if (B_index < (B_index_loop-1))
			B_WIDTH_INT = B_WIDTH_BLOCK;
		else
			B_WIDTH_INT = tail;


		////std::cout << " WL " << WL << " B_WIDTH_INT " << B_WIDTH_INT << std::endl;
		LOOP_WRITE1:    for (int i = 0; i < WL; i+=SPMM_BLOCK) {
			DTYPE C_out;
	  		LOOP_WRITE2: for (int j = 0; j <  B_WIDTH_INT; j++) {
		  		LOOP_WRITE3: for (int z = 0; z <  SPMM_BLOCK; z++) {
					#pragma HLS PIPELINE
	 				if ((z+i) < WL)
					{
						C_out =  write_fifo[j][z].read();
						//////std::cout << "B_index " << B_index << ////std::endl;
						////std::cout << " cout in position " << (i+first_row) << "," << j << " is " << C_out << std::endl;
						#ifdef ENABLE_TRANSPOSE
							//C[i+(j+B_index*B_WIDTH_BLOCK)*(array_c_adjust)] = C_out;
						#else
							C[(i+z)*P+j+B_index*B_WIDTH_BLOCK] = C_out;
							//C[i*B_WIDTH_INT+j] = C_out;
							//C[j] = C_out;
						#endif
					}
				}
			}

		}
}*/



void writec(float deq_factor[5],ap_uint<1> model[5][8],int first_row, int row_count,int N_adj,ap_uint<8> P[5], hls::stream<ITYPE> write_fifo[B_WIDTH_BLOCK],QTYPE linear_pipo[B_HEIGHT][B_WIDTH_BLOCK],hls::stream<OUTTYPE>& CS, int B_index, int layer_loop)
{
		int B_WIDTH_INT;

	    bool linear_mode;
	    bool sage_mode;

		int WL;

		#if defined FLOAT
			WL = row_count;
			//array_c_adjust = array_c_adjust;
		#endif

		#if defined HALF
			WL = row_count;
			//array_c_adjust = array_c_adjust;
		#endif

		#ifdef EIGHTBIT
			//WL = (N>>2);
			//array_c_adjust = (array_c_adjust>>2);
			WL = row_count;
			//array_c_adjust = array_c_adjust;

		#endif

        linear_mode = model[B_index][6];
        sage_mode = model[B_index][7];
		bool gcn_path = !(linear_mode^sage_mode);

		////std::cout << " WL " << WL << " B_WIDTH_INT " << B_WIDTH_INT << std::endl;
		//LOOP_WRITE1:    for (int i = 0; i < WL; i++) {

		DTYPE C_out = DTYPE(0.0);
		DTYPE residual;

        LOOP_WRITE42:    for (int i = 0; i < WL; i++) {
		    LOOP_WRITE52: for (int j = 0; j <  B_WIDTH_BLOCK; j++) {
					 #pragma HLS PIPELINE II=1
                        #if LINEAR_ONLY == 0
		    	        if (gcn_path==1)
						 C_out =  DTYPE(write_fifo[j].read());
		    	        else
		    	         C_out = 0;
                        #endif
                        #if GNN_ONLY == 0
						if (linear_mode==1)
						 residual = DTYPE(linear_pipo[i][j]);
						else
						 residual = 0;
                        #else
						residual = DTYPE(0.0);
                        #endif

						//std::cout << " linear_mode  " << linear_mode << " C_out " << C_out  << std::endl;
						//dequant
                        #if (INT_DEQUANT == 1)
						     //OUTTYPE C_float = (OUTTYPE)C_out*deq_factor[B_index]+(OUTTYPE)residual*deq_factor[B_index]*(sage_mode || linear_mode);
						     OUTTYPE C_float = (OUTTYPE)C_out*deq_factor[B_index]+(OUTTYPE)residual*deq_factor[B_index];
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

                     //terminate stream
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

                     //#if (TRAINING_MODE == 1)
                     //  C[i*B_WIDTH_INT+j+first_row*B_WIDTH_BLOCK+N_adj*B_WIDTH_BLOCK*B_index] = C_float;
                     //#endif
                     //C[i*B_WIDTH_INT+j+first_row*B_WIDTH_BLOCK+N_adj*B_WIDTH_BLOCK*B_index] = last;

			       }
                  }

		}
		else  // write to memory
		{
	        	LOOP_WRITE45:    for (int i = 0; i < WL; i++) {
		 	     LOOP_WRITE55: for (int j = 0; j <  B_WIDTH_INT; j++) {
					#pragma HLS PIPELINE II=1
				 	OUTTYPE C_float =  OUTTYPE(write_fifo.read());
					//C[i*B_WIDTH_INT+j+first_row*B_WIDTH_BLOCK+N_adj*B_WIDTH_BLOCK*B_index] = C_float;
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
						//dequant
                        #if (INT_DEQUANT == 1)
					     OUTTYPE C_float = (OUTTYPE)C_out*deq_factor;
                        #else
						 OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif

						//C[i*P+j+B_index*WL*B_WIDTH_BLOCK] = C_out;
						//std::cout << "first_row " << first_row << std::endl;
						//std::cout << "WL " << WL << std::endl;
						//std::cout << "C float " << C_float << std::endl;
						//std::cout << " cout in position " << i << "," << j << "," << z <<  " is " << C_out << std::endl;
						//std::cout << " cfloat in position " << i << "," << j << " is " << C_float << std::endl;
                        #if (STREAM_MODE_OUT == 1)  //if (stream_mode == 1)
                        {
                           ASTYPE temp;
                           temp.data = C_float;
                           CS.write(temp);
                        }
                        #else
						   //C[i*P+j+first_row*B_WIDTH_BLOCK+B_index*N_adj*B_WIDTH_BLOCK] = C_out;
						   //C[i+j+first_row*B_WIDTH_BLOCK+B_index*N_adj*B_WIDTH_BLOCK] = C_float;
						   C[i*FIFO_DEPTH+j*WL+z+first_row*B_WIDTH_BLOCK+B_index*N_adj*B_WIDTH_BLOCK] = C_float;
                        #endif
						//C[j] = C_out;
					}

			     }

		}

}



void writes(float deq_factor[5],ap_uint<1> model[5][8], int first_row, int row_count,int N_adj,ap_uint<8> P[5], hls::stream<TTYPE> &write_fifo, hls::stream<int> &rnnz_fifo,  OUTTYPE* C,int B_index)
{
		int B_WIDTH_INT;

		int WL;



		//WL = (N>>2);
		//array_c_adjust = (array_c_adjust>>2);
		WL = row_count;
		//array_c_adjust = array_c_adjust;



		B_WIDTH_INT = B_WIDTH_BLOCK;



		////std::cout << " WL " << WL << " B_WIDTH_INT " << B_WIDTH_INT << std::endl;}
		//LOOP_WRITE1:    for (int i = 0; i < WL; i++) {

        //int rnnz_total = 0;

		bool linear_mode = model[B_index][6];
		 bool gat_mode = model[B_index][5];

		if (linear_mode == 0)
		{

		 if (gat_mode == 1)
		 {

			 int rnnz = rnnz_fifo.read();



		  /*LOOP_WRITE4:    for (int i = 0; i < WL; i++) {
			DTYPE C_out;
			int rnnz = rnnz_fifo.read();

			    //LOOP_WRITE5: for (int j = 0; j <  B_WIDTH_INT; j++) {
			     LOOP_WRITE5: for (int j = 0; j <  rnnz; j++) {
					#pragma HLS PIPELINE
						C_out =  write_fifo.read();
						//C[i*P+j+B_index*WL*B_WIDTH_BLOCK] = C_out;
						//std::cout << "first_row " << first_row << std::endl;
						//std::cout << "S out " << C_out << std::endl;
						// for multithreading each thread must know how many rnnz the other thred is generating
						C[j] = C_out;
					}
		  }*/

			     DTYPE C_out;

			    //LOOP_WRITE5: for (int j = 0; j <  B_WIDTH_INT; j++) {
			     LOOP_WRITE5: for (int i = 0; i <  rnnz; i++) {
					#pragma HLS PIPELINE
						C_out =  write_fifo.read();
                        #if (INT_DEQUANT == 1)
				           OUTTYPE C_float = (OUTTYPE)C_out*deq_factor[B_index];
                        #else
				           OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif
						//C[i*P+j+B_index*WL*B_WIDTH_BLOCK] = C_out;
						//std::cout << "first_row " << first_row << std::endl;
						//std::cout << "S out " << C_float << std::endl;
						// for multithreading each thread must know how many rnnz the other thred is generating
						C[i] = C_float;
					}

		  }
		} //linear mode



}



void writesx4(float deq_factor,bool gat_mode,
		int row_count1,int row_count2,int row_count3,int row_count4,
		hls::stream<TTYPE> &write_fifo1,
		hls::stream<TTYPE> &write_fifo2,
		hls::stream<TTYPE> &write_fifo3,
		hls::stream<TTYPE> &write_fifo4,
		hls::stream<int> &rnnz_fifo1,
		hls::stream<int> &rnnz_fifo2,
		hls::stream<int> &rnnz_fifo3,
		hls::stream<int> &rnnz_fifo4,
		OUTTYPE* C,int B_index)
{
		int B_WIDTH_INT;

		//int WL;

		DTYPE C_out;

		//hls::stream<TTYPE> S_fifo;


		//WL = (N>>2);
		//array_c_adjust = (array_c_adjust>>2);

		//array_c_adjust = array_c_adjust;



		B_WIDTH_INT = B_WIDTH_BLOCK;



		////std::cout << " WL " << WL << " B_WIDTH_INT " << B_WIDTH_INT << std::endl;}
		//LOOP_WRITE1:    for (int i = 0; i < WL; i++) {

        int rnnz_total = 0;
        int rnnz1,rnnz2,rnnz3,rnnz4;

		rnnz1 = rnnz_fifo1.read();
		rnnz2 = rnnz_fifo2.read();
		rnnz3 = rnnz_fifo3.read();
		rnnz4 = rnnz_fifo4.read();

    	//int rnnz_offset = rnnz_total.read();

		if (gat_mode == 1)
		{
		  //WL = row_count1;
		  //LOOP_WRITE41:    for (int i = 0; i < WL; i++) {



			    //LOOP_WRITE5: for (int j = 0; j <  B_WIDTH_INT; j++) {
			     LOOP_WRITE51: for (int j = 0; j <  rnnz1; j++) {
					#pragma HLS PIPELINE
						C_out =  write_fifo1.read();
						//C[i*P+j+B_index*WL*B_WIDTH_BLOCK] = C_out;
						//std::cout << "first_row " << first_row << std::endl;
						//std::cout << "S out " << C_out << std::endl;
						// for multithreading each thread must know how many rnnz the other thred is generating
                        #if (INT_DEQUANT == 1)
				           OUTTYPE C_float = (OUTTYPE)C_out*deq_factor;
                        #else
				           OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif
						C[j+rnnz_total] = C_float;
						//S_fifo << C_out;
					}
					rnnz_total+=rnnz1;
		  //}
		  //WL = row_count2;
		  //LOOP_WRITE42:    for (int i = 0; i < WL; i++) {
			//DTYPE C_out;


			    //LOOP_WRITE5: for (int j = 0; j <  B_WIDTH_INT; j++) {
			     LOOP_WRITE52: for (int j = 0; j <  rnnz2; j++) {
					#pragma HLS PIPELINE
						C_out =  write_fifo2.read();
						//C[i*P+j+B_index*WL*B_WIDTH_BLOCK] = C_out;
						//std::cout << "first_row " << first_row << std::endl;
						//std::cout << "S out " << C_out << std::endl;
						// for multithreading each thread must know how many rnnz the other thred is generating
                        #if (INT_DEQUANT == 1)
				           OUTTYPE C_float = (OUTTYPE)C_out*deq_factor;
                        #else
				           OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif
						C[j+rnnz_total] = C_float;
						//S_fifo << C_out;
					}
					rnnz_total+=rnnz2;
		  //}
		  //WL = row_count3;
		  //LOOP_WRITE43:    for (int i = 0; i < WL; i++) {
			//DTYPE C_out;


			    //LOOP_WRITE5: for (int j = 0; j <  B_WIDTH_INT; j++) {
			     LOOP_WRITE53: for (int j = 0; j <  rnnz3; j++) {
					#pragma HLS PIPELINE
						C_out =  write_fifo3.read();
						//C[i*P+j+B_index*WL*B_WIDTH_BLOCK] = C_out;
						//std::cout << "first_row " << first_row << std::endl;
						//std::cout << "S out " << C_out << std::endl;
						// for multithreading each thread must know how many rnnz the other thred is generating
                        #if (INT_QUANT == 1)
				           OUTTYPE C_float = (OUTTYPE)C_out*deq_factor;
                        #else
				           OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif
						C[j+rnnz_total] = C_float;
						//S_fifo << C_out;
					}
					rnnz_total+=rnnz3;
		  //}
		  //WL = row_count4;
		  //LOOP_WRITE44:    for (int i = 0; i < WL; i++) {
			//DTYPE C_out;


			    //LOOP_WRITE5: for (int j = 0; j <  B_WIDTH_INT; j++) {
			     LOOP_WRITE54: for (int j = 0; j <  rnnz4; j++) {
					#pragma HLS PIPELINE
						C_out =  write_fifo4.read();
						//C[i*P+j+B_index*WL*B_WIDTH_BLOCK] = C_out;
						//std::cout << "first_row " << first_row << std::endl;
						//std::cout << "S out " << C_out << std::endl;
						// for multithreading each thread must know how many rnnz the other thred is generating
                        #if (INT_DEQUANT == 1)
				           OUTTYPE C_float = (OUTTYPE)C_out*deq_factor;
                        #else
				           OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif
						C[j+rnnz_total] = C_float;
						//S_fifo << C_out;
					}
					rnnz_total+=rnnz4;
		  //}

		 /* LOOP_WRITE: for (int i = 0; i < rnnz_total; i++) {
				DTYPE C_out;
				#pragma HLS PIPELINE
				C_out =  S_fifo.read();
				C[i] = C_out;
		 }*/

		}


}







void readptr_csr_fea(bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

    #pragma HLS inline off
	int rnnz,current_index,next_index;

	current_index= rowPtr[0];

	if (gemm_mode==0)
	{
		//printf("N rows are %d\n", N);
		LOOP_A_INDEX_SPMM1 : for (int A_index = 0; A_index < N; A_index++) {
			#pragma HLS PIPELINE
					next_index=rowPtr[A_index+1];
					rnnz = next_index-current_index;
					current_index = next_index;
					rnnz_fifo << rnnz;
					//printf ("BRNNZ %d\n",brnnz);
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
	//ADJ dataflow
	LOOP_A_INDEX0 : for (int A_index = 0; A_index <nnz_fea+1 ; A_index++)
	{
		#pragma HLS PIPELINE
  	    next_index1=rowPtr[A_index];
		//printf ("next index %d\n",next_index1);
  	    index_fifo << next_index1;
  	    //index_fifo[A_index] = next_index1;
	}

}



void read_ptr(bool stream_mode,int nnz_fea,int *rowPtr,  hls::stream<int> &index_fifo)
{
	int next_index1;
	//FEA dataflow
	if (stream_mode == 0)
	{
	  LOOP_A_INDEX0 : for (int A_index = 0; A_index <nnz_fea+1 ; A_index++)
	  {
		#pragma HLS PIPELINE
  	    next_index1=rowPtr[A_index];
		//printf ("next index %d\n",next_index1);
  	    index_fifo << next_index1;
  	    //index_fifo[A_index] = next_index1;
	  }
	}

}

/*
void proc_ptr(int nnz_fea,hls::stream<int> &index_fifo,hls::stream<int> &rnnz_fifo)
{
	int next_index2;
	int rnnz = 0;
	int current_index = 0;
	int B_index = 0;
	int first_read = 1;

	LOOP_A_INDEX1 : while(B_index < nnz_fea) {

	//printf ("B_index %d, %d\n",B_index,nnz_fea);
	#pragma HLS PIPELINE
	if(first_read == 1)
	{
	   next_index2=index_fifo.read();
	   //next_index2=index_fifo[0];
	   first_read = 0;
	}
	else if(next_index2 == current_index)
	{
	   rnnz++;
	   next_index2=index_fifo.read();
	   //next_index2=index_fifo[B_index];
	   B_index++;
     }
     else
     {
  	   printf ("RNNZ %d\n",rnnz);
	   rnnz_fifo << rnnz;
	   //printf ("current index %d\n",current_index);

	   current_index++;
	   //printf ("current_index %d\n",current_index);
	   //printf ("next_index2 %d\n",next_index2);
	   rnnz = 0;
	}

   }
   rnnz_fifo << rnnz;
   printf ("RNNZ %d\n",rnnz);

}
*/


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
	//LOOP_A_INDEX1 : while(B_index < nnz_fea) {
	//printf ("B_index %d, %d\n",B_index,nnz_fea);
	#pragma HLS PIPELINE
	next_index2=index_fifo.read();
	B_index++;
	if(next_index2 == current_index)
	{
	   rnnz++;
	   //next_index2=index_fifo.read();
	   //next_index2=index_fifo[B_index];

    }
    else
    {
  	  //printf ("RNNZ %d\n",rnnz);
#if (LINEAR_ONLY == 0)
	  rnnz_fifo << rnnz;
	  //printf ("current index %d\n",current_index);
#endif
	  current_index=next_index2;
	  //printf ("current_index %d\n",current_index);
	  //printf ("next_index2 %d\n",next_index2);
	  rnnz = 1;
	}

   }

   //if(next_index2 == current_index)
   //{
#if (LINEAR_ONLY == 0)
    rnnz_fifo << rnnz;
#endif

    //next_index2=index_fifo.read();
    //printf ("RNNZ eq %d\n",rnnz);
    //printf("next %d\n",next_index2 );
    //printf("current %d\n",current_index );
   //}
   //else
	//printf ("RNNZ diff %d\n",rnnz);

    next_index2=index_fifo.read();
}


void proc_ptr2(bool gcn_path,bool linear_mode,bool stream_mode,int nnz_fea,hls::stream<int> &index_fifo,hls::stream<ASTYPE>&  rowPtrs,hls::stream<int> &rnnz_fifo,hls::stream<int> &rnnz_fifo_sage)
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

	 //printf ("B_index %d, %d\n",B_index,nnz_fea);
	 #pragma HLS PIPELINE
	 next_index2 =index_fifo.read();
	 B_index++;
	 if(next_index2 == current_index)
	 {
	   rnnz++;
	   //next_index2=index_fifo.read();
	   //next_index2=index_fifo[B_index];

     }
     else
     {
       //printf ("RNNZ %d\n",rnnz);
       #if (LINEAR_ONLY == 0)
	   if(gcn_path==1)
	    rnnz_fifo << rnnz;
       #endif
       #if (GNN_ONLY == 0)
	   if(linear_mode==1)
	    rnnz_fifo_sage << rnnz;
       #endif
	   //printf ("current index %d\n",current_index);

	   current_index=next_index2;
	   //printf ("current_index %d\n",current_index);
	   //printf ("next_index2 %d\n",next_index2);
	   rnnz = 1;
	 }

   }

   //printf ("RNNZ %d\n",rnnz);
   #if (LINEAR_ONLY == 0)
   if(gcn_path==1)
    rnnz_fifo << rnnz;
   #endif
   #if (GNN_ONLY == 0)
   if(linear_mode==1)
    rnnz_fifo_sage << rnnz;
   #endif
   next_index2=index_fifo.read();
   }
   else //stream mode on
   {

	 temp=rowPtrs.read();
	 //printf("rowPtrs %d\n",temp.data);
     //rnnz++;
     rnnz=1;
	 current_index= temp.data;

	 if(temp.last!=1)
	 {
	  LOOP_A_INDEX2 : do {
		 //printf ("B_index %d, %d\n",B_index,nnz_fea);
		 #pragma HLS PIPELINE
	  	 //printf ("RNNZ %d\n",rnnz);
		 temp=rowPtrs.read();
		 next_index2= temp.data;
		 //B_index++;
		 if(next_index2 == current_index)
		 {
		   rnnz++;
		   //next_index2=index_fifo.read();
		   //next_index2=index_fifo[B_index];

	     }
	     else
	     {
	       //printf ("RNNZ %d\n",rnnz);
	       #if (LINEAR_ONLY == 0)
		   if(gcn_path==1)
		    rnnz_fifo << rnnz;
	       #endif
	       #if (GNN_ONLY == 0)
		   if(linear_mode==1)
		    rnnz_fifo_sage << rnnz;
	       #endif
		   //printf ("current index %d\n",current_index);

		   current_index=next_index2;
		   //printf ("current_index %d\n",current_index);
		   //printf ("next_index2 %d\n",next_index2);
		   rnnz = 1;
		 }
	   }while(temp.last!=1);
	 }

   //printf ("RNNZ %d\n",rnnz);
   #if (LINEAR_ONLY == 0)
   if(gcn_path==1)
    rnnz_fifo << rnnz;
   #endif
   #if (GNN_ONLY == 0)
   if(linear_mode==1)
    rnnz_fifo_sage << rnnz;
   #endif

   }
    //next_index2=index_fifo.read();
    //printf ("RNNZ eq %d\n",rnnz);
    //printf("next %d\n",next_index2 );
    //printf("current %d\n",current_index );
   //}
   //else
	//printf ("RNNZ diff %d\n",rnnz);

   //next_index2=index_fifo.read();
}

void read_dataflow2(bool gcn_path,bool linear_mode,bool stream_mode,int nnz_fea,int *rowPtr,hls::stream<ASTYPE>&  rowPtrs,hls::stream<int> &rnnz_fifo,hls::stream<int> &rnnz_fifo_sage)
{

    hls::stream<int>  index_fifo("index fifo");
	//hls::stream<ap_axiu<32, 1, 1, 1>>  index_fifo("index fifo");
	#pragma HLS STREAM variable= index_fifo depth=FIFO_DEPTH

    //FEA READDATAFLOW

    #pragma HLS DATAFLOW
    read_ptr(stream_mode,nnz_fea,rowPtr,index_fifo);
	proc_ptr2(gcn_path,linear_mode,stream_mode,nnz_fea,index_fifo,rowPtrs,rnnz_fifo,rnnz_fifo_sage);


}

void read_dataflow(int nnz_fea,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

	hls::stream<int>  index_fifo("index fifo");
	#pragma HLS STREAM variable= index_fifo depth=FIFO_DEPTH

    //ADJ READDATAFLOW

    #pragma HLS DATAFLOW
    read_ptr2(nnz_fea,rowPtr,index_fifo);
	proc_ptr(nnz_fea,index_fifo,rnnz_fifo);


}

void readptr_coo_fea(int nnz_fea,bool sage_mode,bool linear_mode,bool stream_mode,bool gemm_mode,int N,int M,int *rowPtr,hls::stream<ASTYPE>&  rowPtrs,hls::stream<int> &rnnz_fifo,hls::stream<int> &rnnz_fifo_sage)
{

    #pragma HLS inline off

	//int index_fifo[16384];

    bool gcn_path = !(linear_mode^sage_mode);


	if (gemm_mode==0)
	{

		//printf("N rows are %d\n", N);

		read_dataflow2(gcn_path,linear_mode,stream_mode,nnz_fea,rowPtr,rowPtrs,rnnz_fifo,rnnz_fifo_sage);


		//printf ("current index %d\n",current_index);
		//printf ("RNNZ %d\n",rnnz);
	}
	else
	{
		int rnnz;
		LOOP_A_INDEX2 : for (int A_index = 0; A_index <N ; A_index++) {
			#pragma HLS PIPELINE
				rnnz = M;
                #if (LINEAR_ONLY == 0)
				if(gcn_path==1)
				 rnnz_fifo << rnnz;
                #endif
                #if (GNN_ONLY == 0)
				if(linear_mode==1)
				 rnnz_fifo_sage << rnnz;
                #endif
		 }
	} //end else

}

/*

void readptr_coo_fea(int nnz_fea,bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

    #pragma HLS inline off
	int rnnz,current_index,next_index;


	if (gemm_mode==0)
	{
		rnnz = 0;
		current_index = 0;
		next_index=rowPtr[0];
		int A_index = 0;
		//printf("N rows are %d\n", N);
		LOOP_A_INDEX1 : while(A_index < nnz_fea) {
			#pragma HLS PIPELINE
			if(next_index == current_index)
			{
			   rnnz++;
			   next_index=rowPtr[A_index+1];
			   A_index++;
            }
            else
            {
			   rnnz_fifo << rnnz;
			   //printf ("current index %d\n",current_index);
			   //printf ("RNNZ %d\n",rnnz);
			   current_index++;
			   rnnz = 0;
			}

		}
		rnnz_fifo << rnnz;
		//printf ("current index %d\n",current_index);
		//printf ("RNNZ %d\n",rnnz);
	}
	else
	{
		LOOP_A_INDEX2 : for (int A_index = 0; A_index <N ; A_index++) {
			#pragma HLS PIPELINE
				rnnz = M;
				rnnz_fifo << rnnz;
		 }
	} //end else

}
*/
/*
void readptr_coo_fea(int nnz_fea,bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

    #pragma HLS inline off
	int rnnz,current_index,next_index;

	current_index= rowPtr[0];
	rnnz = 1;

	if (gemm_mode==0)
	{
		//printf("N rows are %d\n", N);
		LOOP_A_INDEX1 : for (int A_index = 0; A_index < nnz_fea; A_index++) {
			#pragma HLS PIPELINE
			next_index=rowPtr[A_index+1];
			if(next_index == current_index)
			{
			   rnnz++;
            }
            else
            {
			   rnnz_fifo << rnnz;
			   current_index = next_index;
			   rnnz = 1;
			}
			//printf ("BRNNZ %d\n",brnnz);
		}
	}
	else
	{
		LOOP_A_INDEX2 : for (int A_index = 0; A_index <N ; A_index++) {
			#pragma HLS PIPELINE
				rnnz = M;
				rnnz_fifo << rnnz;
		 }
	} //end else

}
*/

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
				//printf ("RNNZ %d\n",rnnz);
				rnnz_fifo << rnnz;
				//rnnz_fifo[0] << brnnz;

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


/*
void readptr_coo_adj(int nnz_fea,bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

    #pragma HLS inline off
	int rnnz,current_index,next_index;

	current_index= rowPtr[0];
	rnnz = 1;

	if (gemm_mode==0)
	{
		//printf("N rows are %d\n", N);
		LOOP_A_INDEX1 : for (int A_index = 0; A_index < nnz_fea; A_index++) {
			#pragma HLS PIPELINE
			next_index=rowPtr[A_index+1];
			if(next_index == current_index)
			{
			   rnnz++;
            }
            else
            {
			   rnnz_fifo << rnnz;
			   printf ("current index %d\n",current_index);
			   printf ("RNNZ %d\n",rnnz);
			   current_index = next_index;
			   rnnz = 1;
			}
			//printf ("BRNNZ %d\n",brnnz);
		}
	}
	else
	{
		LOOP_A_INDEX2 : for (int A_index = 0; A_index <N ; A_index++) {
			#pragma HLS PIPELINE
				rnnz = M;
				rnnz_fifo << rnnz;
		 }
	} //end else

}
*/



void readptr_coo_adj(int nnz_adj,bool linear_mode,bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

    #pragma HLS inline off

	//int index_fifo[16384];

    if(linear_mode==0)
    {
	 if (gemm_mode==0)
	 {

		//printf("N rows are %d\n", N);

		read_dataflow(nnz_adj,rowPtr,rnnz_fifo);


		//printf ("current index %d\n",current_index);
		//printf ("RNNZ %d\n",rnnz);
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

/*

void readptr_coo_adj(int nnz_adj,bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

    #pragma HLS inline off
	int rnnz,current_index,next_index;


	if (gemm_mode==0)
	{
		rnnz = 0;
		current_index = 0;
		next_index=rowPtr[0];
		int A_index = 0;
		//printf("N rows are %d\n", N);
		LOOP_A_INDEX1 : while(A_index < nnz_adj) {
			#pragma HLS PIPELINE
			if(next_index == current_index)
			{
			   rnnz++;
			   next_index=rowPtr[A_index+1];
			   A_index++;
            }
            else
            {
			   rnnz_fifo << rnnz;
			   //printf ("current index %d\n",current_index);
			   //printf ("RNNZ %d\n",rnnz);
			   current_index++;
			   rnnz = 0;
			}

		}
		rnnz_fifo << rnnz;
		//printf ("current index %d\n",current_index);
		//printf ("RNNZ %d\n",rnnz);
	}
	else
	{
		LOOP_A_INDEX2 : for (int A_index = 0; A_index <N ; A_index++) {
			#pragma HLS PIPELINE
				rnnz = M;
				rnnz_fifo << rnnz;
		 }
	} //end else

}
*/


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

            #if (INT_QUANT_A == 1)
	            quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
            #else
	            value_temp2 = value_temp;
            #endif

			A_fifo <<  value_temp2;
			//std::cout << "ADJ_fifo at " << j << " " << value_temp << " " << values[j] << std::endl;
			col_indices_fifo << columnIndex[j];
			//std::cout << "col indices " << columnIndex[j] << std::endl;
		  }
		}
		else
		{
				int c=0;
				LOOP_J_SPMM2 : for (int j = 0; j < last_index; j++) {
				#pragma HLS PIPELINE
					//if(A_fifo.full())
					//	fifo_full_0++;
					 //if(A_fifo.empty())
						//fifo_empty_0++;

					//if(col_indices_fifo.full())
					//	fifo_full_1++;
					//if(col_indices_fifo.empty())
						//fifo_empty_1++;

				   INTYPE value_temp;
				   ATYPE value_temp2;


				   value_temp = (INTYPE)values[j];

                   #if (INT_QUANT_A == 1)
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
					//std::cout << "A_fifo " << values[j] << std::endl;
					//std::cout << "col index fea " << c << std::endl;
				}
		}

}

void readval_coo_adj(int beta_qu,int f_align,float quantization_scale_fea,bool linear_mode,bool gemm_mode,int ccount,int last_index,hls::stream<ATYPE> &A_fifo,hls::stream<int> &col_indices_fifo,INTYPE *values,int *columnIndex)
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

            #if (INT_QUANT_A == 1)
	            quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
            #else
	            value_temp2 = value_temp;
            #endif

			A_fifo <<  value_temp2;
			//std::cout << "ADJ_fifo at " << j << " " << value_temp << " " << values[j] << std::endl;
			col_indices_fifo << columnIndex[j];
			//std::cout << "col indices " << columnIndex[j] << std::endl;
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

                   #if (INT_QUANT_A == 1)
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
					//std::cout << "A_fifo " << values[j] << std::endl;
					//std::cout << "col index fea " << c << std::endl;
				}
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

	        #if (INT_QUANT_A == 1)
		        quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
	        #else
		        value_temp2 = value_temp;
	        #endif


			A_fifo <<   (ITYPE)value_temp2;
			//std::cout << "ADJ2 fifo at " << j << " " << value_temp << " " << values[j] << std::endl;
			col_indices_fifo << columnIndex[j];
			//std::cout << "col indices " << columnIndex[j] << std::endl;
		 }
   	   }
	   else
	   {
				int c=0;
				LOOP_J_SPMM2 : for (int j = 0; j < last_index; j++) {
				#pragma HLS PIPELINE
					//if(A_fifo.full())
					//	fifo_full_0++;
					 //if(A_fifo.empty())
						//fifo_empty_0++;

					//if(col_indices_fifo.full())
					//	fifo_full_1++;
					//if(col_indices_fifo.empty())
						//fifo_empty_1++;

					INTYPE value_temp;
					ATYPE value_temp2;
					value_temp = (INTYPE)values[j];

			        #if (INT_QUANT_A == 1)
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
					//std::cout << "A_fifo " << values[j] << std::endl;
					//std::cout << "col index fea " << c << std::endl;
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
			//std::cout << "ADJ2 fifo at " << j << " " << value_temp << " " << values[j] << std::endl;
			col_indices_fifo << columnIndex[j];
            #endif
			//std::cout << "col indices " << columnIndex[j] << std::endl;
		 }
   	   }
	   else
	   {
				int c=0;
				LOOP_J_SPMM2 : for (int j = 0; j < last_index; j++) {
				#pragma HLS PIPELINE
					//if(A_fifo.full())
					//	fifo_full_0++;
					 //if(A_fifo.empty())
						//fifo_empty_0++;

					//if(col_indices_fifo.full())
					//	fifo_full_1++;
					//if(col_indices_fifo.empty())
						//fifo_empty_1++;

					INTYPE value_temp;
					ATYPE value_temp2;
					value_temp = (INTYPE)values[j];

			        #if (INT_QUANT_A == 1)
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
					//std::cout << "A_fifo " << values[j] << std::endl;
					//std::cout << "col index fea " << c << std::endl;
				}
	      }
	   }

}

/*void readval_fea(int last_index,hls::stream<bool> &exit_loop,hls::stream<FTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,FTYPE *values,int *columnIndex)
{

		#pragma HLS inline off
	    exit_loop << 0;
		LOOP_J_SPMM : for (int j = 0; j < last_index; j++) {
			#pragma HLS PIPELINE

			A_fifo <<  values[j];
			////std::cout << "A_fifo " << values[j] << std::endl;
			col_indices_fifo << columnIndex[j];
			////std::cout << "col index fea " << columnIndex[j] << std::endl;
		}
		exit_loop << 1;

}
*/
/*
void readval_fea(int last_index,hls::stream<FTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,FTYPE *values,int *columnIndex)
{

		#pragma HLS inline off
	    int j=0;
		LOOP_J_SPMM : while(j < last_index) {
			#pragma HLS PIPELINE
			if(!A_fifo.full() && !col_indices_fifo.full())
			{
				A_fifo << values[j];
				////std::cout << "A_fifo " << values[j] << std::endl;
			    col_indices_fifo << columnIndex[j];
				j++;
			}
			else
			{
				fifo_full_0;
			}

			////std::cout << "col index fea " << columnIndex[j] << std::endl;
		}

}
*/


void readval_coo_fea(int beta_qu,int f_align,int beta_qul,int f_alignl,
		float quantization_scale_fea[5],float quantization_scale_lin[5],bool sage_mode,bool linear_mode,bool stream_mode,bool gemm_mode,int ccount,int last_index,
		hls::stream<FTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,
		hls::stream<LTYPE> &A_fifo_sage,hls::stream<int> &col_indices_fifo_sage,
		INTYPE *values,hls::stream<ASTYPE>&  valuess,int *columnIndex,hls::stream<ASTYPE>&  columnIndex_feas, int B_index)
{

	#pragma HLS inline off

	std::cout << "gemm mode " <<  gemm_mode << std::endl;
	std::cout << "stream mode " <<  stream_mode << std::endl;
	std::cout << "read count " <<  last_index << std::endl;

    bool gcn_path = !(linear_mode^sage_mode);

	if (gemm_mode==0)
	{

    	fp_int C_float_int;


      	if(stream_mode==1)
      	{

          //LOOP_J_SPMM11 : for (int j = 0; j < last_index; j++) {
  	  	  //LOOP_J_SPMM11 : while (c!=0xFFFFFFFF) {
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

             //std::cout << "value " << value_temp << " " << "column "  << c << std::endl;

            #if (INT_QUANT_F == 1)
  			   quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
  			   quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
            #else
  			   value_temp2 = value_temp;
  			   value_temp3 = value_temp;
            #endif


            #if (LINEAR_ONLY == 0)

		      if(gcn_path)
			  {

                A_fifo << value_temp2;

		         //std::cout << "A_fifo at " << j << " " << value_temp << " " << values[j] << std::endl;
		        col_indices_fifo << c;
			  }
		   #endif

              #if (GNN_ONLY == 0)

		      if(linear_mode)
		      {
		       col_indices_fifo_sage << c;
		       A_fifo_sage << value_temp3;
		      }
              #endif


  		    //std::cout << "col index fea " << columnIndex[j] << std::endl;
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

              //std::cout << "values j " << values[j] << std::endl;
			  value_temp = (INTYPE)values[j];
			  col_temp = columnIndex[j];

			  //std::cout << "values temp " << value_temp << std::endl;
			//std::cout << "here " <<  std::endl;


            #if (INT_QUANT_F == 1)
			  quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
			  quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
            #else
			  value_temp2 = value_temp;
			  value_temp3 = value_temp;
            #endif



           #if (LINEAR_ONLY == 0)
		    if(gcn_path)
			{
			  A_fifo << value_temp2;
			  //std::cout << "A_fifo at " << j << " " << value_temp << " " << values[j] << std::endl;
			  col_indices_fifo << col_temp;
			}
			#endif

            #if (GNN_ONLY == 0)

		    if(linear_mode)
		    {
			 col_indices_fifo_sage << col_temp;
			 A_fifo_sage << value_temp3;
		    }

            #endif
		    //std::cout << "col index fea " << columnIndex[j] << std::endl;
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
			//LOOP_J_SPMM21 : do{

			#pragma HLS PIPELINE
				   INTYPE value_temp;
				   FTYPE value_temp2;
				   LTYPE value_temp3;



				    //std::cout << "Read out stream " << std::endl;

					 //if (valuess.empty()) // execute only when producer has already generated some meaningful data
					 //    return;

				    ASTYPE temp = valuess.read();

               	    C_float_int.i = temp.data;

			        value_temp = (INTYPE)C_float_int.f;


					last_index1=temp.last;

					//std::cout << "last index " << last_index1 << std::endl;

			        //std::cout << "value is " << value_temp << " at " << j << std::endl;

                   #if (INT_QUANT_F == 1)
				     quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
					 quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
                   #else
				     value_temp2 = value_temp;
				     value_temp3 = value_temp;
	                #endif

                #if (LINEAR_ONLY == 0)
				 if(gcn_path)
				 {
				   A_fifo <<  value_temp2;
				   col_indices_fifo << c;
				 }
				#endif
                #if (GNN_ONLY == 0)
				if(linear_mode)
				{
				 col_indices_fifo_sage << c;
				 A_fifo_sage <<  value_temp3;
				}
				#endif
				if (c == (ccount-1)) //column count
					c=0;
				else
					c++;
				//std::cout << "A_fifo " << values[j] << std::endl;
				//std::cout << "col index fea " << c << std::endl;
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


                   #if (INT_QUANT_F == 1)
				     quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
					 quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
                   #else
				     value_temp2 = value_temp;
	                #endif

                #if (LINEAR_ONLY == 0)
				 if(gcn_path)
				 {
				  A_fifo <<  value_temp2;
				  col_indices_fifo << c;
				 }
                #endif
                #if (GNN_ONLY == 0)
				if(linear_mode)
				{
				 col_indices_fifo_sage << c;
				 A_fifo_sage <<  value_temp3;
				}
				#endif

				if (c == (ccount-1)) //column count
					c=0;
				else
					c++;
				//std::cout << "A_fifo " << values[j] << std::endl;
				//std::cout << "col index fea " << c << std::endl;
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
	//bool fifo_active = 1;
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
						//fifo_write_0++;
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
					//fifo_empty_0++;
					//fifo_read_0++;
				}
			}
			else //data_buffer not empty
			{
				if (A_fifo_out.write_nb(data_buffer) == 1)
				{
					fifo_write_0++;
					if(A_fifo.read_nb(data_buffer) == 0)
					{
						//fifo_empty_0++;
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
					//fifo_write_0++;
					fifo_full_0++;
				}
			}
	}


}

/*
void check_fifo_0(hls::stream<bool> &exit_loop, hls::stream<ITYPE> &A_fifo, hls::stream<ITYPE> &A_fifo_out)
{
	ITYPE data_buffer;
	//int data_count=0;
	bool loop_done = 0;
	bool data_in_buffer = 0; //data exits in data_buffer
	//bool fifo_active = 1;
	while(loop_done == 0 || data_in_buffer == 1)
	{
		#pragma HLS PIPELINE
			fifo_cycle_0++;
  	       	if (data_in_buffer == 0) //data_buffer empty
			{
				if(A_fifo.read_nb(data_buffer) == 1)
				{
					fifo_read_0++;
					if(A_fifo_out.write_nb(data_buffer) == 0)
					{
						//fifo_write_0++;
						fifo_full_0++;
						data_in_buffer = 1; //fifo full and data stored in data_in_buffer
					}
					else
					{
						//data_count++;
						fifo_write_0++;
					}
				}
				else
				{
					//fifo_empty_0++;
					//fifo_read_0++;
				}
			}
			else //data_buffer not empty
			{
				if (A_fifo_out.write_nb(data_buffer) == 1)
				{
					fifo_write_0++;
					if(A_fifo.read_nb(data_buffer) == 0)
					{
						//fifo_empty_0++;
						data_in_buffer = 0; //data_buffer empty
					}
					else
					{
						fifo_read_0++;
					}
					//data_count++;
				}
				else
				{
					//fifo_write_0++;
					fifo_full_0++;
				}
			}
  	    bool exit_data;
		if(exit_loop.read_nb(exit_data) == 1)
				loop_done == 1;

	}


}
*/
void check_fifo_2(int N, hls::stream<ITYPE> &C_fifo, hls::stream<ITYPE> &C_fifo_out)
{
	ITYPE data_buffer;
	int data_count=0;
	bool data_in_buffer= 0; //data exits in data_buffer


	while(data_count < N)
	{
		#pragma HLS PIPELINE
		//LOOP_CHECK_2 : for (int j = 0; j < B_WIDTH_BLOCK; j++)
		//{
		   //if (j < B_WIDTH_INT)
		   //{
			fifo_cycle_2++;
  	       		if (data_in_buffer == 0) //data_buffer empty
			{
				if(C_fifo.read_nb(data_buffer) == 1)
				{

					fifo_read_2++;
					if(C_fifo_out.write_nb(data_buffer) == 0)
					{
						//fifo_write_2++;
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
					//fifo_read_2++;
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
					//fifo_write_2++;
					fifo_full_2++;
				}
			}
		   //} // j < B_WIDTH_INT
		//} //LOOP_CHECK_2
	} //while

}



void check_fifo_1(int N, int B_index, int B_index_loop, int tail, hls::stream<ITYPE> &C_fifo, hls::stream<ITYPE> &C_fifo_out)
{
	ITYPE data_buffer;
	int data_count=0;
	bool data_in_buffer= 0; //data exits in data_buffer

       /*int B_WIDTH_INT;

       if (B_index < (B_index_loop-1))
 		B_WIDTH_INT = B_WIDTH_BLOCK;
       else
		B_WIDTH_INT = tail;*/

	//while(data_count < N*B_WIDTH_INT)
	while(data_count < N)
	{
		#pragma HLS PIPELINE
		//LOOP_CHECK_2 : for (int j = 0; j < B_WIDTH_BLOCK; j++)
		//{
		   //if (j < B_WIDTH_INT)
		   //{
			fifo_cycle_1++;
  	       		if (data_in_buffer == 0) //data_buffer empty
			{
				if(C_fifo.read_nb(data_buffer) == 1)
				{

					fifo_read_1++;
					if(C_fifo_out.write_nb(data_buffer) == 0)
					{
						//fifo_write_1++;
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
					//fifo_empty_1++;
					//fifo_read_1++;
				}

			}
			else //data_buffer not empty
			{
				if (C_fifo_out.write_nb(data_buffer) == 1)
				{
					fifo_write_1++;
					if(C_fifo.read_nb(data_buffer) == 0)
					{
						//fifo_empty_1++;
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
					//fifo_write_1++;
					fifo_full_1++;
				}
			}
		   //} // j < B_WIDTH_INT
		//} //LOOP_CHECK_2
	} //while

}




/*
void quant1(hls::stream<INTYPE> &A_fifo_fea_in,hls::stream<int> &col_indices_fifo_fea, hls::stream<int> rnnz_fifo_fea[SPMM_BLOCK],
hls::stream<FTYPE> &A_fifo_fea_out,hls::stream<int> &col_indices_fifo_fea_out, hls::stream<int> rnnz_fifo_fea_out[SPMM_BLOCK],
int last_index,float quantization_scale)
{


	LOOP_QUANT : for (int j = 0; j < last_index; j++)
	{
	    #pragma HLS PIPELINE
		INTYPE temp = A_fifo_fea_in.read();
        //std::cout << "QUANT in " << temp << std::endl;
	    INTYPE vfloat = quantization_scale*temp+zero_point;
		INTYPE vround = hls::round(vfloat);

        //std::cout << "VROUND " << vround << std::endl

        //clippping
		if (vround>beta_qu)
			vround = beta_qu;
		else if (vround<alpha_qu)
			vround = alpha_qu;
		ITYPE vquant = ITYPE(vround);

        //std::cout << "FQUANT " << vquant << std::endl;
		ITYPE vnorm = vquant >> (qbits-1);
 		FTYPE fval = FTYPE(vnorm);

        //std::cout << "FNORM " << fval << std::endl;

		A_fifo_fea_out << fval;

	}



}
*/




void reada1_coo(int nnz_fea,int	beta_qu,int f_align,int	beta_qul,int f_alignl,float quantization_scale_fea[5],float quantization_scale_lin[5],
		int &last_index,ap_uint<1> model[5][8],int M, int first_row, int row_count,
		hls::stream<FTYPE> &A_fifo_fea,hls::stream<int> &col_indices_fifo_fea, hls::stream<int> &rnnz_fifo_fea,
		hls::stream<LTYPE> &A_fifo_fea_sage,hls::stream<int> &col_indices_fifo_fea_sage, hls::stream<int> &rnnz_fifo_fea_sage,
int *rowPtr_fea,int *columnIndex_fea,INTYPE *values_fea,
hls::stream<ASTYPE>&  rowPtr_feas,hls::stream<ASTYPE>&  columnIndex_feas,hls::stream<ASTYPE>&  values_feas,
int B_index, int layer_loop)
{

	int last_index_fea;
    bool gemm_mode,stream_mode,linear_mode,sage_mode;
    int M_int;

    gemm_mode = model[B_index][1];
    stream_mode = model[B_index][3];
	linear_mode = model[B_index][6];
	sage_mode = model[B_index][7];

    if (B_index == 0) //first layer
	      M_int = M;
	else
	      M_int = B_WIDTH_BLOCK; //in hidden layers the input width (number of features) is the B_WDITH BLOCK

	if (gemm_mode==0)
	{
		//std::cout << "Thread fea is processing non-zeros " << last_index_fea << "from address " << values_fea << std::endl;
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

	readptr_coo_fea(nnz_fea,sage_mode,linear_mode,stream_mode,gemm_mode,row_count,M_int,rowPtr_fea,rowPtr_feas,rnnz_fifo_fea,rnnz_fifo_fea_sage);
	readval_coo_fea(beta_qu,f_align,beta_qul,f_alignl,quantization_scale_fea,quantization_scale_lin,sage_mode,linear_mode,stream_mode,gemm_mode,M_int,last_index_fea,
			A_fifo_fea,col_indices_fifo_fea,
			A_fifo_fea_sage,col_indices_fifo_fea_sage,
			values_fea,values_feas,columnIndex_fea,columnIndex_feas,B_index);

}


void reada2_csr(int beta_qu,int f_align,float quantization_scale_adj,bool gemm_mode,int M,int first_row, int row_count,  hls::stream<ATYPE> &A_fifo_adj,hls::stream<int> &col_indices_fifo_adj, hls::stream<int> &rnnz_fifo_adj_total_e, hls::stream<int> &rnnz_fifo_adj_total_s,hls::stream<int> &rnnz_fifo_adj,
int *rowPtr_adj,int *columnIndex_adj,INTYPE *values_adj)
{
      //hls::stream<DTYPE>       A_accel;
      //#pragma HLS STREAM variable=A_accel depth=A_WIDTH_FIFO dim=1

      //feature sparse matrix



	int last_index_adj;

	if (gemm_mode==0)
	{

	 last_index_adj=rowPtr_adj[first_row+row_count]-rowPtr_adj[first_row];

	 //std::cout << "Thread adj is processing non-zeros " << last_index_adj << "from address " << values_adj << std::endl;
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

	 //columnIndex_adj += first_row;
	 //values_adj += first_row;


	 //last_index_fea=rowPtr_fea[N_fea];

	 //std::cout << "columnIndex_adj " << columnIndex_adj[0] << " first_row " << first_row << std::endl;

	 //last_index=N*M;

    	//for (int B_index = 0; B_index < B_index_loop; B_index++)
	 //{

        //feature sparse matrix

	 //readptr_fea(N_fea,rowPtr_fea,rnnz_fifo_fea);
	 //readval_fea(last_index_fea,A_fifo_fea,col_indices_fifo_fea,values_fea,columnIndex_fea);


     //adjacency sparse matrix

        ////////std::cout << "last index adj " << last_index_adj << ////std::endl;



	readptr_csr_adj(gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);

	readval_csr_adj(beta_qu,f_align,quantization_scale_adj,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);

    	//}
}


void reada2_coo(int nnz_adj,int beta_qu,int f_align,float quantization_scale_adj,ap_uint<1> model[5][8],int M,int first_row, int row_count,  hls::stream<ATYPE> &A_fifo_adj,hls::stream<int> &col_indices_fifo_adj, hls::stream<int> &rnnz_fifo_adj_total_e, hls::stream<int> &rnnz_fifo_adj_total_s,hls::stream<int> &rnnz_fifo_adj,
int *rowPtr_adj,int *columnIndex_adj,INTYPE *values_adj,int B_index)
{
      //hls::stream<DTYPE>       A_accel;
      //#pragma HLS STREAM variable=A_accel depth=A_WIDTH_FIFO dim=1

      //feature sparse matrix

	bool gemm_mode;

	gemm_mode = model[B_index][0];

	bool linear_mode;

	linear_mode =  model[B_index][6];

	bool gat_mode;

	gat_mode = model[B_index][5];

	int last_index_adj;

	if (gemm_mode==0)
	{


	 //std::cout << "Thread adj is processing non-zeros " << last_index_adj << "from address " << values_adj << std::endl;
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


	if(gat_mode==1)
	{
	 rnnz_fifo_adj_total_e << nnz_adj;
	 rnnz_fifo_adj_total_s << nnz_adj;
	}

	readptr_coo_adj(nnz_adj,linear_mode,gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);

	readval_coo_adj(beta_qu,f_align,quantization_scale_adj,linear_mode,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);


    	//}
}

//CHECK THIS A_FIFO ITYPE should be ATYPE


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


	   //std::cout << "Thread adj is processing non-zeros " << last_index_adj << "from address " << values_adj << std::endl;
	   columnIndex_adj += rowPtr_adj[first_row];
	   values_adj += rowPtr_adj[first_row];
	   rowPtr_adj += first_row;
	   last_index_adj = nnz_adj;
	   //columnIndex_adj += first_row;
	   //values_adj += first_row;
	}
	else
	{
		values_adj+=first_row*M;
		last_index_adj = row_count*M;
	}

	  //last_index_fea=rowPtr_fea[N_fea];

	//std::cout << "columnIndex_adj " << columnIndex_adj[0] << " first_row " << first_row << std::endl;

	//last_index=N*M;

    	//for (int B_index = 0; B_index < B_index_loop; B_index++)
	//{

        //feature sparse matrix

	//readptr_fea(N_fea,rowPtr_fea,rnnz_fifo_fea);
	//readval_fea(last_index_fea,A_fifo_fea,col_indices_fifo_fea,values_fea,columnIndex_fea);


     //adjacency sparse matrix

        ////////std::cout << "last index adj " << last_index_adj << ////std::endl;

	readptr_coo_adj(nnz_adj,linear_mode,gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);
	readval_coo_adj2(beta_qu,f_align,quantization_scale_adj,linear_mode,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);

    	//}
}


void reada22_csr(int beta_qu,int f_align,float quantization_scale_adj,bool gemm_mode,int M,int first_row, int row_count, hls::stream<ITYPE> &A_fifo_adj,hls::stream<int> &col_indices_fifo_adj, hls::stream<int> &rnnz_fifo_adj,
int *rowPtr_adj,int *columnIndex_adj,INTYPE *values_adj)
{
      //hls::stream<DTYPE>       A_accel;
      //#pragma HLS STREAM variable=A_accel depth=A_WIDTH_FIFO dim=1

        //feature sparse matrix




	//int last_index_fea;
	int last_index_adj;

	if (gemm_mode==0)
	{


       last_index_adj=rowPtr_adj[first_row+row_count]-rowPtr_adj[first_row];
	   //std::cout << "Thread adj is processing non-zeros " << last_index_adj << "from address " << values_adj << std::endl;
	   columnIndex_adj += rowPtr_adj[first_row];
	   values_adj += rowPtr_adj[first_row];
	   rowPtr_adj += first_row;
	   //columnIndex_adj += first_row;
	   //values_adj += first_row;
	}
	else
	{
		last_index_adj=row_count*M;
		values_adj+=first_row*M;
	}

	  //last_index_fea=rowPtr_fea[N_fea];

	//std::cout << "columnIndex_adj " << columnIndex_adj[0] << " first_row " << first_row << std::endl;

	//last_index=N*M;

    	//for (int B_index = 0; B_index < B_index_loop; B_index++)
	//{

        //feature sparse matrix

	//readptr_fea(N_fea,rowPtr_fea,rnnz_fifo_fea);
	//readval_fea(last_index_fea,A_fifo_fea,col_indices_fifo_fea,values_fea,columnIndex_fea);


     //adjacency sparse matrix

        ////////std::cout << "last index adj " << last_index_adj << ////std::endl;

	readptr_csr_adj(gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);
	readval_csr_adj2(beta_qu,f_align,quantization_scale_adj,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);

    	//}
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
		  //print

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


			////////std::cout << " compute1 read col fifo" << ////std::endl;



			dsp_kernel_float_adj_4(block_size,v,b_block1,b_block2,b_block3,b_block4,ci,zero_point_lhs,zero_point_rhs,acc_float);


		        for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			  	//#pragma HLS UNROLL
				//#pragma HLS dependence variable=acc_val inter false
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

	//for (int j = 0; j < B_WIDTH_BLOCK; j++) {

	//	#pragma HLS UNROLL

	//		acc2[j] = 0;
	//}



	 DSP_LOOP_SPMM: for (int i = 0; i < M; i++) {
		 	 #pragma HLS PIPELINE
        	 //#pragma HLS UNROLL factor=PARALLEL_ROW
			DTYPE v = A_fifo.read();

			int ci = col_indices_fifo.read();

			dsp_kernel_int_adj_4(block_size,v,b_block1,b_block2,
					b_block3,b_block4,
					ci,zero_point_lhs,zero_point_rhs,acc);



			for (int j = 0; j < B_WIDTH_BLOCK; j++) {

				#pragma HLS UNROLL
				acc2[j] += acc[j];
						//////std::cout << " compute2 acc with j " << j << "acc2[j] is " << acc2[j] << ////std::endl;
			}//j loop

				//////////std::cout << " compute1 acc with j " << j << "acc2[j] is " << acc2[j] << ////std::endl;


	     	} //i loop





	#endif

}

void dsp_kernel_wrapper_adj_2(int block_size,int M[SPMM_BLOCK],hls::stream<ITYPE> &A_fifo,hls::stream<int> &col_indices_fifo,QTYPE b_block1[B_HEIGHT/4][B_WIDTH_BLOCK],
		QTYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],
		//ITYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],
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
         //print


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


				////////std::cout << " compute1 read col fifo" << ////std::endl;



				dsp_kernel_float_adj_2(block_size,v,b_block1,b_block2,ci,zero_point_lhs,zero_point_rhs,acc_float);


			        for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			          	for (int z = 0; z < SPMM_BLOCK; z++)
			          	{
			    		    #pragma HLS UNROLL
			    			if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
			    			  			acc_part[i][j][z] += acc_float[j];
			    			}//#pragma HLS UNROLL
					//#pragma HLS dependence variable=acc_val inter false
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

		//for (int j = 0; j < B_WIDTH_BLOCK; j++) {

		//	#pragma HLS UNROLL

		//		acc2[j] = 0;
		//}


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
	        	 //#pragma HLS UNROLL factor=PARALLEL_ROW
    				DTYPE v = A_fifo.read();

				int ci = col_indices_fifo.read();

				dsp_kernel_int_adj_2(block_size,v,b_block1,b_block2,
						//b_block3,b_block4,
						ci,zero_point_lhs,zero_point_rhs,acc);



				for (int j = 0; j < B_WIDTH_BLOCK; j++) {

					#pragma HLS UNROLL
					for (int z = 0; z < SPMM_BLOCK; z++)
					{
							#pragma HLS UNROLL
							if (i>=M_aux[z]&&i<M_aux[z+1])
									acc2[j][z] += acc[j];
					}//z loop
							//////std::cout << " compute2 acc with j " << j << "acc2[j] is " << acc2[j] << ////std::endl;
				}//j loop

					//////////std::cout << " compute1 acc with j " << j << "acc2[j] is " << acc2[j] << ////std::endl;


		     	} //i loop





		#endif

}

void dsp_kernel_wrapper_adj_1(int block_size,int M,hls::stream<TTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,QTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],
		//ITYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],
		//ITYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],
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
        //print

   		DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_ADJ) {
			#pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

			DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_ADJ; i++) {

				DTYPE v;
				int ci;
				if ((k+i) < BM) //avoid trying to read empty FIFO that only contains M elements
				{

					v = A_fifo.read();
					ci = col_indices_fifo.read();
					//std::cout << "Read A and col fifo (adj) " << v << " " << ci << std::endl;
				}
			        else
				{
					v=0;
					ci=0;
				}


				////////std::cout << " compute1 read col fifo" << ////std::endl;


				dsp_kernel_float_adj_1(v,b_block1,ci,zero_point_lhs,zero_point_rhs,acc_float);

			        for (int j = 0; j < B_WIDTH_BLOCK; j++) {
				  	//#pragma HLS UNROLL
					//#pragma HLS dependence variable=acc_val inter false
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

		//for (int j = 0; j < B_WIDTH_BLOCK; j++) {

		//	#pragma HLS UNROLL

		//		acc2[j] = 0;
		//}




		 DSP_LOOP_SPMM: for (int i = 0; i < M; i++) {
 		 	 #pragma HLS PIPELINE
	        	 //#pragma HLS UNROLL factor=PARALLEL_ROW
    				TTYPE v = A_fifo.read();

    				//std::cout << " ADJ value is " << v << std::endl;

				int ci = col_indices_fifo.read();

				dsp_kernel_int_adj_1(block_size,v,b_block1,//b_block2,
						//b_block3,b_block4,
						ci,zero_point_lhs,zero_point_rhs,acc);



				for (int j = 0; j < B_WIDTH_BLOCK; j++) {
						acc2[j] += acc[j];
							//std::cout << " compute2 acc with j " << j << "acc2[j] is " << acc2[j][0] << std::endl;
				}//j loop

					//std::cout << " compute2 acc with j " << j << "acc2[j] is " << acc2[j][z] << std::endl;


		  } //i loop



		#endif

}


void dsp_kernel_wrapper_fea(bool gemm_mode,int M[SPMM_BLOCK],hls::stream<FTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK])
{



#if defined FLOAT || defined HALF

	//#pragma HLS INLINE

		ITYPE acc_part[FADD_LATENCY_FEA][B_WIDTH_BLOCK][SPMM_BLOCK];
		//#pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0
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
	         //print

	   		DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_FEA) {
				//#pragma HLS PIPELINE II=FADD_LATENCY rewind
	   			#pragma HLS PIPELINE II=FADD_LATENCY_FEA

				DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_FEA; i++) {
					DTYPE v;
					int ci;
					if ((k+i) < BM) //avoid trying to read empty FIFO that only contains BM elements
					{
						v = A_fifo.read();
						//if (gemm_mode==0)
							ci = col_indices_fifo.read();
						//else
						//	ci = k+i;
					}
				        else
					{
						v=0;
						ci=0;
					}

					dsp_kernel_float_fea(v,b_block,ci,zero_point_lhs,zero_point_rhs,acc_float);

					SPMM_BLOCK_LOOP1 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
					  	#pragma HLS UNROLL
						//#pragma HLS dependence variable=acc_val inter false
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

		//LOOP_ACC2_IN: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
        //    #pragma HLS UNROLL
		//	for (int i = 0; i < SPMM_BLOCK; i++) {
		//		acc2[j][i] = 0;
		//	}
		//}

         int BM = M[SPMM_BLOCK-1];

		 int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
		 M_aux[0] = 0;
	     for (int j = 1; j < SPMM_BLOCK+1; j++)
		 {
			#pragma HLS UNROLL
			M_aux[j] = M[j-1];
		 }

         //printf("BM is %d\n",BM);

		 DSP_LOOP_SPMM: for (int i = 0; i < BM; i++) {
 		 	 #pragma HLS PIPELINE
	        	 //#pragma HLS UNROLL factor=PARALLEL_ROW

			    //std::cout << "A fifo/col indices " << std::endl;

    			FTYPE v = A_fifo.read();

    		    //std::cout << "V " << v << std::endl;

    			int ci;
    			//if (gemm_mode==0)
    			ci = col_indices_fifo.read();
    			//else
    			//	ci = i;



			    //std::cout << "Done " << std::endl;

				dsp_kernel_int_fea(v,b_block,ci,zero_point_lhs,zero_point_rhs,acc);




				for (int j = 0; j < B_WIDTH_BLOCK; j++) {

					#pragma HLS UNROLL
					for (int z = 0; z < SPMM_BLOCK; z++)
					{
						//critical #pragma HLS UNROLL
                        #pragma HLS UNROLL
                        //#pragma HLS PIPELINE II=1
						if (i>=M_aux[z]&&i<M_aux[z+1])
						 acc2[j][z] += acc[j];
					}//z loop

				 }//j loop

		     	} //i loop




		#endif

}


void dsp_kernel_wrapper_lin(bool gemm_mode,int M,hls::stream<LTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK])
{



#if defined FLOAT || defined HALF

	//#pragma HLS INLINE

		ITYPE acc_part[FADD_LATENCY_FEA][B_WIDTH_BLOCK][SPMM_BLOCK];
		//#pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0
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
	         //print

	   		DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_FEA) {
				//#pragma HLS PIPELINE II=FADD_LATENCY rewind
	   			#pragma HLS PIPELINE II=FADD_LATENCY_FEA

				DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_FEA; i++) {
					DTYPE v;
					int ci;
					if ((k+i) < BM) //avoid trying to read empty FIFO that only contains BM elements
					{
						v = A_fifo.read();
						//if (gemm_mode==0)
							ci = col_indices_fifo.read();
						//else
						//	ci = k+i;
					}
				        else
					{
						v=0;
						ci=0;
					}

					dsp_kernel_float_fea(v,b_block,ci,zero_point_lhs,zero_point_rhs,acc_float);

					SPMM_BLOCK_LOOP1 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
					  	#pragma HLS UNROLL
						//#pragma HLS dependence variable=acc_val inter false
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

		//LOOP_ACC2_IN: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
        //    #pragma HLS UNROLL
		//	for (int i = 0; i < SPMM_BLOCK; i++) {
		//		acc2[j][i] = 0;
		//	}
		//}



         //printf("BM is %d\n",BM);

		 DSP_LOOP_SPMM: for (int i = 0; i < M; i++) {
 		 	 #pragma HLS PIPELINE
	        	 //#pragma HLS UNROLL factor=PARALLEL_ROW

			    //std::cout << "A fifo/col indices " << std::endl;

    			LTYPE v = A_fifo.read();

    		    //std::cout << "V " << v << std::endl;

    			int ci;
    			//if (gemm_mode==0)
    				ci = col_indices_fifo.read();
    			//else
    			//	ci = i;



			    //std::cout << "Done " << std::endl;

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
			        ////////////std::cout << "Float scale " << ////std::endl;
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
 							//////////std::cout << "C fifo float " << C_fifo_float << "bias_val_float " << bias_val_float << "zero_point_dst_float " << zero_point_dst_float << ////std::endl;
							//////////std::cout << "clamp max float " << clamp_max_float << "clamp min float " << clamp_min_float << ////std::endl;
							FTYPE C_temp_float = C_fifo_float + bias_val_float + zero_point_dst_float;
                         				if (C_temp_float < clamp_min_float) C_temp_float = clamp_min_float;
							if (C_temp_float > clamp_max_float) C_temp_float = clamp_max_float;
							////////////std::cout << "c temp float 2 " << C_temp_float << ////std::endl;
							DTYPE C_out = *(int*)&C_temp_float;
							//DTYPE C_out = (int)C_temp_float;
							write_fifo[j] << C_out;
						#else
							DTYPE C_fifo_int = C_fifo[j].read();
							////////std::cout << "C fifo float" << *(FTYPE*)&C_fifo_int << ////std::endl;
					        	write_fifo[j] << C_fifo_int;
						#endif

					}

				 }
			     }
			#endif


			#if defined EIGHTBIT
			    LOOP_CH1:    for (int i = 0; i < N; i+=4) {
				//#pragma HLS UNROLL factor=BLOCK/32
				//#pragma HLS UNROLL factor=2
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
					//#pragma HLS UNROLL factor=2
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


	//DTYPE A_accel[A_WIDTH];
	        //#pragma HLS array_partition variable=A_accel cyclic factor=


		ITYPE acc[B_WIDTH_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc complete


		ITYPE acc2[B_WIDTH_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0


		//hls::stream<int>             col_indices_fifo;
		//#pragma HLS STREAM variable=col_indices_fifo depth=1024 dim=1

		//int col_indices[A_WIDTH];


	      int B_WIDTH_INT;
	      ITYPE C_fifo_val;

	      //if (B_index < (B_index_loop-1))
		  B_WIDTH_INT = B_WIDTH_BLOCK;
	      //else
			//B_WIDTH_INT = tail;



	        //#pragma HLS DATAFLOW


		for (int A_index = 0; A_index < row_count; A_index++) {


			//computing

			LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
				#pragma HLS UNROLL
					acc2[j] = 0;
				}






			int rnnz;
			rnnz = rnnz_fifo.read();

			//printf("crows %d\n",crows);

	         //std::cout << "The rnnz value is " << rnnz << std::endl;

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
		//ITYPE B_accel3[B_HEIGHT/4][B_WIDTH_BLOCK],ITYPE B_accel4[B_HEIGHT/4][B_WIDTH_BLOCK],
hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK][SPMM_BLOCK],int B_index, int B_index_loop, int tail)
{


	//DTYPE A_accel[A_WIDTH];
	        //#pragma HLS array_partition variable=A_accel cyclic factor=


		ITYPE acc[B_WIDTH_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc complete


		ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0


		//hls::stream<int>             col_indices_fifo;
		//#pragma HLS STREAM variable=col_indices_fifo depth=1024 dim=1

		//int col_indices[A_WIDTH];


	      int B_WIDTH_INT;

	      if (B_index < (B_index_loop-1))
			B_WIDTH_INT = B_WIDTH_BLOCK;
	      else
			B_WIDTH_INT = tail;



	        //#pragma HLS DATAFLOW


		for (int A_index = 0; A_index < row_count; A_index+=SPMM_BLOCK) {


			//computing

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
			//printf("crows %d\n",crows);

	         //std::cout << "The rnnz value is " << rnnz << std::endl;

			dsp_kernel_wrapper_adj_2(block_size,rnnz,A_fifo,col_indices_fifo,B_accel1,
					B_accel2,
					//B_accel3,B_accel4,
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
						////std::cout << "C_fifo " << acc2[j] << std::endl;

						//C_fifo[A_index][j]=acc2[j];
					}
			}


	          } // A_index loop


}

/*void relupipe(bool relu,int row_count,hls::stream<ITYPE> C_fifo_in[B_WIDTH_BLOCK],hls::stream<ITYPE> C_fifo_out[B_WIDTH_BLOCK])
{

	ITYPE C_fifo_read[B_WIDTH_BLOCK];
	for (int A_index = 0; A_index < row_count; A_index++) {

		LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL


							if (i<crows)
							{


							        //std::cout << "relu " << relu << std::endl;
							        if (acc2[j][i] > 0 || relu == 0)
										C_fifo_val = acc2[j][i];
									else
										C_fifo_val = 0.0;
									C_fifo_out[j].write(C_fifo_val);
									//std::cout << "C_fifo_val " << C_fifo_val << std::endl;

					         }




				}
		    }

          } // A_index loop

}*/

void compute2_1(ap_uint<1> model[5][8],float srelu[5],int block_size,ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<TTYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo, QTYPE B_accel1[B_HEIGHT/2][B_WIDTH_BLOCK],
		//ITYPE B_accel2[B_HEIGHT/2][B_WIDTH_BLOCK],
		//ITYPE B_accel3[B_HEIGHT/4][B_WIDTH_BLOCK],ITYPE B_accel4[B_HEIGHT/4][B_WIDTH_BLOCK],
hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK],int B_index)
{


       //#pragma HLS allocation function instances=dsp_kernel limit=1



        //hls::stream<DTYPE>       A_accel;
        //#pragma HLS STREAM variable=A_accel depth=A_WIDTH_FIFO dim=1

	//DTYPE A_accel[A_WIDTH];
        //#pragma HLS array_partition variable=A_accel cyclic factor=


	ITYPE acc[B_WIDTH_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc complete


	ITYPE acc2[B_WIDTH_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0


	//hls::stream<int>             col_indices_fifo;
	//#pragma HLS STREAM variable=col_indices_fifo depth=1024 dim=1

	//int col_indices[A_WIDTH];


      int B_WIDTH_INT;
      ITYPE C_fifo_val;

		B_WIDTH_INT = B_WIDTH_BLOCK;




        //#pragma HLS DATAFLOW

    bool relu;

    relu = model[B_index][4];

    float relu_t = srelu[B_index];

    bool linear_mode;

	linear_mode = model[B_index][6];

    if (linear_mode==0)
    {
	 for (int A_index = 0; A_index < row_count; A_index++) {


		LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			#pragma HLS UNROLL
				acc2[j] = 0;
		}



		int rnnz;
		rnnz = rnnz_fifo.read();

		//std::cout << "ADJ: The rnnz value is " << rnnz << std::endl;

		//printf("crows %d\n",crows);

        //std::cout << "ADJ: The rnnz value is " << rnnz[0] << std::endl;
        //std::cout << "ADJ: A_index is " << A_index << std::endl;
        //std::cout << "ADJ: row_count is " << row_count << std::endl;
		dsp_kernel_wrapper_adj_1(block_size,rnnz,A_fifo,col_indices_fifo,B_accel1,
				//B_accel2,
				//B_accel3,B_accel4,
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
					//std::cout << "C_fifo_val " << C_fifo_val << std::endl;
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
		//ITYPE C_buf2[B_HEIGHT][B_WIDTH_BLOCK],
		//ITYPE C_buf3[B_HEIGHT/4][B_WIDTH_BLOCK],ITYPE C_buf4[B_HEIGHT/4][B_WIDTH_BLOCK],
		QTYPE A_buf1[B_HEIGHT][B_WIDTH_BLOCK], int B_index)
{


	ITYPE acc[B_WIDTH_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc complete


	ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0 //all dimensions are partitioned

  

	//hls::stream<int>             col_indices_fifo;
	//#pragma HLS STREAM variable=col_indices_fifo depth=1024 dim=1

	//int col_indices[A_WIDTH];


      int B_WIDTH_INT;

      //for (int B_index = 0; B_index < B_index_loop; B_index++) {


      		B_WIDTH_INT = B_WIDTH_BLOCK;




        //#pragma HLS DATAFLOW


        //std::cout << " row_count " << row_count << std::endl;
     bool gemm_mode;
	 gemm_mode = model[B_index][1];

     bool linear_mode;

	 linear_mode = model[B_index][6];

	 if(linear_mode==0)
	 {

      for (int A_index = 0; A_index < row_count; A_index+=SPMM_BLOCK) {

         #pragma HLS dataflow


		//computing

    	LOOP_ACC21 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
    	      #pragma HLS UNROLL
    	    	LOOP_ACC22 : for (int i = 0; i < SPMM_BLOCK; i++) {
    	               #pragma HLS UNROLL
    	  		acc2[j][i] = 0;
    	  	}
    	}

    	int rnnz[SPMM_BLOCK];
        //#pragma HLS ARRAY_PARTITION variable=rnnz complete dim=0 //all dimensions are partitioned


    	rnnz[0] = rnnz_fifo.read();

    	int rnnz_temp;

		LOOP_RNNZ_SPMM :for (int i = 1; i < SPMM_BLOCK; i++) {
            //#pragma HLS UNROLL
			#pragma HLS PIPELINE II=2
			if((A_index+i) < row_count)
			    rnnz_temp = rnnz_fifo.read();
			else
				rnnz_temp = 0;
		    rnnz[i] = rnnz_temp+rnnz[i-1];

				//std::cout << " rnnz is " << rnnz[i] << " in " << i << std::endl;

			//else
				//rnnz[i] = 0;
		}



  		//std::cout << " rnnz is " << rnnz[0] << std::endl;

		dsp_kernel_wrapper_fea(gemm_mode,rnnz,A_fifo,col_indices_fifo,B_accel,zero_point_lhs,zero_point_rhs,acc2);

		LOOP_C_BUF1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			//#pragma HLS loop_tripcount min=16 max=16 avg=16
	        #pragma HLS UNROLL
			if (j < B_WIDTH_INT)
			{
				//C_fifo[j].write(acc2[j]);
				#ifdef simulation
				if (acc2[j] < acc2_fea_min)
					acc2_fea_min = acc2[j];
				else if (acc2[j] > acc2_fea_max)
					acc2_fea_max = acc2[j];
				#endif
				LOOP_C_BUF2 : for (int i = 0; i < SPMM_BLOCK; i++) {
                    #pragma HLS UNROLL
					ITYPE cur_val = ITYPE(acc2[j][i]);

					*max_fea = 0;

					//if (cur_val > *max_fea)
					//{
					//	*max_fea= cur_val;
						//std::cout << " MAX FEA " << cur_val << std::endl;
					//}


					//ap_fixed<32, 16> max = quantized_multiplier; //quantized_multiplier is always 32-bit with 16 bit fractional so the same values can be used in python as max and min.
					//ap_fixed<32, 16> min = -quantized_multiplier;

					//ITYPE acc2_temp_1 = (acc2[j][i] >> scale_fea);

					//ap_fixed<32, 16, AP_RND,AP_SAT>  max = quantized_multiplier;
					//ap_fixed<32, 16, AP_RND,AP_SAT>  min = -quantized_multiplier;

					//std::cout << "max " << max << "min " << min << std::endl;
					//std::cout << "acc2[j][i] " << acc2[j][i] << std::endl;

					//ap_fixed<32, 16, AP_RND,AP_SAT>  acc2_temp_1 = acc2[j][i];

					ap_fixed<32, 16>  acc2_temp_1 = acc2[j][i];

                   //acc2_temp_1 = quantized_multiplier&(acc2_temp_1 >> scale_fea); //mask bits outside allowed range

					QTYPE2 acc2_temp_1_2 = QTYPE2(acc2_temp_1 >> scale_fea[B_index]);
					QTYPE4 acc2_temp_1_4 = QTYPE4(acc2_temp_1 >> scale_fea[B_index]);
					QTYPE8 acc2_temp_1_8 = QTYPE8(acc2_temp_1 >> scale_fea[B_index]);
					//QTYPE8 acc2_temp_1_8 = float_to_fix((acc2_temp_1 >> scale_fea),7);
					QTYPE acc2_temp_1_16 = QTYPE(acc2_temp_1 >> scale_fea[B_index]);

					//QTYPE acc2_temp = QTYPE(acc2_temp_1);

					//std::cout << "Internal quantization in " << acc2[j] << " out " << acc2_temp_1 << std::endl;

                   //std::cout << "Internal quantization in " << acc2[j][i] << " out " << acc2_temp_1 << std::endl;

                   #if GAT_ENABLE == 1

                     //if (acc2_temp_1 > max)
					 //{
					 //		C_buf1[A_index+i][j] = (QTYPE)max;
					 //       A_buf1[A_index+i][j] = (QTYPE)max;
					 //}
					 //else if (acc2_temp_1 < min)
					 //{
					 //		C_buf1[A_index+i][j] = (QTYPE)min;
					 //       A_buf1[A_index+i][j] = (QTYPE)min;
					 //}
					 if(quantized_multiplier==2)
					 {
							 C_buf1[A_index+i][j]=acc2_temp_1_2;
					         A_buf1[A_index+i][j]=acc2_temp_1_2;
					 }
					 else if(quantized_multiplier==4)
					 {
						     C_buf1[A_index+i][j]=acc2_temp_1_4;
			                 A_buf1[A_index+i][j]=acc2_temp_1_4;
					 }
			         else if(quantized_multiplier==8)
			         {

			        	     C_buf1[A_index+i][j]=acc2_temp_1_8;
					         A_buf1[A_index+i][j]=acc2_temp_1_8;
			         }
					 else
					 {
						    C_buf1[A_index+i][j]=acc2_temp_1_16;
				            A_buf1[A_index+i][j]=acc2_temp_1_16;
					 }

                   #else

					 if(quantized_multiplier==2)
						 C_buf1[A_index+i][j]=acc2_temp_1_2;
					 else if(quantized_multiplier==4)
						 C_buf1[A_index+i][j]=acc2_temp_1_4;
					 else if(quantized_multiplier==8)
						 C_buf1[A_index+i][j]=acc2_temp_1_8;
					 //if (acc2_temp_1 > max)
					 //{
					 //		C_buf1[A_index+i][j] = (QTYPE)max;
                     //
					 //}
					 //else if (acc2_temp_1 < min)
					 //{
					//		C_buf1[A_index+i][j] = (QTYPE)min;

					 //}
					 else
					 {
						    C_buf1[A_index+i][j]=acc2_temp_1_16;

					 }



                    #endif


					//std::cout << "Internal quantization in " << acc2_temp_1  << " out " << C_buf1[A_index+i][j] << std::endl;
					 //C_buf1[A_index+i][j]=acc2[j][i];

					//std::cout << " Abuf " << A_index+i << " " << j << " " << A_buf1[A_index+i][j] << std::endl;
					//std::cout << " acc2_temp " << acc2_temp << std::endl;

				} //c_buf loop2
				//C_buf2[A_index][j]=acc2[j];
				//C_buf3[A_index][j]=acc2[j];
				//C_buf4[A_index][j]=acc2[j];
			}	//c_buf loop1

		   } // j < B_WIDTH_BLOCK

          } // A_index loop
     } // linear_mode
         //*max_fea = max_val;
	//}
}

//linear operator

void compute1_12(STYPE scale_fea[5],ITYPE* max_fea,int quantized_multiplier,ap_uint<1> model[5][8],ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<LTYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo,BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK], QTYPE linear_pipo[B_HEIGHT][B_WIDTH_BLOCK], int B_index)
{


	ITYPE acc[B_WIDTH_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc complete


	ITYPE acc2[B_WIDTH_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc2 complete //all dimensions are partitioned


	//hls::stream<int>             col_indices_fifo;
	//#pragma HLS STREAM variable=col_indices_fifo depth=1024 dim=1

	//int col_indices[A_WIDTH];


      int B_WIDTH_INT;

      //for (int B_index = 0; B_index < B_index_loop; B_index++) {


      		B_WIDTH_INT = B_WIDTH_BLOCK;




        //#pragma HLS DATAFLOW


        //std::cout << " row_count " << row_count << std::endl;


    bool gemm_mode;
    gemm_mode = model[B_index][1];

    bool linear_mode;

    linear_mode = model[B_index][6];


    if(linear_mode==1)
    {


	 for (int A_index = 0; A_index < row_count; A_index++) {


		//computing

		LOOP_ACC21 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
				acc2[j] = 0;
		}

		int rnnz;

		rnnz = rnnz_fifo.read();


  		//std::cout << " rnnz is " << rnnz[0] << std::endl;

		dsp_kernel_wrapper_lin(gemm_mode,rnnz,A_fifo,col_indices_fifo,B_accel,zero_point_lhs,zero_point_rhs,acc2);

		LOOP_C_BUF1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			//#pragma HLS loop_tripcount min=16 max=16 avg=16
	           	#pragma HLS UNROLL
			if (j < B_WIDTH_INT)
			{
				//C_fifo[j].write(acc2[j]);
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
						//std::cout << " MAX FEA " << cur_val << std::endl;
					}


					//ap_fixed<32, 16> max = quantized_multiplier; //quantized_multiplier is always 32-bit with 16 bit fractional so the same values can be used in python as max and min.
					//ap_fixed<32, 16> min = -quantized_multiplier;

					//ITYPE acc2_temp_1 = (acc2[j][i] >> scale_fea);

					//ap_fixed<32, 16, AP_RND,AP_SAT>  max = quantized_multiplier;
					//ap_fixed<32, 16, AP_RND,AP_SAT>  min = -quantized_multiplier;

					//std::cout << "max " << max << "min " << min << std::endl;
					//std::cout << "acc2[j][i] " << acc2[j][i] << std::endl;

					//ap_fixed<32, 16, AP_RND,AP_SAT>  acc2_temp_1 = acc2[j][i];

					ap_fixed<32, 16>  acc2_temp_1 = acc2[j];

                   //acc2_temp_1 = quantized_multiplier&(acc2_temp_1 >> scale_fea); //mask bits outside allowed range

					QTYPE2 acc2_temp_1_2 = QTYPE2(acc2_temp_1 >> scale_fea[B_index]);
					QTYPE4 acc2_temp_1_4 = QTYPE4(acc2_temp_1 >> scale_fea[B_index]);
					QTYPE8 acc2_temp_1_8 = QTYPE8(acc2_temp_1 >> scale_fea[B_index]);
					//QTYPE8 acc2_temp_1_8 = float_to_fix((acc2_temp_1 >> scale_fea),7);
					QTYPE acc2_temp_1_16 = QTYPE(acc2_temp_1 >> scale_fea[B_index]);

					//QTYPE acc2_temp = QTYPE(acc2_temp_1);

					//std::cout << "Internal quantization in " << acc2[j] << " out " << acc2_temp_1 << std::endl;

                   //std::cout << "Internal quantization in " << acc2[j][i] << " out " << acc2_temp_1 << std::endl;



					 if(quantized_multiplier==2)
						 linear_pipo[A_index][j]=acc2_temp_1_2;
					 else if(quantized_multiplier==4)
						 linear_pipo[A_index][j]=acc2_temp_1_4;
					 else if(quantized_multiplier==8)
						 linear_pipo[A_index][j]=acc2_temp_1_8;
					 else
						 linear_pipo[A_index][j]=acc2_temp_1_16;




				} //c_buf loop
				//C_buf2[A_index][j]=acc2[j];
				//C_buf3[A_index][j]=acc2[j];
				//C_buf4[A_index][j]=acc2[j];

		   } // j < B_WIDTH_BLOCK

          } // A_index loop
      }
         //*max_fea = max_val;
	//}
}




void mxv(int M, int P_w, QTYPE C_mxv[B_HEIGHT][B_WIDTH_BLOCK], BTYPE* A, TTYPE *WH1,TTYPE *WH2)
{

    ITYPE acc1,acc2;
    BTYPE ate_m_int[B_WIDTH_BLOCK*2];
    #pragma HLS array_partition variable=ate_m_int type=complete

    PRELOAD : for(int i = 0; i < B_WIDTH_BLOCK*2; i++) {
       #pragma HLS PIPELINE
    	ate_m_int[i] = A[i];
    }
	//std::cout << "M" << M << std::endl;

	LOOP_MXV3 : for(int i = 0; i < M; i++) {

		ITYPE acc11 = 0;
		ITYPE acc21 = 0;
		ITYPE acc12 = 0;
		ITYPE acc22 = 0;
	     //#pragma HLS UNRO

	     //#pragma HLS UNROLL factor=BLOCK
		LOOP_MXV4 : for (int j = 0; j < B_WIDTH_BLOCK; j+=2) {
                #pragma HLS PIPELINE II=1
	 			acc11 += C_mxv[i][j]*ate_m_int[j];
	 			acc21 += C_mxv[i][j]*ate_m_int[j+B_WIDTH_BLOCK];
	 			acc12 += C_mxv[i][j+1]*ate_m_int[j+1];
	 			acc22 += C_mxv[i][j+1]*ate_m_int[j+1+B_WIDTH_BLOCK];
	            //std::cout << "i " << i << std::endl;
	 			//std::cout << "C_mxv " << C_mxv[i][j] << std::endl;
	 			//std::cout << "A " << A[j] << std::endl;
	 			//std::cout << "acc " << acc[i] << std::endl;
		}
		acc1 = acc11+acc12;
		acc2 = acc21+acc22;
	   	//std::cout << "C is" << C[j] << std::endl;
		//gatv2
        #if GATV2 == 1
	     if (acc1 > 0)
	    	WH1[i] = acc1;
	 	 else
	 	   //relu_out = relu_in; //leaky relu
	 		WH1[i] = acc1*ITYPE(0.2); //leaky relu
	 	 if (acc2 > 0)
	 		WH2[i] = acc2;
	 	 else
	 	   //relu_out = relu_in; //leaky relu
	 		WH2[i] = acc2*ITYPE(0.2); //leaky relu
	 	//standard gat
        #else
		 WH1[i] = acc1;
		 WH2[i] = acc2;
        #endif
		//std::cout << "WH " << WH[i] << std::endl;
	}
	//std::cout << "WH2 " << WH2[1862] << std::endl;
}


void mxv1(int M, int P_w, QTYPE C_mxv[B_HEIGHT/FEA_THREADS][B_WIDTH_BLOCK],
		BTYPE ate_m_int[B_WIDTH_BLOCK*2],
		TTYPE *WH1,TTYPE *WH21,TTYPE *WH22,TTYPE *WH23,TTYPE *WH24)
{

	ITYPE acc1 = 0;
	ITYPE acc2 = 0;

	LOOP_MXV1 : for(int i = 0; i < M; i++) {
		ITYPE acc11 = 0;
		ITYPE acc21 = 0;
		ITYPE acc12 = 0;
		ITYPE acc22 = 0;
	     //#pragma HLS UNROLL factor=BLOCK
		LOOP_MXV5 : for (int j = 0; j < B_WIDTH_BLOCK; j+=2) {
		//LOOP_MXV5 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS PIPELINE
	 			acc11 += C_mxv[i][j]*ate_m_int[j];
	 			acc21 += C_mxv[i][j]*ate_m_int[j+B_WIDTH_BLOCK];
	 			acc12 += C_mxv[i][j+1]*ate_m_int[j+1];
	 			acc22 += C_mxv[i][j+1]*ate_m_int[j+1+B_WIDTH_BLOCK];

	            //std::cout << "i " << i << std::endl;
	 			//std::cout << "C_mxv " << C_mxv[i][j] << std::endl;
	 			//std::cout << "A " << A[j] << std::endl;
	 			//std::cout << "acc " << acc[i] << std::endl;
		}
		acc1 = acc11+acc12;
		acc2 = acc21+acc22;

	   	//std::cout << "C is" << C[j] << std::endl;
		//gatv2
        #if GATV2 == 1
	     if (acc1 > 0)
	    	WH1[i] = acc1;
	 	 else
	 	   //relu_out = relu_in; //leaky relu
	 		WH1[i] = acc1*ITYPE(0.2); //leaky relu
	 	 if (acc2 > 0)
	 	 {
	 		WH21[i] = acc2;
			WH22[i] = acc2;
			WH23[i] = acc2;
			WH24[i] = acc2;
	 	 }
	 	 else
	 	 {
	 	   //relu_out = relu_in; //leaky relu
	 		WH21[i] = acc2*ITYPE(0.2); //leaky relu
	 		WH22[i] = acc2*ITYPE(0.2); //leaky relu
	 		WH23[i] = acc2*ITYPE(0.2); //leaky relu
	 		WH24[i] = acc2*ITYPE(0.2); //leaky relu
	 	 }
	 	//standard gat
        #else
		 WH1[i] = acc1;
		 WH21[i] = acc2;
		 WH22[i] = acc2;
		 WH23[i] = acc2;
		 WH24[i] = acc2;
        #endif
		//std::cout << "WH " << WH[i] << std::endl;
	}
}

void mxv4(int M1, int M2, int M3, int M4, int P_w, QTYPE C_mxv1[B_HEIGHT/FEA_THREADS][B_WIDTH_BLOCK],
		QTYPE C_mxv2[B_HEIGHT/FEA_THREADS][B_WIDTH_BLOCK], QTYPE C_mxv3[B_HEIGHT/FEA_THREADS][B_WIDTH_BLOCK],
		QTYPE C_mxv4[B_HEIGHT/FEA_THREADS][B_WIDTH_BLOCK], BTYPE* A,
		TTYPE *WH11,TTYPE *WH211,TTYPE *WH212,TTYPE *WH213,TTYPE *WH214,
		TTYPE *WH12,TTYPE *WH221,TTYPE *WH222,TTYPE *WH223,TTYPE *WH224,
		TTYPE *WH13,TTYPE *WH231,TTYPE *WH232,TTYPE *WH233,TTYPE *WH234,
		TTYPE *WH14,TTYPE *WH241,TTYPE *WH242,TTYPE *WH243,TTYPE *WH244)
{

    BTYPE ate_m_int[B_WIDTH_BLOCK*2];
    #pragma HLS array_partition variable=ate_m_int type=complete

	//std::cout << "M" << M << std::endl;

    PRELOAD : for(int i = 0; i < B_WIDTH_BLOCK*2; i++) {
       #pragma HLS PIPELINE
    	ate_m_int[i] = A[i];
    }


   mxv1(M1,P_w,C_mxv1,ate_m_int,WH11,WH211,WH212,WH213,WH214);
   mxv1(M2,P_w,C_mxv2,ate_m_int,WH12,WH221,WH222,WH223,WH224),
   mxv1(M3,P_w,C_mxv3,ate_m_int,WH13,WH231,WH232,WH233,WH234);
   mxv1(M4,P_w,C_mxv4,ate_m_int,WH14,WH241,WH242,WH243,WH244);

}




void prepare_attentional_mechanism_input2(
		ap_uint<1> model[5][8],int N_adj, ap_uint<8> P_w[5],
        #if (PIPO_BLOCKS >= 2)
		hls::stream_of_blocks<buf> &A_buffer11,
        #else
		buf A_buffer11,
        #endif
		hls::stream<ATYPE> &A_fifo_adj, hls::stream<int> &col_indices_fifo_adj,
		hls::stream<int> &rnnz_fifo_adj,
		hls::stream<int> &E_col_indices_fifo, hls::stream<int> &E_rnnz_fifo,
		hls::stream<ATYPE> &A_fifo,hls::stream<TTYPE> &E_fifo,hls::stream<TTYPE> &max_fifo,
		BTYPE ate_m[2*C_WIDTH],hls::stream<TTYPE> &EO_fifo,hls::stream<int> &EO_rnnz_fifo,int B_index)
{
    // Wh.shape (N, out_feature)
    // self.a.shape (2 * out_feature, 1)
    // Wh1&2.shape (N, 1)
    // e.shape (N, N)

    TTYPE WH1[A_HEIGHT] = {0};
    TTYPE WH2[A_HEIGHT] = {0};
    TTYPE relu_out,relu_in;
    TTYPE max_val,A_val;
	int rnnz,col;

	ap_uint<8> P_w_attention = P_w[B_index];
	bool gat_mode = model[B_index][5];
	bool linear_mode = model[B_index][6];


    std::cout << "gat mode is " << gat_mode << std::endl;

    #if (PIPO_BLOCKS >= 2)

    hls::read_lock<buf> C_mxv(A_buffer11);

    if(gat_mode == 1 & linear_mode == 0)
        mxv(N_adj, P_w_attention, C_mxv, ate_m,WH1,WH2);

    #else

    if(gat_mode == 1  & linear_mode == 0)
        mxv(N_adj, P_w_attention, A_buffer11, ate_m,WH1,WH2);
    #endif

    //mxv2(N_adj, P_w, C_mxv, ate_m,WH);
    // broadcast add
    //e = Wh1 + Wh2.T;



    if(linear_mode == 0)
    {
     std::cout << "generating attention candidates " << N_adj << std::endl;
	 LOOP_WH1 : for (int i = 0; i < N_adj; i++) {
			 rnnz = rnnz_fifo_adj.read();
		 	 E_rnnz_fifo << rnnz;

		 	 //if(gat_mode == 1)
		 	  //EO_rnnz_fifo << rnnz;

		 	 max_val = 0.0;

     	     int rnnz_loop = rnnz;


             //std::cout << "rnnz " << (rnnz[z+1]-rnnz[z]) << std::endl;
	  	     LOOP_WH2 : for (int j = 0; j <rnnz_loop; j++) {
	  	     #pragma HLS PIPELINE
	  	        col = col_indices_fifo_adj.read();
	  	        E_col_indices_fifo << col;
	  	        A_val = A_fifo_adj.read();
	  	        A_fifo << A_val;

	            if(gat_mode == 1)	            {
	               relu_in = WH1[i]+WH2[col];
	               //std::cout << "[i+z] " << i+z << " [col] " << col << std::endl;
	  	           //std::cout << "WH1[i+z] " << WH1[i+z] << " WH2[col] " << WH2[col] << std::endl;
                #if GATV2 == 1
		  	        //gatv2
		  	        relu_out = relu_in;
               #else
		  	     //standard gat
	  	         if (relu_in >= 0)
	  		       relu_out = relu_in;
	  	         else
	  		      //relu_out = relu_in; //leaky relu
	  		      relu_out = relu_in*ITYPE(0.2); //leaky relu
                #endif
	  	        if (relu_out > max_val)
	  		         max_val = relu_out;
	  	        //std::cout << "generate attention candidate e " << relu_out << std::endl;
                E_fifo << relu_out;
	  	        EO_fifo << relu_out;  //output to extenal memory
	  	        //std::cout << "EO_fifo " << relu_out << std::endl;
	         }

	       } //j
	  	   if(gat_mode == 1)
	  		   max_fifo << max_val;
	  } //i
    } //linear mode


    //return self.leakyrelu(e)
}

void generate_attention_candidates(bool gat_mode,int N_adj,int N_block,
hls::stream<ATYPE> &A_fifo_adj,
hls::stream<int> &col_indices_fifo_adj,hls::stream<int> rnnz_fifo_adj[SPMM_BLOCK],
hls::stream<int> &E_col_indices_fifo,hls::stream<int> E_rnnz_fifo[SPMM_BLOCK],
hls::stream<ATYPE> &A_fifo,hls::stream<TTYPE> &E_fifo,hls::stream<TTYPE> &max_fifo,
hls::stream<TTYPE> &EO_fifo,hls::stream<int> &EO_rnnz_fifo,
TTYPE *WH1,TTYPE *WH21,TTYPE *WH22,TTYPE *WH23,TTYPE *WH24)
{

   TTYPE WH2_col;

   LOOP_WH11 : for (int i = 0; i < N_adj; i++) {
			 int rnnz = rnnz_fifo_adj[0].read();
		 	 E_rnnz_fifo[0] << rnnz;

		 	 //if(gat_mode == 1)
		 	 // EO_rnnz_fifo << rnnz;

		 	 TTYPE max_val = 0.0;

     	     int rnnz_loop = rnnz;
     	     TTYPE relu_out,relu_in;

             //std::cout << "rnnz " << (rnnz[z+1]-rnnz[z]) << std::endl;
	  	     LOOP_WH21 : for (int j = 0; j <rnnz_loop; j++) {
	  	     #pragma HLS PIPELINE
	  	        int col = col_indices_fifo_adj.read();
	  	        E_col_indices_fifo << col;
	  	        TTYPE A_val = A_fifo_adj.read();
	  	        A_fifo << A_val;

	  	        //adjust index col to right segment

	  	  	    if (col < N_block)
	  	  	        WH2_col=WH21[col];
	  	  	    else if (col < N_block*2)
	  	  	  	    WH2_col=WH22[col-N_block];
	  	  	    else if (col < N_block*3)
	  	  	  	    WH2_col=WH23[col-2*N_block];
	  	  	    else
	  	  	  	    WH2_col=WH24[col-3*N_block];



	            if(gat_mode == 1)	            {
	               relu_in = WH1[i]+WH2_col;
	               //std::cout << "[i+z] " << i+z << " [col] " << col << std::endl;
	  	           //std::cout << "WH1[i+z] " << WH1[i+z] << " WH2[col] " << WH2[col] << std::endl;
                #if GATV2 == 1
		  	        //gatv2
		  	        relu_out = relu_in;
               #else
		  	     //standard gat
	  	         if (relu_in >= 0)
	  		       relu_out = relu_in;
	  	         else
	  		      //relu_out = relu_in; //leaky relu
	  		      relu_out = relu_in*ITYPE(0.2); //leaky relu
                #endif
	  	        if (relu_out > max_val)
	  		         max_val = relu_out;
	  	        //std::cout << "generate attention candidate e " << relu_out << std::endl;
                E_fifo << relu_out;
	  	        EO_fifo << relu_out;  //output to extenal memory
	  	        //std::cout << "EO_fifo " << relu_out << std::endl;
	         }

	       } //j
	  	   if(gat_mode == 1)
	  		   max_fifo << max_val;
	  } //i
}

void prepare_attentional_mechanism_inputx4(bool gat_mode,int N_adj1, int N_adj2, int N_adj3, int N_adj4, int P_w,
		hls::stream_of_blocks<buf> &A_buffer11,
		hls::stream_of_blocks<buf> &A_buffer21,
		hls::stream_of_blocks<buf> &A_buffer31,
		hls::stream_of_blocks<buf> &A_buffer41,
		hls::stream<ATYPE> &A_fifo_adj1,
		hls::stream<ATYPE> &A_fifo_adj2,
		hls::stream<ATYPE> &A_fifo_adj3,
		hls::stream<ATYPE> &A_fifo_adj4,
		hls::stream<int> &col_indices_fifo_adj1,
	    hls::stream<int> &col_indices_fifo_adj2,
		hls::stream<int> &col_indices_fifo_adj3,
		hls::stream<int> &col_indices_fifo_adj4,
		hls::stream<int> rnnz_fifo_adj1[SPMM_BLOCK],
		hls::stream<int> rnnz_fifo_adj2[SPMM_BLOCK],
		hls::stream<int> rnnz_fifo_adj3[SPMM_BLOCK],
		hls::stream<int> rnnz_fifo_adj4[SPMM_BLOCK],
		hls::stream<int> &E_col_indices_fifo1,
		hls::stream<int> &E_col_indices_fifo2,
		hls::stream<int> &E_col_indices_fifo3,
		hls::stream<int> &E_col_indices_fifo4,
		hls::stream<int> E_rnnz_fifo1[SPMM_BLOCK],
		hls::stream<int> E_rnnz_fifo2[SPMM_BLOCK],
		hls::stream<int> E_rnnz_fifo3[SPMM_BLOCK],
		hls::stream<int> E_rnnz_fifo4[SPMM_BLOCK],
		hls::stream<ATYPE> &A_fifo1,
		hls::stream<ATYPE> &A_fifo2,
		hls::stream<ATYPE> &A_fifo3,
		hls::stream<ATYPE> &A_fifo4,
		hls::stream<TTYPE> &E_fifo1,
		hls::stream<TTYPE> &E_fifo2,
		hls::stream<TTYPE> &E_fifo3,
		hls::stream<TTYPE> &E_fifo4,
		hls::stream<TTYPE> &max_fifo1,
		hls::stream<TTYPE> &max_fifo2,
		hls::stream<TTYPE> &max_fifo3,
		hls::stream<TTYPE> &max_fifo4,
		BTYPE ate_m[2*C_WIDTH],
		hls::stream<TTYPE> &EO_fifo1,
		hls::stream<TTYPE> &EO_fifo2,
		hls::stream<TTYPE> &EO_fifo3,
		hls::stream<TTYPE> &EO_fifo4,
		hls::stream<int> &EO_rnnz_fifo1,
		hls::stream<int> &EO_rnnz_fifo2,
		hls::stream<int> &EO_rnnz_fifo3,
		hls::stream<int> &EO_rnnz_fifo4)
{
    // Wh.shape (N, out_feature)
    // self.a.shape (2 * out_feature, 1)
    // Wh1&2.shape (N, 1)
    // e.shape (N, N)

    TTYPE WH11[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH211[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH221[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH231[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH241[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH12[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH212[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH222[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH232[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH242[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH13[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH213[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH223[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH233[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH243[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH14[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH214[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH224[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH234[A_HEIGHT/FEA_THREADS]; // = {0};
    TTYPE WH244[A_HEIGHT/FEA_THREADS]; // = {0};


    hls::read_lock<buf> C_mxv1(A_buffer11);
    hls::read_lock<buf> C_mxv2(A_buffer21);
    hls::read_lock<buf> C_mxv3(A_buffer31);
    hls::read_lock<buf> C_mxv4(A_buffer41);

    if(gat_mode == 1)
        mxv4(N_adj1,N_adj2,N_adj3,N_adj4,P_w, C_mxv1, C_mxv2, C_mxv3, C_mxv4,ate_m,
        		WH11,WH211,WH212,WH213,WH214,
				WH12,WH221,WH222,WH223,WH224,
				WH13,WH231,WH232,WH233,WH234,
				WH14,WH241,WH242,WH243,WH244);


    //mxv2(N_adj, P_w, C_mxv, ate_m,WH);
    // broadcast add
    //e = Wh1 + Wh2.T;


    std::cout << "generating attention candidates " << N_adj1 << std::endl;

    generate_attention_candidates(gat_mode,N_adj1,N_adj1,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1,E_col_indices_fifo1,E_rnnz_fifo1,
    A_fifo1,E_fifo1,max_fifo1,EO_fifo1,EO_rnnz_fifo1,WH11,WH211,WH231,WH231,WH241);
    generate_attention_candidates(gat_mode,N_adj2,N_adj1,A_fifo_adj2,col_indices_fifo_adj2,rnnz_fifo_adj2,E_col_indices_fifo2,E_rnnz_fifo2,
    A_fifo2,E_fifo2,max_fifo2,EO_fifo2,EO_rnnz_fifo2,WH12,WH212,WH232,WH232,WH242);
    generate_attention_candidates(gat_mode,N_adj3,N_adj1,A_fifo_adj3,col_indices_fifo_adj3,rnnz_fifo_adj3,E_col_indices_fifo3,E_rnnz_fifo3,
    A_fifo3,E_fifo3,max_fifo3,EO_fifo3,EO_rnnz_fifo3,WH13,WH213,WH233,WH233,WH243);
    generate_attention_candidates(gat_mode,N_adj4,N_adj1,A_fifo_adj4,col_indices_fifo_adj4,rnnz_fifo_adj4,E_col_indices_fifo4,E_rnnz_fifo4,
    A_fifo4,E_fifo4,max_fifo4,EO_fifo4,EO_rnnz_fifo4,WH14,WH214,WH234,WH234,WH244);


    //return self.leakyrelu(e)
}

/*
void prepare_attentional_mechanism_input(int N_adj, int P_w, hls::stream_of_blocks<buf> &C_buffer11,hls::stream<FTYPE> &E_fifo,FTYPE ate_m[2*C_WIDTH])
{
    // Wh.shape (N, out_feature)
    // self.a.shape (2 * out_feature, 1)
    // Wh1&2.shape (N, 1)
    // e.shape (N, N)
    FTYPE WH1[A_HEIGHT];
    FTYPE WH2[A_HEIGHT];
    FTYPE relu_out,relu_in;

	hls::read_lock<buf> C_mxv(C_buffer11);

	mxv(N_adj, P_w, C_mxv, ate_m,WH1,WH2);

    // broadcast add
    //e = Wh1 + Wh2.T;

    //std::cout << "generating attention candidates " << N_adj << std::endl;
	LOOP_WH1 : for (int i = 0; i < N_adj; i++) {
	  	LOOP_WH2 : for (int j = 0; j <N_adj; j++) {
	  	#pragma HLS PIPELINE
	  	relu_in = WH1[i]+WH2[j];
	  	//std::cout << "i " << i << std::endl;
	  	//std::cout << "relu in " << relu_in << std::endl;
	  	if (relu_in > 0)
	  		relu_out = relu_in;
	  	else
	  		relu_out = relu_in*0.01; //leaky relu
	  	//std::cout << "generate attention candidate " << relu_out << std::endl;
	  	E_fifo << relu_out;
	  }
	 }

    //return self.leakyrelu(e)
}

void softmax2(int N_adj,ATYPE row_in[A_WIDTH], hls::stream<ATYPE> &val_att_fifo)
{
	int i;
	int counter =0;
	ATYPE support[A_WIDTH];
	//softmax();
	ATYPE max_val, sum;
	max_val = row_in[0];
	LOOP_SOFTMAX1 : for (i = 1; i < N_adj; i++) {
		#pragma HLS PIPELINE II=1
		if (row_in[i] > max_val)
			max_val = row_in[i];
	}
	sum = 0.0;
	LOOP_SOFTMAX2 : for (i = 0; i < N_adj; i++) {
        #pragma HLS PIPELINE II=2
		support[i] = hls::half_exp(row_in[i] - max_val);
		sum += support[i];
	}
	LOOP_SOFTMAX3 :  for (i = 0; i < N_adj; i++) {
        #pragma HLS PIPELINE II=1
		support[i] /= sum;
	}
	LOOP_SOFTMAX4 : for (i = 0; i < N_adj; i++) {
        #pragma HLS PIPELINE II=1
		//std::cout << "val_att_fifo " << support[i] << std::endl;
		if (support[i] > 0.0)
		   val_att_fifo << support[i];
	}


}

void softmax(int N_adj,ATYPE row_in[A_WIDTH], hls::stream<ATYPE> &val_att_fifo)
{
	int i;
	int counter =0;
	ATYPE support[A_WIDTH];
	//softmax();
	ATYPE max_val, sum;
	max_val = row_in[0];
	LOOP_SOFTMAX1 : for (i = 1; i < N_adj; i++) {
		#pragma HLS PIPELINE II=1
		if (row_in[i] > max_val)
			max_val = row_in[i];
	}
	sum = 0.0;
	LOOP_SOFTMAX2 : for (i = 0; i < N_adj; i++) {
        #pragma HLS PIPELINE II=2
		support[i] = hls::half_exp(row_in[i] - max_val);
		sum += support[i];
	}
	LOOP_SOFTMAX3 :  for (i = 0; i < N_adj; i++) {
        #pragma HLS PIPELINE II=1
		support[i] /= sum;
	}
	LOOP_SOFTMAX4 : for (i = 0; i < N_adj; i++) {
        #pragma HLS PIPELINE II=1
		//std::cout << "val_att_fifo " << support[i] << std::endl;
		if (support[i] > 0.0)
		{
		   //std::cout << "val_att_fifo " << support[i] << std::endl;
		   val_att_fifo << support[i];
		}
	}


}
*/
/*
void compute_attention2(bool gat_mode,int N_adj,
		hls::stream<ATYPE> &A_fifo,
		hls::stream<int> &col_indices_fifo, hls::stream<int> rnnz_fifo[SPMM_BLOCK],hls::stream<FTYPE> &E_fifo,
		hls::stream<FTYPE> &max_fifo,hls::stream<ATYPE> &val_att_fifo,hls::stream<int> &col_att_fifo,hls::stream<int> rnnz_att_fifo[SPMM_BLOCK])
{
		int col;
		ATYPE val,attention_candidate;
		ATYPE fixed_val = ATYPE(-9e3);

    	//ATYPE support[A_WIDTH][ATEN_BLOCK];
    	ATYPE support[A_WIDTH*ATEN_BLOCK];
    	ATYPE fixed_support;
    	ATYPE div_val;
    	ATYPE const_one = ATYPE(1);
    	ATYPE max_val[ATEN_BLOCK];




    	 if (gat_mode==1)
   	 	 {

    	   ATEN_LOOP:for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    	   {

                   //#pragma HLS DATAFLOW
    	           ATYPE sum[ATEN_BLOCK];
    	           int rnnz[ATEN_BLOCK+1];
    	           rnnz[0]=0;
    		       int crows = 0;
    		       LOOP_RNNZ :for (int z = 0; z < ATEN_BLOCK; z++) {
    		            #pragma  HLS PIPELINE II=1
    		        	rnnz[z+1] = rnnz_fifo[0].read();
    		        	max_val[z] = max_fifo.read();
    		        	rnnz_att_fifo[0] << rnnz[z+1]-rnnz[z];
    		       		if ((z+i)<N_adj)
    		       	  	  crows++;
    		       	 }


    			  	 //std::cout << "insert rnnz att fifo with " << N_adj << std::endl;
    			  	 //std::cout << "i is " << i << std::endl;




    			 	  LOOP_SOFTMAX2 : for (int j = 0; j < rnnz[ATEN_BLOCK]; j++) {
    			 		int row_index1;
    	                #pragma HLS PIPELINE II=1
    				         attention_candidate = E_fifo.read();
    				         for(int k = 0; k<ATEN_BLOCK; k++)
    				        	 if ((j >= rnnz[k])&&(j < rnnz[k+1]))
    				        		 row_index1 = k;
    	                     #ifdef EIGHTBIT
    				              //support[j][row_index] = hls::exp(attention_candidate- max_val[row_index]);
    				              support[j] = hls::exp(attention_candidate- max_val[row_index1]);
    	                     #else
    				              support[j]= hls::half_exp(attention_candidate- max_val[row_index1]);
    	                     #endif
    				         //std::cout << "support " << support[j] << std::endl;
    				         //sum[row_index] += support[j][row_index];
    				         sum[row_index1] += support[j];

    					 }


    			      LOOP_FIXED :for (int z = 0; z < ATEN_BLOCK; z++) {
                      #pragma HLS PIPELINE II=1
    	              #ifdef EIGHTBIT
    			 	        fixed_support = hls::exp(fixed_val- max_val[z]);
    	              #else
    			 	        fixed_support = hls::half_exp(fixed_val- max_val[z]);
    	              #endif
    			 	  fixed_support = (N_adj-rnnz[z+1]+rnnz[z])*fixed_support;
    			      sum[z] += fixed_support;
    			      }
    			 	  //std::cout << "sum " << sum << std::endl;
    			 	  //div_val = ATYPE(1)/sum;
    			 	  //div_val = const_one/sum;
    			 	  //div_val = sum;

    			 	  //std::cout << "div_val " << div_val << std::endl;
    			 	 LOOP_SOFTMAX4 : for (int j = 0; j < rnnz[ATEN_BLOCK]; j++) {
    			  		 int row_index2;
    			 		 #pragma HLS PIPELINE II=1
    			  	       for(int k = 0; k<ATEN_BLOCK; k++)
    			  	    		if ((j >= rnnz[k])&&(j < rnnz[k+1]))
    			  	    		 row_index2 = k;
    			    	   val = A_fifo.read();
    				       col = col_indices_fifo.read();
    					   col_att_fifo << col;

    			    	   //if (val>0)
    			    	   //{
    					   //ATYPE out_val = support[j][row_index]/sum[row_index];
    					   ATYPE out_val = support[j]/sum[row_index2];
    			    	   val_att_fifo << out_val;
    			    		  // std::cout << "val_att_fifo " << out_val << std::endl;
    			    	   //}

    			 	 }
    		       }
   	 	 }
	    else
	    {

         for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
	     {

             #pragma HLS DATAFLOW
	         int rnnz[ATEN_BLOCK+1];
		     rnnz[0]=0;
		     int crows = 0;
		     LOOP_RNNZ2 :for (int z = 0; z < ATEN_BLOCK; z++) {
		        rnnz[z+1] = rnnz_fifo[0].read();
		        max_val[z] = max_fifo.read();
		        rnnz_att_fifo[0] << rnnz[z+1]-rnnz[z];
		       	if ((z+i)<N_adj)
		       	  crows++;
		     }
		 	 LOOP_SOFTMAX5 : for (int j = 0; j < rnnz[ATEN_BLOCK]; j++) {
		         #pragma HLS PIPELINE II=1
		    	   val = A_fifo.read();
			       col = col_indices_fifo.read();
				   col_att_fifo << col;
		    	   val_att_fifo << val;
		 	 }
		    }
	       }




	      //attention = torch.where(adj > 0, e, zero_vec)
	      //attention = F.softmax(attention, dim=1)
	      //attention = F.dropout(attention, self.dropout, training=self.training)
}
*/

/*
void compute_attention2(bool gat_mode,int N_adj,
		hls::stream<ATYPE> &A_fifo,
		hls::stream<int> &col_indices_fifo, hls::stream<int> rnnz_fifo[SPMM_BLOCK],hls::stream<FTYPE> &E_fifo,
		hls::stream<FTYPE> &max_fifo,hls::stream<ATYPE> &val_att_fifo,hls::stream<int> &col_att_fifo,hls::stream<int> rnnz_att_fifo[SPMM_BLOCK])
{
		int col;
		ATYPE val,attention_candidate;
		ATYPE fixed_val = ATYPE(-9e3);

    	//ATYPE support[A_WIDTH][ATEN_BLOCK];
    	ATYPE support[A_WIDTH*ATEN_BLOCK];
    	ATYPE fixed_support;
    	ATYPE div_val;
    	ATYPE const_one = ATYPE(1);
    	ATYPE max_val[ATEN_BLOCK];




    	 if (gat_mode==1)
   	 	 {

    	   ATEN_LOOP:for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    	   {

                   //#pragma HLS DATAFLOW
    	           ATYPE sum[ATEN_BLOCK];
    	           int rnnz[ATEN_BLOCK+1];
    		       int crows = 0;
    		       rnnz[0]=0;
    		       LOOP_RNNZ :for (int z = 0; z < ATEN_BLOCK; z++) {
    		            #pragma  HLS PIPELINE II=1
    		        	rnnz[z+1] = rnnz_fifo[0].read();
    		        	max_val[z] = max_fifo.read();
    		        	rnnz_att_fifo[0] << rnnz[z+1]-rnnz[z];
    		       		if ((z+i)<N_adj)
    		       	  	  crows++;
    		       	 }


    			  	 //std::cout << "insert rnnz att fifo with " << N_adj << std::endl;
    			  	 //std::cout << "i is " << i << std::endl;




    			 	  LOOP_SOFTMAX2 : for (int j = 0; j < rnnz[ATEN_BLOCK]; j++) {
    			 		int row_index1;
    	                #pragma HLS PIPELINE II=1
    				         attention_candidate = E_fifo.read();
    				         for(int k = 0; k<ATEN_BLOCK; k++)
    				        	 if ((j >= rnnz[k])&&(j < rnnz[k+1]))
    				        		 row_index1 = k;
    	                     #ifdef EIGHTBIT
    				              //support[j][row_index] = hls::exp(attention_candidate- max_val[row_index]);
    				              support[j] = hls::exp(attention_candidate- max_val[row_index1]);
    	                     #else
    				              support[j]= hls::half_exp(attention_candidate- max_val[row_index1]);
    	                     #endif
    				         //std::cout << "support " << support[j] << std::endl;
    				         //sum[row_index] += support[j][row_index];
    				         sum[row_index1] += support[j];

    					 }


    			      LOOP_FIXED :for (int z = 0; z < ATEN_BLOCK; z++) {
                      #pragma HLS PIPELINE II=1
    	              #ifdef EIGHTBIT
    			 	        fixed_support = hls::exp(fixed_val- max_val[z]);
    	              #else
    			 	        fixed_support = hls::half_exp(fixed_val- max_val[z]);
    	              #endif
    			 	  fixed_support = (N_adj-rnnz[z+1]+rnnz[z])*fixed_support;
    			      sum[z] += fixed_support;
    			      }
    			 	  //std::cout << "sum " << sum << std::endl;
    			 	  //div_val = ATYPE(1)/sum;
    			 	  //div_val = const_one/sum;
    			 	  //div_val = sum;

    			 	  //std::cout << "div_val " << div_val << std::endl;
    			 	 LOOP_SOFTMAX4 : for (int j = 0; j < rnnz[ATEN_BLOCK]; j++) {
    			  		 int row_index2;
    			 		 #pragma HLS PIPELINE II=1
    			  	       for(int k = 0; k<ATEN_BLOCK; k++)
    			  	    		if ((j >= rnnz[k])&&(j < rnnz[k+1]))
    			  	    		 row_index2 = k;
    			    	   val = A_fifo.read();
    				       col = col_indices_fifo.read();
    					   col_att_fifo << col;

    			    	   //if (val>0)
    			    	   //{
    					   //ATYPE out_val = support[j][row_index]/sum[row_index];
    					   ATYPE out_val = support[j]/sum[row_index2];
    			    	   val_att_fifo << out_val;
    			    		  // std::cout << "val_att_fifo " << out_val << std::endl;
    			    	   //}

    			 	 }
    		       }
   	 	 }
	    else
	    {

         for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
	     {

             #pragma HLS DATAFLOW
	         int rnnz[ATEN_BLOCK+1];
		     rnnz[0]=0;
		     int crows = 0;
		     LOOP_RNNZ2 :for (int z = 0; z < ATEN_BLOCK; z++) {
		        rnnz[z+1] = rnnz_fifo[0].read();
		        max_val[z] = max_fifo.read();
		        rnnz_att_fifo[0] << rnnz[z+1]-rnnz[z];
		       	if ((z+i)<N_adj)
		       	  crows++;
		     }
		 	 LOOP_SOFTMAX5 : for (int j = 0; j < rnnz[ATEN_BLOCK]; j++) {
		         #pragma HLS PIPELINE II=1
		    	   val = A_fifo.read();
			       col = col_indices_fifo.read();
				   col_att_fifo << col;
		    	   val_att_fifo << val;
		 	 }
		    }
	       }




	      //attention = torch.where(adj > 0, e, zero_vec)
	      //attention = F.softmax(attention, dim=1)
	      //attention = F.dropout(attention, self.dropout, training=self.training)
}

*/

/*
  good
void compute_attention2(bool gat_mode,int N_adj,
		hls::stream<ATYPE> &A_fifo,
		hls::stream<int> &col_indices_fifo, hls::stream<int> rnnz_fifo[SPMM_BLOCK],hls::stream<FTYPE> &E_fifo,
		hls::stream<FTYPE> &max_fifo,hls::stream<ATYPE> &val_att_fifo,hls::stream<int> &col_att_fifo,hls::stream<int> rnnz_att_fifo[SPMM_BLOCK])
{
		int col;
		ATYPE val,attention_candidate;
		ATYPE fixed_val = ATYPE(-9e3);
        ATYPE sum[ATEN_BLOCK];
    	//ATYPE support[A_WIDTH][ATEN_BLOCK];
    	ATYPE support[A_WIDTH*ATEN_BLOCK];
    	ATYPE fixed_support;
    	ATYPE div_val;
    	ATYPE const_one = ATYPE(1);
    	ATYPE max_val[ATEN_BLOCK];
    	int rnnz[ATEN_BLOCK+1];



    	 if (gat_mode==1)
      	 {

    	   ATEN_LOOP:for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    	   {

                   #pragma HLS DATAFLOW
    		       rnnz[0]=0;
    		       LOOP_RNNZ :for (int z = 0; z < ATEN_BLOCK; z++) {
    		            #pragma  HLS PIPELINE II=1
    		  		   if ((i+z) < N_adj)
    		  		   {
    		    	      sum[z] = 0.0;
    		        	  rnnz[z+1] = rnnz_fifo[0].read();
    		        	  max_val[z] = max_fifo.read();
    		        	  rnnz_att_fifo[0] << rnnz[z+1]-rnnz[z];
    		  		   }
    		  		   else
    		  		   {
    		  			 rnnz[z+1] = rnnz[z];
    		  		   }

    		       	 }


    			  	 //std::cout << "insert rnnz att fifo with " << N_adj << std::endl;
    			  	 //std::cout << "i is " << i << std::endl;




    			 	  LOOP_SOFTMAX2 : for (int j = 0; j < rnnz[ATEN_BLOCK]; j++) {
    			 		int row_index1;
    	                #pragma HLS PIPELINE II=1
    				         attention_candidate = E_fifo.read();
    				         for(int k = 0; k<ATEN_BLOCK; k++)
    				        	 if ((j >= rnnz[k])&&(j < rnnz[k+1]))
    				        		 row_index1 = k;
    	                     #ifdef EIGHTBIT
    				              //support[j][row_index] = hls::exp(attention_candidate- max_val[row_index]);
    				              support[j] = hls::exp(attention_candidate- max_val[row_index1]);
    	                     #else
    				              support[j]= hls::half_exp(attention_candidate- max_val[row_index1]);
    	                     #endif
    				         //std::cout << "support " << support[j] << std::endl;
    				         //sum[row_index] += support[j][row_index];
    				         sum[row_index1] += support[j];

    					 }


    			      LOOP_FIXED :for (int z = 0; z < ATEN_BLOCK; z++) {
                      #pragma HLS PIPELINE II=1
    	              #ifdef EIGHTBIT
    			 	        fixed_support = hls::exp(fixed_val- max_val[z]);
    	              #else
    			 	        fixed_support = hls::half_exp(fixed_val- max_val[z]);
    	              #endif
    			 	  fixed_support = (N_adj-rnnz[z+1]+rnnz[z])*fixed_support;
    			      sum[z] += fixed_support;
    			      }
    			 	  //std::cout << "sum " << sum << std::endl;
    			 	  //div_val = ATYPE(1)/sum;
    			 	  //div_val = const_one/sum;
    			 	  //div_val = sum;

    			 	  //std::cout << "div_val " << div_val << std::endl;
    			 	 LOOP_SOFTMAX4 : for (int j = 0; j < rnnz[ATEN_BLOCK]; j++) {
    			  		 int row_index2;
    			 		 #pragma HLS PIPELINE II=1
    			  	       for(int k = 0; k<ATEN_BLOCK; k++)
    			  	    		if ((j >= rnnz[k])&&(j < rnnz[k+1]))
    			  	    		 row_index2 = k;
    			    	   val = A_fifo.read();
    				       col = col_indices_fifo.read();
    					   col_att_fifo << col;

    			    	   //if (val>0)
    			    	   //{
    					   //ATYPE out_val = support[j][row_index]/sum[row_index];
    					   ATYPE out_val = support[j]/sum[row_index2];
    			    	   val_att_fifo << out_val;
    			    		  // std::cout << "val_att_fifo " << out_val << std::endl;
    			    	   //}

    			 	 }
    		       }
      	 }
    	 else
    	  {

    		     GCN_LOOP: for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    		     {

    	             #pragma HLS DATAFLOW
    		         int rnnz[ATEN_BLOCK+1];
    			     rnnz[0]=0;
    			     int crows = 0;
    			     LOOP_RNNZ2 :for (int z = 0; z < ATEN_BLOCK; z++) {
						#pragma HLS PIPELINE II=1
    			        rnnz[z+1] = rnnz_fifo[0].read();
    			        max_val[z] = max_fifo.read();
    			        rnnz_att_fifo[0] << rnnz[z+1]-rnnz[z];
    			       	if ((z+i)<N_adj)
    			       	  crows++;
    			     }
    			 	 LOOP_SOFTMAX5 : for (int j = 0; j < rnnz[ATEN_BLOCK]; j++) {
    			         #pragma HLS PIPELINE II=1
    			    	   val = A_fifo.read();
    				       col = col_indices_fifo.read();
    					   col_att_fifo << col;
    			    	   val_att_fifo << val;
    			 	 }
    			    }
    		       }







	      //attention = torch.where(adj > 0, e, zero_vec)
	      //attention = F.softmax(attention, dim=1)
	      //attention = F.dropout(attention, self.dropout, training=self.training)
}
*/

void func_rnnz(int i,int N_adj,hls::stream<ATYPE> &max_fifo,hls::stream<int> rnnz_fifo[SPMM_BLOCK],hls::stream<int>  rnnz_f[ATEN_BLOCK],hls::stream<ATYPE> val_f[ATEN_BLOCK])
{
  //#pragma HLS INLINE
//#pragma HLS DATAFLOW
  //std::cout << "loop 1 " << std::endl;
  int rnnz_old = 0;
  int rnnz_val=0;
  ATYPE max_val;

  LOOP_RNNZ :for (int z = 0; z < ATEN_BLOCK; z++) {
		#pragma  HLS PIPELINE II=1

		if ((i+z) < N_adj)
		{
			//std::cout << "i z " << i+z << " " <<  N_adj << std::endl;
			rnnz_val = rnnz_fifo[0].read();
		    rnnz_f[z] << rnnz_val;
		    max_val = max_fifo.read();
		    val_f[z] << max_val;

			std::cout << "rnnz val 1 " << rnnz_val << std::endl;
		 }
		 else
		 {
			std::cout << "rnnz val 2 " << rnnz_val << std::endl;
		    rnnz_f[z] << rnnz_val;
		    val_f[z] << max_val;
		 }
  }
}



void func_exp(hls::stream<int> rnnz_f[ATEN_BLOCK],hls::stream<ATYPE> val_f[ATEN_BLOCK],hls::stream<FTYPE> &E_fifo,
hls::stream<ATYPE>  sum_f[ATEN_BLOCK],hls::stream<ATYPE>  val_f2[ATEN_BLOCK],hls::stream<int>  rnnz_f2[ATEN_BLOCK],
hls::stream<ATYPE>  &support_f)
{
    //#pragma HLS DATAFLOW
    //#pragma HLS INLINE
    std::cout << "loop 2 " << std::endl;
  	int val_rnnz[ATEN_BLOCK+1];
  	ATYPE val_max[ATEN_BLOCK];
	ATYPE support;
	ATYPE attention_candidate;
    ATYPE sum[ATEN_BLOCK];

    val_rnnz[0]=0;
  	LOOP_1 :for (int z = 0; z < ATEN_BLOCK; z++)
  	{
			#pragma HLS UNROLL
            //#pragma HLS PIPELINE II=1
  		    val_rnnz[z+1] = rnnz_f[z].read();
  		    val_max[z] = val_f[z].read();
  		    sum[z]=0;
    }

  	LOOP_SOFTMAX2 : for (int j = 0; j < val_rnnz[ATEN_BLOCK]; j++) {
  			 		int row_index1;
  	                #pragma HLS PIPELINE II=1
  				         attention_candidate = E_fifo.read();
  			  	         for(int k = 0; k<ATEN_BLOCK; k++)
  			    			  if ((j >= val_rnnz[k])&&(j < val_rnnz[k+1]))
  			    			  	 row_index1 = k;
  				         /*if(j < val_rnnz[0])
  				        	 row_index1 = 0;
  				         else if (j < val_rnnz[1])
  	    				     row_index1 = 1;
  				         else if (j < val_rnnz[2])
      	    				 row_index1 = 2;
  				         else
      	    				 row_index1 = 3;*/
  	                     #ifdef EIGHTBIT
  				              //support[j][row_index] = hls::exp(attention_candidate- max_val[row_index]);
  				              support = hls::exp(attention_candidate- val_max[row_index1]);
  	                     #else
  				              support= hls::half_exp(attention_candidate- val_max[row_index1]);
  	                     #endif
  				         //std::cout << "support " << support[j] << std::endl;
  				         //sum[row_index] += support[j][row_index];
  				         sum[row_index1] += support;
  				         support_f << support;

  	}


  	LOOP_2 :for (int z = 0; z < ATEN_BLOCK; z++)
  	{
       #pragma HLS UNROLL
  	   sum_f[z] << sum[z];
       val_f2[z] << val_max[z];
  	   rnnz_f2[z] << val_rnnz[z+1];
  	}

}


void func_fixed(int N_adj,hls::stream<ATYPE>  sum_f[ATEN_BLOCK],hls::stream<ATYPE>  val_f2[ATEN_BLOCK],hls::stream<int>  rnnz_f2[ATEN_BLOCK],
hls::stream<ATYPE>  sum_f2[ATEN_BLOCK],hls::stream<int>  rnnz_f3[ATEN_BLOCK])
{
     //#pragma HLS INLINE
//#pragma HLS DATAFLOW
	 std::cout << "loop 3 " << std::endl;
     int val_rnnz2[ATEN_BLOCK+1];
     ATYPE val_max2[ATEN_BLOCK];
     ATYPE val_sum[ATEN_BLOCK];
  	 ATYPE fixed_val = ATYPE(-9e3);
 	 ATYPE fixed_support;

     val_rnnz2[0] = 0;
  	 LOOP_3 :for (int z = 0; z < ATEN_BLOCK; z++)
  	 {
             #pragma HLS UNROLL
             val_rnnz2[z+1] = rnnz_f2[z].read();
             val_max2[z] = val_f2[z].read();
             val_sum[z] =  sum_f[z].read();
  	 }

   	 LOOP_FIXED :for (int z = 0; z < ATEN_BLOCK; z++) {
           #pragma HLS PIPELINE II=1
   	  	   ATYPE sum_local;
   	       #ifdef EIGHTBIT
   			     fixed_support = hls::exp(fixed_val- val_max2[z]);
   	       #else
   			     fixed_support = hls::half_exp(fixed_val- val_max2[z]);
   	       #endif
   		   fixed_support = (N_adj-val_rnnz2[z+1]+val_rnnz2[z])*fixed_support;
   		   sum_local = val_sum[z]+fixed_support;
   		   sum_f2[z] << sum_local;

   		}
   	LOOP_4 :for (int z = 0; z < ATEN_BLOCK; z++)
   	{
           #pragma HLS UNROLL
   	       rnnz_f3[z] << val_rnnz2[z+1];
   	}

}


void func_div(hls::stream<int> rnnz_att_fifo[SPMM_BLOCK],hls::stream<ATYPE> &A_fifo,hls::stream<ATYPE>  &support_f,
hls::stream<int> &col_indices_fifo,hls::stream<ATYPE> sum_f2[ATEN_BLOCK],hls::stream<int>  rnnz_f3[ATEN_BLOCK],
hls::stream<ATYPE> &val_att_fifo,hls::stream<int> &col_att_fifo)
{
    //#pragma HLS INLINE
//#pragma HLS DATAFLOW
	int col;
	ATYPE val;
	int rnnz_old = 0;
	int rnnz_val=0;

    std::cout << "loop 4 " << std::endl;
 	int val_rnnz3[ATEN_BLOCK+1];
 	ATYPE val_sum2[ATEN_BLOCK];

    val_rnnz3[0]=0;
	LOOP_5 :for (int z = 0; z < ATEN_BLOCK; z++)
	{
       #pragma HLS UNROLL
 	  rnnz_val = rnnz_f3[z].read();
 	  val_rnnz3[z+1] = rnnz_val;
 	  val_sum2[z] = sum_f2[z].read();
 	  rnnz_att_fifo[0] << rnnz_val-rnnz_old;
 	  rnnz_old = rnnz_val;
    }



	LOOP_SOFTMAX4 : for (int j = 0; j < val_rnnz3[ATEN_BLOCK]; j++) {
	  		 int row_index2;
	 		 #pragma HLS PIPELINE II=1
	  	     for(int k = 0; k<ATEN_BLOCK; k++)
	    	   if ((j >= val_rnnz3[k])&&(j < val_rnnz3[k+1]))
	    		  	 row_index2 = k;

	         /*if(j < val_rnnz3[0])
 		      	 row_index2 = 0;
 		     else if (j < val_rnnz3[1])
 	           row_index2 = 1;
 		     else if (j < val_rnnz3[2])
     	  	   row_index2 = 2;
 		     else
     	       row_index2 = 3;*/
	    	  val = A_fifo.read();
		      col = col_indices_fifo.read();
			  col_att_fifo << col;

	    	   //if (val>0)
	    	   //{
			   //ATYPE out_val = support[j][row_index]/sum[row_index];
			   //ATYPE out_val = support_f.read()/val_sum2[row_index2];
			  ATYPE out_val = support_f.read()/val_sum2[row_index2];

	    	   val_att_fifo << out_val;
	    		  // std::cout << "val_att_fifo " << out_val << std::endl;
	    	   //}


	}
}

//#define func_loops

#ifdef func_loops

void compute_attention2(bool gat_mode,int N_adj,
		hls::stream<ATYPE> &A_fifo,
		hls::stream<int> &col_indices_fifo, hls::stream<int> rnnz_fifo[SPMM_BLOCK],hls::stream<ATYPE> &E_fifo,
		hls::stream<ATYPE> &max_fifo,hls::stream<ATYPE> &val_att_fifo,hls::stream<int> &col_att_fifo,hls::stream<int> rnnz_att_fifo[SPMM_BLOCK])


    	hls::stream<int>  rnnz_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f depth=FIFO_DEPTH dim=1

    	hls::stream<int>  rnnz_f2[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f2 depth=FIFO_DEPTH dim=1

    	hls::stream<int>  rnnz_f3[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f3 depth=FIFO_DEPTH dim=1

    	hls::stream<ATYPE>  val_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= val_f depth=FIFO_DEPTH dim=1


    	hls::stream<ATYPE>  val_f2[ATEN_BLOCK];
    	#pragma HLS STREAM variable= val_f2 depth=FIFO_DEPTH dim=1

    	hls::stream<ATYPE>  sum_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= sum_f depth=FIFO_DEPTH dim=1

    	hls::stream<ATYPE>  sum_f2[ATEN_BLOCK];
       	#pragma HLS STREAM variable= sum_f2 depth=FIFO_DEPTH dim=1

       	hls::stream<ATYPE>  support_f;
        #pragma HLS STREAM variable= support_f depth=FIFO_DEPTH_ATTN2 dim=1


    	 if (gat_mode==1)
      	 {

    	   ATEN_LOOP:for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    	   {

                   #pragma HLS DATAFLOW
    		        func_rnnz(i,N_adj,max_fifo,rnnz_fifo,rnnz_f,val_f);

    		        func_exp(rnnz_f,val_f,E_fifo,sum_f,val_f2,rnnz_f2,support_f);

    		        func_fixed(N_adj,sum_f,val_f2,rnnz_f2,sum_f2,rnnz_f3);

    		        func_div(rnnz_att_fifo,A_fifo,support_f,col_indices_fifo,sum_f2,rnnz_f3,val_att_fifo,col_att_fifo);
    	   }

      	 }
    	 else
    	  {
    		     int col;
    		     ATYPE val;

    	         LOOP_GCN : for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    		     {

    	             #pragma HLS DATAFLOW
    		         int rnnz_old=0;
					 int rnnz_val;
    			     LOOP_RNNZ2 :for (int z = 0; z < ATEN_BLOCK; z++) {
						#pragma HLS PIPELINE II=1
    			 		if ((i+z) < N_adj)
    			 		{
    			          rnnz_val = rnnz_fifo[0].read();
    			          ATYPE max_val = max_fifo.read();
    			          rnnz_att_fifo[0] << rnnz_val-rnnz_old;
    			          rnnz_old = rnnz_val;
    			 		}
    			 		//else
    			 		//{
    			 		//  rnnz[z+1] = rnnz[z];
    			 		//}
    			     }
    			 	 LOOP_SOFTMAX5 : for (int j = 0; j < rnnz_old; j++) {
    			         #pragma HLS PIPELINE II=1
    			    	   val = A_fifo.read();
    				       col = col_indices_fifo.read();
    					   col_att_fifo << col;
    			    	   val_att_fifo << val;
    			 	 }
    			    }
    	}



	      //attention = torch.where(adj > 0, e, zero_vec)
	      //attention = F.softmax(attention, dim=1)
	      //attention = F.dropout(attention, self.dropout, training=self.training)
}

#endif

#ifndef func_loops

// good 218K cycles with ATEN BLOCK 1


void compute_attention2(ap_uint<1> model[5][8],int N_adj,
		hls::stream<ATYPE> &A_fifo,
		hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo,hls::stream<TTYPE> &E_fifo,
		hls::stream<TTYPE> &max_fifo,hls::stream<TTYPE> &val_att_fifo,hls::stream<int> &col_att_fifo,hls::stream<int> &rnnz_att_fifo,hls::stream<TTYPE> &SO_fifo,hls::stream<int> &SO_rnnz_fifo,int B_index)
{
		int col;
		TTYPE val,attention_candidate;
		TTYPE fixed_val = TTYPE(-9e3);
	    TTYPE sum[ATEN_BLOCK];
    	TTYPE support;
    	//ATYPE support[A_WIDTH*ATEN_BLOCK];
    	TTYPE fixed_support;
    	TTYPE div_val;
    	TTYPE const_one = TTYPE(1);
    	//ATYPE max_val[ATEN_BLOCK];
    	//int rnnz[ATEN_BLOCK+1];

    	hls::stream<TTYPE>  support_f;
    	#pragma HLS STREAM variable= support_f depth= FIFO_DEPTH_ATTN2
        #pragma HLS bind_storage variable = support_f type=FIFO impl=URAM


    	hls::stream<int>  rnnz_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f depth=FIFO_DEPTH

    	hls::stream<int>  rnnz_f2[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f2 depth=FIFO_DEPTH

    	hls::stream<int>  rnnz_f3[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f3 depth=FIFO_DEPTH

    	hls::stream<TTYPE>  val_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= val_f depth=FIFO_DEPTH


    	hls::stream<TTYPE>  val_f2[ATEN_BLOCK];
    	#pragma HLS STREAM variable= val_f2 depth=FIFO_DEPTH

    	hls::stream<TTYPE>  sum_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= sum_f depth=FIFO_DEPTH

    	hls::stream<TTYPE>  sum_f2[ATEN_BLOCK];
       	#pragma HLS STREAM variable= sum_f2 depth=FIFO_DEPTH

         bool gat_mode = model[B_index][5];

         bool linear_mode = model[B_index][6];

    	 if (linear_mode==0)
      	 {

    	  if (gat_mode==1)
      	  {

    	   ATEN_LOOP:for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    	   {

                   #pragma HLS DATAFLOW
    		       //std::cout << "loop 1 " << std::endl;
    		       int rnnz_old = 0;
    		       ITYPE max_old = 0;
    		       LOOP_RNNZ :for (int z = 0; z < ATEN_BLOCK; z++) {
    		           #pragma  HLS PIPELINE II=1
    		    	   int rnnz_val;
    		    	   ITYPE max_val;
    		  		   if ((i+z) < N_adj)
    		  		   {
    		        	  rnnz_val = rnnz_fifo.read();
    		        	  rnnz_f[z] << rnnz_val;
    		        	  max_val = max_fifo.read();
    		        	  val_f[z] << max_val;
			        	  rnnz_att_fifo << rnnz_val-rnnz_old;
			        	  //SO_rnnz_fifo << rnnz_val-rnnz_old;
			        	  rnnz_old = rnnz_val;
			        	  max_old = max_val;

    		  		   }
    		  		   else
    		  		   {
    		  			 rnnz_f[z] << rnnz_old; //rnnz_val;
    		        	 val_f[z] << max_old; // max_val;
    		  		   }

    		       	 }


    			  	 //std::cout << "insert rnnz att fifo with " << N_adj << std::endl;
    			  	 //std::cout << "i is " << i << std::endl;


    		         //std::cout << "loop 2 " << std::endl;
    		       	  int val_rnnz[4];
    		       	  ATYPE val_max[4];
    		       	  LOOP_1 :for (int z = 0; z < ATEN_BLOCK; z++)
    		       	  {
						#pragma HLS UNROLL
    		            val_rnnz[z] = rnnz_f[z].read();
    		            val_max[z] = val_f[z].read();
    		            sum[z] = 0;

    		       	  }

    			 	  LOOP_SOFTMAX2 : for (int j = 0; j < val_rnnz[ATEN_BLOCK-1]; j++) {
    			 		int row_index1;
    	                #pragma HLS PIPELINE II=1
    				         attention_candidate = E_fifo.read();
    				         if(j < val_rnnz[0])
    				        	 row_index1 = 0;
    				         else if (j < val_rnnz[1])
    	    				     row_index1 = 1;
    				         else if (j < val_rnnz[2])
        	    				 row_index1 = 2;
    				         else
        	    				 row_index1 = 3;
    	                     #ifdef EIGHTBIT
    				              //support[j][row_index] = hls::exp(attention_candidate- max_val[row_index]);
    				              //support[j] = hls::exp(attention_candidate- val_max[row_index1]);
    				              support = hls::exp(attention_candidate- val_max[row_index1]);
    	                     #else
    				              support= hls::half_exp(attention_candidate- val_max[row_index1]);
    	                     #endif
    				         //std::cout << "support " << support[j] << std::endl;
    				         //sum[row_index] += support[j][row_index];
    				         //sum[row_index1] += support[j];
    				         sum[row_index1] += support;
    				         support_f << support;

    					 }

    		     	  LOOP_2 :for (int z = 0; z < ATEN_BLOCK; z++)
    		     	  {
                       #pragma HLS UNROLL
    			 	   sum_f[z] << sum[z];
       			 	   val_f2[z] << val_max[z];
    			 	   rnnz_f2[z] << val_rnnz[z];
    		     	  }



    		     	 //std::cout << "loop 3 " << std::endl;
        		     int val_rnnz2[ATEN_BLOCK+1];
        		     TTYPE val_max2[ATEN_BLOCK];
        			 TTYPE val_sum[ATEN_BLOCK];

        		     val_rnnz2[0] = 0;
   		     	     LOOP_3 :for (int z = 0; z < ATEN_BLOCK; z++)
   		     	     {
                         #pragma HLS UNROLL
       			         val_rnnz2[z+1] = rnnz_f2[z].read();
       			         val_max2[z] = val_f2[z].read();
                         val_sum[z] =  sum_f[z].read();
   		     	     }

    			      LOOP_FIXED :for (int z = 0; z < ATEN_BLOCK; z++) {
    	  			  TTYPE sum_local;
                      #pragma HLS PIPELINE II=1
    	              #ifdef EIGHTBIT
    			 	        fixed_support = hls::exp(fixed_val- val_max2[z]);
    	              #else
    			 	        fixed_support = hls::half_exp(fixed_val- val_max2[z]);
    	              #endif
    			 	  fixed_support = (N_adj-val_rnnz2[z+1]+val_rnnz2[z])*fixed_support;
    			      sum_local = val_sum[z]+fixed_support;
    			      sum_f2[z] << sum_local;

    			      }
    		     	  LOOP_4 :for (int z = 0; z < ATEN_BLOCK; z++)
    	   		      {
                         #pragma HLS UNROLL
    			 	     rnnz_f3[z] << val_rnnz2[z+1];
    	   		      }
    			 	  //std::cout << "sum " << sum << std::endl;
    			 	  //div_val = ATYPE(1)/sum;
    			 	  //div_val = const_one/sum;
    			 	  //div_val = sum;

    			 	  //std::cout << "div_val " << div_val << std::endl;


       		       //std::cout << "loop 4 " << std::endl;
    		    	int val_rnnz3[4];
    		    	TTYPE val_sum2[4];
   	     	        LOOP_5 :for (int z = 0; z < ATEN_BLOCK; z++)
      		     	{
                      #pragma HLS UNROLL
    		    	  val_rnnz3[z] = rnnz_f3[z].read();
    		    	  val_sum2[z] = sum_f2[z].read();
      		     	}

    			 	 LOOP_SOFTMAX4 : for (int j = 0; j < val_rnnz3[ATEN_BLOCK-1]; j++) {
    			  		 int row_index2;
    			 		 #pragma HLS PIPELINE II=1
    			         if(j < val_rnnz3[0])
    		    		      	 row_index2 = 0;
    		    		 else if (j < val_rnnz3[1])
    		    	           row_index2 = 1;
    		    		 else if (j < val_rnnz3[2])
    		        	  	 row_index2 = 2;
    		    		 else
    		        	     row_index2 = 3;
    			    	   val = (ATYPE)(A_fifo.read());
    				       col = col_indices_fifo.read();
    					   col_att_fifo << col;

    			    	   //if (val>0)
    			    	   //{
    					   //ATYPE out_val = support[j][row_index]/sum[row_index];
    					   //ATYPE out_val = support[j]/val_sum2[row_index2];
    					   TTYPE out_val = support_f.read()/val_sum2[row_index2];
    			    	   //val_att_fifo << (ATYPE)(out_val); //cast to A type NO CASTING KEEP PRECISON FOR ATTENTION
    			    	   val_att_fifo << out_val; //NO CASTING KEEP PRECISON FOR ATTENTION
    			    	   //val_att_fifo << out_val; //cast to A type
    			    	   SO_fifo << out_val;
    			    		  // std::cout << "val_att_fifo " << out_val << std::endl;
    			    	   //}


    			 	 }
    		       }
      	 }
    	 else

    	  {

    	         LOOP_GCN : for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    		     {

    	             #pragma HLS DATAFLOW
    		         int rnnz[ATEN_BLOCK+1];
    			     rnnz[0]=0;
    			     int crows = 0;
    			     LOOP_RNNZ2 :for (int z = 0; z < ATEN_BLOCK; z++) {
						#pragma HLS PIPELINE II=1
    			        rnnz[z+1] = rnnz_fifo.read();
    			        //ATYPE max_val = max_fifo.read();
    			        rnnz_att_fifo << rnnz[z+1]-rnnz[z];
    			       	if ((z+i)<N_adj)
    			       	  crows++;
    			     }
    			 	 LOOP_SOFTMAX5 : for (int j = 0; j < rnnz[ATEN_BLOCK]; j++) {
    			         #pragma HLS PIPELINE II=1
    			    	   val = A_fifo.read();
    				       col = col_indices_fifo.read();
    					   col_att_fifo << col;
    			    	   val_att_fifo << (TTYPE)(val);
    			 	 }
    			    }
    		}

         } //linear mode







	      //attention = torch.where(adj > 0, e, zero_vec)
	      //attention = F.softmax(attention, dim=1)
	      //attention = F.dropout(attention, self.dropout, training=self.training)
}
#endif



/*
void compute_attention2(bool gat_mode,int N_adj,
		hls::stream<ATYPE> &A_fifo,
		hls::stream<int> &col_indices_fifo, hls::stream<int> rnnz_fifo[SPMM_BLOCK],hls::stream<FTYPE> &E_fifo,
		hls::stream<FTYPE> &max_fifo,hls::stream<ATYPE> &val_att_fifo,hls::stream<int> &col_att_fifo,hls::stream<int> rnnz_att_fifo[SPMM_BLOCK])
{
		int rnnz,col;
		ATYPE val,attention_candidate;
		ATYPE fixed_val = ATYPE(-9e3);
        ATYPE max_val, sum;
    	ATYPE support[A_WIDTH];
    	ATYPE div_val;
    	ATYPE const_one = ATYPE(1);



	 	 if (gat_mode==1)
	 	 {

    	   for (int i = 0; i < N_adj; i++)
    	   {


                     #pragma HLS DATAFLOW
    		         rnnz = rnnz_fifo[0].read();
    		         rnnz_att_fifo[0] << rnnz;
    			  	 //std::cout << "insert rnnz att fifo with " << N_adj << std::endl;
    			  	 //std::cout << "i is " << i << std::endl;

    			 	 sum = 0.0;
    			 	 max_val = max_fifo.read();

    			 	  LOOP_SOFTMAX2 : for (int j = 0; j < rnnz; j++) {
    	                #pragma HLS PIPELINE II=1
    				         attention_candidate = E_fifo.read();
    	                     #ifdef EIGHTBIT
    				              support[j] = hls::exp(attention_candidate- max_val);
    	                     #else
    				              support[j] = hls::half_exp(attention_candidate- max_val);
    	                     #endif
    				         //std::cout << "support " << support[j] << std::endl;
    				         sum += support[j];
    					 }
    	              #ifdef EIGHTBIT
    			 	        ATYPE fixed_support = hls::exp(fixed_val- max_val);
    	              #else
    			 	        ATYPE fixed_support = hls::half_exp(fixed_val- max_val);
    	              #endif
    			 	  fixed_support = (N_adj-rnnz)*fixed_support,
    			      sum += fixed_support;
    			 	  //std::cout << "sum " << sum << std::endl;
    			 	  //div_val = ATYPE(1)/sum;
    			 	  //div_val = const_one/sum;
    			 	  //div_val = sum;

    			 	  //std::cout << "div_val " << div_val << std::endl;
    			 	 LOOP_SOFTMAX4 : for (int j = 0; j < rnnz; j++) {
    			 		 #pragma HLS PIPELINE II=1 rewind
    			    	   val = A_fifo.read();
    				       col = col_indices_fifo.read();
    					   col_att_fifo << col;

    			    	   //if (val>0)
    			    	   //{
    					   ATYPE out_val = support[j]/sum;
    			    	   val_att_fifo << out_val;
    			    		  // std::cout << "val_att_fifo " << out_val << std::endl;
    			    	   //}

    			 	 }
    		       }
	 	}
    	else
    	{

        for (int i = 0; i < N_adj; i++)
   	     {

             #pragma HLS DATAFLOW
   	         rnnz = rnnz_fifo[0].read();
   	         rnnz_att_fifo[0] << rnnz;
   		  	 //std::cout << "insert rnnz att fifo with " << N_adj << std::endl;
   		  	 //std::cout << "i is " << i << std::endl;

   		 	 sum = 0.0;
   		 	 max_val = max_fifo.read();

   		 	 LOOP_SOFTMAX5 : for (int j = 0; j < rnnz; j++) {
   		         #pragma HLS PIPELINE II=1
   		    	   val = A_fifo.read();
   			       col = col_indices_fifo.read();
   				   col_att_fifo << col;
   		    	   val_att_fifo << val;
   		 	 }
   	       }
    	}






	      //attention = torch.where(adj > 0, e, zero_vec)
	      //attention = F.softmax(attention, dim=1)
	      //attention = F.dropout(attention, self.dropout, training=self.training)
}
 */

/*

void compute_attention2(bool gat_mode,int N_adj,
		hls::stream<ATYPE> &A_fifo,
		hls::stream<int> &col_indices_fifo, hls::stream<int> rnnz_fifo[SPMM_BLOCK],hls::stream<FTYPE> &E_fifo,
		hls::stream<FTYPE> &max_fifo,hls::stream<ATYPE> &val_att_fifo,hls::stream<int> &col_att_fifo,hls::stream<int> rnnz_att_fifo[SPMM_BLOCK])
{
		int rnnz,col;
		ATYPE val,attention_candidate;
		ATYPE fixed_val = -9e3;
        ATYPE max_val, sum;
    	ATYPE support[A_WIDTH];


	     for (int i = 0; i < N_adj; i++)
	     {

             //#pragma HLS DATAFLOW
	         rnnz = rnnz_fifo[0].read();
	         rnnz_att_fifo[0] << rnnz;
		  	 //std::cout << "insert rnnz att fifo with " << N_adj << std::endl;
		  	 //std::cout << "i is " << i << std::endl;

		 	 sum = 0.0;
		 	 max_val = max_fifo.read();
		 	 if (gat_mode==1)
		 	 {
		 	  LOOP_SOFTMAX2 : for (int j = 0; j < rnnz; j++) {
                #pragma HLS PIPELINE II=1
			         attention_candidate = E_fifo.read();
                     #ifdef EIGHTBIT
			              support[j] = hls::exp(attention_candidate- max_val);
                     #else
			              support[j] = hls::half_exp(attention_candidate- max_val);
                     #endif
			         sum += support[j];
				 }
              #ifdef EIGHTBIT
		 	        ATYPE fixed_support = hls::exp(fixed_val- max_val);
              #else
		 	        ATYPE fixed_support = hls::half_exp(fixed_val- max_val);
              #endif
		 	  fixed_support = (N_adj-rnnz)*fixed_support,
		      sum += fixed_support;
		 	  ATYPE div_val = ATYPE(1)/sum;

		 	  LOOP_SOFTMAX3 :  for (int j = 0; j < rnnz; j++) {
		         #pragma HLS PIPELINE II=1
		 		support[j]=support[j]*div_val;
		 	  }
		 	 }
		 	 LOOP_SOFTMAX4 : for (int j = 0; j < rnnz; j++) {
		         #pragma HLS PIPELINE II=1
		    	   val = A_fifo.read();
			       col = col_indices_fifo.read();
				   col_att_fifo << col;
		    	   if (gat_mode==1)
		    	   //if (val>0)
		    	   //{
		    		   val_att_fifo << support[j];
		    		   //std::cout << "val_att_fifo " << support[j] << std::endl;
		    	   //}
		    	   else
		    		   val_att_fifo << val;
		 	 }
	       }



	      //attention = torch.where(adj > 0, e, zero_vec)
	      //attention = F.softmax(attention, dim=1)
	      //attention = F.dropout(attention, self.dropout, training=self.training)
}
*/

/*
void compute_attention(int N_adj,hls::stream<ATYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> rnnz_fifo[SPMM_BLOCK],hls::stream<FTYPE> &E_fifo,
		hls::stream<ATYPE> &val_att_fifo,hls::stream<int> &col_att_fifo,hls::stream<int> rnnz_att_fifo[SPMM_BLOCK])
{
		int rnnz,col;
		ATYPE val,attention_candidate;
		ATYPE fixed_val = -9e3;
        bool vfull; //keep track if val is used or not
        ATYPE support[A_WIDTH];

	     for (int i = 0; i < N_adj; i++)
	     {
	         rnnz = rnnz_fifo[0].read();
	         vfull = 0;
	         rnnz_att_fifo[0] << rnnz;
		  	 //std::cout << "insert rnnz att fifo with " << N_adj << std::endl;
		  	 //std::cout << "i is " << i << std::endl;
		  	 for (int j = 0; j < N_adj; j++)
		  	 {
	  		  	#pragma HLS PIPELINE
		    	 if ((vfull == 0)&&(rnnz > 0))
		    	 {
		    		 val = A_fifo.read();
		    		 col = col_indices_fifo.read();
		 		     col_att_fifo << col;
		    		 rnnz--;
		    		 vfull = 1;
		    	 }

		         attention_candidate = E_fifo.read();
			  	 //std::cout << "attention candidate " << attention_candidate << std::endl;
		    	 if (col == j)
				 {
		    		    vfull = 0; //we use the data in val
					    if (val > 0)
					    	support[j] = attention_candidate;
					    else
					    	support[j] = fixed_val;
	  		  	 }
		    	 else
		    	 {
		    		 support[j] = fixed_val;
		    		 //std::cout << "val is " << fixed_val << std::endl;
		    	 }
		     }
		  	 softmax(N_adj, support,val_att_fifo);

	     }

	      //attention = torch.where(adj > 0, e, zero_vec)
	      //attention = F.softmax(attention, dim=1)
	      //attention = F.dropout(attention, self.dropout, training=self.training)
}
*/

void loop_attention(float deq_factor[5],int beta_qu,int f_align,float quantization_scale_adj,float quantization_scale_w[5],
		ap_uint<1> model[5][8],
		int nnz_adj1,int nnz_adj2,int nnz_adj3,int nnz_adj4,
		int * rowPtr_adj1,int * rowPtr_adj2,int * rowPtr_adj3,int * rowPtr_adj4,
		int *columnIndex_adj1, int *columnIndex_adj2, int *columnIndex_adj3, int *columnIndex_adj4,
		INTYPE *values_adj1, 	INTYPE *values_adj2,	INTYPE *values_adj3,	INTYPE *values_adj4,
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

        /*hls::stream<int>   rnnz_thread0("rnnz_thread0");
        #pragma HLS STREAM variable=rnnz_thread0 depth=1
	    hls::stream<int>   rnnz_thread1("rnnz_thread1");
        #pragma HLS STREAM variable=rnnz_thread1 depth=1
	    hls::stream<int>   rnnz_thread2("rnnz_thread2");
        #pragma HLS STREAM variable=rnnz_thread2 depth=1
	    hls::stream<int>   rnnz_thread3("rnnz_thread3");
        #pragma HLS STREAM variable=rnnz_thread3 depth=1
	    hls::stream<int>   rnnz_thread4("rnnz_thread4");
        #pragma HLS STREAM variable=rnnz_thread4 depth=1*/



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
         //This seems the adj size which is problematic to store
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
         //This seems the adj size which is problematic to store
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
         //This seems the adj size which is problematic to store
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
         //This seems the adj size which is problematic to store
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
         //BTYPE ate_m2[2*C_WIDTH];
         //BTYPE ate_m3[2*C_WIDTH];
         //BTYPE ate_m4[2*C_WIDTH];

         //std::cout << "inside attention " <<  layer_loop << std::endl;

         #if (PIPO_BLOCKS>=2)

	 	   LOOP_ATTN : for (int B_index = 0; B_index < layer_loop; B_index++) {
         #else
	       int B_index = 0;
         #endif

	 	std::cout << "attention layer " << B_index << std::endl;

	    #pragma HLS DATAFLOW

        #if ADJ_THREADS == 1

	 	 //load A (trained parameters in pygat)

		  for (int j = 0; j < 2*B_WIDTH_BLOCK; j++) {
								#pragma HLS PIPELINE
	     	                    BTYPE ate_temp;
	   	                        INTYPE AF = A[j];
	                            #if (INT_QUANT_W == 1)
		  			        	   quantw(ate_temp,AF,quantization_scale_w,f_align,beta_qu,B_index);
	                            #else
		  			        	   ate_temp = AF;
                                #endif
			  			        ate_m1[j] = ate_temp;
			  			        //std::cout << "A is " << A[j] << std::endl;
			  			        //std::cout << "ate_m1 is " << ate_m1[j] << std::endl;
		 }



	  	 //stream adj



		int first_row1;//,first_row2;//,first_row3,first_row4;
		int row_count1;//,row_count2;//,row_count3,row_count4;

		int N_adj_block = N_adj/ADJ_THREADS;
		int N_adj_block_compute = N_adj/FEA_THREADS; // in compute2 each block only contains  N_adj/FEA_THREADS elements
		//int N_adj_rest = N_adj%2;
		row_count1 = N_adj_block;
		//row_count2 = N_adj_block;
		//row_count3 = N_adj_block;
		//row_count4 = N_adj_block+N_adj_rest;
		first_row1 = 0;
		//first_row2 = N_adj_block;
		//first_row3 = 2*N_adj_block;
		//first_row4 = 3*N_adj_block;



       #if GAT_ENABLE == 1
	  	 std::cout << "Read ADJ data" << std::endl;

         #if (COO_MODE == 0)
	 	   reada2_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1_total_e,rnnz_fifo_adj1_total_s,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1);
         #else
	 	   reada2_coo(nnz_adj1,beta_qu,f_align,quantization_scale_adj,model,M_adj,first_row1,row_count1,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1_total_e,rnnz_fifo_adj1_total_s,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1,B_index);
	 	 #endif

	 	 //generate stream with attention

         #if (FAST_ATTENTION==1)

	  	 std::cout << "prepare fast attention mechanism" << std::endl;

	     prepare_attentional_mechanism_input2(model,row_count1, P_w,A_buffer11,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1,E_col_indices_fifo1,E_rnnz_fifo1,A_fifo1,E_fifo1,max_fifo1,ate_m1,EO_fifo1,EO_rnnz_fifo1,B_index);

	     //zero_vec = -9e15*torch.ones_like(e)

	     std::cout << "write e out" << std::endl;
	     //write e out
         //int rnnz_total_e = 0;

	     //writes(gat_mode,first_row1,row_count1,N_adj,P_w, EO_fifo1,EO_rnnz_fifo1, E1,B_index);


	     writes(deq_factor,model,first_row1,row_count1,N_adj,P_w, EO_fifo1,rnnz_fifo_adj1_total_e, E1,B_index);



		 //compute attention

	 	 std::cout << "compute fast attention" << std::endl;

	     compute_attention2(model,row_count1,A_fifo1, E_col_indices_fifo1, E_rnnz_fifo1,E_fifo1,max_fifo1,val_att_fifo1,col_att_fifo1, rnnz_att_fifo1,SO_fifo1,SO_rnnz_fifo1,B_index);

	 	 std::cout << "done fast attention" << std::endl;

	     //write s out
	 	 std::cout << "write s out" << std::endl;
         //int rnnz_total_s = 0;
	     //writes(gat_mode,first_row1,row_count1,N_adj,P_w, SO_fifo1, SO_rnnz_fifo1,S1,B_index);
	     float deq_dummy[5] = {1.0};
	 	 writes(deq_dummy,model,first_row1,row_count1,N_adj,P_w, SO_fifo1, rnnz_fifo_adj1_total_s,S1,B_index);



         #else

	     prepare_attentional_mechanism_input(N_adj, P_w,C_buffer,E_fifo,ate_m);

		 //compute attention

		 std::cout << "compute attention" << std::endl;

		 compute_attention(N_adj,A_fifo_adj, col_indices_fifo_adj, rnnz_fifo_adj,E_fifo,val_att_fifo,col_att_fifo, rnnz_att_fifo);

		 std::cout << "done attention" << std::endl;

         #endif

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

         #if ADJ_THREADS == 4


			  for (int j = 0; j < 2*B_WIDTH_BLOCK; j++) {
										#pragma HLS PIPELINE
					  			        ate_m1[j] = A[j];
					  			        //ate_m2[j] = A[j];
					  			        //ate_m3[j] = A[j];
					  			        //ate_m4[j] = A[j];
					  			        //std::cout << "ate_m is " << ate_m[j] << std::endl;
				 }



			  	 //stream adj


				int first_row1,first_row2,first_row3,first_row4;
				int row_count1,row_count2,row_count3,row_count4;

				int N_adj_block = N_adj/ADJ_THREADS;
				int N_adj_block_compute = N_adj/FEA_THREADS; // in compute2 each block only contains  N_adj/FEA_THREADS elements
				int N_adj_rest = N_adj%4;
				row_count1 = N_adj_block;
				row_count2 = N_adj_block;
				row_count3 = N_adj_block;
				row_count4 = N_adj_block+N_adj_rest;
				first_row1 = 0;
				first_row2 = N_adj_block;
				first_row3 = 2*N_adj_block;
				first_row4 = 3*N_adj_block;



                #if GAT_ENABLE == 1
			  	 std::cout << "Read ADJ data" << std::endl;



                 #if (COO_MODE == 0)
			 	  reada2_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1_total_e,rnnz_fifo_adj1_total_s,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1);
			 	  reada2_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row2,row_count2,A_fifo_adj2,col_indices_fifo_adj2,rnnz_fifo_adj2_total_e,rnnz_fifo_adj2_total_s,rnnz_fifo_adj2,rowPtr_adj2,columnIndex_adj2,values_adj2);
			 	  reada2_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row3,row_count3,A_fifo_adj3,col_indices_fifo_adj3,rnnz_fifo_adj3_total_e,rnnz_fifo_adj3_total_s,rnnz_fifo_adj3,rowPtr_adj3,columnIndex_adj3,values_adj3);
			 	  reada2_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row4,row_count4,A_fifo_adj4,col_indices_fifo_adj4,rnnz_fifo_adj4_total_e,rnnz_fifo_adj4_total_s,rnnz_fifo_adj4,rowPtr_adj4,columnIndex_adj4,values_adj4);
                 #else
			 	  reada2_coo(nnz_adj1,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1_total_e,rnnz_fifo_adj1_total_s,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1);
		    	  reada2_coo(nnz_adj2,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row2,row_count2,A_fifo_adj2,col_indices_fifo_adj2,rnnz_fifo_adj2_total_e,rnnz_fifo_adj2_total_s,rnnz_fifo_adj2,rowPtr_adj2,columnIndex_adj2,values_adj2);
			 	  reada2_coo(nnz_adj3,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row3,row_count3,A_fifo_adj3,col_indices_fifo_adj3,rnnz_fifo_adj3_total_e,rnnz_fifo_adj3_total_s,rnnz_fifo_adj3,rowPtr_adj3,columnIndex_adj3,values_adj3);
			 	  reada2_coo(nnz_adj4,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row4,row_count4,A_fifo_adj4,col_indices_fifo_adj4,rnnz_fifo_adj4_total_e,rnnz_fifo_adj4_total_s,rnnz_fifo_adj4,rowPtr_adj4,columnIndex_adj4,values_adj4);
                 #endif

			 	 //generate stream with attention

		         #if (FAST_ATTENTION==1)

			  	 std::cout << "prepare fast attention mechanism" << std::endl;


			     //prepare_attentional_mechanism_input2(gat_mode,row_count1, P_w,A_buffer11,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1,E_col_indices_fifo1,E_rnnz_fifo1,A_fifo1,E_fifo1,max_fifo1,ate_m1,EO_fifo1,EO_rnnz_fifo1);
			     //prepare_attentional_mechanism_input2(gat_mode,row_count2, P_w,A_buffer21,A_fifo_adj2,col_indices_fifo_adj2,rnnz_fifo_adj2,E_col_indices_fifo2,E_rnnz_fifo2,A_fifo2,E_fifo2,max_fifo2,ate_m2,EO_fifo2,EO_rnnz_fifo2);
			     //prepare_attentional_mechanism_input2(gat_mode,row_count3, P_w,A_buffer31,A_fifo_adj3,col_indices_fifo_adj3,rnnz_fifo_adj3,E_col_indices_fifo3,E_rnnz_fifo3,A_fifo3,E_fifo3,max_fifo3,ate_m3,EO_fifo3,EO_rnnz_fifo3);
			     //prepare_attentional_mechanism_input2(gat_mode,row_count4, P_w,A_buffer41,A_fifo_adj4,col_indices_fifo_adj4,rnnz_fifo_adj4,E_col_indices_fifo4,E_rnnz_fifo4,A_fifo4,E_fifo4,max_fifo4,ate_m4,EO_fifo4,EO_rnnz_fifo4);

			     prepare_attentional_mechanism_inputx4(gat_mode,row_count1, row_count2, row_count3, row_count4, P_w,A_buffer11,A_buffer21,A_buffer31,A_buffer41,
			     A_fifo_adj1,A_fifo_adj2,A_fifo_adj3,A_fifo_adj4,
				 col_indices_fifo_adj1, col_indices_fifo_adj2, col_indices_fifo_adj3, col_indices_fifo_adj4,
				 rnnz_fifo_adj1,rnnz_fifo_adj2,rnnz_fifo_adj3,rnnz_fifo_adj4,
				 E_col_indices_fifo1, E_col_indices_fifo2, E_col_indices_fifo3, E_col_indices_fifo4,
				 E_rnnz_fifo1,E_rnnz_fifo2,E_rnnz_fifo3,E_rnnz_fifo4,
				 A_fifo1, A_fifo2, A_fifo3, A_fifo4,
				 E_fifo1, E_fifo2, E_fifo3, E_fifo4,
				 max_fifo1, max_fifo2, max_fifo3, max_fifo4,
				 ate_m1,
				 EO_fifo1,EO_fifo2,EO_fifo3,EO_fifo4,
				 EO_rnnz_fifo1,EO_rnnz_fifo2,EO_rnnz_fifo3,EO_rnnz_fifo4);

			     writesx4(deq_factor,gat_mode,row_count1,row_count2,row_count3,row_count4,
			    	    EO_fifo1,EO_fifo2,EO_fifo3,EO_fifo4,
						rnnz_fifo_adj1_total_e,rnnz_fifo_adj2_total_e,rnnz_fifo_adj3_total_e,rnnz_fifo_adj4_total_e,
			     		E1,B_index);



			     //writes2(gat_mode,rnnz_thread0,first_row1,row_count1,N_adj,P_w, EO_fifo1,EO_rnnz_fifo1, E1,B_index);
			     //std::cout << "rnnz total e " << rnnz_total_e << std::endl;
			     //writes(gat_mode,rnnz_thread1,first_row2,row_count2,N_adj,P_w, EO_fifo2,EO_rnnz_fifo2, E2,B_index);
			     //std:ccout << "rnnz total e " << rnnz_total_e << std::endl;
			     //writes2(gat_mode,rnnz_thread2,first_row3,row_count3,N_adj,P_w, EO_fifo3,EO_rnnz_fifo3, E3,B_index);
			     //std::cout << "rnnz total e " << rnnz_total_e << std::endl;
			     //writes2(gat_mode,rnnz_thread3,first_row4,row_count4,N_adj,P_w, EO_fifo4,EO_rnnz_fifo4, E4,B_index);


			     //zero_vec = -9e15*torch.ones_like(e)

				 //compute attention

			 	 std::cout << "compute fast attention" << std::endl;

			     compute_attention2(gat_mode,row_count1,A_fifo1, E_col_indices_fifo1, E_rnnz_fifo1,E_fifo1,max_fifo1,val_att_fifo1,col_att_fifo1, rnnz_att_fifo1,SO_fifo1,SO_rnnz_fifo1);
			     compute_attention2(gat_mode,row_count2,A_fifo2, E_col_indices_fifo2, E_rnnz_fifo2,E_fifo2,max_fifo2,val_att_fifo2,col_att_fifo2, rnnz_att_fifo2,SO_fifo2,SO_rnnz_fifo2);
			     compute_attention2(gat_mode,row_count3,A_fifo3, E_col_indices_fifo3, E_rnnz_fifo3,E_fifo3,max_fifo3,val_att_fifo3,col_att_fifo3, rnnz_att_fifo3,SO_fifo3,SO_rnnz_fifo3);
			     compute_attention2(gat_mode,row_count4,A_fifo4, E_col_indices_fifo4, E_rnnz_fifo4,E_fifo4,max_fifo4,val_att_fifo4,col_att_fifo4, rnnz_att_fifo4,SO_fifo4,SO_rnnz_fifo4);

			     //int rnnz_total_s = 0;
			     //writes1(gat_mode,first_row1,row_count1,N_adj,P_w, SO_fifo1, SO_rnnz_fifo1,S1,B_index);
			     //std:cout << "rnnz total s " << rnnz_total_s << std::endl;
			     //writes1(gat_mode,first_row2,row_count2,N_adj,P_w, SO_fifo2, SO_rnnz_fifo2,S2,B_index);
			     //std::cout << "rnnz total s " << rnnz_total_s << std::endl;
			     //writes1(gat_mode,first_row3,row_count3,N_adj,P_w, SO_fifo3, SO_rnnz_fifo3,S3,B_index);
			     //std::cout << "rnnz total s " << rnnz_total_s << std::endl;
			     //writes1(gat_mode,first_row4,row_count4,N_adj,P_w, SO_fifo4, SO_rnnz_fifo4,S4,B_index);

			     float deq_dummy = 1.0;
			     writesx4(deq_dummy,gat_mode,row_count1,row_count2,row_count3,row_count4,
				 SO_fifo1,SO_fifo2,SO_fifo3,SO_fifo4,
				 rnnz_fifo_adj1_total_s,rnnz_fifo_adj2_total_s,rnnz_fifo_adj3_total_s,rnnz_fifo_adj4_total_s,
				 S1,B_index);


			 	 std::cout << "done fast attention" << std::endl;

		         #else

			     prepare_attentional_mechanism_input(N_adj, P_w,C_buffer,E_fifo,ate_m);

				 //compute attention

				 std::cout << "compute attention" << std::endl;

				 compute_attention(N_adj,A_fifo_adj, col_indices_fifo_adj, rnnz_fifo_adj,E_fifo,val_att_fifo,col_att_fifo, rnnz_att_fifo);

				 std::cout << "done attention" << std::endl;

                  #endif

                #else //GAT DISABLE
                std::cout << "Read ADJ data" << std::endl;
                //hls::stream<ATYPE> val_att_fifo1_int;
                #if (COO_MODE == 0)
                  reada22_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,val_att_fifo1,col_att_fifo1,rnnz_att_fifo1,rowPtr_adj1,columnIndex_adj1,values_adj1);
                  reada22_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row2,row_count2,val_att_fifo2,col_att_fifo2,rnnz_att_fifo2,rowPtr_adj2,columnIndex_adj2,values_adj2);
                  reada22_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row3,row_count3,val_att_fifo3,col_att_fifo3,rnnz_att_fifo3,rowPtr_adj3,columnIndex_adj3,values_adj3);
                  reada22_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row4,row_count4,val_att_fifo4,col_att_fifo4,rnnz_att_fifo4,rowPtr_adj4,columnIndex_adj4,values_adj4);
               #else
                  reada22_coo(nnz_adj1,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,val_att_fifo1,col_att_fifo1,rnnz_att_fifo1,rowPtr_adj1,columnIndex_adj1,values_adj1);
                  reada22_coo(nnz_adj2,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row2,row_count2,val_att_fifo2,col_att_fifo2,rnnz_att_fifo2,rowPtr_adj2,columnIndex_adj2,values_adj2);
                  reada22_coo(nnz_adj3,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row3,row_count3,val_att_fifo3,col_att_fifo3,rnnz_att_fifo3,rowPtr_adj3,columnIndex_adj3,values_adj3);
                  reada22_coo(nnz_adj4,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row4,row_count4,val_att_fifo4,col_att_fifo4,rnnz_att_fifo4,rowPtr_adj4,columnIndex_adj4,values_adj4);
               #endif

             #endif
             #endif



         #if (PIPO_BLOCKS>=2)
	 	   }
        #endif

	     //attention = torch.where(adj > 0, e, zero_vec)
	     //attention = F.softmax(attention, dim=1)
	     //attention = F.dropout(attention, self.dropout, training=self.training)
}

void readb(bool load_weights,ap_uint<1> model[5][8],int beta_qu,int f_align,float quantization_scale_w[5],int M_fea,ap_uint<8> P_w[5],int B_index,BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],INTYPES* B)
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

	bool linear_mode = model[B_index][6];
    bool sage_mode = model[B_index][7];
	bool gcn_path = !(linear_mode^sage_mode);

	 
	bool load_weights_gcn = load_weights & gcn_path;


	if(load_weights_gcn==1)
	  {

	     //LOOP_BLOCKB1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
		 LOOP_BLOCKB1 : for (int j = 0; j < P_w[B_index]; j++) {
			        LOOP_BLOCKB2 : for (int i = 0; i < M_fea_current; i++) {
						    //#pragma HLS loop_tripcount min=84 max=84 avg=84
							#pragma HLS PIPELINE
							//#pragma HLS loop_tripcount min=16 max=16 avg=16
			        	    //INTYPES B_TEMP = B[i+j*M_fea+B_index*B_WIDTH_BLOCK*M_fea];
			        	    //BTYPE B_accel_temp = B_TEMP.range(7,4);

			        	    INTYPE BF = (INTYPE)B[i+j*M_fea_current+B_shift];
			        	    BTYPE B_accel_temp;
                          #if (INT_QUANT_W == 1)
			        	     quantw(B_accel_temp,BF,quantization_scale_w,f_align,beta_qu,B_index);
                          #else
			        	     B_accel_temp = BF;
                          #endif
			        		//BTYPE B_accel_temp = (BTYPE)B[i+j*M_fea+B_index*B_WIDTH_BLOCK*M_fea];


			        		B_accel[i][j] = B_accel_temp;
			        		//B_accel2[i][j] = B_accel_temp;
			        		//B_accel3[i][j] = B_accel_temp;
			        		//B_accel4[i][j] = B_accel_temp;

							//std::cout << " " << i << " " << j << " " <<  B_accel_temp  << " " << std::endl;
			        		//printf("%d %d %f %f \n",i,j,B_accel_temp,B[i+j*M_fea+B_index*B_WIDTH_BLOCK*M_fea]);
			          }
		  }

	  }
}

void readbl(bool load_weights,ap_uint<1> model[5][8],int beta_qu,int f_align,float quantization_scale_w[5],int M_fea,ap_uint<8> P_w[5],int B_index,BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],INTYPES* B)
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


    //std::cout << "M fea current " <<  M_fea_current << std::endl;

	bool linear_mode = model[B_index][6];

	 
	bool load_weights_linear = load_weights & linear_mode;

	if(load_weights_linear==1)
	  {

	     //LOOP_BLOCKB1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
		 LOOP_BLOCKB1 : for (int j = 0; j < P_w[B_index]; j++) {
			        LOOP_BLOCKB2 : for (int i = 0; i < M_fea_current; i++) {
						    //#pragma HLS loop_tripcount min=84 max=84 avg=84
							#pragma HLS PIPELINE
							//#pragma HLS loop_tripcount min=16 max=16 avg=16
			        	    //INTYPES B_TEMP = B[i+j*M_fea+B_index*B_WIDTH_BLOCK*M_fea];
			        	    //BTYPE B_accel_temp = B_TEMP.range(7,4);

			        	    INTYPE BF = (INTYPE)B[i+j*M_fea_current+B_shift];
			        	    BTYPE B_accel_temp;
                          #if (INT_QUANT_W == 1)
			        	     quantw(B_accel_temp,BF,quantization_scale_w,f_align,beta_qu,B_index);
                          #else
			        	     B_accel_temp = BF;
                          #endif
			        		//BTYPE B_accel_temp = (BTYPE)B[i+j*M_fea+B_index*B_WIDTH_BLOCK*M_fea];


			        		B_accel[i][j] = B_accel_temp;
			        		//B_accel2[i][j] = B_accel_temp;
			        		//B_accel3[i][j] = B_accel_temp;
			        		//B_accel4[i][j] = B_accel_temp;

							//std::cout << "linear weight" << " " << i << " " << j << " " <<  BF  << " " << std::endl;
			        		//printf("%d %d %f %f \n",i,j,B_accel_temp,B[i+j*M_fea+B_index*B_WIDTH_BLOCK*M_fea]);
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


	 //hls::stream<FTYPE> A_fifo_feaq1;
	 //#pragma HLS STREAM variable=A_fifo_feaq1 depth=FIFO_DEPTH
	 //hls::stream<FTYPE> A_fifo_feaq2;
	 //#pragma HLS STREAM variable=A_fifo_feaq2 depth=FIFO_DEPTH
	 //hls::stream<FTYPE> A_fifo_feaq3;
	 //#pragma HLS STREAM variable=A_fifo_feaq3 depth=FIFO_DEPTH
	 //hls::stream<FTYPE> A_fifo_feaq4;
	 //#pragma HLS STREAM variable=A_fifo_feaq4 depth=FIFO_DEPTH

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

     //bool gemm_mode_int, stream_mode_int;
  	 //int M_fea_int;
     //int last_M_fea_int = 0;




    #if (PIPO_BLOCKS>=2)
	 LOOP_FEA : for (int B_index = 0; B_index < layer_loop; B_index++) {
    #else
	  int B_index = 0;
     #endif
    	#pragma HLS DATAFLOW

		B_WIDTH_INT = B_WIDTH_BLOCK;

	 	std::cout << "fea layer " << B_index << std::endl;


        //if (layer_loop == 1)
        //{
	    //	gemm_mode_int = gemm_mode;
	    //	stream_mode_int = stream_mode;
	    //    M_fea_int = M_fea;
        //}
        //else
        //{


	    //	a_values = N*M;
		//else //SPMM
		//	a_values = nnz_fea;


		/*these are the weights*/

		#if FEA_THREADS == 1

	    //read weights before locking buffer faster?



	  	     //hls::write_lock<buf> C_fea13(C_buffer13);
	  	    //hls::write_lock<buf> C_fea14(C_buffer14);
	  	    //hls::write_lock<buf> C_fea21(C_buffer21);
	  	    //hls::write_lock<buf> C_fea22(C_buffer22);
	  	    //hls::write_lock<buf> C_fea23(C_buffer23);
	  	    //hls::write_lock<buf> C_fea24(C_buffer24);
	  	    //hls::write_lock<buf> C_fea31(C_buffer31);
	  	    //hls::write_lock<buf> C_fea32(C_buffer32);
	  	    //hls::write_lock<buf> C_fea33(C_buffer33);
	  	    //hls::write_lock<buf> C_fea34(C_buffer34);
	  	    //hls::write_lock<buf> C_fea41(C_buffer41);
	  	    //hls::write_lock<buf> C_fea42(C_buffer42);
	  	    //hls::write_lock<buf> C_fea43(C_buffer43);
	  	    //hls::write_lock<buf> C_fea44(C_buffer44);


	  	    //std::cout << "Loop FEA " << std::endl;



          	 std::cout << "load weights " << std::endl;


             #if LINEAR_ONLY == 0

          	 readb(load_weights,model,beta_qu,f_align,quantization_scale_w,M_fea,P_w,B_index,B_accel1,B); //gnn weights

             #endif

             #if GNN_ONLY == 0

          	 readbl(load_weights,model,beta_qu,f_align,quantization_scale_w,M_fea,P_w,B_index,B_accel12,B2); //linear weights

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


	  		  // read sparse matrices

	  	          ////std::cout << "reada " << //std::endl;

	              //int max_fea1;
	              int first_row1,first_row2,first_row3,first_row4;
	              int row_count1,row_count2,row_count3,row_count4;

	              int N_fea_block = N_fea;
	  			  int N_fea_rest = 0;
	  		      row_count1 = N_fea_block;
	  		      //row_count2 = N_fea_block;
	  		      //row_count3 = N_fea_block;
	  		      //row_count2 = N_fea_block+N_fea_rest;
	  		      first_row1 = 0;
	  		      //first_row2 = N_fea_block;
	  		      //first_row3 = 2*N_fea_block;
	  		      //first_row4 = 3*N_fea_block;

	  			  //std::cout << "gemm_mode_int " << gemm_mode_int << std::endl;
                  int last_index1;
	  	          //reada1(exit_loop,first_row1,row_count1,A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,B_index_loop,tail,
	  		      //  rowPtr_fea1,columnIndex_fea1,values_fea1);

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

	  	          //reada1(first_row2,row_count2,A_fifo_fea2,col_indices_fifo_fea2,rnnz_fifo_fea2,B_index_loop,tail,
	  	            //rowPtr_fea2,columnIndex_fea2,values_fea2);
	  	          //reada1(first_row3,row_count3,A_fifo_fea3,col_indices_fifo_fea3,rnnz_fifo_fea3,B_index_loop,tail,
	  	          //  rowPtr_fea3,columnIndex_fea3,values_fea3);
	  	          //reada1(first_row4,row_count4,A_fifo_fea4,col_indices_fifo_fea4,rnnz_fifo_fea4,B_index_loop,tail,
	  	          //  rowPtr_fea4,columnIndex_fea4,values_fea4);



	  	          //quant1(A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,
	  	          //A_fifo_feaq1,col_indices_fifo_feaq1,rnnz_fifo_feaq1,last_index1,quantization_scale_fea);




	  		      //check_fifo_0(105165, A_fifo_fea1, A_fifo_fea1_out);

	  	          // inputs A_fifo_fea, col_indices_fifo_fea, rnnz_fifo_fea   and B_accel

	  		  //outputs C_buffer

	  	          //compute1 FEA * W = C

	  	          ////std::cout << "compute1" << //std::endl;


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

#if FEA_THREADS == 2



	   hls::write_lock<buf> C_fea11(C_buffer11);
		#if ADJ_THREADS == 2
		    hls::write_lock<buf> C_fea12(C_buffer12);
		    hls::write_lock<buf> C_fea22(C_buffer22);
		#endif
		    //hls::write_lock<buf> C_fea13(C_buffer13);
		    //hls::write_lock<buf> C_fea14(C_buffer14);
		    hls::write_lock<buf> C_fea21(C_buffer21);

		    //hls::write_lock<buf> C_fea23(C_buffer23);
		    //hls::write_lock<buf> C_fea24(C_buffer24);
		    //hls::write_lock<buf> C_fea31(C_buffer31);
		    //hls::write_lock<buf> C_fea32(C_buffer32);
		    //hls::write_lock<buf> C_fea33(C_buffer33);
		    //hls::write_lock<buf> C_fea34(C_buffer34);
		    //hls::write_lock<buf> C_fea41(C_buffer41);
		    //hls::write_lock<buf> C_fea42(C_buffer42);
		    //hls::write_lock<buf> C_fea43(C_buffer43);
		    //hls::write_lock<buf> C_fea44(C_buffer44);


		    //std::cout << "Loop FEA " << std::endl;

			for (int j = 0; j < B_WIDTH_INT; j++) {
				        LOOP_BLOCKB : for (int i = 0; i < M_fea; i++) {
							    //#pragma HLS loop_tripcount min=84 max=84 avg=84
								#pragma HLS PIPELINE
								//#pragma HLS loop_tripcount min=16 max=16 avg=16
				        		BTYPE B_accel_temp = B[i+j*M_fea+B_index*B_WIDTH_BLOCK*M_fea];
				        		B_accel1[i][j] = B_accel_temp;
				        		B_accel2[i][j] = B_accel_temp;
				        		//B_accel3[i][j] = B_accel_temp;
				        		//B_accel4[i][j] = B_accel_temp;

								////std::cout << " " << i << " " << j << " " << B_accel[i][j]  << std::endl;
							}
			}

			  int first_row1,first_row2,first_row3,first_row4;
			  int row_count1,row_count2,row_count3,row_count4;

      	  	  int N_fea_block = N_fea/2;
			  int N_fea_rest = N_fea%2;
		      row_count1 = N_fea_block;
		      row_count2 = N_fea_block+N_fea_rest;
		      first_row1 = 0;
		      first_row2 = N_fea_block;

              std::cout << "Thread fea 1" << std::endl;
	          reada1(gemm_mode,M_fea,first_row1,row_count1,A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,B_index_loop,tail,
		        rowPtr_fea1,columnIndex_fea1,values_fea1);
	          std::cout << "Thread fea 2" << std::endl;
	          reada1(gemm_mode,M_fea,first_row2,row_count2,A_fifo_fea2,col_indices_fifo_fea2,rnnz_fifo_fea2,B_index_loop,tail,
	            rowPtr_fea2,columnIndex_fea2,values_fea2);

	#if ADJ_THREADS == 2
		  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_fea11, C_fea12,
				  //C_fea13, C_fea14,
				  B_index, B_index_loop, tail);
		  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,B_accel2,C_fea21, C_fea22,
				  //C_fea23, C_fea24,
				  B_index, B_index_loop, tail);
	#endif

	#if ADJ_THREADS == 1

		  compute1_1(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_fea11,
	  				  //C_fea12,
	  				  //C_fea13, C_fea14,
	  				  );

		  compute1_1(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,B_accel2,C_fea21,
	  				  //C_fea12,
	  				  //C_fea13, C_fea14,
	  				  );
	#endif


	#endif


       #if FEA_THREADS == 4

		#if ADJ_THREADS == 4


	    hls::write_lock<buf> C_fea11(C_buffer11);
	    hls::write_lock<buf> C_fea12(C_buffer12);
	    hls::write_lock<buf> C_fea13(C_buffer13);
	    hls::write_lock<buf> C_fea14(C_buffer14);
	    hls::write_lock<buf> C_fea21(C_buffer21);
	    hls::write_lock<buf> C_fea22(C_buffer22);
	    hls::write_lock<buf> C_fea23(C_buffer23);
	    hls::write_lock<buf> C_fea24(C_buffer24);
	    hls::write_lock<buf> C_fea31(C_buffer31);
	    hls::write_lock<buf> C_fea32(C_buffer32);
	    hls::write_lock<buf> C_fea33(C_buffer33);
	    hls::write_lock<buf> C_fea34(C_buffer34);
	    hls::write_lock<buf> C_fea41(C_buffer41);
	    hls::write_lock<buf> C_fea42(C_buffer42);
	    hls::write_lock<buf> C_fea43(C_buffer43);
	    hls::write_lock<buf> C_fea44(C_buffer44);

       #if GAT_ENABLE == 1
	    hls::write_lock<buf> A_fea11(A_buffer11);
	    hls::write_lock<buf> A_fea21(A_buffer21);
	    hls::write_lock<buf> A_fea31(A_buffer31);
	    hls::write_lock<buf> A_fea41(A_buffer41);
        #else
	    QTYPE A_fea11[B_HEIGHT][B_WIDTH_BLOCK];
	    QTYPE A_fea21[B_HEIGHT][B_WIDTH_BLOCK];
	    QTYPE A_fea31[B_HEIGHT][B_WIDTH_BLOCK];
	    QTYPE A_fea41[B_HEIGHT][B_WIDTH_BLOCK];
        #endif

	    #endif

		#if ADJ_THREADS == 2

	    	hls::write_lock<buf> C_fea11(C_buffer11);
	 	    hls::write_lock<buf> C_fea12(C_buffer12);
	 	    //hls::write_lock<buf> C_fea13(C_buffer13);
	 	    //hls::write_lock<buf> C_fea14(C_buffer14);
	 	    hls::write_lock<buf> C_fea21(C_buffer21);
	 	    hls::write_lock<buf> C_fea22(C_buffer22);
	 	    //hls::write_lock<buf> C_fea23(C_buffer23);
	 	    //hls::write_lock<buf> C_fea24(C_buffer24);
	 	    hls::write_lock<buf> C_fea31(C_buffer31);
	 	    hls::write_lock<buf> C_fea32(C_buffer32);
	 	    //hls::write_lock<buf> C_fea33(C_buffer33);
	 	    //hls::write_lock<buf> C_fea34(C_buffer34);
	 	    hls::write_lock<buf> C_fea41(C_buffer41);
	 	    hls::write_lock<buf> C_fea42(C_buffer42);
	 	    //hls::write_lock<buf> C_fea43(C_buffer43);
	 	    //hls::write_lock<buf> C_fea44(C_buffer44);

		#endif



	    //std::cout << "Loop FEA " << std::endl;

		for (int j = 0; j < B_WIDTH_INT; j++) {
			        LOOP_BLOCKB : for (int i = 0; i < M_fea; i++) {
						    //#pragma HLS loop_tripcount min=84 max=84 avg=84
							#pragma HLS PIPELINE
							//#pragma HLS loop_tripcount min=16 max=16 avg=16
  			        	    INTYPE BF = (INTYPE)B[i+j*M_fea+B_index*B_WIDTH_BLOCK*M_fea];
  			        	    BTYPE B_accel_temp;
                            #if (INT_QUANT_W == 1)
  			        	     quantw(B_accel_temp,BF,quantization_scale_w,f_align,beta_qu);
                            #else
  			        	     B_accel_temp = BF;
                            #endif

			        		B_accel1[i][j] = B_accel_temp;
			        		B_accel2[i][j] = B_accel_temp;
			        		B_accel3[i][j] = B_accel_temp;
			        		B_accel4[i][j] = B_accel_temp;

							////std::cout << " " << i << " " << j << " " << B_accel[i][j]  << std::endl;
						}
		}



		  // read sparse matrices

	          ////std::cout << "reada " << //std::endl;


              int first_row1,first_row2,first_row3,first_row4;
              int row_count1,row_count2,row_count3,row_count4;


              int N_fea_block = N_fea/4;
			  int N_fea_rest = N_fea%4;
		      row_count1 = N_fea_block;
		      row_count2 = N_fea_block;
		      row_count3 = N_fea_block;
		      row_count4 = N_fea_block+N_fea_rest;
		      first_row1 = 0;
		      first_row2 = N_fea_block;
		      first_row3 = 2*N_fea_block;
		      first_row4 = 3*N_fea_block;
		      ITYPE max_fea1,max_fea2,max_fea3,max_fea4;


			    //std::cout << "READA1 " << std::endl;

              int last_index1,last_index2,last_index3,last_index4;
              #if (COO_MODE == 0)
	          reada1_csr(beta_qu,f_align,quantization_scale_fea,last_index1,stream_mode,gemm_mode,M_fea,first_row1,row_count1,A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,
		        rowPtr_fea1,columnIndex_fea1,values_fea1,values_feas1);
	          reada1_csr(beta_qu,f_align,quantization_scale_fea,last_index2,stream_mode,gemm_mode,M_fea,first_row2,row_count2,A_fifo_fea2,col_indices_fifo_fea2,rnnz_fifo_fea2,
	            rowPtr_fea2,columnIndex_fea2,values_fea2,values_feas2);
	          reada1_csr(beta_qu,f_align,quantization_scale_fea,last_index3,stream_mode,gemm_mode,M_fea,first_row3,row_count3,A_fifo_fea3,col_indices_fifo_fea3,rnnz_fifo_fea3,
	            rowPtr_fea3,columnIndex_fea3,values_fea3,values_feas3);
	          reada1_csr(beta_qu,f_align,quantization_scale_fea,last_index4,stream_mode,gemm_mode,M_fea,first_row4,row_count4,A_fifo_fea4,col_indices_fifo_fea4,rnnz_fifo_fea4,
	            rowPtr_fea4,columnIndex_fea4,values_fea4,values_feas4);
              #else
	           reada1_coo(nnz_fea1,beta_qu,f_align,quantization_scale_fea,last_index1,stream_mode,gemm_mode,M_fea,first_row1,row_count1,A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,
		        rowPtr_fea1,columnIndex_fea1,values_fea1,values_feas1);
	           reada1_coo(nnz_fea2,beta_qu,f_align,quantization_scale_fea,last_index2,stream_mode,gemm_mode,M_fea,first_row2,row_count2,A_fifo_fea2,col_indices_fifo_fea2,rnnz_fifo_fea2,
	            rowPtr_fea2,columnIndex_fea2,values_fea2,values_feas2);
	           reada1_coo(nnz_fea3,beta_qu,f_align,quantization_scale_fea,last_index3,stream_mode,gemm_mode,M_fea,first_row3,row_count3,A_fifo_fea3,col_indices_fifo_fea3,rnnz_fifo_fea3,
	            rowPtr_fea3,columnIndex_fea3,values_fea3,values_feas3);
	           reada1_coo(nnz_fea4,beta_qu,f_align,quantization_scale_fea,last_index4,stream_mode,gemm_mode,M_fea,first_row4,row_count4,A_fifo_fea4,col_indices_fifo_fea4,rnnz_fifo_fea4,
	            rowPtr_fea4,columnIndex_fea4,values_fea4,values_feas4);
               #endif
		  //check_fifo_0(a_values, A_fifo, A_fifo_out);

	          // inputs A_fifo_fea, col_indices_fifo_fea, rnnz_fifo_fea   and B_accel

		  //outputs C_buffer

	          //compute1 FEA * W = C

	          ////std::cout << "compute1" << //std::endl;


	  	    //std::cout << "COMPUTE1 " << std::endl;



			#if ADJ_THREADS == 4

	          compute1_4(scale_fea,&max_fea1,quantized_multiplier,gemm_mode,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_fea11, C_fea12, C_fea13, C_fea14,A_fea11);
	          compute1_4(scale_fea,&max_fea2,quantized_multiplier,gemm_mode,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,B_accel2,C_fea21, C_fea22, C_fea23, C_fea24,A_fea21);
	          compute1_4(scale_fea,&max_fea3,quantized_multiplier,gemm_mode,zero_point_lhs,  zero_point_rhs, first_row3,row_count3,A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3,B_accel3,C_fea31, C_fea32, C_fea33, C_fea34,A_fea31);
	          compute1_4(scale_fea,&max_fea4,quantized_multiplier,gemm_mode,zero_point_lhs,  zero_point_rhs, first_row4,row_count4,A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4,B_accel4,C_fea41, C_fea42, C_fea43, C_fea44,A_fea41);

		      *max_fea = max_fea1;

	        #endif

			#if ADJ_THREADS == 2

			  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_fea11, C_fea12, //C_fea13, C_fea14,
					  B_index, B_index_loop, tail);
			  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,B_accel2,C_fea21, C_fea22, //C_fea23, C_fea24,
					  B_index, B_index_loop, tail);
			  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row3,row_count3,A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3,B_accel3,C_fea31, C_fea32, //C_fea33, C_fea34,
					  B_index, B_index_loop, tail);
			  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row4,row_count4,A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4,B_accel4,C_fea41, C_fea42, //C_fea43, C_fea44,
					  B_index, B_index_loop, tail);

			#endif
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




   	   //int B_WIDTH_INT;



    #if (PIPO_BLOCKS>=2)

	   LOOP_ADJ : for (int B_index = 0; B_index < layer_loop; B_index++) {



    #else

		   int B_index = 0;

    #endif

        #pragma HLS DATAFLOW
	    //if (B_index < (B_index_loop-1))
			//B_WIDTH_INT = B_WIDTH_BLOCK;
	      //else
			//B_WIDTH_INT = tail;

            //if (layer_loop == 1)
	        //{
		    //	stream_mode_int = stream_mode;
	        //}
	        //else
	        //{

		 	  std::cout << "adj layer " << B_index << std::endl;


	        //}


	#if ADJ_THREADS == 1

		   //while (C_buffer11.empty()); // execute only when producer has already generated some meaningful data
         #if (PIPO_BLOCKS>=2)
		     hls::read_lock<buf> C_adj11(C_buffer11);
		     hls::read_lock<buf> linear_adj(linear_pipo);
         #endif
		     //hls::read_lock<buf> C_adj12(C_buffer12);
		 		    //hls::read_lock<buf> C_adj13(C_buffer13);
		 		    //hls::read_lock<buf> C_adj14(C_buffer14);
			#if FEA_THREADS == 2
		    	hls::read_lock<buf> C_adj21(C_buffer21);
			#endif
		 		    //hls::read_lock<buf> C_adj22(C_buffer22);
		 		    /*hls::read_lock<buf> C_adj23(C_buffer23);
		 		    hls::read_lock<buf> C_adj24(C_buffer24);
		 		    hls::read_lock<buf> C_adj31(C_buffer31);
		 		    hls::read_lock<buf> C_adj32(C_buffer32);
		 		    hls::read_lock<buf> C_adj33(C_buffer33);
		 		    hls::read_lock<buf> C_adj34(C_buffer34);
		 		    hls::read_lock<buf> C_adj41(C_buffer41);
		 		    hls::read_lock<buf> C_adj42(C_buffer42);
		 		    hls::read_lock<buf> C_adj43(C_buffer43);
		 		    hls::read_lock<buf> C_adj44(C_buffer44);*/



		 	    int first_row1;//,first_row2;//,first_row3,first_row4;
		 	    int row_count1;//,row_count2;//,row_count3,row_count4;

		         int N_adj_block = N_adj/ADJ_THREADS;
		         int N_adj_block_compute = N_adj/FEA_THREADS; // in compute2 each block only contains  N_adj/FEA_THREADS elements
		 	    //int N_adj_rest = N_adj%2;
		 		row_count1 = N_adj_block;
		 		//row_count2 = N_adj_block;
		 		//row_count3 = N_adj_block;
		 		//row_count4 = N_adj_block+N_adj_rest;
		 		first_row1 = 0;
		 		//first_row2 = N_adj_block;
		 		//first_row3 = 2*N_adj_block;
		 		//first_row4 = 3*N_adj_block;


		 	    //std::cout << "READA2 " << std::endl;

		 	    //reada2(first_row1,row_count1,B_index_loop,tail,A_fifo_adj1,coindices_fifo_adj1,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1);

		 	    //std::cout << "COMPUTE2 " << std::endl;
				#if FEA_THREADS == 1
                   #if(PIPO_BLOCKS>=2)
                    #if LINEAR_ONLY == 0
		 	    	 compute2_1(model,srelu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_adj11,
		 	    		//C_adj21,
		 	    		//C_adj31,C_adj41,
		 	    		D_fifo1,B_index);
                    #endif
                   #else
                     #if LINEAR_ONLY == 0
		 	    	 compute2_1(relu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_buffer11,
			 	    		//C_adj21,
			 	    		//C_adj31,C_adj41,
			 	    		D_fifo1);
                       #endif
                   #endif
				#endif
				#if FEA_THREADS == 2
		 	    	compute2_2(N_adj_block_compute,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_adj11,C_adj21,
			   	    		//C_adj31,C_adj41,
			   	    		D_fifo1);

				#endif
		         //compute2(N_adj_block,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,C_adj12,C_adj22,
		         		//C_adj32,C_adj42,
		         	//	D_fifo2, B_index, B_index_loop, tail);
		         //compute2(N_adj_block,zero_point_lhs,  zero_point_rhs, first_row3,row_count3,A_fifo_adj3, col_indices_fifo_adj3, rnnz_fifo_adj3,C_adj13,C_adj23,C_adj33,C_adj43,D_fifo3, B_index, B_index_loop, tail);
		         //compute2(N_adj_block,zero_point_lhs,  zero_point_rhs, first_row4,row_count4,A_fifo_adj4, col_indices_fifo_adj4, rnnz_fifo_adj4,C_adj14,C_adj24,C_adj34,C_adj44,D_fifo4, B_index, B_index_loop, tail);

		 	          //compute2(zero_point_lhs,  zero_point_rhs, N_adj,M_fea,A_fifo_adj, col_indices_fifo_adj, rnnz_fifo_adj,B,D_fifo, B_index_loop, tail);

		 	          //////std::cout << "scale" << //std::endl;

		 		  //scale(quantized_multiplier, shift, bias, zero_point_dst, clamp_max,clamp_min,N_adj, M_adj, P_w, D_fifo, B_index, B_index_loop, tail,write_fifo);

		 	          //check_fifo_2(N/4, write_fifo_0, write_fifo_out_0);

		 		 // write write _fifo into D

		 	          ////std::cout << "write matrix size " << N_adj << "," << P_w << //std::endl;

		 	    //std::cout << "WRITEC " << std::endl;

		 	    	  //relupipe(first_row1,row_count1,N_adj,P_w, D_fifo1, D1,B_index);


                      #if(PIPO_BLOCKS>=2)
		 	           writec(deq_factor,model,first_row1,row_count1,N_adj,P_w, D_fifo1,linear_adj,out_fifo1,B_index,layer_loop);
                      #else
		 	           writec(deq_factor,model,first_row1,row_count1,N_adj,P_w, D_fifo1,linear_pipo,D1,DS1,B_index,layer_loop);
                      #endif

		 	          writeout(model,first_row1,row_count1,N_adj,P_w, out_fifo1,D1,DS1,DS1R, DS1C,B_index,layer_loop);
		 	           //writec_transpose(deq_factor,stream_mode,first_row1,row_count1,N_adj,P_w, D_fifo1, D1,DS1,B_index);

	#endif

#if ADJ_THREADS == 2

	    hls::read_lock<buf> C_adj11(C_buffer11);
	    hls::read_lock<buf> C_adj12(C_buffer12);
	   	hls::read_lock<buf> C_adj21(C_buffer21);
	   	hls::read_lock<buf> C_adj22(C_buffer22);

		#if FEA_THREADS == 4
	    	hls::read_lock<buf> C_adj31(C_buffer31);
		    hls::read_lock<buf> C_adj32(C_buffer32);
		    hls::read_lock<buf> C_adj41(C_buffer41);
		    hls::read_lock<buf> C_adj42(C_buffer42);
		#endif


	   	int first_row1,first_row2;//,first_row3,first_row4;
	   	int row_count1,row_count2;//,row_count3,row_count4;

	    int N_adj_block = N_adj/ADJ_THREADS;
	   	int N_adj_rest = N_adj%ADJ_THREADS;
	    int N_adj_block_compute = N_adj/FEA_THREADS; // in compute2 each block only contains  N_adj/FEA_THREADS elements
	   	row_count1 = N_adj_block;
	   	row_count2 = N_adj_block+N_adj_rest;;
	   	first_row1 = 0;
	   	first_row2 = N_adj_block;


        std::cout << "Thread adj 1" << std::endl;
	   	reada2(first_row1,row_count1,B_index_loop,tail,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1);
        std::cout << "Thread adj 2" << std::endl;
	   	reada2(first_row2,row_count2,B_index_loop,tail,A_fifo_adj2,col_indices_fifo_adj2,rnnz_fifo_adj2,rowPtr_adj2,columnIndex_adj2,values_adj2);

		#if FEA_THREADS == 2

	   	compute2_2(N_adj_block_compute,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_adj11,C_adj21,
	   	    		//C_adj31,C_adj41,
	   	    		D_fifo1, B_index, B_index_loop, tail);

	    compute2_2(N_adj_block_compute,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,C_adj12,C_adj22,
	           		//C_adj32,C_adj42,
	           		D_fifo2, B_index, B_index_loop, tail);

		#endif

		#if FEA_THREADS == 4


	   	compute2_4(N_adj_block_compute,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_adj11,C_adj21,
	   	    		C_adj31,C_adj41,
	   	    		D_fifo1, B_index, B_index_loop, tail);

	    compute2_4(N_adj_block_compute,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,C_adj12,C_adj22,
	           		C_adj32,C_adj42,
	           		D_fifo2, B_index, B_index_loop, tail);

		#endif

	   	 writec(first_row1,row_count1,P_w, D_fifo1, D1,B_index,B_index_loop, tail);
	   	 writec(first_row2,row_count2,P_w, D_fifo2, D2,B_index,B_index_loop, tail);

		#endif

	#if ADJ_THREADS == 4

		    hls::read_lock<buf> C_adj11(C_buffer11);
		    hls::read_lock<buf> C_adj12(C_buffer12);
		    hls::read_lock<buf> C_adj13(C_buffer13);
		    hls::read_lock<buf> C_adj14(C_buffer14);
		    hls::read_lock<buf> C_adj21(C_buffer21);
		    hls::read_lock<buf> C_adj22(C_buffer22);
		    hls::read_lock<buf> C_adj23(C_buffer23);
		    hls::read_lock<buf> C_adj24(C_buffer24);
		    hls::read_lock<buf> C_adj31(C_buffer31);
		    hls::read_lock<buf> C_adj32(C_buffer32);
		    hls::read_lock<buf> C_adj33(C_buffer33);
		    hls::read_lock<buf> C_adj34(C_buffer34);
		    hls::read_lock<buf> C_adj41(C_buffer41);
		    hls::read_lock<buf> C_adj42(C_buffer42);
		    hls::read_lock<buf> C_adj43(C_buffer43);
		    hls::read_lock<buf> C_adj44(C_buffer44);

	    int first_row1,first_row2,first_row3,first_row4;
	    int row_count1,row_count2,row_count3,row_count4;



        int N_adj_block = N_adj/4;
	    int N_adj_rest = N_adj%4;
		row_count1 = N_adj_block;
		row_count2 = N_adj_block;
		row_count3 = N_adj_block;
		row_count4 = N_adj_block+N_adj_rest;
		first_row1 = 0;
		first_row2 = N_adj_block;
		first_row3 = 2*N_adj_block;
		first_row4 = 3*N_adj_block;


	    //std::cout << "READA2 " << std::endl;




	    //std::cout << "COMPUTE2 " << std::endl;

	    compute2_4(relu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_adj11,C_adj21,C_adj31,C_adj41,D_fifo1);
        compute2_4(relu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,C_adj12,C_adj22,C_adj32,C_adj42,D_fifo2);
        compute2_4(relu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row3,row_count3,A_fifo_adj3, col_indices_fifo_adj3, rnnz_fifo_adj3,C_adj13,C_adj23,C_adj33,C_adj43,D_fifo3);
        compute2_4(relu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row4,row_count4,A_fifo_adj4, col_indices_fifo_adj4, rnnz_fifo_adj4,C_adj14,C_adj24,C_adj34,C_adj44,D_fifo4);

	    writec(deq_factor,stream_mode,first_row1,row_count1,N_adj,P_w, D_fifo1, D1,DS1,B_index);
	    writec(deq_factor,stream_mode,first_row2,row_count2,N_adj,P_w, D_fifo2, D2,DS2,B_index);
	    writec(deq_factor,stream_mode,first_row3,row_count3,N_adj,P_w, D_fifo3, D3,DS3,B_index);
	    writec(deq_factor,stream_mode,first_row4,row_count4,N_adj,P_w, D_fifo4, D4,DS4,B_index);

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
INTYPE *values_adj1, 	INTYPE *values_adj2,	INTYPE *values_adj3,	INTYPE *values_adj4,
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

//std::cout << "Start loop adj with gemm mode "  << gemm_mode << std::endl;

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





      //hls::stream<QTYPE>  linear_fifo;
      //#pragma HLS STREAM variable= linear_fifo depth=LINEAR_DEPTH
      //#pragma HLS bind_storage variable = linear_fifo type=FIFO impl=URAM

      #if (PIPO_BLOCKS>=2)
	    hls::stream_of_blocks<buf,PIPO_BLOCKS> linear_pipo;
      #else
	    buf linear_pipo;
      #endif
	  #pragma HLS array_partition variable=linear_pipo block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=linear_pipo cyclic factor= SBLOCK_LIN dim=1




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
      //#pragma HLS array_partition variable=A_buffer11 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> A_buffer21;
	  #pragma HLS array_partition variable=A_buffer21 block factor= BLOCK/2 dim=2
      //#pragma HLS array_partition variable=A_buffer21 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> A_buffer31;
      #pragma HLS array_partition variable=A_buffer31 block factor= BLOCK/2 dim=2
      //#pragma HLS array_partition variable=A_buffer31 cyclic factor= SBLOCK dim=1


      hls::stream_of_blocks<buf,PIPO_BLOCKS> A_buffer41;
      #pragma HLS array_partition variable=A_buffer41 block factor= BLOCK/2 dim=2
      //#pragma HLS array_partition variable=A_buffer41 cyclic factor= SBLOCK dim=1




      //hls::stream_of_blocks<buf> D_buffer11;
	  //#pragma HLS array_partition variable=D_buffer11 block factor= BLOCK/2 dim=2
      //#pragma HLS array_partition variable=D_buffer11 cyclic factor= SBLOCK dim=1



      //hls::stream<ITYPE> D_fifo_0;
      //#pragma HLS STREAM variable=D_fifo_0 depth=128 dim=1

      //hls::stream<ITYPE> write_fifo_0;
      //#pragma HLS STREAM variable=write_fifo_0 depth=128 dim=1

      //hls::stream<ITYPE> write_fifo_out_0;
      //#pragma HLS STREAM variable=write_fifo_out_0 depth=8 dim=1



      int B_WIDTH_INT,a_values;

      #if (PIPO_BLOCKS>=2)
      //LOOP_FEA : for (int B_index = 0; B_index < layer_loop; B_index++) {
        #pragma HLS DATAFLOW
      #endif

      //std::cout << "Start loop fea with gemm mode " << gemm_mode[0] << std::endl;



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
      //}




}


typedef unsigned long u32;

/*
 The amount of data saved in the FPGA is B_HEIGHT*B_WIDTH_BLOCK+A_WIDTH+B_WIDTH_BLOCK which should be less than FPGA BRAM size
*/

//gemm_mode fea adj
// 0 0 0 dense dense not in used in graph layers
// 1 0 1 dense sparse normal mode for layer 2
// 2 1 0 sparse dense used in training
// 3 1 1 sparse sparse normal mode for layer 1
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

     //#pragma HLS INTERFACE ap_none port = stream_mode

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
     //#pragma HLS INTERFACE m_axi port=D1 depth=32000 offset=slave latency=0 num_write_outstanding=2048 bundle=D1
	 //#pragma HLS INTERFACE m_axi port=D2 depth=32000 offset=slave latency=0 num_write_outstanding=2048 bundle=D2
     //#pragma HLS INTERFACE m_axi port=D3 depth=32000 offset=slave latency=0 num_write_outstanding=2048 bundle=D3
     //#pragma HLS INTERFACE m_axi port=D4 depth=32000 offset=slave latency=0 num_write_outstanding=2048 bundle=D4
     //#pragma HLS INTERFACE m_axi port=D1 depth=64000 offset=slave max_widen_bitwidth=512 max_write_burst_length=16 bundle=D1
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


	 //c_fifo_stream_t       C_fifo[B_WIDTH_BLOCK];
	 //#pragma HLS STREAM variable=C_fifo depth=1024 dim=1


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

	 //hls::stream<ASTYPE> DS2;
	 //hls::stream<ASTYPE> DS3;
	 //hls::stream<ASTYPE>  DS4;

	 //hls::stream<ASTYPE>  OUTS1;
	 //#pragma HLS STREAM variable=OUTS1 depth=FIFO_DEPTH

	 //hls::stream<ASTYPE>&  values_feas1,hls::stream<ASTYPE>& values_feas2,hls::stream<ASTYPE>&  values_feas3,hls::stream<ASTYPE>& values_feas4,

	 //ap_int<32> quantized_multiplier_data[1024];


	 //load bias
         //preloading bias and param data seems to be a good idea but in practice performance is the same and we save preloading overhead
         //param data is loaded in demand in this case
         //preloading is important for certain matrix configurations with small A and large B so I am going to leave it
        // if (bias_count > 0)
        // {

	 	//for(int bias_index=0;bias_index<bias_count;bias_index++)
	 	//{
	//		#pragma HLS PIPELINE
		//	 bias_data[bias_index]=bias[bias_index];
		//	 shift_data[bias_index]=shift[bias_index];
		//	 quantized_multiplier_data[bias_index]=quantized_multiplier[bias_index];
	 	//}


	 //}
     //else
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

     //load model
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

    	 //quantization_scale_w_int[i] = 127.0;
    	 //quantization_scale_fea_int[i] =   255.0;
    	 //deq_factor_int[i] =  4.0631776391272885;

    	 std::cout << " Instruction is "<< model_int[i][7] <<  model_int[i][6] <<  model_int[i][5] <<  model_int[i][4] <<
    	 model_int[i][3] <<  model_int[i][1] <<  model_int[i][1] <<  model_int[i][0] << std::endl;
     }



	 //else
	 //{
	  /*simulation run short, remove in normal synthesis*/
	  //B_index_loop = 2;
	  //tail = 0;
	 //}

       //std::cout << " B_index_loop is "<< B_index_loop<< " tail is "<< tail << std::endl;

         //for (int B_index = 0; B_index < B_index_loop; B_index++) {
	  //*max_fea = 0;
	  ITYPE max_fea_val = 0;

	  //bool fixed_gat = 1;

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

	  //std::cout << "max fea val " << max_fea_val << std::endl;

     *max_fea = max_fea_val;

     //std::cout << "max fea " << *max_fea << std::endl;

	  std::cout << "Done mmult wrapper" << std::endl;


        //}

	/*profiling[0] = fifo_full_0;
	profiling[1] = fifo_full_1;
	profiling[2] = fifo_full_2;
	profiling[3] = fifo_empty_0;
	profiling[4] = fifo_empty_1;
	profiling[5] = fifo_empty_2;
	profiling[6] = fifo_read_0;
	profiling[7] = fifo_read_1;
	profiling[8] = fifo_read_2;
	profiling[9] = fifo_write_0;
	profiling[5] = fifo_write_1;
	profiling[11] = fifo_write_2;
	profiling[12] = fifo_cycle_0;
	profiling[13] = fifo_cycle_1;
	profiling[14] = fifo_cycle_2;
  */


     }
}

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
)
{

    int array_c_adjust=N_adj;
    ap_uint<8> P_w_int[5];

    P_w_int[0]=P_w;
    float srelu[5];
	srelu[0]=0.0;

    //int N_adj_block = N_adj/ADJ_THREADS;
    //array_d2+=N_adj_block*P_w;
    //array_d3+=2*N_adj_block*P_w;
    //array_d4+=3*N_adj_block*P_w;

    //#pragma SDS resource(1)
    //#pragma SDS async(1)


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

    //#pragma SDS wait(1)

    std::cout << " kernel done " << std::endl;
}
