/*===============================================================================
* This file is part of the SGRACE GNN accelerator
* has been written at Linkoping/UPM University
* Author : Jose Nunez-Yanez
*Copyright (C) 2026 Jose Nunez-Yanez
*This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
*This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
===============================================================================
*/

#ifndef __MATRIX_MULT_H__
#define __MATRIX_MULT_H__

#include <vector>

//WARNING: this accerator uses a writec loop that ignores possible tails. This means
//that the width of the WEIGHT matrix has to be a multiple of the number of cores.
//For example 64 for P_w and then 16 cores so a total of 4 EXACT tiles are processed.
//use_sblocks is configurable to use SPMM_BLOCK or not. This optimizes the writec loop
//without SPMM_BLOCK so that the writing to memory is more efficient. This is important
//if the amount of work is so little that the bottleneck is the write loop

// 64 * 16(values in 32-bit word (ternary)) = 1024
//1536 and 16 in a word so 96 words of 32-bit
//1536 and 4 in a word so 384 words of 8-bit

//if you want to generate CSR then set this DTYPE_LENGTH to 8 since spmm uses packing to 8-bit for A matrix
// if you set it to 32 then A matrix will be formed with 32-bits words instead of 8-bit
//8-bit packs
// ./generate_csr_8.elf /mnt/weights_tern_090_block_32_1.csv 1 6 336  512
//32-bit packs
// ./generate_csr_32.elf /mnt/weights_tern_090_block_32_1.csv 1 6 84  512
//quad
//8-bit packs
// ./generate_csr_8.elf /mnt/weights_quad_block_095_32_1.csv 2 6 672  512
//32-bit packs
// ./generate_csr_32.elf /mnt/weights_quad_block_095_32_1.csv 2 6 168  512

//#define GENERATE_CSR 0
//#define DTYPE_LENGTH 32 //8//32 //8 //32//32 //8//32//8

//#define simulation

#define MAX_N    4096 //20480  //4096 //38000 //4096 //20480 //32768 //8192 //20480  //4096 //8192 //4096 //1024 //98304 //4096 //20480 // 4096 //256 // 6144 //20480//16384 //4096 //32768 //20480 //64
#define MAX_M    2048 //4096 //4096 //20480  //4096 //38000 ///4096 //20480 //32768 //8192 //20480  //4096 //8192 //4096 //1024 //98304 //4096 //20480 // 4096 // 256 // 6144 //20480//16384 //4096 //24576 //2048 //384 //1536 //384 //1536 // 384 //48 //768 //96 //384 //96 //384 //96 //384// 96
#define MAX_P    64 //2048 //512 //64//1//64//1

#define MAX_FIFO 16
#define PIPO_BLOCKS 2 // use a PIPO with only one block to save memory instead of standard 2

//#define SN    64 //2048 //64
//#define SM    384 //1536 //384 //1536 //384 //1536 // 384 //48 //768 //96 //384 //96 //384 //96 //384// 96
//#define SP    512//64//1//64//1


//48 for WL 64
//96 for WL 32
// 384 for WL 8
// 768 for WL 4

#define A_HEIGHT   MAX_N
//#define A_WIDTH    MAX_N

#define F_HEIGHT   MAX_N
//#define F_WIDTH    MAX_M

#define B_HEIGHT   MAX_M
#define B_WIDTH    MAX_P

//#define C_HEIGHT   MAX_N
#define C_WIDTH    MAX_P


//#define TWOBIT_TERN
//#define TWOBIT_SIX
//#define TWOBIT_OPT
//#define TWOBIT_POS
//#define TWOBIT_NAIVE
//#define TWOBIT
//#define FOURBIT_NAIVE
//#define FOURBIT
#define EIGHTBIT
//#define FOURBIT_XIL
//#define ONEBIT
//#define HALF
//#define FLOAT
//#define TEST

//#define TRAINING_MODE 1 //hardware generates data for backpropagation (inference + training)
//(TRAINING_MODE SWITCH not in used, training mode write output to memory, inference mode write output to stream)
// training mode one layer at a time, inference mode pack all layers together.

#define SIGNED_MODE 0 //use when training configuration so all inputs are signed
#define COO_MODE 1
//#define BINARY_MODE 1 no in used //use for 1-bit precission mode
#define INT_QUANT 1 //internal quantization so interface to DDR set to float
#define INT_QUANT_A 1 //internal hardware quantization of adjacency (can be done in software if adjacency is fixed).
#define INT_QUANT_F 1 //internal hardware quantization of features (important since input/features is not constant).
#define INT_QUANT_W 1 //internal hardware quantization of weights (can be done in software if weights are fixed/inference only).
#define INT_DEQUANT 1 //remove the scaling of the quantization and the internal scaling in the output (needed and done in hardware)
#define FEA_THREADS 1
#define ADJ_THREADS 1
#define GAT_ENABLE 0 //implement support for GAT
#define GNN_ONLY 0 //implement support only for GNN (not linear)
#define LINEAR_ONLY 0 //implement support only for LINEAR
#define GATV2 0 //usegatv2 modificatino or standard gat
#define SPMM_BLOCK 1 //Fused row execution optimization to improve sparse performance. 1 no fusion or 4 for 4-row fusion.
#define ATEN_BLOCK 1 //leave aten block at 1 since dataflow issues otherwise
#define OPT_ATTN 1 ////OPT_ATTN control how much sparsity expected in adj (e.g. OPT_ATTN = 1 worse case with fully dense adjacency,OPT_ATTN = 8 12.5% non zeros in adj) (this is per row, some rows have quite a few nonzeros)
#define B_WIDTH_BLOCK MAX_P //96 //attention cannot do blocks so this value has to match the W width

//with OPT_ATTN 8 and 4096 adj you have buffer for 512 nonzeros
#ifdef EIGHTBIT

	////typedef ap_fixed<16, 2> ATYPE;
	//typedef ap_int<8> ATYPE;
	////typedef ap_fixed<16, 2> BTYPE;
	//typedef ap_int<8> BTYPE;
	//typedef ap_int<8> DTYPE;
	////typedef ap_fixed<16, 2> DTYPE;
	////typedef ap_fixed<16, 2> FTYPE;
	//typedef ap_int<16> ITYPE;
	////typedef ap_fixed<16, 2> ITYPE;


	//typedef ap_ufixed<8, 0> ATYPE;
	//typedef ap_ufixed<8, 0> BTYPE;
	//typedef ap_ufixed<8, 0> DTYPE;
	//typedef ap_ufixed<8, 0> FTYPE;
	//typedef ap_ufixed<16, 8> ITYPE;

    //typedef ap_fixed<32, 7> ATYPE;
    //typedef ap_fixed<32, 7, AP_RND, AP_SAT> ATYPE;
    //typedef ap_fixed<32, 7, AP_RND, AP_SAT> BTYPE;
	//typedef ap_fixed<32, 7, AP_RND, AP_SAT> DTYPE;
	//typedef ap_fixed<32, 7, AP_RND, AP_SAT> FTYPE;
	//typedef ap_fixed<32, 7, AP_RND, AP_SAT> ITYPE;
	//typedef ap_fixed<32, 7, AP_RND, AP_SAT> QTYPE;
	//typedef ap_ufixed<32, 1> STYPE; //scaling factor type

    //typedef ap_fixed<16, 3> ATYPE;
    //typedef ap_fixed<8, 0> ATYPE;
    //typedef ap_fixed<16, 3> BTYPE;
	//typedef ap_fixed<16, 3> DTYPE;
	//typedef ap_fixed<16, 3> FTYPE;
	//typedef ap_fixed<32, 7, AP_RND,AP_SAT> ITYPE;
	//typedef ap_fixed<32, 7> ITYPE;
	//typedef ap_fixed<16, 3> QTYPE;
    //typedef ap_ufixed<16, 1> STYPE; //scaling factor type
    //#define frac_bits 13

    //binary mode
    //typedef ap_ufixed<1, 1> ATYPE;
    //typedef ap_fixed<1, 1> BTYPE;
	//typedef ap_ufixed<1, 1> FTYPE;
	//typedef ap_fixed<32, 16> DTYPE;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
	//typedef ap_fixed<8, 1> QTYPE;
    //typedef ap_ufixed<32, 1> STYPE; //scaling factor type

    //ternary mode
    //typedef ap_ufixed<2, 1> ATYPE;
    //typedef ap_fixed<2, 1> BTYPE;
	//typedef ap_ufixed<2, 1> FTYPE;
	//typedef ap_fixed<32, 16> DTYPE;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
	//typedef ap_fixed<8, 1> QTYPE;
    //typedef ap_ufixed<32, 1> STYPE; //scaling factor type

    //8-bit signed mode
    //typedef ap_fixed<8, 1> INTYPE; //interface to DDR type set to 8-bit
    //typedef ap_fixed<8, 1> ATYPE;
    //typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to 8-bit
    //typedef ap_fixed<8, 1> BTYPE;
	//typedef ap_fixed<8, 1> FTYPE;
	//typedef ap_fixed<32, 16> DTYPE;
    //typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE; //narrow types does not save much because it is qtype the one buffer. This type makes sure that there are enough bits so dense data can still be processed.
    //typedef ap_fixed<8, 1, AP_TRN_ZERO>  QTYPE;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> TTYPE; //attention computing type
    //typedef ap_ufixed<32, 1> STYPE; //scaling factor type

    //16-bit unsigned mode
    //typedef ap_ufixed<16, 1> INTYPE; //interface to DDR type set to 8-bit
    //typedef ap_ufixed<16, 1> ATYPE;
    //typedef ap_fixed<16, 1> INTYPES; //interface to DDR type set to 8-bit
    //typedef ap_fixed<16, 1> BTYPE;
	//typedef ap_ufixed<16, 1> FTYPE;
	//typedef ap_fixed<32, 16> DTYPE;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
	//typedef ap_fixed<32, 1, AP_TRN_ZERO>  QTYPE;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> TTYPE; //attention computing type
    //typedef ap_int<8> STYPE; //scaling factor type

    //8-bit unsigned mode normal
    //typedef ap_axiu<8,0,0,0> ASTYPE; //axi stream type
    //typedef ap_ufixed<8, 1> INTYPE; //interface to DDR type set to 8-bit
    //typedef ap_ufixed<8, 1> ATYPE;
    //typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to 8-bit
    //typedef ap_fixed<8, 1> BTYPE;
	//typedef ap_ufixed<8, 1> FTYPE;
	//typedef ap_fixed<32, 16> DTYPE;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
	//typedef ap_fixed<16, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> TTYPE; //attention computing type
    //typedef ap_int<8> STYPE; //scaling factor type

    //8-bit signed mode

#if (SIGNED_MODE == 1)

    typedef ap_axis<32,0,0,0> ASTYPE; //axi stream type


    #if (INT_QUANT == 1)
       typedef float INTYPE; //interface to DDR type set to float
    #else
       typedef ap_fixed<8, 1> INTYPE;
    #endif
    typedef ap_fixed<8, 1> ATYPE;
    #if (INT_QUANT == 1)
       typedef float INTYPES; //interface to DDR type set to float (weights)
    #else
       typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to float (weights)
    #endif
    typedef ap_fixed<8, 1> BTYPE;
	typedef ap_fixed<8, 1> FTYPE;
	typedef ap_fixed<32, 16> DTYPE;

    #if (INT_QUANT == 1)
	   typedef float OUTTYPE;
    #else
		typedef ap_fixed<32, 16> OUTTYPE;
    #endif
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
	typedef ap_fixed<32, 16> ITYPE;
	//typedef ap_fixed<32, 16, AP_TRN_ZERO,AP_SAT> RTYPE;
	//typedef ap_fixed<16, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	typedef ap_fixed<2, 1, AP_TRN_ZERO,AP_SAT>  QTYPE2;
	typedef ap_fixed<4, 1, AP_TRN_ZERO,AP_SAT>  QTYPE4;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE8;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> TTYPE; //attention computing type
	typedef ap_fixed<32, 16> TTYPE; //attention computing type
    typedef ap_int<8> STYPE; //scaling factor type
    //#define alpha_qu  0
    //#define beta_qu  255
    //#define alpha_q  -127
    //#define beta_q  127
    #define zero_point  0.0
    #define qbits 8


    //8-bit unsigned mode
#else


    //typedef float ASTYPE; //axi stream type
    union fp_int{
    int i;
    float f;
    };

    typedef ap_axis<32,0,0,0> ASTYPE; //axi stream type
    #if (INT_QUANT == 1)
       typedef float INTYPE; //interface to DDR type set to float
    #else
       typedef ap_ufixed<8, 1> INTYPE;
    #endif
    typedef ap_ufixed<8, 1> ATYPE;
    #if (INT_QUANT == 1)
       typedef float INTYPES; //interface to DDR type set to float (weights)
    #else
       typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to float (weights)
    #endif
    typedef ap_fixed<8, 1> BTYPE;
	typedef ap_ufixed<8, 1> FTYPE;
	typedef ap_fixed<8, 1> LTYPE; //linear operator type
	typedef ap_fixed<32, 16> DTYPE;

    #if (INT_QUANT == 1)
	   typedef float OUTTYPE;
    #else
		typedef ap_fixed<32, 16> OUTTYPE;
    #endif
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
	typedef ap_fixed<32, 16> ITYPE;
	//typedef ap_fixed<32, 16, AP_TRN_ZERO,AP_SAT> RTYPE;
	//typedef ap_fixed<16, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	typedef ap_fixed<2, 1, AP_TRN_ZERO,AP_SAT>  QTYPE2;
	typedef ap_fixed<4, 1, AP_TRN_ZERO,AP_SAT>  QTYPE4;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE8;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> TTYPE; //attention computing type
	typedef ap_fixed<32, 16> TTYPE; //attention computing type
    typedef ap_int<8> STYPE; //scaling factor type
    //#define alpha_qu  0
    //#defineh beta_qu  255
    //#define alpha_q  -127
    //#define beta_q  127
    #define zero_point  0.0
    #define qbits 8
    //linear EX special precision
    #define qbitsl 8
    //#define f_alignl 0
    //#define beta_qul 255


#endif

    //4-bit unsigned mode
    //typedef ap_axiu<8,0,0,0> ASTYPE; //axi stream type
    //typedef ap_ufixed<8, 1> INTYPE; //interface to DDR type set to 8-bit
    //typedef ap_ufixed<4, 1> ATYPE;
    //typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to 8-bit
    //typedef ap_fixed<4, 1> BTYPE;
	//typedef ap_ufixed<4, 1> FTYPE;
	//typedef ap_fixed<32, 16> DTYPE;
	//typedef ap_fixed<26, 16, AP_RND,AP_SAT> ITYPE;
	//typedef ap_fixed<8, 1, AP_TRN_ZERO>  QTYPE;
	//typedef ap_fixed<26, 16, AP_RND,AP_SAT> TTYPE; //attention computing type
    //typedef ap_int<8> STYPE; //scaling factor type

      //2-bit unsigned mode
/*
    typedef ap_axiu<8,0,0,0> ASTYPE; //axi stream type
    #if (INT_QUANT == 1)
       typedef float INTYPE; //interface to DDR type set to float
    #else
       typedef ap_ufixed<8, 1> INTYPE;
    #endif
    typedef ap_ufixed<2, 1> ATYPE;
    #if (INT_QUANT == 1)
       typedef float INTYPES; //interface to DDR type set to float (weights)
    #else
       typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to float (weights)
    #endif
    typedef ap_fixed<2, 1> BTYPE;
	typedef ap_ufixed<2, 1> FTYPE;
	typedef ap_fixed<32, 16> DTYPE;

    #if (INT_QUANT == 1)
	   typedef float OUTTYPE;
    #else
		typedef ap_fixed<32, 16> OUTTYPE;
    #endif
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
	typedef ap_fixed<22, 16> ITYPE;
	//typedef ap_fixed<32, 16, AP_TRN_ZERO,AP_SAT> RTYPE;
	typedef ap_fixed<4, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	typedef ap_fixed<2, 1, AP_TRN_ZERO,AP_SAT>  QTYPE2;
	typedef ap_fixed<4, 1, AP_TRN_ZERO,AP_SAT>  QTYPE4;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE8;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> TTYPE; //attention computing type
	typedef ap_fixed<22, 16> TTYPE; //attention computing type
    typedef ap_int<8> STYPE; //scaling factor type
    //#define alpha_qu  0
    //#define beta_qu  255
    //#define alpha_q  -127
    //#define beta_q  127
    #define zero_point  0.0
    #define qbits 2
*/

    //1-bit unsigned mode

//typedef float ASTYPE; //axi stream type
/*
union fp_int{
    int i;
    float f;
};


 typedef ap_axiu<8,0,0,0> ASTYPE; //axi stream type
    #if (INT_QUANT == 1)
       typedef float INTYPE; //interface to DDR type set to float
    #else
       typedef ap_ufixed<8, 1> INTYPE;
    #endif
    typedef ap_ufixed<2, 1> ATYPE;
    #if (INT_QUANT == 1)
       typedef float INTYPES; //interface to DDR type set to float (weights)
    #else
       typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to float (weights)
    #endif
    typedef ap_fixed<2, 1> BTYPE;
	typedef ap_ufixed<2, 1> FTYPE;
	typedef ap_fixed<32, 16> DTYPE;

    #if (INT_QUANT == 1)
	   typedef float OUTTYPE;
    #else
		typedef ap_fixed<32, 16> OUTTYPE;
    #endif
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
	typedef ap_fixed<18, 16> ITYPE;
	//typedef ap_fixed<32, 16, AP_TRN_ZERO,AP_SAT> RTYPE;
	typedef ap_fixed<2, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	typedef ap_fixed<2, 1, AP_TRN_ZERO,AP_SAT>  QTYPE2;
	typedef ap_fixed<4, 1, AP_TRN_ZERO,AP_SAT>  QTYPE4;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE8;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> TTYPE; //attention computing type
	typedef ap_fixed<18, 16> TTYPE; //attention computing type
    typedef ap_int<8> STYPE; //scaling factor type
    //#define alpha_qu  0
    //#define beta_qu  255
    //#define alpha_q  -127
    //#define beta_q  127
    #define zero_point  0.0
    #define qbits 1

    */

    //2-bit unsigned mode
    //typedef ap_axiu<8,0,0,0> ASTYPE; //axi stream type
    //typedef ap_ufixed<8, 1> INTYPE; //interface to DDR type set to 8-bit
    //typedef ap_ufixed<2, 1> ATYPE;
    //typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to 8-bit
    //typedef ap_fixed<2, 1> BTYPE;
   	//typedef ap_ufixed<2, 1> FTYPE;
   	//typedef ap_fixed<32, 16> DTYPE;
   	//typedef ap_fixed<18, 16, AP_RND,AP_SAT> ITYPE;
   	//typedef ap_fixed<4, 1, AP_TRN_ZERO>  QTYPE;
   	//typedef ap_fixed<18, 16, AP_RND,AP_SAT> TTYPE; //attention computing type
    //typedef ap_int<8> STYPE; //scaling factor type

    //1-bit unsigned mode
    //typedef ap_axiu<8,0,0,0> ASTYPE; //axi stream type
    //typedef ap_ufixed<8, 1> INTYPE; //interface to DDR type set to 8-bit
    //typedef ap_ufixed<2, 1> ATYPE;
    //typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to 8-bit
    //typedef ap_fixed<2, 1> BTYPE;
    //typedef ap_ufixed<2, 1> FTYPE;
   	//typedef ap_fixed<32, 16> DTYPE;
   	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
   	//typedef ap_fixed<16, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
   	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> TTYPE; //attention computing type
    //typedef ap_int<8> STYPE; //scaling factor type


    //normal mode
    //typedef ap_ufixed<8, 1> ATYPE;
    //typedef ap_fixed<8, 1> BTYPE;
	//typedef ap_ufixed<8, 1> FTYPE;
	//typedef ap_fixed<32, 16> DTYPE;
	//typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
	//typedef ap_fixed<16, 1> QTYPE;
	//typedef ap_fixed<8, 1> QTYPE;
    //typedef ap_ufixed<32, 1> STYPE; //scaling factor type


    //typedef ap_ufixed<8, 0> ATYPE;
    //typedef ap_fixed<8, 0> BTYPE;
	//typedef ap_fixed<32, 7> DTYPE;
	//typedef ap_fixed<8, 0> FTYPE;
	//typedef ap_fixed<32, 7, AP_RND,AP_SAT> ITYPE;
	//typedef ap_fixed<32, 7> QTYPE;
    //typedef ap_ufixed<32, 1> STYPE; //scaling factor type
    //#define frac_bits 25

    /*typedef ap_fixed<32, 16> ATYPE;
    typedef ap_fixed<32, 16> BTYPE;
	typedef ap_fixed<32, 16> DTYPE;
	typedef ap_fixed<32, 16> FTYPE;
	typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
	typedef ap_fixed<32, 16> QTYPE;
    typedef ap_ufixed<32, 1> STYPE; //scaling factor type
    #define frac_bits 16*/



    //typedef ap_fixed<32, 32> ATYPE;
    //typedef ap_fixed<32, 32> BTYPE;
	//typedef ap_fixed<32, 32> DTYPE;
	//typedef ap_fixed<32, 32> FTYPE;
	//typedef ap_fixed<32, 32, AP_RND,AP_SAT> ITYPE;
	//typedef ap_fixed<32, 32> QTYPE;
    //typedef ap_ufixed<32, 1> STYPE; //scaling factor type
    //#define frac_bits 0

    //typedef ap_fixed<16, 9> ATYPE;
    //typedef ap_fixed<16, 9> BTYPE;
	//typedef ap_fixed<16, 9> DTYPE;
	//typedef ap_fixed<16, 9> FTYPE;
	//typedef ap_fixed<16, 9> ITYPE;
	//typedef ap_fixed<16, 9> QTYPE;
    //typedef ap_ufixed<16, 1> STYPE; //scaling factor type
    //#define frac_bits 7

    //typedef ap_fixed<16, 7> ATYPE;
    //typedef ap_fixed<16, 7> BTYPE;
	//typedef ap_fixed<16, 7> DTYPE;
	//typedef ap_fixed<16, 7> FTYPE;
	//typedef ap_fixed<16, 7> ITYPE;
	//typedef ap_fixed<16, 7> QTYPE;
    //typedef ap_ufixed<16, 1> STYPE; //scaling factor type
    //#define frac_bits 9

	//typedef ap_fixed<8, 2> ATYPE;
	//typedef ap_fixed<8, 2> BTYPE;
	//typedef ap_fixed<8, 2> DTYPE;
	//typedef ap_fixed<8, 2> FTYPE;
	//typedef ap_fixed<8, 2> ITYPE;
	//typedef ap_fixed<8, 2> QTYPE;
    //typedef ap_ufixed<32, 1> STYPE; //scaling factor type
    //#define frac_bits 6

    #define FTYPE_LATENCY_ADJ 1
    #define FTYPE_LATENCY_FEA 1

	//typedef ap_fixed<32, 2> ATYPE;
	//typedef ap_fixed<32, 2> BTYPE;
	//typedef ap_fixed<32, 2> DTYPE;
	//typedef ap_fixed<32, 2> FTYPE;
	//typedef ap_fixed<32, 2> ITYPE;


#endif

#ifdef HALF

	typedef half ATYPE;
	typedef half BTYPE;
	typedef half DTYPE;
	typedef half FTYPE;
	typedef half ITYPE;
    typedef half STYPE; //scaling factor type
	typedef half QTYPE;
	#define FTYPE_LATENCY_ADJ 4
	#define FTYPE_LATENCY_FEA 4
    #define frac_bits 0
#endif

#ifdef FLOAT

	typedef float ATYPE;
	typedef float BTYPE;
	typedef float DTYPE;
	typedef float FTYPE;
	typedef float ITYPE;
	//#define FTYPE_LATENCY 4 //100 MHZ
	#define FTYPE_LATENCY_ADJ 6
	#define FTYPE_LATENCY_FEA 6 //200 MHZ
#endif

//#ifdef FLOAT
//	typedef ap_uint<DTYPE_LENGTH> DTYPE;
//	typedef ap_int<32> DTYPE_OUT;
//	typedef ap_int<8> ATYPE;
//  	#define FTYPE_LATENCY 6   //optimal latency is 6 for 200 MHz with 3 II increases but we reduce pressure on logic
//#endif

//#ifdef HALF
//	typedef half FTYPE;
//        #define FTYPE_LATENCY 4
//	typedef float FTYPE;
//#endif


#define USE_SBLOCKS 0

/*the compute unit always uses sblocks, the USE_SBLOCKS controls if the write unit also uses sblocks and reads multiple FIFO channels
 * or reads only one FIFO channel per core and this is generally better because it optimizes the loop
 * that writes data to memory
 */

/* spmm block controls how many rows of the sparse matrix are processed in a single for loop. In principle only one
 * row is processed and then a matrix mult output is written into the C buffer memory. If only a few elements in the row
 * are nonzero then the overhead is significant since the loop needs to start again for the next row. The loop
 * achieves II 1 but if the number of elements of the row is small the flushing the pipeline and restarting the row
 * is an overhead. By grouping several rows in a single loop it is possible to alleviate this problem and have more nonzeros to process
 */

#define A_HEIGHT_BLOCK  1// 4096 //(512/4)

#define FAST_ATTENTION 1


#define B_BLOCK_PARALLEL 1
//#define PES_ADJ 2 // Number of PEs for ADJ processing
//#define PES_FEA 2 // Number iof PEs for FEA processing


#define ENABLE_GEMM
#define ENABLE_SPMM
//#define ENABLE_SCALING
//#define ENABLE_TRANSPOSE


//typedef ap_int<32> c_fifo_t;
//typedef hls::stream<c_fifo_t> c_fifo_stream_t;

//how many rows of B are computed in parallel in multiplication loop
//for example a couple of B rows are multiplied for A 1 row in each loop iteration
//it basically reduces how the loop iterations by 2 if it is 2.
/*
 A_HEIGHT_BLOCK  is for software part data partitioning due to the limitation in 
 the Xilinx kernel sds_alloc so A_HEIGHT_BLOCK should be A_HEIGHT divided by 
 the number of considered blocks
*/
#define C_HEIGHT_BLOCK  A_HEIGHT_BLOCK 


//typedef unsigned long u32;

typedef std::vector<int> vi;



//int mmult_accel(ap_uint<2> ternary, ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int M, DTYPE A[A_HEIGHT_BLOCK][A_WIDTH], DTYPE B[B_HEIGHT][B_WIDTH_BLOCK], DTYPE_OUT C[C_HEIGHT_BLOCK][B_WIDTH_BLOCK]);
//#pragma SDS data access_pattern(A:SEQUENTIAL)


//#pragma SDS data zero_copy(B[0:B_HEIGHT*B_WIDTH], C[0:A_HEIGHT*B_WIDTH])
//#pragma SDS data copy(A[0:A_HEIGHT*A_WIDTH])
//#pragma SDS data access_pattern(A:RANDOM)
//#pragma SDS data zero_copy(A[0:A_HEIGHT*A_WIDTH],B[0:B_HEIGHT*B_WIDTH], C[0:A_HEIGHT*B_WIDTH])
//#pragma SDS data sys_port(A:AFI,B:AFI,C:AFI)
//#pragma SDS data access_pattern(A:ACP;B:ACP;C:ACP)
//#pragma SDS data zero_copy(A[0:line_count*A_WIDTH],B[0:B_HEIGHT*B_WIDTH], C[0:A_HEIGHT*B_WIDTH])
//int mmult_top(ap_uint<2> ternary,ap_int<32> *quantized_multiplier, ap_int<32> *shift, ap_int<32> *bias, ap_int<32> bias_count,ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, ap_int<8> zero_point_dst,ap_int<8> clamp_max,ap_int<8> clamp_min,int N, int M, int P, DTYPE* A, DTYPE* B, DTYPE* C);
//int mmult_top2(int A[A_WIDTH*A_HEIGHT], int B[B_HEIGHT*B_WIDTH], int C[C_HEIGHT*C_WIDTH],int line_count);
//int mmult_top3(int A[A_WIDTH*A_HEIGHT], int B[B_HEIGHT*B_WIDTH], int C[C_HEIGHT*C_WIDTH],int line_count);
//int mmult_top4(int A[A_WIDTH*A_HEIGHT], int B[B_HEIGHT*B_WIDTH], int C[C_HEIGHT*C_WIDTH],int line_count);

#endif //__MATRIX_MULT_H__



