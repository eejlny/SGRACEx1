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

#define MAX_N    4096
#define MAX_M    2048
#define MAX_P    16

#define MAX_FIFO 16
#define PIPO_BLOCKS 2 // use a PIPO with only one block to save memory instead of standard 2

#define A_HEIGHT   MAX_N
#define F_HEIGHT   MAX_N
#define B_HEIGHT   MAX_M
#define B_WIDTH    MAX_P
#define C_WIDTH    MAX_P

#define EIGHTBIT


#define SIGNED_MODE 0 //use when training configuration so all inputs are signed
#define COO_MODE 1
#define INT_QUANT 1 //internal quantization so interface to DDR set to float
#define INT_DEQUANT 1 //remove the scaling of the quantization and the internal scaling in the output (needed and done in hardware)
#define FEA_THREADS 1
#define ADJ_THREADS 1
#define GAT_ENABLE 0 //implement support for GAT
#define GNN_ONLY 0 //implement support only for GNN (not linear)
#define LINEAR_ONLY 0 //implement support only for LINEAR
#define GATV2 0 //usegatv2 modificatino or standard gat
#define SPMM_BLOCK 1
#define ATEN_BLOCK 1 //leave aten block at 1 since dataflow issues otherwise
#define OPT_ATTN 1 ////OPT_ATTN control how much sparsity expected in adj (e.g. OPT_ATTN = 1 worse case with fully dense adjacency,OPT_ATTN = 8 12.5% non zeros in adj) (this is per row, some rows have quite a few nonzeros)
#define B_WIDTH_BLOCK MAX_P //96 //attention cannot do blocks so this value has to match the W width




#ifdef EIGHTBIT


 #if (SIGNED_MODE == 1)


    //8-bit unsigned mode
 #else


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
	typedef ap_fixed<32, 16> ITYPE;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	typedef ap_fixed<2, 1, AP_TRN_ZERO,AP_SAT>  QTYPE2;
	typedef ap_fixed<4, 1, AP_TRN_ZERO,AP_SAT>  QTYPE4;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE8;
	typedef ap_fixed<32, 16> TTYPE; //attention computing type
    typedef ap_int<8> STYPE; //scaling factor type
    #define zero_point  0.0
    #define qbits 8
    #define qbitsl 8

 #endif

 #define FTYPE_LATENCY_ADJ 1
 #define FTYPE_LATENCY_FEA 1

#endif

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
#define B_BLOCK_PARALLEL 1


#define C_HEIGHT_BLOCK  A_HEIGHT_BLOCK 


typedef std::vector<int> vi;


#endif //__MATRIX_MULT_H__



