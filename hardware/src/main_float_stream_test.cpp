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

#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <memory>
#include <sys/time.h>
#include <algorithm> // for std::find
#include <iterator> // for std::begin, std::end\
#include <math.h> 

#include <string>
#include <fstream>
#include <sstream> // std::stringstream

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "matrix_mult.h"
#include "kernelMatrixmult.h"
#include <hls_math.h>


#define max_N_adj MAX_N
#define max_M_fea MAX_M
#define max_P_w MAX_P



float max_adj=0.0;
float min_adj=0.0;
float max_fea=0.0;
float min_fea=0.0;

//cora data
//#define gcn_coo_cora
//#define dense_test
#define sparse_test
//#define linear_dense

//#define gcn_coo_cora //ok pass


#ifdef gcn_coo_cora

float deq_factor_0=4.0631776;
int scale_fea_0=3;

bool gemm_mode = 0;
bool stream_mode1=1;
bool linear_mode=0;

//8-bit
int f_align = 0;
int beta_qu = 255;


float quantization_scale_adj = 255.0;
float quantization_scale_fea_0= 255.0;
float quantization_scale_w_0= 127.0;
float quantization_scale_lin_0= 31.75;



float zero_point_adj = 0.0;
int qbits_adj=8;
int beta_q_adj = 255;

bool gat_mode = 0;
bool relu = 1;
int N_adj = 2708;  // number of nodes
int M_fea = 1433;  // number of input features
int P_w = 16;  // number of features in the hidden layer
int NNZ_adj = 13264;  // number of non-zero values of adjacency
int NNZ_fea = 49216;  // number of non-zero values of feature

static const std::string adj_name = "../../../../data2/gcn_adj_coo_cora.txt";
static const std::string fea_name = "../../../../data2/gcn_fea_coo_cora.txt";
static const std::string d_name = "../../../../data2/gcn_out_cora.txt";
static const std::string w_name = "../../../../data2/gcn_weights_cora.txt";
static const std::string ate_name = "../../../../data2/gcn_ate.txt";



#endif

#ifdef gcn_coo

float deq_factor_0=6.50108422;
int scale_fea_0=3;

bool gemm_mode = 0;
bool stream_mode1=1;
bool linear_mode=0;

//8-bit accb off
int f_align = 0;
int beta_qu = 255;

//8-bit accb off
//int f_align = 7;
//int beta_qu = 1;

float quantization_scale_adj = 1/0.0039215686;
float quantization_scale_fea_0=1/0.0039215686;
float quantization_scale_w_0=1/0.0062992125;
float quantization_scale_lin_0=1/0.0062992125;



float zero_point_adj = 0.0;
int qbits_adj=8;
int beta_q_adj = 255;
//STYPE scale_fea = 3;

bool gat_mode = 0;
bool relu = 1;
int N_adj = 2708;  // number of nodes
int M_fea = 1433;  // number of input features
int P_w = 16;  // number of features in the hidden layer
//int P_w = 2;  // number of features in the hidden layer
int NNZ_adj = 13264;  // number of non-zero values of adjacency
int NNZ_fea = 49216;  // number of non-zero values of feature
static const std::string adj_name = "../../../../../../gat-rfsoc-mt-all-2024-sage/data2/gcn_adj_coo.txt";
static const std::string fea_name = "../../../../../../gat-rfsoc-mt-all-2024-sage/data2/gcn_fea_coo.txt";
//static const std::string fea_name = "../../../../../data2/gcn_fea_coo_linear.txt";
static const std::string d_name = "../../../../../../gat-rfsoc-mt-all-2024-sage/data2/gcn_out.txt";
//static const std::string w_name = "../../../../../data2/gcn_weights.txt";
static const std::string w_name = "../../../../../../gat-rfsoc-mt-all-2024-sage/data2/gcn_weights.txt";
static const std::string ate_name = "../../../../../../gat-rfsoc-mt-all-2024-sage/data2/gcn_ate.txt";
#endif

#ifdef dense_test

float deq_factor_0=4.063;
int scale_fea_0=3;

bool gemm_mode = 1;
bool stream_mode1=1;
bool linear_mode=0;

//8-bit accb off
int f_align = 0;
int beta_qu = 255;

//8-bit accb off
//int f_align = 7;
//int beta_qu = 1;

float quantization_scale_adj=255.0;
float quantization_scale_fea_0=255.0;
float quantization_scale_w_0=127;
float quantization_scale_lin_0=31.75;



float zero_point_adj = 0.0;
int qbits_adj=8;
int beta_q_adj = 255;
//STYPE scale_fea = 3;

bool gat_mode = 0;
bool relu = 1;
int N_adj = 2162;  // number of nodes
int M_fea = 128;  // number of input features
int P_w = 64;  // number of features in the hidden layer
//int P_w = 2;  // number of features in the hidden layer
int NNZ_adj = 9026;  // number of non-zero values of adjacency
//int NNZ_fea = 4480;  // number of non-zero values of feature
int NNZ_fea = 276736;  // number of non-zero values of feature
static const std::string adj_name = "../../../../data2/sparse_adj.txt";
//static const std::string fea_name = "../../../../data2/sparse_fea.txt";
static const std::string fea_name = "../../../../data2/dense_fea.txt";
//static const std::string fea_name = "../../../../../data2/gcn_fea_coo_linear.txt";
static const std::string d_name = "../../../../data2/sparse_out.txt";
//static const std::string w_name = "../../../../../data2/gcn_weights.txt";
static const std::string w_name = "../../../../data2/sparse_weights.txt";
static const std::string ate_name = "../../../../data2/gcn_ate.txt";
#endif

#ifdef sparse_test

float deq_factor_0=4.063;
int scale_fea_0=3;

bool gemm_mode = 0;
bool stream_mode1=1;
bool linear_mode=0;

//8-bit accb off
int f_align = 0;
int beta_qu = 255;

//8-bit accb off
//int f_align = 7;
//int beta_qu = 1;

float quantization_scale_adj=255.0;
float quantization_scale_fea_0=255.0;
float quantization_scale_w_0=127;
float quantization_scale_lin_0=31.75;



float zero_point_adj = 0.0;
int qbits_adj=8;
int beta_q_adj = 255;
//STYPE scale_fea = 3;

bool gat_mode = 0;
bool relu = 1;
int N_adj = 2162;  // number of nodes
int M_fea = 128;  // number of input features
int P_w = 64;  // number of features in the hidden layer
//int P_w = 2;  // number of features in the hidden layer
int NNZ_adj = 9026;  // number of non-zero values of adjacency
int NNZ_fea = 4480;  // number of non-zero values of feature
//int NNZ_fea = 276736;  // number of non-zero values of feature
static const std::string adj_name = "../../../../data2/sparse_adj.txt";
static const std::string fea_name = "../../../../data2/sparse_fea.txt";
static const std::string d_name = "../../../../data2/sparse_out.txt";
static const std::string w_name = "../../../../data2/sparse_weights.txt";
static const std::string ate_name = "../../../../data2/gcn_ate.txt";
#endif

#ifdef linear_dense

float deq_factor_0=32.5;
int scale_fea_0=1;

bool gemm_mode = 1;
bool stream_mode1=0;
bool linear_mode=1;


//int beta_qu = 255;
//int f_align = 0;
int beta_qu = 1; //
int f_align = 7;

float quantization_scale_adj = 1/0.0039215686;
float quantization_scale_fea_0=1/0.0039215686;
float quantization_scale_w_0=254.0;
float quantization_scale_lin_0=1/0.0039215686; //15.8;

float zero_point_adj = 0.0;
int qbits_adj=8;
int beta_q_adj = 255;
//STYPE scale_fea = 3;

bool gat_mode = 0;
bool relu = 1;
int N_adj = 2708;  // number of nodes
int M_fea = 16;  // number of input features
int P_w = 7;  // number of features in the hidden layer
int NNZ_adj = 0;  // number of non-zero values of adjacency
int NNZ_fea = N_adj*M_fea;  // number of non-zero values of feature
static const std::string adj_name = "../../../../../../gat-rfsoc-mt-all-2024-sage/data2/gcn_adj_coo.txt";
static const std::string fea_name = "../../../../../../gat-rfsoc-mt-all-2024-sage/data2/linear_in.txt";
static const std::string d_name = "../../../../../../gat-rfsoc-mt-all-2024-sage/data2/linear_out.txt";
static const std::string w_name = "../../../../../../gat-rfsoc-mt-all-2024-sage/data2/linear_weights.txt";
static const std::string ate_name = "../../../../../../gat-rfsoc-mt-all-2024-sage/data2/gcn_ate.txt";


#endif



#ifdef gat_coo
#define use_gemm 0

//int f_align=7;
//float quantization_scale_adj=1/0.1;
//float quantization_scale_fea=1/1.0;
//float quantization_scale_w=1/0.1;
//float deq_factor=0.6400000001;
//int beta_qu=1;

//float quantization_scale_adj = 1/0.0039;
//float quantization_scale_fea = 1/0.0039215686;
//float quantization_scale_w =   1/0.0062992125;
//float deq_factor = 6.50108422;

float zero_point_adj = 0.0;
int qbits_adj=8;
int beta_q_adj = 255;
//STYPE scale_fea = 2;

bool gat_mode = 1;
bool relu = 1;
int N_adj = 2708;  // number of nodes
int M_fea = 1433;  // number of input features
int P_w = 16;  // number of features in the hidden layer
//int P_w = 2;  // number of features in the hidden layer
int NNZ_adj = 13264;  // number of non-zero values of adjacency
int NNZ_fea = 49216;  // number of non-zero values of feature
static const std::string adj_name = "../../../../../data2/gcn_adj_coo.txt";
static const std::string fea_name = "../../../../../data2/gcn_fea_coo.txt";
static const std::string d_name = "../../../../../data2/gat_out.txt";
//static const std::string w_name = "../../../../../data2/gcn_weights.txt";
static const std::string w_name = "../../../../../data2/gcn_weights.txt";
static const std::string ate_name = "../../../../../data2/gat_ate.txt";
#endif




double getTimestamp() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_usec + tv.tv_sec * 1e6;
}



void quant_adj(ATYPE &BW,float B,float quantization_scale,int f_align, int beta_qu)
{

    //std::cout << "fea in " << B << std::endl;
	float vfloat = quantization_scale*B+zero_point;
	float vround = hls::round(vfloat);


    //std::cout << "VROUND " << vround << std::endl;

	float ibeta_qu = (float)beta_qu;
	float ialpha_qu = (float)(0.0);


    //clippping
	if (vfloat>ibeta_qu)
		vfloat = ibeta_qu;
	else if (vfloat<ialpha_qu)
		vfloat = ialpha_qu;

	ITYPE vquant = ITYPE(vfloat);

    //std::cout << "FQUANT " << vquant << std::endl;
	if(f_align==7) //BINARY MODE
		f_align = 6;
	ITYPE vnorm = vquant >> (qbits-f_align-1);
 	ATYPE fval = ATYPE(vnorm);

    //std::cout << "FNORM " << fval << std::endl;

	BW = fval;


}

void quant_fea(FTYPE &BW,float B,float quantization_scale,int f_align, int beta_qu)
{

    //std::cout << "fea in " << B << std::endl;
	float vfloat = quantization_scale*B+zero_point;
	float vround = hls::round(vfloat);



    //std::cout << "VROUND " << vround << std::endl;

	float ibeta_qu = (float)beta_qu;
	float ialpha_qu = (float)(0.0);


    //clippping
	if (vfloat>ibeta_qu)
		vfloat = ibeta_qu;
	else if (vfloat<ialpha_qu)
		vfloat = ialpha_qu;

	ITYPE vquant = ITYPE(vround);

    //std::cout << "FQUANT " << vquant << std::endl;
	if(f_align==7) //BINARY MODE
		f_align = 6;
	ITYPE vnorm = vquant >> (qbits-f_align-1);
 	FTYPE fval = FTYPE(vnorm);

    //std::cout << "FNORM " << fval << std::endl;



	BW = fval;


}



void quant_w(BTYPE &BW,float B,float quantization_scale,int f_align, int beta_qu)
{

	float vfloat = quantization_scale*B+zero_point;

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
    	vround = hls::round(vfloat);
    }


	//float vround = hls::round(vfloat);


    //std::cout << vround << " ";


	ITYPE vquant = ITYPE(vround);

	//std::cout << "WQUANT " << vquant << std::endl;



    //std::cout << "VROUND " << vround << std::endl;



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



static void init_weights(INTYPE *B, DTYPE *C_sw, DTYPE *C)
{
     for (int i = 0; i < M_fea; i++) {
          for (int j = 0; j < P_w; j++) {
		   float B_float = 0.4;
        	   B[i * P_w + j] =  (INTYPE)B_float;
		   //std::cout << "B value is "<< B[i * P_w + j] << std::endl;
          }
     }
     for (int i = 0; i < N_adj; i++) {
          for (int j = 0; j < P_w; j++) {
               C_sw[i * P_w + j] = 0;
               C[i * P_w + j] = 0;
          }
     }
}

static void load_result_lines(int N,int P,float *A,std::string file_name)
{

	// Create an input filestream
         std::ifstream myFile(file_name);

	// Make sure the file is open
    std::cout <<  "the file is " << file_name << std::endl;
	if(!myFile.is_open()) throw std::runtime_error("Could not open float file");

	// Helper vars
	std::string line;
	float val;
	int val_count=0;
	int val_zero=0;
	float array_val;



    for (int i = 0; i < N; i++) {
    	// Read data, line by line

    	std::getline(myFile, line);

    	std::stringstream ss(line);


	    // Create a stringstream of the current line


        for (int j = 0; j < P; j++) {

	        //fill one array val
        	array_val = 0;
        	// Extract each integer
        	ss >> val;

        	if (val==0)
        		val_zero++;

        	array_val = (float)val;

	        // If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();

	        //A[i * N + j] = 16;
	        //std::cout << i <<" "<< j << " " << array_val << std::endl;
	        A[i*P + j] = array_val;
	        val_count++;
	  		//if (array_val > 1)
	 	  	//{
	  		//	std::cout << "array_val " << array_val << std::endl;
	 		//	exit(0);
	 	  	//}
	        //std::cout <<"result " << i <<" "<< j << " " << array_val << " " << A[i*P+j] << std::endl;

	    }
    }

    //if (M > 16)
	//exit(0);

    std::cout << "Total " << sizeof(float)*8  << " bit values in matrix " << val_count << std::endl;
    std::cout << "Total values set to zero in  matrix " << val_zero << std::endl;


}

static void load_weights(int M,int P,INTYPES *A,std::string file_name)
{

	// Create an input filestream
         std::ifstream myFile(file_name);

	// Make sure the file is open
    std::cout <<  "the file is " << file_name << std::endl;
	if(!myFile.is_open()) throw std::runtime_error("Could not open float file");

	// Helper vars
	std::string line;
	float val;
	int val_count=0;
	int val_zero=0;
	INTYPES array_val;




    for (int i = 0; i < P; i++) {

    	// Read data, line by line
        std::getline(myFile, line);

    	// Create a stringstream of the current line
    	std::stringstream ss(line);


        for (int j = 0; j < M; j++) {

	        //fill one array val
        	array_val = 0;
        	// Extract each integer
        	ss >> val;

        	if (val==0)
        		val_zero++;


            #if (INT_QUANT==1)
        	    array_val = (INTYPES)(val);
            #else

        	   BTYPE quant_val;

        	   quant_w(quant_val,val,quantization_scale_w,f_align,beta_qu);

        	   array_val = (BTYPE)(quant_val);

            #endif
	        // If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();

	        //A[i * N + j] = 16;
	        //std::cout << "weight value " << array_val <<  " float value " << val << std::endl;
	        A[i*M + j] = array_val;
	        val_count++;
	  		//if (array_val > 1)
	 	  	//{
	  		//	std::cout << "array_val " << array_val << std::endl;
	 		//	exit(0);
	 	  	//}
	        //std::cout << i <<" "<< j << " " << array_val << " " <<  A[i*M + j] << std::endl;

	    }
    }

    //if (M > 16)
	//exit(0);

    std::cout << "Total " << sizeof(INTYPE)*8  << " bit values in matrix " << val_count << std::endl;
    std::cout << "Total values set to zero in  matrix " << val_zero << std::endl;


}

static void load_attention(int N,int M,INTYPE *A,std::string file_name)
{

	// Create an input filestream
         std::ifstream myFile(file_name);

	// Make sure the file is open
    std::cout <<  "the file is " << file_name << std::endl;
	if(!myFile.is_open()) throw std::runtime_error("Could not open float file");

	// Helper vars
	std::string line;
	float val;
	int val_count=0;
	int val_zero=0;
	INTYPE array_val;

    for (int i = 0; i < N; i++) {
    	// Read data, line by line
    	std::getline(myFile, line);

	    // Create a stringstream of the current line
	    std::stringstream ss(line);


        for (int j = 0; j < M; j++) {

	        //fill one array val
        	array_val = 0;
        	// Extract each integer
        	ss >> val;

        	if (val==0)
        		val_zero++;

        	array_val = (INTYPE)val;

	        // If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();

	        //A[i * N + j] = 16;
	        //std::cout << i <<" "<< j << " " << array_val << std::endl;
	        A[i*M + j]  = array_val; //transpose so hardware can read column in contiguos address space
	        val_count++;
	  		//if (array_val > 1)
	 	  	//{
	  		//	std::cout << "array_val " << array_val << std::endl;
	 		//	exit(0);
	 	  	//}
	        //std::cout << i <<" "<< j << " " << array_val << " " << A[i * M + j] << std::endl;

	    }
    }

    //if (M > 16)
	//exit(0);

    std::cout << "Total " << sizeof(INTYPE)*8  << " bit values in matrix " << val_count << std::endl;
    std::cout << "Total values set to zero in  matrix " << val_zero << std::endl;
    

}





static void load_fea(int N,int M,INTYPE *A,std::string file_name)
{

	// Create an input filestream
         std::ifstream myFile(file_name);

	// Make sure the file is open
	if(!myFile.is_open()) throw std::runtime_error("Could not open float file");
	else
		std::cout << "reading dense fea " << file_name << " file" << std::endl;


	// Helper vars
	std::string line;
	float val;
	int val_count=0;
	int val_zero=0;
	INTYPE array_val;



    for (int i = 0; i < N; i++) {

    	std::getline(myFile, line);

        // Create a stringstream of the current line
        std::stringstream ss(line);

        for (int j = 0; j < M; j++) {

	        //fill one array val
        	array_val = 0;
        	// Extract each integer
        	ss >> val;


        	if (val==0)
        		val_zero++;

        	array_val = (INTYPE)val;

	        // If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();

	        //std::cout << "feature value " << array_val <<  " float value " << val << std::endl;

	        //A[i * N + j] = 16;
	        //std::cout << i <<" "<< j << " " << array_val << std::endl;
	        A[i * M + j] = array_val;
	        val_count++;
	        //std::cout << i <<" "<< j << " " << array_val << " " << A[i * M + j] << std::endl;

	    }
    }

    //if (M > 16)
	//exit(0);

    std::cout << "Total " << sizeof(INTYPE)*8  << " bit values in fea matrix " << val_count << std::endl;
    std::cout << "Total values set to zero in fea matrix " << val_zero << std::endl;
    

}



static void load_adj(int N,int M,INTYPE *A,std::string file_name)
{

	// Create an input filestream
         std::ifstream myFile(file_name);

	// Make sure the file is open
	if(!myFile.is_open()) throw std::runtime_error("Could not open float adj file");
	else
		std::cout << "reading " << file_name << " file" << std::endl;


	// Helper vars
	std::string line;
	float val;
	int val_count=0;
	int val_zero=0;
	INTYPE array_val;



    for (int i = 0; i < N; i++) {

    	std::getline(myFile, line);

        // Create a stringstream of the current line
        std::stringstream ss(line);

        for (int j = 0; j < M; j++) {

	        //fill one array val
        	array_val = 0;
        	// Extract each integer
        	ss >> val;

        	if (val==0)
        		val_zero++;

        	array_val = (INTYPE)val;

        	//std::cout << "adjacency value " << array_val <<  " float value " << val << std::endl;

	        // If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();

	        //A[i * N + j] = 16;
	        //std::cout << i <<" "<< j << " " << array_val << std::endl;
	        A[i * M + j] = array_val;
	        val_count++;
	        //std::cout << i <<" "<< j << " " << array_val << " " << A[i * M + j] << std::endl;

	    }
    }

    //if (M > 16)
	//exit(0);

    std::cout << "Total " << sizeof(INTYPE)*8  << " bit values in adj matrix " << val_count << std::endl;
    std::cout << "Total values set to zero in adj matrix " << val_zero << std::endl;


}



void mmult_golden(int N,int M, int P, DTYPE *A,  DTYPE *B, DTYPE *C)
{
     for (int row = 0; row < N; row++) {
          for (int col = 0; col < P; col++) {
        	   DTYPE result = 0;
               for (int k = 0; k < M; k++) {
       			//for(int z = 0; z < DTYPE_LENGTH; z+=8) {
       				DTYPE A_temp1 = A[row*M+k];
       				//ap_int<8> A_val = A_temp1.range(z+7,z);
				DTYPE A_val = A_temp1;
      				DTYPE B_temp = B[k*P+col];
           			result+=A_val*B_temp;
       			//}
               }
               C[row*P+col] = result;
               //std::cout << row << " " << col << " result is " << result << std::endl;
          }
          //std::cout << row << " " << col << " result is " << result << std::endl;
     }
}

static int result_check(STYPE scale_fea,int N,int P, OUTTYPE *D, float *D_sw)
{
	 bool pass = 1;
	 int error_count = 0;
     for (int i = 0; i < N*P; i++) {
             #if (INT_QUANT == 1)
    	           float DH = float(D[i]);
             #else
    	           float DH = deq_factor*float(D[i]);
             #endif
        	 if ((abs(D_sw[i]-DH)) > 0.20) {
        	               std::cout << "Mismatch: data index= " << i << " " << "golden = " << D_sw[i]
        	                         << ", kernel = " << DH << std::endl;
        	               return 0;
        	               pass = 0;
        	               error_count++;

        	          }
        	 //else
        	 //{
	         //      std::cout << "Match: data index= " << i << " "  << "golden = " << D_sw[i]
             //                      << ", kernel = " << DH  << std::endl;
        	 //}

     }

     if (pass == 1)
       std::cout << "All results matched" << std::endl;
     else
       std::cout << "Error count " << error_count << std::endl;
     return pass;
/*
     std::cout << "head 1" << std::endl;
     for (int i = 0; i < 2; i++) {
        for (int j = 0; j < P; j++) {
          //if (C_sw[i] != C[i]) {
          //     std::cout << "Mismatch: data index= " << i << " golden = " << C_sw[i]
          //               << ", kernel = " << C[i] << std::endl;
          //     return 1;
          //}
          //else
        	//  std::cout << "out :data index= " << i << " golden = " << C_sw[i] << std::endl;
	        std::cout << "out :data index= " << i << " " << j << " kernel = " << D[i*P+j] << std::endl;
         }
     }
     for (int i = 1000; i < 1002; i++) {
         for (int j = 0; j < P; j++) {
           //if (C_sw[i] != C[i]) {
           //     std::cout << "Mismatch: data index= " << i << " golden = " << C_sw[i]
           //               << ", kernel = " << C[i] << std::endl;
           //     return 1;
           //}
           //else
         	//  std::cout << "out :data index= " << i << " golden = " << C_sw[i] << std::endl;
 	        std::cout << "out :data index= " << i << " " << j << " kernel = " << D[i*P+j] << std::endl;
          }
      }
     /*std::cout << "head 2" << std::endl;
     for (int i = 0; i < 2; i++) {
        for (int j = 0; j < P; j++) {
          //if (C_sw[i] != C[i]) {
          //     std::cout << "Mismatch: data index= " << i << " golden = " << C_sw[i]
          //               << ", kernel = " << C[i] << std::endl;
          //     return 1;
          //}
          //else
        	//  std::cout << "out :data index= " << i << " golden = " << C_sw[i] << std::endl;
	        std::cout << "out :data index= " << i << " " << j << " kernel = " << D[P*N+i*P+j] << std::endl;
         }
     }
     for (int i = 1000; i < 1002; i++) {
          for (int j = 0; j < P; j++) {
            //if (C_sw[i] != C[i]) {
            //     std::cout << "Mismatch: data index= " << i << " golden = " << C_sw[i]
            //               << ", kernel = " << C[i] << std::endl;
            //     return 1;
            //}
            //else
          	//  std::cout << "out :data index= " << i << " golden = " << C_sw[i] << std::endl;
  	        std::cout << "out :data index= " << i << " " << j << " kernel = " << D[i*P+j] << std::endl;
           }
       }*/


  //   return 0;
}

static int softmax_check(int N,int P, DTYPE *D, DTYPE *D_sw)
{
     //for (int i = 0; i < P_w; i++) {
         //for (int j = 0; j < N_adj; j++) {
     for (int i = 0; i < 100; i++) {
	        std::cout << "out :data index= " << i << " kernel = " << D[i] << std::endl;
     }


     return 0;
}



void printVector(const vi& V, char* msg)
{

	std::cout << msg << "[ ";
	for_each(V.begin(), V.end(), [](int a) {
		std::cout << a << " ";
	});
	std::cout << "]" << std::endl;
}


void loadcsr_adj(
std::string file_name,
int   N,
int   M,
INTYPE *array_values,
int   *array_colIndices,
int   *array_rowPtr,
int   nnz_value)
{
	int i;
	// Helper vars
	std::string line;



	// Create an outuptu filestream
	//std::string file_name = "./weights_layer_" + std::to_string(layer_number) + ".csr";
	std::ifstream inFile(file_name);

	// Make sure the file is open
	if(!inFile.is_open()) 
		throw std::runtime_error("Could not open csr file");
	else
		std::cout << "reading " << file_name << " file" << std::endl;

    	// Read data, line by line
        // rowptr
    	std::getline(inFile, line);
	// Create a stringstream of the current line
	std::stringstream ss;

	ss << line;

	for (i = 0; i < N+1; i++) {
		int temp;
		ss >> temp; 
	        //std::cout << "row pointer " << array_rowPtr[i] << std::endl;

		//if (temp > nnz_value)
		//{
		//	std::cout << "Accumulated non-zeros " << array_rowPtr[i-1] << std::endl;
		//	array_rowPtr[i] = array_rowPtr[i-1];
		//}
		//else
		//{
			array_rowPtr[i] = temp;
		//}

        	// If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();
	}

	

        // column_index
    	std::getline(inFile, line);
        //int *check_cols = malloc(100 * max_N_adj*sizeof(int));
 
	// Create a stringstream of the current line

	ss.str("");
        ss.clear();
	ss <<  line;

        //std::cout << "ss: " << ss.str() << std::endl;

        //std::cout << "nnz_value: " << nnz_value << std::endl;

	for (i = 0; i <  nnz_value; i++) {
		ss >> array_colIndices[i]; 
		//check_cols[i] = array_colIndices[i];
		//std::cout << "array colindx " << array_colIndices[i] << std::endl;

        	// If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();
	}	

	//check if column is empty


    //for(i = 0; i< M;i++)
	//{
  	//	bool exists = std::find(std::begin(check_cols), std::end(check_cols), i) != std::end(check_cols);
	//	if (!exists)
	//	{
			//std::cout << "Attention: Column " << i << " is empty (loading it is not efficient) " << std::endl;
			//exit(0);
	//	}
	//}
				

        // values
    	std::getline(inFile, line);
	// Create a stringstream of the current line
	ss.str("");
        ss.clear();
	ss<<line;

	for (i = 0; i <  nnz_value; i++) {
	//for (i = 0; i <  10; i++) {
		float float_val;
		ss >> float_val;


        
        #if(INT_QUANT==1)
		   array_values[i] = (INTYPE)(float_val);
        #else

		  ATYPE quant_val;

		  quant_adj(quant_val,float_val,quantization_scale_adj,f_align,beta_qu);

		  array_values[i] = (ATYPE)(quant_val);
        #endif
        

		//std::cout << " float value " << float_val << " adjacency value " << array_values[i] << std::endl;

		//exit(0);
        // If the next token is a comma, ignore it and move on
	    if(ss.peek() == ',') ss.ignore();
	}


	inFile.close();
	std::cout << "Number of non-zeros adj values in CSR file: " << nnz_value << std::endl;
	std::cout << "adj matrix size: " << M*N << std::endl;
	std::cout << "Total percentage of zero values in adj: " << (float)(M*N-nnz_value)/(float)(M*N) << std::endl;
	//std::cout << "Total percentage of zero values per row in adj: " << std::endl;
        //for(int i=0;i<N;i++)
	//	std::cout << 100*(1-(float)NNZR[i]/(float)M) << ",";

        //std::cout << "array values and col indices size each uses " << A.size()   << " integers" << std::endl;
        //std::cout << "row ptr uses " << IA.size()   << " integers" << std::endl;
	//exit(0);


}


void loadcoo_adj(
std::string file_name,
int   N,
int   M,
INTYPE *array_values,
int   *array_colIndices,
int   *array_rowPtr,
int   nnz_value)
{
	int i;
	// Helper vars
	std::string line;



	// Create an outuptu filestream
	//std::string file_name = "./weights_layer_" + std::to_string(layer_number) + ".csr";
	std::ifstream inFile(file_name);

	// Make sure the file is open
	if(!inFile.is_open())
		throw std::runtime_error("Could not open csr file");
	else
		std::cout << "reading " << file_name << " file" << std::endl;

    	// Read data, line by line
        // rowptr
    	std::getline(inFile, line);
	// Create a stringstream of the current line
	std::stringstream ss;

	ss << line;

	for (i = 0; i < nnz_value; i++) {
		int temp;
		ss >> temp;
	        //std::cout << "row pointer " << array_rowPtr[i] << std::endl;

		//if (temp > nnz_value)
		//{
		//	std::cout << "Accumulated non-zeros " << array_rowPtr[i-1] << std::endl;
		//	array_rowPtr[i] = array_rowPtr[i-1];
		//}
		//else
		//{
			array_rowPtr[i] = temp;
		//}

        	// If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();
	}



        // column_index
    	std::getline(inFile, line);
        //int *check_cols = malloc(100 * max_N_adj*sizeof(int));

	// Create a stringstream of the current line

	ss.str("");
        ss.clear();
	ss <<  line;

        //std::cout << "ss: " << ss.str() << std::endl;

        //std::cout << "nnz_value: " << nnz_value << std::endl;

	for (i = 0; i <  nnz_value; i++) {
		ss >> array_colIndices[i];
		//check_cols[i] = array_colIndices[i];
		//std::cout << "array colindx " << array_colIndices[i] << std::endl;

        	// If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();
	}

	//check if column is empty


    //for(i = 0; i< M;i++)
	//{
  	//	bool exists = std::find(std::begin(check_cols), std::end(check_cols), i) != std::end(check_cols);
	//	if (!exists)
	//	{
			//std::cout << "Attention: Column " << i << " is empty (loading it is not efficient) " << std::endl;
			//exit(0);
	//	}
	//}


        // values
    	std::getline(inFile, line);
	// Create a stringstream of the current line
	ss.str("");
        ss.clear();
	ss<<line;

	for (i = 0; i <  nnz_value; i++) {
	//for (i = 0; i <  10; i++) {
		float float_val;
		ss >> float_val;



        #if(INT_QUANT==1)
		   array_values[i] = (INTYPE)(float_val);
        #else

		  ATYPE quant_val;

		  quant_adj(quant_val,float_val,quantization_scale_adj,f_align,beta_qu);

		  array_values[i] = (ATYPE)(quant_val);
        #endif


		//std::cout << " float value " << float_val << " adjacency value " << array_values[i] << std::endl;

		//exit(0);
        // If the next token is a comma, ignore it and move on
	    if(ss.peek() == ',') ss.ignore();
	}


	inFile.close();
	std::cout << "Number of non-zeros adj values in COO file: " << nnz_value << std::endl;
	std::cout << "adj matrix size: " << M*N << std::endl;
	std::cout << "Total percentage of zero values in adj: " << (float)(M*N-nnz_value)/(float)(M*N) << std::endl;
	//std::cout << "Total percentage of zero values per row in adj: " << std::endl;
        //for(int i=0;i<N;i++)
	//	std::cout << 100*(1-(float)NNZR[i]/(float)M) << ",";

        //std::cout << "array values and col indices size each uses " << A.size()   << " integers" << std::endl;
        //std::cout << "row ptr uses " << IA.size()   << " integers" << std::endl;
	//exit(0);


}

void load_aparam(
INTYPE *A_param,
int P_w)
{
for (int i=0;i<2*P_w;i++)
	A_param[i] = 1.0;
}

void loadcsr_fea(
std::string file_name,
int   N,
int   M,
INTYPE *array_values,
int   *array_colIndices,
int   *array_rowPtr,
int   nnz_value)
{
	int i;
	// Helper vars
	std::string line;



	// Create an outuptu filestream
	//std::string file_name = "./weights_layer_" + std::to_string(layer_number) + ".csr";
	std::ifstream inFile(file_name);

	// Make sure the file is open
	if(!inFile.is_open()) 
		throw std::runtime_error("Could not open csr file");
	else
		std::cout << "reading " << file_name << " file" << std::endl;

    	// Read data, line by line
        // rowptr
    	std::getline(inFile, line);
	// Create a stringstream of the current line
	std::stringstream ss;

	ss << line;

	for (i = 0; i < N+1; i++) {
		int temp;
		ss >> temp; 
	        //std::cout << "row pointer " << array_rowPtr[i] << std::endl;

		//if (temp > nnz_value)
		//{

		//	array_rowPtr[i] = array_rowPtr[i-1];
		//}
		//else
		//{
			array_rowPtr[i] = temp;
			//std::cout << "Accumulated non-zeros at row " << i << " " << array_rowPtr[i] << std::endl;
		//}

        	// If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();
	}

	

        // column_index
    	std::getline(inFile, line);
        //int check_cols[100 * max_N_adj];
 
	// Create a stringstream of the current line

	ss.str("");
        ss.clear();
	ss <<  line;

        //std::cout << "ss: " << ss.str() << std::endl;

        //std::cout << "nnz_value: " << nnz_value << std::endl;

	for (i = 0; i <  nnz_value; i++) {
		ss >> array_colIndices[i]; 
		//check_cols[i] = array_colIndices[i];
		//std::cout << "array colindx " << array_colIndices[i] << std::endl;

        	// If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();
	}	

	//check if column is empty


    //    for(i = 0; i< M;i++)
	//{
  	//	bool exists = std::find(std::begin(check_cols), std::end(check_cols), i) != std::end(check_cols);
	//	if (!exists)
	//	{
			//std::cout << "Attention: Column " << i << " is empty (loading it is not efficient) " << std::endl;
			//exit(0);
	//	}
	//}
				

        // values
    	std::getline(inFile, line);
	// Create a stringstream of the current line
	ss.str("");
        ss.clear();
	ss<<line;

	for (i = 0; i <  nnz_value; i++) {
		float float_val;
		ss >> float_val;

        #if(INT_QUANT==1)
		   array_values[i] = (INTYPE)(float_val);
        #else

		  FTYPE quant_val;

		  quant_fea(quant_val,float_val,quantization_scale_fea,f_align,beta_qu);

		  array_values[i] = (FTYPE)(quant_val);
        #endif
		//array_values[i] = "0b0.0010000";
		//array_values[i] = "0b1.0001000";
		//std::cout << " float value " << float_val << " feature value " << array_values[i] << std::endl;
		//if (abs(float_val-float(array_values[i])) > 0.1)
		//if (abs(float_val-float(array_values[i])) > 0.5)
		//{
		//	std::cout << " feature error "  << std::endl;
		//	std::cout << " float value " << float_val << " feature value " << array_values[i] << std::endl;
		//	exit(1);
		//}
        // If the next token is a comma, ignore it and move on
	    if(ss.peek() == ',') ss.ignore();
	}


	inFile.close();
	std::cout << "Number of non-zeros fea values in CSR file: " << nnz_value << std::endl;
	std::cout << "fea matrix size: " << M*N << std::endl;
	std::cout << "Total percentage of zero values in fea: " << (float)(M*N-nnz_value)/(float)(M*N) << std::endl;
	//std::cout << "Total percentage of zero values per row in adj: " << std::endl;
        //for(int i=0;i<N;i++)
	//	std::cout << 100*(1-(float)NNZR[i]/(float)M) << ",";

        //std::cout << "array values and col indices size each uses " << A.size()   << " integers" << std::endl;
        //std::cout << "row ptr uses " << IA.size()   << " integers" << std::endl;
	//exit(0);


}


void loadcoo_fea(
std::string file_name,
int   N,
int   M,
INTYPE *array_values,
int   *array_colIndices,
int   *array_rowPtr,
int   nnz_value)
{
	int i;
	// Helper vars
	std::string line;



	// Create an outuptu filestream
	//std::string file_name = "./weights_layer_" + std::to_string(layer_number) + ".csr";
	std::ifstream inFile(file_name);

	// Make sure the file is open
	if(!inFile.is_open())
		throw std::runtime_error("Could not open csr file");
	else
		std::cout << "reading " << file_name << " file" << std::endl;

    	// Read data, line by line
        // rowptr
    	std::getline(inFile, line);
	// Create a stringstream of the current line
	std::stringstream ss;

	ss << line;

	for (i = 0; i < nnz_value; i++) {
		int temp;
		ss >> temp;
	        //std::cout << "row pointer " << array_rowPtr[i] << std::endl;

		//if (temp > nnz_value)
		//{

		//	array_rowPtr[i] = array_rowPtr[i-1];
		//}
		//else
		//{
			array_rowPtr[i] = temp;
			//std::cout << "Accumulated non-zeros at row " << i << " " << array_rowPtr[i] << std::endl;
		//}

        	// If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();
	}



        // column_index
    	std::getline(inFile, line);
        //int check_cols[100 * max_N_adj];

	// Create a stringstream of the current line

	ss.str("");
        ss.clear();
	ss <<  line;

        //std::cout << "ss: " << ss.str() << std::endl;

        //std::cout << "nnz_value: " << nnz_value << std::endl;

	for (i = 0; i <  nnz_value; i++) {
		ss >> array_colIndices[i];
		//check_cols[i] = array_colIndices[i];
		//std::cout << "array colindx " << array_colIndices[i] << std::endl;

        	// If the next token is a comma, ignore it and move on
	        if(ss.peek() == ',') ss.ignore();
	}

	//check if column is empty


    //    for(i = 0; i< M;i++)
	//{
  	//	bool exists = std::find(std::begin(check_cols), std::end(check_cols), i) != std::end(check_cols);
	//	if (!exists)
	//	{
			//std::cout << "Attention: Column " << i << " is empty (loading it is not efficient) " << std::endl;
			//exit(0);
	//	}
	//}


        // values
    	std::getline(inFile, line);
	// Create a stringstream of the current line
	ss.str("");
        ss.clear();
	ss<<line;

	for (i = 0; i <  nnz_value; i++) {
		float float_val;
		ss >> float_val;

        #if(INT_QUANT==1)
		   array_values[i] = (INTYPE)(float_val);
        #else

		  FTYPE quant_val;

		  quant_fea(quant_val,float_val,quantization_scale_fea,f_align,beta_qu);

		  array_values[i] = (FTYPE)(quant_val);
        #endif
		//array_values[i] = "0b0.0010000";
		//array_values[i] = "0b1.0001000";
		//std::cout << " float value " << float_val << " feature value " << array_values[i] << std::endl;
		//if (abs(float_val-float(array_values[i])) > 0.1)
		//if (abs(float_val-float(array_values[i])) > 0.5)
		//{
		//	std::cout << " feature error "  << std::endl;
		//	std::cout << " float value " << float_val << " feature value " << array_values[i] << std::endl;
		//	exit(1);
		//}
        // If the next token is a comma, ignore it and move on
	    if(ss.peek() == ',') ss.ignore();
	}


	inFile.close();
	std::cout << "Number of non-zeros fea values in COO file: " << nnz_value << std::endl;
	std::cout << "fea matrix size: " << M*N << std::endl;
	std::cout << "Total percentage of zero values in fea: " << (float)(M*N-nnz_value)/(float)(M*N) << std::endl;
	//std::cout << "Total percentage of zero values per row in adj: " << std::endl;
        //for(int i=0;i<N;i++)
	//	std::cout << 100*(1-(float)NNZR[i]/(float)M) << ",";

        //std::cout << "array values and col indices size each uses " << A.size()   << " integers" << std::endl;
        //std::cout << "row ptr uses " << IA.size()   << " integers" << std::endl;
	//exit(0);


}



// Generate the three vectors A, IA, JA

void arraytocsr_fea(
INTYPE *V,
int N,
int M,
INTYPE *array_values,
int   *array_colIndices,
int   *array_rowPtr,
int   *nnz_value)
{
	int i, j;
	vi IA = { 0 }; // IA matrix has N+1 rows
	vi JA;
	int NNZ = 0;
	int NNZR[2500] = {0}; //number of non-zeros per row


	// Create an outuptu filestream
	//std::string file_name = "./weights_layer_" + std::to_string(layer_number) + ".csr";
	//std::ofstream outFile(file_name);

	// Make sure the file is open
	//if(!outFile.is_open()) 
	//	throw std::runtime_error("Could not open csr file");
	//else
	//	std::cout << "writting " << file_name << " file" << std::endl;

        std::cout << "arraytocsr size: " << N << " "<< M  << std::endl;


	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			//std::cout << " input " << i <<" "<< j << " " << V[i*M+j] << std::endl;
			if (V[i*M+j] != 0) {
				//printf("Non-zero is %f\n", V[i*M+j]);
				//std::cout << "Non zero adj is " << V[i*M+j] << std::endl;
				//A.push_back(V[i*M+j]);
				array_values[i*M+j] = V[i*M+j];
				JA.push_back(j);
				NNZ++;
				// Count Number of Non Zero
				// Elements in row i
				NNZR[i]++; 
			}
		}
		IA.push_back(NNZ);
	}


	//outFile << N << " " << M << " " << NNZ << std::endl;
        *nnz_value = NNZ;

	for(int i=0;i<JA.size();i++)
	{
		//outFile << JA[i] << " " << A[i] << std::endl;
		//array_values[i] = A[i];
		array_colIndices[i] = JA[i];
		//std::cout << "array values " << array_values[i] << " " << "array colindices " << array_colIndices[i] << std::endl;
	}

	for(int i=0;i<IA.size();i++)
	{
		//outFile << IA[i] << std::endl;
		array_rowPtr[i] =  IA[i];
		//std::cout << "row pointer " << IA[i] << std::endl;
	}

	//outFile.close();
	std::cout << "Number of non-zero fea values in CSR file: " << NNZ << std::endl;
	std::cout << "Total Number of fea values in CSR file: " << N*M << std::endl;
	std::cout << "Total percentage of zero values: " << (float)(N*M-NNZ)/(float)(N*M) << std::endl;
	//std::cout << "Total percentage of zero values per row: " << std::endl;
        //for(int i=0;i<N;i++)
	//	std::cout << 100*(1-(float)NNZR[i]/(float)M) << ",";

        std::cout << "array values and col indices size each uses " << JA.size()   << " integers" << std::endl;
        std::cout << "row ptr uses " << IA.size()   << " integers" << std::endl;


}


void arraytocsr_adj(
INTYPE *V,
int N,
int M,
INTYPE *array_values,
int   *array_colIndices,
int   *array_rowPtr,
int   *nnz_value)
{
	int i, j;
	vi IA = { 0 }; // IA matrix has N+1 rows
	vi JA;
	int NNZ = 0;
	int NNZR[2500] = {0}; //number of non-zeros per row


	// Create an outuptu filestream
	//std::string file_name = "./weights_layer_" + std::to_string(layer_number) + ".csr";
	//std::ofstream outFile(file_name);

	// Make sure the file is open
	//if(!outFile.is_open()) 
	//	throw std::runtime_error("Could not open csr file");
	//else
	//	std::cout << "writting " << file_name << " file" << std::endl;

        std::cout << "arraytocsr size: " << N << " "<< M  << std::endl;


	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			//std::cout << " input " << i <<" "<< j << " " << V[i*M+j] << std::endl;
			if (V[i*M+j] != 0) {
				//printf("Non-zero is %f\n", V[i*M+j]);
				//std::cout << "Non zero adj is " << V[i*M+j] << std::endl;
				//A.push_back(V[i*M+j]);
				array_values[i*M+j] = V[i*M+j];
				JA.push_back(j);
				NNZ++;
				// Count Number of Non Zero
				// Elements in row i
				NNZR[i]++; 
			}
		}
		IA.push_back(NNZ);
	}


	//outFile << N << " " << M << " " << NNZ << std::endl;
        *nnz_value = NNZ;

	for(int i=0;i<JA.size();i++)
	{
		//outFile << JA[i] << " " << A[i] << std::endl;
		//array_values[i] = A[i];
		array_colIndices[i] = JA[i];
		//std::cout << "array values " << array_values[i] << " " << "array colindices " << array_colIndices[i] << std::endl;
	}

	for(int i=0;i<IA.size();i++)
	{
		//outFile << IA[i] << std::endl;
		array_rowPtr[i] =  IA[i];
		//std::cout << "row pointer " << IA[i] << std::endl;
	}

	//outFile.close();
	std::cout << "Number of non-zero adj values in CSR file: " << NNZ << std::endl;
	std::cout << "Total Number of adj values in CSR file: " << N*M << std::endl;
	std::cout << "Total percentage of zero values: " << (float)(N*M-NNZ)/(float)(N*M) << std::endl;
	//std::cout << "Total percentage of zero values per row: " << std::endl;
        //for(int i=0;i<N;i++)
	//	std::cout << 100*(1-(float)NNZR[i]/(float)M) << ",";

        std::cout << "array values and col indices size each uses " << JA.size()   << " integers" << std::endl;
        std::cout << "row ptr uses " << IA.size()   << " integers" << std::endl;


}

int gnn_test(int layer_count,ITYPE max_fea,int quantized_multiplier,ap_int<32> *shift,ap_int<32> *bias,ap_int<32> bias_count,ap_int<64> *profiling,ap_int<8> zero_point_lhs,
ap_int<8> zero_point_rhs,ap_int<8> zero_point_dst,ap_int<8> clamp_max,ap_int<8> clamp_min,INTYPES *w_m,float *D_sw,
OUTTYPE *D1,OUTTYPE *D2,OUTTYPE *D3,OUTTYPE *D4,
OUTTYPE *E1,
OUTTYPE *S1,
INTYPE *ate_m,INTYPE *values_fea,int *colIndices_fea,int *rowPtr_fea,int nnz_fea,INTYPE *values_adj,int *colIndices_adj,int *rowPtr_adj,int nnz_adj,INTYPE *adj_m,  INTYPE *fea_m,
int N_adj,int M_adj,int M_fea,int P_w,std::string adj_file,std::string fea_file,std::string w_file)
{
     //std::cout << "Testing " << std::endl;

	bool test_result;

     for (int i = 0; i < 1; i++) 
     {

     std::cout << "Loading sparse arrays" << std::endl;

	
    bool coo_mode = COO_MODE;



    if (gemm_mode == 0)
    {
    	if(coo_mode == 1)
        	loadcoo_fea(fea_name,M_adj,M_fea,values_fea,colIndices_fea,rowPtr_fea,NNZ_fea);
    	else
    	    loadcsr_fea(fea_name,M_adj,M_fea,values_fea,colIndices_fea,rowPtr_fea,NNZ_fea);
    }
   	else if (gemm_mode == 2)
    	loadcsr_fea(fea_name,M_adj,M_fea,values_fea,colIndices_fea,rowPtr_fea,NNZ_fea);
    else
    	load_fea(N_adj,M_fea,values_fea,fea_name);


    if (gemm_mode == 2)
    	load_adj(N_adj,M_adj,values_adj,adj_name);
    else
    {
    	if(coo_mode == 1)
    		loadcoo_adj(adj_name,N_adj,M_adj,values_adj,colIndices_adj,rowPtr_adj,NNZ_adj);
    	else
            loadcsr_adj(adj_name,N_adj,M_adj,values_adj,colIndices_adj,rowPtr_adj,NNZ_adj);
    }

    load_weights(M_fea,P_w,w_m,w_name);


    //load expected results
    std::cout << "result size" << N_adj << " " << P_w << std::endl;

    load_result_lines(N_adj,P_w,D_sw,d_name);

    load_attention(1,2*P_w,ate_m,ate_name);

    std::cout << "done loading" << std::endl;



	double start_time, end_time, execution_time;

    //======================ONLY CPU ==========================================


    std::cout << "Running GNN accelerator" << std::endl;
    start_time = getTimestamp();
    
    hls::stream<ASTYPE> DS1("out stream");
    #pragma HLS STREAM variable=DS1 depth=640000

    hls::stream<ASTYPE> DS1R("out stream rows");
    #pragma HLS STREAM variable=DS1R depth=640000

    hls::stream<ASTYPE> DS1C("out stream columns");
    #pragma HLS STREAM variable=DS1C depth=640000

    hls::stream<ASTYPE> DS2;
    //#pragma HLS STREAM variable=DS2 depth=4096

    hls::stream<ASTYPE> DS3;
    //#pragma HLS STREAM variable=DS3 depth=4096

    hls::stream<ASTYPE> DS4;
    //#pragma HLS STREAM variable=DS4 depth=4096

    hls::stream<ASTYPE>  values_feas1("in stream");
    #pragma HLS STREAM variable=values_feas1 depth=640000

	hls::stream<ASTYPE> values_feas2,values_feas3,values_feas4;
    hls::stream<ASTYPE> values_feas12, values_feas22,  values_feas32,values_feas42;

    hls::stream<ASTYPE> rowPtr_feas1("in stream rows");
    #pragma HLS STREAM variable=rowPtr_feas1 depth=640000

	hls::stream<ASTYPE> rowPtr_feas2,rowPtr_feas3,rowPtr_feas4;

    hls::stream<ASTYPE> colIndices_feas1("in stream columns");;
    #pragma HLS STREAM variable=colIndices_feas1 depth=640000

	hls::stream<ASTYPE> colIndices_feas2,colIndices_feas3,colIndices_feas4;


    //0 gemm in none
    //1 fea is dense
    //2 adj is dense
    //3 both are dense





    bool load_weights = 1;



    ASTYPE DS1_stream;

	fp_int C_float_int;

    if(stream_mode1==1)
    {
     bool last=0;
     for(int s_loop=0;s_loop<NNZ_fea;s_loop++)
     {
	    if(s_loop==(NNZ_fea-1))
	      last = 1;
	    C_float_int.f=values_fea[s_loop];
    	DS1_stream.data=C_float_int.i;
        DS1_stream.last = last;
        values_feas1.write(DS1_stream);
    	DS1_stream.data=colIndices_fea[s_loop];
        DS1_stream.last = last;
        colIndices_feas1.write(DS1_stream);
        //printf("rowPtr_fea %d\n",rowPtr_fea[s_loop]);
    	DS1_stream.data=rowPtr_fea[s_loop];
        DS1_stream.last = last;
        rowPtr_feas1.write(DS1_stream);
     }
    }
     /*bool last=0;
     for(int s_loop_i=0;s_loop_i<N_adj;s_loop_i++)
     {
    		   for(int s_loop_j=0;s_loop_j<B_WIDTH_BLOCK;s_loop_j++)
    	       {

    		    if(s_loop_j*s_loop_i==(N_adj-1)*(B_WIDTH_BLOCK-1))
    		      last = 1;
    	       //std::cout << "loading stream buffer " << s_loop_i << " " << s_loop_j << std::endl;
    		   C_float_int.f=1.0;
    		   DS1_stream.data=C_float_int.i;
    		   DS1_stream.last = last;
    		   values_feas1.write(DS1_stream);
    	       DS1_stream.data = s_loop_i;
    	       DS1_stream.last = last;

    	       rowPtr_feas1.write(DS1_stream);
    	       DS1_stream.data = s_loop_j;
    	       DS1_stream.last = last;

    	       colIndices_feas1.write(DS1_stream);

    	     }
     }


   //}






   //DS1_stream.last = 1;
   //rowPtr_feas1.write(DS1_stream);

    /*
      DS1_stream.data = 0x2; DS1_stream.last = 0;
      //DS1_stream.keep = 1;
      //DS1_stream.strb = 1;
      //DS1_stream.user = 1;
      DS1.write(DS1_stream);
      DS1_stream.data = 0x3; DS1_stream.last = 0;
      //DS1_stream.keep = 1;
      //DS1_stream.strb = 1;
      //DS1_stream.user = 1;
      DS1.write(DS1_stream);
      DS1_stream.data = 0x4; DS1_stream.last = 0;
      //DS1_stream.keep = 1;
      //DS1_stream.strb = 1;
      //DS1_stream.user = 1;
      DS1.write(DS1_stream);
      DS1_stream.data = 0x5; DS1_stream.last = 1;
      //DS1_stream.keep = 1;
      //DS1_stream.strb = 1;
      //DS1_stream.user = 1;
      DS1.write(DS1_stream);*/


    ap_uint<8> model[10] = {0};

    //7	             6	       5	     4	         3	              2	           1	         0
    //sage_mode	linear_mode	gat_mode	relu	stream_mode1	stream_mode0	gemm_mode1	gemm_mode0


    //test gcnconv layer

    model[0][0]= 0;
    model[0][1]= gemm_mode;
    model[0][2]= 0;
    model[0][3]= stream_mode1;
    model[0][4]= relu;
    model[0][5]= gat_mode;
    model[0][6]= linear_mode;
    model[0][7]=0;




    std::cout << "start with gemm mode " <<  model[0][1] << std::endl;
    std::cout << "start with linear mode " <<   model[0][6] << std::endl;


    float quantization_scale_fea[5];
    float quantization_scale_w[5];
    float quantization_scale_lin[5];
    float deq_factor[5];
    STYPE scale_fea[5];

    quantization_scale_fea[0]=quantization_scale_fea_0;
    quantization_scale_w[0]=quantization_scale_w_0;
    quantization_scale_lin[0]=quantization_scale_lin_0;
    deq_factor[0]=deq_factor_0;
    scale_fea[0]=scale_fea_0;


    kernelmult1(load_weights,beta_qu,f_align,
    quantization_scale_adj,quantization_scale_fea,quantization_scale_w,quantization_scale_lin,deq_factor,
    layer_count,
	model,
	scale_fea,&max_fea,
    quantized_multiplier,shift,bias,bias_count,profiling,zero_point_lhs,zero_point_rhs,zero_point_dst,clamp_max,clamp_min,
    w_m,w_m,
	D1,D2,D3,D4,
	DS1,DS1R,DS1C,
	DS2,DS3,DS4,
	E1,
	S1,
	ate_m,
	values_fea,values_fea,values_fea,values_fea,
	values_feas1,values_feas2,values_feas3,values_feas4,
	colIndices_fea,colIndices_fea,colIndices_fea,colIndices_fea,
	colIndices_feas1,colIndices_feas2,colIndices_feas3,colIndices_feas4,
    NNZ_fea, NNZ_fea, NNZ_fea, NNZ_fea,
	rowPtr_fea,rowPtr_fea,rowPtr_fea,rowPtr_fea,
	rowPtr_feas1,rowPtr_feas2,rowPtr_feas3,rowPtr_feas4,
	values_adj,values_adj,values_adj,values_adj,
	colIndices_adj,colIndices_adj,colIndices_adj,colIndices_adj,
	NNZ_adj,NNZ_adj,NNZ_adj,NNZ_adj,
	rowPtr_adj,rowPtr_adj,rowPtr_adj,rowPtr_adj,
	N_adj,N_adj,M_fea,P_w);
/*
    ap_uint<2> gemm_mode2 = 1; // features are dense in layer 2
    ap_int<2> stream_mode2 = 2; // read stream as input and write to memory in layer 2


    ASTYPE out_stream;
    for(int s_loop=0;s_loop<B_WIDTH_BLOCK*N_adj*(layer_count-1);s_loop++)
    {


         //DS1_stream.keep = 1;
         //DS1_stream.strb = 1;
         //DS1_stream.user = 1;
         out_stream = values_feas1.read();
         std::cout << "out data " <<  out_stream.data << std::endl;


       }
*/
    //std::cout << "start l2 with gemm mode " <<  gemm_mode2 << std::endl;

/*       ASTYPE DS1_stream;

    DS1_stream.data = 0x1; DS1_stream.last = 0;
    //DS1_stream.keep = 1;
    //DS1_stream.strb = 1;
    //DS1_stream.user = 1;
    DS1.write(DS1_stream);
    DS1_stream.data = 0x2; DS1_stream.last = 0;
    //DS1_stream.keep = 1;
    //DS1_stream.strb = 1;
    //DS1_stream.user = 1;
    DS1.write(DS1_stream);
    DS1_stream.data = 0x3; DS1_stream.last = 0;
    //DS1_stream.keep = 1;
    //DS1_stream.strb = 1;
    //DS1_stream.user = 1;
    DS1.write(DS1_stream);
    DS1_stream.data = 0x4; DS1_stream.last = 0;
    //DS1_stream.keep = 1;
    //DS1_stream.strb = 1;
    //DS1_stream.user = 1;
    DS1.write(DS1_stream);
    DS1_stream.data = 0x5; DS1_stream.last = 1;
    //DS1_stream.keep = 1;
    //DS1_stream.strb = 1;
    //DS1_stream.user = 1;
    DS1.write(DS1_stream);
*/

/*

    kernelmult1(load_weights,beta_qu,f_align,quantization_scale_adj,quantization_scale_fea,quantization_scale_w,deq_factor,
    layer_count,stream_mode2,gat_mode,gemm_mode2,relu,scale_fea,&max_fea,
    quantized_multiplier,shift,bias,bias_count,profiling,zero_point_lhs,zero_point_rhs,zero_point_dst,clamp_max,clamp_min,
    w_m,
   	D1,D2,D3,D4,
	values_feas12,values_feas22,values_feas32,values_feas42,
   	E1,
   	S1,
   	ate_m,
   	values_fea,values_fea,values_fea,values_fea,
	DS1,DS2,DS3,DS4,
   	colIndices_fea,colIndices_fea,colIndices_fea,colIndices_fea,
    NNZ_fea, NNZ_fea, NNZ_fea, NNZ_fea,
   	rowPtr_fea,rowPtr_fea,rowPtr_fea,rowPtr_fea,
   	values_adj,values_adj,values_adj,values_adj,
   	colIndices_adj,colIndices_adj,colIndices_adj,colIndices_adj,
   	NNZ_adj,NNZ_adj,NNZ_adj,NNZ_adj,
   	rowPtr_adj,rowPtr_adj,rowPtr_adj,rowPtr_adj,
   	N_adj,N_adj,B_WIDTH_BLOCK,P_w); //2 layer the input features are B_WIDTH_BLOCK

*/

   
    end_time = getTimestamp();

  	execution_time = (end_time - start_time) / (1000);


	std::cout << "MAX FEA " <<  max_fea << std::endl;


	std::cout << "CPU " << " Total execution time = " << execution_time << " msec" << std::endl;

	std::cout << "Checking Results" << std::endl;
	test_result = result_check(scale_fea,N_adj,P_w,D1, D_sw);
	  
     }

     return test_result;
}

/**
 * Design principles to achieve performance
 *
 * 1. sds_alloc to guarantee physically contiguous buffer allocation
 *    that enables the most efficient DMA configuration (axidma_simple)
 */
int main(int argc, char* argv[]){
	 int test_passed = 0;
     INTYPE *adj_m;
     INTYPE *fea_m;
     INTYPES *w_m;

     ap_int<8> zero_point_lhs,zero_point_rhs,zero_point_dst,clamp_max,clamp_min;

     float *D_sw;
	 OUTTYPE *D; //, *D1,*D2,*D3,*D4;
     OUTTYPE *E, *S;
     INTYPE *A_param;
     INTYPE *values_fea;
     int *colIndices_fea,*rowPtr_fea;
     INTYPE *values_adj;
     int *colIndices_adj,*rowPtr_adj;
     ap_int<32> *shift,*bias;

     ap_int<64> *profiling;
     int bias_count,nnz_fea,nnz_adj;
     ITYPE max_fea = 0;
     //STYPE scale_fea = 3;
     //int quantized_multiplier = 16;


     int quantized_multiplier = 16;


     #ifdef input_arguments

     if(argc<6)
     {
      	std::cout << "Error not enough arguments " << std::endl;
     	exit(1);
     }


     //std::ifstream adj_file(argv[1]);
     //std::ifstream fea_file(argv[2]);

     std::string adj_file(argv[1]);
     std::string fea_file(argv[2]);

     //SN,SM,SP
     N_adj = atoi(argv[3]);
     M_fea = atoi(argv[4]); 
     P_w = atoi(argv[5]);
     #else

     std::string adj_file(adj_name);
     std::string fea_file(fea_name);
     std::string w_file(w_name);
     std::string ate_file(ate_name);

     #endif

     std::cout << "Matrix dimensions N_adj/M_adj " << N_adj << " M_fea " << M_fea << " P_w " << P_w << std::endl;

     std::cout << "Scale fea is " << scale_fea_0 << std::endl;


     bias_count = 0;

     int layer_count = 1;

     D = (OUTTYPE *)malloc(layer_count*max_N_adj*max_P_w*sizeof(OUTTYPE));
     E = (OUTTYPE *)malloc(layer_count*max_N_adj*max_P_w*sizeof(OUTTYPE));
     S = (OUTTYPE *)malloc(layer_count*max_N_adj*max_P_w*sizeof(OUTTYPE));
     A_param = (INTYPE *)malloc(max_P_w*2*sizeof(INTYPE));
     //D1 = D;
     //D2 = D+N_adj/4;
     //D3 = (DTYPE *)malloc(N_adj*P_w *sizeof(DTYPE));
     //D4 = (DTYPE *)malloc(N_adj*P_w *sizeof(DTYPE));


     shift = (ap_int<32> *)malloc(max_N_adj*sizeof(ap_int<32>));
     bias = (ap_int<32> *)malloc(max_N_adj*sizeof(ap_int<32>));
     profiling = (ap_int<64> *)malloc(max_N_adj*sizeof(ap_int<32>));



     //values_fea = (INTYPE *)(malloc(max_M_fea * max_N_adj * sizeof(INTYPE)));
     //colIndices_fea  = (int *)(malloc(max_M_fea * max_N_adj  * sizeof(int)));
     //rowPtr_fea  = (int *)malloc(max_M_fea * max_N_adj * sizeof(int));
     values_fea = (INTYPE *)(malloc((NNZ_fea+1)  * sizeof(INTYPE)));
     colIndices_fea  = (int *)(malloc((NNZ_fea+1)   * sizeof(int)));
     rowPtr_fea  = (int *)malloc((NNZ_fea+1) * sizeof(int));


     //values_adj  = (INTYPE *)(malloc(max_N_adj * max_N_adj  * sizeof(INTYPE)));
     //colIndices_adj  = (int *)(malloc(max_N_adj * max_N_adj  * sizeof(int)));
     //rowPtr_adj  = (int *)malloc(max_N_adj * max_N_adj * sizeof(int));

     values_adj  = (INTYPE *)(malloc((NNZ_adj+1)  * sizeof(INTYPE)));
     colIndices_adj  = (int *)(malloc((NNZ_adj+1)   * sizeof(int)));
     rowPtr_adj  = (int *)malloc((NNZ_adj+1)  * sizeof(int));
     
     w_m = (INTYPES *)malloc(layer_count*max_M_fea * max_P_w * sizeof(INTYPES));

     //result matrix
     D_sw = (float *)malloc(max_N_adj * max_P_w * sizeof(float));


     adj_m = (INTYPE *)malloc(max_N_adj * max_N_adj * sizeof(INTYPE));
     fea_m = (INTYPE *)malloc(max_N_adj * max_M_fea * sizeof(INTYPE));
     


     if (!values_adj || !colIndices_adj || !rowPtr_adj) {
          if (values_adj) free(values_adj);
          if (colIndices_adj) free(colIndices_adj);
          if (rowPtr_adj) free(rowPtr_adj);

	  std::cout << "Error allocating sparse adj memory " << std::endl;
          return 1;
     }

     if (!values_fea || !colIndices_fea || !rowPtr_fea) {
          if (values_fea) free(values_fea);
          if (colIndices_fea) free(colIndices_fea);
          if (rowPtr_fea) free(rowPtr_fea);

	  std::cout << "Error allocating sparse fea memory " << std::endl;
          return 1;
     }


     if (!adj_m || !fea_m || !D || !D_sw) {
          if (adj_m) free(adj_m);
          if (fea_m) free(fea_m);
          if (D) free(D);
          if (D_sw) free(D_sw);
	  std::cout << "Error allocating dense memory " << std::endl;
          return 1;
     }






     test_passed = gnn_test(layer_count,max_fea,quantized_multiplier,shift,bias,bias_count,profiling,zero_point_lhs,zero_point_rhs,zero_point_dst,clamp_max,
                  clamp_min,w_m,D_sw,
				  D,D,D,D,
				  E,
				  S,
				  A_param,values_fea,colIndices_fea,rowPtr_fea,NNZ_fea,values_adj,colIndices_adj,rowPtr_adj,NNZ_adj,adj_m, fea_m,
		   N_adj,N_adj,M_fea,P_w,adj_file,fea_file,w_file);
     
     std::cout << "TEST " << (test_passed ? "PASSED" : "FAILED") << std::endl;


     //std::cout << "MAX ADJ " << max_adj << std::endl;
     //std::cout << "MIN ADJ " << min_adj << std::endl;
     //std::cout << "MAX FEA " << max_fea << std::endl;
     //std::cout << "MIN FEA " << min_fea << std::endl;


     //free(adj_m);
     //free(fea_m);
     //free(D);

     //free(D_sw);
     //free(A_param);
     //free(shift);
    //free(bias);
     //free(profiling);

     //free(values_fea);
     //free(colIndices_fea);
     //free(rowPtr_fea);

     //free(values_adj);
     //free(colIndices_adj);
     //free(rowPtr_adj);

     //free(w_m);



     std::cout << "All memory has been released " << std::endl;


    // return (test_passed ? -1 : 0);

     
}

