#include "myblas_internal.h"
#include <stdlib.h>
//#include <stdio.h>

void myblas_dgemm_copy_t_detail(size_t K, size_t M, const double* A, size_t k, size_t i, size_t lda, double* A2 ){

	//size_t k = K;
	//while( k-- ){
	//  size_t m = M;
	//  while( m-- ){
	//    (*A2) = (*A);
	//    A++;
	//    A2+=K;
	//  }
	//  A2 = A2 - K*M + 1;
	//  A  = A  - M + lda;
	//}
	const double* a0 = A;
	const double* a1 = A + lda;
	size_t  lda2 = 2*lda*sizeof(double);
	size_t  lda3 = 3*lda*sizeof(double);
	size_t  KK = K*sizeof(double);

	double* A2_2 = A2   + K*( M & ~3 );
	double* A2_1 = A2_2 + K*( M &  2 );;

	//double x00,x01,x02,x03;
	//double x10,x11,x12,x13;
	//double x20,x21,x22,x23;
	//double x30,x31,x32,x33;

	if( K >> 3 ){
	  size_t k8 = ( K >> 3 );
	  while( k8-- ){
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      m4--;

	        __asm__ __volatile__ (
	          "prefetcht0  16*8(%[a0]          ) \n\t"
	          "prefetcht0  16*8(%[a1]          ) \n\t"
	          "prefetcht0  16*8(%[a0],%[lda2]  ) \n\t"
	          "prefetcht0  16*8(%[a1],%[lda2]  ) \n\t"
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t"
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t"
	          "vmovupd  0*8(%[a0],%[lda2]  ), %%ymm2 \n\t"
	          "vmovupd  0*8(%[a1],%[lda2]  ), %%ymm3 \n\t"
	          "prefetcht0  16*8(%[a0],%[lda2],2) \n\t"
	          "prefetcht0  16*8(%[a1],%[lda2],2) \n\t"
	          "prefetcht0  16*8(%[a0],%[lda3],2) \n\t"
	          "prefetcht0  16*8(%[a1],%[lda3],2) \n\t"
	          "vmovupd  0*8(%[a0],%[lda2],2), %%ymm4 \n\t"
	          "vmovupd  0*8(%[a1],%[lda2],2), %%ymm5 \n\t"
	          "vmovupd  0*8(%[a0],%[lda3],2), %%ymm6 \n\t"
	          "vmovupd  0*8(%[a1],%[lda3],2), %%ymm7 \n\t"
	          "\n\t"
	          "addq  $4*8, %[a0]\n\t"
	          "addq  $4*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K],4), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3),[K]"r"(KK));


	      while( m4-- ){

	        //for( size_t l=0; l<8; l++ ){
	        //  for( size_t i=0; i<4; i++ ){
	        //    (*A2) = *(A+i+l*lda);
	        //    A2++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        //x10 = *(A+1+0*lda); x11 = *(A+1+1*lda); x12 = *(A+1+2*lda); x13 = *(A+1+3*lda);
	        //x20 = *(A+2+0*lda); x21 = *(A+2+1*lda); x22 = *(A+2+2*lda); x23 = *(A+2+3*lda);
	        //x30 = *(A+3+0*lda); x31 = *(A+3+1*lda); x32 = *(A+3+2*lda); x33 = *(A+3+3*lda);
	        //*(A2+0+0*4) = x00; *(A2+0+1*4) = x01; *(A2+0+2*4) = x02; *(A2+0+3*4) = x03;
	        //*(A2+1+0*4) = x10; *(A2+1+1*4) = x11; *(A2+1+2*4) = x12; *(A2+1+3*4) = x13;
	        //*(A2+2+0*4) = x20; *(A2+2+1*4) = x21; *(A2+2+2*4) = x22; *(A2+2+3*4) = x23;
	        //*(A2+3+0*4) = x30; *(A2+3+1*4) = x31; *(A2+3+2*4) = x32; *(A2+3+3*4) = x33;
	        //x00 = *(A+0+4*lda); x01 = *(A+0+5*lda); x02 = *(A+0+6*lda); x03 = *(A+0+7*lda);
	        //x10 = *(A+1+4*lda); x11 = *(A+1+5*lda); x12 = *(A+1+6*lda); x13 = *(A+1+7*lda);
	        //x20 = *(A+2+4*lda); x21 = *(A+2+5*lda); x22 = *(A+2+6*lda); x23 = *(A+2+7*lda);
	        //x30 = *(A+3+4*lda); x31 = *(A+3+5*lda); x32 = *(A+3+6*lda); x33 = *(A+3+7*lda);
	        //*(A2+0+4*4) = x00; *(A2+0+5*4) = x01; *(A2+0+6*4) = x02; *(A2+0+7*4) = x03;
	        //*(A2+1+4*4) = x10; *(A2+1+5*4) = x11; *(A2+1+6*4) = x12; *(A2+1+7*4) = x13;
	        //*(A2+2+4*4) = x20; *(A2+2+5*4) = x21; *(A2+2+6*4) = x22; *(A2+2+7*4) = x23;
	        //*(A2+3+4*4) = x30; *(A2+3+5*4) = x31; *(A2+3+6*4) = x32; *(A2+3+7*4) = x33;

	        //A  = A  + 4;
	        ////A2 = A2 - 4*8 + 4*K;
	        //A2 = A2 + 4*K;

	        __asm__ __volatile__ (
	          "prefetcht0  20*8(%[a0]          ) \n\t"
	          "prefetcht0  20*8(%[a1]          ) \n\t"
	          "prefetcht0  20*8(%[a0],%[lda2]  ) \n\t"
	          "prefetcht0  20*8(%[a1],%[lda2]  ) \n\t"
	          "vmovapd  %%ymm0 ,  0*8(%[a2]) \n\t"
	          "vmovapd  %%ymm1 ,  4*8(%[a2]) \n\t"
	          "vmovupd  4*8(%[a0]          ), %%ymm0 \n\t"
	          "vmovupd  4*8(%[a1]          ), %%ymm1 \n\t"
	          "vmovapd  %%ymm2 ,  8*8(%[a2]) \n\t"
	          "vmovapd  %%ymm3 , 12*8(%[a2]) \n\t"
	          "vmovupd  4*8(%[a0],%[lda2]  ), %%ymm2 \n\t"
	          "vmovupd  4*8(%[a1],%[lda2]  ), %%ymm3 \n\t"
	          "prefetcht0  20*8(%[a0],%[lda2],2) \n\t"
	          "prefetcht0  20*8(%[a1],%[lda2],2) \n\t"
	          "prefetcht0  20*8(%[a0],%[lda3],2) \n\t"
	          "prefetcht0  20*8(%[a1],%[lda3],2) \n\t"
	          "vmovapd  %%ymm4 , 16*8(%[a2]) \n\t"
	          "vmovapd  %%ymm5 , 20*8(%[a2]) \n\t"
	          "vmovupd  4*8(%[a0],%[lda2],2), %%ymm4 \n\t"
	          "vmovupd  4*8(%[a1],%[lda2],2), %%ymm5 \n\t"
	          "vmovapd  %%ymm6 , 24*8(%[a2]) \n\t"
	          "vmovapd  %%ymm7 , 28*8(%[a2]) \n\t"
	          "vmovupd  4*8(%[a0],%[lda3],2), %%ymm6 \n\t"
	          "vmovupd  4*8(%[a1],%[lda3],2), %%ymm7 \n\t"
	          "\n\t"
	          "addq  $4*8, %[a0]\n\t"
	          "addq  $4*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K],4), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3),[K]"r"(KK));

	        //A  = A  + 4;

	      }

	        __asm__ __volatile__ (
	          "vmovapd  %%ymm0 ,  0*8(%[a2]) \n\t"
	          "vmovapd  %%ymm1 ,  4*8(%[a2]) \n\t"
	          "vmovapd  %%ymm2 ,  8*8(%[a2]) \n\t"
	          "vmovapd  %%ymm3 , 12*8(%[a2]) \n\t"
	          "vmovapd  %%ymm4 , 16*8(%[a2]) \n\t"
	          "vmovapd  %%ymm5 , 20*8(%[a2]) \n\t"
	          "vmovapd  %%ymm6 , 24*8(%[a2]) \n\t"
	          "vmovapd  %%ymm7 , 28*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8, %[a0]\n\t"
	          "addq  $4*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K],4), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3),[K]"r"(KK));


	      A2 = A2 - (M&~3)*K + 4*8; // move to next row block
	    }
	    if( M & 2 ){

	        //for( size_t l=0; l<8; l++ ){
	        //  for( size_t i=0; i<2; i++ ){
	        //    (*A2_2) = *(A+i+l*lda);
	        //    A2_2++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        //x10 = *(A+1+0*lda); x11 = *(A+1+1*lda); x12 = *(A+1+2*lda); x13 = *(A+1+3*lda);
	        //*(A2_2+0+0*2) = x00; *(A2_2+0+1*2) = x01; *(A2_2+0+2*2) = x02; *(A2_2+0+3*2) = x03;
	        //*(A2_2+1+0*2) = x10; *(A2_2+1+1*2) = x11; *(A2_2+1+2*2) = x12; *(A2_2+1+3*2) = x13;
	        //x00 = *(A+0+4*lda); x01 = *(A+0+5*lda); x02 = *(A+0+6*lda); x03 = *(A+0+7*lda);
	        //x10 = *(A+1+4*lda); x11 = *(A+1+5*lda); x12 = *(A+1+6*lda); x13 = *(A+1+7*lda);
	        //*(A2_2+0+4*2) = x00; *(A2_2+0+5*2) = x01; *(A2_2+0+6*2) = x02; *(A2_2+0+7*2) = x03;
	        //*(A2_2+1+4*2) = x10; *(A2_2+1+5*2) = x11; *(A2_2+1+6*2) = x12; *(A2_2+1+7*2) = x13;
	        //A  = A  + 2;
	        //A2_2 += 2*8;

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t"
	          "movupd  0*8(%[a1]          ), %%xmm1 \n\t"
	          "movupd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t"
	          "movupd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t"
	          "movapd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movapd  %%xmm1 ,  2*8(%[a2]) \n\t"
	          "movapd  %%xmm2 ,  4*8(%[a2]) \n\t"
	          "movapd  %%xmm3 ,  6*8(%[a2]) \n\t"
	          "movupd  0*8(%[a0],%[lda2],2), %%xmm4 \n\t"
	          "movupd  0*8(%[a1],%[lda2],2), %%xmm5 \n\t"
	          "movupd  0*8(%[a0],%[lda3],2), %%xmm6 \n\t"
	          "movupd  0*8(%[a1],%[lda3],2), %%xmm7 \n\t"
	          "movapd  %%xmm4 ,  8*8(%[a2]) \n\t"
	          "movapd  %%xmm5 , 10*8(%[a2]) \n\t"
	          "movapd  %%xmm6 , 12*8(%[a2]) \n\t"
	          "movapd  %%xmm7 , 14*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8, %[a0]\n\t"
	          "addq  $2*8, %[a1]\n\t"
	          "addq  $2*8*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	        //A  = A  + 2;

	    }
	    if( M & 1 ){

	        //for( size_t l=0; l<8; l++ ){
	        //  for( size_t i=0; i<1; i++ ){
	        //    (*A2_1) = *(A+i+l*lda);
	        //    A2_1++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        //*(A2_1+0+0*1) = x00; *(A2_1+0+1*1) = x01; *(A2_1+0+2*1) = x02; *(A2_1+0+3*1) = x03;
	        //x00 = *(A+0+4*lda); x01 = *(A+0+5*lda); x02 = *(A+0+6*lda); x03 = *(A+0+7*lda);
	        //*(A2_1+0+4*1) = x00; *(A2_1+0+5*1) = x01; *(A2_1+0+6*1) = x02; *(A2_1+0+7*1) = x03;
	        //A  = A  + 1;
	        //A2_1 += 1*8;

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t"
	          "movsd  0*8(%[a1]          ), %%xmm1 \n\t"
	          "movsd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t"
	          "movsd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movsd  %%xmm1 ,  1*8(%[a2]) \n\t"
	          "movsd  %%xmm2 ,  2*8(%[a2]) \n\t"
	          "movsd  %%xmm3 ,  3*8(%[a2]) \n\t"
	          "movsd  0*8(%[a0],%[lda2],2), %%xmm4 \n\t"
	          "movsd  0*8(%[a1],%[lda2],2), %%xmm5 \n\t"
	          "movsd  0*8(%[a0],%[lda3],2), %%xmm6 \n\t"
	          "movsd  0*8(%[a1],%[lda3],2), %%xmm7 \n\t"
	          "movsd  %%xmm4 ,  4*8(%[a2]) \n\t"
	          "movsd  %%xmm5 ,  5*8(%[a2]) \n\t"
	          "movsd  %%xmm6 ,  6*8(%[a2]) \n\t"
	          "movsd  %%xmm7 ,  7*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8, %[a0]\n\t"
	          "addq  $1*8, %[a1]\n\t"
	          "addq  $1*8*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :[lda2]"r"(lda2),[lda3]"r"(lda3));

	        //A  = A  + 1;

	    }
	    //A  = A - M + 8*lda;
	    a0 = a0- M + 8*lda;
	    a1 = a1- M + 8*lda;
	  }
	}
	if( K & 4  ){
	    if( M >> 2 ){
	      //printf("K4M4:A2-A0=%d\n",A2-A0);
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        //for( size_t l=0; l<4; l++ ){
	        //  for( size_t i=0; i<4; i++ ){
	        //    (*A2) = *(A+i+l*lda);
	        //    A2++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        //x10 = *(A+1+0*lda); x11 = *(A+1+1*lda); x12 = *(A+1+2*lda); x13 = *(A+1+3*lda);
	        //x20 = *(A+2+0*lda); x21 = *(A+2+1*lda); x22 = *(A+2+2*lda); x23 = *(A+2+3*lda);
	        //x30 = *(A+3+0*lda); x31 = *(A+3+1*lda); x32 = *(A+3+2*lda); x33 = *(A+3+3*lda);
	        //*(A2+0+0*4) = x00; *(A2+0+1*4) = x01; *(A2+0+2*4) = x02; *(A2+0+3*4) = x03;
	        //*(A2+1+0*4) = x10; *(A2+1+1*4) = x11; *(A2+1+2*4) = x12; *(A2+1+3*4) = x13;
	        //*(A2+2+0*4) = x20; *(A2+2+1*4) = x21; *(A2+2+2*4) = x22; *(A2+2+3*4) = x23;
	        //*(A2+3+0*4) = x30; *(A2+3+1*4) = x31; *(A2+3+2*4) = x32; *(A2+3+3*4) = x33;
	        //A  = A  + 4;
	        ////A2 = A2 - 4*4 + 4*K;
	        //A2 = A2 + 4*K;

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t"
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t"
	          "vmovupd  0*8(%[a0],%[lda2]  ), %%ymm2 \n\t"
	          "vmovupd  0*8(%[a1],%[lda2]  ), %%ymm3 \n\t"
	          "vmovapd  %%ymm0 ,  0*8(%[a2]) \n\t"
	          "vmovapd  %%ymm1 ,  4*8(%[a2]) \n\t"
	          "vmovapd  %%ymm2 ,  8*8(%[a2]) \n\t"
	          "vmovapd  %%ymm3 , 12*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8, %[a0]\n\t"
	          "addq  $4*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K],4), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[K]"r"(KK));

	        //A  = A  + 4;

	      }
	      A2 = A2 - (M&~3)*K + 4*4;
	    }
	    if( M & 2 ){

	        //printf("K4M2:A2_2-A0=%d\n",A2_2-A0);
	        //for( size_t l=0; l<4; l++ ){
	        //  for( size_t i=0; i<2; i++ ){
	        //    (*A2_2) = *(A+i+l*lda);
	        //    A2_2++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        //x10 = *(A+1+0*lda); x11 = *(A+1+1*lda); x12 = *(A+1+2*lda); x13 = *(A+1+3*lda);
	        //*(A2_2+0+0*2) = x00; *(A2_2+0+1*2) = x01; *(A2_2+0+2*2) = x02; *(A2_2+0+3*2) = x03;
	        //*(A2_2+1+0*2) = x10; *(A2_2+1+1*2) = x11; *(A2_2+1+2*2) = x12; *(A2_2+1+3*2) = x13;
	        //A  = A  + 2;
	        //A2_2 += 2*4;

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t"
	          "movupd  0*8(%[a1]          ), %%xmm1 \n\t"
	          "movupd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t"
	          "movupd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t"
	          "movapd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movapd  %%xmm1 ,  2*8(%[a2]) \n\t"
	          "movapd  %%xmm2 ,  4*8(%[a2]) \n\t"
	          "movapd  %%xmm3 ,  6*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8, %[a0]\n\t"
	          "addq  $2*8, %[a1]\n\t"
	          "addq  $2*4*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2));

	        //A  = A  + 2;


	    }
	    if( M & 1 ){

	        //printf("K4M1:A2_1-A0=%d\n",A2_1-A0);
	        //for( size_t l=0; l<4; l++ ){
	        //  for( size_t i=0; i<1; i++ ){
	        //    (*A2_1) = *(A+i+l*lda);
	        //    A2_1++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda); x01 = *(A+0+1*lda); x02 = *(A+0+2*lda); x03 = *(A+0+3*lda);
	        //*(A2_1+0+0*1) = x00; *(A2_1+0+1*1) = x01; *(A2_1+0+2*1) = x02; *(A2_1+0+3*1) = x03;
	        //A  = A  + 1;
	        //A2_1 += 1*4;

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t"
	          "movsd  0*8(%[a1]          ), %%xmm1 \n\t"
	          "movsd  0*8(%[a0],%[lda2]  ), %%xmm2 \n\t"
	          "movsd  0*8(%[a1],%[lda2]  ), %%xmm3 \n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movsd  %%xmm1 ,  1*8(%[a2]) \n\t"
	          "movsd  %%xmm2 ,  2*8(%[a2]) \n\t"
	          "movsd  %%xmm3 ,  3*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8, %[a0]\n\t"
	          "addq  $1*8, %[a1]\n\t"
	          "addq  $1*4*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :[lda2]"r"(lda2));

	        //A  = A  + 1;


	    }
	    //A  = A - M + 4*lda;
	    a0 = a0- M + 4*lda;
	    a1 = a1- M + 4*lda;
	}
	if( K & 2  ){
	    if( M >> 2 ){
	      //printf("K2M4:A2-A0=%d\n",A2-A0);
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        //for( size_t l=0; l<2; l++ ){
	        //  for( size_t i=0; i<4; i++ ){
	        //    (*A2) = *(A+i+l*lda);
	        //    A2++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda); x01 = *(A+0+1*lda);
	        //x10 = *(A+1+0*lda); x11 = *(A+1+1*lda);
	        //x20 = *(A+2+0*lda); x21 = *(A+2+1*lda);
	        //x30 = *(A+3+0*lda); x31 = *(A+3+1*lda);
	        //*(A2+0+0*4) = x00; *(A2+0+1*4) = x01;
	        //*(A2+1+0*4) = x10; *(A2+1+1*4) = x11;
	        //*(A2+2+0*4) = x20; *(A2+2+1*4) = x21;
	        //*(A2+3+0*4) = x30; *(A2+3+1*4) = x31;
	        //A  = A  + 4;
	        ////A2 = A2 - 4*2 + 4*K;
	        //A2 = A2 + 4*K;

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t"
	          "vmovupd  0*8(%[a1]          ), %%ymm1 \n\t"
	          "vmovapd  %%ymm0 ,  0*8(%[a2]) \n\t"
	          "vmovapd  %%ymm1 ,  4*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8, %[a0]\n\t"
	          "addq  $4*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K],4), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[K]"r"(KK));

	        //A  = A  + 4;


	      }
	      A2 = A2 - (M&~3)*K + 4*2;

	    }
	    if( M & 2 ){

	        //printf("K2M2:A2_2-A0=%d\n",A2_2-A0);
	        //for( size_t l=0; l<2; l++ ){
	        //  for( size_t i=0; i<2; i++ ){
	        //    (*A2_2) = *(A+i+l*lda);
	        //    A2_2++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda); x01 = *(A+0+1*lda);
	        //x10 = *(A+1+0*lda); x11 = *(A+1+1*lda);
	        //*(A2_2+0+0*2) = x00; *(A2_2+0+1*2) = x01;
	        //*(A2_2+1+0*2) = x10; *(A2_2+1+1*2) = x11;
	        //A  = A  + 2;
	        //A2_2 += 2*2;

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t"
	          "movupd  0*8(%[a1]          ), %%xmm1 \n\t"
	          "movapd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movapd  %%xmm1 ,  2*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8, %[a0]\n\t"
	          "addq  $2*8, %[a1]\n\t"
	          "addq  $2*2*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2));

	        //A  = A  + 2;


	    }
	    if( M & 1 ){

	        //printf("K2M1:A2_1-A0=%d\n",A2_1-A0);
	        //for( size_t l=0; l<2; l++ ){
	        //  for( size_t i=0; i<1; i++ ){
	        //    (*A2_1) = *(A+i+l*lda);
	        //    A2_1++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda); x01 = *(A+0+1*lda);
	        //*(A2_1+0+0*1) = x00; *(A2_1+0+1*1) = x01;
	        //A  = A  + 1;
	        //A2_1 += 1*2;

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t"
	          "movsd  0*8(%[a1]          ), %%xmm1 \n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "movsd  %%xmm1 ,  1*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8, %[a0]\n\t"
	          "addq  $1*8, %[a1]\n\t"
	          "addq  $1*2*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :);

	        //A  = A  + 1;


	    }
	    //A  = A - M + 2*lda;
	    a0 = a0- M + 2*lda;
	    a1 = a1- M + 2*lda;

	}
	if( K & 1  ){
	    if( M >> 2 ){
	      //printf("K1M4:A2-A0=%d\n",A2-A0);
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        //for( size_t l=0; l<1; l++ ){
	        //  for( size_t i=0; i<4; i++ ){
	        //    (*A2) = *(A+i+l*lda);
	        //    A2++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda);
	        //x10 = *(A+1+0*lda);
	        //x20 = *(A+2+0*lda);
	        //x30 = *(A+3+0*lda);
	        //*(A2+0+0*4) = x00;
	        //*(A2+1+0*4) = x10;
	        //*(A2+2+0*4) = x20;
	        //*(A2+3+0*4) = x30;
	        //A  = A  + 4;
	        ////A2 = A2 - 4*1 + 4*K;
	        //A2 = A2 + 4*K;

	        __asm__ __volatile__ (
	          "vmovupd  0*8(%[a0]          ), %%ymm0 \n\t"
	          "vmovapd  %%ymm0 ,  0*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $4*8, %[a0]\n\t"
	          "addq  $4*8, %[a1]\n\t"
	          "leaq  0*8(%[a2],%[K],4), %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2)
	        :[lda2]"r"(lda2),[K]"r"(KK));

	        //A  = A  + 4;


	      }
	      A2 = A2 - (M&~3)*K + 4*1;

	    }
	    if( M & 2 ){

	        //printf("K1M2:A2_2-A0=%d\n",A2_2-A0);
	        //for( size_t l=0; l<1; l++ ){
	        //  for( size_t i=0; i<2; i++ ){
	        //    (*A2_2) = *(A+i+l*lda);
	        //    A2_2++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda);
	        //x10 = *(A+1+0*lda);
	        //*(A2_2+0+0*2) = x00;
	        //*(A2_2+1+0*2) = x10;
	        //A  = A  + 2;
	        //A2_2 += 2*1;

	        __asm__ __volatile__ (
	          "movupd  0*8(%[a0]          ), %%xmm0 \n\t"
	          "movapd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $2*8, %[a0]\n\t"
	          "addq  $2*8, %[a1]\n\t"
	          "addq  $2*1*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_2)
	        :[lda2]"r"(lda2));

	        //A  = A  + 2;


	    }
	    if( M & 1 ){

	        //printf("K1M1:A2_1-A0=%d\n",A2_1-A0);
	        //for( size_t l=0; l<1; l++ ){
	        //  for( size_t i=0; i<1; i++ ){
	        //    (*A2_1) = *(A+i+l*lda);
	        //    A2_1++;
	        //  }
	        //}
	        //x00 = *(A+0+0*lda);
	        //*(A2_1+0+0*1) = x00;
	        //A  = A  + 1;
	        //A2_1 += 1*1;

	        __asm__ __volatile__ (
	          "movsd  0*8(%[a0]          ), %%xmm0 \n\t"
	          "movsd  %%xmm0 ,  0*8(%[a2]) \n\t"
	          "\n\t"
	          "addq  $1*8, %[a0]\n\t"
	          "addq  $1*8, %[a1]\n\t"
	          "addq  $112*8, %[a2]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(A2_1)
	        :);

	        //A  = A  + 1;


	    }
	    //A  = A - M + 1*lda;
	    a0 = a0- M + 1*lda;
	    a1 = a1- M + 1*lda;

	}

	
}

