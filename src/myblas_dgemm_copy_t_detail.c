#include "myblas_internal.h"

void myblas_dgemm_copy_t_detail(size_t K, size_t M, const double* A, size_t k, size_t i, size_t lda, double* A2 ){

	double x00,x10,x20,x30;
	double x01,x11,x21,x31;
	double x02,x12,x22,x32;
	double x03,x13,x23,x33;

	A = A + i + k*lda;

	const double* a0 = A;
	const double* a1 = A + lda;
	const double* a2 = A + lda*2;
	const double* a3 = A + lda*3;
	size_t        lda1 = lda * sizeof(double);
	size_t        lda2 = lda * sizeof(double) * 2;

	if( M >> 2 ){
	  size_t m4 = ( M >> 2 );
	  while( m4-- ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );

	      __asm__ __volatile__ (
	        "\n\t"
	        "prefetcht0  0(%[a0],%[lda1],8)\n\t"
	        "prefetcht0  0(%[a1],%[lda1],8)\n\t"
	        "prefetcht0  0(%[a2],%[lda1],8)\n\t"
	        "prefetcht0  0(%[a3],%[lda1],8)\n\t"
	        "\n\t"
	      :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(a2),[a3]"+r"(a3),[aa]"+r"(A2)
	      :[lda1]"r"(lda1),[lda2]"r"(lda2) );

	      while( k8-- ){

	        //x00 = *(A + 0 + 0*lda); x01 = *(A + 0 + 1*lda); x02 = *(A + 0 + 2*lda); x03 = *(A + 0 + 3*lda);
	        //x10 = *(A + 1 + 0*lda); x11 = *(A + 1 + 1*lda); x12 = *(A + 1 + 2*lda); x13 = *(A + 1 + 3*lda);
	        //x20 = *(A + 2 + 0*lda); x21 = *(A + 2 + 1*lda); x22 = *(A + 2 + 2*lda); x23 = *(A + 2 + 3*lda);
	        //x30 = *(A + 3 + 0*lda); x31 = *(A + 3 + 1*lda); x32 = *(A + 3 + 2*lda); x33 = *(A + 3 + 3*lda);
	        //*(A2 + 0*4 + 0) = x00; *(A2 + 1*4 + 0) = x01; *(A2 + 2*4 + 0) = x02; *(A2 + 3*4 + 0) = x03;
	        //*(A2 + 0*4 + 1) = x10; *(A2 + 1*4 + 1) = x11; *(A2 + 2*4 + 1) = x12; *(A2 + 3*4 + 1) = x13;
	        //*(A2 + 0*4 + 2) = x20; *(A2 + 1*4 + 2) = x21; *(A2 + 2*4 + 2) = x22; *(A2 + 3*4 + 2) = x23;
	        //*(A2 + 0*4 + 3) = x30; *(A2 + 1*4 + 3) = x31; *(A2 + 2*4 + 3) = x32; *(A2 + 3*4 + 3) = x33;
	        //x00 = *(A + 0 + 4*lda); x01 = *(A + 0 + 5*lda); x02 = *(A + 0 + 6*lda); x03 = *(A + 0 + 7*lda);
	        //x10 = *(A + 1 + 4*lda); x11 = *(A + 1 + 5*lda); x12 = *(A + 1 + 6*lda); x13 = *(A + 1 + 7*lda);
	        //x20 = *(A + 2 + 4*lda); x21 = *(A + 2 + 5*lda); x22 = *(A + 2 + 6*lda); x23 = *(A + 2 + 7*lda);
	        //x30 = *(A + 3 + 4*lda); x31 = *(A + 3 + 5*lda); x32 = *(A + 3 + 6*lda); x33 = *(A + 3 + 7*lda);
	        //*(A2 + 4*4 + 0) = x00; *(A2 + 5*4 + 0) = x01; *(A2 + 6*4 + 0) = x02; *(A2 + 7*4 + 0) = x03;
	        //*(A2 + 4*4 + 1) = x10; *(A2 + 5*4 + 1) = x11; *(A2 + 6*4 + 1) = x12; *(A2 + 7*4 + 1) = x13;
	        //*(A2 + 4*4 + 2) = x20; *(A2 + 5*4 + 2) = x21; *(A2 + 6*4 + 2) = x22; *(A2 + 7*4 + 2) = x23;
	        //*(A2 + 4*4 + 3) = x30; *(A2 + 5*4 + 3) = x31; *(A2 + 6*4 + 3) = x32; *(A2 + 7*4 + 3) = x33;
	        //A += 8*lda ;
	        //A2+= 32;;

	        __asm__ __volatile__ (
	          "\n\t"
	          //"prefetcht0  0(%[a0],%[lda1],8)\n\t"
	          //"prefetcht0  0(%[a1],%[lda1],8)\n\t"
	          //"prefetcht0  0(%[a2],%[lda1],8)\n\t"
	          //"prefetcht0  0(%[a3],%[lda1],8)\n\t"
	          "\n\t"
	          "vmovupd   0*8(%[a0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	          "vmovupd   0*8(%[a1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	          "vmovupd   0*8(%[a2]        ), %%ymm2 \n\t" // [x02,x12,x22,x32]
	          "vmovupd   0*8(%[a3]        ), %%ymm3 \n\t" // [x03,x13,x23,x33]
	          "\n\t"
	          "vmovapd   %%ymm0 ,   0*8(%[aa])\n\t"
	          "vmovapd   %%ymm1 ,   4*8(%[aa])\n\t"
	          "vmovapd   %%ymm2 ,   8*8(%[aa])\n\t"
	          "vmovapd   %%ymm3 ,  12*8(%[aa])\n\t"
	          "\n\t"
	          //"prefetcht0  0(%[a0],%[lda2],8)\n\t"
	          //"prefetcht0  0(%[a1],%[lda2],8)\n\t"
	          //"prefetcht0  0(%[a2],%[lda2],8)\n\t"
	          //"prefetcht0  0(%[a3],%[lda2],8)\n\t"
	          "\n\t"
	          "vmovupd   0*8(%[a0],%[lda1],4), %%ymm4 \n\t" // [x00,x10,x20,x30]
	          "vmovupd   0*8(%[a1],%[lda1],4), %%ymm5 \n\t" // [x01,x11,x21,x31]
	          "vmovupd   0*8(%[a2],%[lda1],4), %%ymm6 \n\t" // [x02,x12,x22,x32]
	          "vmovupd   0*8(%[a3],%[lda1],4), %%ymm7 \n\t" // [x03,x13,x23,x33]
	          "\n\t"
	          "vmovapd   %%ymm4 ,  16*8(%[aa])\n\t"
	          "vmovapd   %%ymm5 ,  20*8(%[aa])\n\t"
	          "vmovapd   %%ymm6 ,  24*8(%[aa])\n\t"
	          "vmovapd   %%ymm7 ,  28*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1],8), %[a0]\n\t"
	          "leaq    0(%[a1],%[lda1],8), %[a1]\n\t"
	          "leaq    0(%[a2],%[lda1],8), %[a2]\n\t"
	          "leaq    0(%[a3],%[lda1],8), %[a3]\n\t"
	          "addq    $32*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(a2),[a3]"+r"(a3),[aa]"+r"(A2)
	        :[lda1]"r"(lda1),[lda2]"r"(lda2) );

	        A += 8*lda ;
	      }
	    }
	    if( K & 4 ){
	        //x00 = *(A + 0 + 0*lda); x01 = *(A + 0 + 1*lda); x02 = *(A + 0 + 2*lda); x03 = *(A + 0 + 3*lda);
	        //x10 = *(A + 1 + 0*lda); x11 = *(A + 1 + 1*lda); x12 = *(A + 1 + 2*lda); x13 = *(A + 1 + 3*lda);
	        //x20 = *(A + 2 + 0*lda); x21 = *(A + 2 + 1*lda); x22 = *(A + 2 + 2*lda); x23 = *(A + 2 + 3*lda);
	        //x30 = *(A + 3 + 0*lda); x31 = *(A + 3 + 1*lda); x32 = *(A + 3 + 2*lda); x33 = *(A + 3 + 3*lda);
	        //*(A2 + 0*4 + 0) = x00; *(A2 + 1*4 + 0) = x01; *(A2 + 2*4 + 0) = x02; *(A2 + 3*4 + 0) = x03;
	        //*(A2 + 0*4 + 1) = x10; *(A2 + 1*4 + 1) = x11; *(A2 + 2*4 + 1) = x12; *(A2 + 3*4 + 1) = x13;
	        //*(A2 + 0*4 + 2) = x20; *(A2 + 1*4 + 2) = x21; *(A2 + 2*4 + 2) = x22; *(A2 + 3*4 + 2) = x23;
	        //*(A2 + 0*4 + 3) = x30; *(A2 + 1*4 + 3) = x31; *(A2 + 2*4 + 3) = x32; *(A2 + 3*4 + 3) = x33;
	        //A += 4*lda ;
	        //A2+= 16;;

	        __asm__ __volatile__ (
	          "\n\t"
	          "vmovupd   0*8(%[a0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	          "vmovupd   0*8(%[a1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	          "vmovupd   0*8(%[a2]        ), %%ymm2 \n\t" // [x02,x12,x22,x32]
	          "vmovupd   0*8(%[a3]        ), %%ymm3 \n\t" // [x03,x13,x23,x33]
	          "\n\t"
	          "vmovapd   %%ymm0 ,   0*8(%[aa])\n\t"
	          "vmovapd   %%ymm1 ,   4*8(%[aa])\n\t"
	          "vmovapd   %%ymm2 ,   8*8(%[aa])\n\t"
	          "vmovapd   %%ymm3 ,  12*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1],4), %[a0]\n\t"
	          "leaq    0(%[a1],%[lda1],4), %[a1]\n\t"
	          "leaq    0(%[a2],%[lda1],4), %[a2]\n\t"
	          "leaq    0(%[a3],%[lda1],4), %[a3]\n\t"
	          "addq    $16*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(a2),[a3]"+r"(a3),[aa]"+r"(A2)
	        :[lda1]"r"(lda1) );

	        A += 4*lda ;

	    }
	    if( K & 2 ){
	        //x00 = *(A + 0 + 0*lda); x01 = *(A + 0 + 1*lda);
	        //x10 = *(A + 1 + 0*lda); x11 = *(A + 1 + 1*lda);
	        //x20 = *(A + 2 + 0*lda); x21 = *(A + 2 + 1*lda);
	        //x30 = *(A + 3 + 0*lda); x31 = *(A + 3 + 1*lda);
	        //*(A2 + 0*4 + 0) = x00; *(A2 + 1*4 + 0) = x01;
	        //*(A2 + 0*4 + 1) = x10; *(A2 + 1*4 + 1) = x11;
	        //*(A2 + 0*4 + 2) = x20; *(A2 + 1*4 + 2) = x21;
	        //*(A2 + 0*4 + 3) = x30; *(A2 + 1*4 + 3) = x31;
	        //A += 2*lda ;
	        //A2+= 8;

	        __asm__ __volatile__ (
	          "\n\t"
	          "vmovupd   0*8(%[a0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	          "vmovupd   0*8(%[a1]        ), %%ymm1 \n\t" // [x01,x11,x21,x31]
	          "\n\t"
	          "vmovapd   %%ymm0 ,   0*8(%[aa])\n\t"
	          "vmovapd   %%ymm1 ,   4*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1],2), %[a0]\n\t"
	          "leaq    0(%[a1],%[lda1],2), %[a1]\n\t"
	          "addq    $8*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[aa]"+r"(A2)
	        :[lda1]"r"(lda1) );

	        A += 2*lda ;

	    }
	    if( K & 1 ){
	        //x00 = *(A + 0 + 0*lda);
	        //x10 = *(A + 1 + 0*lda);
	        //x20 = *(A + 2 + 0*lda);
	        //x30 = *(A + 3 + 0*lda);
	        //*(A2 + 0*4 + 0) = x00;
	        //*(A2 + 0*4 + 1) = x10;
	        //*(A2 + 0*4 + 2) = x20;
	        //*(A2 + 0*4 + 3) = x30;
	        //A += 1*lda ;
	        //A2+= 4;

	        __asm__ __volatile__ (
	          "\n\t"
	          "vmovupd   0*8(%[a0]        ), %%ymm0 \n\t" // [x00,x10,x20,x30]
	          "\n\t"
	          "vmovapd   %%ymm0 ,   0*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1]  ), %[a0]\n\t"
	          "addq    $4*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[aa]"+r"(A2)
	        :[lda1]"r"(lda1) );

	        A += 1*lda ;

	    }
	    A  = A  - lda *K + 4;
	    a0 = a0 - lda *K + 4;
	    a1 = a0 + lda;
	    a2 = a1 + lda;
	    a3 = a2 + lda;

	  }
	}
	if( M & 2 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){

	        //x00 = *(A + 0 + 0*lda); x01 = *(A + 0 + 1*lda); x02 = *(A + 0 + 2*lda); x03 = *(A + 0 + 3*lda);
	        //x10 = *(A + 1 + 0*lda); x11 = *(A + 1 + 1*lda); x12 = *(A + 1 + 2*lda); x13 = *(A + 1 + 3*lda);
	        //*(A2 + 0*2 + 0) = x00; *(A2 + 1*2 + 0) = x01; *(A2 + 2*2 + 0) = x02; *(A2 + 3*2 + 0) = x03;
	        //*(A2 + 0*2 + 1) = x10; *(A2 + 1*2 + 1) = x11; *(A2 + 2*2 + 1) = x12; *(A2 + 3*2 + 1) = x13;
	        //x00 = *(A + 0 + 4*lda); x01 = *(A + 0 + 5*lda); x02 = *(A + 0 + 6*lda); x03 = *(A + 0 + 7*lda);
	        //x10 = *(A + 1 + 4*lda); x11 = *(A + 1 + 5*lda); x12 = *(A + 1 + 6*lda); x13 = *(A + 1 + 7*lda);
	        //*(A2 + 4*2 + 0) = x00; *(A2 + 5*2 + 0) = x01; *(A2 + 6*2 + 0) = x02; *(A2 + 7*2 + 0) = x03;
	        //*(A2 + 4*2 + 1) = x10; *(A2 + 5*2 + 1) = x11; *(A2 + 6*2 + 1) = x12; *(A2 + 7*2 + 1) = x13;
	        //A += 8*lda ;
	        //A2+= 16;;

	        __asm__ __volatile__ (
	          "\n\t"
	          "movupd   0*8(%[a0]        ), %%xmm0 \n\t" // [x00,x10]
	          "movupd   0*8(%[a1]        ), %%xmm1 \n\t" // [x01,x11]
	          "movupd   0*8(%[a2]        ), %%xmm2 \n\t" // [x02,x12]
	          "movupd   0*8(%[a3]        ), %%xmm3 \n\t" // [x03,x13]
	          "\n\t"
	          "movapd   %%xmm0 ,   0*8(%[aa])\n\t"
	          "movapd   %%xmm1 ,   2*8(%[aa])\n\t"
	          "movapd   %%xmm2 ,   4*8(%[aa])\n\t"
	          "movapd   %%xmm3 ,   6*8(%[aa])\n\t"
	          "\n\t"
	          "movupd   0*8(%[a0],%[lda1],4), %%xmm4 \n\t" // [x00,x10]
	          "movupd   0*8(%[a1],%[lda1],4), %%xmm5 \n\t" // [x01,x11]
	          "movupd   0*8(%[a2],%[lda1],4), %%xmm6 \n\t" // [x02,x12]
	          "movupd   0*8(%[a3],%[lda1],4), %%xmm7 \n\t" // [x03,x13]
	          "\n\t"
	          "movapd   %%xmm4 ,   8*8(%[aa])\n\t"
	          "movapd   %%xmm5 ,  10*8(%[aa])\n\t"
	          "movapd   %%xmm6 ,  12*8(%[aa])\n\t"
	          "movapd   %%xmm7 ,  14*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1],8), %[a0]\n\t"
	          "leaq    0(%[a1],%[lda1],8), %[a1]\n\t"
	          "leaq    0(%[a2],%[lda1],8), %[a2]\n\t"
	          "leaq    0(%[a3],%[lda1],8), %[a3]\n\t"
	          "addq    $16*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(a2),[a3]"+r"(a3),[aa]"+r"(A2)
	        :[lda1]"r"(lda1) );

	        A += 8*lda ;

	      }
	    }
	    if( K & 4 ){
	        //x00 = *(A + 0 + 0*lda); x01 = *(A + 0 + 1*lda); x02 = *(A + 0 + 2*lda); x03 = *(A + 0 + 3*lda);
	        //x10 = *(A + 1 + 0*lda); x11 = *(A + 1 + 1*lda); x12 = *(A + 1 + 2*lda); x13 = *(A + 1 + 3*lda);
	        //*(A2 + 0*2 + 0) = x00; *(A2 + 1*2 + 0) = x01; *(A2 + 2*2 + 0) = x02; *(A2 + 3*2 + 0) = x03;
	        //*(A2 + 0*2 + 1) = x10; *(A2 + 1*2 + 1) = x11; *(A2 + 2*2 + 1) = x12; *(A2 + 3*2 + 1) = x13;
	        //A += 4*lda ;
	        //A2+= 8;;

	        __asm__ __volatile__ (
	          "\n\t"
	          "movupd   0*8(%[a0]        ), %%xmm0 \n\t" // [x00,x10]
	          "movupd   0*8(%[a1]        ), %%xmm1 \n\t" // [x01,x11]
	          "movupd   0*8(%[a2]        ), %%xmm2 \n\t" // [x02,x12]
	          "movupd   0*8(%[a3]        ), %%xmm3 \n\t" // [x03,x13]
	          "\n\t"
	          "movapd   %%xmm0 ,   0*8(%[aa])\n\t"
	          "movapd   %%xmm1 ,   2*8(%[aa])\n\t"
	          "movapd   %%xmm2 ,   4*8(%[aa])\n\t"
	          "movapd   %%xmm3 ,   6*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1],4), %[a0]\n\t"
	          "leaq    0(%[a1],%[lda1],4), %[a1]\n\t"
	          "leaq    0(%[a2],%[lda1],4), %[a2]\n\t"
	          "leaq    0(%[a3],%[lda1],4), %[a3]\n\t"
	          "addq    $8*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(a2),[a3]"+r"(a3),[aa]"+r"(A2)
	        :[lda1]"r"(lda1) );

	        A += 4*lda ;

	    }
	    if( K & 2 ){
	        //x00 = *(A + 0 + 0*lda); x01 = *(A + 0 + 1*lda);
	        //x10 = *(A + 1 + 0*lda); x11 = *(A + 1 + 1*lda);
	        //*(A2 + 0*2 + 0) = x00; *(A2 + 1*2 + 0) = x01;
	        //*(A2 + 0*2 + 1) = x10; *(A2 + 1*2 + 1) = x11;
	        //A += 2*lda ;
	        //A2+= 4;

	        __asm__ __volatile__ (
	          "\n\t"
	          "movupd   0*8(%[a0]        ), %%xmm0 \n\t" // [x00,x10]
	          "movupd   0*8(%[a1]        ), %%xmm1 \n\t" // [x01,x11]
	          "\n\t"
	          "movapd   %%xmm0 ,   0*8(%[aa])\n\t"
	          "movapd   %%xmm1 ,   2*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1],2), %[a0]\n\t"
	          "leaq    0(%[a1],%[lda1],2), %[a1]\n\t"
	          "addq    $4*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[aa]"+r"(A2)
	        :[lda1]"r"(lda1) );

	        A += 2*lda ;


	    }
	    if( K & 1 ){
	        //x00 = *(A + 0 + 0*lda);
	        //x10 = *(A + 1 + 0*lda);
	        //*(A2 + 0*2 + 0) = x00;
	        //*(A2 + 0*2 + 1) = x10;
	        //A += 1*lda ;
	        //A2+= 2;

	        __asm__ __volatile__ (
	          "\n\t"
	          "movupd   0*8(%[a0]        ), %%xmm0 \n\t" // [x00,x10]
	          "\n\t"
	          "movapd   %%xmm0 ,   0*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1]), %[a0]\n\t"
	          "addq    $2*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[aa]"+r"(A2)
	        :[lda1]"r"(lda1) );

	        A += 1*lda ;
	    }
	    A  = A  - lda *K + 2;
	    a0 = a0 - lda *K + 2;
	    a1 = a0 + lda;
	    a2 = a1 + lda;
	    a3 = a2 + lda;

	}
	if( M & 1 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){

	        //x00 = *(A + 0 + 0*lda); x01 = *(A + 0 + 1*lda); x02 = *(A + 0 + 2*lda); x03 = *(A + 0 + 3*lda);
	        //*(A2 + 0*1 + 0) = x00; *(A2 + 1*1 + 0) = x01; *(A2 + 2*1 + 0) = x02; *(A2 + 3*1 + 0) = x03;
	        //x00 = *(A + 0 + 4*lda); x01 = *(A + 0 + 5*lda); x02 = *(A + 0 + 6*lda); x03 = *(A + 0 + 7*lda);
	        //*(A2 + 4*1 + 0) = x00; *(A2 + 5*1 + 0) = x01; *(A2 + 6*1 + 0) = x02; *(A2 + 7*1 + 0) = x03;
	        //A += 8*lda ;
	        //A2+= 8;;

	        __asm__ __volatile__ (
	          "\n\t"
	          "movsd   0*8(%[a0]        ), %%xmm0 \n\t" // [x00,x10]
	          "movsd   0*8(%[a1]        ), %%xmm1 \n\t" // [x01,x11]
	          "movsd   0*8(%[a2]        ), %%xmm2 \n\t" // [x02,x12]
	          "movsd   0*8(%[a3]        ), %%xmm3 \n\t" // [x03,x13]
	          "\n\t"
	          "movsd   %%xmm0 ,   0*8(%[aa])\n\t"
	          "movsd   %%xmm1 ,   1*8(%[aa])\n\t"
	          "movsd   %%xmm2 ,   2*8(%[aa])\n\t"
	          "movsd   %%xmm3 ,   3*8(%[aa])\n\t"
	          "\n\t"
	          "movsd   0*8(%[a0],%[lda1],4), %%xmm4 \n\t" // [x00,x10]
	          "movsd   0*8(%[a1],%[lda1],4), %%xmm5 \n\t" // [x01,x11]
	          "movsd   0*8(%[a2],%[lda1],4), %%xmm6 \n\t" // [x02,x12]
	          "movsd   0*8(%[a3],%[lda1],4), %%xmm7 \n\t" // [x03,x13]
	          "\n\t"
	          "movsd   %%xmm4 ,   4*8(%[aa])\n\t"
	          "movsd   %%xmm5 ,   5*8(%[aa])\n\t"
	          "movsd   %%xmm6 ,   6*8(%[aa])\n\t"
	          "movsd   %%xmm7 ,   7*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1],8), %[a0]\n\t"
	          "leaq    0(%[a1],%[lda1],8), %[a1]\n\t"
	          "leaq    0(%[a2],%[lda1],8), %[a2]\n\t"
	          "leaq    0(%[a3],%[lda1],8), %[a3]\n\t"
	          "addq    $8*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(a2),[a3]"+r"(a3),[aa]"+r"(A2)
	        :[lda1]"r"(lda1) );

	        A += 8*lda ;


	      }
	    }
	    if( K & 4 ){
	        //x00 = *(A + 0 + 0*lda); x01 = *(A + 0 + 1*lda); x02 = *(A + 0 + 2*lda); x03 = *(A + 0 + 3*lda);
	        //*(A2 + 0*1 + 0) = x00; *(A2 + 1*1 + 0) = x01; *(A2 + 2*1 + 0) = x02; *(A2 + 3*1 + 0) = x03;
	        //A += 4*lda ;
	        //A2+= 4;;

	        __asm__ __volatile__ (
	          "\n\t"
	          "movsd   0*8(%[a0]        ), %%xmm0 \n\t" // [x00,x10]
	          "movsd   0*8(%[a1]        ), %%xmm1 \n\t" // [x01,x11]
	          "movsd   0*8(%[a2]        ), %%xmm2 \n\t" // [x02,x12]
	          "movsd   0*8(%[a3]        ), %%xmm3 \n\t" // [x03,x13]
	          "\n\t"
	          "movsd   %%xmm0 ,   0*8(%[aa])\n\t"
	          "movsd   %%xmm1 ,   1*8(%[aa])\n\t"
	          "movsd   %%xmm2 ,   2*8(%[aa])\n\t"
	          "movsd   %%xmm3 ,   3*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1],4), %[a0]\n\t"
	          "leaq    0(%[a1],%[lda1],4), %[a1]\n\t"
	          "leaq    0(%[a2],%[lda1],4), %[a2]\n\t"
	          "leaq    0(%[a3],%[lda1],4), %[a3]\n\t"
	          "addq    $4*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[a2]"+r"(a2),[a3]"+r"(a3),[aa]"+r"(A2)
	        :[lda1]"r"(lda1) );

	        A += 4*lda ;


	    }
	    if( K & 2 ){
	        //x00 = *(A + 0 + 0*lda); x01 = *(A + 0 + 1*lda);
	        //*(A2 + 0*1 + 0) = x00; *(A2 + 1*1 + 0) = x01;
	        //A += 2*lda ;
	        //A2+= 2;

	        __asm__ __volatile__ (
	          "\n\t"
	          "movsd   0*8(%[a0]        ), %%xmm0 \n\t" // [x00]
	          "movsd   0*8(%[a1]        ), %%xmm1 \n\t" // [x01]
	          "\n\t"
	          "movsd   %%xmm0 ,   0*8(%[aa])\n\t"
	          "movsd   %%xmm1 ,   1*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1],2), %[a0]\n\t"
	          "leaq    0(%[a1],%[lda1],2), %[a1]\n\t"
	          "addq    $2*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[a1]"+r"(a1),[aa]"+r"(A2)
	        :[lda1]"r"(lda1) );

	        A += 2*lda ;

	    }
	    if( K & 1 ){
	        //x00 = *(A + 0 + 0*lda);
	        //*(A2 + 0*1 + 0) = x00;
	        //A += 1*lda ;
	        //A2+= 1;

	        __asm__ __volatile__ (
	          "\n\t"
	          "movsd   0*8(%[a0]        ), %%xmm0 \n\t" // [x00]
	          "\n\t"
	          "movsd   %%xmm0 ,   0*8(%[aa])\n\t"
	          "\n\t"
	          "leaq    0(%[a0],%[lda1]), %[a0]\n\t"
	          "addq    $1*8,  %[aa]\n\t"
	          "\n\t"
	        :[a0]"+r"(a0),[aa]"+r"(A2)
	        :[lda1]"r"(lda1) );

	        A += 1*lda ;


	    }
	    A  = A  - lda *K + 1;
	    a0 = a0 - lda *K + 2;
	    a1 = a0 + lda;
	    a2 = a1 + lda;
	    a3 = a2 + lda;
	}


}

