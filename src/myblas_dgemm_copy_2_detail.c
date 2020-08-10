#include "myblas_internal.h"

void myblas_dgemm_copy_2_detail( size_t M, size_t K, const double* A, double *A2 ){

	if( M >> 2 ){
	  size_t m4 = ( M >> 2 );
	  while( m4-- ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	    
	        __asm__ __volatile__ (
	            "\n\t"
	            "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	            "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	            "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t"
	            "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t"
	            "vbroadcastf128  8*8(%[a]), %%ymm4 \n\t"
	            "vbroadcastf128 10*8(%[a]), %%ymm5 \n\t"
	            "vbroadcastf128 12*8(%[a]), %%ymm6 \n\t"
	            "vbroadcastf128 14*8(%[a]), %%ymm7 \n\t"
	            "vbroadcastf128 16*8(%[a]), %%ymm8 \n\t"
	            "vbroadcastf128 18*8(%[a]), %%ymm9 \n\t"
	            "vbroadcastf128 20*8(%[a]), %%ymm10\n\t"
	            "vbroadcastf128 22*8(%[a]), %%ymm11\n\t"
	            "vbroadcastf128 24*8(%[a]), %%ymm12\n\t"
	            "vbroadcastf128 26*8(%[a]), %%ymm13\n\t"
	            "vbroadcastf128 28*8(%[a]), %%ymm14\n\t"
	            "vbroadcastf128 30*8(%[a]), %%ymm15\n\t"
	            "\n\t"
	            "vmovapd      %%ymm0  ,  0*8(%[a2])\n\t"
	            "vmovapd      %%ymm1  ,  4*8(%[a2])\n\t"
	            "vmovapd      %%ymm2  ,  8*8(%[a2])\n\t"
	            "vmovapd      %%ymm3  , 12*8(%[a2])\n\t"
	            "vmovapd      %%ymm4  , 16*8(%[a2])\n\t"
	            "vmovapd      %%ymm5  , 20*8(%[a2])\n\t"
	            "vmovapd      %%ymm6  , 24*8(%[a2])\n\t"
	            "vmovapd      %%ymm7  , 28*8(%[a2])\n\t"
	            "vmovapd      %%ymm8  , 32*8(%[a2])\n\t"
	            "vmovapd      %%ymm9  , 36*8(%[a2])\n\t"
	            "vmovapd      %%ymm10 , 40*8(%[a2])\n\t"
	            "vmovapd      %%ymm11 , 44*8(%[a2])\n\t"
	            "vmovapd      %%ymm12 , 48*8(%[a2])\n\t"
	            "vmovapd      %%ymm13 , 52*8(%[a2])\n\t"
	            "vmovapd      %%ymm14 , 56*8(%[a2])\n\t"
	            "vmovapd      %%ymm15 , 60*8(%[a2])\n\t"
	            "\n\t"
	            "addq  $32*8, %[a]\n\t"
	            "addq  $64*8, %[a2]\n\t"
	            "\n\t"
	            :[a]"+r"(A),[a2]"+r"(A2)
	        :);
	      }
	    }
	    if( K & 4 ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	            "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	            "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t"
	            "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t"
	            "vbroadcastf128  8*8(%[a]), %%ymm4 \n\t"
	            "vbroadcastf128 10*8(%[a]), %%ymm5 \n\t"
	            "vbroadcastf128 12*8(%[a]), %%ymm6 \n\t"
	            "vbroadcastf128 14*8(%[a]), %%ymm7 \n\t"
	            "\n\t"
	            "vmovapd      %%ymm0  ,  0*8(%[a2])\n\t"
	            "vmovapd      %%ymm1  ,  4*8(%[a2])\n\t"
	            "vmovapd      %%ymm2  ,  8*8(%[a2])\n\t"
	            "vmovapd      %%ymm3  , 12*8(%[a2])\n\t"
	            "vmovapd      %%ymm4  , 16*8(%[a2])\n\t"
	            "vmovapd      %%ymm5  , 20*8(%[a2])\n\t"
	            "vmovapd      %%ymm6  , 24*8(%[a2])\n\t"
	            "vmovapd      %%ymm7  , 28*8(%[a2])\n\t"
	            "\n\t"
	            "addq  $16*8, %[a]\n\t"
	            "addq  $32*8, %[a2]\n\t"
	            "\n\t"
	            :[a]"+r"(A),[a2]"+r"(A2)
	        :);
	       
	    }
	    if( K & 2 ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	            "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	            "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t"
	            "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t"
	            "\n\t"
	            "vmovapd      %%ymm0  ,  0*8(%[a2])\n\t"
	            "vmovapd      %%ymm1  ,  4*8(%[a2])\n\t"
	            "vmovapd      %%ymm2  ,  8*8(%[a2])\n\t"
	            "vmovapd      %%ymm3  , 12*8(%[a2])\n\t"
	            "\n\t"
	            "addq  $8*8, %[a]\n\t"
	            "addq  $16*8, %[a2]\n\t"
	            "\n\t"
	            :[a]"+r"(A),[a2]"+r"(A2)
	        :);

	    }
	    if( K & 1 ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	            "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	            "\n\t"
	            "vmovapd      %%ymm0  ,  0*8(%[a2])\n\t"
	            "vmovapd      %%ymm1  ,  4*8(%[a2])\n\t"
	            "\n\t"
	            "addq  $4*8, %[a]\n\t"
	            "addq  $8*8, %[a2]\n\t"
	            "\n\t"
	            :[a]"+r"(A),[a2]"+r"(A2)
	        :);


	    }

	  }

	}
	if( M & 2 ){

	    if( K >> 3 ){
	      size_t k8 = ( K >> 3 );
	      while( k8-- ){
	        __asm__ __volatile__ (
	            "\n\t"
	            "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	            "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	            "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t"
	            "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t"
	            "vbroadcastf128  8*8(%[a]), %%ymm4 \n\t"
	            "vbroadcastf128 10*8(%[a]), %%ymm5 \n\t"
	            "vbroadcastf128 12*8(%[a]), %%ymm6 \n\t"
	            "vbroadcastf128 14*8(%[a]), %%ymm7 \n\t"
	            "\n\t"
	            "vmovapd      %%ymm0  ,  0*8(%[a2])\n\t"
	            "vmovapd      %%ymm1  ,  4*8(%[a2])\n\t"
	            "vmovapd      %%ymm2  ,  8*8(%[a2])\n\t"
	            "vmovapd      %%ymm3  , 12*8(%[a2])\n\t"
	            "vmovapd      %%ymm4  , 16*8(%[a2])\n\t"
	            "vmovapd      %%ymm5  , 20*8(%[a2])\n\t"
	            "vmovapd      %%ymm6  , 24*8(%[a2])\n\t"
	            "vmovapd      %%ymm7  , 28*8(%[a2])\n\t"
	            "\n\t"
	            "addq  $16*8, %[a]\n\t"
	            "addq  $32*8, %[a2]\n\t"
	            "\n\t"
	            :[a]"+r"(A),[a2]"+r"(A2)
	        :);


	      }
	    }
	    if( K & 4 ){
	        __asm__ __volatile__ (
	            "\n\t"
	            "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	            "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	            "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t"
	            "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t"
	            "\n\t"
	            "vmovapd      %%ymm0  ,  0*8(%[a2])\n\t"
	            "vmovapd      %%ymm1  ,  4*8(%[a2])\n\t"
	            "vmovapd      %%ymm2  ,  8*8(%[a2])\n\t"
	            "vmovapd      %%ymm3  , 12*8(%[a2])\n\t"
	            "\n\t"
	            "addq  $8*8 , %[a]\n\t"
	            "addq  $16*8, %[a2]\n\t"
	            "\n\t"
	            :[a]"+r"(A),[a2]"+r"(A2)
	        :);


	    }
	    if( K & 2 ){
	        __asm__ __volatile__ (
	            "\n\t"
	            "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	            "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	            "\n\t"
	            "vmovapd      %%ymm0  ,  0*8(%[a2])\n\t"
	            "vmovapd      %%ymm1  ,  4*8(%[a2])\n\t"
	            "\n\t"
	            "addq  $4*8, %[a]\n\t"
	            "addq  $8*8, %[a2]\n\t"
	            "\n\t"
	            :[a]"+r"(A),[a2]"+r"(A2)
	        :);


	    }
	    if( K & 1 ){

	        __asm__ __volatile__ (
	            "\n\t"
	            "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	            "vmovapd      %%ymm0  ,  0*8(%[a2])\n\t"
	            "\n\t"
	            "addq  $2*8, %[a]\n\t"
	            "addq  $4*8, %[a2]\n\t"
	            "\n\t"
	            :[a]"+r"(A),[a2]"+r"(A2)
	        :);

	    }

	}
	if( M & 1 ){

	    //if( K >> 3 ){
	    //  size_t k8 = ( K >> 3 );
	    //  while( k8-- ){

	    //    __asm__ __volatile__ (
	    //        "\n\t"
	    //        "vbroadcastsd  0*8(%[a]), %%ymm0 \n\t"
	    //        "vbroadcastsd  1*8(%[a]), %%ymm1 \n\t"
	    //        "vbroadcastsd  2*8(%[a]), %%ymm2 \n\t"
	    //        "vbroadcastsd  3*8(%[a]), %%ymm3 \n\t"
	    //        "\n\t"
	    //        "vbroadcastsd  4*8(%[a]), %%ymm9 \n\t"
	    //        "vbroadcastsd  5*8(%[a]), %%ymm10\n\t"
	    //        "vbroadcastsd  6*8(%[a]), %%ymm11\n\t"
	    //        "vbroadcastsd  7*8(%[a]), %%ymm12\n\t"
	    //        "\n\t"
	    //        "addq  $8*8 , %[a]\n\t"
	    //        "addq  $32*8, %[b]\n\t"
	    //        "\n\t"
	    //        :[a]"+r"(A),[b]"+r"(B)
	    //    :);


	    //  }
	    //}
	    //if( K & 4 ){
	    //    __asm__ __volatile__ (
	    //        "\n\t"
	    //        "vbroadcastsd  0*8(%[a]), %%ymm0 \n\t"
	    //        "vbroadcastsd  1*8(%[a]), %%ymm1 \n\t"
	    //        "vbroadcastsd  2*8(%[a]), %%ymm2 \n\t"
	    //        "vbroadcastsd  3*8(%[a]), %%ymm3 \n\t"
	    //        "\n\t"
	    //        "addq  $4*8 , %[a]\n\t"
	    //        "addq  $16*8, %[b]\n\t"
	    //        "\n\t"
	    //        :[a]"+r"(A),[b]"+r"(B)
	    //    :);

	    //}
	    //if( K & 2 ){

	    //    __asm__ __volatile__ (
	    //        "\n\t"
	    //        "vbroadcastsd  0*8(%[a]), %%ymm0 \n\t"
	    //        "vbroadcastsd  1*8(%[a]), %%ymm1 \n\t"
	    //        "\n\t"
	    //        "addq  $2*8 , %[a]\n\t"
	    //        "addq  $8*8, %[b]\n\t"
	    //        "\n\t"
	    //        :[a]"+r"(A),[b]"+r"(B)
	    //    :);

	    //}
	    //if( K & 1 ){

	    //    __asm__ __volatile__ (
	    //        "\n\t"
	    //        "vbroadcastsd  0*8(%[a]), %%ymm0 \n\t"
	    //        "\n\t"
	    //        "addq  $1*8 , %[a]\n\t"
	    //        "addq  $4*8, %[b]\n\t"
	    //        "\n\t"
	    //        :[a]"+r"(A),[b]"+r"(B)
	    //    :);

	    //}

	    A  = A  +   K;
	    A2 = A2 + 2*K;

	}
	A = A + M*K;
	A2= A2+ M*K*2;

}
