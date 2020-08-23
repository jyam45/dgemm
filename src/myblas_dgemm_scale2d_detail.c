#include "myblas_internal.h"
#include <stdio.h>

void myblas_dgemm_scale2d_detail(size_t M, size_t N, double beta, double *C, size_t ldc ){

	double* c0 = C;
	double* c1 = C + ldc;
	size_t  ldc2 = 2*ldc*sizeof(double); // *** byte unit ***
	size_t  ldc3 = 3*ldc*sizeof(double); // *** byte unit ***

	//printf("M=%d, N=%d, ldc=%d\n",M,N,ldc);

	__asm__ __volatile__ (
	   "vbroadcastsd  %[beta],  %%ymm15\n\t"
	   :: [beta]"m"(beta)
	);

	// scaling beta*C
	size_t n = N;
	if( n >> 2 ){
	    size_t n4 = ( n >> 2 ); // unrolling with 4 elements
	    while( n4-- )
	    {
	        __asm__ __volatile__ (
	          "prefetcht1   0*8(%[c0],%[ldc2],2)\n\t"
	          "prefetcht1   0*8(%[c1],%[ldc2],2)\n\t"
	          "prefetcht1   0*8(%[c0],%[ldc3],2)\n\t"
	          "prefetcht1   0*8(%[c1],%[ldc3],2)\n\t"
	          :[c0]"+r"(c0),[c1]"+r"(c1)
	          :[ldc2]"r"(ldc2),[ldc3]"r"(ldc3)
	        );
	        size_t m = M;
	        if( m >> 4 ){
	            size_t m16 = ( m >> 4 );
	            while( m16-- ){

	                __asm__ __volatile__ (
	                    "prefetcht0  16*8(%[c0]        )\n\t"
	                    "prefetcht0  16*8(%[c1]        )\n\t"
	                    "prefetcht0  16*8(%[c0],%[ldc2])\n\t"
	                    "prefetcht0  16*8(%[c1],%[ldc2])\n\t"
	                    "vmovupd   0*8(%[c0]        ), %%ymm0\n\t"
	                    "vmovupd   0*8(%[c1]        ), %%ymm1\n\t"
	                    "vmovupd   0*8(%[c0],%[ldc2]), %%ymm2\n\t"
	                    "vmovupd   0*8(%[c1],%[ldc2]), %%ymm3\n\t"
	                    "vmulpd    %%ymm15,  %%ymm0 ,  %%ymm0\n\t"
	                    "vmulpd    %%ymm15,  %%ymm1 ,  %%ymm1\n\t"
	                    "vmulpd    %%ymm15,  %%ymm2 ,  %%ymm2\n\t"
	                    "vmulpd    %%ymm15,  %%ymm3 ,  %%ymm3\n\t"
	                    "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	                    "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	                    "vmovupd   %%ymm2,   0*8(%[c0],%[ldc2])\n\t"
	                    "vmovupd   %%ymm3,   0*8(%[c1],%[ldc2])\n\t"
	                    "\n\t"
	                    "vmovupd   4*8(%[c0]        ), %%ymm8 \n\t"
	                    "vmovupd   4*8(%[c1]        ), %%ymm9 \n\t"
	                    "vmovupd   4*8(%[c0],%[ldc2]), %%ymm10\n\t"
	                    "vmovupd   4*8(%[c1],%[ldc2]), %%ymm11\n\t"
	                    "vmulpd    %%ymm15,  %%ymm8 ,  %%ymm8 \n\t"
	                    "vmulpd    %%ymm15,  %%ymm9 ,  %%ymm9 \n\t"
	                    "vmulpd    %%ymm15,  %%ymm10,  %%ymm10\n\t"
	                    "vmulpd    %%ymm15,  %%ymm11,  %%ymm11\n\t"
	                    "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	                    "vmovupd   %%ymm9 ,   4*8(%[c1]        )\n\t"
	                    "vmovupd   %%ymm10,   4*8(%[c0],%[ldc2])\n\t"
	                    "vmovupd   %%ymm11,   4*8(%[c1],%[ldc2])\n\t"
	                    "\n\t"
	                    "vmovupd   8*8(%[c0]        ), %%ymm4\n\t"
	                    "vmovupd   8*8(%[c1]        ), %%ymm5\n\t"
	                    "vmovupd   8*8(%[c0],%[ldc2]), %%ymm6\n\t"
	                    "vmovupd   8*8(%[c1],%[ldc2]), %%ymm7\n\t"
	                    "vmulpd    %%ymm15,  %%ymm4 ,  %%ymm4\n\t"
	                    "vmulpd    %%ymm15,  %%ymm5 ,  %%ymm5\n\t"
	                    "vmulpd    %%ymm15,  %%ymm6 ,  %%ymm6\n\t"
	                    "vmulpd    %%ymm15,  %%ymm7 ,  %%ymm7\n\t"
	                    "vmovupd   %%ymm4,   8*8(%[c0]        )\n\t"
	                    "vmovupd   %%ymm5,   8*8(%[c1]        )\n\t"
	                    "vmovupd   %%ymm6,   8*8(%[c0],%[ldc2])\n\t"
	                    "vmovupd   %%ymm7,   8*8(%[c1],%[ldc2])\n\t"
	                    "\n\t"
	                    "vmovupd  12*8(%[c0]        ), %%ymm12\n\t"
	                    "vmovupd  12*8(%[c1]        ), %%ymm13\n\t"
	                    "vmovupd  12*8(%[c0],%[ldc2]), %%ymm14\n\t"
	                    "vmovupd  12*8(%[c1],%[ldc2]), %%ymm11\n\t"
	                    "vmulpd    %%ymm15,  %%ymm12,  %%ymm12\n\t"
	                    "vmulpd    %%ymm15,  %%ymm13,  %%ymm13\n\t"
	                    "vmulpd    %%ymm15,  %%ymm14,  %%ymm14\n\t"
	                    "vmulpd    %%ymm15,  %%ymm11,  %%ymm11\n\t"
	                    "vmovupd   %%ymm12,  12*8(%[c0]        )\n\t"
	                    "vmovupd   %%ymm13,  12*8(%[c1]        )\n\t"
	                    "vmovupd   %%ymm14,  12*8(%[c0],%[ldc2])\n\t"
	                    "vmovupd   %%ymm11,  12*8(%[c1],%[ldc2])\n\t"
	                    "\n\t"
	                    "addq   $16*8, %[c0]\n\t"
	                    "addq   $16*8, %[c1]\n\t"
	                    "\n\t"
	                    :[c0]"=r"(c0),[c1]"=r"(c1)
	                    :"0"(c0),"1"(c1),[ldc2]"r"(ldc2)
	                    );
	            }
	        }

	        if( m & 8 ){

	            __asm__ __volatile__ (
	                "vmovupd   0*8(%[c0]        ), %%ymm0\n\t"
	                "vmovupd   0*8(%[c1]        ), %%ymm1\n\t"
	                "vmovupd   0*8(%[c0],%[ldc2]), %%ymm2\n\t"
	                "vmovupd   0*8(%[c1],%[ldc2]), %%ymm3\n\t"
	                "vmulpd    %%ymm15,  %%ymm0 ,  %%ymm0\n\t"
	                "vmulpd    %%ymm15,  %%ymm1 ,  %%ymm1\n\t"
	                "vmulpd    %%ymm15,  %%ymm2 ,  %%ymm2\n\t"
	                "vmulpd    %%ymm15,  %%ymm3 ,  %%ymm3\n\t"
	                "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	                "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	                "vmovupd   %%ymm2,   0*8(%[c0],%[ldc2])\n\t"
	                "vmovupd   %%ymm3,   0*8(%[c1],%[ldc2])\n\t"
	                "\n\t"
	                "vmovupd   4*8(%[c0]        ), %%ymm8 \n\t"
	                "vmovupd   4*8(%[c1]        ), %%ymm9 \n\t"
	                "vmovupd   4*8(%[c0],%[ldc2]), %%ymm10\n\t"
	                "vmovupd   4*8(%[c1],%[ldc2]), %%ymm11\n\t"
	                "vmulpd    %%ymm15,  %%ymm8 ,  %%ymm8 \n\t"
	                "vmulpd    %%ymm15,  %%ymm9 ,  %%ymm9 \n\t"
	                "vmulpd    %%ymm15,  %%ymm10,  %%ymm10\n\t"
	                "vmulpd    %%ymm15,  %%ymm11,  %%ymm11\n\t"
	                "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	                "vmovupd   %%ymm9 ,   4*8(%[c1]        )\n\t"
	                "vmovupd   %%ymm10,   4*8(%[c0],%[ldc2])\n\t"
	                "vmovupd   %%ymm11,   4*8(%[c1],%[ldc2])\n\t"
	                "\n\t"
	                "addq   $8*8, %[c0]\n\t"
	                "addq   $8*8, %[c1]\n\t"
	                "\n\t"
	                :[c0]"=r"(c0),[c1]"=r"(c1)
	                :"0"(c0),"1"(c1),[ldc2]"r"(ldc2)
	                );
	        }
	        if( m & 4 ){

	            __asm__ __volatile__ (
	                "vmovupd   0*8(%[c0]        ), %%ymm0\n\t"
	                "vmovupd   0*8(%[c1]        ), %%ymm1\n\t"
	                "vmovupd   0*8(%[c0],%[ldc2]), %%ymm2\n\t"
	                "vmovupd   0*8(%[c1],%[ldc2]), %%ymm3\n\t"
	                "vmulpd    %%ymm15,  %%ymm0 ,  %%ymm0\n\t"
	                "vmulpd    %%ymm15,  %%ymm1 ,  %%ymm1\n\t"
	                "vmulpd    %%ymm15,  %%ymm2 ,  %%ymm2\n\t"
	                "vmulpd    %%ymm15,  %%ymm3 ,  %%ymm3\n\t"
	                "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	                "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	                "vmovupd   %%ymm2,   0*8(%[c0],%[ldc2])\n\t"
	                "vmovupd   %%ymm3,   0*8(%[c1],%[ldc2])\n\t"
	                "\n\t"
	                "addq   $4*8, %[c0]\n\t"
	                "addq   $4*8, %[c1]\n\t"
	                "\n\t"
	                :[c0]"=r"(c0),[c1]"=r"(c1)
	                :"0"(c0),"1"(c1),[ldc2]"r"(ldc2)
	                );
	
		}
	        if( m & 2 ){

	            __asm__ __volatile__ (
	                "movupd   0*8(%[c0]        ), %%xmm0\n\t"
	                "movupd   0*8(%[c1]        ), %%xmm1\n\t"
	                "movupd   0*8(%[c0],%[ldc2]), %%xmm2\n\t"
	                "movupd   0*8(%[c1],%[ldc2]), %%xmm3\n\t"
	                "mulpd    %%xmm15,  %%xmm0 \n\t"
	                "mulpd    %%xmm15,  %%xmm1 \n\t"
	                "mulpd    %%xmm15,  %%xmm2 \n\t"
	                "mulpd    %%xmm15,  %%xmm3 \n\t"
	                "movupd   %%xmm0,   0*8(%[c0]        )\n\t"
	                "movupd   %%xmm1,   0*8(%[c1]        )\n\t"
	                "movupd   %%xmm2,   0*8(%[c0],%[ldc2])\n\t"
	                "movupd   %%xmm3,   0*8(%[c1],%[ldc2])\n\t"
	                "\n\t"
	                "addq   $2*8, %[c0]\n\t"
	                "addq   $2*8, %[c1]\n\t"
	                "\n\t"
	                :[c0]"=r"(c0),[c1]"=r"(c1)
	                :"0"(c0),"1"(c1),[ldc2]"r"(ldc2)
	                );
	
	        }
	        if( m & 1 ){

	            __asm__ __volatile__ (
	                "movsd   0*8(%[c0]        ), %%xmm0\n\t"
	                "movsd   0*8(%[c1]        ), %%xmm1\n\t"
	                "movsd   0*8(%[c0],%[ldc2]), %%xmm2\n\t"
	                "movsd   0*8(%[c1],%[ldc2]), %%xmm3\n\t"
	                "mulsd    %%xmm15,  %%xmm0 \n\t"
	                "mulsd    %%xmm15,  %%xmm1 \n\t"
	                "mulsd    %%xmm15,  %%xmm2 \n\t"
	                "mulsd    %%xmm15,  %%xmm3 \n\t"
	                "movlpd   %%xmm0,   0*8(%[c0]        )\n\t"
	                "movlpd   %%xmm1,   0*8(%[c1]        )\n\t"
	                "movlpd   %%xmm2,   0*8(%[c0],%[ldc2])\n\t"
	                "movlpd   %%xmm3,   0*8(%[c1],%[ldc2])\n\t"
	                "\n\t"
	                "addq   $1*8, %[c0]\n\t"
	                "addq   $1*8, %[c1]\n\t"
	                "\n\t"
	                :[c0]"=r"(c0),[c1]"=r"(c1)
	                :"0"(c0),"1"(c1),[ldc2]"r"(ldc2)
	                );

		}
	        c0= c0- M + 4*ldc;
	        c1= c1- M + 4*ldc;
	    }
	}
	if( n & 2 ){
	    size_t m = M;
	    if( m >> 4 ){
	        size_t m16 = ( m >> 4 );
	        while( m16-- ){

	            __asm__ __volatile__ (
	                "vmovupd   0*8(%[c0]        ), %%ymm0\n\t"
	                "vmovupd   0*8(%[c1]        ), %%ymm1\n\t"
	                "vmulpd    %%ymm15,  %%ymm0 ,  %%ymm0\n\t"
	                "vmulpd    %%ymm15,  %%ymm1 ,  %%ymm1\n\t"
	                "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	                "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	                "\n\t"
	                "vmovupd   4*8(%[c0]        ), %%ymm8 \n\t"
	                "vmovupd   4*8(%[c1]        ), %%ymm9 \n\t"
	                "vmulpd    %%ymm15,  %%ymm8 ,  %%ymm8 \n\t"
	                "vmulpd    %%ymm15,  %%ymm9 ,  %%ymm9 \n\t"
	                "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	                "vmovupd   %%ymm9 ,   4*8(%[c1]        )\n\t"
	                "\n\t"
	                "vmovupd   8*8(%[c0]        ), %%ymm4\n\t"
	                "vmovupd   8*8(%[c1]        ), %%ymm5\n\t"
	                "vmulpd    %%ymm15,  %%ymm4 ,  %%ymm4\n\t"
	                "vmulpd    %%ymm15,  %%ymm5 ,  %%ymm5\n\t"
	                "vmovupd   %%ymm4,   8*8(%[c0]        )\n\t"
	                "vmovupd   %%ymm5,   8*8(%[c1]        )\n\t"
	                "\n\t"
	                "vmovupd  12*8(%[c0]        ), %%ymm12\n\t"
	                "vmovupd  12*8(%[c1]        ), %%ymm13\n\t"
	                "vmulpd    %%ymm15,  %%ymm12,  %%ymm12\n\t"
	                "vmulpd    %%ymm15,  %%ymm13,  %%ymm13\n\t"
	                "vmovupd   %%ymm12,  12*8(%[c0]        )\n\t"
	                "vmovupd   %%ymm13,  12*8(%[c1]        )\n\t"
	                "\n\t"
	                "addq   $16*8, %[c0]\n\t"
	                "addq   $16*8, %[c1]\n\t"
	                "\n\t"
	                :[c0]"=r"(c0),[c1]"=r"(c1)
	                :"0"(c0),"1"(c1)
	                );

	        }
	    }
	    if( m & 8  ){
	        
	        __asm__ __volatile__ (
	            "vmovupd   0*8(%[c0]        ), %%ymm0\n\t"
	            "vmovupd   0*8(%[c1]        ), %%ymm1\n\t"
	            "vmulpd    %%ymm15,  %%ymm0 ,  %%ymm0\n\t"
	            "vmulpd    %%ymm15,  %%ymm1 ,  %%ymm1\n\t"
	            "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	            "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	            "\n\t"
	            "vmovupd   4*8(%[c0]        ), %%ymm8 \n\t"
	            "vmovupd   4*8(%[c1]        ), %%ymm9 \n\t"
	            "vmulpd    %%ymm15,  %%ymm8 ,  %%ymm8 \n\t"
	            "vmulpd    %%ymm15,  %%ymm9 ,  %%ymm9 \n\t"
	            "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	            "vmovupd   %%ymm9 ,   4*8(%[c1]        )\n\t"
	            "\n\t"
	            "addq   $8*8, %[c0]\n\t"
	            "addq   $8*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"=r"(c0),[c1]"=r"(c1)
	            :"0"(c0),"1"(c1),[ldc2]"r"(ldc2)
	            );
	    }
	    if( m & 4 ){

	        __asm__ __volatile__ (
	            "vmovupd   0*8(%[c0]        ), %%ymm0\n\t"
	            "vmovupd   0*8(%[c1]        ), %%ymm1\n\t"
	            "vmulpd    %%ymm15,  %%ymm0 ,  %%ymm0\n\t"
	            "vmulpd    %%ymm15,  %%ymm1 ,  %%ymm1\n\t"
	            "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	            "vmovupd   %%ymm1,   0*8(%[c1]        )\n\t"
	            "\n\t"
	            "addq   $4*8, %[c0]\n\t"
	            "addq   $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"=r"(c0),[c1]"=r"(c1)
	            :"0"(c0),"1"(c1),[ldc2]"r"(ldc2)
	            );
	
	    }
	    if( m & 2 ){

	        __asm__ __volatile__ (
	            "movupd   0*8(%[c0]        ), %%xmm0\n\t"
	            "movupd   0*8(%[c1]        ), %%xmm1\n\t"
	            "mulpd    %%xmm15,  %%xmm0 \n\t"
	            "mulpd    %%xmm15,  %%xmm1 \n\t"
	            "movupd   %%xmm0,   0*8(%[c0]        )\n\t"
	            "movupd   %%xmm1,   0*8(%[c1]        )\n\t"
	            "\n\t"
	            "addq   $2*8, %[c0]\n\t"
	            "addq   $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"=r"(c0),[c1]"=r"(c1)
	            :"0"(c0),"1"(c1),[ldc2]"r"(ldc2)
	            );
	
	    }
	    if( m & 1 ){

	        __asm__ __volatile__ (
	            "movsd   0*8(%[c0]        ), %%xmm0\n\t"
	            "movsd   0*8(%[c1]        ), %%xmm1\n\t"
	            "mulsd    %%xmm15,  %%xmm0 \n\t"
	            "mulsd    %%xmm15,  %%xmm1 \n\t"
	            "movlpd   %%xmm0,   0*8(%[c0]        )\n\t"
	            "movlpd   %%xmm1,   0*8(%[c1]        )\n\t"
	            "\n\t"
	            "addq   $1*8, %[c0]\n\t"
	            "addq   $1*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"=r"(c0),[c1]"=r"(c1)
	            :"0"(c0),"1"(c1),[ldc2]"r"(ldc2)
	            );

	    }
	    c0= c0- M + 2*ldc;
	    c1= c1- M + 2*ldc;

	}
	if( n & 1 ){
	    size_t m = M;
	    if( m >> 4 ){
	        size_t m16 = ( m >> 4 );
	        while( m16-- ){

	            __asm__ __volatile__ (
	                "vmovupd   0*8(%[c0]        ), %%ymm0\n\t"
	                "vmulpd    %%ymm15,  %%ymm0 ,  %%ymm0\n\t"
	                "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	                "\n\t"
	                "vmovupd   4*8(%[c0]        ), %%ymm8 \n\t"
	                "vmulpd    %%ymm15,  %%ymm8 ,  %%ymm8 \n\t"
	                "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	                "\n\t"
	                "vmovupd   8*8(%[c0]        ), %%ymm4\n\t"
	                "vmulpd    %%ymm15,  %%ymm4 ,  %%ymm4\n\t"
	                "vmovupd   %%ymm4,   8*8(%[c0]        )\n\t"
	                "\n\t"
	                "vmovupd  12*8(%[c0]        ), %%ymm12\n\t"
	                "vmulpd    %%ymm15,  %%ymm12,  %%ymm12\n\t"
	                "vmovupd   %%ymm12,  12*8(%[c0]        )\n\t"
	                "\n\t"
	                "addq   $16*8, %[c0]\n\t"
	                "\n\t"
	                :[c0]"=r"(c0)
	                :"0"(c0)
	                );

	        }
	    }
	    if( m & 8 ){
	    
	        __asm__ __volatile__ (
	            "vmovupd   0*8(%[c0]        ), %%ymm0\n\t"
	            "vmulpd    %%ymm15,  %%ymm0 ,  %%ymm0\n\t"
	            "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	            "\n\t"
	            "vmovupd   4*8(%[c0]        ), %%ymm8 \n\t"
	            "vmulpd    %%ymm15,  %%ymm8 ,  %%ymm8 \n\t"
	            "vmovupd   %%ymm8 ,   4*8(%[c0]        )\n\t"
	            "\n\t"
	            "addq   $8*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"=r"(c0)
	            :"0"(c0)
	            );

	    }
	    if( m & 4 ){
	   
	        __asm__ __volatile__ (
	            "vmovupd   0*8(%[c0]        ), %%ymm0\n\t"
	            "vmulpd    %%ymm15,  %%ymm0 ,  %%ymm0\n\t"
	            "vmovupd   %%ymm0,   0*8(%[c0]        )\n\t"
	            "\n\t"
	            "addq   $4*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"=r"(c0)
	            :"0"(c0)
	            );
	
	    }
	    if( m & 2 ){

	        __asm__ __volatile__ (
	            "movupd   0*8(%[c0]        ), %%xmm0\n\t"
	            "mulpd    %%xmm15,  %%xmm0 \n\t"
	            "movupd   %%xmm0,   0*8(%[c0]        )\n\t"
	            "\n\t"
	            "addq   $2*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"=r"(c0)
	            :"0"(c0)
	            );
	
	    }
	    if( m & 1 ){
	    
	        __asm__ __volatile__ (
	            "movsd   0*8(%[c0]        ), %%xmm0\n\t"
	            "mulsd    %%xmm15,  %%xmm0 \n\t"
	            "movlpd   %%xmm0,   0*8(%[c0]        )\n\t"
	            "\n\t"
	            "addq   $1*8, %[c0]\n\t"
	            "\n\t"
	            :[c0]"=r"(c0)
	            :"0"(c0)
	            );

	    }
	    c0= c0- M + ldc;

	}
	c0= c0- ldc*N; // retern to head of pointer.

}

