#include "myblas_internal.h"
#include <stdio.h>
#include <stdlib.h>

void myblas_dgemm_kernel_detail(
         size_t M, size_t N, size_t K,
         double alpha, const double *A, const double *B, 
         double *C, size_t ldc )
{
	//double a00,a01,a02,a03;
	//double a10,a11,a12,a13;
	//double a20,a21,a22,a23;
	//double a30,a31,a32,a33;

	//double b00,b01,b02,b03;
	//double b10,b11,b12,b13;
	//double b20,b21,b22,b23;
	//double b30,b31,b32,b33;

	//double c00,c01,c02,c03;
	//double c10,c11,c12,c13;
	//double c20,c21,c22,c23;
	//double c30,c31,c32,c33;

	double *c0 = C;
	double *c1 = C + ldc;
	size_t ldc2 = ldc * 2 * sizeof(double);
	double alpha4[4] = {alpha,alpha,alpha,alpha};

	//double cc[16] = {0e0};
	//double *cc = (double*)calloc(32,sizeof(double));
	//printf("%x\n",(uint64_t)cc);

	// ---- Kernel
	if( N >> 2 ){
	  size_t n4 = ( N >> 2 );
	  while( n4-- ){
	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){


	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;

	        __asm__ __volatile__ (
	            "\n\t"
	            //"prefetcht1   32*8(%[a])\n\t"
	            //"prefetcht0   64*8(%[b])\n\t"
	            "prefetchw     0*8(%[c0]        )\n\t"
	            "prefetchw     0*8(%[c1]        )\n\t"
	            "prefetchw     0*8(%[c0],%[ldc2])\n\t"
	            "prefetchw     0*8(%[c1],%[ldc2])\n\t"
	        ::[a]"r"(A),[b]"r"(B),[c0]"r"(c0),[c1]"r"(c1),[ldc2]"r"(ldc2)
	        );

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 ); a02 = *(A + 0 + 2*4 ); a03 = *(A + 0 + 3*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 ); a12 = *(A + 1 + 2*4 ); a13 = *(A + 1 + 3*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 ); a22 = *(A + 2 + 2*4 ); a23 = *(A + 2 + 3*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 ); a32 = *(A + 3 + 2*4 ); a33 = *(A + 3 + 3*4 );
	            //b00 = *(B + 0 + 0*4 ); b01 = *(B +  0 + 1*4 ); b02 = *(B + 0 + 2*4 ); b03 = *(B + 0 + 3*4 );
	            //b10 = *(B + 1 + 0*4 ); b11 = *(B +  1 + 1*4 ); b12 = *(B + 1 + 2*4 ); b13 = *(B + 1 + 3*4 );
	            //b20 = *(B + 2 + 0*4 ); b21 = *(B +  2 + 1*4 ); b22 = *(B + 2 + 2*4 ); b23 = *(B + 2 + 3*4 );
	            //b30 = *(B + 3 + 0*4 ); b31 = *(B +  3 + 1*4 ); b32 = *(B + 3 + 2*4 ); b33 = *(B + 3 + 3*4 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c02 += a00 * b20; c02 += a01 * b21; c02 += a02 * b22; c02 += a03 * b23; 
	            //c03 += a00 * b30; c03 += a01 * b31; c03 += a02 * b32; c03 += a03 * b33; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c12 += a10 * b20; c12 += a11 * b21; c12 += a12 * b22; c12 += a13 * b23; 
	            //c13 += a10 * b30; c13 += a11 * b31; c13 += a12 * b32; c13 += a13 * b33; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c21 += a20 * b10; c21 += a21 * b11; c21 += a22 * b12; c21 += a23 * b13; 
	            //c22 += a20 * b20; c22 += a21 * b21; c22 += a22 * b22; c22 += a23 * b23; 
	            //c23 += a20 * b30; c23 += a21 * b31; c23 += a22 * b32; c23 += a23 * b33; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //c31 += a30 * b10; c31 += a31 * b11; c31 += a32 * b12; c31 += a33 * b13; 
	            //c32 += a30 * b20; c32 += a31 * b21; c32 += a32 * b22; c32 += a33 * b23; 
	            //c33 += a30 * b30; c33 += a31 * b31; c33 += a32 * b32; c33 += a33 * b33; 
	            //a00 = *(A + 0 + 4*4 ); a01 = *(A +  0 + 5*4 ); a02 = *(A + 0 + 6*4 ); a03 = *(A + 0 + 7*4 );
	            //a10 = *(A + 1 + 4*4 ); a11 = *(A +  1 + 5*4 ); a12 = *(A + 1 + 6*4 ); a13 = *(A + 1 + 7*4 );
	            //a20 = *(A + 2 + 4*4 ); a21 = *(A +  2 + 5*4 ); a22 = *(A + 2 + 6*4 ); a23 = *(A + 2 + 7*4 );
	            //a30 = *(A + 3 + 4*4 ); a31 = *(A +  3 + 5*4 ); a32 = *(A + 3 + 6*4 ); a33 = *(A + 3 + 7*4 );
	            //b00 = *(B + 0 + 4*4 ); b01 = *(B +  0 + 5*4 ); b02 = *(B + 0 + 6*4 ); b03 = *(B + 0 + 7*4 );
	            //b10 = *(B + 1 + 4*4 ); b11 = *(B +  1 + 5*4 ); b12 = *(B + 1 + 6*4 ); b13 = *(B + 1 + 7*4 );
	            //b20 = *(B + 2 + 4*4 ); b21 = *(B +  2 + 5*4 ); b22 = *(B + 2 + 6*4 ); b23 = *(B + 2 + 7*4 );
	            //b30 = *(B + 3 + 4*4 ); b31 = *(B +  3 + 5*4 ); b32 = *(B + 3 + 6*4 ); b33 = *(B + 3 + 7*4 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c02 += a00 * b20; c02 += a01 * b21; c02 += a02 * b22; c02 += a03 * b23; 
	            //c03 += a00 * b30; c03 += a01 * b31; c03 += a02 * b32; c03 += a03 * b33; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c12 += a10 * b20; c12 += a11 * b21; c12 += a12 * b22; c12 += a13 * b23; 
	            //c13 += a10 * b30; c13 += a11 * b31; c13 += a12 * b32; c13 += a13 * b33; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c21 += a20 * b10; c21 += a21 * b11; c21 += a22 * b12; c21 += a23 * b13; 
	            //c22 += a20 * b20; c22 += a21 * b21; c22 += a22 * b22; c22 += a23 * b23; 
	            //c23 += a20 * b30; c23 += a21 * b31; c23 += a22 * b32; c23 += a23 * b33; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //c31 += a30 * b10; c31 += a31 * b11; c31 += a32 * b12; c31 += a33 * b13; 
	            //c32 += a30 * b20; c32 += a31 * b21; c32 += a32 * b22; c32 += a33 * b23; 
	            //c33 += a30 * b30; c33 += a31 * b31; c33 += a32 * b32; c33 += a33 * b33; 
	            ////A+=32;
	            ////B+=32;

	            //c00 = a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 = a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c02 = a00 * b20; c02 += a01 * b21; c02 += a02 * b22; c02 += a03 * b23; 
	            //c12 = a10 * b20; c12 += a11 * b21; c12 += a12 * b22; c12 += a13 * b23; 
                    //                                                    
	            //c20 = a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c30 = a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //c22 = a20 * b20; c22 += a21 * b21; c22 += a22 * b22; c22 += a23 * b23; 
	            //c32 = a30 * b20; c32 += a31 * b21; c32 += a32 * b22; c32 += a33 * b23; 
                    //                                                    
	            //c01 = a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c11 = a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c03 = a00 * b30; c03 += a01 * b31; c03 += a02 * b32; c03 += a03 * b33; 
	            //c13 = a10 * b30; c13 += a11 * b31; c13 += a12 * b32; c13 += a13 * b33; 
                    //                                                    
	            //c21 = a20 * b10; c21 += a21 * b11; c21 += a22 * b12; c21 += a23 * b13; 
	            //c31 = a30 * b10; c31 += a31 * b11; c31 += a32 * b12; c31 += a33 * b13; 
	            //c23 = a20 * b30; c23 += a21 * b31; c23 += a22 * b32; c23 += a23 * b33; 
	            //c33 = a30 * b30; c33 += a31 * b31; c33 += a32 * b32; c33 += a33 * b33; 

	            //c00 =b00;// a00;
	            //c10 =b10;// a10;
	            //c02 =b20;// a20;
	            //c12 =b30;// a30;
	            //c20 =b01;// a01;
	            //c30 =b11;// a11;
	            //c22 =b21;// a21;
	            //c32 =b31;// a31;
	            //c01 =b02;// a02;
	            //c11 =b12;// a12;
	            //c03 =b22;// a22;
	            //c13 =b32;// a32;
	            //c21 =b03;// a03;
	            //c31 =b13;// a13;
	            //c23 =b23;// a23;
	            //c33 =b33;// a33;



	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  64*8(%[a])\n\t"
	                //"prefetcht0  64*8(%[b])\n\t"
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	                "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t"
	                "vmovapd         0*8(%[b]), %%ymm4 \n\t"
	                "vmovapd         4*8(%[b]), %%ymm5 \n\t"
	                "\n\t"
	                "vshufpd    $0x00, %%ymm4, %%ymm4, %%ymm8 \n\t"
	                "vshufpd    $0x0f, %%ymm4, %%ymm4, %%ymm9 \n\t"
	                "vshufpd    $0x00, %%ymm5, %%ymm5, %%ymm10\n\t"
	                "vshufpd    $0x0f, %%ymm5, %%ymm5, %%ymm11\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm8 , %%ymm12\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm8 , %%ymm13\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm9 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm9 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm10, %%ymm12\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm10, %%ymm13\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm11, %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "vbroadcastf128  8*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128 10*8(%[a]), %%ymm1 \n\t"
	                "vbroadcastf128 12*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastf128 14*8(%[a]), %%ymm3 \n\t"
	                "vmovapd         8*8(%[b]), %%ymm6 \n\t"
	                "vmovapd        12*8(%[b]), %%ymm7 \n\t"
	                "\n\t"
	                "vshufpd    $0x00, %%ymm6, %%ymm6, %%ymm8 \n\t"
	                "vshufpd    $0x0f, %%ymm6, %%ymm6, %%ymm9 \n\t"
	                "vshufpd    $0x00, %%ymm7, %%ymm7, %%ymm10\n\t"
	                "vshufpd    $0x0f, %%ymm7, %%ymm7, %%ymm11\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm8 , %%ymm12\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm8 , %%ymm13\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm9 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm9 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm10, %%ymm12\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm10, %%ymm13\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm11, %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "vbroadcastf128 16*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128 18*8(%[a]), %%ymm1 \n\t"
	                "vbroadcastf128 20*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastf128 22*8(%[a]), %%ymm3 \n\t"
	                "vmovapd        16*8(%[b]), %%ymm4 \n\t"
	                "vmovapd        20*8(%[b]), %%ymm5 \n\t"
	                "\n\t"
	                "vshufpd    $0x00, %%ymm4, %%ymm4, %%ymm8 \n\t"
	                "vshufpd    $0x0f, %%ymm4, %%ymm4, %%ymm9 \n\t"
	                "vshufpd    $0x00, %%ymm5, %%ymm5, %%ymm10\n\t"
	                "vshufpd    $0x0f, %%ymm5, %%ymm5, %%ymm11\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm8 , %%ymm12\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm8 , %%ymm13\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm9 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm9 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm10, %%ymm12\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm10, %%ymm13\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm11, %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "vbroadcastf128 24*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128 26*8(%[a]), %%ymm1 \n\t"
	                "vbroadcastf128 28*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastf128 30*8(%[a]), %%ymm3 \n\t"
	                "vmovapd        24*8(%[b]), %%ymm6 \n\t"
	                "vmovapd        28*8(%[b]), %%ymm7 \n\t"
	                "\n\t"
	                "vshufpd    $0x00, %%ymm6, %%ymm6, %%ymm8 \n\t"
	                "vshufpd    $0x0f, %%ymm6, %%ymm6, %%ymm9 \n\t"
	                "vshufpd    $0x00, %%ymm7, %%ymm7, %%ymm10\n\t"
	                "vshufpd    $0x0f, %%ymm7, %%ymm7, %%ymm11\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm8 , %%ymm12\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm8 , %%ymm13\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm9 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm9 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm10, %%ymm12\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm10, %%ymm13\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm11, %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $32*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	            //printf("%x\n",(uint64_t)cc);
	            //// Dump
	            //__asm__ __volatile__ (
	            //    "\n\t"
	            //    "vmovupd  %%ymm12, 0*8(%[c]) \n\t"
	            //    "vmovupd  %%ymm13, 4*8(%[c]) \n\t"
	            //    "vmovupd  %%ymm14, 8*8(%[c]) \n\t"
	            //    "vmovupd  %%ymm15,12*8(%[c]) \n\t"
	            //    "\n\t"
	            //    :[c]"=r"(cc)
	            //:);
	            //printf("%x\n",(uint64_t)cc);

	            //printf("n-------------------------------\n");
	            //printf("Dump n4=%d m4=%d k8=%d\n",n4,m4,k8);
	            //printf("n-------------------------------\n");
	            //printf("Unroll Expansion\n");
	            //printf("c00=%G, c10=%G, c02=%G, c12=%G \n",c00,c10,c02,c12);
	            //printf("c20=%G, c30=%G, c22=%G, c32=%G \n",c20,c30,c22,c32);
	            //printf("c01=%G, c11=%G, c03=%G, c13=%G \n",c01,c11,c03,c13);
	            //printf("c21=%G, c31=%G, c23=%G, c33=%G \n",c21,c31,c23,c33);
	            //printf("AVX\n");
	            //printf("c00=%G, c10=%G, c02=%G, c12=%G \n",*(cc+ 0),*(cc+ 1),*(cc+ 2),*(cc+ 3));
	            //printf("c20=%G, c30=%G, c22=%G, c32=%G \n",*(cc+ 4),*(cc+ 5),*(cc+ 6),*(cc+ 7));
	            //printf("c01=%G, c11=%G, c03=%G, c13=%G \n",*(cc+ 8),*(cc+ 9),*(cc+10),*(cc+11));
	            //printf("c21=%G, c31=%G, c23=%G, c33=%G \n",*(cc+12),*(cc+13),*(cc+14),*(cc+15));
	          }
	        }
	        if( K & 4 ){
/*
	            a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 ); a02 = *(A + 0 + 2*4 ); a03 = *(A + 0 + 3*4 );
	            a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 ); a12 = *(A + 1 + 2*4 ); a13 = *(A + 1 + 3*4 );
	            a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 ); a22 = *(A + 2 + 2*4 ); a23 = *(A + 2 + 3*4 );
	            a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 ); a32 = *(A + 3 + 2*4 ); a33 = *(A + 3 + 3*4 );
	            b00 = *(B + 0 + 0*4 ); b01 = *(B +  0 + 1*4 ); b02 = *(B + 0 + 2*4 ); b03 = *(B + 0 + 3*4 );
	            b10 = *(B + 1 + 0*4 ); b11 = *(B +  1 + 1*4 ); b12 = *(B + 1 + 2*4 ); b13 = *(B + 1 + 3*4 );
	            b20 = *(B + 2 + 0*4 ); b21 = *(B +  2 + 1*4 ); b22 = *(B + 2 + 2*4 ); b23 = *(B + 2 + 3*4 );
	            b30 = *(B + 3 + 0*4 ); b31 = *(B +  3 + 1*4 ); b32 = *(B + 3 + 2*4 ); b33 = *(B + 3 + 3*4 );
	            c00 = a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            c01 = a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            c02 = a00 * b20; c02 += a01 * b21; c02 += a02 * b22; c02 += a03 * b23; 
	            c03 = a00 * b30; c03 += a01 * b31; c03 += a02 * b32; c03 += a03 * b33; 
	            c10 = a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            c11 = a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            c12 = a10 * b20; c12 += a11 * b21; c12 += a12 * b22; c12 += a13 * b23; 
	            c13 = a10 * b30; c13 += a11 * b31; c13 += a12 * b32; c13 += a13 * b33; 
	            c20 = a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            c21 = a20 * b10; c21 += a21 * b11; c21 += a22 * b12; c21 += a23 * b13; 
	            c22 = a20 * b20; c22 += a21 * b21; c22 += a22 * b22; c22 += a23 * b23; 
	            c23 = a20 * b30; c23 += a21 * b31; c23 += a22 * b32; c23 += a23 * b33; 
	            c30 = a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            c31 = a30 * b10; c31 += a31 * b11; c31 += a32 * b12; c31 += a33 * b13; 
	            c32 = a30 * b20; c32 += a31 * b21; c32 += a32 * b22; c32 += a33 * b23; 
	            c33 = a30 * b30; c33 += a31 * b31; c33 += a32 * b32; c33 += a33 * b33; 
	            //A+=16;
	            //B+=16;
*/
	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	                "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t"
	                "vmovapd         0*8(%[b]), %%ymm4 \n\t"
	                "vmovapd         4*8(%[b]), %%ymm5 \n\t"
	                "\n\t"
	                "vshufpd    $0x00, %%ymm4, %%ymm4, %%ymm8 \n\t"
	                "vshufpd    $0x0f, %%ymm4, %%ymm4, %%ymm9 \n\t"
	                "vshufpd    $0x00, %%ymm5, %%ymm5, %%ymm10\n\t"
	                "vshufpd    $0x0f, %%ymm5, %%ymm5, %%ymm11\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm8 , %%ymm12\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm8 , %%ymm13\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm9 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm9 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm10, %%ymm12\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm10, %%ymm13\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm11, %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "vbroadcastf128  8*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128 10*8(%[a]), %%ymm1 \n\t"
	                "vbroadcastf128 12*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastf128 14*8(%[a]), %%ymm3 \n\t"
	                "vmovapd         8*8(%[b]), %%ymm6 \n\t"
	                "vmovapd        12*8(%[b]), %%ymm7 \n\t"
	                "\n\t"
	                "vshufpd    $0x00, %%ymm6, %%ymm6, %%ymm8 \n\t"
	                "vshufpd    $0x0f, %%ymm6, %%ymm6, %%ymm9 \n\t"
	                "vshufpd    $0x00, %%ymm7, %%ymm7, %%ymm10\n\t"
	                "vshufpd    $0x0f, %%ymm7, %%ymm7, %%ymm11\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm8 , %%ymm12\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm8 , %%ymm13\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm9 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm9 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm10, %%ymm12\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm10, %%ymm13\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm11, %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);
	           
	        }
	        if( K & 2 ){
/*
	            a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 );
	            a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 );
	            a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 );
	            a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 );
	            b00 = *(B + 0 + 0*4 ); b01 = *(B +  0 + 1*4 );
	            b10 = *(B + 1 + 0*4 ); b11 = *(B +  1 + 1*4 );
	            b20 = *(B + 2 + 0*4 ); b21 = *(B +  2 + 1*4 );
	            b30 = *(B + 3 + 0*4 ); b31 = *(B +  3 + 1*4 );
	            c00 += a00 * b00; c00 += a01 * b01;
	            c01 += a00 * b10; c01 += a01 * b11;
	            c02 += a00 * b20; c02 += a01 * b21;
	            c03 += a00 * b30; c03 += a01 * b31;
	            c10 += a10 * b00; c10 += a11 * b01;
	            c11 += a10 * b10; c11 += a11 * b11;
	            c12 += a10 * b20; c12 += a11 * b21;
	            c13 += a10 * b30; c13 += a11 * b31;
	            c20 += a20 * b00; c20 += a21 * b01;
	            c21 += a20 * b10; c21 += a21 * b11;
	            c22 += a20 * b20; c22 += a21 * b21;
	            c23 += a20 * b30; c23 += a21 * b31;
	            c30 += a30 * b00; c30 += a31 * b01;
	            c31 += a30 * b10; c31 += a31 * b11;
	            c32 += a30 * b20; c32 += a31 * b21;
	            c33 += a30 * b30; c33 += a31 * b31;
	            A+=8;
	            B+=8;
*/
	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	                "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t"
	                "vmovapd         0*8(%[b]), %%ymm4 \n\t"
	                "vmovapd         4*8(%[b]), %%ymm5 \n\t"
	                "\n\t"
	                "vshufpd    $0x00, %%ymm4, %%ymm4, %%ymm8 \n\t"
	                "vshufpd    $0x0f, %%ymm4, %%ymm4, %%ymm9 \n\t"
	                "vshufpd    $0x00, %%ymm5, %%ymm5, %%ymm10\n\t"
	                "vshufpd    $0x0f, %%ymm5, %%ymm5, %%ymm11\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm8 , %%ymm12\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm8 , %%ymm13\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm9 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm9 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm10, %%ymm12\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm10, %%ymm13\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm11, %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){
/*
	            a00 = *(A + 0 + 0*4 );
	            a10 = *(A + 1 + 0*4 );
	            a20 = *(A + 2 + 0*4 );
	            a30 = *(A + 3 + 0*4 );
	            b00 = *(B + 0 + 0*4 );
	            b10 = *(B + 1 + 0*4 );
	            b20 = *(B + 2 + 0*4 );
	            b30 = *(B + 3 + 0*4 );
	            c00 += a00 * b00;
	            c01 += a00 * b10;
	            c02 += a00 * b20;
	            c03 += a00 * b30;
	            c10 += a10 * b00;
	            c11 += a10 * b10;
	            c12 += a10 * b20;
	            c13 += a10 * b30;
	            c20 += a20 * b00;
	            c21 += a20 * b10;
	            c22 += a20 * b20;
	            c23 += a20 * b30;
	            c30 += a30 * b00;
	            c31 += a30 * b10;
	            c32 += a30 * b20;
	            c33 += a30 * b30;
	            A+=4;
	            B+=4;
*/

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	                "vmovapd         0*8(%[b]), %%ymm4 \n\t"
	                "\n\t"
	                "vshufpd    $0x00, %%ymm4, %%ymm4, %%ymm8 \n\t"
	                "vshufpd    $0x0f, %%ymm4, %%ymm4, %%ymm9 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm8 , %%ymm12\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm8 , %%ymm13\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm9 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm9 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
/*
	        *(C+0+0*ldc) += alpha*c00;
	        *(C+0+1*ldc) += alpha*c01;
	        *(C+0+2*ldc) += alpha*c02;
	        *(C+0+3*ldc) += alpha*c03;
	        *(C+1+0*ldc) += alpha*c10;
	        *(C+1+1*ldc) += alpha*c11;
	        *(C+1+2*ldc) += alpha*c12;
	        *(C+1+3*ldc) += alpha*c13;
	        *(C+2+0*ldc) += alpha*c20;
	        *(C+2+1*ldc) += alpha*c21;
	        *(C+2+2*ldc) += alpha*c22;
	        *(C+2+3*ldc) += alpha*c23;
	        *(C+3+0*ldc) += alpha*c30;
	        *(C+3+1*ldc) += alpha*c31;
	        *(C+3+2*ldc) += alpha*c32;
	        *(C+3+3*ldc) += alpha*c33;
	        B = B - 4*K;//N*K
	        C+=4;
*/


	        //printf("C =0x%x\n",(uint64_t)C);
	        //printf("c0=0x%x\n",(uint64_t)c0);
	        //printf("c1=0x%x\n",(uint64_t)c1);

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vbroadcastsd %[alpha], %%ymm0 \n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vperm2f128 $0x20, %%ymm13, %%ymm12, %%ymm8 \n\t" // src2, src1, dest
	            "vperm2f128 $0x31, %%ymm13, %%ymm12, %%ymm10\n\t"
	            "vperm2f128 $0x20, %%ymm15, %%ymm14, %%ymm9 \n\t"
	            "vperm2f128 $0x31, %%ymm15, %%ymm14, %%ymm11\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%ymm0, %%ymm8 \n\t"
	            "vfmadd213pd 0*8(%[c1]        ), %%ymm0, %%ymm9 \n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]), %%ymm0, %%ymm10\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]), %%ymm0, %%ymm11\n\t"
	            "\n\t"
	            "vmovupd  %%ymm8 , 0*8(%[c0]        )\n\t"
	            "vmovupd  %%ymm9 , 0*8(%[c1]        )\n\t"
	            "vmovupd  %%ymm10, 0*8(%[c0],%[ldc2])\n\t"
	            "vmovupd  %%ymm11, 0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	            ////printf("%x\n",(uint64_t)cc);
	            ////// Dump
	            //__asm__ __volatile__ (
	            //    "\n\t"
	            //    "vmovupd  %%ymm8 , 0*8(%[c]) \n\t"
	            //    "vmovupd  %%ymm9 , 4*8(%[c]) \n\t"
	            //    "vmovupd  %%ymm10, 8*8(%[c]) \n\t"
	            //    "vmovupd  %%ymm11,12*8(%[c]) \n\t"
	            //    "\n\t"
	            //    :[c]"=r"(cc)
	            //:);
	            ////printf("%x\n",(uint64_t)cc);

	            //printf("n-------------------------------\n");
	            //printf("Dump n4=%d m4=%d\n",n4,m4);
	            //printf("n-------------------------------\n");
	            //printf("Register\n");
	            //printf("c00=%G, c10=%G, c20=%G, c30=%G \n",*(cc+ 0),*(cc+ 1),*(cc+ 2),*(cc+ 3));
	            //printf("c01=%G, c11=%G, c21=%G, c31=%G \n",*(cc+ 4),*(cc+ 5),*(cc+ 6),*(cc+ 7));
	            //printf("c02=%G, c12=%G, c22=%G, c32=%G \n",*(cc+ 8),*(cc+ 9),*(cc+10),*(cc+11));
	            //printf("c03=%G, c13=%G, c23=%G, c33=%G \n",*(cc+12),*(cc+13),*(cc+14),*(cc+15));
	            //printf("Memory\n");
	            //printf("c00=%G, c10=%G, c20=%G, c30=%G \n",*(C+ 0+ 0*ldc),*(C+ 1+ 0*ldc),*(C+ 2+ 0*ldc),*(C+ 3+ 0*ldc));
	            //printf("c01=%G, c11=%G, c21=%G, c31=%G \n",*(C+ 0+ 1*ldc),*(C+ 1+ 1*ldc),*(C+ 2+ 1*ldc),*(C+ 3+ 1*ldc));
	            //printf("c02=%G, c12=%G, c22=%G, c32=%G \n",*(C+ 0+ 2*ldc),*(C+ 1+ 2*ldc),*(C+ 2+ 2*ldc),*(C+ 3+ 2*ldc));
	            //printf("c03=%G, c13=%G, c23=%G, c33=%G \n",*(C+ 0+ 3*ldc),*(C+ 1+ 3*ldc),*(C+ 2+ 3*ldc),*(C+ 3+ 3*ldc));
	
	        B = B - 4*K;//N*K
	        //C+=4;

	      }

	    }
	    if( M & 2 ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;

	        __asm__ __volatile__ (
	            "\n\t"
	            //"prefetcht1   32*8(%[a])\n\t"
	            //"prefetcht0   64*8(%[b])\n\t"
	            "prefetchw     0*8(%[c0]        )\n\t"
	            "prefetchw     0*8(%[c1]        )\n\t"
	            "prefetchw     0*8(%[c0],%[ldc2])\n\t"
	            "prefetchw     0*8(%[c1],%[ldc2])\n\t"
	        ::[a]"r"(A),[b]"r"(B),[c0]"r"(c0),[c1]"r"(c1),[ldc2]"r"(ldc2)
	        );


	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);


	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 ); a02 = *(A + 0 + 2*2 ); a03 = *(A + 0 + 3*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 ); a12 = *(A + 1 + 2*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*4 ); b01 = *(B +  0 + 1*4 ); b02 = *(B + 0 + 2*4 ); b03 = *(B + 0 + 3*4 );
	            //b10 = *(B + 1 + 0*4 ); b11 = *(B +  1 + 1*4 ); b12 = *(B + 1 + 2*4 ); b13 = *(B + 1 + 3*4 );
	            //b20 = *(B + 2 + 0*4 ); b21 = *(B +  2 + 1*4 ); b22 = *(B + 2 + 2*4 ); b23 = *(B + 2 + 3*4 );
	            //b30 = *(B + 3 + 0*4 ); b31 = *(B +  3 + 1*4 ); b32 = *(B + 3 + 2*4 ); b33 = *(B + 3 + 3*4 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c02 += a00 * b20; c02 += a01 * b21; c02 += a02 * b22; c02 += a03 * b23; 
	            //c03 += a00 * b30; c03 += a01 * b31; c03 += a02 * b32; c03 += a03 * b33; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c12 += a10 * b20; c12 += a11 * b21; c12 += a12 * b22; c12 += a13 * b23; 
	            //c13 += a10 * b30; c13 += a11 * b31; c13 += a12 * b32; c13 += a13 * b33; 
	            //a00 = *(A + 0 + 4*2 ); a01 = *(A +  0 + 5*2 ); a02 = *(A + 0 + 6*2 ); a03 = *(A + 0 + 7*2 );
	            //a10 = *(A + 1 + 4*2 ); a11 = *(A +  1 + 5*2 ); a12 = *(A + 1 + 6*2 ); a13 = *(A + 1 + 7*2 );
	            //b00 = *(B + 0 + 4*4 ); b01 = *(B +  0 + 5*4 ); b02 = *(B + 0 + 6*4 ); b03 = *(B + 0 + 7*4 );
	            //b10 = *(B + 1 + 4*4 ); b11 = *(B +  1 + 5*4 ); b12 = *(B + 1 + 6*4 ); b13 = *(B + 1 + 7*4 );
	            //b20 = *(B + 2 + 4*4 ); b21 = *(B +  2 + 5*4 ); b22 = *(B + 2 + 6*4 ); b23 = *(B + 2 + 7*4 );
	            //b30 = *(B + 3 + 4*4 ); b31 = *(B +  3 + 5*4 ); b32 = *(B + 3 + 6*4 ); b33 = *(B + 3 + 7*4 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c02 += a00 * b20; c02 += a01 * b21; c02 += a02 * b22; c02 += a03 * b23; 
	            //c03 += a00 * b30; c03 += a01 * b31; c03 += a02 * b32; c03 += a03 * b33; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c12 += a10 * b20; c12 += a11 * b21; c12 += a12 * b22; c12 += a13 * b23; 
	            //c13 += a10 * b30; c13 += a11 * b31; c13 += a12 * b32; c13 += a13 * b33; 
	            //A+=16;
	            //B+=32;

	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c02 += a00 * b20; c02 += a01 * b21; c02 += a02 * b22; c02 += a03 * b23; 
	            //c12 += a10 * b20; c12 += a11 * b21; c12 += a12 * b22; c12 += a13 * b23; 

	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c03 += a00 * b30; c03 += a01 * b31; c03 += a02 * b32; c03 += a03 * b33; 
	            //c13 += a10 * b30; c13 += a11 * b31; c13 += a12 * b32; c13 += a13 * b33; 

	            __asm__ __volatile__ (
	                "\n\t"
	                "prefetcht0  64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	                "vmovapd         0*8(%[b]), %%ymm4 \n\t"
	                "vmovapd         4*8(%[b]), %%ymm5 \n\t"
	                "\n\t"
	                "vshufpd    $0x0f, %%ymm4, %%ymm4, %%ymm10\n\t"
	                "vshufpd    $0x0f, %%ymm5, %%ymm5, %%ymm11\n\t"
	                "vshufpd    $0x00, %%ymm4, %%ymm4, %%ymm4 \n\t"
	                "vshufpd    $0x00, %%ymm5, %%ymm5, %%ymm5 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm10, %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t"
	                "vmovapd         8*8(%[b]), %%ymm6 \n\t"
	                "vmovapd        12*8(%[b]), %%ymm7 \n\t"
	                "\n\t"
	                "vshufpd    $0x0f, %%ymm6, %%ymm6, %%ymm12\n\t"
	                "vshufpd    $0x0f, %%ymm7, %%ymm7, %%ymm13\n\t"
	                "vshufpd    $0x00, %%ymm6, %%ymm6, %%ymm6 \n\t"
	                "vshufpd    $0x00, %%ymm7, %%ymm7, %%ymm7 \n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm12, %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm13, %%ymm15\n\t"
	                "\n\t"
	                "vbroadcastf128  8*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128 10*8(%[a]), %%ymm1 \n\t"
	                "vmovapd        16*8(%[b]), %%ymm4 \n\t"
	                "vmovapd        20*8(%[b]), %%ymm5 \n\t"
	                "\n\t"
	                "vshufpd    $0x0f, %%ymm4, %%ymm4, %%ymm10\n\t"
	                "vshufpd    $0x0f, %%ymm5, %%ymm5, %%ymm11\n\t"
	                "vshufpd    $0x00, %%ymm4, %%ymm4, %%ymm4 \n\t"
	                "vshufpd    $0x00, %%ymm5, %%ymm5, %%ymm5 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm10, %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "vbroadcastf128 12*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastf128 14*8(%[a]), %%ymm3 \n\t"
	                "vmovapd        24*8(%[b]), %%ymm6 \n\t"
	                "vmovapd        28*8(%[b]), %%ymm7 \n\t"
	                "\n\t"
	                "vshufpd    $0x0f, %%ymm6, %%ymm6, %%ymm12\n\t"
	                "vshufpd    $0x0f, %%ymm7, %%ymm7, %%ymm13\n\t"
	                "vshufpd    $0x00, %%ymm6, %%ymm6, %%ymm6 \n\t"
	                "vshufpd    $0x00, %%ymm7, %%ymm7, %%ymm7 \n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm12, %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm13, %%ymm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $32*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 ); a02 = *(A + 0 + 2*2 ); a03 = *(A + 0 + 3*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 ); a12 = *(A + 1 + 2*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*4 ); b01 = *(B +  0 + 1*4 ); b02 = *(B + 0 + 2*4 ); b03 = *(B + 0 + 3*4 );
	            //b10 = *(B + 1 + 0*4 ); b11 = *(B +  1 + 1*4 ); b12 = *(B + 1 + 2*4 ); b13 = *(B + 1 + 3*4 );
	            //b20 = *(B + 2 + 0*4 ); b21 = *(B +  2 + 1*4 ); b22 = *(B + 2 + 2*4 ); b23 = *(B + 2 + 3*4 );
	            //b30 = *(B + 3 + 0*4 ); b31 = *(B +  3 + 1*4 ); b32 = *(B + 3 + 2*4 ); b33 = *(B + 3 + 3*4 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c02 += a00 * b20; c02 += a01 * b21; c02 += a02 * b22; c02 += a03 * b23; 
	            //c03 += a00 * b30; c03 += a01 * b31; c03 += a02 * b32; c03 += a03 * b33; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c12 += a10 * b20; c12 += a11 * b21; c12 += a12 * b22; c12 += a13 * b23; 
	            //c13 += a10 * b30; c13 += a11 * b31; c13 += a12 * b32; c13 += a13 * b33; 
	            //A+=8;
	            //B+=16;

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	                "vmovapd         0*8(%[b]), %%ymm4 \n\t"
	                "vmovapd         4*8(%[b]), %%ymm5 \n\t"
	                "\n\t"
	                "vshufpd    $0x0f, %%ymm4, %%ymm4, %%ymm10\n\t"
	                "vshufpd    $0x0f, %%ymm5, %%ymm5, %%ymm11\n\t"
	                "vshufpd    $0x00, %%ymm4, %%ymm4, %%ymm4 \n\t"
	                "vshufpd    $0x00, %%ymm5, %%ymm5, %%ymm5 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm10, %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "vbroadcastf128  4*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastf128  6*8(%[a]), %%ymm3 \n\t"
	                "vmovapd         8*8(%[b]), %%ymm6 \n\t"
	                "vmovapd        12*8(%[b]), %%ymm7 \n\t"
	                "\n\t"
	                "vshufpd    $0x0f, %%ymm6, %%ymm6, %%ymm12\n\t"
	                "vshufpd    $0x0f, %%ymm7, %%ymm7, %%ymm13\n\t"
	                "vshufpd    $0x00, %%ymm6, %%ymm6, %%ymm6 \n\t"
	                "vshufpd    $0x00, %%ymm7, %%ymm7, %%ymm7 \n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm12, %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm13, %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 );
	            //b00 = *(B + 0 + 0*4 ); b01 = *(B +  0 + 1*4 );
	            //b10 = *(B + 1 + 0*4 ); b11 = *(B +  1 + 1*4 );
	            //b20 = *(B + 2 + 0*4 ); b21 = *(B +  2 + 1*4 );
	            //b30 = *(B + 3 + 0*4 ); b31 = *(B +  3 + 1*4 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c01 += a00 * b10; c01 += a01 * b11;
	            //c02 += a00 * b20; c02 += a01 * b21;
	            //c03 += a00 * b30; c03 += a01 * b31;
	            //c10 += a10 * b00; c10 += a11 * b01;
	            //c11 += a10 * b10; c11 += a11 * b11;
	            //c12 += a10 * b20; c12 += a11 * b21;
	            //c13 += a10 * b30; c13 += a11 * b31;
	            //A+=4;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastf128  2*8(%[a]), %%ymm1 \n\t"
	                "vmovapd         0*8(%[b]), %%ymm4 \n\t"
	                "vmovapd         4*8(%[b]), %%ymm5 \n\t"
	                "\n\t"
	                "vshufpd    $0x0f, %%ymm4, %%ymm4, %%ymm10\n\t"
	                "vshufpd    $0x0f, %%ymm5, %%ymm5, %%ymm11\n\t"
	                "vshufpd    $0x00, %%ymm4, %%ymm4, %%ymm4 \n\t"
	                "vshufpd    $0x00, %%ymm5, %%ymm5, %%ymm5 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm10, %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm11, %%ymm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*2 );
	            //a10 = *(A + 1 + 0*2 );
	            //b00 = *(B + 0 + 0*4 );
	            //b10 = *(B + 1 + 0*4 );
	            //b20 = *(B + 2 + 0*4 );
	            //b30 = *(B + 3 + 0*4 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b10;
	            //c02 += a00 * b20;
	            //c03 += a00 * b30;
	            //c10 += a10 * b00;
	            //c11 += a10 * b10;
	            //c12 += a10 * b20;
	            //c13 += a10 * b30;
	            //A+=2;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastf128  0*8(%[a]), %%ymm0 \n\t"
	                "vmovapd         0*8(%[b]), %%ymm4 \n\t"
	                "\n\t"
	                "vshufpd    $0x0f, %%ymm4, %%ymm4, %%ymm10\n\t"
	                "vshufpd    $0x00, %%ymm4, %%ymm4, %%ymm4 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm10, %%ymm15\n\t"
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;
	        //*(C+0+2*ldc) += alpha*c02;
	        //*(C+0+3*ldc) += alpha*c03;
	        //*(C+1+0*ldc) += alpha*c10;
	        //*(C+1+1*ldc) += alpha*c11;
	        //*(C+1+2*ldc) += alpha*c12;
	        //*(C+1+3*ldc) += alpha*c13;

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vbroadcastsd %[alpha], %%ymm0 \n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vperm2f128 $0x01, %%ymm14, %%ymm14, %%ymm12\n\t" // ymm14[00,10], ymm12[02,12]
	            "vperm2f128 $0x01, %%ymm15, %%ymm15, %%ymm13\n\t" // ymm15[01,11], ymm13[03,13]
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]        ), %%xmm0, %%xmm14\n\t"
	            "vfmadd213pd 0*8(%[c1]        ), %%xmm0, %%xmm15\n\t"
	            "vfmadd213pd 0*8(%[c0],%[ldc2]), %%xmm0, %%xmm12\n\t"
	            "vfmadd213pd 0*8(%[c1],%[ldc2]), %%xmm0, %%xmm13\n\t"
	            "\n\t"
	            "vmovupd  %%xmm14, 0*8(%[c0]        )\n\t"
	            "vmovupd  %%xmm15, 0*8(%[c1]        )\n\t"
	            "vmovupd  %%xmm12, 0*8(%[c0],%[ldc2])\n\t"
	            "vmovupd  %%xmm13, 0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 4*K;//N*K
	        //C+=2;
	        //c0+=2;
	        //c1+=2;

	    }
	    if( M & 1 ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;
/*
	        __asm__ __volatile__ (
	            "\n\t"
	            //"prefetcht1   32*8(%[a])\n\t"
	            //"prefetcht0   64*8(%[b])\n\t"
	            "prefetchw     4*8(%[c0]        )\n\t"
	            "prefetchw     4*8(%[c1]        )\n\t"
	            "prefetchw     4*8(%[c0],%[ldc2])\n\t"
	            "prefetchw     4*8(%[c1],%[ldc2])\n\t"
	        ::[a]"r"(A),[b]"r"(B),[c0]"r"(c0),[c1]"r"(c1),[ldc2]"r"(ldc2)
	        );
*/

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 ); a02 = *(A + 0 + 2*1 ); a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*4 ); b01 = *(B +  0 + 1*4 ); b02 = *(B + 0 + 2*4 ); b03 = *(B + 0 + 3*4 );
	            //b10 = *(B + 1 + 0*4 ); b11 = *(B +  1 + 1*4 ); b12 = *(B + 1 + 2*4 ); b13 = *(B + 1 + 3*4 );
	            //b20 = *(B + 2 + 0*4 ); b21 = *(B +  2 + 1*4 ); b22 = *(B + 2 + 2*4 ); b23 = *(B + 2 + 3*4 );
	            //b30 = *(B + 3 + 0*4 ); b31 = *(B +  3 + 1*4 ); b32 = *(B + 3 + 2*4 ); b33 = *(B + 3 + 3*4 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c02 += a00 * b20; c02 += a01 * b21; c02 += a02 * b22; c02 += a03 * b23; 
	            //c03 += a00 * b30; c03 += a01 * b31; c03 += a02 * b32; c03 += a03 * b33; 
	            //a00 = *(A + 0 + 4*1 ); a01 = *(A +  0 + 5*1 ); a02 = *(A + 0 + 6*1 ); a03 = *(A + 0 + 7*1 );
	            //b00 = *(B + 0 + 4*4 ); b01 = *(B +  0 + 5*4 ); b02 = *(B + 0 + 6*4 ); b03 = *(B + 0 + 7*4 );
	            //b10 = *(B + 1 + 4*4 ); b11 = *(B +  1 + 5*4 ); b12 = *(B + 1 + 6*4 ); b13 = *(B + 1 + 7*4 );
	            //b20 = *(B + 2 + 4*4 ); b21 = *(B +  2 + 5*4 ); b22 = *(B + 2 + 6*4 ); b23 = *(B + 2 + 7*4 );
	            //b30 = *(B + 3 + 4*4 ); b31 = *(B +  3 + 5*4 ); b32 = *(B + 3 + 6*4 ); b33 = *(B + 3 + 7*4 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c02 += a00 * b20; c02 += a01 * b21; c02 += a02 * b22; c02 += a03 * b23; 
	            //c03 += a00 * b30; c03 += a01 * b31; c03 += a02 * b32; c03 += a03 * b33; 
	            //A+=8;
	            //B+=32;

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastsd  0*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastsd  1*8(%[a]), %%ymm1 \n\t"
	                "vbroadcastsd  2*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastsd  3*8(%[a]), %%ymm3 \n\t"
	                "vmovapd       0*8(%[b]), %%ymm4 \n\t"
	                "vmovapd       4*8(%[b]), %%ymm5 \n\t"
	                "vmovapd       8*8(%[b]), %%ymm6 \n\t"
	                "vmovapd      12*8(%[b]), %%ymm7 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm15\n\t"
	                "\n\t"
	                "vbroadcastsd  4*8(%[a]), %%ymm9 \n\t"
	                "vbroadcastsd  5*8(%[a]), %%ymm10\n\t"
	                "vbroadcastsd  6*8(%[a]), %%ymm11\n\t"
	                "vbroadcastsd  7*8(%[a]), %%ymm12\n\t"
	                "vmovapd      16*8(%[b]), %%ymm4 \n\t"
	                "vmovapd      20*8(%[b]), %%ymm5 \n\t"
	                "vmovapd      24*8(%[b]), %%ymm6 \n\t"
	                "vmovapd      28*8(%[b]), %%ymm7 \n\t"
	                "vfmadd231pd %%ymm9  , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm10 , %%ymm5 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm11 , %%ymm6 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm12 , %%ymm7 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $32*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 ); a02 = *(A + 0 + 2*1 ); a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*4 ); b01 = *(B +  0 + 1*4 ); b02 = *(B + 0 + 2*4 ); b03 = *(B + 0 + 3*4 );
	            //b10 = *(B + 1 + 0*4 ); b11 = *(B +  1 + 1*4 ); b12 = *(B + 1 + 2*4 ); b13 = *(B + 1 + 3*4 );
	            //b20 = *(B + 2 + 0*4 ); b21 = *(B +  2 + 1*4 ); b22 = *(B + 2 + 2*4 ); b23 = *(B + 2 + 3*4 );
	            //b30 = *(B + 3 + 0*4 ); b31 = *(B +  3 + 1*4 ); b32 = *(B + 3 + 2*4 ); b33 = *(B + 3 + 3*4 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c02 += a00 * b20; c02 += a01 * b21; c02 += a02 * b22; c02 += a03 * b23; 
	            //c03 += a00 * b30; c03 += a01 * b31; c03 += a02 * b32; c03 += a03 * b33; 
	            //A+=4;
	            //B+=16;
	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastsd  0*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastsd  1*8(%[a]), %%ymm1 \n\t"
	                "vbroadcastsd  2*8(%[a]), %%ymm2 \n\t"
	                "vbroadcastsd  3*8(%[a]), %%ymm3 \n\t"
	                "vmovapd       0*8(%[b]), %%ymm4 \n\t"
	                "vmovapd       4*8(%[b]), %%ymm5 \n\t"
	                "vmovapd       8*8(%[b]), %%ymm6 \n\t"
	                "vmovapd      12*8(%[b]), %%ymm7 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $4*8 , %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 );
	            //b00 = *(B + 0 + 0*4 ); b01 = *(B +  0 + 1*4 );
	            //b10 = *(B + 1 + 0*4 ); b11 = *(B +  1 + 1*4 );
	            //b20 = *(B + 2 + 0*4 ); b21 = *(B +  2 + 1*4 );
	            //b30 = *(B + 3 + 0*4 ); b31 = *(B +  3 + 1*4 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c01 += a00 * b10; c01 += a01 * b11;
	            //c02 += a00 * b20; c02 += a01 * b21;
	            //c03 += a00 * b30; c03 += a01 * b31;
	            //A+=2;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastsd  0*8(%[a]), %%ymm0 \n\t"
	                "vbroadcastsd  1*8(%[a]), %%ymm1 \n\t"
	                "vmovapd       0*8(%[b]), %%ymm4 \n\t"
	                "vmovapd       4*8(%[b]), %%ymm5 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $2*8 , %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*1 );
	            //b00 = *(B + 0 + 0*4 );
	            //b10 = *(B + 1 + 0*4 );
	            //b20 = *(B + 2 + 0*4 );
	            //b30 = *(B + 3 + 0*4 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b10;
	            //c02 += a00 * b20;
	            //c03 += a00 * b30;
	            //A+=1;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                //"prefetcht0  64*8(%[a])\n\t"
	                "\n\t"
	                "vbroadcastsd  0*8(%[a]), %%ymm0 \n\t"
	                "vmovapd       0*8(%[b]), %%ymm4 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $1*8 , %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;
	        //*(C+0+2*ldc) += alpha*c02;
	        //*(C+0+3*ldc) += alpha*c03;

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vbroadcastsd %[alpha], %%ymm0 \n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vmulpd  %%ymm0, %%ymm15, %%ymm15\n\t"
	            "\n\t"
	            "vshufpd    $0x05, %%ymm15, %%ymm15, %%ymm14\n\t" // imm8=[00000101], ymm15[00,01,02,03] ymm14[01,00,03,02]
	            "vperm2f128 $0x01, %%ymm14, %%ymm14, %%ymm12\n\t" // ymm14[01], ymm12[03]
	            "vperm2f128 $0x01, %%ymm15, %%ymm15, %%ymm13\n\t" // ymm15[00], ymm13[02]
	            //"\n\t"
	            //"vmulsd  %%xmm0, %%xmm15, %%xmm15\n\t"
	            //"vmulsd  %%xmm0, %%xmm14, %%xmm14\n\t"
	            //"vmulsd  %%xmm0, %%xmm13, %%xmm13\n\t"
	            //"vmulsd  %%xmm0, %%xmm12, %%xmm12\n\t"
	            "\n\t"
	            "vaddsd  0*8(%[c0]        ), %%xmm15, %%xmm15\n\t"
	            "vaddsd  0*8(%[c1]        ), %%xmm14, %%xmm14\n\t"
	            "vaddsd  0*8(%[c0],%[ldc2]), %%xmm13, %%xmm13\n\t"
	            "vaddsd  0*8(%[c1],%[ldc2]), %%xmm12, %%xmm12\n\t"
	            "\n\t"
	            "vmovsd  %%xmm15, 0*8(%[c0]        )\n\t"
	            "vmovsd  %%xmm14, 0*8(%[c1]        )\n\t"
	            "vmovsd  %%xmm13, 0*8(%[c0],%[ldc2])\n\t"
	            "vmovsd  %%xmm12, 0*8(%[c1],%[ldc2])\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "addq  $1*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 4*K;//N*K
	        //C+=1;
	        //c0+=1;
	        //c1+=1;

	    }
	    A = A - M*K;
	    B = B + 4*K;
	    //C  = C - M + 4*ldc;
	    c0 = c0 - M + 4*ldc;
	    c1 = c1 - M + 4*ldc;
	  }
	}
	if( N & 2 ){

	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;

	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);

	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 ); a02 = *(A + 0 + 2*4 ); a03 = *(A + 0 + 3*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 ); a12 = *(A + 1 + 2*4 ); a13 = *(A + 1 + 3*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 ); a22 = *(A + 2 + 2*4 ); a23 = *(A + 2 + 3*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 ); a32 = *(A + 3 + 2*4 ); a33 = *(A + 3 + 3*4 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c21 += a20 * b10; c21 += a21 * b11; c21 += a22 * b12; c21 += a23 * b13; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //c31 += a30 * b10; c31 += a31 * b11; c31 += a32 * b12; c31 += a33 * b13; 
	            //a00 = *(A + 0 + 4*4 ); a01 = *(A +  0 + 5*4 ); a02 = *(A + 0 + 6*4 ); a03 = *(A + 0 + 7*4 );
	            //a10 = *(A + 1 + 4*4 ); a11 = *(A +  1 + 5*4 ); a12 = *(A + 1 + 6*4 ); a13 = *(A + 1 + 7*4 );
	            //a20 = *(A + 2 + 4*4 ); a21 = *(A +  2 + 5*4 ); a22 = *(A + 2 + 6*4 ); a23 = *(A + 2 + 7*4 );
	            //a30 = *(A + 3 + 4*4 ); a31 = *(A +  3 + 5*4 ); a32 = *(A + 3 + 6*4 ); a33 = *(A + 3 + 7*4 );
	            //b00 = *(B + 0 + 4*2 ); b01 = *(B +  0 + 5*2 ); b02 = *(B + 0 + 6*2 ); b03 = *(B + 0 + 7*2 );
	            //b10 = *(B + 1 + 4*2 ); b11 = *(B +  1 + 5*2 ); b12 = *(B + 1 + 6*2 ); b13 = *(B + 1 + 7*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c21 += a20 * b10; c21 += a21 * b11; c21 += a22 * b12; c21 += a23 * b13; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //c31 += a30 * b10; c31 += a31 * b11; c31 += a32 * b12; c31 += a33 * b13; 
	            //A+=32;
	            //B+=16;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd          0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd          4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "vshufpd     $0x0f, %%ymm0 , %%ymm0 , %%ymm8 \n\t"
	                "vshufpd     $0x0f, %%ymm1 , %%ymm1 , %%ymm9 \n\t"
	                "vshufpd     $0x00, %%ymm0 , %%ymm0 , %%ymm0 \n\t"
	                "vshufpd     $0x00, %%ymm1 , %%ymm1 , %%ymm1 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm8 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm9 , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "vmovapd          8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd         12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastf128   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b02,b12]
	                "vbroadcastf128   6*8(%[b]), %%ymm7 \n\t" // [b03,b13,b03,b13]
	                "vshufpd     $0x0f, %%ymm2 , %%ymm2 , %%ymm10\n\t"
	                "vshufpd     $0x0f, %%ymm3 , %%ymm3 , %%ymm11\n\t"
	                "vshufpd     $0x00, %%ymm2 , %%ymm2 , %%ymm2 \n\t"
	                "vshufpd     $0x00, %%ymm3 , %%ymm3 , %%ymm3 \n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm10, %%ymm6 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm11, %%ymm7 , %%ymm15\n\t"
	                "\n\t"
	                "vmovapd         16*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd         20*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vbroadcastf128   8*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128  10*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "vshufpd     $0x0f, %%ymm0 , %%ymm0 , %%ymm8 \n\t"
	                "vshufpd     $0x0f, %%ymm1 , %%ymm1 , %%ymm9 \n\t"
	                "vshufpd     $0x00, %%ymm0 , %%ymm0 , %%ymm0 \n\t"
	                "vshufpd     $0x00, %%ymm1 , %%ymm1 , %%ymm1 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm8 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm9 , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "vmovapd         24*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd         28*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastf128  12*8(%[b]), %%ymm6 \n\t" // [b02,b12,b02,b12]
	                "vbroadcastf128  14*8(%[b]), %%ymm7 \n\t" // [b03,b13,b03,b13]
	                "vshufpd     $0x0f, %%ymm2 , %%ymm2 , %%ymm10\n\t"
	                "vshufpd     $0x0f, %%ymm3 , %%ymm3 , %%ymm11\n\t"
	                "vshufpd     $0x00, %%ymm2 , %%ymm2 , %%ymm2 \n\t"
	                "vshufpd     $0x00, %%ymm3 , %%ymm3 , %%ymm3 \n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm10, %%ymm6 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm11, %%ymm7 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 ); a02 = *(A + 0 + 2*4 ); a03 = *(A + 0 + 3*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 ); a12 = *(A + 1 + 2*4 ); a13 = *(A + 1 + 3*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 ); a22 = *(A + 2 + 2*4 ); a23 = *(A + 2 + 3*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 ); a32 = *(A + 3 + 2*4 ); a33 = *(A + 3 + 3*4 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c21 += a20 * b10; c21 += a21 * b11; c21 += a22 * b12; c21 += a23 * b13; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //c31 += a30 * b10; c31 += a31 * b11; c31 += a32 * b12; c31 += a33 * b13; 
	            //A+=16;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd          0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd          4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "vshufpd     $0x0f, %%ymm0 , %%ymm0 , %%ymm8 \n\t"
	                "vshufpd     $0x0f, %%ymm1 , %%ymm1 , %%ymm9 \n\t"
	                "vshufpd     $0x00, %%ymm0 , %%ymm0 , %%ymm0 \n\t"
	                "vshufpd     $0x00, %%ymm1 , %%ymm1 , %%ymm1 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm8 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm9 , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "vmovapd          8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd         12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastf128   4*8(%[b]), %%ymm6 \n\t" // [b02,b12,b02,b12]
	                "vbroadcastf128   6*8(%[b]), %%ymm7 \n\t" // [b03,b13,b03,b13]
	                "vshufpd     $0x0f, %%ymm2 , %%ymm2 , %%ymm10\n\t"
	                "vshufpd     $0x0f, %%ymm3 , %%ymm3 , %%ymm11\n\t"
	                "vshufpd     $0x00, %%ymm2 , %%ymm2 , %%ymm2 \n\t"
	                "vshufpd     $0x00, %%ymm3 , %%ymm3 , %%ymm3 \n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm10, %%ymm6 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm11, %%ymm7 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 2 ){

	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c10 += a10 * b00; c10 += a11 * b01;
	            //c20 += a20 * b00; c20 += a21 * b01;
	            //c30 += a30 * b00; c30 += a31 * b01;
	            //c01 += a00 * b10; c01 += a01 * b11;
	            //c11 += a10 * b10; c11 += a11 * b11;
	            //c21 += a20 * b10; c21 += a21 * b11;
	            //c31 += a30 * b10; c31 += a31 * b11;
	            //A+=8;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd          0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd          4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vbroadcastf128   2*8(%[b]), %%ymm5 \n\t" // [b01,b11,b01,b11]
	                "vshufpd     $0x0f, %%ymm0 , %%ymm0 , %%ymm8 \n\t"
	                "vshufpd     $0x0f, %%ymm1 , %%ymm1 , %%ymm9 \n\t"
	                "vshufpd     $0x00, %%ymm0 , %%ymm0 , %%ymm0 \n\t"
	                "vshufpd     $0x00, %%ymm1 , %%ymm1 , %%ymm1 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm8 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm9 , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*4 );
	            //a10 = *(A + 1 + 0*4 );
	            //a20 = *(A + 2 + 0*4 );
	            //a30 = *(A + 3 + 0*4 );
	            //b00 = *(B + 0 + 0*2 );
	            //b10 = *(B + 1 + 0*2 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b10;
	            //c10 += a10 * b00;
	            //c11 += a10 * b10;
	            //c20 += a20 * b00;
	            //c21 += a20 * b10;
	            //c30 += a30 * b00;
	            //c31 += a30 * b10;
	            //A+=4;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "\n\t"
	                "vmovapd          0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vbroadcastf128   0*8(%[b]), %%ymm4 \n\t" // [b00,b10,b00,b10]
	                "vshufpd     $0x0f, %%ymm0 , %%ymm0 , %%ymm8 \n\t"
	                "vshufpd     $0x00, %%ymm0 , %%ymm0 , %%ymm0 \n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm14\n\t"
	                "vfmadd231pd %%ymm8 , %%ymm4 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;
	        //*(C+1+0*ldc) += alpha*c10;
	        //*(C+1+1*ldc) += alpha*c11;
	        //*(C+2+0*ldc) += alpha*c20;
	        //*(C+2+1*ldc) += alpha*c21;
	        //*(C+3+0*ldc) += alpha*c30;
	        //*(C+3+1*ldc) += alpha*c31;

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vbroadcastsd %[alpha], %%ymm0 \n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vshufpd  $0x00, %%ymm15, %%ymm14, %%ymm12\n\t"
	            "vshufpd  $0x0f, %%ymm15, %%ymm14, %%ymm13\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]), %%ymm0, %%ymm12\n\t"
	            "vfmadd213pd 0*8(%[c1]), %%ymm0, %%ymm13\n\t"
	            "\n\t"
	            "vmovupd  %%ymm12, 0*8(%[c0])\n\t"
	            "vmovupd  %%ymm13, 0*8(%[c1])\n\t"
	            "\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4)
	        );

	        B = B - 2*K;//N*K
	        //C+=4;
	        //c0+=4;
	        //c1+=4;
	      }

	    }
	    if( M & 2 ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 ); a02 = *(A + 0 + 2*2 ); a03 = *(A + 0 + 3*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 ); a12 = *(A + 1 + 2*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //a00 = *(A + 0 + 4*2 ); a01 = *(A +  0 + 5*2 ); a02 = *(A + 0 + 6*2 ); a03 = *(A + 0 + 7*2 );
	            //a10 = *(A + 1 + 4*2 ); a11 = *(A +  1 + 5*2 ); a12 = *(A + 1 + 6*2 ); a13 = *(A + 1 + 7*2 );
	            //b00 = *(B + 0 + 4*2 ); b01 = *(B +  0 + 5*2 ); b02 = *(B + 0 + 6*2 ); b03 = *(B + 0 + 7*2 );
	            //b10 = *(B + 1 + 4*2 ); b11 = *(B +  1 + 5*2 ); b12 = *(B + 1 + 6*2 ); b13 = *(B + 1 + 7*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //A+=16;
	            //B+=16;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   4*8(%[a]), %%xmm2 \n\t" // [a02,a12]
	                "movapd   6*8(%[a]), %%xmm3 \n\t" // [a03,a13]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "movapd   4*8(%[b]), %%xmm6 \n\t" // [b02,b12]
	                "movapd   6*8(%[b]), %%xmm7 \n\t" // [b03,b13]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm8 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b10,b10]
	                "vshufpd   $0x00, %%xmm5 , %%xmm5 , %%xmm9 \n\t" // [b01,b01]
	                "vshufpd   $0x03, %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b11,b11]
	                "vfmadd231pd %%xmm0 , %%xmm8 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm9 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "vshufpd   $0x00, %%xmm6 , %%xmm6 , %%xmm10\n\t" // [b02,b02]
	                "vshufpd   $0x03, %%xmm6 , %%xmm6 , %%xmm6 \n\t" // [b12,b12]
	                "vshufpd   $0x00, %%xmm7 , %%xmm7 , %%xmm11\n\t" // [b03,b03]
	                "vshufpd   $0x03, %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b13,b13]
	                "vfmadd231pd %%xmm2 , %%xmm10, %%xmm14\n\t"
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm11, %%xmm14\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "movapd   8*8(%[a]), %%xmm0 \n\t" // [a04,a14]
	                "movapd  10*8(%[a]), %%xmm1 \n\t" // [a05,a15]
	                "movapd  12*8(%[a]), %%xmm2 \n\t" // [a06,a16]
	                "movapd  14*8(%[a]), %%xmm3 \n\t" // [a07,a17]
	                "movapd   8*8(%[b]), %%xmm4 \n\t" // [b04,b14]
	                "movapd  10*8(%[b]), %%xmm5 \n\t" // [b05,b15]
	                "movapd  12*8(%[b]), %%xmm6 \n\t" // [b06,b16]
	                "movapd  14*8(%[b]), %%xmm7 \n\t" // [b07,b17]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm8 \n\t" // [b04,b04]
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b14,b14]
	                "vshufpd   $0x00, %%xmm5 , %%xmm5 , %%xmm9 \n\t" // [b05,b05]
	                "vshufpd   $0x03, %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b15,b15]
	                "vfmadd231pd %%xmm0 , %%xmm8 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm9 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "vshufpd   $0x00, %%xmm6 , %%xmm6 , %%xmm10\n\t" // [b06,b06]
	                "vshufpd   $0x03, %%xmm6 , %%xmm6 , %%xmm6 \n\t" // [b16,b16]
	                "vshufpd   $0x00, %%xmm7 , %%xmm7 , %%xmm11\n\t" // [b07,b07]
	                "vshufpd   $0x03, %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b17,b17]
	                "vfmadd231pd %%xmm2 , %%xmm10, %%xmm14\n\t"
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm11, %%xmm14\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 ); a02 = *(A + 0 + 2*2 ); a03 = *(A + 0 + 3*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 ); a12 = *(A + 1 + 2*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c11 += a10 * b10; c11 += a11 * b11; c11 += a12 * b12; c11 += a13 * b13; 
	            //A+=8;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm8 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b10,b10]
	                "vshufpd   $0x00, %%xmm5 , %%xmm5 , %%xmm9 \n\t" // [b01,b01]
	                "vshufpd   $0x03, %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b11,b11]
	                "vfmadd231pd %%xmm0 , %%xmm8 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm9 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "\n\t"
	                "movapd   4*8(%[a]), %%xmm2 \n\t" // [a02,a12]
	                "movapd   6*8(%[a]), %%xmm3 \n\t" // [a03,a13]
	                "movapd   4*8(%[b]), %%xmm6 \n\t" // [b02,b12]
	                "movapd   6*8(%[b]), %%xmm7 \n\t" // [b03,b13]
	                "vshufpd   $0x00, %%xmm6 , %%xmm6 , %%xmm10\n\t" // [b02,b02]
	                "vshufpd   $0x03, %%xmm6 , %%xmm6 , %%xmm6 \n\t" // [b12,b12]
	                "vshufpd   $0x00, %%xmm7 , %%xmm7 , %%xmm11\n\t" // [b03,b03]
	                "vshufpd   $0x03, %%xmm7 , %%xmm7 , %%xmm7 \n\t" // [b13,b13]
	                "vfmadd231pd %%xmm2 , %%xmm10, %%xmm14\n\t"
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm11, %%xmm14\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c01 += a00 * b10; c01 += a01 * b11;
	            //c10 += a10 * b00; c10 += a11 * b01;
	            //c11 += a10 * b10; c11 += a11 * b11;
	            //A+=4;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm8 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b10,b10]
	                "vshufpd   $0x00, %%xmm5 , %%xmm5 , %%xmm9 \n\t" // [b01,b01]
	                "vshufpd   $0x03, %%xmm5 , %%xmm5 , %%xmm5 \n\t" // [b11,b11]
	                "vfmadd231pd %%xmm0 , %%xmm8 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm9 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*2 );
	            //a10 = *(A + 1 + 0*2 );
	            //b00 = *(B + 0 + 0*2 );
	            //b10 = *(B + 1 + 0*2 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b10;
	            //c10 += a10 * b00;
	            //c11 += a10 * b10;
	            //A+=2;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm8 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b10,b10]
	                "vfmadd231pd %%xmm0 , %%xmm8 , %%xmm14\n\t"
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;
	        //*(C+1+0*ldc) += alpha*c10;
	        //*(C+1+1*ldc) += alpha*c11;

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vbroadcastsd %[alpha], %%ymm0 \n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "\n\t"
	            "vfmadd213pd 0*8(%[c0]), %%xmm0, %%xmm14\n\t"
	            "vfmadd213pd 0*8(%[c1]), %%xmm0, %%xmm15\n\t"
	            "\n\t"
	            "vmovupd  %%xmm14, 0*8(%[c0])\n\t"
	            "vmovupd  %%xmm15, 0*8(%[c1])\n\t"
	            "\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4)
	        );


	        B = B - 2*K;//N*K
	        //C+=2;
	        //c0+=2;
	        //c1+=2;

	    }
	    if( M & 1 ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm12, %%ymm12, %%ymm12\n\t"
	            "vpxor  %%ymm13, %%ymm13, %%ymm13\n\t"
	            "vpxor  %%ymm14, %%ymm14, %%ymm14\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 ); a02 = *(A + 0 + 2*1 ); a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //a00 = *(A + 0 + 4*1 ); a01 = *(A +  0 + 5*1 ); a02 = *(A + 0 + 6*1 ); a03 = *(A + 0 + 7*1 );
	            //b00 = *(B + 0 + 4*2 ); b01 = *(B +  0 + 5*2 ); b02 = *(B + 0 + 6*2 ); b03 = *(B + 0 + 7*2 );
	            //b10 = *(B + 1 + 4*2 ); b11 = *(B +  1 + 5*2 ); b12 = *(B + 1 + 6*2 ); b13 = *(B + 1 + 7*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //A+=8;
	            //B+=16;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a01]
	                "movapd   2*8(%[a]), %%xmm2 \n\t" // [a02,a03]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "movapd   4*8(%[b]), %%xmm6 \n\t" // [b02,b12]
	                "movapd   6*8(%[b]), %%xmm7 \n\t" // [b03,b13]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm0 , %%xmm0 , %%xmm8 \n\t" // [a00,a00]
	                "vshufpd   $0x03, %%xmm0 , %%xmm0 , %%xmm9 \n\t" // [a01,a01]
	                "vshufpd   $0x00, %%xmm2 , %%xmm2 , %%xmm10\n\t" // [a02,a02]
	                "vshufpd   $0x03, %%xmm2 , %%xmm2 , %%xmm11\n\t" // [a03,a03]
	                "vfmadd231pd %%xmm8  , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm9  , %%xmm5 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm10 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm11 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "movapd   4*8(%[a]), %%xmm0 \n\t" // [a00,a01]
	                "movapd   6*8(%[a]), %%xmm2 \n\t" // [a02,a03]
	                "movapd   8*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd  10*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "movapd  12*8(%[b]), %%xmm6 \n\t" // [b02,b12]
	                "movapd  14*8(%[b]), %%xmm7 \n\t" // [b03,b13]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm0 , %%xmm0 , %%xmm8 \n\t" // [a00,a00]
	                "vshufpd   $0x03, %%xmm0 , %%xmm0 , %%xmm9 \n\t" // [a01,a01]
	                "vshufpd   $0x00, %%xmm2 , %%xmm2 , %%xmm10\n\t" // [a02,a02]
	                "vshufpd   $0x03, %%xmm2 , %%xmm2 , %%xmm11\n\t" // [a03,a03]
	                "vfmadd231pd %%xmm8  , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm9  , %%xmm5 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm10 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm11 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $8*8 , %[a]\n\t"
	                "addq  $16*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 ); a02 = *(A + 0 + 2*1 ); a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 ); b02 = *(B + 0 + 2*2 ); b03 = *(B + 0 + 3*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 ); b12 = *(B + 1 + 2*2 ); b13 = *(B + 1 + 3*2 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c01 += a00 * b10; c01 += a01 * b11; c01 += a02 * b12; c01 += a03 * b13; 
	            //A+=4;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a01]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "vshufpd   $0x03, %%xmm0 , %%xmm0 , %%xmm1 \n\t" // [a01,a01]
	                "vshufpd   $0x00, %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a00,a00]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "\n\t"
	                "movapd   2*8(%[a]), %%xmm2 \n\t" // [a02,a03]
	                "movapd   4*8(%[b]), %%xmm6 \n\t" // [b02,b12]
	                "movapd   6*8(%[b]), %%xmm7 \n\t" // [b03,b13]
	                "vshufpd   $0x03, %%xmm2 , %%xmm2 , %%xmm3 \n\t" // [a03,a03]
	                "vshufpd   $0x00, %%xmm2 , %%xmm2 , %%xmm2 \n\t" // [a02,a02]
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 );
	            //b00 = *(B + 0 + 0*2 ); b01 = *(B +  0 + 1*2 );
	            //b10 = *(B + 1 + 0*2 ); b11 = *(B +  1 + 1*2 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c01 += a00 * b10; c01 += a01 * b11;
	            //A+=2;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a01]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "movapd   2*8(%[b]), %%xmm5 \n\t" // [b01,b11]
	                "\n\t"
	                "vshufpd   $0x03, %%xmm0 , %%xmm0 , %%xmm1 \n\t" // [a01,a01]
	                "vshufpd   $0x00, %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a00,a00]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*1 );
	            //b00 = *(B + 0 + 0*2 );
	            //b10 = *(B + 1 + 0*2 );
	            //c00 += a00 * b00;
	            //c01 += a00 * b10;
	            //A+=1;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movsd    0*8(%[a]), %%xmm0 \n\t" // [a00]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b10]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm0 , %%xmm0 , %%xmm0 \n\t" // [a00,a00]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $1*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+0+1*ldc) += alpha*c01;

	        __asm__ __volatile__ (
	            "\n\t"
	            "movupd %[alpha], %%xmm0\n\t"
	            "\n\t"
	            "mulpd   %%xmm0 , %%xmm15\n\t"
	            "movapd  %%xmm15, %%xmm14\n\t"
	            "shufpd  $0x01, %%xmm15, %%xmm14\n\t"
	            "addsd   0*8(%[c0]), %%xmm15\n\t"
	            "addsd   0*8(%[c1]), %%xmm14\n\t"
	            "movlpd  %%xmm15, 0*8(%[c0])\n\t"
	            "movlpd  %%xmm14, 0*8(%[c1])\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "addq  $1*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4),[ldc2]"r"(ldc2)
	        );

	        B = B - 2*K;//N*K
	        //C+=1;
	        //c0+=1;
	        //c1+=1;

	    }
	    A = A - M*K;
	    B = B + 2*K;
	    //C  = C - M + 2*ldc;
	    c0 = c0- M + 2*ldc;
	    c1 = c1- M + 2*ldc;

	}
	if( N & 1 ){

	    if( M >> 2 ){
	      size_t m4 = ( M >> 2 );
	      while( m4-- ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 ); a02 = *(A + 0 + 2*4 ); a03 = *(A + 0 + 3*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 ); a12 = *(A + 1 + 2*4 ); a13 = *(A + 1 + 3*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 ); a22 = *(A + 2 + 2*4 ); a23 = *(A + 2 + 3*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 ); a32 = *(A + 3 + 2*4 ); a33 = *(A + 3 + 3*4 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //a00 = *(A + 0 + 4*4 ); a01 = *(A +  0 + 5*4 ); a02 = *(A + 0 + 6*4 ); a03 = *(A + 0 + 7*4 );
	            //a10 = *(A + 1 + 4*4 ); a11 = *(A +  1 + 5*4 ); a12 = *(A + 1 + 6*4 ); a13 = *(A + 1 + 7*4 );
	            //a20 = *(A + 2 + 4*4 ); a21 = *(A +  2 + 5*4 ); a22 = *(A + 2 + 6*4 ); a23 = *(A + 2 + 7*4 );
	            //a30 = *(A + 3 + 4*4 ); a31 = *(A +  3 + 5*4 ); a32 = *(A + 3 + 6*4 ); a33 = *(A + 3 + 7*4 );
	            //b00 = *(B + 0 + 4*1 ); b01 = *(B +  0 + 5*1 ); b02 = *(B + 0 + 6*1 ); b03 = *(B + 0 + 7*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //A+=32;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd        4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd        8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd       12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   1*8(%[b]), %%ymm5 \n\t" // [b01,b01,b01,b01]
	                "vbroadcastsd   2*8(%[b]), %%ymm6 \n\t" // [b02,b02,b02,b02]
	                "vbroadcastsd   3*8(%[b]), %%ymm7 \n\t" // [b03,b03,b03,b03]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm15\n\t"
	                "\n\t"
	                "vmovapd       16*8(%[a]), %%ymm8 \n\t" // [a00,a10,a20,a30]
	                "vmovapd       20*8(%[a]), %%ymm9 \n\t" // [a01,a11,a21,a31]
	                "vmovapd       24*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd       28*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastsd   4*8(%[b]), %%ymm10\n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   5*8(%[b]), %%ymm11\n\t" // [b01,b01,b01,b01]
	                "vbroadcastsd   6*8(%[b]), %%ymm12\n\t" // [b02,b02,b02,b02]
	                "vbroadcastsd   7*8(%[b]), %%ymm13\n\t" // [b03,b03,b03,b03]
	                "\n\t"
	                "vfmadd231pd %%ymm8 , %%ymm10, %%ymm15\n\t"
	                "vfmadd231pd %%ymm9 , %%ymm11, %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm12, %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm13, %%ymm15\n\t"
	                "\n\t"
	                "addq  $32*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 ); a02 = *(A + 0 + 2*4 ); a03 = *(A + 0 + 3*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 ); a12 = *(A + 1 + 2*4 ); a13 = *(A + 1 + 3*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 ); a22 = *(A + 2 + 2*4 ); a23 = *(A + 2 + 3*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 ); a32 = *(A + 3 + 2*4 ); a33 = *(A + 3 + 3*4 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //c20 += a20 * b00; c20 += a21 * b01; c20 += a22 * b02; c20 += a23 * b03; 
	            //c30 += a30 * b00; c30 += a31 * b01; c30 += a32 * b02; c30 += a33 * b03; 
	            //A+=16;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd        4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vmovapd        8*8(%[a]), %%ymm2 \n\t" // [a02,a12,a22,a32]
	                "vmovapd       12*8(%[a]), %%ymm3 \n\t" // [a03,a13,a23,a33]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   1*8(%[b]), %%ymm5 \n\t" // [b01,b01,b01,b01]
	                "vbroadcastsd   2*8(%[b]), %%ymm6 \n\t" // [b02,b02,b02,b02]
	                "vbroadcastsd   3*8(%[b]), %%ymm7 \n\t" // [b03,b03,b03,b03]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm2 , %%ymm6 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm3 , %%ymm7 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $4*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*4 ); a01 = *(A +  0 + 1*4 );
	            //a10 = *(A + 1 + 0*4 ); a11 = *(A +  1 + 1*4 );
	            //a20 = *(A + 2 + 0*4 ); a21 = *(A +  2 + 1*4 );
	            //a30 = *(A + 3 + 0*4 ); a31 = *(A +  3 + 1*4 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c10 += a10 * b00; c10 += a11 * b01;
	            //c20 += a20 * b00; c20 += a21 * b01;
	            //c30 += a30 * b00; c30 += a31 * b01;
	            //A+=8;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vmovapd        4*8(%[a]), %%ymm1 \n\t" // [a01,a11,a21,a31]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "vbroadcastsd   1*8(%[b]), %%ymm5 \n\t" // [b01,b01,b01,b01]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1 , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*4 );
	            //a10 = *(A + 1 + 0*4 );
	            //a20 = *(A + 2 + 0*4 );
	            //a30 = *(A + 3 + 0*4 );
	            //b00 = *(B + 0 + 0*1 );
	            //c00 += a00 * b00;
	            //c10 += a10 * b00;
	            //c20 += a20 * b00;
	            //c30 += a30 * b00;
	            //A+=4;
	            //B+=1;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovapd        0*8(%[a]), %%ymm0 \n\t" // [a00,a10,a20,a30]
	                "vbroadcastsd   0*8(%[b]), %%ymm4 \n\t" // [b00,b00,b00,b00]
	                "\n\t"
	                "vfmadd231pd %%ymm0 , %%ymm4 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $1*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+1+0*ldc) += alpha*c10;
	        //*(C+2+0*ldc) += alpha*c20;
	        //*(C+3+0*ldc) += alpha*c30;

	        __asm__ __volatile__ (
	            "\n\t"
	            //"vbroadcastsd %[alpha], %%ymm0 \n\t"
	            "vmovupd %[alpha], %%ymm0\n\t"
	            "vfmadd213pd 0*8(%[c0]), %%ymm0, %%ymm15\n\t"
	            "vmovupd  %%ymm15, 0*8(%[c0])\n\t"
	            "addq  $4*8, %[c0]\n\t"
	            "addq  $4*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4)
	        );

	        B = B - 1*K;//N*K
	        //C+=4;
	        //c0+=4;
	        //c1+=4;
	      }

	    }
	    if( M & 2 ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 ); a02 = *(A + 0 + 2*2 ); a03 = *(A + 0 + 3*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 ); a12 = *(A + 1 + 2*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //a00 = *(A + 0 + 4*2 ); a01 = *(A +  0 + 5*2 ); a02 = *(A + 0 + 6*2 ); a03 = *(A + 0 + 7*2 );
	            //a10 = *(A + 1 + 4*2 ); a11 = *(A +  1 + 5*2 ); a12 = *(A + 1 + 6*2 ); a13 = *(A + 1 + 7*2 );
	            //b00 = *(B + 0 + 4*1 ); b01 = *(B +  0 + 5*1 ); b02 = *(B + 0 + 6*1 ); b03 = *(B + 0 + 7*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //A+=16;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   4*8(%[a]), %%xmm2 \n\t" // [a02,a12]
	                "movapd   6*8(%[a]), %%xmm3 \n\t" // [a03,a13]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b01]
	                "movapd   2*8(%[b]), %%xmm6 \n\t" // [b02,b03]
	                "\n\t"
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm5 \n\t" // [b01,b01]
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm6 , %%xmm6 , %%xmm7 \n\t" // [b03,b03]
	                "vshufpd   $0x00, %%xmm6 , %%xmm6 , %%xmm6 \n\t" // [b02,b02]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "movapd   8*8(%[a]), %%xmm8 \n\t" // [a00,a10]
	                "movapd  10*8(%[a]), %%xmm9 \n\t" // [a01,a11]
	                "movapd  12*8(%[a]), %%xmm10\n\t" // [a02,a12]
	                "movapd  14*8(%[a]), %%xmm11\n\t" // [a03,a13]
	                "movapd   4*8(%[b]), %%xmm12\n\t" // [b00,b01]
	                "movapd   6*8(%[b]), %%xmm13\n\t" // [b02,b03]
	                "\n\t"
	                "vshufpd   $0x03, %%xmm12, %%xmm12, %%xmm5 \n\t" // [b01,b01]
	                "vshufpd   $0x00, %%xmm12, %%xmm12, %%xmm12\n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm13, %%xmm13, %%xmm7 \n\t" // [b03,b03]
	                "vshufpd   $0x00, %%xmm13, %%xmm13, %%xmm13\n\t" // [b02,b02]
	                "vfmadd231pd %%xmm8 , %%xmm12, %%xmm15\n\t"
	                "vfmadd231pd %%xmm9 , %%xmm5 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm10, %%xmm13, %%xmm15\n\t"
	                "vfmadd231pd %%xmm11, %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $16*8, %[a]\n\t"
	                "addq  $8*8 , %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 ); a02 = *(A + 0 + 2*2 ); a03 = *(A + 0 + 3*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 ); a12 = *(A + 1 + 2*2 ); a13 = *(A + 1 + 3*2 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //c10 += a10 * b00; c10 += a11 * b01; c10 += a12 * b02; c10 += a13 * b03; 
	            //A+=8;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   4*8(%[a]), %%xmm2 \n\t" // [a02,a12]
	                "movapd   6*8(%[a]), %%xmm3 \n\t" // [a03,a13]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b01]
	                "movapd   2*8(%[b]), %%xmm6 \n\t" // [b02,b03]
	                "\n\t"
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm5 \n\t" // [b01,b01]
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b00,b00]
	                "vshufpd   $0x03, %%xmm6 , %%xmm6 , %%xmm7 \n\t" // [b03,b03]
	                "vshufpd   $0x00, %%xmm6 , %%xmm6 , %%xmm6 \n\t" // [b02,b02]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm2 , %%xmm6 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm3 , %%xmm7 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);

	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*2 ); a01 = *(A +  0 + 1*2 );
	            //a10 = *(A + 1 + 0*2 ); a11 = *(A +  1 + 1*2 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //c10 += a10 * b00; c10 += a11 * b01;
	            //A+=4;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movapd   2*8(%[a]), %%xmm1 \n\t" // [a01,a11]
	                "movapd   0*8(%[b]), %%xmm4 \n\t" // [b00,b01]
	                "\n\t"
	                "vshufpd   $0x03, %%xmm4 , %%xmm4 , %%xmm5 \n\t" // [b01,b01]
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b00,b00]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "vfmadd231pd %%xmm1 , %%xmm5 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*2 );
	            //a10 = *(A + 1 + 0*2 );
	            //b00 = *(B + 0 + 0*1 );
	            //c00 += a00 * b00;
	            //c10 += a10 * b00;
	            //A+=2;
	            //B+=1;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movapd   0*8(%[a]), %%xmm0 \n\t" // [a00,a10]
	                "movsd    0*8(%[b]), %%xmm4 \n\t" // [b00]
	                "\n\t"
	                "vshufpd   $0x00, %%xmm4 , %%xmm4 , %%xmm4 \n\t" // [b00,b00]
	                "vfmadd231pd %%xmm0 , %%xmm4 , %%xmm15\n\t"
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $1*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        //*(C+0+0*ldc) += alpha*c00;
	        //*(C+1+0*ldc) += alpha*c10;

	        __asm__ __volatile__ (
	            "\n\t"
	            "movupd %[alpha], %%xmm0\n\t"
	            "vfmadd213pd 0*8(%[c0]), %%xmm0, %%xmm15\n\t"
	            "movupd  %%xmm15, 0*8(%[c0])\n\t"
	            "addq  $2*8, %[c0]\n\t"
	            "addq  $2*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4)
	        );


	        B = B - 1*K;//N*K
	        //C+=2;
	        //c0+=2;
	        //c1+=2;

	    }
	    if( M & 1 ){

	        //c00=0e0;c01=0e0;c02=0e0;c03=0e0;
	        //c10=0e0;c11=0e0;c12=0e0;c13=0e0;
	        //c20=0e0;c21=0e0;c22=0e0;c23=0e0;
	        //c30=0e0;c31=0e0;c32=0e0;c33=0e0;
	        __asm__ __volatile__ (
	            "\n\t"
	            "vpxor  %%ymm15, %%ymm15, %%ymm15\n\t"
	        ::);
	        if( K >> 3 ){
	          size_t k8 = ( K >> 3 );
	          while( k8-- ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 ); a02 = *(A + 0 + 2*1 ); a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //a00 = *(A + 0 + 4*1 ); a01 = *(A +  0 + 5*1 ); a02 = *(A + 0 + 6*1 ); a03 = *(A + 0 + 7*1 );
	            //b00 = *(B + 0 + 4*1 ); b01 = *(B +  0 + 5*1 ); b02 = *(B + 0 + 6*1 ); b03 = *(B + 0 + 7*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //A+=8;
	            //B+=8;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a02,a03]
	                "vmovupd   4*8(%[a]), %%ymm1 \n\t" // [a04,a05,a06,a07]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b01,b02,b03]
	                "vmovupd   4*8(%[b]), %%ymm5 \n\t" // [b04,b05,b06,b07]
	                "\n\t"
	                "vfmadd231pd %%ymm0  , %%ymm4 , %%ymm15\n\t"
	                "vfmadd231pd %%ymm1  , %%ymm5 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $8*8, %[a]\n\t"
	                "addq  $8*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	          }
	        }
	        if( K & 4 ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 ); a02 = *(A + 0 + 2*1 ); a03 = *(A + 0 + 3*1 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 ); b02 = *(B + 0 + 2*1 ); b03 = *(B + 0 + 3*1 );
	            //c00 += a00 * b00; c00 += a01 * b01; c00 += a02 * b02; c00 += a03 * b03; 
	            //A+=4;
	            //B+=4;

	            __asm__ __volatile__ (
	                "\n\t"
	                "vmovupd   0*8(%[a]), %%ymm0 \n\t" // [a00,a01,a02,a03]
	                "vmovupd   0*8(%[b]), %%ymm4 \n\t" // [b00,b01,b02,b03]
	                "\n\t"
	                "vfmadd231pd %%ymm0  , %%ymm4 , %%ymm15\n\t"
	                "\n\t"
	                "addq  $4*8, %[a]\n\t"
	                "addq  $4*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 2 ){
	            //a00 = *(A + 0 + 0*1 ); a01 = *(A +  0 + 1*1 );
	            //b00 = *(B + 0 + 0*1 ); b01 = *(B +  0 + 1*1 );
	            //c00 += a00 * b00; c00 += a01 * b01;
	            //A+=2;
	            //B+=2;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movupd   0*8(%[a]), %%xmm0 \n\t" // [a00,a01]
	                "movupd   0*8(%[b]), %%xmm4 \n\t" // [b00,b01]
	                "\n\t"
	                "mulpd  %%xmm0, %%xmm4 \n\t"
	                "addpd  %%xmm4, %%xmm15\n\t"
	                "\n\t"
	                "addq  $2*8, %[a]\n\t"
	                "addq  $2*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        if( K & 1 ){
	            //a00 = *(A + 0 + 0*1 );
	            //b00 = *(B + 0 + 0*1 );
	            //c00 += a00 * b00;
	            //A+=1;
	            //B+=1;

	            __asm__ __volatile__ (
	                "\n\t"
	                "movsd   0*8(%[a]), %%xmm0 \n\t" // [a00,0]
	                "movsd   0*8(%[b]), %%xmm4 \n\t" // [b00,0]
	                "\n\t"
	                "mulsd  %%xmm0, %%xmm4 \n\t" // [a00*b00,0]
	                "addsd  %%xmm4, %%xmm15\n\t"
	                "\n\t"
	                "addq  $1*8, %[a]\n\t"
	                "addq  $1*8, %[b]\n\t"
	                "\n\t"
	                :[a]"+r"(A),[b]"+r"(B)
	            :);


	        }
	        //*(C+0+0*ldc) += alpha*c00;

	        __asm__ __volatile__ (
	            "\n\t"
	            "movupd           %[alpha], %%xmm0 \n\t"
	            "movsd          0*8(%[c0]), %%xmm1 \n\t"
	            "vperm2f128 $0x01, %%ymm15, %%ymm15, %%ymm14\n\t"
	            "vhaddpd           %%ymm14, %%ymm15, %%ymm13\n\t"
	            "vdppd      $0x31, %%xmm13, %%xmm0 , %%xmm12\n\t"
	            "addsd             %%xmm1 , %%xmm12\n\t"
	            "movlpd            %%xmm12, 0*8(%[c0])\n\t"
	            "\n\t"
	            "addq  $1*8, %[c0]\n\t"
	            "addq  $1*8, %[c1]\n\t"
	            "\n\t"
	            :[c0]"+r"(c0),[c1]"+r"(c1)
	            :[alpha]"m"(alpha4)
	        );


	        B = B - 1*K;//N*K
	        //C+=1;
	        //c0+=1;
	        //c1+=1;

	    }
	    A = A - M*K;
	    B = B + 1*K;
	    //C  = C - M + 1*ldc;
	    c0 = c0- M + 1*ldc;
	    c1 = c1- M + 1*ldc;
	}

	A = A + M*K;
	B = B - K*N;
	//C  = C - ldc*N + M;
	c0 = c0- ldc*N + M;
	c1 = c1- ldc*N + M;
	// ---- Kernel

	//free(cc);
}

