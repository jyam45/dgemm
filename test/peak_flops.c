#include "cpuid.h"
#include "dgemm_test.h"

void peak_flops( flops_info_t* out ){

	cpuid_info_t info;
	read_cpuid_info( &info );

	// Frequency (MHz)
	size_t base_freq = info.proc_freq_info.proc_base_freq;
	size_t max_freq  = info.proc_freq_info.max_freq; // on Tarbo Boost

	// Number of cores
	size_t num_cores = 0;
	for( int i=0; i< info.num_tplevel; i++ ){
		num_cores += info.topology[i].num_log_procs;
	}

	// Number of Floting-point operators
	size_t fp_operator = 2;

	// Floting-point Operations in an operator
	size_t fp_operation = 1;
	if( info.basic_info.features[0] & F_FMA ){ fp_operation = 2; } 
	
	// Vector length
	size_t vlen_bits = 0;
	if( info.basic_info.features[1]      & F_MMX     ){ vlen_bits =  64; } // introduced  MM registers
	if( info.basic_info.features[1]      & F_SSE2    ){ vlen_bits = 128; } // introduced XMM registers
	if( info.more_feature[0].features[1] & F_AVX2    ){ vlen_bits = 256; } // introduced YMM registers
	if( info.more_feature[0].features[1] & F_AVX512F ){ vlen_bits = 512; } // introduced ZMM registers

	out->base_freq    = base_freq;
	out->num_cores    = num_cores;
	out->max_freq     = max_freq;
	out->fp_operator  = fp_operator;
	out->fp_operation = fp_operation;
	out->vlen_bits    = vlen_bits;

	out->mflops_single_base = num_cores * fp_operator * fp_operation * (vlen_bits/(8*sizeof(float))) * base_freq; 
	out->mflops_single_max  = num_cores * fp_operator * fp_operation * (vlen_bits/(8*sizeof(float))) * max_freq; 

	out->mflops_double_base = num_cores * fp_operator * fp_operation * (vlen_bits/(8*sizeof(double))) * base_freq; 
	out->mflops_double_max  = num_cores * fp_operator * fp_operation * (vlen_bits/(8*sizeof(double))) * max_freq; 
}
