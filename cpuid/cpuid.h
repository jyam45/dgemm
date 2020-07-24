#ifndef CPUID_H_HEADER_GUIRD
#define CPUID_H_HEADER_GUIRD

#include <stddef.h>

#define MAX_CACHES 5
#define MAX_TLBS   5
#define MAX_FTLEVEL 5
#define MAX_TPLEVEL 5
#define MAX_PLLEVEL 5

#ifdef __cplusplus
extern "C" {
#endif

/* Structures */
typedef struct _cpuid_t { 
        unsigned int eax;
        unsigned int ebx;
        unsigned int ecx;
        unsigned int edx; 
        } cpuid_t;

typedef struct _verinfo_t{
        unsigned int ex_family;
        unsigned int ex_model;
        unsigned int proc_type;
        unsigned int family;
        unsigned int model;
        unsigned int stepping;
        } version_info_t;
 
// 00H - 03H, 07H
typedef struct _basic_info_t {
        // max input value
        unsigned int  max_support;
        unsigned char vendor_id[13];
        // version info
        version_info_t version;
        unsigned int  brand_index;
        unsigned int  clflush_linesize;
        unsigned int  max_address_id;
        unsigned int  initial_apic_id;
        unsigned int  features[2];
        // cache and TLB
        unsigned char cache_info[16];
        // processor serial number (p3 only)
        unsigned int  proc_serial_number[3];

        } basic_info_t;
// 04H
typedef struct _cache_param_t {
	//eax
	unsigned int  cache_type;
	unsigned int  cache_level;// start from 1
	unsigned int  self_init_level;;
	unsigned int  assoc_fully;
	unsigned int  max_addr_procs; //sharing this cache 
	unsigned int  max_addr_cores; //in physical package
	//ebx,ecx
	unsigned int  line_size;
	unsigned int  line_parts;
	unsigned int  assoc_ways;
	unsigned int  assoc_sets;
	//edx
	unsigned int  write_back_invd;
	unsigned int  cache_inclusive;
	unsigned int  complex_indexing;
	//extra
	size_t        cache_size_b;
	} cache_param_t;

// 07H
typedef struct _extend_feature_t {
	unsigned int  features[4];
	} extend_feature_t;

// 0BH
typedef struct _extend_topology_t {
	unsigned int  shift_apic2topo;
	unsigned int  num_log_procs;
	unsigned int  level_number;
	unsigned int  level_type;
	unsigned int  apic_id;
	} extend_topology_t;

// 16H
typedef struct _proc_freq_info_t {
	unsigned int  proc_base_freq;
	unsigned int  max_freq;
	unsigned int  bus_freq;
	} proc_freq_info_t;

// 18H
typedef struct _tlb_param_t {
	// ebx
	unsigned int  page_size_flags;
	unsigned int  partitioning;
	unsigned int  assoc_ways;
	// ecx
	unsigned int  assoc_sets;
	// edx
	unsigned int  tlb_type;
	unsigned int  tlb_level;
	unsigned int  assoc_fully;;
	unsigned int  max_addr_procs;
	} tlb_param_t;

// 1DH
typedef struct _tile_info_t {
	unsigned int  total_tile_bytes;
	unsigned int  bytes_per_tile;
	unsigned int  bytes_per_row;
	unsigned int  max_names;//num of tile regs
	unsigned int  max_rows;
	} tile_info_t;

// 1EH
typedef struct _tmul_info_t {
	unsigned int  tmul_maxk;
	unsigned int  tmul_maxn;
	} tmul_info_t;

// 80000000H-80000008H
typedef struct _extend_info_t {
	unsigned int  max_supported;
	unsigned int  ex_proc_sign;
	unsigned int  features[4];
	unsigned char proc_brand[49];
	unsigned int  line_size_b;
	unsigned int  l2_assoc_fld;
	unsigned int  l2_size_kb;
	unsigned int  inv_tsc;
	unsigned int  phys_addr_bits;
	unsigned int  linear_addr_bits;
	} extend_info_t;

// Composed Structure
typedef struct _cpuid_info_t {
	basic_info_t      basic_info;
	extend_info_t     extend_info;
	cache_param_t     cache_info[MAX_CACHES];
	tlb_param_t       tlb_info[MAX_TLBS];
        extend_feature_t  more_feature[MAX_FTLEVEL];
        extend_topology_t topology[MAX_FTLEVEL];
	tile_info_t       tile_info[MAX_PLLEVEL];
	tmul_info_t       tmul_info;
	proc_freq_info_t  proc_freq_info;
	size_t            num_caches;
	size_t            num_tlbs;
	size_t            num_ftlevel;
	size_t            num_tplevel;
	size_t            max_palette;
	} cpuid_info_t;


/* Enumerates */
// basic_info.features[0]
enum Features1 {
         F_SSE3            = 0x00000001 // 00
        ,F_PCLMULQDQ       = 0x00000002 // 01
        ,F_DTES64          = 0x00000004 // 02
        ,F_MONITOR         = 0x00000008 // 03
        ,F_DS_CPL          = 0x00000010 // 04
        ,F_VMX             = 0x00000020 // 05
        ,F_SMX             = 0x00000040 // 06
        ,F_EIST            = 0x00000080 // 07
        ,F_TM2             = 0x00000100 // 08
        ,F_SSSE3           = 0x00000200 // 09
        ,F_CNXT_ID         = 0x00000400 // 10
        ,F_SDBG            = 0x00000800 // 11
        ,F_FMA             = 0x00001000 // 12
        ,F_CMPXCHG16B      = 0x00002000 // 13
        ,F_XTPR            = 0x00004000 // 14
        ,F_PDCM            = 0x00008000 // 15
      //,F_RESERVED        = 0x00010000 // 16
        ,F_PCID            = 0x00020000 // 17
        ,F_DCA             = 0x00040000 // 18
        ,F_SSE4_1          = 0x00080000 // 19
        ,F_SSE4_2          = 0x00100000 // 20
        ,F_X2APIC          = 0x00200000 // 21
        ,F_MOVBE           = 0x00400000 // 22
        ,F_POPCNT          = 0x00800000 // 23
        ,F_TSCDEADLINE     = 0x01000000 // 24
        ,F_AESNI           = 0x02000000 // 25
        ,F_XSAVE           = 0x04000000 // 26
        ,F_OSXSAVE         = 0x08000000 // 27
        ,F_AVX             = 0x10000000 // 28
        ,F_F16C            = 0x20000000 // 29
        ,F_RDRAND          = 0x40000000 // 30
      //,F_NOT_USED        = 0x80000000 // 31
};

// basic_info.features[1]
enum Features2 {
         F_FPU             = 0x00000001 // 00
        ,F_VME             = 0x00000002 // 01
        ,F_DE              = 0x00000004 // 02
        ,F_PSE             = 0x00000008 // 03
        ,F_TSC             = 0x00000010 // 04
        ,F_MSR             = 0x00000020 // 05
        ,F_PAE             = 0x00000040 // 06
        ,F_MCE             = 0x00000080 // 07
        ,F_CX8             = 0x00000100 // 08
        ,F_APIC            = 0x00000200 // 09
      //,F_RESERVED        = 0x00000400 // 10
        ,F_SEP             = 0x00000800 // 11
        ,F_MTRR            = 0x00001000 // 12
        ,F_PGE             = 0x00002000 // 13
        ,F_MCA             = 0x00004000 // 14
        ,F_CMOV            = 0x00008000 // 15
        ,F_PAT             = 0x00010000 // 16
        ,F_PSE36           = 0x00020000 // 17
        ,F_PSN             = 0x00040000 // 18
        ,F_CLFSH           = 0x00080000 // 19
      //,F_RESERVED        = 0x00100000 // 20
        ,F_DS              = 0x00200000 // 21
        ,F_ACPI            = 0x00400000 // 22
        ,F_MMX             = 0x00800000 // 23
        ,F_FXSR            = 0x01000000 // 24
        ,F_SSE             = 0x02000000 // 25
        ,F_SSE2            = 0x04000000 // 26
        ,F_SS              = 0x08000000 // 27
        ,F_HTT             = 0x10000000 // 28
        ,F_TM              = 0x20000000 // 29
      //,F_RESERVED        = 0x40000000 // 30
        ,F_PBE             = 0x80000000 // 31
};

// 07H Leaf=0 EBX
// more_feature[0].features[1]
enum Features3 {
         F_FSGSBASE        = 0x00000001 // 00
        ,F_IA32TSC         = 0x00000002 // 01
        ,F_SGX             = 0x00000004 // 02
        ,F_BMI1            = 0x00000008 // 03
        ,F_HLE             = 0x00000010 // 04
        ,F_AVX2            = 0x00000020 // 05
      //,F_RESERVED        = 0x00000040 // 06
        ,F_SMEP            = 0x00000080 // 07
        ,F_BMI2            = 0x00000100 // 08
        ,F_REPMOVSB        = 0x00000200 // 09
        ,F_INVPCID         = 0x00000400 // 10
        ,F_RTM             = 0x00000800 // 11
        ,F_PQM             = 0x00001000 // 12
        ,F_FPUCSDS         = 0x00002000 // 13
        ,F_IPMX            = 0x00004000 // 14
        ,F_PQE             = 0x00008000 // 15
        ,F_AVX512F         = 0x00010000 // 16
        ,F_AVX512DQ        = 0x00020000 // 17
        ,F_RDSEED          = 0x00040000 // 18
        ,F_ADX             = 0x00080000 // 19
        ,F_SMAP            = 0x00100000 // 20
        ,F_AVX512_IFMA     = 0x00200000 // 21
      //,F_RESERVED        = 0x00400000 // 22
        ,F_CLFLUSHOPT      = 0x00800000 // 23
        ,F_CLWB            = 0x01000000 // 24
        ,F_PROCTRACE       = 0x02000000 // 25
        ,F_AVX512PF        = 0x04000000 // 26
        ,F_AVX512ER        = 0x08000000 // 27
        ,F_AVX512CD        = 0x10000000 // 28
        ,F_SHA             = 0x20000000 // 29
        ,F_AVX512BW        = 0x40000000 // 30
        ,F_AVX512VL        = 0x80000000 // 31
};

// 07H Leaf=0 ECX
// more_feature[0].features[2]
enum Features4 {
         F_PREFETCHWT1     = 0x00000001 // 00
        ,F_AVX512_VBMI     = 0x00000002 // 01
        ,F_UMIP            = 0x00000004 // 02
        ,F_PKU             = 0x00000008 // 03
        ,F_OSPKE           = 0x00000010 // 04
        ,F_WAITPKG         = 0x00000020 // 05
        ,F_AVX512_VBMI2    = 0x00000040 // 06
      //,F_RESERVED        = 0x00000080 // 07
        ,F_GFNI            = 0x00000100 // 08
        ,F_VAES            = 0x00000200 // 09
        ,F_VPCLMULQDQ      = 0x00000400 // 10
        ,F_AVX512_VNNI     = 0x00000800 // 11
        ,F_AVX512_BITALG   = 0x00001000 // 12
      //,F_RESERVED        = 0x00002000 // 13
        ,F_AVX512_VPOPCNTDQ= 0x00004000 // 14
      //,F_RESERVED        = 0x00008000 // 15
      //,F_RESERVED        = 0x00010000 // 16
      //,F_RESERVED        = 0x00020000 // 17
      //,F_RESERVED        = 0x00040000 // 18
      //,F_RESERVED        = 0x00080000 // 19
      //,F_RESERVED        = 0x00100000 // 20
      //,F_RESERVED        = 0x00200000 // 21
        ,F_RDPID           = 0x00400000 // 22
      //,F_RESERVED        = 0x00800000 // 23
      //,F_RESERVED        = 0x01000000 // 24
        ,F_CLDEMOTE        = 0x02000000 // 25
      //,F_RESERVED        = 0x04000000 // 26
        ,F_MOVDIRI         = 0x08000000 // 27
        ,F_MOVDIR64B       = 0x10000000 // 28
        ,F_ENQCMD          = 0x20000000 // 29
        ,F_SGX_LC          = 0x40000000 // 30
        ,F_PKS             = 0x80000000 // 31
};

// 07H Leaf=0 EDX
// more_feature[0].features[3]
enum Features5 {
      // F_RESERVED            = 0x00000001 // 00
      //,F_RESERVED            = 0x00000002 // 01
         F_AVX512_4VNNIW       = 0x00000004 // 02
        ,F_AVX512_4FMDPS       = 0x00000008 // 03
        ,F_FASTSHORTREPMOV     = 0x00000010 // 04
      //,F_RESERVED            = 0x00000020 // 05
      //,F_RESERVED            = 0x00000040 // 06
      //,F_RESERVED            = 0x00000080 // 07
        ,F_AVX512_VP2INTERSECT = 0x00000100 // 08
      //,F_RESERVED            = 0x00000200 // 09
        ,F_MD_CLEAR            = 0x00000400 // 10
      //,F_RESERVED            = 0x00000800 // 11
      //,F_RESERVED            = 0x00001000 // 12
      //,F_RESERVED            = 0x00002000 // 13
        ,F_SERIALIZE           = 0x00004000 // 14
        ,F_HYBLID              = 0x00008000 // 15
        ,F_TSXLDTRK            = 0x00010000 // 16
      //,F_RESERVED            = 0x00020000 // 17
        ,F_PCONFIG             = 0x00040000 // 18
      //,F_RESERVED            = 0x00080000 // 19
      //,F_RESERVED            = 0x00100000 // 20
      //,F_RESERVED            = 0x00200000 // 21
        ,F_AMX_BF16            = 0x00400000 // 22
      //,F_RESERVED            = 0x00800000 // 23
        ,F_AMX_TILE            = 0x01000000 // 24
        ,F_AMX_INT8            = 0x02000000 // 25
        ,F_IPRS_IBPB           = 0x04000000 // 26
        ,F_STIBP               = 0x08000000 // 27
      //,F_RESERVED            = 0x10000000 // 28
        ,F_ENUM_IA32_ARCH      = 0x20000000 // 29
        ,F_ENUM_IA32_CORE      = 0x40000000 // 30
        ,F_ENUM_SSDB           = 0x80000000 // 31
};

// 07H Leaf=1 EAX
// more_feature[1].features[0]
enum Features6 {
      //,F_RESERVED            = 0x00000001 // 00
      //,F_RESERVED            = 0x00000002 // 01
      //,F_RESERVED            = 0x00000004 // 02
      //,F_RESERVED            = 0x00000008 // 03
      //,F_RESERVED            = 0x00000010 // 04
         F_AVX512_BF16         = 0x00000020 // 05
      //,F_RESERVED            = 0x00000040 // 06
      //,F_RESERVED            = 0x00000080 // 07
      //,F_RESERVED            = 0x00000100 // 08
      //,F_RESERVED            = 0x00000200 // 09
      //,F_RESERVED            = 0x00000400 // 10
      //,F_RESERVED            = 0x00000800 // 11
      //,F_RESERVED            = 0x00001000 // 12
      //,F_RESERVED            = 0x00002000 // 13
      //,F_RESERVED            = 0x00004000 // 14
      //,F_RESERVED            = 0x00008000 // 15
      //,F_RESERVED            = 0x00010000 // 16
      //,F_RESERVED            = 0x00020000 // 17
      //,F_RESERVED            = 0x00040000 // 18
      //,F_RESERVED            = 0x00080000 // 19
      //,F_RESERVED            = 0x00100000 // 20
      //,F_RESERVED            = 0x00200000 // 21
      //,F_RESERVED            = 0x00400000 // 22
      //,F_RESERVED            = 0x00800000 // 23
      //,F_RESERVED            = 0x01000000 // 24
      //,F_RESERVED            = 0x02000000 // 25
      //,F_RESERVED            = 0x04000000 // 26
      //,F_RESERVED            = 0x08000000 // 27
      //,F_RESERVED            = 0x10000000 // 28
      //,F_RESERVED            = 0x20000000 // 29
      //,F_RESERVED            = 0x40000000 // 30
      //,F_RESERVED            = 0x80000000 // 31
};



enum CacheParam{
         F_CP_WBINVD       = 0x00000001
        ,F_CP_INCLSV       = 0x00000002
        ,F_CP_CMPLXIDX     = 0x00000004
};

//80000001H ECX
enum ExFeatures1 {
         F_X_LAHF          = 0x00000001 // 00
      //,F_X_RESERVED      = 0x00000002 // 01
      //,F_X_RESERVED      = 0x00000004 // 02
      //,F_X_RESERVED      = 0x00000008 // 03
      //,F_X_RESERVED      = 0x00000010 // 04
        ,F_X_LZCNT         = 0x00000020 // 05
      //,F_X_RESERVED      = 0x00000040 // 06
      //,F_X_RESERVED      = 0x00000080 // 07
        ,F_X_PREFETCHW     = 0x00000100 // 08
      //,F_X_RESERVED      = 0x00000200 // 09
      //,F_X_RESERVED      = 0x00000400 // 10
      //,F_X_RESERVED      = 0x00000800 // 11
      //,F_X_RESERVED      = 0x00001000 // 12
      //,F_X_RESERVED      = 0x00002000 // 13
      //,F_X_RESERVED      = 0x00004000 // 14
      //,F_X_RESERVED      = 0x00008000 // 15
      //,F_X_RESERVED      = 0x00010000 // 16
      //,F_X_RESERVED      = 0x00020000 // 17
      //,F_X_RESERVED      = 0x00040000 // 18
      //,F_X_RESERVED      = 0x00080000 // 19
      //,F_X_RESERVED      = 0x00100000 // 20
      //,F_X_RESERVED      = 0x00200000 // 21
      //,F_X_RESERVED      = 0x00400000 // 22
      //,F_X_RESERVED      = 0x00800000 // 23
      //,F_X_RESERVED      = 0x01000000 // 24
      //,F_X_RESERVED      = 0x02000000 // 25
      //,F_X_RESERVED      = 0x04000000 // 26
      //,F_X_RESERVED      = 0x08000000 // 27
      //,F_X_RESERVED      = 0x10000000 // 28
      //,F_X_RESERVED      = 0x20000000 // 29
      //,F_X_RESERVED      = 0x40000000 // 30
      //,F_X_RESERVED      = 0x80000000 // 31
};

//80000001H EDX
enum ExFeatures2 {
      // F_X_RESERVED      = 0x00000001 // 00
      //,F_X_RESERVED      = 0x00000002 // 01
      //,F_X_RESERVED      = 0x00000004 // 02
      //,F_X_RESERVED      = 0x00000008 // 03
      //,F_X_RESERVED      = 0x00000010 // 04
      //,F_X_RESERVED      = 0x00000020 // 05
      //,F_X_RESERVED      = 0x00000040 // 06
      //,F_X_RESERVED      = 0x00000080 // 07
      //,F_X_RESERVED      = 0x00000100 // 08
      //,F_X_RESERVED      = 0x00000200 // 09
      //,F_X_RESERVED      = 0x00000400 // 10
         F_X_SYSCALL       = 0x00000800 // 11
      //,F_X_RESERVED      = 0x00001000 // 12
      //,F_X_RESERVED      = 0x00002000 // 13
      //,F_X_RESERVED      = 0x00004000 // 14
      //,F_X_RESERVED      = 0x00008000 // 15
      //,F_X_RESERVED      = 0x00010000 // 16
      //,F_X_RESERVED      = 0x00020000 // 17
      //,F_X_RESERVED      = 0x00040000 // 18
      //,F_X_RESERVED      = 0x00080000 // 19
        ,F_X_EXEDSBL       = 0x00100000 // 20
      //,F_X_RESERVED      = 0x00200000 // 21
      //,F_X_RESERVED      = 0x00400000 // 22
      //,F_X_RESERVED      = 0x00800000 // 23
      //,F_X_RESERVED      = 0x01000000 // 24
      //,F_X_RESERVED      = 0x02000000 // 25
        ,F_X_1GBPAGE       = 0x04000000 // 26
        ,F_X_RDTSCP        = 0x08000000 // 27
      //,F_X_RESERVED      = 0x10000000 // 28
        ,F_X_INTEL64       = 0x20000000 // 29
      //,F_X_RESERVED      = 0x40000000 // 30
      //,F_X_RESERVED      = 0x80000000 // 31
};

//80000007H EDX
enum ExFeatures3 {
      // F_X_RESERVED      = 0x00000001 // 00
      //,F_X_RESERVED      = 0x00000002 // 01
      //,F_X_RESERVED      = 0x00000004 // 02
      //,F_X_RESERVED      = 0x00000008 // 03
      //,F_X_RESERVED      = 0x00000010 // 04
      //,F_X_RESERVED      = 0x00000020 // 05
      //,F_X_RESERVED      = 0x00000040 // 06
      //,F_X_RESERVED      = 0x00000080 // 07
         F_X_INVALTSC      = 0x00000100 // 08
      //,F_X_RESERVED      = 0x00000200 // 09
      //,F_X_RESERVED      = 0x00000400 // 10
      //,F_X_RESERVED      = 0x00000800 // 11
      //,F_X_RESERVED      = 0x00001000 // 12
      //,F_X_RESERVED      = 0x00002000 // 13
      //,F_X_RESERVED      = 0x00004000 // 14
      //,F_X_RESERVED      = 0x00008000 // 15
      //,F_X_RESERVED      = 0x00010000 // 16
      //,F_X_RESERVED      = 0x00020000 // 17
      //,F_X_RESERVED      = 0x00040000 // 18
      //,F_X_RESERVED      = 0x00080000 // 19
      //,F_X_RESERVED      = 0x00100000 // 20
      //,F_X_RESERVED      = 0x00200000 // 21
      //,F_X_RESERVED      = 0x00400000 // 22
      //,F_X_RESERVED      = 0x00800000 // 23
      //,F_X_RESERVED      = 0x01000000 // 24
      //,F_X_RESERVED      = 0x02000000 // 25
      //,F_X_RESERVED      = 0x04000000 // 26
      //,F_X_RESERVED      = 0x08000000 // 27
      //,F_X_RESERVED      = 0x10000000 // 28
      //,F_X_RESERVED      = 0x20000000 // 29
      //,F_X_RESERVED      = 0x40000000 // 30
      //,F_X_RESERVED      = 0x80000000 // 31
};

//80000008H EBX
enum ExFeatures4 {
      // F_X_RESERVED      = 0x00000001 // 00
      //,F_X_RESERVED      = 0x00000002 // 01
      //,F_X_RESERVED      = 0x00000004 // 02
      //,F_X_RESERVED      = 0x00000008 // 03
      //,F_X_RESERVED      = 0x00000010 // 04
      //,F_X_RESERVED      = 0x00000020 // 05
      //,F_X_RESERVED      = 0x00000040 // 06
      //,F_X_RESERVED      = 0x00000080 // 07
      //,F_X_RESERVED      = 0x00000100 // 08
         F_X_WBNOINVD      = 0x00000200 // 09
      //,F_X_RESERVED      = 0x00000400 // 10
      //,F_X_RESERVED      = 0x00000800 // 11
      //,F_X_RESERVED      = 0x00001000 // 12
      //,F_X_RESERVED      = 0x00002000 // 13
      //,F_X_RESERVED      = 0x00004000 // 14
      //,F_X_RESERVED      = 0x00008000 // 15
      //,F_X_RESERVED      = 0x00010000 // 16
      //,F_X_RESERVED      = 0x00020000 // 17
      //,F_X_RESERVED      = 0x00040000 // 18
      //,F_X_RESERVED      = 0x00080000 // 19
      //,F_X_RESERVED      = 0x00100000 // 20
      //,F_X_RESERVED      = 0x00200000 // 21
      //,F_X_RESERVED      = 0x00400000 // 22
      //,F_X_RESERVED      = 0x00800000 // 23
      //,F_X_RESERVED      = 0x01000000 // 24
      //,F_X_RESERVED      = 0x02000000 // 25
      //,F_X_RESERVED      = 0x04000000 // 26
      //,F_X_RESERVED      = 0x08000000 // 27
      //,F_X_RESERVED      = 0x10000000 // 28
      //,F_X_RESERVED      = 0x20000000 // 29
      //,F_X_RESERVED      = 0x40000000 // 30
      //,F_X_RESERVED      = 0x80000000 // 31
};



/* Fuctions */
void   cpuid( int eax, int ecx, cpuid_t *reg );
void   read_cpuid_info( cpuid_info_t* out );
void   write_cpuid_info( const cpuid_info_t* out );

#ifdef __cplusplus
}
#endif

#endif//CPUID_H_HEADER_GUIRD
