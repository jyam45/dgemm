
#include <stdio.h>
#include "cpuid.h"
#include "cache.h"

static void   cpuid_basic_info(basic_info_t* out);
static void   cpuid_print_basic_info( const basic_info_t* out );
static size_t cpuid_cache_params(size_t n, cache_param_t* out );
static void   cpuid_print_cache_params(size_t n, const cache_param_t* out );
static void   cpuid_extend_info(extend_info_t* out);
static void   cpuid_print_extend_info( const extend_info_t* out );
static size_t cpuid_extend_features( size_t n, extend_feature_t* out );
static void   cpuid_print_extend_features( size_t n, const extend_feature_t* out );
static size_t cpuid_extend_topology( size_t n, extend_topology_t* out );
static void   cpuid_print_extend_topology( size_t n, const extend_topology_t* out );
static size_t cpuid_tile_palettes( size_t max_pllevel, tile_info_t* out );
static void   cpuid_print_tile_palettes( size_t max_palette, const tile_info_t* out );
static void   cpuid_tmul_info( tmul_info_t* out );
static void   cpuid_print_tmul_info( const tmul_info_t* out );
static void   cpuid_proc_freq_info( proc_freq_info_t* out );
static void   cpuid_print_proc_freq_info( const proc_freq_info_t* out );
static size_t cpuid_tlb_params( size_t max_tlbs, tlb_param_t* out );
static void   cpuid_print_tlb_params( size_t n, const tlb_param_t* out );

void cpuid( int eax, int ecx, cpuid_t *reg )
{
  __asm__ __volatile__(
          "cpuid\n\t"
          : "=a"(reg->eax),"=b"(reg->ebx),"=c"(reg->ecx),"=d"(reg->edx)
          : "a"(eax),"c"(ecx) );
}

/*****************************************************************/
/**                  READ FUNCTIONS                             **/
/*****************************************************************/
void cpuid_basic_info(basic_info_t* out)
{
  cpuid_t reg;

  // 00H
  cpuid(0x0,0x0,&reg);
  out->max_support = reg.eax;
  sprintf(out->vendor_id,
          "%.4s%.4s%.4s\n",
          (char*)(&(reg.ebx)),
          (char*)(&(reg.edx)),
          (char*)(&(reg.ecx)));
  out->vendor_id[12]='\0';
 
  //reg={0,0,0,0};

  // 01H
  cpuid(0x1,0x0,&reg);

  //out->version = reg.eax;
  out->version.stepping = ( reg.eax      & 0x0f);
  out->version.model    = ((reg.eax>>4 ) & 0x0f);
  out->version.family   = ((reg.eax>>8 ) & 0x0f);
  out->version.proc_type= ((reg.eax>>12) & 0x03);
  out->version.ex_model = ((reg.eax>>16) & 0x03);
  out->version.ex_family= ((reg.eax>>20) & 0xff);

  out->brand_index      = ( reg.ebx      & 0xff);
  out->clflush_linesize = ((reg.ebx>>8 ) & 0xff);
  out->max_address_id   = ((reg.ebx>>16) & 0xff);
  out->initial_apic_id  = ((reg.ebx>>24) & 0xff);

  out->features[0]      = reg.ecx;
  out->features[1]      = reg.edx;

  // 02H
  cpuid(0x2,0x0,&reg);
  out->cache_info[ 0]   = ( reg.eax      & 0xff);
  out->cache_info[ 1]   = ((reg.eax>>8 ) & 0xff);
  out->cache_info[ 2]   = ((reg.eax>>16) & 0xff);
  out->cache_info[ 3]   = ((reg.eax>>24) & 0xff);
  out->cache_info[ 4]   = ( reg.ebx      & 0xff);
  out->cache_info[ 5]   = ((reg.ebx>>8 ) & 0xff);
  out->cache_info[ 6]   = ((reg.ebx>>16) & 0xff);
  out->cache_info[ 7]   = ((reg.ebx>>24) & 0xff);
  out->cache_info[ 8]   = ( reg.ecx      & 0xff);
  out->cache_info[ 9]   = ((reg.ecx>>8 ) & 0xff);
  out->cache_info[10]   = ((reg.ecx>>16) & 0xff);
  out->cache_info[11]   = ((reg.ecx>>24) & 0xff);
  out->cache_info[12]   = ( reg.edx      & 0xff);
  out->cache_info[13]   = ((reg.edx>>8 ) & 0xff);
  out->cache_info[14]   = ((reg.edx>>16) & 0xff);
  out->cache_info[15]   = ((reg.edx>>24) & 0xff);

  // 03H
  cpuid(0x3,0x0,&reg);
  out->proc_serial_number[0]=0;
  out->proc_serial_number[1]=0;
  out->proc_serial_number[2]=0;
  if( out->features[1] & F_PSN ){
    out->proc_serial_number[0]=reg.ecx;
    out->proc_serial_number[1]=reg.edx;
  }

}

// 04H
size_t cpuid_cache_params( size_t n, cache_param_t* out )
{
  cpuid_t reg={0};
  size_t i=0;
  size_t retval=0;
  cache_param_t* p=out;

  for( i=0, p=out; i<n; i++,p++ ){
    // 04H
    cpuid(0x4,i,&reg);
    if( (reg.eax & 0x1f) != 0 ) {
      // eax
      p->cache_type        = ((reg.eax     ) & 0x001f); // 5 bits
      p->cache_level       = ((reg.eax>> 5 ) & 0x0007); // 3 bits
      p->self_init_level   = ((reg.eax>> 8 ) & 0x0001); // 1 bit
      p->assoc_fully       = ((reg.eax>> 9 ) & 0x0001); // 1 bit
      p->max_addr_procs    = ((reg.eax>>14 ) & 0x0fff)+1; // 12 bits
      p->max_addr_cores    = ((reg.eax>>26 ) & 0x003f)+1; // 6 bits
      // ebx
      p->line_size         = ((reg.ebx     ) & 0x0fff)+1; // 12 bits
      p->line_parts        = ((reg.ebx>>12 ) & 0x03ff)+1; // 10 bits
      p->assoc_ways        = ((reg.ebx>>22 ) & 0x03ff)+1; // 10 bits
      // ecx
      p->assoc_sets        = reg.ecx+1;
      // edx
      p->write_back_invd   = ((reg.edx     ) & 0x0001); // 1 bit
      p->cache_inclusive   = ((reg.edx>> 1 ) & 0x0001); // 1 bit
      p->complex_indexing  = ((reg.edx>> 2 ) & 0x0001); // 1 bit
      // extra
      p->cache_size_b = (p->line_size) * (p->line_parts) * (p->assoc_ways) * (p->assoc_sets);
      retval = i+1;
    }else{
      //if( (reg.ebx|reg.ecx|reg.edx)==0 ) retval = 0; // sub-leaf index is invalid.
      break;
    }
  }
  return retval;
}

// 07H
size_t cpuid_extend_features( size_t n, extend_feature_t* out )
{
  cpuid_t reg={0};
  size_t i=0;
  size_t retval=0;
  extend_feature_t* p=out;

  for( i=0, p=out; i<n; i++,p++ ){
    // 07H
    cpuid(0x7,i,&reg);
    if( (reg.eax|reg.ebx|reg.ecx|reg.edx) != 0 ) {
      p->features[0] = reg.eax;
      p->features[1] = reg.ebx;
      p->features[2] = reg.ecx;
      p->features[3] = reg.edx;
      retval = i+1;
    }else{
      break;
    }
  }
  return retval;
}

// 0BH
size_t cpuid_extend_topology( size_t n, extend_topology_t* out )
{
  cpuid_t reg={0};
  size_t i=0;
  size_t retval=0;
  extend_topology_t* p=out;

  for( i=0, p=out; i<n; i++,p++ ){
    // 0BH
    cpuid(0xb,i,&reg);
    if( (reg.eax|reg.ebx|(reg.ecx&0x00f0)) != 0 ) {
      p->shift_apic2topo = ((reg.eax    )&0x001f);
      p->num_log_procs   = ((reg.ebx    )&0xffff);
      p->level_number    = ((reg.ecx    )&0x00ff);
      p->level_type      = ((reg.ecx>>8 )&0x00ff);
      p->apic_id         = ((reg.edx    )       );
      retval = i+1;
    }else{
      break;
    }
  }
  return retval;
}

// 16H
void   cpuid_proc_freq_info( proc_freq_info_t* out )
{
  cpuid_t reg={0};
  cpuid(0x1d,0,&reg);
  out->proc_base_freq = ((reg.eax)&0xffff);
  out->max_freq       = ((reg.ebx)&0xffff);
  out->bus_freq       = ((reg.ecx)&0xffff);
}

// 18H
size_t cpuid_tlb_params( size_t max_tlbs, tlb_param_t* out )
{
  cpuid_t reg={0};
  size_t i=0;
  size_t n=0;
  size_t retval=0;
  tlb_param_t* p=out;

  cpuid(0x18,0,&reg);
  n = reg.eax+1; /// num of leaves
  if( n > max_tlbs ){ fprintf(stderr,"[ERROR] MAX_TLBS is less than size of leaves in cpuid_tlb_params(18H)\n"); return 0; }

  for( i=0, p=out; i<n; i++, p++ ){
    if( reg.eax|reg.ebx|reg.ecx|reg.edx != 0 ){
      // ebx
      p->page_size_flags  = ((reg.ebx    )&0x00ff); // 8 bits
      p->partitioning     = ((reg.ebx>> 8)&0x0008); // 3 bits 
      p->assoc_ways       = ((reg.ebx>>16)&0xffff); //16 bits 
      // ecx
      p->assoc_sets       = ((reg.ecx    )       ); //32 bits
      // edx
      p->tlb_type         = ((reg.edx    )&0x001f); // 5 bits 
      p->tlb_level        = ((reg.edx>> 5)&0x0008); // 3 bits 
      p->assoc_fully      = ((reg.edx>> 8)&0x0001); // 1 bits 
      p->max_addr_procs   = ((reg.edx>>14)&0x0fff); //12 bits
      retval++;
      // next
      cpuid(0x18,i+1,&reg);
    }else{
      break;
    }
  }

  return retval;
}

// 1DH
size_t cpuid_tile_palettes( size_t max_pllevel, tile_info_t* out )
{
  cpuid_t reg={0};
  size_t i=0;
  size_t n=0;
  size_t max_palette=0;
  size_t retval=0;
  tile_info_t* p=out;

  // max_palette
  cpuid(0x1d,0,&reg);
  max_palette = reg.eax;

  //printf("max_palette = %z\n",max_palette);

  if( max_pllevel <= max_palette ){ fprintf(stderr,"[ERROR] MAX_PLLEVEL is less than size of subleaf in cpuid_tile_palettes 1DH\n");  return 0; }// ERROR

  // palettes
  n = max_palette + 1;
  for( i=1, p=out; i<n; i++,p++ ){
    cpuid(0x1d,i,&reg);
    p->total_tile_bytes = ((reg.eax    )&0xffff) ; // 16 bits
    p->bytes_per_tile   = ((reg.eax>>16)&0xffff) ; // 16 bits
    p->bytes_per_row    = ((reg.ebx    )&0xffff) ; // 16 bits
    p->max_names        = ((reg.ebx>>16)&0xffff) ; // 16 bits
    p->max_rows         = ((reg.ecx    )&0xffff) ; // 16 bits
  }
  return max_palette;
}

// 1EH
void cpuid_tmul_info( tmul_info_t* out )
{
  cpuid_t reg={0};

  cpuid(0x1e,0,&reg);
  out->tmul_maxk = ((reg.ebx    )&0x00ff) ; //  8 bits
  out->tmul_maxn = ((reg.ebx>>8 )&0xffff) ; // 16 bits
}


//80000000H
void cpuid_extend_info( extend_info_t* out )
{
  cpuid_t reg={0};
  extend_info_t *p=out;
  unsigned int  *c=(unsigned int *)(p->proc_brand);
  
  // 80000000H
  cpuid(0x80000000,0x0,&reg);
  p->max_supported = reg.eax;

  if( p->max_supported & 0x80000000 ){

    if( p->max_supported >= 0x80000004 ){

      // 80000001H
      cpuid(0x80000001,0x0,&reg);
      p->ex_proc_sign = reg.eax;
      p->features[0]  = reg.ecx;
      p->features[1]  = reg.edx;

      // 80000002H
      cpuid(0x80000002,0x0,&reg);
      *c = reg.eax; c++;
      *c = reg.ebx; c++;
      *c = reg.ecx; c++;
      *c = reg.edx; c++;

      // 80000003H
      cpuid(0x80000003,0x0,&reg);
      *c = reg.eax; c++;
      *c = reg.ebx; c++;
      *c = reg.ecx; c++;
      *c = reg.edx; c++;

      // 80000004H
      cpuid(0x80000004,0x0,&reg);
      *c = reg.eax; c++;
      *c = reg.ebx; c++;
      *c = reg.ecx; c++;
      *c = reg.edx; c++;

      *((unsigned char*)c) = '\0';
    }

    if( p->max_supported >= 0x80000008 ){

      // 80000006H
      cpuid(0x80000006,0x0,&reg);
      p->line_size_b = ((reg.eax    ) & 0x00ff );
      p->l2_assoc_fld= ((reg.eax>>12) & 0x000f );
      p->l2_size_kb  = ((reg.eax>>16) & 0xffff );

      // 80000007H
      cpuid(0x80000007,0x0,&reg);
      p->features[2]  = reg.edx;

      // 80000008H
      cpuid(0x80000008,0x0,&reg);
      p->phys_addr_bits  = ((reg.eax    ) & 0x00ff );
      p->linear_addr_bits= ((reg.eax>>8 ) & 0x00ff );

    }
  }
  
}

/*****************************************************************/
/**                 PRINT FUNCTIONS                             **/
/*****************************************************************/
void cpuid_print_basic_info( const basic_info_t* out )
{
  const char* proc_type_str[4]={
    "Original OEM Processor",
    "Intel OverDirve Processor",
    "Dual processor (not applicable to Intel486 processors)",
    "Intel reserved"
  };

  const char* feature1_item[32]={
    "sse3","pclmulqdq","dtes64","monitor","ds-cpl","vmx","smx","eist",
    "tm2","ssse3","cnxt-id","sdbg","fma","cmpxchg16b","xtpr","pdcm",
    "","pcid","dca","sse4.1","sse4.2","x2apic","movbe","popcnt",
    "tsc-deadline","aesni","xsave","osxsave","avx","f16c","rdrand",""
  };
  const char* feature2_item[32]={
    "fpu","vme","de","pse","tsc","msr","pae","mce",
    "cx8","apic","","sep","mtrr","pge","mca","cmov",
    "pat","pse-36","psn","clfsh","","ds","acpi","mmx",
    "fxsr","sse","sse2","ss","htt","tm","","pbe"
  };

  const char* features[2][32]=
  {
    {
      "sse3","pclmulqdq","dtes64","monitor","ds-cpl","vmx","smx","eist",
      "tm2","ssse3","cnxt-id","sdbg","fma","cmpxchg16b","xtpr","pdcm",
      "","pcid","dca","sse4.1","sse4.2","x2apic","movbe","popcnt",
      "tsc-deadline","aesni","xsave","osxsave","avx","f16c","rdrand",""
    }
   ,{
      "fpu","vme","de","pse","tsc","msr","pae","mce",
      "cx8","apic","","sep","mtrr","pge","mca","cmov",
      "pat","pse-36","psn","clfsh","","ds","acpi","mmx",
      "fxsr","sse","sse2","ss","htt","tm","","pbe"
    }
  };

  unsigned int i,j;
  unsigned int mask;
  unsigned int family;
  unsigned int model;

  const char* cache_description[256]={
     MSGTLB00, MSGTLB01, MSGTLB02, MSGTLB03, MSGTLB04, MSGTLB05, MSGTLB06, MSGTLB07, 
     MSGTLB08, MSGTLB09, MSGTLB0A, MSGTLB0B, MSGTLB0C, MSGTLB0D, MSGTLB0E, MSGTLB0F,
     MSGTLB10, MSGTLB11, MSGTLB12, MSGTLB13, MSGTLB14, MSGTLB15, MSGTLB16, MSGTLB17, 
     MSGTLB18, MSGTLB19, MSGTLB1A, MSGTLB1B, MSGTLB1C, MSGTLB1D, MSGTLB1E, MSGTLB1F,
     MSGTLB20, MSGTLB21, MSGTLB22, MSGTLB23, MSGTLB24, MSGTLB25, MSGTLB26, MSGTLB27, 
     MSGTLB28, MSGTLB29, MSGTLB2A, MSGTLB2B, MSGTLB2C, MSGTLB2D, MSGTLB2E, MSGTLB2F,
     MSGTLB30, MSGTLB31, MSGTLB32, MSGTLB33, MSGTLB34, MSGTLB35, MSGTLB36, MSGTLB37, 
     MSGTLB38, MSGTLB39, MSGTLB3A, MSGTLB3B, MSGTLB3C, MSGTLB3D, MSGTLB3E, MSGTLB3F,
     MSGTLB40, MSGTLB41, MSGTLB42, MSGTLB43, MSGTLB44, MSGTLB45, MSGTLB46, MSGTLB47, 
     MSGTLB48, MSGTLB49, MSGTLB4A, MSGTLB4B, MSGTLB4C, MSGTLB4D, MSGTLB4E, MSGTLB4F,
     MSGTLB50, MSGTLB51, MSGTLB52, MSGTLB53, MSGTLB54, MSGTLB55, MSGTLB56, MSGTLB57, 
     MSGTLB58, MSGTLB59, MSGTLB5A, MSGTLB5B, MSGTLB5C, MSGTLB5D, MSGTLB5E, MSGTLB5F,
     MSGTLB60, MSGTLB61, MSGTLB62, MSGTLB63, MSGTLB64, MSGTLB65, MSGTLB66, MSGTLB67, 
     MSGTLB68, MSGTLB69, MSGTLB6A, MSGTLB6B, MSGTLB6C, MSGTLB6D, MSGTLB6E, MSGTLB6F,
     MSGTLB70, MSGTLB71, MSGTLB72, MSGTLB73, MSGTLB74, MSGTLB75, MSGTLB76, MSGTLB77, 
     MSGTLB78, MSGTLB79, MSGTLB7A, MSGTLB7B, MSGTLB7C, MSGTLB7D, MSGTLB7E, MSGTLB7F,
     MSGTLB80, MSGTLB81, MSGTLB82, MSGTLB83, MSGTLB84, MSGTLB85, MSGTLB86, MSGTLB87, 
     MSGTLB88, MSGTLB89, MSGTLB8A, MSGTLB8B, MSGTLB8C, MSGTLB8D, MSGTLB8E, MSGTLB8F,
     MSGTLB90, MSGTLB91, MSGTLB92, MSGTLB93, MSGTLB94, MSGTLB95, MSGTLB96, MSGTLB97, 
     MSGTLB98, MSGTLB99, MSGTLB9A, MSGTLB9B, MSGTLB9C, MSGTLB9D, MSGTLB9E, MSGTLB9F,
     MSGTLBA0, MSGTLBA1, MSGTLBA2, MSGTLBA3, MSGTLBA4, MSGTLBA5, MSGTLBA6, MSGTLBA7, 
     MSGTLBA8, MSGTLBA9, MSGTLBAA, MSGTLBAB, MSGTLBAC, MSGTLBAD, MSGTLBAE, MSGTLBAF,
     MSGTLBB0, MSGTLBB1, MSGTLBB2, MSGTLBB3, MSGTLBB4, MSGTLBB5, MSGTLBB6, MSGTLBB7, 
     MSGTLBB8, MSGTLBB9, MSGTLBBA, MSGTLBBB, MSGTLBBC, MSGTLBBD, MSGTLBBE, MSGTLBBF,
     MSGTLBC0, MSGTLBC1, MSGTLBC2, MSGTLBC3, MSGTLBC4, MSGTLBC5, MSGTLBC6, MSGTLBC7, 
     MSGTLBC8, MSGTLBC9, MSGTLBCA, MSGTLBCB, MSGTLBCC, MSGTLBCD, MSGTLBCE, MSGTLBCF,
     MSGTLBD0, MSGTLBD1, MSGTLBD2, MSGTLBD3, MSGTLBD4, MSGTLBD5, MSGTLBD6, MSGTLBD7, 
     MSGTLBD8, MSGTLBD9, MSGTLBDA, MSGTLBDB, MSGTLBDC, MSGTLBDD, MSGTLBDE, MSGTLBDF,
     MSGTLBE0, MSGTLBE1, MSGTLBE2, MSGTLBE3, MSGTLBE4, MSGTLBE5, MSGTLBE6, MSGTLBE7, 
     MSGTLBE8, MSGTLBE9, MSGTLBEA, MSGTLBEB, MSGTLBEC, MSGTLBED, MSGTLBEE, MSGTLBEF,
     MSGTLBF0, MSGTLBF1, MSGTLBF2, MSGTLBF3, MSGTLBF4, MSGTLBF5, MSGTLBF6, MSGTLBF7, 
     MSGTLBF8, MSGTLBF9, MSGTLBFA, MSGTLBFB, MSGTLBFC, MSGTLBFD, MSGTLBFE, MSGTLBFF };

  // 00H
  printf("Maximum Input Value                : %02xH\n",out->max_support);
  printf("vender id                          : %12s\n",out->vendor_id);

  // 01H
  printf("proccessor type                    : %s\n",proc_type_str[out->version.proc_type]);

  if( out->version.family != 0x0f ){
    family = out->version.family;
  }else{
    family = out->version.ex_family + out->version.family;
  }
  if( out->version.family == 0x06 || out->version.family == 0x0f ){
    model = (out->version.ex_model<<4) + out->version.model;
  }else{
    model = out->version.model;
  }
  printf("family                             : %02xH\n",family);
  printf("model                              : %02xH\n",model);
  printf("stepping                           : %u\n",out->version.stepping);
  printf("brand index                        : %u\n",out->brand_index);
  printf("clflush line size                  : %uB\n",out->clflush_linesize*8);
  printf("Max num. of addr.                  : %u\n",out->max_address_id);
  printf("Initial APIC ID                    : %u\n",out->initial_apic_id);

  printf("Features                           :");
  for( j=2; j>0 ; j-- ){
    mask=0x01;
    for( i=0; i<32; i++ ){
      if( out->features[j-1] & mask ){
        printf(" %s",features[j-1][i]);
      }
      mask=(mask<<1);
    }
  }
  printf("\n");

  // 02H
  if( out->cache_info[0]==0x01 ){
    for( i=1; i<16; i++ ){
      if( out->cache_info[i] != 0x00 && out->cache_info[i] != 0xff ){
        printf("Cache or TLB                       : ");
        printf("%s",cache_description[out->cache_info[i]]);
        printf("\n");
      }
    }
  }

  // 03H
  if( out->features[1] & F_PSN ){
    printf("Processor Serial No. : %u %u\n",out->proc_serial_number[1],out->proc_serial_number[0]);
  }

}

void cpuid_print_cache_params( size_t n, const cache_param_t* out )
{
  const char* yesno_str[2]={"No","Yes"};
  const char* cache_type_str[4]={"Null","Data Cache","Instruction Cache","Unified Cache"};
  const char* write_back_msg[2]={
     "WBINVD/INVD from threads sharing this cache acts upon lower level caches for threads sharing this cache."
    ,"WBINVD/INVD is not guaranteed to act upon lower level caches of non-originating threads sharing this cache."
  };
  const char* inclusiveness_msg[2]={
     "Cache is not inclusive of lower cache levels."
    ,"Cache is inclusive of lower cache levels."
  };
  const char* complex_indexing_msg[2]={
     "Direct mapped cache."
    ,"A complex function is used to index the cache, potentially using all address bits."
  };
  const cache_param_t* p=out;
 
  size_t i=0;

  for( i=0,p=out; i<n; i++,p++ ){ 
    printf("\n");
    printf("Cache No.                          : %u\n",i);
    printf("Cache Type Field                   : %s\n",cache_type_str[p->cache_type]);
    printf("Cache Level                        : %u\n",p->cache_level);
    printf("Self Init. Cache Lv.               : %u\n",p->self_init_level);
    printf("Fully Aassociative                 : %s\n",yesno_str[p->assoc_fully]);
    printf("Max Shared Logical Processors      : %u\n",p->max_addr_procs);
    printf("Max Packed Processor Cores         : %u\n",p->max_addr_cores);
    printf("System Coherency Line Size (B)     : %u\n",p->line_size);
    printf("Physical Line partitions           : %u\n",p->line_parts);
    printf("Ways of associativity              : %u\n",p->assoc_ways);
    printf("Number of Sets                     : %u\n",p->assoc_sets);
    printf("Write-Back Invalidate/Invalidate   : %s. %s\n",
            yesno_str[p->write_back_invd],write_back_msg[p->write_back_invd]);
    printf("Cache Inclusiveness                : %s. %s\n",
            yesno_str[p->cache_inclusive],inclusiveness_msg[p->cache_inclusive]);
    printf("Complex Cache Indexing             : %s. %s\n",
            yesno_str[p->complex_indexing],complex_indexing_msg[p->complex_indexing]);
    printf("Cache size (B)                     : %zu\n",p->cache_size_b);
    printf("Cache size (KB)                    : %zu\n",(p->cache_size_b)>>10);
    printf("Cache size (MB)                    : %zu\n",(p->cache_size_b)>>20);
  }
}

// 07H
void cpuid_print_extend_features( size_t n, const extend_feature_t* out )
{
  const extend_feature_t* p=out;
  const char* item_null[32]={""};
  const char* item_lv1b[32]={
      "fsgsbase","ia32_tsc_adjust","sgx","bmi1","hle","avx2","","smep"
     ,"bmi2","movsb/stosb","invpcid","rtm","pqm","fpucs/fpuds","impx","pqe"
     ,"avx512f","avx512dq","rdseed","adx","smap","avx512_ifma","","clflushopt"
     ,"clwb","proctrace","avx512pf","avx512er","avx512cd","sha","avx512bw","avx512vl"
  };
  const char* item_lv1c[32]={
      "prefetchwt1","avx512_vbmi","umip","pku","ospke","waitpkg","avx512_vbmi2",""
     ,"gfni","vaes","vpclmulqdq","avx512_vnni","avx512_bitalg","","avx512_vpopcntdq",""
     ,"","","","","","","rdpid",""
     ,"","cldemote","","movdiri","movdir64b","enqcmd","sgx_lc","pks"
  };
  const char* item_lv1d[32]={
      "","","avx512_4vnniw","avx512_4fmaps","fast_short_rep_mov","","",""
     ,"avx512_vp2intersect","","md_clear","","","","serialize","hybrid"
     ,"tsxldtrk","","pconfig","","","","amx-bf16",""
     ,"amx-tile","amx-int8","iprs/ibpb","stibp","","enum_ia32_arch","enum_ia32_core","enum_ssbd"
  };

  const char* item_lv2a[32]={
      "","","","","","avx512-bf16","",""
     ,"","","","","","","",""
     ,"","","","","","","",""
     ,"","","","","","","",""
  };

  const char** features[MAX_FTLEVEL][4];
  size_t i,j,k,j0;
  unsigned int mask;

  size_t max_subleaves = out->features[0];

  for( k=0; k<MAX_FTLEVEL; k++ ){
    features[k][0]=item_null;
    features[k][1]=item_null;
    features[k][2]=item_null;
    features[k][3]=item_null;
  } 

  features[0][1]=item_lv1b;
  features[0][2]=item_lv1c;
  features[0][3]=item_lv1d;

  features[1][0]=item_lv2a;

  printf("\n");
  for( k=0,p=out; k<n; k++,p++ ){ 
    printf("Extended Features                  : ");
    j0=(k==0?1:0);
    for( j=j0; j<4 ; j++ ){
      mask=0x01;
      for( i=0; i<32; i++ ){
        if( p->features[j] & mask ){
          printf("%s ",features[k][j][i]);
        }
        mask=(mask<<1);
      }
    }
    printf("\n");
  }
}
// 0BH
void cpuid_print_extend_topology( size_t n, const extend_topology_t* out )
{
  const extend_topology_t* p=out;
  size_t k;

  for( k=0,p=out; k<n; k++,p++ ){ 
    printf("\n");
    printf("Shift size from x2APIC to Topology : %u\n",p->shift_apic2topo);
    printf("Number of logical processors       : %u\n",p->num_log_procs  );
    printf("Level number                       : %u\n",p->level_number   );
    printf("Level type                         : ");
    if      ( p->level_type == 0 ){
      printf("invalid\n");
    }else if( p->level_type == 1 ){
      printf("SMT\n");
    }else if( p->level_type == 2 ){
      printf("Core\n");
    }else{
      printf("unkown (error)\n");
    }
    printf("x2APIC ID                          : %u\n",p->apic_id        );
  }

}

// 16H
void cpuid_print_proc_freq_info( const proc_freq_info_t* out )
{
  printf("\n");
  printf("Processor Base Frequency (MHz)     : %u\n",out->proc_base_freq);
  printf("Maximum Frequency (MHz)            : %u\n",out->max_freq);
  printf("Bus (Reference) Frequency (MHz)    : %u\n",out->bus_freq);
}

// 18H
void cpuid_print_tlb_params( size_t n, const tlb_param_t* out )
{
  const char* yesno_str[2]={"No","Yes"};
  const char* tlb_type_str[6]={"Null","Data TLB","Instruction TLB","Unified TLB","Load Only TLB","Store Only TLB"};
  const char* page_size_str[4]={
     "4K page size entries supported by this structure."
    ,"2MB page size entries supported by this structure."
    ,"4MB page size entries supported by this structure."
    ,"1 GB page size entries supported by this structure."
  };
  size_t i=0;
  size_t j=0;
  const tlb_param_t* p;

  for( i=0, p=out; i<n; i++,p++ ){ 
    printf("\n");
    printf("TLB No.                            : %u\n",i);
    printf("Translation Cache Type             : %s\n",tlb_type_str[p->tlb_type]);
    printf("Translation Cache Level            : %u\n",p->tlb_level);
    printf("Fully Aassociative                 : %s\n",yesno_str[p->assoc_fully]);
    printf("Max Shared Logical Processors      : %u\n",p->max_addr_procs);
    printf("Partitioning (0=soft partition)    : %u\n",p->partitioning);
    printf("Ways of associativity              : %u\n",p->assoc_ways);
    printf("Number of Sets                     : %u\n",p->assoc_sets);
    for( j=0; j<4 ; j++ ){
      if( ((p->page_size_flags)>>j)&0x1 ){
        printf("Supported Page size                : %s\n",page_size_str[j]);
      }
    }
  }

}


// 1DH
void cpuid_print_tile_palettes( size_t max_palette, const tile_info_t* out )
{
  const tile_info_t* p=out;
  size_t k;

  for( k=0,p=out; k<max_palette; k++,p++ ){ 
    printf("\n");
    printf("Palette No.                        : %u\n",k);
    printf("Total tile bytes                   : %u\n",p->total_tile_bytes);
    printf("Bytes per tile                     : %u\n",p->bytes_per_tile  );
    printf("Bytes per row                      : %u\n",p->bytes_per_row   );
    printf("Number of tile registers           : %u\n",p->max_names       );
    printf("Max rows                           : %u\n",p->max_rows        );
  }

}

// 1EH
void cpuid_print_tmul_info( const tmul_info_t* out )
{
  printf("Max K in Tile Multiply (TMUL)      : %u\n",out->tmul_maxk);
  printf("Max N in Tile Multiply (TMUL)      : %u\n",out->tmul_maxn);
}



// 80000000H
void cpuid_print_extend_info( const extend_info_t* out )
{
  const char* features[4][32]=
  {
    {
      "lahf/sahf ","","","","","lzcnt ","",""
     ,"prefetchw ","","","","","","",""
     ,"","","","","","","",""
     ,"","","","","","","",""
    }
   ,{
      "","","","","","","",""
     ,"","","","syscall/sysret ","","","",""
     ,"","","","","exe_disable ","","",""
     ,"","","1gb_page ","rdtscp ","","intel64 ","",""
    }
   ,{
      "","","","","","","",""
     ,"invaliant_tsc","","","","","","",""
     ,"","","","","","","",""
     ,"","","","","","","",""
    } 
   ,{ // 0x80000008/ebx
      "","","","","","","",""
     ,"","wbnoinvd","","","","","",""
     ,"","","","","","","",""
     ,"","","","","","","",""
    } 
  };
  unsigned int i,j,mask;
  const extend_info_t *p=out;
  
  if( p->max_supported & 0x80000000 ){

    if( p->max_supported >= 0x80000004 ){

      printf("\n");
      printf("Extended Processor Signature       : %02xH\n",p->ex_proc_sign);

      printf("Extended Processor Information     : ");
      for( j=0; j<4 ; j++ ){
        mask=0x01;
        for( i=0; i<32; i++ ){
          if( p->features[j] & mask ){
            printf("%s",features[j][i]);
          }
          mask=(mask<<1);
        }
      }
      printf("\n");

      printf("Processor Brand String             : %s\n",p->proc_brand);
    }

    if( p->max_supported >= 0x80000008 ){

      printf("Cache Line size (B)                : %u\n",p->line_size_b);
      printf("L2 Associativity field             : ");
      if( p->l2_assoc_fld == 0x00 ) {
        printf("Disabled\n");
      }else if( p->l2_assoc_fld == 0x01 ) {
        printf("Direct mapped\n");
      }else if( p->l2_assoc_fld == 0x02 ) {
        printf("2-way\n");
      }else if( p->l2_assoc_fld == 0x04 ) {
        printf("4-way\n");
      }else if( p->l2_assoc_fld == 0x06 ) {
        printf("8-way\n");
      }else if( p->l2_assoc_fld == 0x08 ) {
        printf("16-way\n");
      }else if( p->l2_assoc_fld == 0x0f ) {
        printf("Fully associative\n");
      }else{
        printf("unknown (error)\n");
      }
      printf("Cache size (KB)                    : %u\n",p->l2_size_kb);
      printf("Physical Address Bits              : %u\n",p->phys_addr_bits);
      printf("Linear   Address Bits              : %u\n",p->linear_addr_bits);

    }
  }
  
}

void read_cpuid_info( cpuid_info_t* out )
{
  cpuid_basic_info(&(out->basic_info));
  cpuid_extend_info(&(out->extend_info));

  //printf("max support : 0x%x\n",out->basic_info.max_support);

  // 04H
  if( out->basic_info.max_support >= 0x04 ){
    out->num_caches = 
     cpuid_cache_params(MAX_CACHES,out->cache_info);
  }else{
    out->num_caches = 0;
  }

  // 07H
  if( out->basic_info.max_support >= 0x07 ){
    out->num_ftlevel = 
     cpuid_extend_features(MAX_FTLEVEL,out->more_feature);
  }else{
    out->num_ftlevel = 0;
  }

  // 0BH
  if( out->basic_info.max_support >= 0x0B ){
    out->num_tplevel = 
     cpuid_extend_topology(MAX_TPLEVEL,out->topology);
  }else{
    out->num_tplevel = 0;
  }

  // 16H
  if( out->basic_info.max_support >= 0x16 ){
     cpuid_proc_freq_info(&(out->proc_freq_info));
  } 

  // 18H
  if( out->basic_info.max_support >= 0x18 ){
    out->num_tlbs =
     cpuid_tlb_params(MAX_TLBS,out->tlb_info);
  }else{
    out->num_tlbs = 0;
  } 

  // 1DH
  if( out->basic_info.max_support >= 0x1D ){
    out->max_palette = 
     cpuid_tile_palettes(MAX_PLLEVEL,out->tile_info);
  }else{
    out->max_palette = 0;
  }

  // 1EH
  if( out->basic_info.max_support >= 0x1E ){
     cpuid_tmul_info(&(out->tmul_info));
  } 

}

void write_cpuid_info( const cpuid_info_t* out )
{
  cpuid_print_basic_info(&(out->basic_info));
  cpuid_print_extend_info(&(out->extend_info));

  // 16H
  if( out->basic_info.max_support >= 0x16 ){
     cpuid_print_proc_freq_info(&(out->proc_freq_info));
  } 
  // 07H
  if( out->basic_info.max_support >= 0x07 ){
    cpuid_print_extend_features(out->num_ftlevel,out->more_feature);
  }
  // 0BH
  if( out->basic_info.max_support >= 0x0B ){
    cpuid_print_extend_topology(out->num_ftlevel,out->topology);
  }
  // 04H
  if( out->basic_info.max_support >= 0x04 ){
    cpuid_print_cache_params(out->num_caches,out->cache_info);
  }
  // 18H
  if( out->basic_info.max_support >= 0x18 ){
    cpuid_print_tlb_params(out->num_tlbs,out->tlb_info);
  }
  // 1EH
  if( out->basic_info.max_support >= 0x1E ){
     cpuid_print_tmul_info(&(out->tmul_info));
  } 
  // 1DH
  if( out->basic_info.max_support >= 0x1D ){
     cpuid_print_tile_palettes(out->max_palette,out->tile_info);
  }
}

