#include <stdio.h>

void myblas_xerbla( const char* name, int info ){

  printf(" ** On entry to %s parameter number %d had an illegal value\n",name,info);

}
