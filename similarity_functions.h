//#ifndef SIMILARITY_FUNCTIONS_H_INCLUDED 
//#define SIMILARITY_FUNCTIONS_H_INCLUDED 
/** 
 * This header contains the C-macros to create the SQLite3 interface functions for 
 * the various similarity functions.
 */
#include "similarity_functions_basic.h"

#if defined(__aarch64__)

#include "similarity_functions_neon.h"

#elif defined(__riscv) && defined(__riscv_vector)

#include "similarity_functions_rvv.h"

#elif defined(__riscv)

// This defaults to the basic functions.

#else

#include "similarity_functions_sse41.h"
#include "similarity_functions_avx.h"
#include "similarity_functions_avx2.h"
#include "similarity_functions_avx512f.h"

#endif 


//#endif 