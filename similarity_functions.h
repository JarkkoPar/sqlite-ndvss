#ifndef SIMILARITY_FUNCTIONS_H_INCLUDED 
#define SIMILARITY_FUNCTIONS_H_INCLUDED 
/** 
 * This header contains the C-macros to create the SQLite3 interface functions for 
 * the various similarity functions.
 */
#include "similarity_functions_basic.h"
#include "similarity_functions_avx.h"
#include "similarity_functions_avx2.h"
#include "similarity_functions_avx512f.h"



/**
 * Function generation macros. These will create the fucntions into sqlite-ndvss.c for 
 * registering them into sqlite.
 */


//----------------------------------------------------------------------------------------
// Name: ndvss_cosine_similarity_f
// Desc: Calculates the cosine similarity to a BLOB-converted array of floats.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as an angle DOUBLE
//----------------------------------------------------------------------------------------
#define GENERATE_COSINE_FUNC_F(NAME, SIMILARITY_FUNC) \
static void ndvss_cosine_similarity_f_##NAME( sqlite3_context *context \
                                             ,int argc \
                                             ,sqlite3_value **argv) \
{ \
    if( argc < 2 ) { \
        sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); \
        return; \
    } \
    if( sqlite3_value_type(argv[0]) == SQLITE_NULL || \
        sqlite3_value_type(argv[1]) == SQLITE_NULL ) { \
        sqlite3_result_error(context, "One of the required arguments is null.", -1); \
        return; \
    } \
    int arg1_size_bytes = sqlite3_value_bytes(argv[0]); \
    int arg2_size_bytes = sqlite3_value_bytes(argv[1]); \
    if( arg1_size_bytes != arg2_size_bytes ) { \
        sqlite3_result_error(context, "The arrays are not the same length.", -1); \
        return; \
    } \
    int vector_size = -1; \
    if( argc > 2 ) { \
        if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { \
            vector_size = sqlite3_value_int(argv[2]); \
        } \
    } \
    if( vector_size < 1 ) { \
        vector_size = arg1_size_bytes / sizeof(float); \
    } \
    const float* searched_array = (const float *)sqlite3_value_blob(argv[0]); \
    const float* column_array = (const float *)sqlite3_value_blob(argv[1]); \
    float dividerA = 0.0f; \
    float dividerB = 0.0f; \
    float similarity = SIMILARITY_FUNC( searched_array, column_array, vector_size, &dividerA, &dividerB ); \
    if( dividerA == 0.0f || dividerB == 0.0f ) { \
        /* There'd be a division by zero, so assume no similarity. */ \
        sqlite3_result_error(context, "Division by zero.", -1); \
        return; \
    } \
    float divider = sqrtf(dividerA * dividerB); \
    similarity = similarity / divider; \
    sqlite3_result_double(context, (double)similarity); \
}



//----------------------------------------------------------------------------------------
// Name: ndvss_cosine_similarity_d
// Desc: Calculates the cosine similarity to a BLOB-converted array of doubles.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as an angle DOUBLE
//----------------------------------------------------------------------------------------
#define GENERATE_COSINE_FUNC_D(NAME, SIMILARITY_FUNC) \
static void ndvss_cosine_similarity_d_##NAME( sqlite3_context *context \
                                             ,int argc \
                                             ,sqlite3_value **argv) \
{ \
    if( argc < 2 ) { \
        sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); \
        return; \
    } \
    if( sqlite3_value_type(argv[0]) == SQLITE_NULL || \
        sqlite3_value_type(argv[1]) == SQLITE_NULL ) { \
        sqlite3_result_error(context, "One of the required arguments is null.", -1); \
        return; \
    } \
    int arg1_size_bytes = sqlite3_value_bytes(argv[0]); \
    int arg2_size_bytes = sqlite3_value_bytes(argv[1]); \
    if( arg1_size_bytes != arg2_size_bytes ) { \
        sqlite3_result_error(context, "The arrays are not the same length.", -1); \
        return; \
    } \
    int vector_size = -1; \
    if( argc > 2 ) { \
        if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { \
            vector_size = sqlite3_value_int(argv[2]); \
        } \
    } \
    if( vector_size < 1 ) { \
        vector_size = arg1_size_bytes / sizeof(double); \
    } \
    const double* searched_array = (const double *)sqlite3_value_blob(argv[0]); \
    const double* column_array = (const double *)sqlite3_value_blob(argv[1]); \
    double dividerA = 0.0; \
    double dividerB = 0.0; \
    double similarity = SIMILARITY_FUNC( searched_array, column_array, vector_size, &dividerA, &dividerB ); \
    if( dividerA == 0.0 || dividerB == 0.0 ) { \
        /* There'd be a division by zero, so assume no similarity. */ \
        sqlite3_result_error(context, "Division by zero.", -1); \
        return; \
    } \
    double divider = sqrt(dividerA * dividerB); \
    similarity = similarity / divider; \
    sqlite3_result_double(context, similarity); \
}



//----------------------------------------------------------------------------------------
// Name: ndvss_euclidean_distance_similarity_f
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of floats.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
#define GENERATE_EUCLIDEAN_FUNC_F(NAME, SIMILARITY_FUNC) \
static void ndvss_euclidean_distance_similarity_f_##NAME( sqlite3_context* context, \
                                                   int argc, \
                                                   sqlite3_value** argv ) \
{ \
  if( argc < 2 ) { \
    /* Not enough arguments.*/ \
    sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); \
    return; \
  } \
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL || \
      sqlite3_value_type(argv[1]) == SQLITE_NULL ) { \
    /* Missing one of the required arguments. */ \
    sqlite3_result_error(context, "One of the given arguments is null.", -1); \
    return; \
  } \
  int arg1_size_bytes = sqlite3_value_bytes(argv[0]); \
  int arg2_size_bytes = sqlite3_value_bytes(argv[1]); \
  if( arg1_size_bytes != arg2_size_bytes ) { \
    sqlite3_result_error(context, "The arrays are not the same length.", -1); \
    return; \
  } \
  int vector_size = -1; \
  if( argc > 2 ) { \
    if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { \
      vector_size = sqlite3_value_int(argv[2]); \
    } \
  } \
  if( vector_size < 1 ) { \
    vector_size = arg1_size_bytes / sizeof(float); \
  } \
  const float* searched_array = (const float *)sqlite3_value_blob(argv[0]); \
  const float* column_array = (const float *)sqlite3_value_blob(argv[1]); \
  float similarity = SIMILARITY_FUNC( searched_array, column_array, vector_size ); \
  similarity = sqrtf(similarity); \
  sqlite3_result_double(context, (double)similarity); \
} 




//----------------------------------------------------------------------------------------
// Name: ndvss_euclidean_distance_similarity_d
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of doubles.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
#define GENERATE_EUCLIDEAN_FUNC_D(NAME, SIMILARITY_FUNC) \
static void ndvss_euclidean_distance_similarity_d_##NAME( sqlite3_context* context, \
                                                          int argc, \
                                                          sqlite3_value** argv ) \
{ \
  if( argc < 2 ) { \
    /* Not enough arguments.*/ \
    sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); \
    return; \
  } \
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL || \
      sqlite3_value_type(argv[1]) == SQLITE_NULL ) { \
    /* Missing one of the required arguments. */ \
    sqlite3_result_error(context, "One of the given arguments is null.", -1); \
    return; \
  } \
  int arg1_size_bytes = sqlite3_value_bytes(argv[0]); \
  int arg2_size_bytes = sqlite3_value_bytes(argv[1]); \
  if( arg1_size_bytes != arg2_size_bytes ) { \
    sqlite3_result_error(context, "The arrays are not the same length.", -1); \
    return; \
  } \
  int vector_size = -1; \
  if( argc > 2 ) { \
    if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { \
      vector_size = sqlite3_value_int(argv[2]); \
    } \
  } \
  if( vector_size < 1 ) { \
    vector_size = arg1_size_bytes / sizeof(double); \
  } \
  const double* searched_array = (const double *)sqlite3_value_blob(argv[0]); \
  const double* column_array = (const double *)sqlite3_value_blob(argv[1]); \
  double similarity = SIMILARITY_FUNC( searched_array, column_array, vector_size ); \
  similarity = sqrt(similarity); \
  sqlite3_result_double(context, similarity); \
} 


//----------------------------------------------------------------------------------------
// Name: ndvss_euclidean_distance_similarity_squared_f
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of floats.
//       Uses the non-squared result (faster).
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
#define GENERATE_EUCLIDEAN_SQUARED_FUNC_F(NAME, SIMILARITY_FUNC) \
static void ndvss_euclidean_distance_similarity_squared_f_##NAME( sqlite3_context* context, \
                                                                  int argc, \
                                                                  sqlite3_value** argv ) \
{ \
  if( argc < 2 ) { \
    /* Not enough arguments.*/ \
    sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); \
    return; \
  } \
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL || \
      sqlite3_value_type(argv[1]) == SQLITE_NULL ) { \
    /* Missing one of the required arguments. */ \
    sqlite3_result_error(context, "One of the given arguments is null.", -1); \
    return; \
  } \
  int arg1_size_bytes = sqlite3_value_bytes(argv[0]); \
  int arg2_size_bytes = sqlite3_value_bytes(argv[1]); \
  if( arg1_size_bytes != arg2_size_bytes ) { \
    sqlite3_result_error(context, "The arrays are not the same length.", -1); \
    return; \
  } \
  int vector_size = -1; \
  if( argc > 2 ) { \
    if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { \
      vector_size = sqlite3_value_int(argv[2]); \
    } \
  } \
  if( vector_size < 1 ) { \
    vector_size = arg1_size_bytes / sizeof(float); \
  } \
  const float* searched_array = (const float *)sqlite3_value_blob(argv[0]); \
  const float* column_array = (const float *)sqlite3_value_blob(argv[1]); \
  float similarity = SIMILARITY_FUNC( searched_array, column_array, vector_size ); \
  sqlite3_result_double(context, (double)similarity); \
} 


//----------------------------------------------------------------------------------------
// Name: ndvss_euclidean_distance_similarity_squared_d
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of doubles.
//       Uses the non-squared result (faster).
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
#define GENERATE_EUCLIDEAN_SQUARED_FUNC_D(NAME, SIMILARITY_FUNC) \
static void ndvss_euclidean_distance_similarity_squared_d_##NAME( sqlite3_context* context, \
                                                                  int argc, \
                                                                  sqlite3_value** argv ) \
{ \
  if( argc < 2 ) { \
    /* Not enough arguments.*/ \
    sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); \
    return; \
  } \
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL || \
      sqlite3_value_type(argv[1]) == SQLITE_NULL ) { \
    /* Missing one of the required arguments. */ \
    sqlite3_result_error(context, "One of the given arguments is null.", -1); \
    return; \
  } \
  int arg1_size_bytes = sqlite3_value_bytes(argv[0]); \
  int arg2_size_bytes = sqlite3_value_bytes(argv[1]); \
  if( arg1_size_bytes != arg2_size_bytes ) { \
    sqlite3_result_error(context, "The arrays are not the same length.", -1); \
    return; \
  } \
  int vector_size = -1; \
  if( argc > 2 ) { \
    if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { \
      vector_size = sqlite3_value_int(argv[2]); \
    } \
  } \
  if( vector_size < 1 ) { \
    vector_size = arg1_size_bytes / sizeof(double); \
  } \
  const double* searched_array = (const double *)sqlite3_value_blob(argv[0]); \
  const double* column_array = (const double *)sqlite3_value_blob(argv[1]); \
  double similarity = SIMILARITY_FUNC( searched_array, column_array, vector_size ); \
  sqlite3_result_double(context, similarity); \
} 


//----------------------------------------------------------------------------------------
// Name: ndvss_dot_product_similarity_f
// Desc: Calculates the dot product similarity to a BLOB-converted array of floats.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
#define GENERATE_DOT_PRODUCT_FUNC_F(NAME, SIMILARITY_FUNC) \
static void ndvss_dot_product_similarity_f_##NAME( sqlite3_context* context, \
                                                   int argc, \
                                                   sqlite3_value** argv ) \
{ \
  if( argc < 2 ) { \
    /* Not enough arguments. */ \
    sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, array length.", -1); \
    return; \
  } \
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL || \
      sqlite3_value_type(argv[1]) == SQLITE_NULL ) { \
    /* Missing one of the required arguments. */ \
    sqlite3_result_error(context, "One of the given arguments is NULL.", -1); \
    return; \
  } \
  int arg1_size_bytes = sqlite3_value_bytes(argv[0]); \
  int arg2_size_bytes = sqlite3_value_bytes(argv[1]); \
  if( arg1_size_bytes != arg2_size_bytes ) { \
    sqlite3_result_error(context, "The arrays are not the same length.", -1); \
    return; \
  } \
  int vector_size = -1; \
  if( argc > 2 ) { \
    if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { \
      vector_size = sqlite3_value_int(argv[2]); \
    } \
  }  \
  if( vector_size < 1 ) { \
    vector_size = arg1_size_bytes / sizeof(float);\
  } \
  const float* searched_array = (const float *)sqlite3_value_blob(argv[0]); \
  const float* column_array = (const float *)sqlite3_value_blob(argv[1]); \
  float similarity = SIMILARITY_FUNC( searched_array, column_array, vector_size ); \
  sqlite3_result_double(context, (double)similarity); \
}



//----------------------------------------------------------------------------------------
// Name: ndvss_dot_product_similarity_d
// Desc: Calculates the dot product similarity to a BLOB-converted array of doubles.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
#define GENERATE_DOT_PRODUCT_FUNC_D(NAME, SIMILARITY_FUNC) \
static void ndvss_dot_product_similarity_d_##NAME( sqlite3_context* context, \
                                            int argc, \
                                            sqlite3_value** argv ) \
{ \
  if( argc < 2 ) { \
    /* Not enough arguments. */ \
    sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, array length.", -1); \
    return; \
  } \
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL || \
      sqlite3_value_type(argv[1]) == SQLITE_NULL ) { \
    /* Missing one of the required arguments. */ \
    sqlite3_result_error(context, "One of the given arguments is NULL.", -1); \
    return; \
  } \
  int arg1_size_bytes = sqlite3_value_bytes(argv[0]); \
  int arg2_size_bytes = sqlite3_value_bytes(argv[1]); \
  if( arg1_size_bytes != arg2_size_bytes ) { \
    sqlite3_result_error(context, "The arrays are not the same length.", -1); \
    return; \
  } \
  int vector_size = -1; \
  if( argc > 2 ) { \
    if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { \
      vector_size = sqlite3_value_int(argv[2]); \
    } \
  }  \
  if( vector_size < 1 ) { \
    vector_size = arg1_size_bytes / sizeof(double);\
  } \
  const double* searched_array = (const double *)sqlite3_value_blob(argv[0]); \
  const double* column_array = (const double *)sqlite3_value_blob(argv[1]); \
  double similarity = SIMILARITY_FUNC( searched_array, column_array, vector_size ); \
  sqlite3_result_double(context, similarity); \
}



/**
 * Function instantiation using the macros.
 */

// Cosine similarity functions.
GENERATE_COSINE_FUNC_F(basic, cosine_similarity_f_basic)

GENERATE_COSINE_FUNC_F(avx,  cosine_similarity_f_avx)

GENERATE_COSINE_FUNC_F(avx2,  cosine_similarity_f_avx2)

GENERATE_COSINE_FUNC_F(avx512f,  cosine_similarity_f_avx512f)

GENERATE_COSINE_FUNC_D(basic, cosine_similarity_d_basic)

GENERATE_COSINE_FUNC_D(avx,  cosine_similarity_d_avx)

GENERATE_COSINE_FUNC_D(avx2,  cosine_similarity_d_avx2)

GENERATE_COSINE_FUNC_D(avx512f,  cosine_similarity_d_avx512f)

// Euclidean distance similarity functions. 
GENERATE_EUCLIDEAN_FUNC_F(basic, euclidean_distance_similarity_f_basic)

GENERATE_EUCLIDEAN_FUNC_F(avx,  euclidean_distance_similarity_f_avx)

GENERATE_EUCLIDEAN_FUNC_F(avx2,  euclidean_distance_similarity_f_avx2)

GENERATE_EUCLIDEAN_FUNC_F(avx512f,  euclidean_distance_similarity_f_avx512f)

GENERATE_EUCLIDEAN_FUNC_D(basic, euclidean_distance_similarity_d_basic)

GENERATE_EUCLIDEAN_FUNC_D(avx,  euclidean_distance_similarity_d_avx)

GENERATE_EUCLIDEAN_FUNC_D(avx2,  euclidean_distance_similarity_d_avx2)

GENERATE_EUCLIDEAN_FUNC_D(avx512f,  euclidean_distance_similarity_d_avx512f)

// Euclidean distance squared similarity functions. 
GENERATE_EUCLIDEAN_SQUARED_FUNC_F(basic, euclidean_distance_similarity_f_basic)

GENERATE_EUCLIDEAN_SQUARED_FUNC_F(avx,  euclidean_distance_similarity_f_avx)

GENERATE_EUCLIDEAN_SQUARED_FUNC_F(avx2,  euclidean_distance_similarity_f_avx2)

GENERATE_EUCLIDEAN_SQUARED_FUNC_F(avx512f,  euclidean_distance_similarity_f_avx512f)

GENERATE_EUCLIDEAN_SQUARED_FUNC_D(basic, euclidean_distance_similarity_d_basic)

GENERATE_EUCLIDEAN_SQUARED_FUNC_D(avx,  euclidean_distance_similarity_d_avx)

GENERATE_EUCLIDEAN_SQUARED_FUNC_D(avx2,  euclidean_distance_similarity_d_avx2)

GENERATE_EUCLIDEAN_SQUARED_FUNC_D(avx512f,  euclidean_distance_similarity_d_avx512f)

// Dot product similarity functions. 
GENERATE_DOT_PRODUCT_FUNC_F(basic, dot_product_similarity_f_basic)

GENERATE_DOT_PRODUCT_FUNC_F(avx,  dot_product_similarity_f_avx)

GENERATE_DOT_PRODUCT_FUNC_F(avx2,  dot_product_similarity_f_avx2)

GENERATE_DOT_PRODUCT_FUNC_F(avx512f,  dot_product_similarity_f_avx512f)

GENERATE_DOT_PRODUCT_FUNC_D(basic, dot_product_similarity_d_basic)

GENERATE_DOT_PRODUCT_FUNC_D(avx,  dot_product_similarity_d_avx)

GENERATE_DOT_PRODUCT_FUNC_D(avx2,  dot_product_similarity_d_avx2)

GENERATE_DOT_PRODUCT_FUNC_D(avx512f,  dot_product_similarity_d_avx512f)



#endif 