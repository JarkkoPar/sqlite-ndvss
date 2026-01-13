/*
** This SQLite extension implements functions to perform vector
** similarity searches. 
*/
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
#include <cpuid.h>
#endif 

#include "similarity_functions.h"


#define NDVSS_VERSION_DOUBLE    0.50


//========================================================================================
// Similarity function handling.
//========================================================================================

// Function pointer types for the basic similarity functions. 
typedef float  (*similarity_function_f)(const float*,  const float*,  const int);
typedef double (*similarity_function_d)(const double*, const double*, const int);

// Function pointer types for the cosine similarity functions. 
typedef float  (*cos_similarity_function_f)(const float*,  const float*,  const int, float*,  float*);
typedef double (*cos_similarity_function_d)(const double*, const double*, const int, double*, double*);

// The function pointers. These are set as the basic implementation by default.

// Cosine specific
static cos_similarity_function_f cosine_func_f  = cosine_similarity_f_basic;
static cos_similarity_function_d cosine_func_d  = cosine_similarity_d_basic;
    
// Standard metrics
static similarity_function_f euclidean_func_f   = euclidean_distance_similarity_f_basic;
static similarity_function_d euclidean_func_d   = euclidean_distance_similarity_d_basic;
static similarity_function_f dot_product_func_f = dot_product_similarity_f_basic;
static similarity_function_d dot_product_func_d = dot_product_similarity_d_basic;

// Information about which instruction set is in use. 
static char g_instruction_set[8] = "none";

// String helper
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// Function to load the functions.
#define LOAD_SIMILARITY_FUNCTIONS(SUFFIX) \
    cosine_func_f            = cosine_similarity_f_##SUFFIX; \
    cosine_func_d            = cosine_similarity_d_##SUFFIX; \
    euclidean_func_f         = euclidean_distance_similarity_f_##SUFFIX; \
    euclidean_func_d         = euclidean_distance_similarity_d_##SUFFIX; \
    dot_product_func_f       = dot_product_similarity_f_##SUFFIX; \
    dot_product_func_d       = dot_product_similarity_d_##SUFFIX; \
    snprintf(g_instruction_set, 8, "%s", STR(SUFFIX) );



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
static void ndvss_cosine_similarity_f( sqlite3_context *context 
                                      ,int argc 
                                      ,sqlite3_value **argv ) 
{ 
    int vector_size = -1; 
    if( argc < 2 ) { 
        sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); 
        return; 
    } 
    if( sqlite3_value_type(argv[0]) == SQLITE_NULL || 
        sqlite3_value_type(argv[1]) == SQLITE_NULL ) { 
        sqlite3_result_error(context, "One of the required arguments is null.", -1); 
        return; 
    } 
    int arg1_size_bytes = sqlite3_value_bytes(argv[0]); 
    int arg2_size_bytes = sqlite3_value_bytes(argv[1]); 
    if( arg1_size_bytes != arg2_size_bytes ) { 
        sqlite3_result_error(context, "The arrays are not the same length.", -1); 
        return; 
    } 
    
    if( argc > 2 ) { 
        if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { 
            vector_size = sqlite3_value_int(argv[2]); 
        } 
    } 
    if( vector_size < 1 ) { 
        vector_size = arg1_size_bytes / sizeof(float); 
    } 
    float* searched_array = (float *)sqlite3_value_blob(argv[0]); 
    float* column_array   = (float *)sqlite3_value_blob(argv[1]); 
    float dividerA = 0.0f; 
    float dividerB = 0.0f; 
    float similarity = cosine_func_f( searched_array, column_array, vector_size, &dividerA, &dividerB ); 
    if( dividerA == 0.0f || dividerB == 0.0f ) { 
        /* There'd be a division by zero, so assume no similarity. */ 
        sqlite3_result_double(context, 0.0);
        return; 
    } 
    float divider = sqrtf(dividerA * dividerB); 
    similarity = similarity / divider; 
    sqlite3_result_double(context, (double)similarity); 
}



//----------------------------------------------------------------------------------------
// Name: ndvss_cosine_similarity_d
// Desc: Calculates the cosine similarity to a BLOB-converted array of doubles.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as an angle DOUBLE
//----------------------------------------------------------------------------------------
static void ndvss_cosine_similarity_d( sqlite3_context *context 
                                      ,int argc 
                                      ,sqlite3_value **argv) 
{ 
    if( argc < 2 ) { 
        sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); 
        return; 
    } 
    if( sqlite3_value_type(argv[0]) == SQLITE_NULL || 
        sqlite3_value_type(argv[1]) == SQLITE_NULL ) { 
        sqlite3_result_error(context, "One of the required arguments is null.", -1); 
        return; 
    } 
    int arg1_size_bytes = sqlite3_value_bytes(argv[0]); 
    int arg2_size_bytes = sqlite3_value_bytes(argv[1]); 
    if( arg1_size_bytes != arg2_size_bytes ) { 
        sqlite3_result_error(context, "The arrays are not the same length.", -1); 
        return; 
    } 
    int vector_size = -1; 
    if( argc > 2 ) { 
        if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { 
            vector_size = sqlite3_value_int(argv[2]); 
        } 
    } 
    if( vector_size < 1 ) { 
        vector_size = arg1_size_bytes / sizeof(double); 
    } 
    const double* searched_array = (const double *)sqlite3_value_blob(argv[0]); 
    const double* column_array = (const double *)sqlite3_value_blob(argv[1]); 
    double dividerA = 0.0; 
    double dividerB = 0.0; 
    double similarity = cosine_func_d( searched_array, column_array, vector_size, &dividerA, &dividerB ); 
    if( dividerA == 0.0 || dividerB == 0.0 ) { 
        /* There'd be a division by zero, so assume no similarity. */ 
        sqlite3_result_double(context, 0.0);    
        return; 
    } 
    double divider = sqrt(dividerA * dividerB); 
    similarity = similarity / divider; 
    sqlite3_result_double(context, similarity); 
}



//----------------------------------------------------------------------------------------
// Name: ndvss_euclidean_distance_similarity_f
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of floats.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
static void ndvss_euclidean_distance_similarity_f( sqlite3_context* context, 
                                                   int argc, 
                                                   sqlite3_value** argv ) 
{ 
  if( argc < 2 ) { 
    /* Not enough arguments.*/ 
    sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); 
    return; 
  } 
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL || 
      sqlite3_value_type(argv[1]) == SQLITE_NULL ) { 
    /* Missing one of the required arguments. */ 
    sqlite3_result_error(context, "One of the given arguments is null.", -1); 
    return; 
  } 
  int arg1_size_bytes = sqlite3_value_bytes(argv[0]); 
  int arg2_size_bytes = sqlite3_value_bytes(argv[1]); 
  if( arg1_size_bytes != arg2_size_bytes ) { 
    sqlite3_result_error(context, "The arrays are not the same length.", -1); 
    return; 
  } 
  int vector_size = -1; 
  if( argc > 2 ) { 
    if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { 
      vector_size = sqlite3_value_int(argv[2]); 
    } 
  } 
  if( vector_size < 1 ) { 
    vector_size = arg1_size_bytes / sizeof(float); 
  } 
  const float* searched_array = (const float *)sqlite3_value_blob(argv[0]); 
  const float* column_array = (const float *)sqlite3_value_blob(argv[1]); 
  float similarity = euclidean_func_f( searched_array, column_array, vector_size ); 
  similarity = sqrtf(similarity); 
  sqlite3_result_double(context, (double)similarity); 
} 




//----------------------------------------------------------------------------------------
// Name: ndvss_euclidean_distance_similarity_d
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of doubles.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
static void ndvss_euclidean_distance_similarity_d(  sqlite3_context* context, 
                                                    int argc, 
                                                    sqlite3_value** argv ) 
{ 
    if( argc < 2 ) { 
        /* Not enough arguments.*/ 
        sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); \
        return; 
    } 
    if( sqlite3_value_type(argv[0]) == SQLITE_NULL || 
        sqlite3_value_type(argv[1]) == SQLITE_NULL ) { 
        /* Missing one of the required arguments. */ 
        sqlite3_result_error(context, "One of the given arguments is null.", -1); 
        return; 
    } 
    int arg1_size_bytes = sqlite3_value_bytes(argv[0]); 
    int arg2_size_bytes = sqlite3_value_bytes(argv[1]); 
    if( arg1_size_bytes != arg2_size_bytes ) { 
        sqlite3_result_error(context, "The arrays are not the same length.", -1); 
        return; 
    } 
    int vector_size = -1; 
    if( argc > 2 ) { 
        if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { 
        vector_size = sqlite3_value_int(argv[2]); 
        } 
    } 
    if( vector_size < 1 ) { 
        vector_size = arg1_size_bytes / sizeof(double); 
    } 
    const double* searched_array = (const double *)sqlite3_value_blob(argv[0]); 
    const double* column_array = (const double *)sqlite3_value_blob(argv[1]); 
    double similarity = euclidean_func_d( searched_array, column_array, vector_size ); 
    similarity = sqrt(similarity); 
    sqlite3_result_double(context, similarity); 
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
static void ndvss_euclidean_distance_similarity_squared_f(  sqlite3_context* context, 
                                                            int argc, 
                                                            sqlite3_value** argv ) 
{ 
    if( argc < 2 ) { 
        /* Not enough arguments.*/ 
        sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); 
        return; 
    } 
    if( sqlite3_value_type(argv[0]) == SQLITE_NULL || 
        sqlite3_value_type(argv[1]) == SQLITE_NULL ) { 
        /* Missing one of the required arguments. */ 
        sqlite3_result_error(context, "One of the given arguments is null.", -1); 
        return; 
    } 
    int arg1_size_bytes = sqlite3_value_bytes(argv[0]); 
    int arg2_size_bytes = sqlite3_value_bytes(argv[1]); 
    if( arg1_size_bytes != arg2_size_bytes ) { 
        sqlite3_result_error(context, "The arrays are not the same length.", -1); 
        return; 
    } 
    int vector_size = -1; 
    if( argc > 2 ) { 
        if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { 
        vector_size = sqlite3_value_int(argv[2]); 
        } 
    } 
    if( vector_size < 1 ) { 
        vector_size = arg1_size_bytes / sizeof(float); 
    } 
    const float* searched_array = (const float *)sqlite3_value_blob(argv[0]); 
    const float* column_array = (const float *)sqlite3_value_blob(argv[1]); 
    float similarity = euclidean_func_f( searched_array, column_array, vector_size ); 
    sqlite3_result_double(context, (double)similarity); 
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
static void ndvss_euclidean_distance_similarity_squared_d(  sqlite3_context* context, 
                                                            int argc, 
                                                            sqlite3_value** argv ) 
{ 
    if( argc < 2 ) { 
        /* Not enough arguments.*/ 
        sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, optionally the array length.", -1); \
        return; 
    } 
    if( sqlite3_value_type(argv[0]) == SQLITE_NULL || 
        sqlite3_value_type(argv[1]) == SQLITE_NULL ) { 
        /* Missing one of the required arguments. */ 
        sqlite3_result_error(context, "One of the given arguments is null.", -1); 
        return; 
    } 
    int arg1_size_bytes = sqlite3_value_bytes(argv[0]); 
    int arg2_size_bytes = sqlite3_value_bytes(argv[1]); 
    if( arg1_size_bytes != arg2_size_bytes ) { 
        sqlite3_result_error(context, "The arrays are not the same length.", -1); 
        return; 
    } 
    int vector_size = -1; 
    if( argc > 2 ) { 
        if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { 
        vector_size = sqlite3_value_int(argv[2]); 
        } 
    } 
    if( vector_size < 1 ) { 
        vector_size = arg1_size_bytes / sizeof(double); 
    } 
    const double* searched_array = (const double *)sqlite3_value_blob(argv[0]); 
    const double* column_array = (const double *)sqlite3_value_blob(argv[1]); 
    double similarity = euclidean_func_d( searched_array, column_array, vector_size ); 
    sqlite3_result_double(context, similarity); 
} 


//----------------------------------------------------------------------------------------
// Name: ndvss_dot_product_similarity_f
// Desc: Calculates the dot product similarity to a BLOB-converted array of floats.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
static void ndvss_dot_product_similarity_f( sqlite3_context* context, 
                                            int argc, 
                                            sqlite3_value** argv ) 
{ 
    if( argc < 2 ) { 
        /* Not enough arguments. */ 
        sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, array length.", -1); 
        return; 
    } 
    if( sqlite3_value_type(argv[0]) == SQLITE_NULL || 
        sqlite3_value_type(argv[1]) == SQLITE_NULL ) { 
        /* Missing one of the required arguments. */ 
        sqlite3_result_error(context, "One of the given arguments is NULL.", -1); 
        return; 
    } 
    int arg1_size_bytes = sqlite3_value_bytes(argv[0]); 
    int arg2_size_bytes = sqlite3_value_bytes(argv[1]); 
    if( arg1_size_bytes != arg2_size_bytes ) { 
        sqlite3_result_error(context, "The arrays are not the same length.", -1); 
        return; 
    } 
    int vector_size = -1; 
    if( argc > 2 ) { 
        if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { 
        vector_size = sqlite3_value_int(argv[2]); 
        } 
    }  
    if( vector_size < 1 ) { 
        vector_size = arg1_size_bytes / sizeof(float);
    } 
    const float* searched_array = (const float *)sqlite3_value_blob(argv[0]); 
    const float* column_array = (const float *)sqlite3_value_blob(argv[1]); 
    float similarity = dot_product_func_f( searched_array, column_array, vector_size ); 
    sqlite3_result_double(context, (double)similarity); 
}



//----------------------------------------------------------------------------------------
// Name: ndvss_dot_product_similarity_d
// Desc: Calculates the dot product similarity to a BLOB-converted array of doubles.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
static void ndvss_dot_product_similarity_d( sqlite3_context* context, 
                                            int argc, 
                                            sqlite3_value** argv ) 
{ 
    if( argc < 2 ) { 
        /* Not enough arguments. */ 
        sqlite3_result_error(context, "2 arguments needs to be given: searched array, column/compared array, array length.", -1); \
        return; 
    } 
    if( sqlite3_value_type(argv[0]) == SQLITE_NULL || 
        sqlite3_value_type(argv[1]) == SQLITE_NULL ) { 
        /* Missing one of the required arguments. */ 
        sqlite3_result_error(context, "One of the given arguments is NULL.", -1); 
        return; 
    } 
    int arg1_size_bytes = sqlite3_value_bytes(argv[0]); 
    int arg2_size_bytes = sqlite3_value_bytes(argv[1]); 
    if( arg1_size_bytes != arg2_size_bytes ) { 
        sqlite3_result_error(context, "The arrays are not the same length.", -1); 
        return; 
    } 
    int vector_size = -1; 
    if( argc > 2 ) { 
        if( sqlite3_value_type(argv[2]) != SQLITE_NULL ) { 
        vector_size = sqlite3_value_int(argv[2]); 
        } 
    }  
    if( vector_size < 1 ) { 
        vector_size = arg1_size_bytes / sizeof(double);
    } 
    const double* searched_array = (const double *)sqlite3_value_blob(argv[0]); 
    const double* column_array = (const double *)sqlite3_value_blob(argv[1]); 
    double similarity = dot_product_func_d( searched_array, column_array, vector_size ); 
    sqlite3_result_double(context, similarity); 
}





//========================================================================================
// Helper utils.
//========================================================================================

//----------------------------------------------------------------------------------------
// Name: ndvss_version
// Desc: Returns the current version of ndvss.
// Args: None.
// Returns: Version number as a DOUBLE.
//----------------------------------------------------------------------------------------
static void ndvss_version( sqlite3_context* context,
                           int argc,
                           sqlite3_value** argv ) 
{
    sqlite3_result_double(context, NDVSS_VERSION_DOUBLE );
}



//----------------------------------------------------------------------------------------
// Name: ndvss_instruction_set
// Desc: Returns the current instruction set used by ndvss.
// Args: None.
// Returns: The instruction set as a STRING.
//----------------------------------------------------------------------------------------
static void ndvss_instruction_set(  sqlite3_context* context,
                                    int argc,
                                    sqlite3_value** argv ) 
{
    sqlite3_result_text(context, g_instruction_set, -1, SQLITE_STATIC );
}

//----------------------------------------------------------------------------------------
// Name: ndvss_convert_str_to_array_d
// Desc: Converts a list of decimal numbers from a string to an array of doubles.
// Args: List of decimal numbers TEXT, 
//       Number of dimensions INTEGER
// Returns: The double-array as a BLOB.
//----------------------------------------------------------------------------------------
static void ndvss_convert_str_to_array_d( sqlite3_context* context,
                                          int argc,
                                          sqlite3_value** argv ) 
{
  if( argc < 2 ) {
    sqlite3_result_error(context, "2 arguments needs to be given: string to convert, array length.", -1);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ) {
    sqlite3_result_error(context, "One of the given arguments is null.", -1);
    return;
  }

  int num_dimensions = sqlite3_value_int(argv[1]);
  if( num_dimensions <= 0 ) {
    sqlite3_result_error(context, "Number of dimensions is 0.", -1);
    return;
  }
  int allocated_size = sizeof(double)*num_dimensions;
  double* output = (double*)sqlite3_malloc(allocated_size);
  if( output == 0 ) {
    sqlite3_result_error(context, "Out of memory.", -1);
    return;
  }
  char* input = (char*)sqlite3_value_text(argv[0]);
  char* end = input;
  double* index = output;
  int i = 0;
  while( end != 0 && i < num_dimensions ) {
    // Skip the JSON-array characters.
    if (*end == '[' || *end == ']' || *end == ',') {
      end++;
      continue;
    } 
    *index = strtod(end, &end);
    ++index;
    ++i;  
  }//endwhile processing string
  sqlite3_result_blob(context, output, allocated_size, sqlite3_free );
}


//----------------------------------------------------------------------------------------
// Name: ndvss_convert_str_to_array_f
// Desc: Converts a list of decimal numbers from a string to an array of floats.
// Args: List of decimal numbers TEXT, 
//       Number of dimensions INTEGER
// Returns: The float-array as a BLOB.
//----------------------------------------------------------------------------------------
static void ndvss_convert_str_to_array_f( sqlite3_context* context,
                                          int argc,
                                          sqlite3_value** argv ) 
{
  if( argc < 2 ) {
    sqlite3_result_error(context, "2 arguments needs to be given: string to convert, array length.", -1);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ) {
    sqlite3_result_error(context, "One of the given arguments is null.", -1);
    return;
  }

  int num_dimensions = sqlite3_value_int(argv[1]);
  if( num_dimensions <= 0 ) {
    sqlite3_result_error(context, "Number of dimensions is 0.", -1);
    return;
  }
  int allocated_size = sizeof(float)*num_dimensions;
  float* output = (float*)sqlite3_malloc(allocated_size);
  if( output == 0 ) {
    sqlite3_result_error(context, "Out of memory.", -1);
    return;
  }
  char* input = (char*)sqlite3_value_text(argv[0]);
  char* end = input;
  float* index = output;
  int i = 0;
  while( end != 0 && i < num_dimensions ) {
    // Skip the JSON-array characters.
    if (*end == '[' || *end == ']' || *end == ',') {
      end++;
      continue;
    } 
    *index = (float)strtod(end, &end);
    ++index;
    ++i;  
  }//endwhile processing string
  sqlite3_result_blob(context, output, allocated_size, sqlite3_free );
}



//----------------------------------------------------------------------------------------
// Name: ndvss_dot_product_similarity_str
// Desc: Calculates the dot product similarity between two strings containing arrays of
//       decimal numbers. The first argument (searched array) is cached as a BLOB.
//       Note! This is an inefficient implementation mainly meant for testing.
// Args: Searched array TEXT,
//       Compared array (usually a column) TEXT, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
static void ndvss_dot_product_similarity_str( sqlite3_context* context,
                                                    int argc,
                                                    sqlite3_value** argv ) 
{
  if( argc < 3 ) {
    // Not enough arguments.
    sqlite3_result_error(context, "3 arguments needs to be given: searched array, column/compared array, array length.", -1);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ||
      sqlite3_value_type(argv[2]) == SQLITE_NULL ) {
    // Missing one of the required argument.
    sqlite3_result_error(context, "One of the given arguments is NULL.", -1);
    return;
  }

  int vector_size = sqlite3_value_int(argv[2]);
  // Parse through the searched value and save it to aux-data for future
  // use.
  void* aux_array = sqlite3_get_auxdata( context, 0 );
  double* comparison_vector = 0; 
  if( aux_array == 0 ) {
    comparison_vector = (double*)sqlite3_malloc(vector_size*sizeof(double));
    if(comparison_vector == 0 ) {
      sqlite3_result_error(context, "Out of memory.", -1);
      return;
    }
    char* search_vector = (char*)sqlite3_value_text(argv[0]);
    char* end = search_vector;
    double* index = comparison_vector;
    int i = 0;
    while( end != 0 && i < vector_size ) {
      if (*end == '[' || *end == ']' || *end == ',') {
            // Increment endptr to skip the character
            end++;
            continue;
      } 
      *index = strtod(end, &end);
      ++index;
      ++i;
    }//endwhile processing searched string
    sqlite3_set_auxdata(context, 0, comparison_vector, sqlite3_free);
  } else {
    comparison_vector = (double *)aux_array;
  }

  // Start parsing through the second argument and calculate the
  // similarity.
  double similarity = 0.0;
  char* rowvalue_input = (char*)sqlite3_value_text(argv[1]);
  char* end = rowvalue_input;
  double* index = comparison_vector;
  int i = 0;
  while( end != 0 && i < vector_size ) {
    // Skip the JSON-array characters.
    if (*end == '[' || *end == ']' || *end == ',') {
      end++;
      continue;
    } 
    double d = strtod(end, &end);
    similarity += (*index * d);
    ++index;
    ++i;  
  }//endwhile processing searched string
  sqlite3_result_double(context, similarity);
}


//-----------------------------------------------------------------------------------
// ENTRYPOINT.
//-----------------------------------------------------------------------------------
#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_ndvss_init( sqlite3 *db, 
                        char **pzErrMsg, 
                        const sqlite3_api_routines *pApi )
{
    int rc = SQLITE_OK;
    SQLITE_EXTENSION_INIT2(pApi);

    // Detect the CPU and initialize functions based on the available 
    // instruction sets. 
#if defined(__riscv) && defined(__riscv_vector)
    LOAD_SIMILARITY_FUNCTIONS(rvv)
#elif defined(__aarch64__)
    LOAD_SIMILARITY_FUNCTIONS(neon)
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
    
    // For x86_64 do a runtime check for cpu capabilities. 
    int has_sse41   = 0
       ,has_avx     = 0
       ,has_avx2    = 0
       ,has_avx512f = 0;

    unsigned int eax, ebx, ecx, edx;

    // 1. Check Standard Features
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (ecx & (1 << 19)) has_sse41 = 1;
        if (ecx & (1 << 28)) has_avx = 1;
        
        if ((ecx & (1 << 27)) == 0) {
            has_avx = 0;
        }
    }
    // 2. Check Extended Features
    if (has_avx && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        if (ebx & (1 << 5))  has_avx2 = 1;
        if (ebx & (1 << 16)) has_avx512f = 1;
    }

    /**/
    if (has_avx512f) {
        LOAD_SIMILARITY_FUNCTIONS(avx512f)
    } else if (has_avx2) {
        LOAD_SIMILARITY_FUNCTIONS(avx2)
    } else if (has_avx) {
        LOAD_SIMILARITY_FUNCTIONS(avx)
    } else if (has_sse41) {
        LOAD_SIMILARITY_FUNCTIONS(sse41)
    }
    /**/

#endif

    (void)pzErrMsg;  /* Unused parameter */

    /* Register the functions. */
    rc = sqlite3_create_function( db, 
                                "ndvss_cosine_similarity_f", // Function name 
                                -1, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_cosine_similarity_f, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }

    rc = sqlite3_create_function( db, 
                                "ndvss_euclidean_distance_similarity_f", // Function name 
                                -1, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_euclidean_distance_similarity_f, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }

    rc = sqlite3_create_function( db, 
                                "ndvss_euclidean_distance_similarity_squared_f", // Function name 
                                -1, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_euclidean_distance_similarity_squared_f, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }

    rc = sqlite3_create_function( db, 
                                "ndvss_dot_product_similarity_f", // Function name 
                                -1, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_dot_product_similarity_f, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }
    
    rc = sqlite3_create_function( db, 
                                "ndvss_cosine_similarity_d", // Function name 
                                -1, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_cosine_similarity_d, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }

    rc = sqlite3_create_function( db, 
                                "ndvss_euclidean_distance_similarity_d", // Function name 
                                -1, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_euclidean_distance_similarity_d, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }

    rc = sqlite3_create_function( db, 
                                "ndvss_euclidean_distance_similarity_squared_d", // Function name 
                                -1, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_euclidean_distance_similarity_squared_d, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }

    rc = sqlite3_create_function( db, 
                                "ndvss_dot_product_similarity_d", // Function name 
                                -1, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_dot_product_similarity_d, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }


   

    // General functions.
    rc = sqlite3_create_function( db, 
                                "ndvss_version", // Function name 
                                0, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_version, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }

    rc = sqlite3_create_function( db, 
                                "ndvss_instruction_set", // Function name 
                                0, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_instruction_set, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }
  
    rc = sqlite3_create_function( db, 
                                "ndvss_convert_str_to_array_d", // Function name 
                                2, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_convert_str_to_array_d, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }


    rc = sqlite3_create_function( db, 
                                    "ndvss_convert_str_to_array_f", // Function name 
                                    2, // Number of arguments
                                    SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                    0, // *pApp?
                                    ndvss_convert_str_to_array_f, // xFunc -> Function pointer 
                                    0, // xStep?
                                    0  // xFinal?
                                    );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }

   

    rc = sqlite3_create_function( db, 
                                    "ndvss_dot_product_similarity_str", // Function name 
                                    3, // Number of arguments
                                    SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                    0, // *pApp?
                                    ndvss_dot_product_similarity_str, // xFunc -> Function pointer 
                                    0, // xStep?
                                    0  // xFinal?
                                    );
    if (rc != SQLITE_OK) {
        *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
        return rc;
    }

  return rc;
}

