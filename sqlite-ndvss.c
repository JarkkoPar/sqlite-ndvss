/*
** 2024-04-08
**
** This SQLite extension implements functions to perform vector
** similarity searches. The extension's goal is to create a 
** dependency-free solution that can easily be used in different
** platforms.
*/
#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1
//#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NDVSS_VERSION_STRING  "0.1"


// Returns the current version of ndvss.
static void ndvss_version( sqlite3_context* context,
                           int argc,
                           sqlite3_value** argv ) 
{
  sqlite3_result_text(context, NDVSS_VERSION_STRING, -1, SQLITE_STATIC );
}


//----------------------------------------------------------------------------------------
// Converts a list of decimal numbers from a string to an array of doubles.
// Arguments: The list of doubles TEXT, Number of dimensions INTEGER. 
// Return value: The double-array as a BLOB.
//----------------------------------------------------------------------------------------
static void ndvss_convert_str_to_array_d( sqlite3_context* context,
                                          int argc,
                                          sqlite3_value** argv ) 
{
  if( argc < 2 ) {
    // Not enough arguments.
    sqlite3_result_null(context);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ) {
    // Text argument is NULL.
    sqlite3_result_null(context);
    return;
  }
  if( sqlite3_value_type(argv[1]) == SQLITE_NULL ) {
    // Dimension count argument is NULL.
    sqlite3_result_null(context);
    return;
  }

  int num_dimensions = sqlite3_value_int(argv[1]);
  if( num_dimensions <= 0 ) {
    // Seriously? Passing a 0 or negative-length arrays..?
    sqlite3_result_null(context);
    return;
  }
  int allocated_size = sizeof(double)*num_dimensions;
  double* output = (double*)sqlite3_malloc(allocated_size);
  if( output == 0 ) {
    // Out of memory.
    sqlite3_result_null(context);
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
// Converts a list of decimal numbers from a string to an array of floats.
// This will sacrifice precision for performance.
// Arguments: The list of doubles TEXT, Number of dimensions INTEGER. 
// Return value: The float-array as a BLOB.
//----------------------------------------------------------------------------------------
static void ndvss_convert_str_to_array_f(sqlite3_context* context,
                                        int argc,
                                        sqlite3_value** argv ) 
{
  if( argc < 2 ) {
    // Not enough arguments.
    sqlite3_result_null(context);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ) {
    // Text argument is NULL.
    sqlite3_result_null(context);
    return;
  }
  if( sqlite3_value_type(argv[1]) == SQLITE_NULL ) {
    // Dimension count argument is NULL.
    sqlite3_result_null(context);
    return;
  }

  int num_dimensions = sqlite3_value_int(argv[1]);
  if( num_dimensions <= 0 ) {
    // Seriously? Passing a 0 or negative-length arrays..?
    sqlite3_result_null(context);
    return;
  }
  int allocated_size = sizeof(float)*num_dimensions;
  float* output = (float*)sqlite3_malloc(allocated_size);
  if( output == 0 ) {
    // Out of memory.
    sqlite3_result_null(context);
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
// Calculates the cosine similarity of a vector compared to the given other 
// vector (assumed to be a column in the database).
// Assumes the column is of BLOB-type and that it contains an array of doubles.
// Third parameter is the vector size.
//----------------------------------------------------------------------------------------
static void ndvss_cosine_similarity_d( sqlite3_context* context,
                                       int argc,
                                       sqlite3_value** argv ) 
{
  if( argc < 3 ) {
    // Not enough arguments.
    sqlite3_result_double(context, -1.0);
    sqlite3_result_error(context, "3 arguments needs to be given: searched array, column/compared array, array length.", -1);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ||
      sqlite3_value_type(argv[2]) == SQLITE_NULL ) {
    // Missing one of the required arguments.
    sqlite3_result_error(context, "One of the given arguments is null.", -1);
    return;
  }
  if( sqlite3_value_bytes(argv[0]) != sqlite3_value_bytes(argv[1])) {
    // Mismatching array lengths.
    sqlite3_result_error(context, "The arrays are not the same length.", -1);
    return;
  }

  int vector_size = sqlite3_value_int(argv[2]);
  const double* searched_array = (const double *)sqlite3_value_blob(argv[0]);
  const double* column_array = (const double *)sqlite3_value_blob(argv[1]);
  double similarity = 0.0;
  double dividerA = 0.0;
  double dividerB = 0.0;
  for( int i = 0; i < vector_size; ++i ) {
    double A = searched_array[i];
    double B = column_array[i];
    similarity += (A*B);
    dividerA += (A*A);
    dividerB += (B*B);
  }
  if( dividerA == 0.0 || dividerB == 0.0 ) {
    // There'd be a division by zero, so assume no similarity.
    sqlite3_result_double(context, -1.0);  
    return;
  }
  double divider = sqrt(dividerA * dividerB);
  similarity = similarity / divider;
  sqlite3_result_double(context, similarity);
}


//----------------------------------------------------------------------------------------
// Calculates the cosine similarity of a vector compared to the given other 
// vector (assumed to be a column in the database).
// Assumes the column is of BLOB-type and that it contains an array of floats.
// Third parameter is the vector size.
//----------------------------------------------------------------------------------------
static void ndvss_cosine_similarity_f( sqlite3_context* context,
                                       int argc,
                                       sqlite3_value** argv ) 
{
  if( argc < 3 ) {
    // Not enough arguments.
    sqlite3_result_double(context, -1.0);
    sqlite3_result_error(context, "3 arguments needs to be given: searched array, column/compared array, array length.", -1);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ||
      sqlite3_value_type(argv[2]) == SQLITE_NULL ) {
    // Missing one of the required arguments.
    sqlite3_result_error(context, "One of the given arguments is null.", -1);
    return;
  }
  if( sqlite3_value_bytes(argv[0]) != sqlite3_value_bytes(argv[1])) {
    // Mismatching array lengths.
    sqlite3_result_error(context, "The arrays are not the same length.", -1);
    return;
  }

  int vector_size = sqlite3_value_int(argv[2]);
  const float* searched_array = (const float *)sqlite3_value_blob(argv[0]);
  const float* column_array = (const float *)sqlite3_value_blob(argv[1]);
  float similarity = 0.0f;
  float dividerA = 0.0f;
  float dividerB = 0.0f;
  for( int i = 0; i < vector_size; ++i ) {
    float A = searched_array[i];
    float B = column_array[i];
    similarity += (A*B);
    dividerA += (A*A);
    dividerB += (B*B);
  }
  if( dividerA == 0.0f || dividerB == 0.0f ) {
    // There'd be a division by zero, so assume no similarity.
    sqlite3_result_double(context, -1.0);  
    return;
  }
  float divider = sqrtf(dividerA * dividerB);
  similarity = similarity / divider;
  sqlite3_result_double(context, (double)similarity);
}

//----------------------------------------------------------------------------------------
// Calculates the euclidean distance similarity of a vector compared to the given other 
// vector (assumed to be a column in the database).
// Assumes the column is of BLOB-type and that it contains an array of doubles.
// Third parameter is the vector size.
//----------------------------------------------------------------------------------------
static void ndvss_euclidean_distance_similarity_d( sqlite3_context* context,
                                                   int argc,
                                                   sqlite3_value** argv ) 
{
  if( argc < 3 ) {
    // Not enough arguments.
    sqlite3_result_double(context, -1.0);
    sqlite3_result_error(context, "3 arguments needs to be given: searched array, column/compared array, array length.", -1);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ||
      sqlite3_value_type(argv[2]) == SQLITE_NULL ) {
    // Missing one of the required arguments.
    sqlite3_result_error(context, "One of the given arguments is null.", -1);
    return;
  }
  if( sqlite3_value_bytes(argv[0]) != sqlite3_value_bytes(argv[1])) {
    // Mismatching array lengths.
    sqlite3_result_error(context, "The arrays are not the same length.", -1);
    return;
  }

  int vector_size = sqlite3_value_int(argv[2]);
  const double* searched_array = (const double *)sqlite3_value_blob(argv[0]);
  const double* column_array = (const double *)sqlite3_value_blob(argv[1]);
  double similarity = 0.0;
  for( int i = 0; i < vector_size; ++i ) {
    double AB = (searched_array[i] - column_array[i]);
    similarity += (AB * AB);
  }
  similarity = sqrt(similarity);
  sqlite3_result_double(context, similarity);
}


//----------------------------------------------------------------------------------------
// Calculates the euclidean distance similarity of a vector compared to the given other 
// vector (assumed to be a column in the database).
// Assumes the column is of BLOB-type and that it contains an array of floats.
// Third parameter is the vector size.
//----------------------------------------------------------------------------------------
static void ndvss_euclidean_distance_similarity_f( sqlite3_context* context,
                                                   int argc,
                                                   sqlite3_value** argv ) 
{
  if( argc < 3 ) {
    // Not enough arguments.
    sqlite3_result_double(context, -1.0);
    sqlite3_result_error(context, "3 arguments needs to be given: searched array, column/compared array, array length.", -1);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ||
      sqlite3_value_type(argv[2]) == SQLITE_NULL ) {
    // Missing one of the required arguments.
    sqlite3_result_error(context, "One of the given arguments is null.", -1);
    return;
  }
  if( sqlite3_value_bytes(argv[0]) != sqlite3_value_bytes(argv[1])) {
    // Mismatching array lengths.
    sqlite3_result_error(context, "The arrays are not the same length.", -1);
    return;
  }

  int vector_size = sqlite3_value_int(argv[2]);
  const float* searched_array = (const float *)sqlite3_value_blob(argv[0]);
  const float* column_array = (const float *)sqlite3_value_blob(argv[1]);
  float similarity = 0.0f;
  for( int i = 0; i < vector_size; ++i ) {
    float AB = (searched_array[i] - column_array[i]);
    similarity += (AB * AB);
  }
  similarity = sqrtf(similarity);
  sqlite3_result_double(context, (double)similarity);
}


//----------------------------------------------------------------------------------------
// Calculates the euclidean distance similarity of a vector compared to the given other 
// vector (assumed to be a column in the database).
// Returns the squared results (no squareroot taken).
// Assumes the column is of BLOB-type and that it contains an array of doubles.
// Third parameter is the vector size.
//----------------------------------------------------------------------------------------
static void ndvss_euclidean_distance_similarity_squared_d( sqlite3_context* context,
                                                           int argc,
                                                           sqlite3_value** argv ) 
{
  if( argc < 3 ) {
    // Not enough arguments.
    sqlite3_result_double(context, -1.0);
    sqlite3_result_error(context, "3 arguments needs to be given: searched array, column/compared array, array length.", -1);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ||
      sqlite3_value_type(argv[2]) == SQLITE_NULL ) {
    // Missing one of the required arguments.
    sqlite3_result_error(context, "One of the given arguments is null.", -1);
    return;
  }
  if( sqlite3_value_bytes(argv[0]) != sqlite3_value_bytes(argv[1])) {
    // Mismatching array lengths.
    sqlite3_result_error(context, "The arrays are not the same length.", -1);
    return;
  }

  int vector_size = sqlite3_value_int(argv[2]);
  const double* searched_array = (const double *)sqlite3_value_blob(argv[0]);
  const double* column_array = (const double *)sqlite3_value_blob(argv[1]);
  double similarity = 0.0;
  for( int i = 0; i < vector_size; ++i ) {
    double AB = (searched_array[i] - column_array[i]);
    similarity += (AB * AB);
  }
  sqlite3_result_double(context, similarity);
}


//----------------------------------------------------------------------------------------
// Calculates the euclidean distance similarity of a vector compared to the given other 
// vector (assumed to be a column in the database). 
// Returns the squared results (no squareroot taken).
// Assumes the column is of BLOB-type and that it contains an array of floats.
// Third parameter is the vector size.
//----------------------------------------------------------------------------------------
static void ndvss_euclidean_distance_similarity_squared_f( sqlite3_context* context,
                                                           int argc,
                                                           sqlite3_value** argv ) 
{
  if( argc < 3 ) {
    // Not enough arguments.
    sqlite3_result_double(context, -1.0);
    sqlite3_result_error(context, "3 arguments needs to be given: searched array, column/compared array, array length.", -1);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ||
      sqlite3_value_type(argv[2]) == SQLITE_NULL ) {
    // Missing one of the required arguments.
    sqlite3_result_error(context, "One of the given arguments is null.", -1);
    return;
  }
  if( sqlite3_value_bytes(argv[0]) != sqlite3_value_bytes(argv[1])) {
    // Mismatching array lengths.
    sqlite3_result_error(context, "The arrays are not the same length.", -1);
    return;
  }

  int vector_size = sqlite3_value_int(argv[2]);
  const float* searched_array = (const float *)sqlite3_value_blob(argv[0]);
  const float* column_array = (const float *)sqlite3_value_blob(argv[1]);
  float similarity = 0.0f;
  for( int i = 0; i < vector_size; ++i ) {
    float AB = (searched_array[i] - column_array[i]);
    similarity += (AB * AB);
  }
  sqlite3_result_double(context, (float)similarity);
}


//----------------------------------------------------------------------------------------
// Calculates the dot product similarity of a normalized vector compared to the given other 
// normalized vector (assumed to be a column in the database).
// Assumes the column is of BLOB-type and contains an array of doubles.
// Third parameter is the vector size.
//----------------------------------------------------------------------------------------
static void ndvss_dot_product_similarity_d( sqlite3_context* context,
                                            int argc,
                                            sqlite3_value** argv ) 
{
  if( argc < 3 ) {
    // Not enough arguments.
    sqlite3_result_double(context, -1.0);
    sqlite3_result_error(context, "3 arguments needs to be given: searched array, column/compared array, array length.", -1);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ||
      sqlite3_value_type(argv[2]) == SQLITE_NULL ) {
    // Missing one of the required arguments.
    sqlite3_result_error(context, "One of the given arguments is null.", -1);
    return;
  }
  if( sqlite3_value_bytes(argv[0]) != sqlite3_value_bytes(argv[1])) {
    // Mismatching array lengths.
    sqlite3_result_error(context, "The arrays are not the same length.", -1);
    return;
  }

  int vector_size = sqlite3_value_int(argv[2]);
  const double* searched_array = (const double *)sqlite3_value_blob(argv[0]);
  const double* column_array = (const double *)sqlite3_value_blob(argv[1]);
  double similarity = 0.0;
  for( int i = 0; i < vector_size; ++i ) {
    similarity += ((searched_array[i]) * (column_array[i]));
    //++searched_array;
    //++column_array;
  }

  sqlite3_result_double(context, similarity);
}


//----------------------------------------------------------------------------------------
// Calculates the dot product similarity of a normalized vector compared to the given other 
// normalized vector (assumed to be a column in the database).
// Assumes the column is of BLOB-type and contains an array of float.
// Third parameter is the vector size.
//----------------------------------------------------------------------------------------
static void ndvss_dot_product_similarity_f( sqlite3_context* context,
                                            int argc,
                                            sqlite3_value** argv ) 
{
  if( argc < 3 ) {
    // Not enough arguments.
    sqlite3_result_double(context, -1.0);
    sqlite3_result_error(context, "3 arguments needs to be given: searched array, column/compared array, array length.", -1);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ||
      sqlite3_value_type(argv[2]) == SQLITE_NULL ) {
    // Missing one of the required arguments.
    sqlite3_result_error(context, "One of the given arguments is null.", -1);
    return;
  }
  if( sqlite3_value_bytes(argv[0]) != sqlite3_value_bytes(argv[1])) {
    // Mismatching array lengths.
    sqlite3_result_error(context, "The arrays are not the same length.", -1);
    return;
  }

  int vector_size = sqlite3_value_int(argv[2]);
  const float* searched_array = (const float *)sqlite3_value_blob(argv[0]);
  const float* column_array = (const float *)sqlite3_value_blob(argv[1]);
  float similarity = 0.0f;
  for( int i = 0; i < vector_size; ++i ) {
    similarity += ((searched_array[i]) * (column_array[i]));
    //++searched_array;
    //++column_array;
  }

  sqlite3_result_double(context, (double)similarity);
}


//----------------------------------------------------------------------------------------
// Calculates the cosine similarity of a normalized vector compared to the given other 
// normalized vector (assumed to be a column in the database).
// Third parameter is the vector size.
// The data is expected to be in a JSON array-format [0.0003403, 0.0343422, ... 0.0482384]
// or just a list of doubles.
//----------------------------------------------------------------------------------------
static void ndvss_cosine_similarity_normalized_str( sqlite3_context* context,
                                                    int argc,
                                                    sqlite3_value** argv ) 
{
  if( argc < 3 ) {
    // Not enough arguments.
    //sqlite3_mprintf("Error - Not enough arguments, 3 needed.\n");
    sqlite3_result_double(context, -1.0);
    return;
  }
  if( sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL ||
      sqlite3_value_type(argv[2]) == SQLITE_NULL ) {
    // Missing one of the required argument.
    sqlite3_mprintf("Error - One of the arguments is NULL.\n");
    sqlite3_result_double(context, -2.0);
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
      // Out of memory!
      //sqlite3_mprintf("Error - could not allocate memory for the double-array of size %i.\n", vector_size);
      sqlite3_result_double(context, -3.0);
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
  (void)pzErrMsg;  /* Unused parameter */
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
                                "ndvss_cosine_similarity_d", // Function name 
                                3, // Number of arguments
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
                                "ndvss_cosine_similarity_f", // Function name 
                                3, // Number of arguments
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
                                "ndvss_euclidean_distance_similarity_d", // Function name 
                                3, // Number of arguments
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
                                3, // Number of arguments
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
                                "ndvss_euclidean_distance_similarity_f", // Function name 
                                3, // Number of arguments
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
                                3, // Number of arguments
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
                                "ndvss_dot_product_similarity_d", // Function name 
                                3, // Number of arguments
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

  rc = sqlite3_create_function( db, 
                                "ndvss_dot_product_similarity_f", // Function name 
                                3, // Number of arguments
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
                                "ndvss_cosine_similarity_normalized_str", // Function name 
                                3, // Number of arguments
                                SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC,
                                0, // *pApp?
                                ndvss_cosine_similarity_normalized_str, // xFunc -> Function pointer 
                                0, // xStep?
                                0  // xFinal?
                                );
  if (rc != SQLITE_OK) {
      *pzErrMsg = sqlite3_mprintf("%s", sqlite3_errmsg(db));
      return rc;
  }
  return rc;
}

