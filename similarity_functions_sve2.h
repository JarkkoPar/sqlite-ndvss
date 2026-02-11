#ifndef SIMILARITY_FUNCTIONS_SVE2_H_INCLUDED
#define SIMILARITY_FUNCTIONS_SVE2_H_INCLUDED
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
/**
 * This file contains the Arm SVE2 versions of the similarity function definitions. 
 */
#include <arm_sve.h>


//----------------------------------------------------------------------------------------
// Name: cosine_similarity_f_sve2
// Desc: Calculates the cosine similarity using two given float arrays. ARM SVE2 version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a FLOAT 
//       Pointer to divider_b FLOAT
// Returns: Similarity as an angle float
//----------------------------------------------------------------------------------------
float cosine_similarity_f_sve2( 
     const float*   searched_array 
    ,const float*   column_array 
    ,const int      vector_size
    ,float*         divider_a 
    ,float*         divider_b )
{
    float dividerA   = 0.0f
         ,dividerB   = 0.0f
         ,similarity = 0.0f;
    svfloat32_t A
               ,B
               ,mmdividerA   = svdup_f32(0.0f)
               ,mmdividerB   = svdup_f32(0.0f)
               ,mmsimilarity = svdup_f32(0.0f);

    int i = 0;

    // Prepare the predicate that is used to handle each iteration.
    svbool_t while_lessthan_bool = svwhilelt_b32( i, vector_size );
    while( svptest_any( svptrue_b32(), while_lessthan_bool )) 
    {
        A = svld1_f32( while_lessthan_bool, &searched_array[i] );
        B = svld1_f32( while_lessthan_bool, &column_array[i]   );
        
        // Do a fused multiply-add. 
        mmdividerA = svmla_f32_m( while_lessthan_bool, mmdividerA, A, A );
        mmdividerB = svmla_f32_m( while_lessthan_bool, mmdividerB, B, B );
        mmsimilarity = svmla_f32_m( while_lessthan_bool, mmsimilarity, A, B );

        // Move the index according to number elements processed.
        i += svcntw();

        // Update the predicate for the next iteration.
        while_lessthan_bool = svwhilelt_b32( i, vector_size ); 
    }//endwhile 
    
    // Do a horizontal reduction to get the result.
    svbool_t all_true = svptrue_b32();
    similarity = svaddv_f32(all_true, mmsimilarity );
    *divider_a = svaddv_f32(all_true, mmdividerA   );
    *divider_b = svaddv_f32(all_true, mmdividerB   );

    // Return the result.
    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: cosine_similarity_d_sve2
// Desc: Calculates the cosine similarity using two given double arrays. ARM SVE2 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a DOUBLE 
//       Pointer to divider_b DOUBLE
// Returns: Similarity as an angle DOUBLE
//----------------------------------------------------------------------------------------
double cosine_similarity_d_sve2( 
     const double*   searched_array 
    ,const double*   column_array 
    ,const int       vector_size
    ,double*         divider_a 
    ,double*         divider_b )
{
    double similarity = 0.0
          ,dividerA   = 0.0
          ,dividerB   = 0.0;
    svfloat64_t A
               ,B
               ,mmdividerA   = svdup_f64(0.0f)
               ,mmdividerB   = svdup_f64(0.0f)
               ,mmsimilarity = svdup_f64(0.0f);

    int i = 0;

    // Prepare the predicate that is used to handle each iteration.
    svbool_t while_lessthan_bool = svwhilelt_b64( i, vector_size );
    while( svptest_any( svptrue_b64(), while_lessthan_bool )) 
    {
        A = svld1_f64( while_lessthan_bool, &searched_array[i] );
        B = svld1_f64( while_lessthan_bool, &column_array[i]   );
        
        // Do a fused multiply-add. 
        mmdividerA = svmla_f64_m( while_lessthan_bool, mmdividerA, A, A );
        mmdividerB = svmla_f64_m( while_lessthan_bool, mmdividerB, B, B );
        mmsimilarity = svmla_f64_m( while_lessthan_bool, mmsimilarity, A, B );

        // Move the index according to number elements processed.
        i += svcntd();

        // Update the predicate for the next iteration.
        while_lessthan_bool = svwhilelt_b64( i, vector_size ); 
    }//endwhile 
    
    // Do a horizontal reduction to get the result.
    svbool_t all_true = svptrue_b64();
    similarity = svaddv_f64(all_true, mmsimilarity );
    *divider_a = svaddv_f64(all_true, mmdividerA   );
    *divider_b = svaddv_f64(all_true, mmdividerB   );

    // Return the result.
    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_f_sve2
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of floats.
//       ARM SVE2 version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
float euclidean_distance_similarity_f_sve2( const float* searched_array
                                          ,const float* column_array
                                          ,const int    vector_size ) 
{
    float similarity = 0.0f;
    svfloat32_t A, B, AB, sumAB = svdup_f32(0.0f);
    int i = 0;
    // Prepare the predicate that is used to handle each iteration.
    svbool_t while_lessthan_bool = svwhilelt_b32( i, vector_size );
    while( svptest_any( svptrue_b32(), while_lessthan_bool )) 
    {
        A = svld1_f32( while_lessthan_bool, &searched_array[i] );
        B = svld1_f32( while_lessthan_bool, &column_array[i]   );
        
        // Substract and accumulate. 
        AB = svsub_f32_x( while_lessthan_bool, A, B );
        sumAB = svmla_f32_m( while_lessthan_bool, sumAB, AB, AB );

        // Move the index according to number elements processed.
        i += svcntw();

        // Update the predicate for the next iteration.
        while_lessthan_bool = svwhilelt_b32( i, vector_size ); 
    }//endwhile 
    
    // Return the result as a float.
    similarity = svaddv_f32( svptrue_b32(), sumAB );
    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_d_sve2
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of doubles.
//       ARM SVE2 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
double euclidean_distance_similarity_d_sve2( const double* searched_array
                                           ,const double* column_array
                                           ,const int    vector_size ) 
{
    double similarity = 0.0;
    svfloat64_t A, B, AB, sumAB = svdup_f64(0.0f);
    int i = 0;
    // Prepare the predicate that is used to handle each iteration.
    svbool_t while_lessthan_bool = svwhilelt_b64( i, vector_size );
    while( svptest_any( svptrue_b64(), while_lessthan_bool )) 
    {
        A = svld1_f64( while_lessthan_bool, &searched_array[i] );
        B = svld1_f64( while_lessthan_bool, &column_array[i]   );
        
        // Substract and accumulate. 
        AB = svsub_f64_x( while_lessthan_bool, A, B );
        sumAB = svmla_f64_m( while_lessthan_bool, sumAB, AB, AB );

        // Move the index according to number elements processed.
        i += svcntw();

        // Update the predicate for the next iteration.
        while_lessthan_bool = svwhilelt_b64( i, vector_size ); 
    }//endwhile 
    
    // Return the result as a float.
    similarity = svaddv_f64( svptrue_b64(), sumAB );
    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_f_sve2
// Desc: Calculates the dot product similarity to a BLOB-converted array of floats.
//       ARM SVE2 version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product FLOAT
//----------------------------------------------------------------------------------------
float dot_product_similarity_f_sve2( const float* searched_array 
                                    ,const float* column_array 
                                    ,const int    vector_size ) 
{
    float similarity = 0.0f;
    
    svfloat32_t A, B, sumAB = svdup_f32(0.0f);
    int i = 0;
    // Prepare the predicate that is used to handle each iteration.
    svbool_t while_lessthan_bool = svwhilelt_b32( i, vector_size );
    while( svptest_any( svptrue_b32(), while_lessthan_bool )) 
    {
        A = svld1_f32( while_lessthan_bool, &searched_array[i] );
        B = svld1_f32( while_lessthan_bool, &column_array[i]   );
        
        // Do a fused multiply-add. 
        sumAB = svmla_f32_m( while_lessthan_bool, sumAB, A, B );

        // Move the index according to number elements processed.
        i += svcntw();

        // Update the predicate for the next iteration.
        while_lessthan_bool = svwhilelt_b32( i, vector_size ); 
    }//endwhile 
    
    // Return the result as a float.
    similarity = svaddv_f32( svptrue_b32(), sumAB );
    return similarity;
}




//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_d_sve2
// Desc: Calculates the dot product similarity to a BLOB-converted array of doubles.
//       ARM SVE2 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
double dot_product_similarity_d_sve2( const double* searched_array 
                                     ,const double* column_array 
                                     ,const int     vector_size ) 
{
    double similarity = 0.0f;
    svfloat64_t A, B, sumAB = svdup_f64(0.0f);
    int i = 0;
    // Prepare the predicate that is used to handle each iteration.
    svbool_t while_lessthan_bool = svwhilelt_b64( i, vector_size );
    while( svptest_any( svptrue_b64(), while_lessthan_bool )) 
    {
        A = svld1_f64( while_lessthan_bool, &searched_array[i] );
        B = svld1_f64( while_lessthan_bool, &column_array[i]   );
        
        // Do a fused multiply-add. 
        sumAB = svmla_f64_m( while_lessthan_bool, sumAB, A, B );

        // Move the index according to number elements processed.
        i += svcntd();

        // Update the predicate for the next iteration.
        while_lessthan_bool = svwhilelt_b64( i, vector_size ); 
    }//endwhile 
    
    // Return the result as a float.
    similarity = svaddv_f64( svptrue_b64(), sumAB );
    return similarity;
}


#endif // if aarch64
#endif 