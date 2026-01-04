#ifndef SIMILARITY_FUNCTIONS_NEON_H_INCLUDED
#define SIMILARITY_FUNCTIONS_NEON_H_INCLUDED
#if defined(__aarch64__)
/**
 * This file contains the ARM NEON versions of the similarity function definitions. 
 */
#include <arm_neon.h>


//----------------------------------------------------------------------------------------
// Name: cosine_similarity_f_neon
// Desc: Calculates the cosine similarity using two given float arrays. ARM Neon version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a FLOAT 
//       Pointer to divider_b FLOAT
// Returns: Similarity as an angle float
//----------------------------------------------------------------------------------------
float cosine_similarity_f_neon( 
     const float*   searched_array 
    ,const float*   column_array 
    ,const int      vector_size
    ,float*         divider_a 
    ,float*         divider_b )
{
    int   i = 0;
    float dividerA   = 0.0f
         ,dividerB   = 0.0f
         ,similarity = 0.0f;
    float32x4_t A
          ,B
          ,mmdividerA   = vdupq_n_f32(0.0f)
          ,mmdividerB   = vdupq_n_f32(0.0f)
          ,mmsimilarity = vdupq_n_f32(0.0f);
    
    for( ; i + 3 < vector_size; i += 4 ) {
        A = vld1q_f32(&searched_array[i]);
        B = vld1q_f32(&column_array[i]);
        mmdividerA = vfmaq_f32(mmdividerA, A, A);
        mmdividerB = vfmaq_f32(mmdividerB, B, B);
        mmsimilarity = vfmaq_f32(mmsimilarity, A, B);    
    }//endfor i+4

    // Divider A
    dividerA = vaddvq_f32(mmdividerA); 

    // Divider B
    dividerB = vaddvq_f32(mmdividerB); 

    // Similarity
    similarity = vaddvq_f32(mmsimilarity);

    // Calculate the remaining elements. 
    for(; i < vector_size; ++i ) {
        float A = searched_array[i];
        float B = column_array[i];
        similarity += (A*B);
        dividerA   += (A*A);
        dividerB   += (B*B);
    }

    *divider_a = dividerA;
    *divider_b = dividerB;
    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: cosine_similarity_d_neon
// Desc: Calculates the cosine similarity using two given double arrays. ARM Neon version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a DOUBLE 
//       Pointer to divider_b DOUBLE
// Returns: Similarity as an angle DOUBLE
//----------------------------------------------------------------------------------------
double cosine_similarity_d_neon( 
     const double*   searched_array 
    ,const double*   column_array 
    ,const int       vector_size
    ,double*         divider_a 
    ,double*         divider_b )
{
    int   i = 0;
    double dividerA   = 0.0
          ,dividerB   = 0.0
          ,similarity = 0.0;
    float64x2_t A
           ,B
           ,mmdividerA   = vdupq_n_f64(0.0)
           ,mmdividerB   = vdupq_n_f64(0.0)
           ,mmsimilarity = vdupq_n_f64(0.0);
    
    for( ; i + 1 < vector_size; i += 2 ) {
        A = vld1q_f64(&searched_array[i]);
        B = vld1q_f64(&column_array[i]);
        mmdividerA = vfmaq_f64(mmdividerA, A, A);
        mmdividerB = vfmaq_f64(mmdividerB, B, B);
        mmsimilarity = vfmaq_f64(mmsimilarity, A, B);    
    }//endfor i+2

    // Divider A
    dividerA = vaddvq_f64(mmdividerA);

    // Divider B
    dividerB = vaddvq_f64(mmdividerB);

    // Similarity
    similarity = vaddvq_f64(mmsimilarity);

    // Calculate the remaining elements. 
    for(; i < vector_size; ++i ) {
        double A = searched_array[i];
        double B = column_array[i];
        similarity += (A*B);
        dividerA   += (A*A);
        dividerB   += (B*B);
    }

    *divider_a = dividerA;
    *divider_b = dividerB;
    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_f_neon
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of floats.
//       AVX version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
float euclidean_distance_similarity_f_neon( const float* searched_array
                                           ,const float* column_array
                                           ,const int    vector_size ) 
{
    float similarity = 0.0f;
    int i = 0; 
    float32x4_t A, B, AB, ABAB, sumAB = vdupq_n_f32(0.0f);
    for( ; i + 3 < vector_size; i += 4 ) {
        A = vld1q_f32(&searched_array[i]);
        B = vld1q_f32(&column_array[i]);
        AB = vsubq_f32( A, B );
        
        sumAB = vfmaq_f32(sumAB, AB, AB);
    }//endfor i+4

    similarity = vaddvq_f32( sumAB );

    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        float AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_d_neon
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of doubles.
//       ARM Neon version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
double euclidean_distance_similarity_d_neon( const double* searched_array
                                            ,const double* column_array
                                            ,const int    vector_size ) 
{
    double similarity = 0.0;
    int i = 0; 
    float64x2_t A, B, AB, ABAB, sumAB = vdupq_n_f64(0.0);
    for( ; i + 1 < vector_size; i += 2 ) {
        A = vld1q_f64(&searched_array[i]);
        B = vld1q_f64(&column_array[i]);
        AB = vsubq_f64( A, B );
        
        sumAB = vfmaq_f64(sumAB, AB, AB);
    }//endfor i+2

    similarity = vaddvq_f64( sumAB );

    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        double AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_f_neon
// Desc: Calculates the dot product similarity to a BLOB-converted array of floats.
//       ARM Neon version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product FLOAT
//----------------------------------------------------------------------------------------
float dot_product_similarity_f_neon( const float* searched_array 
                                    ,const float* column_array 
                                    ,const int    vector_size ) 
{
    float similarity = 0.0f;
    int i = 0;
    float32x4_t A, B, AB, sumAB = vdupq_n_f32(0.0f);
    for( ; i + 3 < vector_size; i += 4 ) {
        A = vld1q_f32(&searched_array[i]);
        B = vld1q_f32(&column_array[i]);
        sumAB = vfmaq_f32(sumAB, A, B);
    }// endfor i+4
    
    similarity = vaddvq_f32( sumAB );

    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

    return similarity;
}




//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_d_neon
// Desc: Calculates the dot product similarity to a BLOB-converted array of doubles.
//       AVX version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
double dot_product_similarity_d_neon(  const double* searched_array 
                                      ,const double* column_array 
                                      ,const int     vector_size ) 
{
    double similarity = 0.0;
    int i = 0;
    float64x2_t A, B, AB, sumAB = vdupq_n_f64(0.0);
    for( ; i + 1 < vector_size; i += 2 ) {
        A = vld1q_f64(&searched_array[i]);
        B = vld1q_f64(&column_array[i]);
        sumAB = vfmaq_f64(sumAB, A, B);
    }//endfor i + 2

    similarity = vaddvq_f64( sumAB ); 
    
    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

  return similarity;
}



#endif // if aarch64
#endif 