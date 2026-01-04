#ifndef SIMILARITY_FUNCTIONS_AVX2_H_INCLUDED
#define SIMILARITY_FUNCTIONS_AVX2_H_INCLUDED
/**
 * This file contains the AVX2 versions of the similarity function definitions. 
 */
#include <immintrin.h>




//----------------------------------------------------------------------------------------
// Name: cosine_similarity_f_avx2
// Desc: Calculates the cosine similarity using two given float arrays. AVX2 version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a FLOAT 
//       Pointer to divider_b FLOAT
// Returns: Similarity as an angle float
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif 
float cosine_similarity_f_avx2( 
     const float*   searched_array 
    ,const float*   column_array 
    ,const int      vector_size
    ,float*         divider_a 
    ,float*         divider_b )
{
    int i = 0;
    float dividerA   = 0.0f
         ,dividerB   = 0.0f
         ,similarity = 0.0f;
    __m256 A
          ,B 
          ,mmdividerA   = _mm256_setzero_ps()
          ,mmdividerB   = _mm256_setzero_ps()
          ,mmsimilarity = _mm256_setzero_ps();
    
    for( ; i + 7 < vector_size; i += 8 ) {
        A = _mm256_loadu_ps(&searched_array[i]);
        B = _mm256_loadu_ps(&column_array[i]);
         // Fused multiply-add supported (AVX2).
        mmdividerA = _mm256_fmadd_ps(A, A, mmdividerA);
        mmdividerB = _mm256_fmadd_ps(B, B, mmdividerB);
        mmsimilarity = _mm256_fmadd_ps(A, B, mmsimilarity);
    }//endfor i+8

    __m128 vlow   = _mm256_castps256_ps128(mmdividerA);
    __m128 vhigh  = _mm256_extractf128_ps(mmdividerA, 1);
            vlow   = _mm_add_ps(vlow, vhigh);
    __m128 high64 = _mm_movehl_ps( vlow, vlow );
    __m128 sum    = _mm_add_ps(vlow, high64);
            sum    = _mm_add_ss(sum, _mm_shuffle_ps( sum, sum, 0x55));
    dividerA = _mm_cvtss_f32(sum); 

    vlow   = _mm256_castps256_ps128(mmdividerB);
    vhigh  = _mm256_extractf128_ps(mmdividerB, 1);
    vlow   = _mm_add_ps(vlow, vhigh);
    high64 = _mm_movehl_ps( vlow, vlow );
    sum    = _mm_add_ps(vlow, high64);
    sum    = _mm_add_ss(sum, _mm_shuffle_ps( sum, sum, 0x55));
    dividerB = _mm_cvtss_f32(sum); 

    vlow   = _mm256_castps256_ps128(mmsimilarity);
    vhigh  = _mm256_extractf128_ps(mmsimilarity, 1);
    vlow   = _mm_add_ps(vlow, vhigh);
    high64 = _mm_movehl_ps( vlow, vlow );
    sum    = _mm_add_ps(vlow, high64);
    sum    = _mm_add_ss(sum, _mm_shuffle_ps( sum, sum, 0x55));
    similarity = _mm_cvtss_f32(sum); 

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
// Name: cosine_similarity_d_avx2
// Desc: Calculates the cosine similarity using two given double arrays. AVX2 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a DOUBLE 
//       Pointer to divider_b DOUBLE
// Returns: Similarity as an angle DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif 
double cosine_similarity_d_avx2( 
     const double*   searched_array 
    ,const double*   column_array 
    ,const int      vector_size
    ,double*         divider_a 
    ,double*         divider_b )
{
    int i = 0;
    double dividerA   = 0.0
          ,dividerB   = 0.0
          ,similarity = 0.0;
    __m256d A
           ,B 
           ,mmdividerA   = _mm256_setzero_pd()
           ,mmdividerB   = _mm256_setzero_pd()
           ,mmsimilarity = _mm256_setzero_pd();
    
    for( ; i + 3 < vector_size; i += 4 ) {
        A = _mm256_loadu_pd(&searched_array[i]);
        B = _mm256_loadu_pd(&column_array[i]);
         // Fused multiply-add supported (AVX2).
        mmdividerA = _mm256_fmadd_pd(A, A, mmdividerA);
        mmdividerB = _mm256_fmadd_pd(B, B, mmdividerB);
        mmsimilarity = _mm256_fmadd_pd(A, B, mmsimilarity);
    }//endfor i+4

    // The following is based on code from stack overflow. 
    // https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx/49943540#49943540
    __m128d vlow  = _mm256_castpd256_pd128(mmdividerA);
    __m128d vhigh = _mm256_extractf128_pd(mmdividerA, 1); 
            vlow  = _mm_add_pd(vlow, vhigh);     

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    dividerA = _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); 

    vlow  = _mm256_castpd256_pd128(mmdividerB);
    vhigh = _mm256_extractf128_pd(mmdividerB, 1); 
    vlow  = _mm_add_pd(vlow, vhigh);     

    high64 = _mm_unpackhi_pd(vlow, vlow);
    dividerB = _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); 

    vlow  = _mm256_castpd256_pd128(mmsimilarity);
    vhigh = _mm256_extractf128_pd(mmsimilarity, 1); 
    vlow  = _mm_add_pd(vlow, vhigh);     

    high64 = _mm_unpackhi_pd(vlow, vlow);
    similarity = _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); 

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
// Name: euclidean_distance_similarity_f_avx2
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of floats.
//       AVX2 version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif 
float euclidean_distance_similarity_f_avx2(  const float* searched_array
                                            ,const float* column_array
                                            ,const int    vector_size ) 
{
    float similarity = 0.0f;
    int i = 0; 

    // AVX2 can handle 8 at a time.
    __m256 A, B, AB, ABAB, sumAB = _mm256_setzero_ps();
    for( ; i + 7 < vector_size; i += 8 ) {
        A = _mm256_loadu_ps(&searched_array[i]);
        B = _mm256_loadu_ps(&column_array[i]);
        AB = _mm256_sub_ps( A, B );
        // Fused multiply-add supported (AVX2).
        sumAB = _mm256_fmadd_ps(AB, AB, sumAB );
    }//endfor i+8
    __m128 vlow   = _mm256_castps256_ps128(sumAB);
    __m128 vhigh  = _mm256_extractf128_ps(sumAB, 1);
           vlow   = _mm_add_ps(vlow, vhigh);
    __m128 high64 = _mm_movehl_ps( vlow, vlow );
    __m128 sum    = _mm_add_ps(vlow, high64);
           sum    = _mm_add_ss(sum, _mm_shuffle_ps( sum, sum, 0x55));
    similarity = _mm_cvtss_f32(sum);
    
    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        float AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}






//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_d_avx2
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of double.
//       AVX2 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif 
double euclidean_distance_similarity_d_avx2( const double* searched_array
                                            ,const double* column_array
                                            ,const int    vector_size ) 
{
    double similarity = 0.0;
    int i = 0; 

    // AVX2 can handle 4 at a time.
    __m256d A, B, AB, ABAB, sumAB = _mm256_setzero_pd();
    for( ; i + 3 < vector_size; i += 4 ) {
        A = _mm256_loadu_pd(&searched_array[i]);
        B = _mm256_loadu_pd(&column_array[i]);
        AB = _mm256_sub_pd( A, B );
        // Fused multiply-add supported (AVX2).
        sumAB = _mm256_fmadd_pd(AB, AB, sumAB );
    }//endfor i+4
    __m128d vlow  = _mm256_castpd256_pd128(sumAB);
    __m128d vhigh = _mm256_extractf128_pd(sumAB, 1); 
            vlow  = _mm_add_pd(vlow, vhigh);     

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    similarity = _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); 
    
    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        double AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_f_avx2
// Desc: Calculates the dot product similarity to a BLOB-converted array of floats.
//       AVX2 version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product FLOAT
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif 
float dot_product_similarity_f_avx2(  const float* searched_array 
                                     ,const float* column_array 
                                     ,const int    vector_size ) 
{
    float similarity = 0.0f;
    int i = 0;
    __m256 A, B, AB, sumAB = _mm256_setzero_ps();
    for( ; i + 7 < vector_size; i += 8 ) {
        A = _mm256_loadu_ps(&searched_array[i]);
        B = _mm256_loadu_ps(&column_array[i]);
        sumAB = _mm256_fmadd_ps(A, B, sumAB );
    }//endfor i + 8

    __m128 vlow   = _mm256_castps256_ps128(sumAB);
    __m128 vhigh  = _mm256_extractf128_ps(sumAB, 1);
           vlow   = _mm_add_ps(vlow, vhigh);
    __m128 high64 = _mm_movehl_ps( vlow, vlow );
    __m128 sum    = _mm_add_ps(vlow, high64);
           sum    = _mm_add_ss(sum, _mm_shuffle_ps( sum, sum, 0x55));
    similarity = _mm_cvtss_f32(sum);
    
    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

  return similarity;
}




//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_d_avx2
// Desc: Calculates the dot product similarity to a BLOB-converted array of doubles.
//       AVX2 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif 
double dot_product_similarity_d_avx2( const double* searched_array 
                                     ,const double* column_array 
                                     ,const int     vector_size ) 
{
    double similarity = 0.0;
    int i = 0;
    __m256d A, B, AB, sumAB = _mm256_setzero_pd();
    for( ; i + 3 < vector_size; i += 4 ) {
        A = _mm256_loadu_pd(&searched_array[i]);
        B = _mm256_loadu_pd(&column_array[i]);
        sumAB = _mm256_fmadd_pd(A, B, sumAB );
    }//endfor i + 4

    __m128d vlow  = _mm256_castpd256_pd128(sumAB);
    __m128d vhigh = _mm256_extractf128_pd(sumAB, 1); 
            vlow  = _mm_add_pd(vlow, vhigh);     

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    similarity = _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); 
    
    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

  return similarity;
}



#endif 