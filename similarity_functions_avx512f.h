#ifndef SIMILARITY_FUNCTINOS_AVX512F_H_INCLUDED
#define SIMILARITY_FUNCTINOS_AVX512F_H_INCLUDED
/**
 * This file contains the AVX512f versions of the similarity function definitions. 
 */
#include <immintrin.h>



//----------------------------------------------------------------------------------------
// Name: cosine_similarity_f_avx512f
// Desc: Calculates the cosine similarity using two given float arrays. 
//       AVX512f version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a FLOAT 
//       Pointer to divider_b FLOAT
// Returns: Similarity as an angle float
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif 
static float cosine_similarity_f_avx512f( 
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
    __m512 A
          ,B 
          ,mmdividerA   = _mm512_setzero_ps()
          ,mmdividerB   = _mm512_setzero_ps()
          ,mmsimilarity = _mm512_setzero_ps();
    
    for( ; i + 15 < vector_size; i += 16 ) {
        A = _mm512_loadu_ps(&searched_array[i]);
        B = _mm512_loadu_ps(&column_array[i]);
        
        mmdividerA = _mm512_fmadd_ps(A, A, mmdividerA);
        mmdividerB = _mm512_fmadd_ps(B, B, mmdividerB);
        mmsimilarity = _mm512_fmadd_ps(A, B, mmsimilarity);
    }//endfor i+16

    dividerA = _mm512_reduce_add_ps(mmdividerA); 
    dividerB = _mm512_reduce_add_ps(mmdividerB);
    similarity = _mm512_reduce_add_ps(mmsimilarity); 

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
// Name: cosine_similarity_d_avx512f
// Desc: Calculates the cosine similarity using two given double arrays. 
//       AVX512f version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a DOUBLE 
//       Pointer to divider_b DOUBLE
// Returns: Similarity as an angle DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif 
static double cosine_similarity_d_avx512f( 
     const double*   searched_array 
    ,const double*   column_array 
    ,const int       vector_size
    ,double*         divider_a 
    ,double*         divider_b )
{
    int i = 0;
    double dividerA   = 0.0
          ,dividerB   = 0.0
          ,similarity = 0.0;
    __m512d A
           ,B 
           ,mmdividerA   = _mm512_setzero_pd()
           ,mmdividerB   = _mm512_setzero_pd()
           ,mmsimilarity = _mm512_setzero_pd();
    
    for( ; i + 7 < vector_size; i += 8 ) {
        A = _mm512_loadu_pd(&searched_array[i]);
        B = _mm512_loadu_pd(&column_array[i]);
        
        mmdividerA = _mm512_fmadd_pd(A, A, mmdividerA);
        mmdividerB = _mm512_fmadd_pd(B, B, mmdividerB);
        mmsimilarity = _mm512_fmadd_pd(A, B, mmsimilarity);
    }//endfor i+8

    dividerA = _mm512_reduce_add_pd(mmdividerA); 
    dividerB = _mm512_reduce_add_pd(mmdividerB); 
    similarity = _mm512_reduce_add_pd(mmsimilarity); 

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
// Name: euclidean_distance_similarity_f_avx512f
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of floats.
//       AVX512f version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif 
static float euclidean_distance_similarity_f_avx512f(  const float* searched_array,
                                                       const float* column_array,
                                                       const int    vector_size ) 
{
    float similarity = 0.0f;
    int i = 0; 

    // AVX512f can handle 16 at a time.
    __m512 A, B, AB, ABAB, sumAB = _mm512_setzero_ps();
    for( ; i + 15 < vector_size; i += 16 ) {
        A = _mm512_loadu_ps(&searched_array[i]);
        B = _mm512_loadu_ps(&column_array[i]);
        AB = _mm512_sub_ps( A, B );

        sumAB = _mm512_fmadd_ps(AB, AB, sumAB );
    }//endfor i+16

    similarity = _mm512_reduce_add_ps(sumAB);
    
    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        float AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_d_avx512f
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of double.
//       AVX2 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif 
static double euclidean_distance_similarity_d_avx512f( const double* searched_array,
                                                       const double* column_array,
                                                       const int    vector_size ) 
{
    double similarity = 0.0;
    int i = 0; 

    __m512d A, B, AB, ABAB, sumAB = _mm512_setzero_pd();
    for( ; i + 7 < vector_size; i += 8 ) {
        A = _mm512_loadu_pd(&searched_array[i]);
        B = _mm512_loadu_pd(&column_array[i]);
        AB = _mm512_sub_pd( A, B );
        
        sumAB = _mm512_fmadd_pd(AB, AB, sumAB );
    }//endfor i+8

    similarity = _mm512_reduce_add_pd(sumAB); 
    
    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        float AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}





//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_f_avx512f
// Desc: Calculates the dot product similarity to a BLOB-converted array of floats.
//       AVX512f version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product FLOAT
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif 
static float dot_product_similarity_f_avx512f(  const float* searched_array 
                                               ,const float* column_array 
                                               ,const int    vector_size ) 
{
    float similarity = 0.0f;
    int i = 0;
    __m512 A, B, AB, sumAB = _mm512_setzero_ps();
    for( ; i + 15 < vector_size; i += 16 ) {
        A = _mm512_loadu_ps(&searched_array[i]);
        B = _mm512_loadu_ps(&column_array[i]);
        sumAB = _mm512_fmadd_ps(A, B, sumAB );
    }//endfor i + 8

    similarity = _mm512_reduce_add_ps(sumAB);
    
    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

  return similarity;
}




//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_d_avx512f
// Desc: Calculates the dot product similarity to a BLOB-converted array of doubles.
//       AVX2 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif 
static double dot_product_similarity_d_avx512f( const double* searched_array 
                                               ,const double* column_array 
                                               ,const int     vector_size ) 
{
    double similarity = 0.0;
    int i = 0;
    
    __m512d A, B, AB, sumAB = _mm512_setzero_pd();
    for( ; i + 7 < vector_size; i += 8 ) {
        A = _mm512_loadu_pd(&searched_array[i]);
        B = _mm512_loadu_pd(&column_array[i]);
        sumAB = _mm512_fmadd_pd(A, B, sumAB );
    }//endfor i + 8

    similarity = _mm512_reduce_add_pd(sumAB);
    
    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

  return similarity;
}



#endif 