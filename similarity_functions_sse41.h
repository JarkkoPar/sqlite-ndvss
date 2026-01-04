#ifndef SIMILARITY_FUNCTIONS_SSE41_H_INCLUDED
#define SIMILARITY_FUNCTIONS_SSE41_H_INCLUDED
/**
 * This file contains the SSE 4.1 versions of the similarity function definitions. 
 */
#include <immintrin.h>


//----------------------------------------------------------------------------------------
// Name: cosine_similarity_f_sse41
// Desc: Calculates the cosine similarity using two given float arrays. SSE 4.1 version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a FLOAT 
//       Pointer to divider_b FLOAT
// Returns: Similarity as an angle float
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__) || defined(__llvm__)
__attribute__((target("sse4.1")))
#endif 
float cosine_similarity_f_sse41( 
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
    __m128 A
          ,B
          ,AA
          ,BB
          ,AB
          ,mmdividerA   = _mm_setzero_ps()
          ,mmdividerB   = _mm_setzero_ps()
          ,mmsimilarity = _mm_setzero_ps();
    
    for( ; i + 3 < vector_size; i += 4 ) {
        A = _mm_loadu_ps(&searched_array[i]);
        B = _mm_loadu_ps(&column_array[i]);
        AA = _mm_mul_ps(A, A);
        BB = _mm_mul_ps(B, B);
        AB = _mm_mul_ps(A, B);
        mmdividerA = _mm_add_ps(AA, mmdividerA);
        mmdividerB = _mm_add_ps(BB, mmdividerB);
        mmsimilarity = _mm_add_ps(AB, mmsimilarity);    
    }//endfor i+4

    // Divider A
    __m128 shuf = _mm_movehl_ps(mmdividerA, mmdividerA);
    __m128 sums = _mm_add_ps(mmdividerA, shuf);
    shuf        = _mm_shuffle_ps(sums, sums, 1);
    sums        = _mm_add_ss(sums, shuf);
    dividerA    = _mm_cvtss_f32(sums); 

    // Divider B
    shuf     = _mm_movehl_ps(mmdividerB, mmdividerB);
    sums     = _mm_add_ps(mmdividerB, shuf);
    shuf     = _mm_shuffle_ps(sums, sums, 1);
    sums     = _mm_add_ss(sums, shuf);
    dividerB = _mm_cvtss_f32(sums); 

    // Similarity
    shuf       = _mm_movehl_ps(mmsimilarity, mmsimilarity);
    sums       = _mm_add_ps(mmsimilarity, shuf);
    shuf       = _mm_shuffle_ps(sums, sums, 1);
    sums       = _mm_add_ss(sums, shuf);
    similarity = _mm_cvtss_f32(sums);

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
// Name: cosine_similarity_d_sse41
// Desc: Calculates the cosine similarity using two given double arrays. SSE 4.1 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a DOUBLE 
//       Pointer to divider_b DOUBLE
// Returns: Similarity as an angle DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__) || defined(__llvm__)
__attribute__((target("sse4.1")))
#endif 
double cosine_similarity_d_sse41( 
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
    __m128d A
           ,B
           ,AA
           ,BB
           ,AB
           ,mmdividerA   = _mm_setzero_pd()
           ,mmdividerB   = _mm_setzero_pd()
           ,mmsimilarity = _mm_setzero_pd();
    
    for( ; i + 1 < vector_size; i += 2 ) {
        A = _mm_loadu_pd(&searched_array[i]);
        B = _mm_loadu_pd(&column_array[i]);
        AA = _mm_mul_pd(A, A);
        BB = _mm_mul_pd(B, B);
        AB = _mm_mul_pd(A, B);
        mmdividerA = _mm_add_pd(AA, mmdividerA);
        mmdividerB = _mm_add_pd(BB, mmdividerB);
        mmsimilarity = _mm_add_pd(AB, mmsimilarity);    
    }//endfor i+2

    // Divider A
    __m128d high = _mm_unpackhi_pd(mmdividerA, mmdividerA);
    __m128d sum  = _mm_add_sd(mmdividerA, high); 
    dividerA     = _mm_cvtsd_f64(sum);

    // Divider B
    high     = _mm_unpackhi_pd(mmdividerB, mmdividerB);
    sum      = _mm_add_sd(mmdividerB, high);
    dividerB = _mm_cvtsd_f64(sum);

    // Similarity
    high       = _mm_unpackhi_pd(mmsimilarity, mmsimilarity);
    sum        = _mm_add_sd(mmsimilarity, high);
    similarity = _mm_cvtsd_f64(sum);

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
// Name: euclidean_distance_similarity_f_sse41
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of floats.
//       AVX version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__) || defined(__llvm__)
__attribute__((target("sse4.1")))
#endif 
float euclidean_distance_similarity_f_sse41( const float* searched_array
                                            ,const float* column_array
                                            ,const int    vector_size ) 
{
    float similarity = 0.0f;
    int i = 0; 
    // SSE 4.1 can handle 4 at a time. 
    __m128 A, B, AB, ABAB, sumAB = _mm_setzero_ps();
    for( ; i + 3 < vector_size; i += 4 ) {
        A = _mm_loadu_ps(&searched_array[i]);
        B = _mm_loadu_ps(&column_array[i]);
        AB = _mm_sub_ps( A, B );
        
        ABAB = _mm_mul_ps(AB, AB);
        sumAB = _mm_add_ps(ABAB, sumAB);
    }//endfor i+4

    __m128 shuffle = _mm_movehl_ps( sumAB, sumAB );
    __m128 sums    = _mm_add_ps( sumAB, shuffle );
           shuffle = _mm_shuffle_ps( sums, sums, 1 );
           sumAB   = _mm_add_ss( sums, shuffle );
    similarity = _mm_cvtss_f32( sumAB );

    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        float AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_d_sse41
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of doubles.
//       SSE 4.1 version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__) || defined(__llvm__)
__attribute__((target("sse4.1")))
#endif 
double euclidean_distance_similarity_d_sse41( const double* searched_array
                                             ,const double* column_array
                                             ,const int    vector_size ) 
{
    double similarity = 0.0;
    int i = 0; 
    // SSE 4.1 can handle 2 at a time. 
    __m128d A, B, AB, ABAB, sumAB = _mm_setzero_pd();
    for( ; i + 1 < vector_size; i += 2 ) {
        A = _mm_loadu_pd(&searched_array[i]);
        B = _mm_loadu_pd(&column_array[i]);
        AB = _mm_sub_pd( A, B );
        
        ABAB = _mm_mul_pd(AB, AB);
        sumAB = _mm_add_pd(ABAB, sumAB);
    }//endfor i+2

    __m128d vhigh  = _mm_unpackhi_pd( sumAB, sumAB );
    __m128d result = _mm_add_sd(sumAB, vhigh );
    similarity = _mm_cvtsd_f64( result ); 

    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        double AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_f_sse41
// Desc: Calculates the dot product similarity to a BLOB-converted array of floats.
//       SSE 4.1 version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product FLOAT
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__) || defined(__llvm__)
__attribute__((target("sse4.1")))
#endif 
float dot_product_similarity_f_sse41( const float* searched_array 
                                     ,const float* column_array 
                                     ,const int    vector_size ) 
{
    float similarity = 0.0f;
    int i = 0;
    __m128 A, B, AB, sumAB = _mm_setzero_ps();
    for( ; i + 3 < vector_size; i += 4 ) {
        A = _mm_loadu_ps(&searched_array[i]);
        B = _mm_loadu_ps(&column_array[i]);
        AB = _mm_mul_ps(A, B);
        sumAB = _mm_add_ps(AB, sumAB);
    }// endfor i+4
    
    __m128 shuffle = _mm_movehl_ps( sumAB, sumAB );
    __m128 sums    = _mm_add_ps( sumAB, shuffle );
           shuffle = _mm_shuffle_ps( sums, sums, 1 );
           sumAB   = _mm_add_ss( sums, shuffle );
    similarity = _mm_cvtss_f32( sumAB );

    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

    return similarity;
}




//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_d_sse41
// Desc: Calculates the dot product similarity to a BLOB-converted array of doubles.
//       AVX version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__) || defined(__llvm__)
__attribute__((target("sse4.1")))
#endif 
double dot_product_similarity_d_sse41(  const double* searched_array 
                                       ,const double* column_array 
                                       ,const int     vector_size ) 
{
    double similarity = 0.0;
    int i = 0;
    __m128d A, B, AB, sumAB = _mm_setzero_pd();
    for( ; i + 1 < vector_size; i += 2 ) {
        A = _mm_loadu_pd(&searched_array[i]);
        B = _mm_loadu_pd(&column_array[i]);
        AB = _mm_mul_pd(A, B);
        sumAB = _mm_add_pd(AB, sumAB);
    }//endfor i + 2

    __m128d vhigh  = _mm_unpackhi_pd( sumAB, sumAB );
    __m128d result = _mm_add_sd(sumAB, vhigh );
    similarity = _mm_cvtsd_f64( result ); 
    
    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

  return similarity;
}




#endif 