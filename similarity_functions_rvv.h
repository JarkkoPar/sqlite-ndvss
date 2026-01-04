#ifndef SIMILARITY_FUNCTIONS_RVV_H_INCLUDED
#define SIMILARITY_FUNCTIONS_RVV_H_INCLUDED
#if defined(__riscv) && defined(__riscv_vector)
/**
 * This file contains the Risc-V Vector Extension versions of the similarity function definitions. 
 */
#include <riscv_vector.h>


//----------------------------------------------------------------------------------------
// Name: cosine_similarity_f_rvv
// Desc: Calculates the cosine similarity using two given float arrays. RISC-V Vector version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a FLOAT 
//       Pointer to divider_b FLOAT
// Returns: Similarity as an angle float
//----------------------------------------------------------------------------------------
float cosine_similarity_f_rvv( 
     const float*   searched_array 
    ,const float*   column_array 
    ,const int      vector_size
    ,float*         divider_a 
    ,float*         divider_b )
{
    float dividerA   = 0.0f
         ,dividerB   = 0.0f
         ,similarity = 0.0f;
    size_t elements_to_handle = vector_size;
    size_t vector_length = __riscv_vsetvlmax_e32m8(); // Get the vector length. 
    vfloat32m8_t A
                ,B
                ,mmdividerA   = __riscv_vfmv_v_f_f32m8(0.0f, vector_length)
                ,mmdividerB   = __riscv_vfmv_v_f_f32m8(0.0f, vector_length)
                ,mmsimilarity = __riscv_vfmv_v_f_f32m8(0.0f, vector_length);

    int i = 0;
    for( ; elements_to_handle > 0; elements_to_handle -= vector_length ) {
        // We make a query, telling that we have n elements to handle
        // and asking how many the CPU can handle at once. 
        vector_length = __riscv_vsetvl_e32m8(elements_to_handle); 

        A = __riscv_vle32_v_f32m8(&searched_array[i], vector_length);
        B = __riscv_vle32_v_f32m8(&column_array[i], vector_length);

        mmdividerA   = __riscv_vfmacc_vv_f32m8(mmdividerA, A, A, vector_length);
        mmdividerB   = __riscv_vfmacc_vv_f32m8(mmdividerB, B, B, vector_length);
        mmsimilarity = __riscv_vfmacc_vv_f32m8(mmsimilarity, A, B, vector_length);
        i += vector_length;
    }// endfor vector_length

    vfloat32m1_t result_vector = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());

    // Reduce the vector to a float by summing the components. 
    vector_length = __riscv_vsetvlmax_e32m8();
    result_vector = __riscv_vfredusum_vs_f32m8_f32m1( result_vector, mmdividerA, result_vector, vector_length );
    dividerA = __riscv_vfmv_f_s_f32m1_f32( result_vector );
    
    result_vector = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
    result_vector = __riscv_vfredusum_vs_f32m8_f32m1( result_vector, mmdividerB, result_vector, vector_length );
    dividerB = __riscv_vfmv_f_s_f32m1_f32( result_vector );

    result_vector = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
    result_vector = __riscv_vfredusum_vs_f32m8_f32m1( result_vector, mmsimilarity, result_vector, vector_length );
    similarity = __riscv_vfmv_f_s_f32m1_f32( result_vector );

    *divider_a = dividerA;
    *divider_b = dividerB;
    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: cosine_similarity_d_rvv
// Desc: Calculates the cosine similarity using two given double arrays. RISC-V Vector version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a DOUBLE 
//       Pointer to divider_b DOUBLE
// Returns: Similarity as an angle DOUBLE
//----------------------------------------------------------------------------------------
double cosine_similarity_d_rvv( 
     const double*   searched_array 
    ,const double*   column_array 
    ,const int       vector_size
    ,double*         divider_a 
    ,double*         divider_b )
{
    double similarity = 0.0
          ,dividerA   = 0.0
          ,dividerB   = 0.0;
    
    size_t elements_to_handle = vector_size;
    size_t vector_length = __riscv_vsetvlmax_e64m8(); // Get the vector length. 
    vfloat64m8_t A
                ,B
                ,mmdividerA   = __riscv_vfmv_v_f_f64m8(0.0, vector_length)
                ,mmdividerB   = __riscv_vfmv_v_f_f64m8(0.0, vector_length)
                ,mmsimilarity = __riscv_vfmv_v_f_f64m8(0.0, vector_length);

    int i = 0;
    for( ; elements_to_handle > 0; elements_to_handle -= vector_length ) {
        // We make a query, telling that we have n elements to handle
        // and asking how many the CPU can handle at once. 
        vector_length = __riscv_vsetvl_e64m8(elements_to_handle); 

        A = __riscv_vle64_v_f64m8(&searched_array[i], vector_length);
        B = __riscv_vle64_v_f64m8(&column_array[i], vector_length);

        mmdividerA   = __riscv_vfmacc_vv_f64m8(mmdividerA, A, A, vector_length);
        mmdividerB   = __riscv_vfmacc_vv_f64m8(mmdividerB, B, B, vector_length);
        mmsimilarity = __riscv_vfmacc_vv_f64m8(mmsimilarity, A, B, vector_length);
        i += vector_length;
    }// endfor vector_length

    vfloat64m1_t result_vector = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvlmax_e64m1());

    // Reduce the vector to a float by summing the components. 
    vector_length = __riscv_vsetvlmax_e64m8();
    result_vector = __riscv_vfredusum_vs_f64m8_f64m1( result_vector, mmdividerA, result_vector, vector_length );
    dividerA = __riscv_vfmv_f_s_f64m1_f64( result_vector );
    
    result_vector = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvlmax_e64m1());
    result_vector = __riscv_vfredusum_vs_f64m8_f64m1( result_vector, mmdividerB, result_vector, vector_length );
    dividerB = __riscv_vfmv_f_s_f64m1_f64( result_vector );

    result_vector = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvlmax_e64m1());
    result_vector = __riscv_vfredusum_vs_f64m8_f64m1( result_vector, mmsimilarity, result_vector, vector_length );
    similarity = __riscv_vfmv_f_s_f64m1_f64( result_vector );

    *divider_a = dividerA;
    *divider_b = dividerB;
    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_f_rvv
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of floats.
//       AVX version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
float euclidean_distance_similarity_f_rvv( const float* searched_array
                                          ,const float* column_array
                                          ,const int    vector_size ) 
{
    float similarity = 0.0f;
    size_t elements_to_handle = vector_size;
    size_t vector_length = __riscv_vsetvlmax_e32m8(); // Get the vector length. 
    vfloat32m8_t A, B, AB, sumAB = __riscv_vfmv_v_f_f32m8(0.0f, vector_length);

    int i = 0;
    for( ; elements_to_handle > 0; elements_to_handle -= vector_length ) {
        // We make a query, telling that we have n elements to handle
        // and asking how many the CPU can handle at once. 
        vector_length = __riscv_vsetvl_e32m8(elements_to_handle); 

        A = __riscv_vle32_v_f32m8(&searched_array[i], vector_length);
        B = __riscv_vle32_v_f32m8(&column_array[i], vector_length);

        AB = __riscv_vfsub_vv_f32m8(A, B, vector_length);

        sumAB = __riscv_vfmacc_vv_f32m8(sumAB, AB, AB, vector_length);
        i += vector_length;
    }// endfor vector_length

    vfloat32m1_t result_vector = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());

    // Reduce the vector to a float by summing the components. 
    vector_length = __riscv_vsetvlmax_e32m8();
    result_vector = __riscv_vfredusum_vs_f32m8_f32m1( result_vector, sumAB, result_vector, vector_length );
    
    // Return the result as a float.
    similarity = __riscv_vfmv_f_s_f32m1_f32( result_vector );

    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_d_rvv
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of doubles.
//       RISC-V Vector version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
double euclidean_distance_similarity_d_rvv( const double* searched_array
                                           ,const double* column_array
                                           ,const int    vector_size ) 
{
    double similarity = 0.0;
    size_t elements_to_handle = vector_size;
    size_t vector_length = __riscv_vsetvlmax_e64m8(); // Get the vector length. 
    vfloat64m8_t A, B, AB, sumAB = __riscv_vfmv_v_f_f64m8(0.0, vector_length);

    int i = 0;
    for( ; elements_to_handle > 0; elements_to_handle -= vector_length ) {
        // We make a query, telling that we have n elements to handle
        // and asking how many the CPU can handle at once. 
        vector_length = __riscv_vsetvl_e64m8(elements_to_handle); 

        A = __riscv_vle64_v_f64m8(&searched_array[i], vector_length);
        B = __riscv_vle64_v_f64m8(&column_array[i], vector_length);

        AB = __riscv_vfsub_vv_f64m8(A, B, vector_length);

        sumAB = __riscv_vfmacc_vv_f64m8(sumAB, AB, AB, vector_length);
        i += vector_length;
    }// endfor vector_length

    vfloat64m1_t result_vector = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvlmax_e64m1());

    // Reduce the vector to a double by summing the components. 
    vector_length = __riscv_vsetvlmax_e64m8();
    result_vector = __riscv_vfredusum_vs_f64m8_f64m1( result_vector, sumAB, result_vector, vector_length );
    
    // Return the result as a double.
    similarity = __riscv_vfmv_f_s_f64m1_f64( result_vector );

    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_f_rvv
// Desc: Calculates the dot product similarity to a BLOB-converted array of floats.
//       RISC-V Vector version.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product FLOAT
//----------------------------------------------------------------------------------------
float dot_product_similarity_f_rvv( const float* searched_array 
                                   ,const float* column_array 
                                   ,const int    vector_size ) 
{
    float similarity = 0.0f;
    size_t elements_to_handle = vector_size;
    size_t vector_length = __riscv_vsetvlmax_e32m8(); // Get the vector length. 
    vfloat32m8_t A, B, sumAB = __riscv_vfmv_v_f_f32m8(0.0f, vector_length);

    // Contrary to the others, here we go through the arrays until
    // they are fully handled. 
    int i = 0;
    for( ; elements_to_handle > 0; elements_to_handle -= vector_length ) {
        // We make a query, telling that we have n elements to handle
        // and asking how many the CPU can handle at once. 
        vector_length = __riscv_vsetvl_e32m8(elements_to_handle); 

        A = __riscv_vle32_v_f32m8(&searched_array[i], vector_length);
        B = __riscv_vle32_v_f32m8(&column_array[i], vector_length);
        sumAB = __riscv_vfmacc_vv_f32m8(sumAB, A, B, vector_length);
        i += vector_length;
    }// endfor vector_length
    
    vfloat32m1_t result_vector = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());

    // Reduce the vector to a float by summing the components. 
    vector_length = __riscv_vsetvlmax_e32m8();
    result_vector = __riscv_vfredusum_vs_f32m8_f32m1( result_vector, sumAB, result_vector, vector_length );
    
    // Return the result as a float.
    similarity = __riscv_vfmv_f_s_f32m1_f32( result_vector );
    return similarity;
}




//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_d_rvv
// Desc: Calculates the dot product similarity to a BLOB-converted array of doubles.
//       AVX version.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
double dot_product_similarity_d_rvv( const double* searched_array 
                                    ,const double* column_array 
                                    ,const int     vector_size ) 
{
    double similarity = 0.0f;
    size_t elements_to_handle = vector_size;
    size_t vector_length = __riscv_vsetvlmax_e64m8(); // Get the vector length. 
    vfloat64m8_t A, B, sumAB = __riscv_vfmv_v_f_f64m8(0.0, vector_length);

    // Contrary to the others, here we go through the arrays until
    // they are fully handled. 
    int i = 0;
    for( ; elements_to_handle > 0; elements_to_handle -= vector_length ) {
        // We make a query, telling that we have n elements to handle
        // and asking how many the CPU can handle at once. 
        vector_length = __riscv_vsetvl_e64m8(elements_to_handle); 

        A = __riscv_vle64_v_f64m8(&searched_array[i], vector_length);
        B = __riscv_vle64_v_f64m8(&column_array[i], vector_length);
        sumAB = __riscv_vfmacc_vv_f64m8(sumAB, A, B, vector_length);
        i += vector_length;
    }// endfor vector_length
    
    vfloat64m1_t result_vector = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvlmax_e64m1());

    // Reduce the vector to a double by summing the components. 
    vector_length = __riscv_vsetvlmax_e64m8();
    result_vector = __riscv_vfredusum_vs_f64m8_f64m1( result_vector, sumAB, result_vector, vector_length );
    
    // Return the result as a double.
    similarity = __riscv_vfmv_f_s_f64m1_f64( result_vector );
    return similarity;
}



#endif // if riscv & riscv vector
#endif 