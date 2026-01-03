#ifndef SIMILARITY_FUNCTIONS_BASIC_INCLUDED
#define SIMILARITY_FUNCTIONS_BASIC_INCLUDED

/**
 * This file contains the basic similarity function definitions. 
 */



//----------------------------------------------------------------------------------------
// Name: cosine_similarity_f_basic
// Desc: Calculates the cosine similarity using two given float arrays.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a FLOAT 
//       Pointer to divider_b FLOAT
// Returns: Similarity as an angle float
//----------------------------------------------------------------------------------------
static float cosine_similarity_f_basic( 
     const float*   searched_array
    ,const float*   column_array 
    ,const int      vector_size
    ,float*         divider_a 
    ,float*         divider_b ) 
{   
    int i = 0;
    float dividerA = 0.0f, dividerB = 0.0f;
    float similarity = 0.0f;
    for( ; i + 3 < vector_size; i += 4 ) {
        float A = searched_array[i];
        float B = column_array[i];
        similarity += (A*B);
        dividerA += (A*A);
        dividerB += (B*B);

        A = searched_array[i+1];
        B = column_array[i+1];
        similarity += (A*B);
        dividerA += (A*A);
        dividerB += (B*B);
        A = searched_array[i+2];
        B = column_array[i+2];
        similarity += (A*B);
        dividerA += (A*A);
        dividerB += (B*B);
        A = searched_array[i+3];
        B = column_array[i+3];
        similarity += (A*B);
        dividerA += (A*A);
        dividerB += (B*B);
    }//endfor i + 4 

    // Calculate the remaining elements. 
    for( ; i < vector_size; ++i ) {
        float A = searched_array[i];
        float B = column_array[i];
        similarity += (A*B);
        dividerA += (A*A);
        dividerB += (B*B);
    }//endfor i 

    *divider_a = dividerA; 
    *divider_b = dividerB; 

    return similarity; 
}



//----------------------------------------------------------------------------------------
// Name: cosine_similarity_d_basic
// Desc: Calculates the cosine similarity using two given double arrays.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
//       Pointer to divider_a DOUBLE 
//       Pointer to divider_b DOUBLE
// Returns: Similarity as an angle DOUBLE
//----------------------------------------------------------------------------------------
static double cosine_similarity_d_basic( 
     const double*  searched_array
    ,const double*  column_array 
    ,const int      vector_size
    ,double*        divider_a 
    ,double*        divider_b ) 
{   
    int i = 0;
    double dividerA = 0.0, dividerB = 0.0;
    double similarity = 0.0;
    for( ; i + 3 < vector_size; i += 4 ) {
        double A = searched_array[i];
        double B = column_array[i];
        similarity += (A*B);
        dividerA += (A*A);
        dividerB += (B*B);

        A = searched_array[i+1];
        B = column_array[i+1];
        similarity += (A*B);
        dividerA += (A*A);
        dividerB += (B*B);
        A = searched_array[i+2];
        B = column_array[i+2];
        similarity += (A*B);
        dividerA += (A*A);
        dividerB += (B*B);
        A = searched_array[i+3];
        B = column_array[i+3];
        similarity += (A*B);
        dividerA += (A*A);
        dividerB += (B*B);
    }//endfor i + 4 

    // Calculate the remaining elements. 
    for( ; i < vector_size; ++i ) {
        double A = searched_array[i];
        double B = column_array[i];
        similarity += (A*B);
        dividerA += (A*A);
        dividerB += (B*B);
    }//endfor i 

    *divider_a = dividerA; 
    *divider_b = dividerB; 

    return similarity; 
}




//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_f_basic
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of floats.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
static float euclidean_distance_similarity_f_basic( const float* searched_array,
                                                    const float* column_array,
                                                    const int    vector_size ) 
{
    float similarity = 0.0f;
    int i = 0; 
    // Handle in steps of 4 to enable basic optimizations.
    for( ; i + 3 < vector_size; i += 4 ) {
        float AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
        AB = (searched_array[i+1] - column_array[i+1]);
        similarity += (AB * AB);
        AB = (searched_array[i+2] - column_array[i+2]);
        similarity += (AB * AB);
        AB = (searched_array[i+3] - column_array[i+3]);
        similarity += (AB * AB);
    }//endfor i+4
    
    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        float AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: euclidean_distance_similarity_d_basic
// Desc: Calculates the euclidean distance similarity to a BLOB-converted array of doubles.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a distance DOUBLE
//----------------------------------------------------------------------------------------
static double euclidean_distance_similarity_d_basic( const double* searched_array,
                                                     const double* column_array,
                                                     const int     vector_size ) 
{
    double similarity = 0.0;
    int i = 0; 
    // Handle in steps of 4 to enable basic optimizations.
    for( ; i + 3 < vector_size; i += 4 ) {
        double AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
        AB = (searched_array[i+1] - column_array[i+1]);
        similarity += (AB * AB);
        AB = (searched_array[i+2] - column_array[i+2]);
        similarity += (AB * AB);
        AB = (searched_array[i+3] - column_array[i+3]);
        similarity += (AB * AB);
    }//endfor i+4
    
    // Handle the remaining elements.
    for( ; i < vector_size; ++i ) {
        double AB = (searched_array[i] - column_array[i]);
        similarity += (AB * AB);
    }

    return similarity;
}



//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_f_basic
// Desc: Calculates the dot product similarity to a BLOB-converted array of floats.
// Args: Searched float array BLOB,
//       Compared float array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product FLOAT
//----------------------------------------------------------------------------------------
static float dot_product_similarity_f_basic( const float* searched_array 
                                            ,const float* column_array 
                                            ,const int    vector_size ) 
{
    float similarity = 0.0f;  
    int i = 0;
    for( ; i + 3 < vector_size; i += 4 ) {
        similarity += ((searched_array[i]) * (column_array[i]));
        similarity += ((searched_array[i+1]) * (column_array[i+1]));
        similarity += ((searched_array[i+2]) * (column_array[i+2]));
        similarity += ((searched_array[i+3]) * (column_array[i+3]));
    }//endfor i+4

    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

    return similarity;
}


//----------------------------------------------------------------------------------------
// Name: dot_product_similarity_d_basic
// Desc: Calculates the dot product similarity to a BLOB-converted array of doubles.
// Args: Searched double array BLOB,
//       Compared double array (usually a column) BLOB, 
//       Number of dimensions INTEGER
// Returns: Similarity as a dot product DOUBLE
//----------------------------------------------------------------------------------------
static float dot_product_similarity_d_basic( const double* searched_array 
                                            ,const double* column_array 
                                            ,const int     vector_size ) 
{
    double similarity = 0.0;  
    int i = 0;
    for( ; i + 3 < vector_size; i += 4 ) {
        similarity += ((searched_array[i]) * (column_array[i]));
        similarity += ((searched_array[i+1]) * (column_array[i+1]));
        similarity += ((searched_array[i+2]) * (column_array[i+2]));
        similarity += ((searched_array[i+3]) * (column_array[i+3]));
    }//endfor i+4

    for( ; i < vector_size; ++i ) {
        similarity += ((searched_array[i]) * (column_array[i]));
    }

    return similarity;
}


#endif 