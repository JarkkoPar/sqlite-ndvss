# sqlite-ndvss
sqlite-ndvss is a No-Dependency Vector Similarity Search extension for SQLite. 

It enables conversion of a string containing a list of decimal numbers to a BLOB of floats or doubles for storing the data, and the use of euclidean, dot product and cosine similarity functions to perform searches.

## Installation

Copy the binaries to the folder where you have your sqlite3 executable. 

## Usage

Open a database and load the extension by running `.load ./ndvss`. Change the path if needed to match where you've saved the extension files.
You can now use the ndvss-functions in your SQL code.

## Functions

|Function|Parameters|Return values|Description|
|--|--|--|--|
|**ndvss_version**|none|Version number (TEXT)|Returns the version number of the extension.|
|**ndvss_convert_str_to_array_f**|Array to convert (TEXT), Number of dimensions (INT)|float-array (BLOB)|Converts the given text string containing an array of decimal numbers to a BLOB containing an array of floats.|
|**ndvss_convert_str_to_array_d**|Array to convert (TEXT), Number of dimensions (INT)|double-array (BLOB)|Converts the given text string containing an array of decimal numbers to a BLOB containing an array of doubles.|
|**ndvss_cosine_similarity_f**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the cosine similarity between the vectors of floats given as arguments. The vectors need to be of the same data type (float) and contain the same number of dimensions.|
|**ndvss_cosine_similarity_d**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the cosine similarity between the vectors of doubles given as arguments. The vectors need to be of the same data type (double) and contain the same number of dimensions.|
|**ndvss_euclidean_distance_similarity_f**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the euclidean distance similarity between the vectors of floats given as arguments. The vectors need to be of the same data type (float) and contain the same number of dimensions.|
|**ndvss_euclidean_distance_similarity_d**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the euclidean distance similarity between the vectors of doubles given as arguments. The vectors need to be of the same data type (double) and contain the same number of dimensions.|
|**ndvss_euclidean_distance_squared_similarity_f**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Does the same as *ndvss_euclidean_distance_similarity_f* but returns the squared distance (i.e. doesn't calculate the square root).|
|**ndvss_euclidean_distance_squared_similarity_d**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Does the same as *ndvss_euclidean_distance_similarity_d* but returns the squared distance (i.e. doesn't calculate the square root).|
|**ndvss_dot_product_similarity_f**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the dot product similarity between the vectors of floats given as arguments. The vectors need to be of the same data type (float) and contain the same number of dimensions.|
|**ndvss_dot_product_similarity_d**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the dot product similarity between the vectors of doubles given as arguments. The vectors need to be of the same data type (double) and contain the same number of dimensions.|
|**ndvss_dot_product_similarity_str**|Vector to search for (TEXT), Vector to compare to TEXT), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the dot product similarity between the strings containing arrays of decimal numbers given as arguments. The vectors need to be of the same data type (double) and contain the same number of dimensions. The first argument is cached and is expected to be the array that is being searched.|

