# sqlite-ndvss
sqlite-ndvss is a No-Dependency Vector Similarity Search (VSS) extension for SQLite. 

It enables conversion of a string containing a list of decimal numbers to a BLOB of floats or doubles for storing the data, and the use of euclidean, dot product and cosine similarity functions to perform searches. 

sqlite-ndvss doesn't use any external dependencies to do its thing, making it portable and easy to install: just download the shared library file and copy it to where you'd like to use it. 

sqlite-ndvss was created to try out RAG with LLM's without having to install more full-fledged vector databases, and because SQLite is amazing.

You can find example SQL queries [here](examples/examples.md).

## What kind of performance can I expect?

The similarity functions are a *naïve* implementation, meaning they don't use any additional logic or structures to speed up the search. The only optimization in place is a static loop unrolling, which speeds up the for-loops in the similarity functions.

On my Asus laptop with an Intel Core i7 3610QM @ 2.3 GHz, 10 GB of RAM and an SSD from year 2012, I get following results with 200 000 random vectors with 1536 dimensions running a query with sorting based on similarity and limiting the output to 10 rows:

|Similarity function|DOUBLE/FLOAT|Runtime (s)|
|--|--|--|
|Cosine|DOUBLE|2.20|
|Cosine|FLOAT|1.83|
|Euclidean distance|DOUBLE|1.80|
|Euclidean distance|FLOAT|1.50|
|Euclidean distance squared|DOUBLE|1.80|
|Euclidean distance squared|FLOAT|1.50|
|Dot product|DOUBLE|1.79|
|Dot product|FLOAT|1.49|

Modern hardware gets of course much better results.


## Installation

Copy the binaries to the folder where you have your sqlite3 executable. 
Currently builds for Linux and Windows are available, for Mac you need to compile (see instructions below).

## Compilation

1. Download the source code and extract it to some folder.
2. Copy in the sqlite3.c, sqlite3.h and sqlite3ext.h files to the same folder (get them from https://sqlite.org/download.html). 
3. Open terminal/command prompt and change to the directory where you have the source code files.
4. Compile using the platform-specific command below:

**Windows**:`gcc -g -shared sqlite-ndvss.c -o ndvss.dll`. 

**Linux**:`gcc -g -fPIC -shared sqlite-ndvss.c -o ndvss.so`

**Mac**:`gcc -g -fPIC -dynamiclib sqlite-ndvss.c -o ndvss.dylib`

## Loading the extension

Open a database and load the extension by running `.load ./ndvss`. Change the path if needed to match where you've saved the extension files, or copy the dll/so/dylib to a directory that is included in your system path variables.
Once loaded, you can use the ndvss-functions in your SQL code.

## Functions

|Function|Parameters|Return values|Description|
|--|--|--|--|
|**ndvss_version**|none|Version number (DOUBLE)|Returns the version number of the extension.|
|**ndvss_convert_str_to_array_f**|Array to convert (TEXT), Number of dimensions (INT)|float-array (BLOB)|Converts the given text string containing an array of decimal numbers to a BLOB containing an array of floats. The textual array can be a JSON formatted array or just a space-delimited or comma-delimeted list of decimal numbers.|
|**ndvss_convert_str_to_array_d**|Array to convert (TEXT), Number of dimensions (INT)|double-array (BLOB)|Converts the given text string containing an array of decimal numbers to a BLOB containing an array of doubles. The textual array can be a JSON formatted array or just a space-delimited or comma-delimeted list of decimal numbers.|
|**ndvss_cosine_similarity_f**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the cosine similarity between the vectors of floats given as arguments. The vectors need to be of the same data type (float) and contain the same number of dimensions.|
|**ndvss_cosine_similarity_d**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the cosine similarity between the vectors of doubles given as arguments. The vectors need to be of the same data type (double) and contain the same number of dimensions.|
|**ndvss_euclidean_distance_similarity_f**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the euclidean distance similarity between the vectors of floats given as arguments. The vectors need to be of the same data type (float) and contain the same number of dimensions.|
|**ndvss_euclidean_distance_similarity_d**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the euclidean distance similarity between the vectors of doubles given as arguments. The vectors need to be of the same data type (double) and contain the same number of dimensions.|
|**ndvss_euclidean_distance_squared_similarity_f**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Does the same as *ndvss_euclidean_distance_similarity_f* but returns the squared distance (i.e. doesn't calculate the square root).|
|**ndvss_euclidean_distance_squared_similarity_d**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Does the same as *ndvss_euclidean_distance_similarity_d* but returns the squared distance (i.e. doesn't calculate the square root).|
|**ndvss_dot_product_similarity_f**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the dot product similarity between the vectors of floats given as arguments. The vectors need to be of the same data type (float) and contain the same number of dimensions.|
|**ndvss_dot_product_similarity_d**|Vector to search for (BLOB), Vector to compare to (BLOB), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the dot product similarity between the vectors of doubles given as arguments. The vectors need to be of the same data type (double) and contain the same number of dimensions.|
|**ndvss_dot_product_similarity_str**|Vector to search for (TEXT), Vector to compare to (TEXT), Number of dimensions (INT)|Similarity score (DOUBLE)|Calculates the dot product similarity between the strings containing arrays of decimal numbers given as arguments. The vectors need to be of the same data type (double) and contain the same number of dimensions. The first argument is cached and is expected to be the array that is being searched.|



## If you find a bug

Please report it with steps on how to reproduce the issue. If possible, please include some example data.
Once a fix is done, please help by verifying that the fix is working.

