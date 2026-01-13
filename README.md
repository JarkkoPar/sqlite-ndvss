# sqlite-ndvss
sqlite-ndvss is a No-Dependency Vector Similarity Search (VSS) extension for SQLite. sqlite-ndvss doesn't use any external dependencies to do its thing, making it portable and easy to install: just download the shared library file and copy it to where you'd like to use it. 

It enables conversion of a string containing a list of decimal numbers to a BLOB of floats or doubles for storing the data, and the use of euclidean, dot product and cosine similarity functions to perform searches. 

This extension has been used in real-world projects to perform similarity searches for manuals, product data, publicly available standards documents in PDF and other text formats. 

There are versions available for x86_64 and Arm Linux, Windows and Mac, and even RISC-V. Of these the Mac and RISC-V versions are currently untested.  

sqlite-ndvss was originally created to try out RAG with LLM's without having to install more full-fledged vector databases, and because SQLite is amazing.

You can find example SQL queries [here](examples/examples.md).

## What kind of performance can I expect?

The similarity functions are a *naïve* implementation, meaning they don't use any additional logic or structures to speed up the search. The only optimization in place is the use of intrinsics if any are available (on x86 SSE4.1/AVX/AVX2/AVX512F, on ARMv8 Neon, and on RISC-V RVV-extension). In the examples-folder there are instructions on clustering the data to improve performance, however this is done outside of the extension itself.



On my 2012 Asus laptop (Intel Core i7 3610QM @ 2.3 GHz, 10 GB of RAM and an SSD, supports AVX but not AVX2) running Fedora Linux 39, I get following results for 200 000 random vectors with 1536 dimensions running a query with sorting based on similarity and limiting the output to 10 rows:


|Similarity function|DOUBLE/FLOAT|Runtime (s)|
|--|--|--|
|Cosine|DOUBLE|0.46|
|Cosine|FLOAT|0.31|
|Euclidean distance|DOUBLE|0.45|
|Euclidean distance|FLOAT|0.30|
|Euclidean distance squared|DOUBLE|0.44|
|Euclidean distance squared|FLOAT|0.30|
|Dot product|DOUBLE|0.44|
|Dot product|FLOAT|0.29|


The tests were done by loading the database into a `:memory:` database and timing the duration to run a SELECT statement that calculates the similarity for a random 1536 vector, ordering by the similarity score: `SELECT ID, ndvss_cosine_similarity_d( ndvss_convert_str_to_array_d('" + vector + "', 1536), EMBEDDING, 1536) FROM embeddings_d ORDER BY 2` for doubles and similarly with a table containing floats. If you run your query in a database on disk the speed of your SSD/HDD will cause differences in the results. On the afore mentioned Asus, running from the SSD causes the Cosine similarity query (double) to run in about 1.12 seconds. 

Modern hardware gets of course much better results.

In the examples.md there is a Python-script that you can use to benchmark your machine. 


## Installation

Copy the binaries to the folder where you have your sqlite3 executable. 

Currently builds for x86_64 & Arm Linux, Windows and Mac are available, as well as RISC-V.

## Compilation

The latest version uses zig for cross-compilation and a Makefile has been added that makes use of it. 

1. Install zig. 
2. Download the source code and extract it to some folder. 
3. Copy in the sqlite3.c, sqlite3.h and sqlite3ext.h files to the same folder (get them from https://sqlite.org/download.html). 
4. Open terminal/command prompt and change to the directory where you have the source code files.
5. Compile by running the command `make` in the folder. 


You should still be able to compile ndvss using gcc as before:

1. Download the source code and extract it to some folder.
2. Copy in the sqlite3.c, sqlite3.h and sqlite3ext.h files to the same folder (get them from https://sqlite.org/download.html). 
3. Open terminal/command prompt and change to the directory where you have the source code files.
4. Compile using the platform-specific command below:

**Windows**:`gcc -g -shared sqlite-ndvss.c -o ndvss.dll -mavx2 -mfma -Ofast -ffast-math` 

**Linux**:`gcc -g -fPIC -shared sqlite-ndvss.c -o ndvss.so -mavx2 -mfma -Ofast -ffast-math`

**Mac**:`gcc -g -fPIC -dynamiclib sqlite-ndvss.c -o ndvss.dylib -mavx2 -mfma -Ofast -ffast-math`

The ARMv8 and RISC-V libraries are compiled using zig on Linux (see Makefile).

**Note** If you are running a pre-2013 machine that does not have AVX2 support, use the following compile options:

**Windows**:`gcc -g -shared sqlite-ndvss.c -o ndvss.dll -mavx -Ofast -ffast-math`. 

**Linux**:`gcc -g -fPIC -shared sqlite-ndvss.c -o ndvss.so -mavx -Ofast -ffast-math`

**Mac**:`gcc -g -fPIC -dynamiclib sqlite-ndvss.c -o ndvss.dylib -mavx -Ofast -ffast-math`


The default compile options above use the -ffast-math option, which trades some accuracy for some speed. If you want more accuracy, simply compile without the -ffast-math option.


## Loading the extension

Open a database and load the extension by running `.load ./ndvss`. Change the path if needed to match where you've saved the extension files, or copy the dll/so/dylib to a directory that is included in your system path variables.
Once loaded, you can use the ndvss-functions in your SQL code.

## Functions

|Function|Parameters|Return values|Description|
|--|--|--|--|
|**ndvss_version**|none|Version number (DOUBLE)|Returns the version number of the extension.|
|**ndvss_instruction_set**|none|Instruction set name (STRING)|Returns which extension is in use ("basic" if none).|
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

