# sqlite-ndvss
sqlite-ndvss is a No-Dependency Vector Similarity Search (VSS) extension for SQLite. sqlite-ndvss doesn't use any external dependencies to do its thing, making it portable and easy to install: just download the shared library file and copy it to where you'd like to use it. 

It enables conversion of a string containing a list of decimal numbers to a BLOB of floats or doubles for storing the data, and the use of euclidean, dot product and cosine similarity functions to perform searches. 

This extension has been used in real-world projects to perform similarity searches for manuals, product data, publicly available standards documents in PDF and other text formats. 

There are versions available for x86_64 and Arm Linux, Windows and Mac, and even RISC-V. Of these the Mac and RISC-V versions are currently untested.  

sqlite-ndvss was originally created to try out RAG with LLM's without having to install more full-fledged vector databases, and because SQLite is amazing.

You can find example SQL queries and Python code [here](examples/examples.md).

## What kind of performance can I expect?

The similarity functions are a *naïve* implementation, meaning they don't use any additional logic or structures to speed up the search. The only optimization in place is the use of intrinsics if any are available (on x86 SSE4.1/AVX/AVX2/AVX512F, on ARMv8 Neon, and on RISC-V RVV-extension). In the examples-folder there are instructions on clustering the data to improve performance, however this is done outside of the extension itself.

Below you can find benchmark results across different hardware. On Linux, the test are run on the specific cores. Those marked with Windows are executed on what ever core the OS has decided to run the code on.

Results for DOUBLE:
|System|CPU|DOUBLE/FLOAT|Instructions|Cos|Euc|Euc.Sq.|Dot|
|---|---|---|---|---|---|---|---|
|(Windows) Asus TUF A16|AMD Ryzen 9 7940HX|DOUBLE|AVX512f|0.3252s|0.3224s|0.3203s|0.3239s|
|(Windows) MSI Claw|Intel Core Ultra 7 155H|DOUBLE|AVX2|0.5230s|0.3697s|0.3125s|0.2936s|
|Asus NV56vz|Intel Core i7 - 3610QM|DOUBLE|AVX|0.4577s|0.4419s|0.4460s|0.4707s|
|Radxa Rock 5B|RK3588 - Cortex-A76|DOUBLE|Neon|0.6426s|0.6369s|0.6088s|0.5974s|
|Radxa Rock 4 SE|RK3399-T - Cortex-A72|DOUBLE|Neon|1.9030s|1.8493s|1.8215s|1.7814s|
|Radxa Rock 5B|RK3588 - Cortex-A55|DOUBLE|Neon|2.0746s|2.3460s|2.2850s|2.0168s|
|Radxa Rock 4 SE|RK3399-T - Cortex-A53|DOUBLE|Neon|5.0274s|5.1935s|5.1899s|4.5772s|

Results for FLOAT:
|System|CPU|DOUBLE/FLOAT|Instructions|Cos|Euc|Euc.Sq.|Dot|
|---|---|---|---|---|---|---|---|
|(Windows) Asus TUF A16|AMD Ryzen 9 7940HX|FLOAT|AVX512f|0.2313s|0.2270s|0.2206s|0.2239s|
|(Windows) MSI Claw|Intel Core Ultra 7 155H|FLOAT|AVX2|0.2316s|0.2440s|0.2341s|0.2161s|
|Asus NV56vz|Intel Core i7|FLOAT|AVX|0.3125s|0.3022s|0.3235s|0.3211s|
|Radxa Rock 5B|RK3588 - Cortex-A76|FLOAT|Neon|0.4309s|0.4190s|0.3978s|0.4017s|
|Radxa Rock 4 SE|RK3399-T - Cortex-A72|FLOAT|Neon|1.3502s|1.3070s|1.3568s|1.1668s|
|Radxa Rock 5B|RK3588 - Cortex-A55|FLOAT|Neon|1.3446s|1.4941s|1.4779s|1.3456s|
|Radxa Rock 4 SE|RK3399-T - Cortex-A53|FLOAT|Neon|3.1416s|3.2427s|3.2409s|2.8837s|

Clarification of terms:
|Term|Meaning|
|---|---|
|Cos|Cosine similarity|
|Euc|Euclidean distance similarity|
|Euc.Sq.|Squared euclidean distance similarity| 
|Dot|Dotproduct similarity|


The tests were done by running the benchmark code in the example. It creates a `:memory:` database with 200,000 vectors with 1536 dimensions. It then times the duration to run a SELECT statement that calculates the similarity to a random 1536 vector, ordering by the similarity score. The timing is done for vectors using doubles and floats.

If you run your query in a database on disk the speed of your SSD/HDD will cause differences in the results. On the afore mentioned Asus, running from the SSD causes the Cosine similarity query (double) to run in about 1.12 seconds Modern hardware gets of course much better results.

To benchmark your machine, in the examples.md there is a Python-script that you can use to do that. 


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

