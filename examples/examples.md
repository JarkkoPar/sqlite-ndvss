# Example queries

## Load the extension

```SQL
.load ./ndvss
```

## Show the version number

Easiest way to check if the extension has been loaded.
```SQL
SELECT ndvss_version();
```

## Create a table for the embeddings

```SQL
CREATE TABLE my_embeddings(
    ID INTEGER,    -- ID to join the embedding to the actual content
    EMBEDDING BLOB -- The embeddings as an array of doubles
);
```

## Insert rows to the table for the embeddings

```SQL
-- As a space-delimeted list of decimals.
INSERT INTO my_embeddings( ID, EMBEDDING ) 
VALUES( 1, ndvss_convert_str_to_array_d('0.001 0.4 0.005 0.9', 4));

-- As a JSON array.
INSERT INTO my_embeddings( ID, EMBEDDING ) 
VALUES( 2, ndvss_convert_str_to_array_d('[0.9, 0.1, 0.0, 0.881]', 4));

-- As a comma-separated array.
INSERT INTO my_embeddings( ID, EMBEDDING ) 
VALUES( 3, ndvss_convert_str_to_array_d('0.4522, 0.0075, 0.8, 0.2234', 4));

-- As an array in square brackets.
INSERT INTO my_embeddings( ID, EMBEDDING ) 
VALUES( 4, ndvss_convert_str_to_array_d('[0.372 0.0096 0.1097 0.0041]', 4));

-- And the following are just some extra data to make the SQL queries below more interesting.
INSERT INTO my_embeddings( ID, EMBEDDING ) 
VALUES( 5, ndvss_convert_str_to_array_d('-0.556 0.104 -0.299 0.0', 4));

INSERT INTO my_embeddings( ID, EMBEDDING ) 
VALUES( 6, ndvss_convert_str_to_array_d('0.009 -0.654 0.0 -1.0', 4));

INSERT INTO my_embeddings( ID, EMBEDDING ) 
VALUES( 7, ndvss_convert_str_to_array_d('-0.443 -0.142 -0.984 -0.332', 4));

INSERT INTO my_embeddings( ID, EMBEDDING ) 
VALUES( 8, ndvss_convert_str_to_array_d('-0.316 0.999 -1.0 -1.0', 4));

INSERT INTO my_embeddings( ID, EMBEDDING ) 
VALUES( 9, ndvss_convert_str_to_array_d('1.0 0.0 -1.0 0.0', 4));

INSERT INTO my_embeddings( ID, EMBEDDING ) 
VALUES( 10, ndvss_convert_str_to_array_d('-0.0005 -0.0023 0.9872 -0.0421', 4));

```

## Query the data

```SQL
SELECT ID, -- ID to connect to other data
       ndvss_cosine_similarity_d(   -- Selected similarity function
            ndvss_convert_str_to_array_d('0.372 0.0096 0.1097 0.0041', 4), -- What to search for
            EMBEDDING, -- Column to compare to
            4 ) -- Number of dimensions
FROM my_embeddings;
```

## Query the data with sort and limit to 2 rows

```SQL
SELECT ID, -- ID to connect to other data
       ndvss_euclidean_distance_similarity_d(   -- Selected similarity function
            ndvss_convert_str_to_array_d('0.9, 0.1, 0.0, 0.881', 4), -- What to search for
            EMBEDDING, -- Column to compare to
            4 ) -- Number of dimensions
FROM my_embeddings
ORDER BY 2
LIMIT 2;
```


# Python integration 

This example demonstrates how to:

* Parse all PDF files in a given folder.
* Chunk the texts into smaller segments. 
* Calculate vector embeddings (using google-genai for embedding).
* Store the chunks and binary vectors using SQLite and ndvss.
* Perform a similarity search using ndvss.

While the example uses google-genai, you can easily change it to use some other API and AI-model for embedding (such as Ollama and some Open Source model).

## Prerequisites

``` Bash
pip install pypdf google-genai
```

## Code

```Python
import sqlite3
import struct
import os
from google import genai
from google.genai import types
from pypdf import PdfReader

# 1. Configuration
# ----------------
# Set your folder path here
PDF_DIRECTORY = "./some/folder" 

# Set your API Key here
client = genai.Client(api_key="WRITE_YOUR_API_KEY_HERE")

EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_DIM = 768

# 2. Database Setup
# -----------------
conn = sqlite3.connect("documents_genai.db")
conn.enable_load_extension(True)

try:
    conn.load_extension("./ndvss") # Adjust path if needed (e.g. ./x86/ndvss)
except sqlite3.OperationalError as e:
    print(f"Error loading extension: {e}")
    exit(1)

cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS pdf_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        chunk_text TEXT,
        embedding BLOB
    )
""")

# 3. Helper Functions
# -------------------

# This function extracts text from all the pages in a pdf-document.
def get_pdf_text(filepath):
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            extract = page.extract_text()
            if extract:
                text += extract + "\n"
        return text
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return ""

# This function does the text chunking. Adjust chunk size and overlap
# to tweak your search results. 
def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk) > 10: 
            chunks.append(chunk)
    return chunks

# This function converts a list of floats into a BLOB that can be used with ndvss.
def float_list_to_blob(float_list):
    return struct.pack(f'{len(float_list)}f', *float_list)


# This function uses an LLM to create the embeddings.
def get_embeddings(text_chunks, task_type="RETRIEVAL_DOCUMENT"):
    if not text_chunks: return []
    
    all_embeddings = []
    BATCH_SIZE = 100  # Google GenAI limit is 100 items per request
    
    try:
        # Loop through chunks in batches of 100
        for i in range(0, len(text_chunks), BATCH_SIZE):
            batch = text_chunks[i : i + BATCH_SIZE]
            
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            
            # Extract values from this batch and add to main list
            batch_embeddings = [e.values for e in response.embeddings]
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    except Exception as e:
        print(f"API Error: {e}")
        return []

# 4. Ingestion Process
# ------------------------------

# This function reads all PDF-files in the given directory, extracts the text on
# all the pages, gets embeddings and stores them to the database.
def ingest_pdf(): 
    if not os.path.isdir(PDF_DIRECTORY):
        print(f"Error: The directory '{PDF_DIRECTORY}' does not exist.")
        exit(1)

    files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]

    if not files:
        print(f"No PDF files found in '{PDF_DIRECTORY}'.")
    else:
        print(f"Found {len(files)} PDF files in '{PDF_DIRECTORY}'. Starting ingestion...")

        for filename in files:

            # Check if file is already indexed to avoid duplicates
            exists = cursor.execute("SELECT 1 FROM pdf_chunks WHERE filename = ? LIMIT 1", (filename,)).fetchone()
            if exists:
                print(f"Skipping {filename} (already indexed).")
                continue

            full_path = os.path.join(PDF_DIRECTORY, filename)
            print(f"Processing {filename}...")
            
            full_text = get_pdf_text(full_path)
            if not full_text: continue

            text_chunks = chunk_text(full_text)
            print(f"  - Generated {len(text_chunks)} chunks.")

            if text_chunks:
                vectors = get_embeddings(text_chunks)
                
                if len(vectors) != len(text_chunks):
                    print("  ! Mismatch in vector count. Skipping.")
                    continue

                data_to_insert = []
                for chunk, vector in zip(text_chunks, vectors):
                    vector_blob = float_list_to_blob(vector)
                    data_to_insert.append((filename, chunk, vector_blob))
                
                cursor.executemany("""
                    INSERT INTO pdf_chunks (filename, chunk_text, embedding)
                    VALUES (?, ?, ?)
                """, data_to_insert)
                
                conn.commit()

        print("Ingestion complete.")

# Comment out this if you only want to test querying after ingestion.
ingest_pdf()

# 5. Search Example
# -----------------
query_text = "Write your query text here"
print(f"\nSearching for: '{query_text}'")

try:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query_text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_blob = float_list_to_blob(response.embeddings[0].values)

    results = cursor.execute(f"""
        SELECT filename, chunk_text, 
               ndvss_cosine_similarity_f(?, embedding, {VECTOR_DIM}) as similarity
        FROM pdf_chunks
        ORDER BY similarity DESC LIMIT 3
    """, (query_blob,)).fetchall()

    for i, (fname, text, score) in enumerate(results):
        print(f"\nResult {i+1} (Score: {score:.4f}) from [{fname}]:")
        print(f"{text}")

except Exception as e:
    print(f"Search failed: {e}")

conn.close()

```

# Benchmark

Here's a simple benchmark that you can run to test ndvss on your machine. It creates 200,000 vectors and does a similarity search using a random vector. It benchmarks doubles and floats separately, to enable testing on hardware with less memory (such as Radxa Rock 4SE with 4GB RAM I have run this benchmark on).

```Python
import sqlite3
import time
import random
import os
import platform
import array
import gc

# --- Configuration ---
EXT_PATH = "./ndvss.so" 
if platform.system() == "Windows":
    EXT_PATH = "C:\\YOUR\\PATH\\HERE\\ndvss.dll"

ROW_COUNT  = 200000 
DIMENSIONS = 1536

def get_random_blob(dim, typecode):
    """Generates a raw binary blob for a C-array."""
    vec = [random.random() for _ in range(dim)]
    return array.array(typecode, vec).tobytes()

def run_single_query_test(cursor, label, func_name, query_blob, dim, order_by="DESC"):
    """Helper to run a specific similarity test."""
    print(f"   Testing: {label} ({func_name})")
    start_time = time.monotonic()
    
    sql = f"""
        SELECT ID, {func_name}(?, EMBEDDING, {dim}) as score
        FROM embeddings 
        ORDER BY score {order_by}
        LIMIT 10
    """
    
    cursor.execute(sql, (query_blob,))
    results = cursor.fetchone()
    duration = time.monotonic() - start_time
    
    if results:
        print(f"     -> Time: {duration:.4f}s | Top ID: {results[0]} | Score: {results[1]:.6f}")
    else:
        print(f"     -> Time: {duration:.4f}s | No results found.")

def run_suite(typecode, type_name):
    """
    Runs the full lifecycle for a specific data type (d/f).
    Opens a FRESH connection and closes it at the end to guarantee RAM release.
    """
    print(f"\n==========================================")
    print(f"[PHASE] {type_name} ({'8 bytes' if typecode=='d' else '4 bytes'})")
    print(f"==========================================")
    
    # 1. Setup Fresh Connection
    try:
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        conn.load_extension(EXT_PATH)
    except Exception as e:
        print(f"!! Failed to load extension: {e}")
        return

    cursor = conn.cursor()
    
    # 2. Insert Data
    cursor.execute("CREATE TABLE embeddings (ID INTEGER PRIMARY KEY, EMBEDDING BLOB)")
    print("   -> Generating and inserting data...")
    start_gen = time.time()
    
    def data_generator():
        for i in range(ROW_COUNT):
            yield (i, get_random_blob(DIMENSIONS, typecode))

    cursor.executemany("INSERT INTO embeddings (ID, EMBEDDING) VALUES (?, ?)", data_generator())
    conn.commit()
    print(f"   -> Inserted {ROW_COUNT} rows in {time.time() - start_gen:.2f}s")

    # 3. Prepare Query Vector
    query_blob = get_random_blob(DIMENSIONS, typecode)

    # 4. Run All Tests
    # Adjust suffixes (_d or _f) based on typecode
    suffix = "_d" if typecode == 'd' else "_f"
    
    # A. Cosine (Similarity -> DESC)
    run_single_query_test(cursor, "Cosine", f"ndvss_cosine_similarity{suffix}", query_blob, DIMENSIONS, "DESC")
    
    # B. Euclidean Distance (Distance -> ASC)
    run_single_query_test(cursor, "Euclidean", f"ndvss_euclidean_distance_similarity{suffix}", query_blob, DIMENSIONS, "ASC")
    
    # C. Euclidean Squared (Distance -> ASC)
    run_single_query_test(cursor, "Euclidean Sq", f"ndvss_euclidean_distance_similarity_squared{suffix}", query_blob, DIMENSIONS, "ASC")
    
    # D. Dot Product (Similarity -> DESC)
    run_single_query_test(cursor, "Dot Product", f"ndvss_dot_product_similarity{suffix}", query_blob, DIMENSIONS, "DESC")

    # 5. HARD RESET
    print(f"   -> Closing connection to free RAM...")
    conn.close()
    gc.collect() # Force Python to release objects
    print(f"   -> RAM cleared.")

def run_benchmark():
    print(f"--- NDVSS Benchmark ---")
    print(f"System: {platform.machine()}")
    print(f"Rows: {ROW_COUNT} | Dims: {DIMENSIONS}")

    # Check the instruction set that is in use.
    try:
        conn_check = sqlite3.connect(":memory:")
        conn_check.enable_load_extension(True)
        conn_check.load_extension(EXT_PATH)
        cursor_check = conn_check.cursor()
        
        cursor_check.execute("SELECT ndvss_instruction_set()")
        iset = cursor_check.fetchone()[0]
        print(f"Active Instruction Set: {iset}")
        
        conn_check.close()
    except Exception as e:
        print(f"!! Warning: Could not determine instruction set: {e}")

    # Run Doubles
    run_suite('d', "DOUBLES")
    
    # Run Floats
    run_suite('f', "FLOATS")

    print("\nBenchmark Complete.")

if __name__ == "__main__":
    run_benchmark()



```

