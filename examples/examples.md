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

* Parse a PDF file.
* Chunk the text into smaller segments.
* Calculate vector embeddings (using sentence-transformers).
* Store the chunks and binary vectors using SQLite and ndvss.
* Perform a similarity search using ndvss.

## Prerequisites
Bash
``` 
pip install pypdf sentence-transformers
Python Script (ingest_pdf.py)
```
Python
```
import sqlite3
import struct
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# 1. Setup Models and DB
#-----------------------
# Using a standard lightweight model. Output dimension is 384.

model = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_DIM = 384

conn = sqlite3.connect("documents.db")
conn.enable_load_extension(True)

# Load the compiled extension (adjust path as necessary, e.g. "./ndvss.so" or "./ndvss.dll")
try:
    conn.load_extension("./ndvss")
except sqlite3.OperationalError as e:
    print(f"Error loading extension: {e}")
    exit(1)

cursor = conn.cursor()

# Create table. Note we store the embedding as a raw BLOB.
cursor.execute("""
    CREATE TABLE IF NOT EXISTS pdf_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        chunk_text TEXT,
        embedding BLOB
    )
""")

# 2. Helper Functions
#-----------------


def get_pdf_text(filepath):
    """Reads the text from a PDF-file in the given path."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=100, overlap=20):
    """Simple word-based sliding window chunker."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk) > 10: # filter tiny chunks
            chunks.append(chunk)
    return chunks

def float_list_to_blob(float_list):
    """
    Packs a list of Python floats into a C-compatible binary blob (float32).
    Alternatively, we could use the ndvss_convert_str_to_array_f function, but
    this is faster and requires less code.
    """
    return struct.pack(f'{len(float_list)}f', *float_list)

# 3. Ingestion Process
# --------------------
pdf_file = "manual.pdf" # Replace with your PDF
print(f"Processing {pdf_file}...")

# A. Extract
full_text = get_pdf_text(pdf_file)

# B. Chunk
text_chunks = chunk_text(full_text) # Adjust the parameters to fine-tune your search results.
print(f"Generated {len(text_chunks)} chunks.")

# C. Embed & Insert
print("Embedding and storing...")
for chunk in text_chunks:
    # Generate vector (returns numpy array)
    vector = model.encode(chunk)
    
    # Convert numpy array to binary blob (float32)
    vector_blob = float_list_to_blob(vector.tolist())
    
    cursor.execute("""
        INSERT INTO pdf_chunks (filename, chunk_text, embedding)
        VALUES (?, ?, ?)
    """, (pdf_file, chunk, vector_blob))

conn.commit()
print("Ingestion complete.")

# 4. Search Example
# -----------------
query_text = "How do I install the software?"
print(f"\nSearching for: '{query_text}'")

query_vector = model.encode(query_text).tolist()
query_blob = float_list_to_blob(query_vector)

# Use ndvss_cosine_similarity_f (float version)
results = cursor.execute(f"""
    SELECT 
        chunk_text, 
        ndvss_cosine_similarity_f(?, embedding, {VECTOR_DIM}) as similarity
    FROM pdf_chunks
    ORDER BY similarity DESC
    LIMIT 3
""", (query_blob,)).fetchall()

for i, (text, score) in enumerate(results):
    print(f"\nResult {i+1} (Score: {score:.4f}):")
    print(f"{text[:150]}...")

conn.close()

```