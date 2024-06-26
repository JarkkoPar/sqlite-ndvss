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
