-- =============================================================================
-- DB Queries: Panel / Plaque-Type Art (Yoruba Doors, Ifa Boards, Benin Plaques)
-- =============================================================================
-- These queries search museum_objects for panel-type works already in the DB.
-- Two approaches are shown:
--   A. Keyword / full-text search (fast, no index needed)
--   B. Semantic (text embedding) similarity search (slower without vector index)
--
-- Run against the african-artifacts Postgres database.
-- For embedding search, the text_embedding column must be populated first via
-- the generateTextEmbeddings script.
-- =============================================================================


-- =============================================================================
-- A. KEYWORD / FULL-TEXT SEARCH
-- =============================================================================

-- ─── A1. Yoruba door panels ───────────────────────────────────────────────────

SELECT
    id,
    title,
    culture,
    object_type,
    classification,
    place->>'country' AS country,
    date->>'display'  AS date_display,
    data_source->>'museum' AS museum,
    data_source->>'url'    AS record_url,
    images->0->>'url'      AS primary_image_url
FROM museum_objects
WHERE
    to_tsvector('english',
        coalesce(title, '') || ' ' ||
        coalesce(description, '') || ' ' ||
        coalesce(physical_description, '') || ' ' ||
        coalesce(culture, '') || ' ' ||
        coalesce(classification, '') || ' ' ||
        coalesce(object_type, '') || ' ' ||
        coalesce(array_to_string(materials, ' '), '')
    ) @@ to_tsquery('english', 'Yoruba & (door | panel | carved)')
ORDER BY title
LIMIT 100;


-- ─── A2. Ifa divination boards (opon Ifa) ─────────────────────────────────────

SELECT
    id,
    title,
    culture,
    object_type,
    classification,
    place->>'country'      AS country,
    date->>'display'       AS date_display,
    data_source->>'museum' AS museum,
    data_source->>'url'    AS record_url,
    images->0->>'url'      AS primary_image_url
FROM museum_objects
WHERE
    to_tsvector('english',
        coalesce(title, '') || ' ' ||
        coalesce(description, '') || ' ' ||
        coalesce(physical_description, '') || ' ' ||
        coalesce(culture, '') || ' ' ||
        coalesce(classification, '') || ' ' ||
        coalesce(object_type, '') || ' ' ||
        coalesce(array_to_string(materials, ' '), '')
    ) @@ to_tsquery('english', '(Ifa | opon) & (divination | board | tray)')
ORDER BY title
LIMIT 100;


-- ─── A3. Benin relief plaques — individual figures ────────────────────────────

SELECT
    id,
    title,
    culture,
    object_type,
    classification,
    place->>'country'      AS country,
    date->>'display'       AS date_display,
    data_source->>'museum' AS museum,
    data_source->>'url'    AS record_url,
    images->0->>'url'      AS primary_image_url,
    physical_description
FROM museum_objects
WHERE
    to_tsvector('english',
        coalesce(title, '') || ' ' ||
        coalesce(description, '') || ' ' ||
        coalesce(physical_description, '') || ' ' ||
        coalesce(culture, '') || ' ' ||
        coalesce(classification, '') || ' ' ||
        coalesce(object_type, '')
    ) @@ to_tsquery('english', 'Benin & (plaque | relief | Reliefplatte)')
    AND (
        -- Exclude clearly non-plaque items
        object_type NOT ILIKE '%mask%'
        OR object_type IS NULL
    )
ORDER BY title
LIMIT 200;


-- ─── A4. Benin palace scenes (multi-figure compositions) ─────────────────────

SELECT
    id,
    title,
    culture,
    object_type,
    classification,
    place->>'country'      AS country,
    date->>'display'       AS date_display,
    data_source->>'museum' AS museum,
    data_source->>'url'    AS record_url,
    images->0->>'url'      AS primary_image_url,
    description
FROM museum_objects
WHERE
    to_tsvector('english',
        coalesce(title, '') || ' ' ||
        coalesce(description, '') || ' ' ||
        coalesce(physical_description, '') || ' ' ||
        coalesce(culture, '') || ' ' ||
        coalesce(classification, '') || ' ' ||
        coalesce(object_type, '')
    ) @@ to_tsquery('english', 'Benin & (scene | court | palace | warrior | ceremony | plaque | relief)')
    AND (
        -- Must have images for visual analysis
        images IS NOT NULL
        AND jsonb_array_length(images) > 0
    )
ORDER BY title
LIMIT 200;


-- ─── A5. Combined: all panel / plaque / board / door types (broad) ────────────

SELECT
    id,
    title,
    culture,
    object_type,
    classification,
    place->>'country'      AS country,
    date->>'display'       AS date_display,
    data_source->>'museum' AS museum,
    data_source->>'url'    AS record_url,
    images->0->>'url'      AS primary_image_url
FROM museum_objects
WHERE
    (
        -- Match "panel" or "plaque" or "board" or "door" in key text fields
        title ILIKE ANY(ARRAY['%panel%', '%plaque%', '%board%', '%door%', '%tray%', '%relief%'])
        OR object_type ILIKE ANY(ARRAY['%panel%', '%plaque%', '%board%', '%door%', '%tray%', '%relief%'])
        OR classification ILIKE ANY(ARRAY['%panel%', '%plaque%', '%board%', '%door%', '%tray%', '%relief%'])
        OR description ILIKE ANY(ARRAY['%panel%', '%plaque%', '%board%', '%door%', '%divination%'])
        OR physical_description ILIKE ANY(ARRAY['%panel%', '%plaque%', '%board%', '%door%', '%divination%'])
    )
    AND (
        -- Restrict to relevant cultures / regions
        culture ILIKE ANY(ARRAY['%Yoruba%', '%Benin%', '%Edo%', '%Nigerian%', '%West African%', '%African%'])
        OR place->>'country' ILIKE ANY(ARRAY['%Nigeria%', '%Benin%'])
        OR place->>'region' ILIKE ANY(ARRAY['%West Africa%', '%Nigeria%'])
        OR title ILIKE ANY(ARRAY['%Yoruba%', '%Benin%', '%Edo%', '%Nigeria%', '%Ifa%'])
    )
ORDER BY
    -- Prioritise objects with images
    (CASE WHEN jsonb_array_length(COALESCE(images, '[]'::jsonb)) > 0 THEN 0 ELSE 1 END),
    title
LIMIT 300;


-- ─── A6. SMB-specific: Benin Reliefplatte (German term used in their records) ──

SELECT
    id,
    title,
    culture,
    object_type,
    classification,
    data_source->>'museum' AS museum,
    data_source->>'url'    AS record_url,
    images->0->>'url'      AS primary_image_url,
    place->>'country'      AS country
FROM museum_objects
WHERE
    data_source->>'museum' ILIKE '%Berlin%'
    AND (
        title ILIKE '%Reliefplatte%'
        OR classification ILIKE '%Reliefplatte%'
        OR object_type ILIKE '%Reliefplatte%'
        OR description ILIKE '%Reliefplatte%'
        OR title ILIKE '%Benin%plaque%'
        OR title ILIKE '%Benin%relief%'
    )
ORDER BY title
LIMIT 100;


-- =============================================================================
-- B. SEMANTIC EMBEDDING SEARCH (cosine similarity via pgvector)
-- =============================================================================
-- NOTE: text_embedding column must be populated.
--       Without an IVFFlat index this performs a full table scan — acceptable
--       for moderate table sizes but will be slow on large datasets.
--       To create an approximate index (after enough data is loaded):
--
--         CREATE INDEX ON museum_objects USING ivfflat (text_embedding vector_cosine_ops)
--         WITH (lists = 100);
--
-- Replace the literal vector below with the actual embedding generated for the
-- semantic query string using generateTextEmbeddings or the embeddings module.
-- Placeholder shown as zeros for illustration only.

-- ─── B1. Semantic search: Yoruba carved wooden door panel ─────────────────────
-- (replace the embedding vector with one generated from your query string)

/*
SELECT
    id,
    title,
    culture,
    object_type,
    data_source->>'museum'  AS museum,
    data_source->>'url'     AS record_url,
    images->0->>'url'       AS primary_image_url,
    1 - (text_embedding <=> '<YOUR_EMBEDDING_VECTOR>'::vector) AS similarity
FROM museum_objects
WHERE
    text_embedding IS NOT NULL
    AND 1 - (text_embedding <=> '<YOUR_EMBEDDING_VECTOR>'::vector) > 0.70
ORDER BY text_embedding <=> '<YOUR_EMBEDDING_VECTOR>'::vector
LIMIT 50;
*/

-- ─── B2. Hybrid: keyword pre-filter + embedding re-rank ──────────────────────
-- More efficient than pure embedding scan; combine both signals.

/*
WITH keyword_candidates AS (
    SELECT id, text_embedding
    FROM museum_objects
    WHERE
        to_tsvector('english',
            coalesce(title, '') || ' ' ||
            coalesce(description, '') || ' ' ||
            coalesce(culture, '') || ' ' ||
            coalesce(object_type, '') || ' ' ||
            coalesce(classification, '')
        ) @@ to_tsquery('english', 'Yoruba | Benin | Ifa | plaque | panel | door | divination | relief')
)
SELECT
    mo.id,
    mo.title,
    mo.culture,
    mo.object_type,
    mo.data_source->>'museum'  AS museum,
    mo.data_source->>'url'     AS record_url,
    mo.images->0->>'url'       AS primary_image_url,
    1 - (mo.text_embedding <=> '<YOUR_EMBEDDING_VECTOR>'::vector) AS similarity
FROM museum_objects mo
JOIN keyword_candidates kc ON mo.id = kc.id
WHERE
    mo.text_embedding IS NOT NULL
    AND 1 - (mo.text_embedding <=> '<YOUR_EMBEDDING_VECTOR>'::vector) > 0.65
ORDER BY mo.text_embedding <=> '<YOUR_EMBEDDING_VECTOR>'::vector
LIMIT 50;
*/


-- =============================================================================
-- C. UTILITY / DIAGNOSTIC QUERIES
-- =============================================================================

-- ─── C1. Count panel-type objects by museum ──────────────────────────────────

SELECT
    data_source->>'museum' AS museum,
    count(*) AS panel_type_count
FROM museum_objects
WHERE
    (
        title ILIKE ANY(ARRAY['%panel%', '%plaque%', '%board%', '%door%', '%relief%'])
        OR object_type ILIKE ANY(ARRAY['%panel%', '%plaque%', '%board%', '%door%', '%relief%'])
        OR classification ILIKE ANY(ARRAY['%panel%', '%plaque%', '%board%', '%door%', '%relief%'])
    )
GROUP BY data_source->>'museum'
ORDER BY panel_type_count DESC;


-- ─── C2. Objects with images, ordered by similarity to a reference object ─────
-- Useful after finding one good example: find visually/semantically similar ones.

/*
WITH ref AS (
    SELECT id, text_embedding
    FROM museum_objects
    WHERE id = '<REFERENCE_OBJECT_UUID>'
)
SELECT
    mo.id,
    mo.title,
    mo.culture,
    mo.object_type,
    mo.data_source->>'museum' AS museum,
    mo.data_source->>'url'    AS record_url,
    mo.images->0->>'url'      AS primary_image_url,
    1 - (mo.text_embedding <=> ref.text_embedding) AS similarity
FROM museum_objects mo, ref
WHERE
    mo.id != ref.id
    AND mo.text_embedding IS NOT NULL
    AND ref.text_embedding IS NOT NULL
    AND mo.images IS NOT NULL
    AND jsonb_array_length(mo.images) > 0
ORDER BY mo.text_embedding <=> ref.text_embedding
LIMIT 20;
*/


-- ─── C3. All Yoruba-culture objects, grouped by object_type ──────────────────

SELECT
    object_type,
    count(*) AS count,
    array_agg(DISTINCT data_source->>'museum') AS museums
FROM museum_objects
WHERE
    culture ILIKE '%Yoruba%'
    OR title ILIKE '%Yoruba%'
GROUP BY object_type
ORDER BY count DESC;


-- ─── C4. All Benin-culture objects with images ────────────────────────────────

SELECT
    id,
    title,
    object_type,
    classification,
    data_source->>'museum' AS museum,
    data_source->>'url'    AS record_url,
    images->0->>'url'      AS primary_image_url
FROM museum_objects
WHERE
    (
        culture ILIKE '%Benin%'
        OR culture ILIKE '%Edo%'
        OR title ILIKE '%Benin%'
        OR place->>'country' ILIKE '%Nigeria%'
    )
    AND images IS NOT NULL
    AND jsonb_array_length(images) > 0
ORDER BY title
LIMIT 200;
