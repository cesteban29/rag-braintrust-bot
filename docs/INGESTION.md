# Braintrust Documentation Ingestion

## Overview

This system ingests the complete Braintrust documentation from https://www.braintrust.dev/docs/llms.txt into a Pinecone vector database for RAG (Retrieval-Augmented Generation) queries.

## What Changed

**Before**: Static ingestion of 28 local MDX files from `data/documents/`
**Now**: Dynamic ingestion of the complete Braintrust documentation (30,000+ words) from the web

## Architecture

```
https://www.braintrust.dev/docs/llms.txt
            ↓
    [GitHub Action: update-docs.yml]
            ↓
    [ingest_web.py script]
            ↓
    [VoyageAI Embeddings]
            ↓
    [Pinecone Vector DB]
            ↓
    [RAG Retrieval Tool]
```

## Key Files

- `src/rag_braintrust_bot/ingestion/ingest_web.py` - Main ingestion script
- `.github/workflows/update-docs.yml` - Scheduled GitHub Action (runs Monday & Thursday at 2 AM UTC)
- `src/rag_braintrust_bot/tools/retrieval_tool.py` - Updated retrieval tool with new metadata

## Running Manually

### First Time Setup (Clear and Ingest)
```bash
python -m src.rag_braintrust_bot.ingestion.ingest_web --clear
```

### Regular Updates (Incremental)
```bash
python -m src.rag_braintrust_bot.ingestion.ingest_web
```

## Environment Variables Required

```bash
PINECONE_API_KEY=your_key_here
VOYAGEAI_API_KEY=your_key_here
INDEX_NAME=your_index_name
EMBEDDING_MODEL=voyage-3  # optional, defaults to voyage-3
UPLOAD_BATCH_SIZE=50      # optional, defaults to 50
```

## Cost Analysis

**Free Tier Usage:**
- Storage: ~800KB (0.04% of 2GB limit)
- Write Units: ~120/month (0.006% of 2M limit)
- Read Units: ~9,000/month (0.9% of 1M limit)

You're using less than 1% of Pinecone's free tier limits!

## Document Structure

The ingested content includes:
- **Changelog entries** (weekly updates from 2023-2025)
- **SDK documentation** (Python, TypeScript)
- **API references**
- **Feature guides** (tracing, evals, prompts)
- **Code examples**

Each chunk is tagged with:
- `source`: Always "braintrust_docs"
- `url`: Source URL
- `section_type`: Type of content (changelog, api, sdk, etc.)
- `date`: For changelog entries
- `title`: Section heading
- `content`: Actual documentation text

## Monitoring

- Check ingestion logs: `ingest_web.log`
- GitHub Actions dashboard for scheduled runs
- Pinecone dashboard for index statistics

## Cleanup

To remove the old local MDX files (now redundant):
```bash
rm -rf data/documents/
```

The old `ingest.py` script can also be removed once you confirm the new system works.