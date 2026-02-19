# WCM Setlist API Reconciliation Agent

A production-grade matching pipeline that reconciles messy live tour setlists against an internal song catalog, using deterministic code logic first and LLM-powered fuzzy matching only when needed.

## Quick Start

```bash
pip install -r requirements.txt

cp .env.example .env
# Add your OpenAI API key to .env

python main.py
```

## Output

The script produces `output/matched_setlists.csv` with 11 rows reconciling 10 setlist tracks across 2 shows (the medley "Desert Rain / Ocean Avenue" correctly expands to 2 rows). Each row includes the matched catalog title alongside the ID for human reviewability.

| Track | Catalog Match | Confidence | How |
|---|---|---|---|
| Neon Dreams | CAT-001 | Exact | Deterministic |
| Tokyo (Acoustic) | CAT-002 | High | LLM (abbreviation) |
| Desert Rain / Ocean Avenue | CAT-004 + CAT-005 | High | LLM (medley split) |
| Wonderwall | None | None | LLM (flagged as Oasis cover) |
| Shattered Glass | CAT-003 | Exact | Deterministic |
| Velocity (Extended Jam) | CAT-007 | Exact | Deterministic (suffix stripped) |
| Golden Gate | CAT-006 | Exact | Deterministic |
| Midnight In Tokyo | CAT-002 | Exact | Deterministic (case-insensitive) |
| Smsls Lk Tn Sprt | None | None | LLM (recognized as "Smells Like Teen Spirit" — not controlled) |

## Project Structure

```
wcm-setlist-agent/
  main.py              — Slim entry point (config → ingest → match → output)
  config.py            — Centralized config, logging setup, startup validation
  ingestion.py         — API fetch with fallback, catalog loading, setlist flattening
  matching.py          — Deterministic matching: normalize, exact match, validation
  llm_matching.py      — LLM fuzzy matching with caching, backoff, prompt engineering
  reconciler.py        — Two-stage pipeline orchestration
  output.py            — CSV output writer
  requirements.txt     — Pinned dependencies
  .env.example         — API key template
  data/
    catalog.csv        — Internal song catalog (15 songs, includes controlled_percentage)
    tour_setlist.json  — Tour API payload (local fallback)
  output/
    matched_setlists.csv — Reconciliation results (11 rows)
  tests/
    test_matching.py   — Unit tests for normalize, match, validate (20 tests)
    test_ingestion.py  — Catalog, flattening, validation, API fallback (16 tests)
    test_llm.py        — Mocked LLM: parsing, medley, cover, retry, cache (11 tests)
    test_pipeline.py   — End-to-end pipeline + CSV output (2 tests)
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Architecture

### Two-Stage Matching Pipeline

```
Setlist Track
     │
     ▼
┌─────────────────────────┐
│  Stage 1: Deterministic │  Free, instant, 100% reliable
│  • Case-insensitive     │  Matches ~60% of tracks
│  • Suffix stripping     │
└────────┬────────────────┘
         │ unmatched
         ▼
┌─────────────────────────┐
│  Stage 2: LLM Fuzzy     │  OpenAI gpt-4o-mini
│  • Abbreviations        │  Only for remaining ~40%
│  • Medley splitting     │  Cached per normalized name
│  • Cover detection      │  Exponential backoff on failure
└────────┬────────────────┘
         │
         ▼
    CSV Output
```

### Production Features

| Feature | Implementation |
|---|---|
| **Structured logging** | Python `logging` with timestamps, levels, module names |
| **Exponential backoff** | `2^attempt + jitter`, configurable max retries (default: 3) |
| **LLM result cache** | In-memory dict keyed on normalized track name |
| **Input validation** | Tour JSON structure + catalog CSV columns checked at ingestion |
| **Startup validation** | `validate_config()` checks files/env before pipeline runs |
| **Rate limiting** | Configurable delay between consecutive LLM calls |
| **Log rotation** | Optional file logging with automatic rotation (10 MB, 5 backups) |

## Design Decisions

### 1. API & Data Handling

The `fetch_tour_data()` function makes a GET request to the API endpoint with a 10-second timeout. If the request fails for any reason (network error, invalid JSON, timeout), it falls back to a local copy of the JSON payload. This defensive pattern means the script always produces output, even if the API is temporarily unavailable.

The nested JSON structure (tour → shows → setlists) is flattened by `flatten_setlists()` into individual track rows, each carrying the show's date and venue. This makes the data suitable for row-by-row matching and direct CSV output.

### 2. Cost & Speed Optimization

The pipeline uses a **two-stage matching architecture**:

**Stage 1 (Deterministic — free, instant):**
- Case-insensitive exact matching
- Normalization: strip parenthetical suffixes like `(Acoustic)`, `(Extended Jam)`, `(Live)` using regex, then compare

This stage matched **6 out of 10 tracks (60%)** without any LLM call. In a production scenario with hundreds of setlist entries per week, this pre-filter would significantly reduce API costs and latency.

**Stage 2 (LLM — only for unmatched tracks):**
- Only 4 tracks required LLM processing
- Medleys (tracks containing "/") are detected by code and routed directly to the LLM with explicit instructions to split and match each part
- Results are cached per normalized track name — repeat appearances across shows trigger no additional API calls

**Why this split matters:** At scale, if we process 1,000 tracks per week and 60% match deterministically, we save ~600 LLM API calls per week. At ~$0.01 per call with gpt-4o-mini, that's modest savings — but with a larger model or higher volume, the savings become significant. More importantly, deterministic matches are **instant and 100% reliable** — no risk of LLM hallucination.

### 3. Prompt Engineering / Agent Design

**Medley handling:** The code detects "/" in a track name and adds explicit instructions to the prompt: "This is a MEDLEY containing N songs. You MUST return a separate match entry for each part." This structural hint prevents the LLM from treating the medley as a single track. The JSON response is parsed to produce one CSV row per match.

**Preventing false positives (covers):** The system prompt includes explicit rules:
- "If a track is a well-known song by ANOTHER artist (a cover), it is NOT CONTROLLED by us"
- Named examples teach the pattern (e.g., recognizing abbreviations of well-known non-catalog songs)
- Confidence levels include "Review" as a middle ground — the LLM can express uncertainty rather than being forced into a binary match/no-match

**Preventing aggressive matching:** The prompt states "Only match if you are genuinely confident. Do NOT force-match vaguely similar titles." Combined with temperature=0.0 (deterministic), this minimizes creative-but-wrong matches.

**Structured JSON output:** Using `response_format={"type": "json_object"}` guarantees parseable output. The code also handles multiple possible JSON structures the LLM might return (`{"matches": [...]}`, direct arrays, etc.) as a resilience measure.

**Process orchestration:** This solution uses a **hybrid code-AI pipeline** rather than a pure agentic loop or LangChain chain. The pipeline is orchestrated in code (`reconciler.py`), which calls the LLM as a stateless tool — one prompt per unmatched track. This was chosen over a fully agentic approach (where the LLM decides what to do next) because the matching workflow is predictable: every track follows the same deterministic → LLM fallback path. An agent would add unnecessary complexity and token cost for a pipeline that doesn't require dynamic decision-making. It was chosen over a single mega-prompt because per-track calls allow caching, isolated retries, and clearer error attribution.

**Conflict & exception handling:** If a setlist track partially matches multiple catalog songs (e.g., "Neon" could match "Neon Dreams" or "Neon Nights"), the deterministic stage requires an exact normalized match — partial matches are rejected and forwarded to the LLM. The LLM is instructed to return "Review" confidence when uncertain, surfacing ambiguity for human review rather than making a wrong pick. The `validate_match()` function also catches invalid catalog IDs returned by the LLM, replacing them with "None" rather than passing through hallucinated data.

### 4. Scalability & Reliability

**At 10x volume (100 tracks per run):**
- The deterministic stage scales linearly and remains instant (O(n*m) where n=tracks, m=catalog size)
- LLM calls could be batched — instead of one call per track, group 5-10 unmatched tracks into a single prompt to reduce API round-trips
- The in-memory match cache prevents re-processing the same track across different shows in a single run

**If the data format shifted:**
- The JSON flattening logic handles arbitrary numbers of shows and setlist items — adding more shows requires no code changes
- New catalog columns (e.g., genre, release_date) could be added to the LLM prompt to improve matching accuracy
- The normalization regex could be extended with additional suffix patterns as new variations appear

**Production improvements:**
- Exponential backoff with jitter for API failures (configurable max retries)
- Structured logging for all operations with timestamps and severity levels
- Optional log rotation for long-running deployments
- Rate limiting between consecutive LLM calls to stay within API quotas
- Input validation at ingestion to catch malformed data early

### 5. Data Model Awareness

The catalog includes a `controlled_percentage` field (0–100) representing the publisher’s ownership stake in each song. This field is loaded at ingestion but is not used in the matching stage — it would be consumed by downstream royalty calculation pipelines to determine payout amounts per matched performance.

### 6. Handling False Negatives

The pipeline addresses the risk of missed matches (false negatives) through:
- **"Review" confidence level**: When the LLM is uncertain, it returns "Review" rather than "None", flagging the track for human verification rather than silently discarding it
- **Per-track prompting**: Each unmatched track is sent individually to the LLM (rather than batched), ensuring maximum context and attention per track. Per-track prompts were chosen over batching for this scope because they simplify response parsing and provide clearer error isolation — a single malformed LLM response only affects one track. Batching (5–10 tracks per prompt) is a planned optimization for higher-volume deployments where round-trip reduction outweighs parsing complexity
- **Fail-safe defaults**: If all LLM retries are exhausted, the track is returned with `catalog_id=None` and `confidence=None`, ensuring it appears in the output for manual review rather than being dropped

## Configuration

All configuration is centralized in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `MAX_RETRIES` | 3 | Max LLM retry attempts |
| `BACKOFF_BASE` | 2 | Exponential backoff base (seconds) |
| `BACKOFF_MAX` | 30 | Max backoff delay cap (seconds) |
| `LLM_MODEL` | gpt-4o-mini | OpenAI model |
| `LLM_TEMPERATURE` | 0.0 | Deterministic LLM output |
| `RATE_LIMIT_DELAY` | 0.1 | Seconds between LLM calls |
| `LOG_TO_FILE` | false | Enable file logging (env: `LOG_TO_FILE=true`) |
| `LOG_FILE_MAX_BYTES` | 10 MB | Log rotation threshold |
| `LOG_FILE_BACKUP_COUNT` | 5 | Number of rotated log files kept |
