---
name: data-pipeline-engineering
description: Comprehensive data pipeline engineering toolkit for building production-grade ETL/ELT systems. Use when designing data pipelines, implementing data transformations, setting up data quality checks, working with data orchestration (Airflow, Prefect, Dagster), building batch or streaming pipelines, integrating data sources (APIs, databases, files), implementing data validation, or creating data processing workflows. Includes patterns for AWS/GCP/Azure data services, database optimization, and pipeline monitoring.
---

# Data Pipeline Engineering

This skill provides comprehensive guidance for building production-grade data pipelines with focus on reliability, scalability, and maintainability.

## Core Pipeline Patterns

### ETL vs ELT Decision Framework

**Use ETL (Extract-Transform-Load) when:**
- Transform logic is complex and resource-intensive
- Data needs significant cleaning before storage
- Target system has limited compute capacity
- Working with legacy systems or strict schemas
- Data volume is moderate and transformations are stable

**Use ELT (Extract-Load-Transform) when:**
- Target system has powerful compute (e.g., Snowflake, BigQuery, Redshift)
- Raw data should be preserved for auditing
- Schema flexibility is needed
- Multiple downstream consumers need different transformations
- Working with large-scale data (modern approach)

### Pipeline Architecture Layers

```
1. INGESTION LAYER
   ├── API connectors (REST, GraphQL, webhooks)
   ├── Database replication (CDC, batch)
   ├── File ingestion (S3, GCS, SFTP)
   └── Stream ingestion (Kafka, Kinesis, Pub/Sub)

2. PROCESSING LAYER
   ├── Data validation and quality checks
   ├── Schema evolution handling
   ├── Transformations (cleaning, enrichment, aggregation)
   └── Deduplication and conflict resolution

3. STORAGE LAYER
   ├── Raw/bronze: Unprocessed data
   ├── Staging/silver: Cleaned and validated
   └── Curated/gold: Business-ready aggregates

4. ORCHESTRATION LAYER
   ├── Workflow scheduling (Airflow, Prefect, Dagster)
   ├── Dependency management
   ├── Error handling and retries
   └── Monitoring and alerting
```

## Implementation Guidelines

### 1. Data Validation Framework

Implement multi-level validation:

**Schema Validation:**
- Column presence and data types
- Nullable constraints
- Enum/categorical value ranges
- Primary key uniqueness

**Business Logic Validation:**
- Value ranges (e.g., age between 0-120)
- Referential integrity across tables
- Time-based constraints (no future dates)
- Cross-field validation (e.g., end_date > start_date)

**Data Quality Metrics:**
- Completeness: % of non-null required fields
- Accuracy: match against known truth sets
- Consistency: cross-system reconciliation
- Timeliness: freshness vs SLA requirements

**Implementation pattern:**
```python
# Use great_expectations for validation
def validate_data(df, expectations):
    results = df.validate(expectations)
    if not results['success']:
        log_failures(results['results'])
        raise DataQualityException(results)
    return df
```

### 2. Incremental Processing Patterns

**Timestamp-based (preferred when available):**
- Track max timestamp processed
- Query: `WHERE updated_at > last_processed_timestamp`
- Handle timezone conversions carefully
- Account for late-arriving data

**ID-based (when no timestamps):**
- Track max ID processed
- Query: `WHERE id > last_processed_id`
- Ensure IDs are monotonically increasing
- Handle ID gaps from deletions

**Change Data Capture (CDC):**
- Debezium for database change streams
- Captures inserts, updates, deletes
- Lower latency than batch pulls
- Preserves full change history

**Full snapshot with comparison:**
- Load complete dataset
- Compare with previous snapshot
- Identify changes via hashing or diff
- Expensive but handles all edge cases

### 3. Error Handling Strategy

**Transient vs Permanent Errors:**

Transient (retry with exponential backoff):
- Network timeouts
- Rate limiting (429 errors)
- Temporary service unavailability (503)
- Database connection failures

Permanent (alert and skip):
- Authentication failures (401)
- Not found (404)
- Bad request/malformed data (400)
- Schema mismatches

**Implementation pattern:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def extract_with_retry(source):
    # Transient error handling
    pass

def process_with_dead_letter_queue(record):
    try:
        transform_and_load(record)
    except PermanentError as e:
        send_to_dlq(record, error=str(e))
        log_error(e)
```

### 4. Idempotency Design

Ensure pipelines can be safely re-run:

**Upsert patterns:**
- Use MERGE statements for databases
- Unique constraints prevent duplicates
- Update existing records, insert new ones

**Partitioning strategies:**
- Date-based partitions (common for time-series)
- Drop and rewrite partition on re-run
- Enables efficient backfills

**Deduplication:**
- Use row_number() with PARTITION BY unique keys
- Keep latest record based on timestamp
- Hash-based deduplication for exact duplicates

### 5. Performance Optimization

**Parallelization:**
- Partition data for parallel processing
- Use multiprocessing/threading appropriately
- Balance between parallelism and resource contention

**Batch sizing:**
- API calls: respect rate limits, batch requests
- Database writes: bulk inserts (1000-5000 rows)
- File processing: chunk large files

**Database optimization:**
- Create indexes on filter/join columns
- Use EXPLAIN to analyze query plans
- Partition large tables by date
- Consider columnar formats (Parquet) for analytics

**Memory management:**
- Stream data instead of loading fully into memory
- Use iterators/generators in Python
- Clear DataFrames after processing
- Monitor memory usage and set limits

## Orchestration Best Practices

### Airflow Patterns

**DAG Design:**
- One DAG per data source or business domain
- Use sensors for external dependencies
- Set appropriate SLAs and timeouts
- Implement proper task dependencies

**Task granularity:**
- Each task should be independently retryable
- Avoid monolithic tasks that do everything
- Use XCom sparingly (only for metadata)
- Store large data in external systems, not XCom

**Configuration management:**
- Use Variables for environment config
- Store secrets in Airflow Connections or external vault
- Parameterize with macros ({{ ds }}, {{ execution_date }})
- Avoid hardcoding values

### Monitoring and Alerting

**Key metrics to track:**
- Pipeline run duration and success rate
- Data volume processed (row counts, bytes)
- Data quality check pass/fail rates
- Lag/staleness (time since last successful run)
- Resource utilization (CPU, memory, I/O)

**Alert triggers:**
- Pipeline failure or timeout
- Data quality threshold breaches
- Volume anomalies (±50% from baseline)
- SLA violations
- Prolonged stale data

**Logging best practices:**
- Structured logging (JSON format)
- Include correlation IDs for tracing
- Log at appropriate levels (ERROR for failures, INFO for milestones)
- Include data volumes and timestamps
- Avoid logging PII or sensitive data

## Technology-Specific Patterns

### Cloud Data Services

**AWS:**
- S3: use Parquet/ORC with partitioning
- Glue: serverless ETL with PySpark
- Lambda: lightweight transformations (<15min)
- Kinesis: real-time streaming
- DMS: database migration and replication

**GCP:**
- BigQuery: use partitioned tables, clustering
- Dataflow: Apache Beam pipelines
- Cloud Functions: event-driven processing
- Pub/Sub: messaging and streaming
- Dataproc: managed Spark clusters

**Azure:**
- Data Factory: orchestration and ETL
- Databricks: Spark-based processing
- Synapse: integrated analytics
- Event Hubs: streaming ingestion
- Blob Storage: data lake foundation

### Database Patterns

**PostgreSQL/MySQL:**
- Use COPY/LOAD DATA for bulk inserts
- Implement connection pooling
- Handle long-running transactions carefully
- Use read replicas for analytics queries

**Data Warehouses (Snowflake, Redshift, BigQuery):**
- Leverage native COPY/LOAD commands
- Use appropriate distribution/clustering keys
- Partition by date for time-series data
- Separate compute from storage concerns
- Use materialized views for complex aggregations

### Streaming vs Batch

**Choose streaming when:**
- Real-time decision-making required (<1 min latency)
- Event-driven architecture
- Continuous data generation (IoT, logs)
- Immediate anomaly detection needed

**Choose batch when:**
- Latency requirements are relaxed (hourly/daily)
- Processing requires full dataset context
- Cost optimization is priority
- Complex transformations that benefit from parallelism

## Testing and Quality Assurance

**Unit testing:**
- Test transformation functions with sample data
- Mock external dependencies (APIs, databases)
- Validate business logic independently

**Integration testing:**
- Test full pipeline with test datasets
- Verify data flow through all stages
- Check error handling with bad data

**Data testing:**
- Schema validation tests
- Statistical property tests (distributions, outliers)
- Consistency checks across systems
- Historical comparison tests

## Migration and Deployment

**Blue-green deployment:**
- Run new pipeline version alongside old
- Compare outputs for validation period
- Switch traffic once verified
- Maintain rollback capability

**Backfilling strategies:**
- Partition historical data into manageable chunks
- Process oldest data first
- Monitor resource usage
- Implement checkpointing for long backfills

**Schema evolution:**
- Add new columns as nullable initially
- Deprecate before removing columns
- Version schemas explicitly
- Document breaking changes

## Quick Reference

See `references/common_patterns.md` for detailed code examples of:
- API pagination handling
- Database connection pooling
- Parquet file operations
- Error handling patterns
- Testing templates

See `scripts/` for executable utilities:
- Data validation framework
- Pipeline performance profiler
- Schema comparison tool
- Data quality checker
