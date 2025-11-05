# Common Data Pipeline Patterns

This reference provides production-ready code examples for common data pipeline operations.

## API Pagination Patterns

### Offset-based Pagination

```python
def fetch_all_pages_offset(base_url, params=None, limit=100):
    """Fetch all pages using offset-based pagination"""
    all_data = []
    offset = 0
    
    while True:
        params = params or {}
        params.update({'limit': limit, 'offset': offset})
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        records = data.get('results', [])
        
        if not records:
            break
            
        all_data.extend(records)
        offset += limit
        
        if len(records) < limit:
            break
    
    return all_data
```

### Cursor-based Pagination

```python
def fetch_all_pages_cursor(base_url, params=None):
    """Fetch all pages using cursor-based pagination"""
    all_data = []
    next_cursor = None
    
    while True:
        params = params or {}
        if next_cursor:
            params['cursor'] = next_cursor
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        records = data.get('data', [])
        
        if not records:
            break
            
        all_data.extend(records)
        
        next_cursor = data.get('next_cursor')
        if not next_cursor:
            break
    
    return all_data
```

### Rate-limited API Fetching

```python
import time
from functools import wraps

def rate_limit(calls_per_second=10):
    """Decorator to rate limit API calls"""
    interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = interval - elapsed
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

@rate_limit(calls_per_second=5)
def fetch_data(url):
    return requests.get(url).json()
```

## Database Connection Patterns

### Connection Pooling with PostgreSQL

```python
from psycopg2 import pool
from contextlib import contextmanager

class DatabasePool:
    def __init__(self, minconn=1, maxconn=10, **kwargs):
        self.pool = pool.ThreadedConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            **kwargs
        )
    
    @contextmanager
    def get_connection(self):
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)
    
    def execute_query(self, query, params=None):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

# Usage
db_pool = DatabasePool(
    minconn=2,
    maxconn=10,
    host='localhost',
    database='mydb',
    user='user',
    password='password'
)

results = db_pool.execute_query(
    "SELECT * FROM users WHERE created_at > %s",
    (last_timestamp,)
)
```

### Bulk Insert with PostgreSQL

```python
from psycopg2.extras import execute_batch

def bulk_insert(conn, table, records, batch_size=1000):
    """Efficient bulk insert using execute_batch"""
    if not records:
        return
    
    columns = records[0].keys()
    query = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(['%s'] * len(columns))})
        ON CONFLICT DO NOTHING
    """
    
    values = [tuple(r.values()) for r in records]
    
    with conn.cursor() as cur:
        execute_batch(cur, query, values, page_size=batch_size)
    
    conn.commit()
```

### Incremental Processing with Timestamp

```python
import json
from datetime import datetime

class IncrementalProcessor:
    def __init__(self, state_file='pipeline_state.json'):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self):
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)
    
    def get_last_timestamp(self, source):
        return self.state.get(source, {}).get('last_timestamp')
    
    def update_timestamp(self, source, timestamp):
        if source not in self.state:
            self.state[source] = {}
        self.state[source]['last_timestamp'] = timestamp
        self._save_state()
    
    def process_incremental(self, source, fetch_func):
        last_ts = self.get_last_timestamp(source)
        records = fetch_func(last_ts)
        
        if records:
            max_ts = max(r['updated_at'] for r in records)
            self.update_timestamp(source, max_ts)
        
        return records

# Usage
processor = IncrementalProcessor()
new_records = processor.process_incremental(
    'users_table',
    lambda last_ts: fetch_users_since(last_ts)
)
```

## Parquet File Operations

### Write to Parquet with Partitioning

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

def write_partitioned_parquet(df, base_path, partition_cols=['year', 'month']):
    """Write DataFrame to partitioned Parquet files"""
    # Add partition columns if date column exists
    if 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
    
    # Convert to PyArrow Table
    table = pa.Table.from_pandas(df)
    
    # Write with partitioning
    pq.write_to_dataset(
        table,
        root_path=base_path,
        partition_cols=partition_cols,
        existing_data_behavior='overwrite_or_ignore',
        compression='snappy'
    )

# Usage
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'user_id': range(100),
    'value': range(100)
})

write_partitioned_parquet(df, 's3://bucket/data/')
```

### Read Parquet with Filters

```python
def read_parquet_filtered(path, date_filter=None):
    """Read Parquet files with partition pruning"""
    filters = []
    
    if date_filter:
        year, month = date_filter.year, date_filter.month
        filters = [
            ('year', '=', year),
            ('month', '=', month)
        ]
    
    df = pd.read_parquet(
        path,
        filters=filters,
        engine='pyarrow'
    )
    
    return df

# Usage
df = read_parquet_filtered(
    's3://bucket/data/',
    date_filter=datetime(2024, 1, 1)
)
```

## Data Validation Patterns

### Schema Validation with Pydantic

```python
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import date

class UserRecord(BaseModel):
    user_id: int = Field(gt=0)
    email: str
    age: Optional[int] = Field(None, ge=0, le=120)
    signup_date: date
    
    @validator('email')
    def email_valid(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v
    
    @validator('signup_date')
    def no_future_dates(cls, v):
        if v > date.today():
            raise ValueError('Date cannot be in future')
        return v

def validate_records(records):
    """Validate a list of records"""
    valid_records = []
    errors = []
    
    for idx, record in enumerate(records):
        try:
            validated = UserRecord(**record)
            valid_records.append(validated.dict())
        except Exception as e:
            errors.append({
                'index': idx,
                'record': record,
                'error': str(e)
            })
    
    return valid_records, errors
```

### Data Quality Checks

```python
def run_quality_checks(df):
    """Run comprehensive data quality checks"""
    checks = {
        'total_rows': len(df),
        'null_checks': {},
        'duplicate_checks': {},
        'value_range_checks': {}
    }
    
    # Null checks
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        checks['null_checks'][col] = {
            'null_count': int(null_count),
            'null_percentage': round(null_pct, 2)
        }
    
    # Duplicate checks
    duplicate_count = df.duplicated().sum()
    checks['duplicate_checks'] = {
        'duplicate_rows': int(duplicate_count),
        'duplicate_percentage': round((duplicate_count / len(df)) * 100, 2)
    }
    
    # Value range checks for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        checks['value_range_checks'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'outliers': int((df[col] < df[col].quantile(0.01)) | 
                          (df[col] > df[col].quantile(0.99))).sum()
        }
    
    return checks
```

## Error Handling Patterns

### Retry with Exponential Backoff

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import requests

@retry(
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def fetch_with_retry(url):
    """Fetch URL with automatic retry on transient errors"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()
```

### Dead Letter Queue Pattern

```python
import json
from datetime import datetime

class DeadLetterQueue:
    def __init__(self, dlq_path='dlq.jsonl'):
        self.dlq_path = dlq_path
    
    def add(self, record, error, source):
        """Add failed record to DLQ"""
        dlq_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'source': source,
            'error': str(error),
            'record': record
        }
        
        with open(self.dlq_path, 'a') as f:
            f.write(json.dumps(dlq_entry) + '\n')
    
    def process_with_dlq(self, records, process_func, source):
        """Process records with DLQ for failures"""
        successful = []
        
        for record in records:
            try:
                result = process_func(record)
                successful.append(result)
            except Exception as e:
                self.add(record, e, source)
        
        return successful

# Usage
dlq = DeadLetterQueue()
processed = dlq.process_with_dlq(
    records,
    lambda r: transform_and_load(r),
    source='api_users'
)
```

## Testing Patterns

### Unit Test Template

```python
import pytest
import pandas as pd

def transform_data(df):
    """Example transformation function"""
    df['total'] = df['price'] * df['quantity']
    return df[df['total'] > 0]

class TestTransformData:
    def test_basic_transformation(self):
        input_df = pd.DataFrame({
            'price': [10, 20, 30],
            'quantity': [1, 2, 3]
        })
        
        result = transform_data(input_df)
        
        assert 'total' in result.columns
        assert len(result) == 3
        assert result['total'].tolist() == [10, 40, 90]
    
    def test_filters_zero_values(self):
        input_df = pd.DataFrame({
            'price': [10, 0, 30],
            'quantity': [0, 2, 3]
        })
        
        result = transform_data(input_df)
        
        assert len(result) == 1
        assert result['total'].iloc[0] == 90
    
    def test_empty_dataframe(self):
        input_df = pd.DataFrame({'price': [], 'quantity': []})
        
        result = transform_data(input_df)
        
        assert len(result) == 0
        assert 'total' in result.columns
```

### Integration Test Template

```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_database():
    """Mock database for testing"""
    db = Mock()
    db.execute_query.return_value = [
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200}
    ]
    return db

def test_pipeline_integration(mock_database):
    """Test full pipeline flow"""
    # Extract
    data = extract_from_database(mock_database)
    assert len(data) == 2
    
    # Transform
    transformed = transform_data(data)
    assert all('processed' in r for r in transformed)
    
    # Load (mock)
    with patch('pipeline.load_to_warehouse') as mock_load:
        load_to_warehouse(transformed)
        mock_load.assert_called_once()
```

## Performance Monitoring

### Pipeline Metrics Collection

```python
import time
import logging
from functools import wraps

class PipelineMetrics:
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def track(self, metric_name):
        """Decorator to track execution time and success rate"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self._record_success(metric_name, duration)
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    self._record_failure(metric_name, duration, e)
                    raise
            
            return wrapper
        return decorator
    
    def _record_success(self, name, duration):
        if name not in self.metrics:
            self.metrics[name] = {'successes': 0, 'failures': 0, 'durations': []}
        
        self.metrics[name]['successes'] += 1
        self.metrics[name]['durations'].append(duration)
        
        self.logger.info(f"{name} completed in {duration:.2f}s")
    
    def _record_failure(self, name, duration, error):
        if name not in self.metrics:
            self.metrics[name] = {'successes': 0, 'failures': 0, 'durations': []}
        
        self.metrics[name]['failures'] += 1
        
        self.logger.error(f"{name} failed after {duration:.2f}s: {error}")
    
    def get_summary(self):
        """Get metrics summary"""
        summary = {}
        for name, data in self.metrics.items():
            total = data['successes'] + data['failures']
            avg_duration = sum(data['durations']) / len(data['durations']) if data['durations'] else 0
            
            summary[name] = {
                'total_runs': total,
                'success_rate': (data['successes'] / total * 100) if total > 0 else 0,
                'avg_duration': round(avg_duration, 2)
            }
        
        return summary

# Usage
metrics = PipelineMetrics()

@metrics.track('extract_users')
def extract_users():
    # extraction logic
    pass

@metrics.track('transform_users')
def transform_users(data):
    # transformation logic
    pass
```
