#!/usr/bin/env python3
"""
Data Validation Framework Script

Validates data against schema and business rules.
Supports CSV, Parquet, and JSON file formats.
"""

import pandas as pd
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


class DataValidator:
    """Comprehensive data validation framework"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
    
    def validate_file(self, file_path, schema=None, rules=None):
        """
        Validate a data file
        
        Args:
            file_path: Path to data file
            schema: Dict with expected schema {col_name: dtype}
            rules: Dict with validation rules
        
        Returns:
            bool: True if validation passes
        """
        df = self._load_file(file_path)
        
        if df is None:
            return False
        
        results = {
            'file': str(file_path),
            'rows': len(df),
            'columns': len(df.columns),
            'checks': {}
        }
        
        # Schema validation
        if schema:
            results['checks']['schema'] = self._validate_schema(df, schema)
        
        # Business rules validation
        if rules:
            results['checks']['rules'] = self._validate_rules(df, rules)
        
        # Data quality checks
        results['checks']['quality'] = self._check_quality(df)
        
        # Print results
        self._print_results(results)
        
        return len(self.errors) == 0
    
    def _load_file(self, file_path):
        """Load file into DataFrame"""
        path = Path(file_path)
        
        if not path.exists():
            self.errors.append(f"File not found: {file_path}")
            return None
        
        try:
            if path.suffix == '.csv':
                return pd.read_csv(file_path)
            elif path.suffix == '.parquet':
                return pd.read_parquet(file_path)
            elif path.suffix == '.json':
                return pd.read_json(file_path)
            else:
                self.errors.append(f"Unsupported file format: {path.suffix}")
                return None
        except Exception as e:
            self.errors.append(f"Error loading file: {e}")
            return None
    
    def _validate_schema(self, df, schema):
        """Validate DataFrame schema"""
        results = {'passed': True, 'details': []}
        
        # Check column presence
        expected_cols = set(schema.keys())
        actual_cols = set(df.columns)
        
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        
        if missing:
            results['passed'] = False
            results['details'].append(f"Missing columns: {missing}")
            self.errors.append(f"Schema: Missing columns {missing}")
        
        if extra:
            results['details'].append(f"Extra columns: {extra}")
            self.warnings.append(f"Schema: Extra columns {extra}")
        
        # Check data types
        for col, expected_dtype in schema.items():
            if col not in df.columns:
                continue
            
            actual_dtype = str(df[col].dtype)
            
            if not self._dtype_matches(actual_dtype, expected_dtype):
                results['passed'] = False
                results['details'].append(
                    f"Column '{col}': expected {expected_dtype}, got {actual_dtype}"
                )
                self.errors.append(
                    f"Schema: Column '{col}' has wrong type"
                )
        
        return results
    
    def _dtype_matches(self, actual, expected):
        """Check if dtypes match (with flexibility)"""
        type_map = {
            'int': ['int64', 'int32', 'int16'],
            'float': ['float64', 'float32'],
            'string': ['object', 'string'],
            'datetime': ['datetime64']
        }
        
        for generic, specifics in type_map.items():
            if expected == generic and any(s in actual for s in specifics):
                return True
        
        return expected in actual
    
    def _validate_rules(self, df, rules):
        """Validate business rules"""
        results = {'passed': True, 'details': []}
        
        for rule_name, rule in rules.items():
            rule_type = rule.get('type')
            column = rule.get('column')
            
            if rule_type == 'not_null':
                nulls = df[column].isna().sum()
                if nulls > 0:
                    results['passed'] = False
                    results['details'].append(
                        f"{rule_name}: {nulls} null values in '{column}'"
                    )
                    self.errors.append(f"Rule: {rule_name} failed")
            
            elif rule_type == 'unique':
                dupes = df[column].duplicated().sum()
                if dupes > 0:
                    results['passed'] = False
                    results['details'].append(
                        f"{rule_name}: {dupes} duplicates in '{column}'"
                    )
                    self.errors.append(f"Rule: {rule_name} failed")
            
            elif rule_type == 'range':
                min_val = rule.get('min')
                max_val = rule.get('max')
                
                if min_val is not None:
                    violations = (df[column] < min_val).sum()
                    if violations > 0:
                        results['passed'] = False
                        results['details'].append(
                            f"{rule_name}: {violations} values below {min_val}"
                        )
                
                if max_val is not None:
                    violations = (df[column] > max_val).sum()
                    if violations > 0:
                        results['passed'] = False
                        results['details'].append(
                            f"{rule_name}: {violations} values above {max_val}"
                        )
            
            elif rule_type == 'enum':
                allowed = set(rule.get('values', []))
                invalid = ~df[column].isin(allowed)
                violations = invalid.sum()
                
                if violations > 0:
                    results['passed'] = False
                    results['details'].append(
                        f"{rule_name}: {violations} invalid values in '{column}'"
                    )
        
        return results
    
    def _check_quality(self, df):
        """Run data quality checks"""
        results = {}
        
        # Completeness check
        total_cells = len(df) * len(df.columns)
        null_cells = df.isna().sum().sum()
        completeness = ((total_cells - null_cells) / total_cells * 100)
        
        results['completeness'] = {
            'percentage': round(completeness, 2),
            'null_cells': int(null_cells),
            'total_cells': int(total_cells)
        }
        
        # Duplicate rows check
        duplicate_rows = df.duplicated().sum()
        results['duplicates'] = {
            'count': int(duplicate_rows),
            'percentage': round(duplicate_rows / len(df) * 100, 2)
        }
        
        if duplicate_rows > 0:
            self.warnings.append(f"Quality: {duplicate_rows} duplicate rows found")
        
        # Column-level statistics
        results['columns'] = {}
        for col in df.columns:
            col_stats = {
                'null_count': int(df[col].isna().sum()),
                'null_percentage': round(df[col].isna().sum() / len(df) * 100, 2)
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean())
                })
            
            results['columns'][col] = col_stats
        
        return results
    
    def _print_results(self, results):
        """Print validation results"""
        print("\n" + "="*70)
        print(f"DATA VALIDATION REPORT")
        print("="*70)
        print(f"File: {results['file']}")
        print(f"Rows: {results['rows']:,}")
        print(f"Columns: {results['columns']}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        for check_name, check_results in results['checks'].items():
            print(f"\n{check_name.upper()} CHECKS:")
            print("-" * 70)
            
            if isinstance(check_results, dict):
                if 'passed' in check_results:
                    status = "✓ PASSED" if check_results['passed'] else "✗ FAILED"
                    print(f"Status: {status}")
                    
                    if self.verbose or not check_results['passed']:
                        for detail in check_results.get('details', []):
                            print(f"  - {detail}")
                
                if 'completeness' in check_results:
                    comp = check_results['completeness']
                    print(f"Completeness: {comp['percentage']}%")
                    print(f"  Null cells: {comp['null_cells']:,} / {comp['total_cells']:,}")
                
                if 'duplicates' in check_results:
                    dup = check_results['duplicates']
                    print(f"Duplicates: {dup['count']} rows ({dup['percentage']}%)")
        
        print("\n" + "="*70)
        print(f"SUMMARY:")
        print(f"  Errors: {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")
        
        if self.errors:
            print(f"\nERRORS:")
            for error in self.errors:
                print(f"  ✗ {error}")
        
        if self.warnings and self.verbose:
            print(f"\nWARNINGS:")
            for warning in self.warnings:
                print(f"  ! {warning}")
        
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate data files against schema and business rules"
    )
    parser.add_argument("file", help="Path to data file (CSV, Parquet, or JSON)")
    parser.add_argument("--schema", help="Path to schema JSON file")
    parser.add_argument("--rules", help="Path to rules JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load schema if provided
    schema = None
    if args.schema:
        with open(args.schema) as f:
            schema = json.load(f)
    
    # Load rules if provided
    rules = None
    if args.rules:
        with open(args.rules) as f:
            rules = json.load(f)
    
    # Run validation
    validator = DataValidator(verbose=args.verbose)
    passed = validator.validate_file(args.file, schema, rules)
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
