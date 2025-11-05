#!/usr/bin/env python3
"""
Schema Comparison Tool

Compares database schemas or file schemas to detect changes.
Useful for data pipeline development and migration validation.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass
from enum import Enum


class ChangeType(Enum):
    ADDED = "ADDED"
    REMOVED = "REMOVED"
    MODIFIED = "MODIFIED"
    UNCHANGED = "UNCHANGED"


@dataclass
class SchemaChange:
    change_type: ChangeType
    entity: str  # table/file name
    field: str  # column/field name
    old_value: str = None
    new_value: str = None
    details: str = None


class SchemaComparator:
    """Compare schemas and detect changes"""
    
    def __init__(self):
        self.changes = []
    
    def compare_files(self, old_file, new_file):
        """Compare schemas of two data files"""
        old_schema = self._extract_file_schema(old_file)
        new_schema = self._extract_file_schema(new_file)
        
        return self.compare_schemas(old_schema, new_schema, 
                                   entity=f"{Path(old_file).stem}")
    
    def compare_schemas(self, old_schema: Dict, new_schema: Dict, entity: str):
        """Compare two schema dictionaries"""
        old_fields = set(old_schema.keys())
        new_fields = set(new_schema.keys())
        
        # Detect added fields
        added = new_fields - old_fields
        for field in added:
            self.changes.append(SchemaChange(
                change_type=ChangeType.ADDED,
                entity=entity,
                field=field,
                new_value=new_schema[field],
                details=f"New field added: {field} ({new_schema[field]})"
            ))
        
        # Detect removed fields
        removed = old_fields - new_fields
        for field in removed:
            self.changes.append(SchemaChange(
                change_type=ChangeType.REMOVED,
                entity=entity,
                field=field,
                old_value=old_schema[field],
                details=f"Field removed: {field} (was {old_schema[field]})"
            ))
        
        # Detect modified fields
        common = old_fields & new_fields
        for field in common:
            old_type = old_schema[field]
            new_type = new_schema[field]
            
            if old_type != new_type:
                self.changes.append(SchemaChange(
                    change_type=ChangeType.MODIFIED,
                    entity=entity,
                    field=field,
                    old_value=old_type,
                    new_value=new_type,
                    details=f"Type changed: {field} ({old_type} → {new_type})"
                ))
            else:
                self.changes.append(SchemaChange(
                    change_type=ChangeType.UNCHANGED,
                    entity=entity,
                    field=field,
                    old_value=old_type,
                    new_value=new_type
                ))
        
        return self.changes
    
    def _extract_file_schema(self, file_path):
        """Extract schema from data file"""
        path = Path(file_path)
        
        if path.suffix == '.csv':
            df = pd.read_csv(file_path, nrows=0)
            return {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        elif path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
            return {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        elif path.suffix == '.json':
            with open(file_path) as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    return {k: type(v).__name__ for k, v in sample.items()}
                elif isinstance(data, dict):
                    return {k: type(v).__name__ for k, v in data.items()}
        
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def get_summary(self):
        """Get summary of changes"""
        summary = {
            'total_changes': len(self.changes),
            'by_type': {
                'added': sum(1 for c in self.changes if c.change_type == ChangeType.ADDED),
                'removed': sum(1 for c in self.changes if c.change_type == ChangeType.REMOVED),
                'modified': sum(1 for c in self.changes if c.change_type == ChangeType.MODIFIED),
                'unchanged': sum(1 for c in self.changes if c.change_type == ChangeType.UNCHANGED)
            }
        }
        return summary
    
    def has_breaking_changes(self):
        """Check if there are breaking changes"""
        return any(
            c.change_type in [ChangeType.REMOVED, ChangeType.MODIFIED]
            for c in self.changes
        )
    
    def print_report(self, show_unchanged=False):
        """Print comparison report"""
        print("\n" + "="*70)
        print("SCHEMA COMPARISON REPORT")
        print("="*70)
        
        summary = self.get_summary()
        print(f"\nTotal Changes: {summary['total_changes']}")
        print(f"  Added:    {summary['by_type']['added']}")
        print(f"  Removed:  {summary['by_type']['removed']}")
        print(f"  Modified: {summary['by_type']['modified']}")
        print(f"  Unchanged: {summary['by_type']['unchanged']}")
        
        if self.has_breaking_changes():
            print("\n⚠️  WARNING: Breaking changes detected!")
        
        # Group by entity and change type
        by_entity = {}
        for change in self.changes:
            if change.entity not in by_entity:
                by_entity[change.entity] = []
            by_entity[change.entity].append(change)
        
        # Print changes by entity
        for entity, changes in by_entity.items():
            print(f"\n{entity}")
            print("-" * 70)
            
            for change in changes:
                if not show_unchanged and change.change_type == ChangeType.UNCHANGED:
                    continue
                
                icon = {
                    ChangeType.ADDED: "✓ +",
                    ChangeType.REMOVED: "✗ -",
                    ChangeType.MODIFIED: "⚠ ~",
                    ChangeType.UNCHANGED: "  ="
                }.get(change.change_type, "  ?")
                
                print(f"{icon} {change.details}")
        
        print("\n" + "="*70 + "\n")
    
    def export_json(self, output_path):
        """Export changes to JSON"""
        data = {
            'summary': self.get_summary(),
            'has_breaking_changes': self.has_breaking_changes(),
            'changes': [
                {
                    'type': c.change_type.value,
                    'entity': c.entity,
                    'field': c.field,
                    'old_value': c.old_value,
                    'new_value': c.new_value,
                    'details': c.details
                }
                for c in self.changes
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Report exported to: {output_path}")


def compare_database_schemas(old_schema_json, new_schema_json):
    """Compare database schemas from JSON files"""
    with open(old_schema_json) as f:
        old_db = json.load(f)
    
    with open(new_schema_json) as f:
        new_db = json.load(f)
    
    comparator = SchemaComparator()
    
    # Compare each table
    old_tables = set(old_db.keys())
    new_tables = set(new_db.keys())
    
    # Check for added/removed tables
    for table in new_tables - old_tables:
        for field, dtype in new_db[table].items():
            comparator.changes.append(SchemaChange(
                change_type=ChangeType.ADDED,
                entity=table,
                field=field,
                new_value=dtype,
                details=f"New table.field: {table}.{field} ({dtype})"
            ))
    
    for table in old_tables - new_tables:
        for field, dtype in old_db[table].items():
            comparator.changes.append(SchemaChange(
                change_type=ChangeType.REMOVED,
                entity=table,
                field=field,
                old_value=dtype,
                details=f"Removed table.field: {table}.{field} (was {dtype})"
            ))
    
    # Compare common tables
    for table in old_tables & new_tables:
        comparator.compare_schemas(
            old_db[table],
            new_db[table],
            entity=table
        )
    
    return comparator


def main():
    parser = argparse.ArgumentParser(
        description="Compare schemas to detect changes"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comparison mode')
    
    # File comparison
    file_parser = subparsers.add_parser('files', help='Compare data files')
    file_parser.add_argument('old_file', help='Old data file')
    file_parser.add_argument('new_file', help='New data file')
    
    # Database schema comparison
    db_parser = subparsers.add_parser('database', help='Compare database schemas')
    db_parser.add_argument('old_schema', help='Old schema JSON file')
    db_parser.add_argument('new_schema', help='New schema JSON file')
    
    # Common arguments
    for p in [file_parser, db_parser]:
        p.add_argument('--show-unchanged', action='store_true', 
                      help='Show unchanged fields')
        p.add_argument('--export', help='Export report to JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run comparison
    if args.command == 'files':
        comparator = SchemaComparator()
        comparator.compare_files(args.old_file, args.new_file)
    else:
        comparator = compare_database_schemas(args.old_schema, args.new_schema)
    
    # Print report
    comparator.print_report(show_unchanged=args.show_unchanged)
    
    # Export if requested
    if args.export:
        comparator.export_json(args.export)
    
    # Exit with error code if breaking changes
    import sys
    sys.exit(1 if comparator.has_breaking_changes() else 0)


if __name__ == "__main__":
    main()
