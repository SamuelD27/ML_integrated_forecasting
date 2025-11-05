#!/usr/bin/env python3
"""
Pipeline Performance Profiler

Analyzes and profiles data pipeline performance, identifying bottlenecks
and providing optimization recommendations.
"""

import argparse
import time
import json
import psutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List


class PipelineProfiler:
    """Profile data pipeline performance"""
    
    def __init__(self, name="pipeline"):
        self.name = name
        self.stages = []
        self.current_stage = None
        self.start_time = None
    
    @contextmanager
    def profile_stage(self, stage_name, record_count=None):
        """Context manager to profile a pipeline stage"""
        stage_data = {
            'name': stage_name,
            'record_count': record_count,
            'start_time': time.time(),
            'start_memory_mb': self._get_memory_usage(),
            'start_cpu_percent': psutil.cpu_percent(interval=0.1)
        }
        
        self.current_stage = stage_data
        
        try:
            yield self
        finally:
            stage_data['end_time'] = time.time()
            stage_data['duration_seconds'] = stage_data['end_time'] - stage_data['start_time']
            stage_data['end_memory_mb'] = self._get_memory_usage()
            stage_data['memory_delta_mb'] = (
                stage_data['end_memory_mb'] - stage_data['start_memory_mb']
            )
            stage_data['end_cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            # Calculate throughput if records provided
            if record_count:
                stage_data['records_per_second'] = (
                    record_count / stage_data['duration_seconds']
                )
            
            self.stages.append(stage_data)
            self.current_stage = None
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_summary(self):
        """Get performance summary"""
        if not self.stages:
            return {}
        
        total_duration = sum(s['duration_seconds'] for s in self.stages)
        total_records = sum(s.get('record_count', 0) for s in self.stages 
                          if s.get('record_count'))
        
        summary = {
            'pipeline_name': self.name,
            'total_stages': len(self.stages),
            'total_duration_seconds': round(total_duration, 2),
            'total_records_processed': total_records,
            'average_memory_mb': round(
                sum(s['end_memory_mb'] for s in self.stages) / len(self.stages), 2
            ),
            'peak_memory_mb': round(
                max(s['end_memory_mb'] for s in self.stages), 2
            ),
            'stages': []
        }
        
        # Add stage details
        for stage in self.stages:
            stage_summary = {
                'name': stage['name'],
                'duration_seconds': round(stage['duration_seconds'], 2),
                'duration_percentage': round(
                    stage['duration_seconds'] / total_duration * 100, 1
                ),
                'memory_delta_mb': round(stage['memory_delta_mb'], 2),
                'avg_cpu_percent': round(
                    (stage['start_cpu_percent'] + stage['end_cpu_percent']) / 2, 1
                )
            }
            
            if stage.get('record_count'):
                stage_summary['records_per_second'] = round(
                    stage['records_per_second'], 0
                )
            
            summary['stages'].append(stage_summary)
        
        return summary
    
    def identify_bottlenecks(self, threshold_percent=20):
        """Identify performance bottlenecks"""
        if not self.stages:
            return []
        
        total_duration = sum(s['duration_seconds'] for s in self.stages)
        bottlenecks = []
        
        for stage in self.stages:
            percentage = (stage['duration_seconds'] / total_duration) * 100
            
            if percentage >= threshold_percent:
                bottleneck = {
                    'stage': stage['name'],
                    'duration_seconds': round(stage['duration_seconds'], 2),
                    'percentage': round(percentage, 1),
                    'recommendations': []
                }
                
                # Memory-related recommendations
                if stage['memory_delta_mb'] > 500:
                    bottleneck['recommendations'].append(
                        "High memory usage - consider streaming data or processing in chunks"
                    )
                
                # Throughput recommendations
                if stage.get('records_per_second', float('inf')) < 1000:
                    bottleneck['recommendations'].append(
                        "Low throughput - consider parallel processing or batch optimization"
                    )
                
                # CPU recommendations
                avg_cpu = (stage['start_cpu_percent'] + stage['end_cpu_percent']) / 2
                if avg_cpu < 30:
                    bottleneck['recommendations'].append(
                        "Low CPU utilization - pipeline may be I/O bound, consider async operations"
                    )
                elif avg_cpu > 80:
                    bottleneck['recommendations'].append(
                        "High CPU utilization - consider optimizing computations or adding more cores"
                    )
                
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def print_report(self, detailed=True):
        """Print performance report"""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print(f"PIPELINE PERFORMANCE REPORT: {self.name}")
        print("="*70)
        print(f"Total Duration: {summary['total_duration_seconds']}s")
        print(f"Total Stages: {summary['total_stages']}")
        print(f"Records Processed: {summary.get('total_records_processed', 'N/A')}")
        print(f"Average Memory: {summary['average_memory_mb']} MB")
        print(f"Peak Memory: {summary['peak_memory_mb']} MB")
        
        print("\n" + "-"*70)
        print("STAGE BREAKDOWN:")
        print("-"*70)
        
        for stage in summary['stages']:
            print(f"\n{stage['name']}")
            print(f"  Duration:    {stage['duration_seconds']}s ({stage['duration_percentage']}%)")
            print(f"  Memory Δ:    {stage['memory_delta_mb']:+.2f} MB")
            print(f"  CPU Usage:   {stage['avg_cpu_percent']}%")
            
            if 'records_per_second' in stage:
                print(f"  Throughput:  {stage['records_per_second']:.0f} records/sec")
        
        # Bottleneck analysis
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            print("\n" + "-"*70)
            print("⚠️  BOTTLENECKS DETECTED:")
            print("-"*70)
            
            for i, bottleneck in enumerate(bottlenecks, 1):
                print(f"\n{i}. {bottleneck['stage']}")
                print(f"   Takes {bottleneck['percentage']}% of total time")
                
                if bottleneck['recommendations']:
                    print("   Recommendations:")
                    for rec in bottleneck['recommendations']:
                        print(f"   • {rec}")
        
        print("\n" + "="*70 + "\n")
    
    def export_json(self, output_path):
        """Export profile data to JSON"""
        data = {
            'summary': self.get_summary(),
            'bottlenecks': self.identify_bottlenecks(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Profile exported to: {output_path}")


# Example usage function
def example_pipeline():
    """Example pipeline to demonstrate profiler usage"""
    profiler = PipelineProfiler("example_etl")
    
    # Stage 1: Extract
    with profiler.profile_stage("extract", record_count=10000):
        # Simulate data extraction
        time.sleep(0.5)
        data = pd.DataFrame({
            'id': range(10000),
            'value': range(10000)
        })
    
    # Stage 2: Transform
    with profiler.profile_stage("transform", record_count=10000):
        # Simulate transformation
        time.sleep(1.0)
        data['value'] = data['value'] * 2
    
    # Stage 3: Load
    with profiler.profile_stage("load", record_count=10000):
        # Simulate loading
        time.sleep(0.3)
        # data.to_parquet('output.parquet')
    
    return profiler


def main():
    parser = argparse.ArgumentParser(
        description="Profile data pipeline performance"
    )
    parser.add_argument('--example', action='store_true',
                       help='Run example pipeline')
    parser.add_argument('--export', help='Export results to JSON file')
    
    args = parser.parse_args()
    
    if args.example:
        print("Running example pipeline...")
        profiler = example_pipeline()
        profiler.print_report()
        
        if args.export:
            profiler.export_json(args.export)
    else:
        print("Pipeline Profiler")
        print("\nUsage in your code:")
        print("""
from pipeline_profiler import PipelineProfiler

profiler = PipelineProfiler("my_pipeline")

# Profile each stage
with profiler.profile_stage("extract", record_count=1000):
    data = extract_data()

with profiler.profile_stage("transform", record_count=1000):
    data = transform_data(data)

with profiler.profile_stage("load", record_count=1000):
    load_data(data)

# Print report
profiler.print_report()
profiler.export_json("profile.json")
        """)
        print("\nRun with --example to see a demonstration")


if __name__ == "__main__":
    main()
