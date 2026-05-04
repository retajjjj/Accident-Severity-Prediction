#!/usr/bin/env python3
"""
Test runner script for Accident Severity Prediction project.

This script provides a comprehensive testing interface with:
- Multiple test execution modes
- Coverage reporting
- Performance profiling
- Test result analysis
- Automated reporting

Usage:
    python scripts/run_tests.py --mode all
    python scripts/run_tests.py --mode unit
    python scripts/run_tests.py --mode integration
    python scripts/run_tests.py --coverage
    python scripts/run_tests.py --performance
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_execution.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Comprehensive test runner for the project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "reports"
        self.coverage_dir = self.reports_dir / "coverage"
        self.test_results_dir = self.reports_dir / "test_results"
        
        # Create directories
        for directory in [self.reports_dir, self.coverage_dir, self.test_results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def run_command(self, command: List[str], description: str) -> subprocess.CompletedProcess:
        """Run a command and log the execution."""
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(command)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info(f"Return code: {result.returncode}")
            
            if result.stdout:
                logger.info(f"STDOUT:\n{result.stdout}")
            
            if result.stderr:
                logger.warning(f"STDERR:\n{result.stderr}")
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after 300s: {description}")
            raise
        except Exception as e:
            logger.error(f"Command failed: {description} - {e}")
            raise
    
    def run_unit_tests(self) -> subprocess.CompletedProcess:
        """Run unit tests only."""
        command = [
            "poetry", "run", "python", "-m", "pytest",
            "tests/",
            "-p", "no:postgresql",
            "-m", "unit",
            "--verbose",
            "--tb=short",
            "--html=reports/test_results/unit_tests.html",
            "--self-contained-html",
            "--json-report",
            "--json-report-file=reports/test_results/unit_tests.json"
        ]
        
        return self.run_command(command, "Unit Tests")
    
    def run_integration_tests(self) -> subprocess.CompletedProcess:
        """Run integration tests only."""
        command = [
            "poetry", "run", "python", "-m", "pytest",
            "tests/",
            "-p", "no:postgresql",
            "-m", "integration",
            "--verbose",
            "--tb=short",
            "--html=reports/test_results/integration_tests.html",
            "--self-contained-html",
            "--json-report",
            "--json-report-file=reports/test_results/integration_tests.json"
        ]
        
        return self.run_command(command, "Integration Tests")
    
    def run_all_tests(self) -> subprocess.CompletedProcess:
        """Run all tests with coverage."""
        command = [
            "poetry", "run", "python", "-m", "pytest",
            "tests/",
            "-p", "no:postgresql",
            "--verbose",
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:reports/coverage",
            "--cov-report=xml:reports/coverage.xml",
            "--cov-report=json:reports/coverage.json",
            "--html=reports/test_results/all_tests.html",
            "--self-contained-html",
            "--json-report",
            "--json-report-file=reports/test_results/all_tests.json",
            "--durations=10"
        ]
        
        return self.run_command(command, "All Tests with Coverage")
    
    def run_performance_tests(self) -> subprocess.CompletedProcess:
        """Run performance-focused tests."""
        command = [
            "poetry", "run", "python", "-m", "pytest",
            "tests/",
            "-p", "no:postgresql",
            "-m", "slow",
            "--verbose",
            "--tb=short",
            "--benchmark-only",
            "--benchmark-json=reports/test_results/performance.json",
            "--html=reports/test_results/performance_tests.html",
            "--self-contained-html"
        ]
        
        return self.run_command(command, "Performance Tests")
    
    def run_coverage_analysis(self) -> subprocess.CompletedProcess:
        """Run detailed coverage analysis."""
        command = [
            "poetry", "run", "python", "-m", "pytest",
            "tests/",
            "-p", "no:postgresql",
            "--cov=src",
            "--cov-report=html:reports/coverage",
            "--cov-report=xml:reports/coverage.xml",
            "--cov-report=json:reports/coverage.json",
            "--cov-report=annotate:reports/coverage/annotated",
            "--cov-report=term-missing",
            "--cov-fail-under=0",
            "--verbose"
        ]
        
        return self.run_command(command, "Coverage Analysis")
    
    def run_data_quality_tests(self) -> subprocess.CompletedProcess:
        """Run data quality validation tests."""
        command = [
            "poetry", "run", "python", "-m", "pytest",
            "tests/",
            "-p", "no:postgresql",
            "-m", "data_quality",
            "--verbose",
            "--tb=short",
            "--html=reports/test_results/data_quality_tests.html",
            "--self-contained-html"
        ]
        
        return self.run_command(command, "Data Quality Tests")
    
    def run_model_validation_tests(self) -> subprocess.CompletedProcess:
        """Run model validation tests."""
        command = [
            "poetry", "run", "python", "-m", "pytest",
            "tests/",
            "-p", "no:postgresql",
            "-m", "model_validation",
            "--verbose",
            "--tb=short",
            "--html=reports/test_results/model_validation_tests.html",
            "--self-contained-html"
        ]
        
        return self.run_command(command, "Model Validation Tests")
    
    def generate_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        logger.info("Generating comprehensive test report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_root": str(self.project_root),
            "test_results": {},
            "coverage_summary": {},
            "performance_summary": {},
            "recommendations": []
        }
        
        # Load test results if available
        test_files = {
            "unit_tests": "test_results/unit_tests.json",
            "integration_tests": "test_results/integration_tests.json",
            "all_tests": "test_results/all_tests.json",
            "performance": "test_results/performance.json"
        }
        
        for test_type, file_path in test_files.items():
            full_path = self.reports_dir / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        report["test_results"][test_type] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        # Load coverage results if available
        # Coverage JSON is saved to reports/coverage.json (not reports/coverage/coverage.json)
        coverage_file = self.reports_dir / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    report["coverage_summary"] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load coverage.json: {e}")
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)
        
        # Save comprehensive report
        report_file = self.reports_dir / "comprehensive_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive test report saved to: {report_file}")
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Coverage recommendations
        coverage = report.get("coverage_summary", {})
        # Handle different coverage JSON formats
        if "totals" in coverage:
            total_coverage = coverage.get("totals", {}).get("percent_covered", 0)
        elif "metadata" in coverage and "coverage_percent" in coverage:
            total_coverage = coverage.get("coverage_percent", 0)
        else:
            # Try to calculate from coverage data if available
            total_coverage = 0
        
        if total_coverage < 80:
            recommendations.append(
                f"Code coverage is {total_coverage:.1f}% (below 80% target). "
                "Consider adding more tests for uncovered code paths."
            )
        elif total_coverage < 90:
            recommendations.append(
                f"Code coverage is {total_coverage:.1f}%. "
                "Good, but consider aiming for 90%+ for critical modules."
            )
        
        # Test failure recommendations
        for test_type, results in report.get("test_results", {}).items():
            summary = results.get("summary", {})
            if summary.get("failed", 0) > 0:
                recommendations.append(
                    f"{test_type.replace('_', ' ').title()} has {summary.get('failed', 0)} failed tests. "
                    "Review and fix failing tests."
                )
        
        # Performance recommendations
        performance = report.get("performance_summary", {})
        if performance:
            recommendations.append(
                "Performance tests completed. Review benchmark results "
                "to identify potential performance bottlenecks."
            )
        
        if not recommendations:
            recommendations.append("All tests passed successfully! Maintain current testing standards.")
        
        return recommendations
    
    def print_summary(self, report: Dict):
        """Print test execution summary."""
        print("\n" + "="*70)
        print("TEST EXECUTION SUMMARY")
        print("="*70)
        
        print(f"Timestamp: {report['timestamp']}")
        print(f"Project: {report['project_root']}")
        
        # Test results summary
        print("\nTEST RESULTS:")
        for test_type, results in report.get("test_results", {}).items():
            summary = results.get("summary", {})
            total = summary.get("total", 0)
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            skipped = summary.get("skipped", 0)
            
            print(f"  {test_type.replace('_', ' ').title()}:")
            print(f"    Total: {total}, Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
        
        # Coverage summary
        coverage = report.get("coverage_summary", {})
        if coverage:
            totals = coverage.get("totals", {})
            print(f"\nCOVERAGE:")
            print(f"  Overall: {totals.get('percent_covered', 0):.1f}%")
            print(f"  Lines: {totals.get('covered_lines', 0)}/{totals.get('num_statements', 0)}")
            print(f"  Missing: {totals.get('missing_lines', 0)} lines")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        for i, recommendation in enumerate(report.get("recommendations", []), 1):
            print(f"  {i}. {recommendation}")
        
        print("\n" + "="*70)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for Accident Severity Prediction project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py --mode all                    # Run all tests with coverage
  python scripts/run_tests.py --mode unit                   # Run unit tests only
  python scripts/run_tests.py --mode integration            # Run integration tests only
  python scripts/run_tests.py --coverage                    # Coverage analysis only
  python scripts/run_tests.py --performance                 # Performance tests only
  python scripts/run_tests.py --mode all --report          # Run all tests and generate report
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["all", "unit", "integration", "coverage", "performance", "data_quality", "model_validation"],
        default="all",
        help="Test execution mode"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive test report"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(args.project_root)
    
    # Execute tests based on mode
    results = {}
    
    try:
        if args.mode == "all":
            results["all_tests"] = runner.run_all_tests()
        elif args.mode == "unit":
            results["unit_tests"] = runner.run_unit_tests()
        elif args.mode == "integration":
            results["integration_tests"] = runner.run_integration_tests()
        elif args.mode == "coverage":
            results["coverage"] = runner.run_coverage_analysis()
        elif args.mode == "performance":
            results["performance"] = runner.run_performance_tests()
        elif args.mode == "data_quality":
            results["data_quality"] = runner.run_data_quality_tests()
        elif args.mode == "model_validation":
            results["model_validation"] = runner.run_model_validation_tests()
        
        # Generate report if requested
        if args.report:
            report = runner.generate_test_report()
            runner.print_summary(report)
        
        # Check for failures
        failed_tests = any(result.returncode != 0 for result in results.values())
        
        if failed_tests:
            logger.error("Some tests failed. Check logs for details.")
            sys.exit(1)
        else:
            logger.info("All tests completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.error("Test execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
