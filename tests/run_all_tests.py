#!/usr/bin/env python3
"""
Test runner for all Viggo tests.
"""

import sys
import os
import subprocess
import glob
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_test_file(test_file):
    """Run a single test file and return the result."""
    print(f"\n🧪 Running {test_file}...")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✅ {test_file} PASSED")
            return True
        else:
            print(f"❌ {test_file} FAILED")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_file} TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {test_file} CRASHED: {e}")
        return False


def run_multi_agent_tests():
    """Run multi-agent specific tests."""
    print("🤖 Running Multi-Agent Framework Tests")
    print("=" * 60)
    
    multi_agent_tests = [
        "test_multi_agent_standalone.py",
        "test_multi_agent_framework.py"
    ]
    
    results = []
    for test_file in multi_agent_tests:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        if os.path.exists(test_path):
            result = run_test_file(test_path)
            results.append((test_file, result))
        else:
            print(f"⚠️ {test_file} not found, skipping...")
    
    return results


def run_core_tests():
    """Run core system tests."""
    print("\n🏗️ Running Core System Tests")
    print("=" * 60)
    
    core_tests = [
        "test_solid_architecture.py",
        "test_rag_service.py",
        "test_document_processors.py",
        "test_graph_service.py",
        "test_aliasing_service.py",
        "test_entity_utils.py"
    ]
    
    results = []
    for test_file in core_tests:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        if os.path.exists(test_path):
            result = run_test_file(test_path)
            results.append((test_file, result))
        else:
            print(f"⚠️ {test_file} not found, skipping...")
    
    return results


def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 Running Integration Tests")
    print("=" * 60)
    
    integration_tests = [
        "test_api.py",
        "test_hybrid_rag.py",
        "test_azure_search_only.py",
        "test_simple_azure_search.py"
    ]
    
    results = []
    for test_file in integration_tests:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        if os.path.exists(test_path):
            result = run_test_file(test_path)
            results.append((test_file, result))
        else:
            print(f"⚠️ {test_file} not found, skipping...")
    
    return results


def run_all_tests():
    """Run all available tests."""
    print("🚀 Viggo Test Suite Runner")
    print("=" * 60)
    
    all_results = []
    
    # Run multi-agent tests
    multi_agent_results = run_multi_agent_tests()
    all_results.extend(multi_agent_results)
    
    # Run core tests
    core_results = run_core_tests()
    all_results.extend(core_results)
    
    # Run integration tests
    integration_results = run_integration_tests()
    all_results.extend(integration_results)
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in all_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n🎯 Overall Results:")
    print(f"   Total Tests: {len(all_results)}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {(passed/len(all_results)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All Tests Passed! System is Ready!")
    else:
        print(f"\n⚠️ {failed} tests failed. Please check the errors above.")
    
    return failed == 0


def list_available_tests():
    """List all available test files."""
    print("📋 Available Test Files")
    print("=" * 60)
    
    test_dir = os.path.dirname(__file__)
    test_files = glob.glob(os.path.join(test_dir, "test_*.py"))
    
    categories = {
        "Multi-Agent Framework": [],
        "Core System": [],
        "Integration": [],
        "Other": []
    }
    
    for test_file in sorted(test_files):
        filename = os.path.basename(test_file)
        
        if "multi_agent" in filename:
            categories["Multi-Agent Framework"].append(filename)
        elif filename in ["test_solid_architecture.py", "test_rag_service.py", 
                         "test_document_processors.py", "test_graph_service.py",
                         "test_aliasing_service.py", "test_entity_utils.py"]:
            categories["Core System"].append(filename)
        elif filename in ["test_api.py", "test_hybrid_rag.py", 
                         "test_azure_search_only.py", "test_simple_azure_search.py"]:
            categories["Integration"].append(filename)
        else:
            categories["Other"].append(filename)
    
    for category, files in categories.items():
        if files:
            print(f"\n{category}:")
            for file in files:
                print(f"   - {file}")
    
    print(f"\nTotal: {len(test_files)} test files")


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "list":
            list_available_tests()
            return True
        elif command == "multi-agent":
            results = run_multi_agent_tests()
            return all(result for _, result in results)
        elif command == "core":
            results = run_core_tests()
            return all(result for _, result in results)
        elif command == "integration":
            results = run_integration_tests()
            return all(result for _, result in results)
        else:
            print(f"Unknown command: {command}")
            print("Available commands: list, multi-agent, core, integration")
            return False
    else:
        return run_all_tests()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
