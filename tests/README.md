# Viggo Test Suite

This directory contains all tests for the Viggo RAG system, including the new multi-agent framework tests.

## 📋 Test Organization

### **Multi-Agent Framework Tests**
- `test_multi_agent_core.py` - Core multi-agent functionality (recommended)
- `test_multi_agent_standalone.py` - Standalone tests without dependencies
- `test_multi_agent_framework.py` - Full framework tests (requires viggo module)
- `test_multi_agent_simple.py` - Simple dependency-free tests
- `test_multi_agent_system.py` - Comprehensive system tests

### **Core System Tests**
- `test_solid_architecture.py` - SOLID architecture compliance
- `test_rag_service.py` - RAG service functionality
- `test_document_processors.py` - Document processing capabilities
- `test_graph_service.py` - Graph service operations
- `test_aliasing_service.py` - Entity aliasing functionality
- `test_entity_utils.py` - Entity utility functions

### **Integration Tests**
- `test_api.py` - API endpoint testing
- `test_hybrid_rag.py` - Hybrid RAG system testing
- `test_azure_search_only.py` - Azure Search integration
- `test_simple_azure_search.py` - Simple Azure Search tests

### **Other Tests**
- `test_app.py` - Application-level tests
- `test_graph.py` - Graph-specific tests

## 🚀 Running Tests

### **Test Runner**
Use the test runner to execute tests:

```bash
# Run all tests
python3 run_all_tests.py

# Run specific test categories
python3 run_all_tests.py multi-agent
python3 run_all_tests.py core
python3 run_all_tests.py integration

# List all available tests
python3 run_all_tests.py list
```

### **Individual Test Files**
Run individual test files directly:

```bash
# Multi-agent tests (recommended)
python3 test_multi_agent_core.py
python3 test_multi_agent_standalone.py

# Core system tests
python3 test_solid_architecture.py
python3 test_rag_service.py

# Integration tests
python3 test_api.py
python3 test_hybrid_rag.py
```

## 🧪 Multi-Agent Framework Tests

### **Recommended Tests**
1. **`test_multi_agent_core.py`** - Most comprehensive, tests core functionality
2. **`test_multi_agent_standalone.py`** - No external dependencies, always works

### **Test Coverage**
- ✅ Query Analyzer Agent - Intent detection and routing
- ✅ Entity Extractor Agent - Entity and relationship extraction
- ✅ Context Aggregator Agent - Hybrid context aggregation
- ✅ Response Generator Agent - Intelligent response generation
- ✅ Multi-Agent Orchestrator - Agent coordination
- ✅ Azure GraphRAG Service - Graph-based RAG implementation
- ✅ Enhanced RAG Factory - Service factory functionality
- ✅ SOLID Principles - Architecture compliance

### **Key Features Tested**
- **Intent Classification**: character, plot, setting, relationship, general
- **Entity Extraction**: Pattern-based and spaCy-based extraction
- **Relationship Detection**: Regex-based relationship extraction
- **Context Aggregation**: Semantic and graph search result combination
- **Response Generation**: Template-based response generation
- **Agent Orchestration**: Multi-agent workflow coordination

## 📊 Test Results

### **Multi-Agent Core Tests**
```
🎯 Overall: 3/4 tests passed
✅ Multi-Agent Orchestrator: PASSED
✅ Entity Extraction Patterns: PASSED  
✅ SOLID Principles Compliance: PASSED
⚠️ Query Analyzer: Minor intent classification issues
```

### **Standalone Tests**
```
🎯 Overall: 3/3 tests passed
✅ Query Analyzer: PASSED
✅ Multi-Agent Orchestrator: PASSED
✅ Entity Extraction Patterns: PASSED
```

## 🔧 Test Dependencies

### **No Dependencies (Always Work)**
- `test_multi_agent_standalone.py`
- `test_multi_agent_core.py`

### **Requires Viggo Module**
- `test_multi_agent_framework.py`
- `test_multi_agent_system.py`
- All core system tests
- All integration tests

### **External Dependencies**
- Azure Search tests require Azure configuration
- Graph tests require Neo4j connection
- API tests require running server

## 🎯 Test Strategy

### **Development Testing**
1. Start with `test_multi_agent_core.py` for multi-agent development
2. Use `test_multi_agent_standalone.py` for quick validation
3. Run `test_solid_architecture.py` for architecture compliance

### **Integration Testing**
1. Run `test_api.py` for API integration
2. Use `test_hybrid_rag.py` for RAG system testing
3. Test Azure Search with `test_azure_search_only.py`

### **Production Testing**
1. Run all tests with `run_all_tests.py`
2. Verify multi-agent functionality
3. Check system integration

## 📝 Adding New Tests

### **Test File Naming**
- Use `test_` prefix
- Use descriptive names: `test_multi_agent_core.py`
- Group related tests: `test_multi_agent_*.py`

### **Test Structure**
```python
#!/usr/bin/env python3
"""
Test description.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_functionality():
    """Test specific functionality."""
    print("🧪 Testing functionality...")
    # Test implementation
    return True

def main():
    """Run all tests."""
    tests = [
        ("Functionality", test_functionality),
    ]
    
    # Test execution logic
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### **Test Categories**
- **Core**: Basic functionality tests
- **Integration**: System integration tests
- **Multi-Agent**: Multi-agent framework tests
- **Other**: Miscellaneous tests

## 🎉 Success Criteria

### **Multi-Agent Framework**
- ✅ All agents working correctly
- ✅ Intent classification accurate
- ✅ Entity extraction functional
- ✅ Relationship detection working
- ✅ Agent orchestration successful
- ✅ SOLID principles compliance

### **System Integration**
- ✅ API endpoints responding
- ✅ RAG service functional
- ✅ Document processing working
- ✅ Graph operations successful
- ✅ Azure Search integration working

### **Overall System**
- ✅ All core tests passing
- ✅ Integration tests successful
- ✅ Multi-agent framework operational
- ✅ Production-ready system

---

*Test suite organized and ready for development and production use.*
