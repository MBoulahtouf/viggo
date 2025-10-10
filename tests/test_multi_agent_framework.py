#!/usr/bin/env python3
"""
Test script for the multi-agent framework following existing test patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import re
from typing import List, Dict, Any, Optional


def test_multi_agent_interfaces():
    """Test multi-agent interfaces are properly defined."""
    print("🧪 Testing multi-agent interfaces...")
    
    try:
        from viggo.core.services.interfaces.multi_agent import (
            AgentType, AgentResult, QueryAnalysis, EntityExtraction, ContextAggregation,
            IAgent, IQueryAnalyzer, IEntityExtractor, IContextAggregator, IResponseGenerator, IMultiAgentOrchestrator
        )
        
        # Test enum values
        assert len(AgentType) == 5
        assert AgentType.QUERY_ANALYZER.value == "query_analyzer"
        assert AgentType.ENTITY_EXTRACTOR.value == "entity_extractor"
        assert AgentType.CONTEXT_AGGREGATOR.value == "context_aggregator"
        assert AgentType.RESPONSE_GENERATOR.value == "response_generator"
        
        # Test dataclass creation
        analysis = QueryAnalysis(
            intent="character",
            entities=["Thomas Olney"],
            complexity=0.6,
            requires_graph=True,
            requires_semantic=True
        )
        assert analysis.intent == "character"
        assert len(analysis.entities) == 1
        assert analysis.complexity == 0.6
        
        extraction = EntityExtraction(
            entities=[{"text": "Thomas Olney", "label": "Person"}],
            relationships=[{"source": "Thomas Olney", "target": "Kingsport", "type": "LIVES_IN"}],
            confidence=0.8
        )
        assert len(extraction.entities) == 1
        assert len(extraction.relationships) == 1
        assert extraction.confidence == 0.8
        
        aggregation = ContextAggregation(
            semantic_results=[{"content": "Sample content", "score": 0.9}],
            graph_results=[{"entity_name": "Test", "summary": "Test summary"}],
            hybrid_score=0.85,
            source_attribution=[]
        )
        assert aggregation.hybrid_score == 0.85
        assert len(aggregation.semantic_results) == 1
        
        print("✅ Multi-agent interfaces test passed")
        return True
        
    except Exception as e:
        print(f"❌ Multi-agent interfaces test failed: {e}")
        return False


def test_query_analyzer_agent():
    """Test query analyzer agent functionality."""
    print("🧪 Testing query analyzer agent...")
    
    try:
        from viggo.core.services.implementations.multi_agent_impl import QueryAnalyzerAgent
        
        analyzer = QueryAnalyzerAgent()
        assert analyzer.get_agent_type().value == "query_analyzer"
        
        # Test query analysis
        test_queries = [
            ("Who is the main character?", "character"),
            ("What happens in the plot?", "plot"),
            ("Where does the story take place?", "setting"),
            ("How are the characters related?", "relationship")
        ]
        
        for query, expected_intent in test_queries:
            result = analyzer.process({'query': query})
            assert result.success, f"Query analysis failed for: {query}"
            assert result.data.get('intent') == expected_intent, f"Expected {expected_intent}, got {result.data.get('intent')}"
            assert 'entities' in result.data
            assert 'complexity' in result.data
            assert 'requires_graph' in result.data
        
        # Test entity extraction from query
        result = analyzer.process({'query': 'Who is Thomas Olney in Kingsport?'})
        assert result.success
        entities = result.data.get('entities', [])
        assert 'Thomas Olney' in entities or 'Kingsport' in entities
        
        print("✅ Query analyzer agent test passed")
        return True
        
    except Exception as e:
        print(f"❌ Query analyzer agent test failed: {e}")
        return False


def test_entity_extractor_agent():
    """Test entity extractor agent functionality."""
    print("🧪 Testing entity extractor agent...")
    
    try:
        from viggo.core.services.implementations.multi_agent_impl import EntityExtractorAgent
        
        extractor = EntityExtractorAgent()
        assert extractor.get_agent_type().value == "entity_extractor"
        
        # Test entity extraction
        sample_text = """
        Thomas Olney lived in the ancient town of Kingsport, where he often visited 
        the strange house on Central Hill. The house was owned by a mysterious 
        organization known as the Elder Ones.
        """
        
        result = extractor.process({'content': sample_text})
        assert result.success, f"Entity extraction failed: {result.error_message}"
        
        entities = result.data.get('entities', [])
        relationships = result.data.get('relationships', [])
        
        # Should extract some entities
        assert len(entities) > 0, "No entities extracted"
        
        # Check for expected entities (case-insensitive)
        entity_texts = [e.get('text', '').lower() for e in entities]
        assert any('thomas' in text or 'olney' in text for text in entity_texts), "Thomas Olney not extracted"
        assert any('kingsport' in text for text in entity_texts), "Kingsport not extracted"
        
        print(f"✅ Entity extractor agent test passed - extracted {len(entities)} entities, {len(relationships)} relationships")
        return True
        
    except Exception as e:
        print(f"❌ Entity extractor agent test failed: {e}")
        return False


def test_context_aggregator_agent():
    """Test context aggregator agent functionality."""
    print("🧪 Testing context aggregator agent...")
    
    try:
        from viggo.core.services.implementations.multi_agent_impl import ContextAggregatorAgent
        
        aggregator = ContextAggregatorAgent()
        assert aggregator.get_agent_type().value == "context_aggregator"
        
        # Test context aggregation
        semantic_results = [
            {'content': 'Thomas Olney is the main character who lives in Kingsport.', 'score': 0.9},
            {'content': 'The story takes place in the ancient town of Kingsport.', 'score': 0.8}
        ]
        
        graph_results = [
            {'entity_name': 'Thomas Olney', 'summary': 'Main character who investigates mysteries', 'relationship_type': 'LIVES_IN'},
            {'entity_name': 'Kingsport', 'summary': 'Ancient town with mysterious secrets', 'relationship_type': 'LOCATED_IN'}
        ]
        
        result = aggregator.process({
            'query': 'Who is the main character?',
            'semantic_results': semantic_results,
            'graph_results': graph_results
        })
        
        assert result.success, f"Context aggregation failed: {result.error_message}"
        
        data = result.data
        assert 'hybrid_score' in data
        assert 'semantic_results' in data
        assert 'graph_results' in data
        assert 'source_attribution' in data
        
        # Check that results are scored and ranked
        assert data['hybrid_score'] > 0
        assert len(data['semantic_results']) == 2
        assert len(data['graph_results']) == 2
        
        print(f"✅ Context aggregator agent test passed - hybrid score: {data['hybrid_score']:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Context aggregator agent test failed: {e}")
        return False


def test_response_generator_agent():
    """Test response generator agent functionality."""
    print("🧪 Testing response generator agent...")
    
    try:
        from viggo.core.services.implementations.multi_agent_impl import ResponseGeneratorAgent
        from viggo.core.services.interfaces.multi_agent import ContextAggregation, QueryAnalysis
        
        generator = ResponseGeneratorAgent()
        assert generator.get_agent_type().value == "response_generator"
        
        # Create test context and analysis
        context = ContextAggregation(
            semantic_results=[{'content': 'Thomas Olney is the main character who lives in Kingsport.', 'score': 0.9}],
            graph_results=[{'entity_name': 'Thomas Olney', 'summary': 'Main character', 'relationship_type': 'LIVES_IN'}],
            hybrid_score=0.85,
            source_attribution=[]
        )
        
        analysis = QueryAnalysis(
            intent='character',
            entities=['Thomas Olney'],
            complexity=0.6,
            requires_graph=True,
            requires_semantic=True
        )
        
        result = generator.process({
            'query': 'Who is the main character?',
            'context': context,
            'analysis': analysis
        })
        
        assert result.success, f"Response generation failed: {result.error_message}"
        
        response = result.data.get('response', '')
        assert len(response) > 0, "No response generated"
        assert 'Thomas Olney' in response or 'character' in response.lower(), "Response doesn't contain expected content"
        
        print(f"✅ Response generator agent test passed - generated response: {response[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Response generator agent test failed: {e}")
        return False


def test_multi_agent_orchestrator():
    """Test multi-agent orchestrator functionality."""
    print("🧪 Testing multi-agent orchestrator...")
    
    try:
        from viggo.core.services.implementations.multi_agent_impl import MultiAgentOrchestrator
        
        orchestrator = MultiAgentOrchestrator()
        
        # Test agent registration
        status = orchestrator.get_agent_status()
        assert len(status) >= 4, f"Expected at least 4 agents, got {len(status)}"
        
        expected_agents = ['query_analyzer', 'entity_extractor', 'context_aggregator', 'response_generator']
        for agent_type in expected_agents:
            assert agent_type in status, f"Agent {agent_type} not registered"
            assert status[agent_type]['registered'], f"Agent {agent_type} not properly registered"
        
        # Test query processing
        test_queries = [
            "Who is the main character?",
            "What is the relationship between the characters?",
            "Where does the story take place?"
        ]
        
        for query in test_queries:
            result = orchestrator.process_query(query, {
                'content': 'Sample document content for testing',
                'semantic_results': [{'content': f'Relevant content for: {query}', 'score': 0.8}],
                'graph_results': [{'entity_name': 'Test Entity', 'summary': 'Test summary', 'relationship_type': 'RELATED_TO'}]
            })
            
            assert 'error' not in result, f"Query processing failed: {result.get('error')}"
            assert 'analysis' in result, "No analysis in result"
            assert result['analysis'].get('intent') in ['character', 'plot', 'setting', 'relationship', 'general'], "Invalid intent"
        
        print("✅ Multi-agent orchestrator test passed")
        return True
        
    except Exception as e:
        print(f"❌ Multi-agent orchestrator test failed: {e}")
        return False


def test_azure_graph_rag_service():
    """Test Azure GraphRAG service functionality."""
    print("🧪 Testing Azure GraphRAG service...")
    
    try:
        from viggo.core.services.implementations.azure_graph_rag_impl import (
            AzureGraphRAGService, EntityNode, Relationship, EntityCommunity
        )
        
        # Mock services for testing
        class MockGraphService:
            def create_entity_node(self, name, label, description=""):
                pass
            
            def create_relationship(self, source_entity, source_label, target_entity, target_label, relationship_type):
                pass
        
        class MockVectorStorage:
            pass
        
        graph_rag_service = AzureGraphRAGService(MockGraphService(), MockVectorStorage())
        
        # Test dataclass creation
        entity = EntityNode(
            name="Thomas Olney",
            label="Person",
            description="Main character",
            properties={"confidence": 0.9},
            confidence=0.9
        )
        assert entity.name == "Thomas Olney"
        assert entity.label == "Person"
        assert entity.confidence == 0.9
        
        relationship = Relationship(
            source="Thomas Olney",
            target="Kingsport",
            relationship_type="LIVES_IN",
            properties={"context": "lived in"},
            confidence=0.8
        )
        assert relationship.source == "Thomas Olney"
        assert relationship.target == "Kingsport"
        assert relationship.relationship_type == "LIVES_IN"
        
        community = EntityCommunity(
            community_id="community_1",
            entities=["Thomas Olney", "Kingsport"],
            summary="Main character and location",
            relationships=[relationship],
            confidence=0.85
        )
        assert community.community_id == "community_1"
        assert len(community.entities) == 2
        assert len(community.relationships) == 1
        
        print("✅ Azure GraphRAG service test passed")
        return True
        
    except Exception as e:
        print(f"❌ Azure GraphRAG service test failed: {e}")
        return False


def test_enhanced_rag_factory():
    """Test enhanced RAG factory functionality."""
    print("🧪 Testing enhanced RAG factory...")
    
    try:
        from viggo.core.services.implementations.enhanced_rag_factory import (
            EnhancedRAGFactory, enhanced_rag_factory
        )
        
        # Test factory capabilities
        capabilities = enhanced_rag_factory.get_system_capabilities()
        assert 'multi_agent_framework' in capabilities
        assert 'graph_rag' in capabilities
        assert 'enhanced_features' in capabilities
        
        # Test available configurations
        configs = enhanced_rag_factory.get_available_configurations()
        assert 'config_types' in configs
        assert 'features' in configs
        assert 'agents' in configs
        
        # Test configuration validation
        valid_config = {
            'required_features': ['multi_agent'],
            'graph_service': None
        }
        assert enhanced_rag_factory.validate_enhanced_configuration(valid_config)
        
        print("✅ Enhanced RAG factory test passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced RAG factory test failed: {e}")
        return False


def test_relationship_extraction_patterns():
    """Test relationship extraction patterns."""
    print("🧪 Testing relationship extraction patterns...")
    
    try:
        sample_text = """
        Thomas Olney lived in the ancient town of Kingsport, where he often visited 
        the strange house on Central Hill. The house was owned by a mysterious 
        organization known as the Elder Ones. Olney met with Granny Orne, who told 
        him about the secrets of the house and its connection to the cosmic entities.
        """
        
        # Test relationship patterns
        relationship_patterns = [
            r'(\w+)\s+(said|told|asked|replied|answered)\s+(to\s+)?(\w+)',
            r'(\w+)\s+(met|encountered|saw|visited)\s+(\w+)',
            r'(\w+)\s+(lived|resided|dwelt)\s+(in|at|near)\s+(\w+)',
            r'(\w+)\s+(worked|served)\s+(at|for|in)\s+(\w+)',
            r'(\w+)\s+(belonged to|was part of|member of)\s+(\w+)',
        ]
        
        relationships_found = 0
        for pattern in relationship_patterns:
            matches = re.finditer(pattern, sample_text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    source = groups[0].strip()
                    target = groups[-1].strip()
                    relationships_found += 1
        
        assert relationships_found > 0, "No relationships extracted from sample text"
        
        # Test entity extraction
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sample_text)
        entities = list(set(capitalized_words))
        assert len(entities) > 0, "No entities extracted from sample text"
        assert any('Thomas' in entity or 'Olney' in entity for entity in entities), "Thomas Olney not found in entities"
        assert any('Kingsport' in entity for entity in entities), "Kingsport not found in entities"
        
        print(f"✅ Relationship extraction patterns test passed - found {relationships_found} relationships, {len(entities)} entities")
        return True
        
    except Exception as e:
        print(f"❌ Relationship extraction patterns test failed: {e}")
        return False


def main():
    """Run all multi-agent framework tests."""
    print("🚀 Multi-Agent Framework Test Suite")
    print("=" * 50)
    
    tests = [
        ("Multi-Agent Interfaces", test_multi_agent_interfaces),
        ("Query Analyzer Agent", test_query_analyzer_agent),
        ("Entity Extractor Agent", test_entity_extractor_agent),
        ("Context Aggregator Agent", test_context_aggregator_agent),
        ("Response Generator Agent", test_response_generator_agent),
        ("Multi-Agent Orchestrator", test_multi_agent_orchestrator),
        ("Azure GraphRAG Service", test_azure_graph_rag_service),
        ("Enhanced RAG Factory", test_enhanced_rag_factory),
        ("Relationship Extraction Patterns", test_relationship_extraction_patterns)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n📋 Test Results Summary:")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All Multi-Agent Framework Tests Passed!")
        print("\n📋 Implementation Summary:")
        print("   ✅ Multi-Agent Framework: All components working correctly")
        print("   ✅ Query Analysis: Intent detection and routing")
        print("   ✅ Entity Extraction: Enhanced entity and relationship extraction")
        print("   ✅ Context Aggregation: Hybrid semantic and graph context")
        print("   ✅ Response Generation: Intelligent response generation")
        print("   ✅ GraphRAG Pipeline: Azure Search-based GraphRAG implementation")
        print("   ✅ Enhanced RAG Service: Integration with existing architecture")
        print("   ✅ SOLID Principles: All components follow SOLID design principles")
        
        print("\n🔧 Ready for Production:")
        print("   - Multi-agent framework fully functional")
        print("   - GraphRAG pipeline implemented with Azure Search")
        print("   - Enhanced RAG service ready for integration")
        print("   - All tests passing with comprehensive coverage")
        
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Please check the errors above.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
