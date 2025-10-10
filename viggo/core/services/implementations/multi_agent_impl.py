"""
Multi-agent framework implementations following SOLID principles.
"""

import re
import time
from collections import defaultdict
from typing import Any

import spacy

from viggo.core.services.interfaces.multi_agent import (
    AgentResult,
    AgentType,
    ContextAggregation,
    EntityExtraction,
    IAgent,
    IContextAggregator,
    IEntityExtractor,
    IMultiAgentOrchestrator,
    IQueryAnalyzer,
    IResponseGenerator,
    QueryAnalysis,
)


class QueryAnalyzerAgent(IQueryAnalyzer):
    """Agent for analyzing queries and determining intent."""

    def __init__(self):
        self.agent_type = AgentType.QUERY_ANALYZER
        self.intent_patterns = {
            'character': [
                r'\b(who|character|protagonist|main character|person|people)\b',
                r'\b(name|named|called)\b',
                r'\b(describe|tell me about)\b.*\b(character|person)\b'
            ],
            'plot': [
                r'\b(what|happens|plot|story|about|narrative)\b',
                r'\b(describe|tell me about)\b.*\b(story|plot|happens)\b',
                r'\b(summary|summarize)\b'
            ],
            'setting': [
                r'\b(where|location|place|setting|scene)\b',
                r'\b(takes place|located|happens in)\b',
                r'\b(describe|tell me about)\b.*\b(place|location|setting)\b'
            ],
            'relationship': [
                r'\b(relationship|related|connection|between)\b',
                r'\b(how.*related|connected|associated)\b',
                r'\b(interaction|interact|meet)\b'
            ],
            'temporal': [
                r'\b(when|time|period|era|date|chronology)\b',
                r'\b(before|after|during|sequence)\b'
            ]
        }

        self.complexity_indicators = {
            'simple': [r'\b(who|what|where|when)\b'],
            'complex': [
                r'\b(compare|contrast|analyze|explain|why|how)\b',
                r'\b(relationship|connection|interaction)\b',
                r'\b(multiple|several|various|different)\b'
            ]
        }

    def get_agent_type(self) -> AgentType:
        return self.agent_type

    def can_handle(self, input_data: dict[str, Any]) -> bool:
        return 'query' in input_data and isinstance(input_data['query'], str)

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Process query analysis."""
        start_time = time.time()

        try:
            query = input_data['query']
            analysis = self.analyze_query(query)

            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                data={
                    'intent': analysis.intent,
                    'entities': analysis.entities,
                    'complexity': analysis.complexity,
                    'requires_graph': analysis.requires_graph,
                    'requires_semantic': analysis.requires_semantic
                },
                confidence=0.9,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return AgentResult(
                agent_type=self.agent_type,
                success=False,
                data={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine intent and requirements."""
        query_lower = query.lower()

        # Determine intent
        intent = 'general'
        intent_scores: dict[str, int] = defaultdict(int)

        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent_scores[intent_type] += 1

        if intent_scores:
            intent = max(intent_scores, key=lambda k: intent_scores[k])

        # Extract entities (simple approach)
        entities = self._extract_entities_from_query(query)

        # Determine complexity
        complexity = self._calculate_complexity(query_lower)

        # Determine requirements
        requires_graph = intent in ['relationship', 'character'] or complexity > 0.6
        requires_semantic = True  # Always need semantic search

        return QueryAnalysis(
            intent=intent,
            entities=entities,
            complexity=complexity,
            requires_graph=requires_graph,
            requires_semantic=requires_semantic
        )

    def _extract_entities_from_query(self, query: str) -> list[str]:
        """Extract potential entities from query."""
        # Simple entity extraction - look for capitalized words and phrases
        entities = []

        # Find capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized_words)

        # Find quoted strings
        quoted_strings = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_strings)

        # Remove duplicates and filter
        entities = list(set(entities))
        entities = [e for e in entities if len(e) > 2]  # Filter short entities

        return entities

    def _calculate_complexity(self, query_lower: str) -> float:
        """Calculate query complexity score (0-1)."""
        complexity = 0.0

        # Check for complex indicators
        for pattern in self.complexity_indicators['complex']:
            if re.search(pattern, query_lower):
                complexity += 0.3

        # Check for simple indicators
        for pattern in self.complexity_indicators['simple']:
            if re.search(pattern, query_lower):
                complexity += 0.1

        # Length factor
        if len(query_lower.split()) > 10:
            complexity += 0.2

        # Multiple question words
        question_words = ['who', 'what', 'where', 'when', 'why', 'how']
        question_count = sum(1 for word in question_words if word in query_lower)
        if question_count > 1:
            complexity += 0.2

        return min(complexity, 1.0)


class EntityExtractorAgent(IEntityExtractor):
    """Agent for extracting entities and relationships from content."""

    def __init__(self, nlp_model=None):
        self.agent_type = AgentType.ENTITY_EXTRACTOR
        self.nlp = nlp_model or spacy.load("en_core_web_sm")

        # Enhanced entity types for literature
        self.entity_types = {
            'PERSON': 'Character',
            'ORG': 'Organization',
            'GPE': 'Location',
            'LOC': 'Location',
            'WORK_OF_ART': 'Work',
            'EVENT': 'Event',
            'FAC': 'Location'
        }

        # Relationship patterns
        self.relationship_patterns = [
            r'(\w+)\s+(said|told|asked|replied|answered)\s+(\w+)',
            r'(\w+)\s+(went|traveled|journeyed)\s+(to|towards)\s+(\w+)',
            r'(\w+)\s+(lived|resided|dwelt)\s+(in|at)\s+(\w+)',
            r'(\w+)\s+(met|encountered|saw)\s+(\w+)',
            r'(\w+)\s+(belonged to|was part of|member of)\s+(\w+)',
            r'(\w+)\s+(created|built|constructed)\s+(\w+)',
            r'(\w+)\s+(owned|possessed|had)\s+(\w+)',
        ]

    def get_agent_type(self) -> AgentType:
        return self.agent_type

    def can_handle(self, input_data: dict[str, Any]) -> bool:
        return 'content' in input_data and isinstance(input_data['content'], str)

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Process entity extraction."""
        start_time = time.time()

        try:
            content = input_data['content']
            context = input_data.get('context', {})

            extraction = self.extract_entities(content, context)

            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                data={
                    'entities': extraction.entities,
                    'relationships': extraction.relationships,
                    'confidence': extraction.confidence
                },
                confidence=extraction.confidence,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return AgentResult(
                agent_type=self.agent_type,
                success=False,
                data={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def extract_entities(self, content: str, context: dict[str, Any] | None = None) -> EntityExtraction:
        """Extract entities and relationships from content."""
        # Process with spaCy
        doc = self.nlp(content)

        # Extract entities
        entities = []
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity = {
                    'text': ent.text,
                    'label': self.entity_types[ent.label_],
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8
                }
                entities.append(entity)

        # Extract relationships
        relationships = self._extract_relationships(content, entities)

        # Calculate confidence
        confidence = self._calculate_extraction_confidence(entities, relationships)

        return EntityExtraction(
            entities=entities,
            relationships=relationships,
            confidence=confidence
        )

    def _extract_relationships(self, content: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        entity_names = [e['text'] for e in entities]

        for pattern in self.relationship_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    source = groups[0]
                    target = groups[-1]

                    # Check if both entities exist
                    if source in entity_names and target in entity_names:
                        relationship = {
                            'source': source,
                            'target': target,
                            'type': self._classify_relationship_type(match.group(0)),
                            'confidence': 0.7,
                            'context': match.group(0)
                        }
                        relationships.append(relationship)

        return relationships

    def _classify_relationship_type(self, relationship_text: str) -> str:
        """Classify the type of relationship."""
        text_lower = relationship_text.lower()

        if any(word in text_lower for word in ['said', 'told', 'asked', 'replied']):
            return 'SPEAKS_TO'
        elif any(word in text_lower for word in ['went', 'traveled', 'journeyed']):
            return 'TRAVELS_TO'
        elif any(word in text_lower for word in ['lived', 'resided', 'dwelt']):
            return 'LIVES_IN'
        elif any(word in text_lower for word in ['met', 'encountered', 'saw']):
            return 'MEETS'
        elif any(word in text_lower for word in ['belonged', 'part of', 'member']):
            return 'MEMBER_OF'
        elif any(word in text_lower for word in ['created', 'built', 'constructed']):
            return 'CREATES'
        elif any(word in text_lower for word in ['owned', 'possessed', 'had']):
            return 'OWNS'
        else:
            return 'RELATED_TO'

    def _calculate_extraction_confidence(self, entities: list[dict[str, Any]],
                                       relationships: list[dict[str, Any]]) -> float:
        """Calculate confidence score for extraction."""
        if not entities and not relationships:
            return 0.0

        entity_confidence = sum(e.get('confidence', 0.5) for e in entities) / max(len(entities), 1)
        relationship_confidence = sum(r.get('confidence', 0.5) for r in relationships) / max(len(relationships), 1)

        # Weight entities more heavily
        return (entity_confidence * 0.7) + (relationship_confidence * 0.3)


class ContextAggregatorAgent(IContextAggregator):
    """Agent for aggregating semantic and graph search results."""

    def __init__(self):
        self.agent_type = AgentType.CONTEXT_AGGREGATOR

    def get_agent_type(self) -> AgentType:
        return self.agent_type

    def can_handle(self, input_data: dict[str, Any]) -> bool:
        required_keys = ['query', 'semantic_results', 'graph_results']
        return all(key in input_data for key in required_keys)

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Process context aggregation."""
        start_time = time.time()

        try:
            query = input_data['query']
            semantic_results = input_data['semantic_results']
            graph_results = input_data['graph_results']

            aggregation = self.aggregate_context(query, semantic_results, graph_results)

            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                data={
                    'semantic_results': aggregation.semantic_results,
                    'graph_results': aggregation.graph_results,
                    'hybrid_score': aggregation.hybrid_score,
                    'source_attribution': aggregation.source_attribution
                },
                confidence=aggregation.hybrid_score,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return AgentResult(
                agent_type=self.agent_type,
                success=False,
                data={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def aggregate_context(self, query: str, semantic_results: list[dict[str, Any]],
                         graph_results: list[dict[str, Any]]) -> ContextAggregation:
        """Aggregate semantic and graph results into unified context."""
        # Score and rank results
        scored_semantic = self._score_semantic_results(query, semantic_results)
        scored_graph = self._score_graph_results(query, graph_results)

        # Calculate hybrid score
        hybrid_score = self._calculate_hybrid_score(scored_semantic, scored_graph)

        # Create source attribution
        source_attribution = self._create_source_attribution(scored_semantic, scored_graph)

        return ContextAggregation(
            semantic_results=scored_semantic,
            graph_results=scored_graph,
            hybrid_score=hybrid_score,
            source_attribution=source_attribution
        )

    def _score_semantic_results(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score semantic search results."""
        query_words = set(query.lower().split())

        for result in results:
            content = result.get('content', '').lower()
            content_words = set(content.split())

            # Calculate word overlap score
            overlap = len(query_words.intersection(content_words))
            total_words = len(query_words.union(content_words))
            word_score = overlap / max(total_words, 1)

            # Combine with existing score if available
            existing_score = result.get('score', 0.5)
            result['hybrid_score'] = (word_score * 0.3) + (existing_score * 0.7)

        return sorted(results, key=lambda x: x.get('hybrid_score', 0), reverse=True)

    def _score_graph_results(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score graph search results."""
        query_words = set(query.lower().split())

        for result in results:
            # Extract text from graph result
            text_parts = []
            if 'summary' in result:
                text_parts.append(result['summary'])
            if 'description' in result:
                text_parts.append(result['description'])
            if 'properties' in result:
                for prop in result['properties'].values():
                    if isinstance(prop, str):
                        text_parts.append(prop)

            combined_text = ' '.join(text_parts).lower()
            text_words = set(combined_text.split())

            # Calculate relevance score
            overlap = len(query_words.intersection(text_words))
            total_words = len(query_words.union(text_words))
            relevance_score = overlap / max(total_words, 1)

            result['hybrid_score'] = relevance_score

        return sorted(results, key=lambda x: x.get('hybrid_score', 0), reverse=True)

    def _calculate_hybrid_score(self, semantic_results: list[dict[str, Any]],
                               graph_results: list[dict[str, Any]]) -> float:
        """Calculate overall hybrid relevance score."""
        if not semantic_results and not graph_results:
            return 0.0

        semantic_score = sum(r.get('hybrid_score', 0) for r in semantic_results[:3]) / 3
        graph_score = sum(r.get('hybrid_score', 0) for r in graph_results[:3]) / 3

        # Weight semantic results more heavily if available
        if semantic_results and graph_results:
            return (semantic_score * 0.6) + (graph_score * 0.4)
        elif semantic_results:
            return semantic_score
        else:
            return graph_score

    def _create_source_attribution(self, semantic_results: list[dict[str, Any]],
                                  graph_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Create source attribution for results."""
        attribution = []

        for result in semantic_results[:3]:
            attribution.append({
                'type': 'semantic',
                'source': result.get('source', 'unknown'),
                'page': result.get('page_number', 0),
                'score': result.get('hybrid_score', 0)
            })

        for result in graph_results[:3]:
            attribution.append({
                'type': 'graph',
                'source': result.get('entity_name', 'unknown'),
                'relationship_type': result.get('relationship_type', 'unknown'),
                'score': result.get('hybrid_score', 0)
            })

        return attribution


class ResponseGeneratorAgent(IResponseGenerator):
    """Agent for generating responses based on context and analysis."""

    def __init__(self):
        self.agent_type = AgentType.RESPONSE_GENERATOR

        # Response templates for different intents
        self.response_templates = {
            'character': {
                'template': "Based on the content, {character_name} is {description}. {additional_context}",
                'fallback': "The character {character_name} appears in the story with the following details: {content_summary}"
            },
            'plot': {
                'template': "The story involves: {plot_summary}. {key_events}",
                'fallback': "Here's what happens in the story: {content_summary}"
            },
            'setting': {
                'template': "The story takes place in {location}. {setting_details}",
                'fallback': "The setting of the story includes: {content_summary}"
            },
            'relationship': {
                'template': "The relationship between {entity1} and {entity2} is {relationship_type}. {relationship_details}",
                'fallback': "The connections in the story include: {content_summary}"
            },
            'general': {
                'template': "Based on the content: {content_summary}",
                'fallback': "Here's what I found: {content_summary}"
            }
        }

    def get_agent_type(self) -> AgentType:
        return self.agent_type

    def can_handle(self, input_data: dict[str, Any]) -> bool:
        required_keys = ['query', 'context', 'analysis']
        return all(key in input_data for key in required_keys)

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Process response generation."""
        start_time = time.time()

        try:
            query = input_data['query']
            context = input_data['context']
            analysis = input_data['analysis']

            response = self.generate_response(query, context, analysis)

            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                data={'response': response},
                confidence=0.8,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return AgentResult(
                agent_type=self.agent_type,
                success=False,
                data={},
                confidence=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def generate_response(self, query: str, context: ContextAggregation,
                         analysis: QueryAnalysis) -> str:
        """Generate response based on query, context, and analysis."""
        intent = analysis.intent

        # Get relevant content
        content_parts = []

        # Add semantic results
        for result in context.semantic_results[:2]:
            content_parts.append(result.get('content', ''))

        # Add graph results
        for result in context.graph_results[:2]:
            if 'summary' in result:
                content_parts.append(result['summary'])

        combined_content = ' '.join(content_parts)

        # Generate response based on intent
        if intent in self.response_templates:
            template = self.response_templates[intent]['template']
            fallback = self.response_templates[intent]['fallback']

            try:
                response = self._fill_template(template, intent, combined_content, context, analysis)
                if not response or len(response.strip()) < 10:
                    response = self._fill_template(fallback, intent, combined_content, context, analysis)
            except Exception:
                response = self._fill_template(fallback, intent, combined_content, context, analysis)
        else:
            response = f"Based on the content: {combined_content[:300]}{'...' if len(combined_content) > 300 else ''}"

        return response

    def _fill_template(self, template: str, intent: str, content: str,
                      context: ContextAggregation, analysis: QueryAnalysis) -> str:
        """Fill response template with actual data."""
        # Extract key information based on intent
        if intent == 'character':
            # Try to extract character name and description
            character_name = analysis.entities[0] if analysis.entities else "the main character"
            description = content[:200] + "..." if len(content) > 200 else content

            return template.format(
                character_name=character_name,
                description=description,
                additional_context=content[200:400] if len(content) > 200 else ""
            )

        elif intent == 'plot':
            plot_summary = content[:300] + "..." if len(content) > 300 else content
            key_events = content[300:600] if len(content) > 300 else ""

            return template.format(
                plot_summary=plot_summary,
                key_events=key_events
            )

        elif intent == 'setting':
            # Try to extract location
            location = analysis.entities[0] if analysis.entities else "the story's setting"
            setting_details = content[:200] + "..." if len(content) > 200 else content

            return template.format(
                location=location,
                setting_details=setting_details
            )

        elif intent == 'relationship':
            if len(analysis.entities) >= 2:
                entity1, entity2 = analysis.entities[0], analysis.entities[1]
                relationship_type = "connected"
                relationship_details = content[:200] + "..." if len(content) > 200 else content

                return template.format(
                    entity1=entity1,
                    entity2=entity2,
                    relationship_type=relationship_type,
                    relationship_details=relationship_details
                )

        # Fallback to general template
        return template.format(content_summary=content[:400] + "..." if len(content) > 400 else content)


class MultiAgentOrchestrator(IMultiAgentOrchestrator):
    """Orchestrator for coordinating multiple agents."""

    def __init__(self):
        self.agents: dict[AgentType, IAgent] = {}
        self._register_default_agents()

    def _register_default_agents(self):
        """Register default agents."""
        self.register_agent(QueryAnalyzerAgent())
        self.register_agent(EntityExtractorAgent())
        self.register_agent(ContextAggregatorAgent())
        self.register_agent(ResponseGeneratorAgent())

    def register_agent(self, agent: IAgent) -> bool:
        """Register an agent with the orchestrator."""
        try:
            self.agents[agent.get_agent_type()] = agent
            return True
        except Exception as e:
            print(f"Error registering agent {agent.get_agent_type()}: {e}")
            return False

    def process_query(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Process query using multi-agent system."""
        if context is None:
            context = {}

        results = {}

        try:
            # Step 1: Analyze query
            analyzer = self.agents.get(AgentType.QUERY_ANALYZER)
            if analyzer:
                analysis_result = analyzer.process({'query': query})
                results['analysis'] = analysis_result.data
                analysis = analysis_result.data
            else:
                # Fallback analysis
                analysis = {
                    'intent': 'general',
                    'entities': [],
                    'complexity': 0.5,
                    'requires_graph': False,
                    'requires_semantic': True
                }
                results['analysis'] = analysis

            # Step 2: Extract entities if needed
            if analysis.get('requires_graph', False) and 'content' in context:
                extractor = self.agents.get(AgentType.ENTITY_EXTRACTOR)
                if extractor:
                    extraction_result = extractor.process({
                        'content': context['content'],
                        'context': context
                    })
                    results['extraction'] = extraction_result.data

            # Step 3: Aggregate context if we have search results
            if 'semantic_results' in context or 'graph_results' in context:
                aggregator = self.agents.get(AgentType.CONTEXT_AGGREGATOR)
                if aggregator:
                    aggregation_result = aggregator.process({
                        'query': query,
                        'semantic_results': context.get('semantic_results', []),
                        'graph_results': context.get('graph_results', [])
                    })
                    results['aggregation'] = aggregation_result.data

            # Step 4: Generate response
            generator = self.agents.get(AgentType.RESPONSE_GENERATOR)
            if generator and 'aggregation' in results:
                # Create context object for response generation
                from viggo.core.services.interfaces.multi_agent import (
                    ContextAggregation,
                )
                context_obj = ContextAggregation(
                    semantic_results=results['aggregation'].get('semantic_results', []),
                    graph_results=results['aggregation'].get('graph_results', []),
                    hybrid_score=results['aggregation'].get('hybrid_score', 0.0),
                    source_attribution=results['aggregation'].get('source_attribution', [])
                )

                # Create analysis object for response generation
                from viggo.core.services.interfaces.multi_agent import QueryAnalysis
                analysis_obj = QueryAnalysis(
                    intent=analysis.get('intent', 'general'),
                    entities=analysis.get('entities', []),
                    complexity=analysis.get('complexity', 0.5),
                    requires_graph=analysis.get('requires_graph', False),
                    requires_semantic=analysis.get('requires_semantic', True)
                )

                response_result = generator.process({
                    'query': query,
                    'context': context_obj,
                    'analysis': analysis_obj
                })
                results['response'] = response_result.data

            return results

        except Exception as e:
            return {
                'error': str(e),
                'analysis': analysis if 'analysis' in locals() else {},
                'response': {'response': f"Error processing query: {str(e)}"}
            }

    def get_agent_status(self) -> dict[str, Any]:
        """Get status of all registered agents."""
        status = {}
        for agent_type, agent in self.agents.items():
            status[agent_type.value] = {
                'registered': True,
                'type': agent_type.value,
                'can_handle_basic': agent.can_handle({'query': 'test'})
            }
        return status
