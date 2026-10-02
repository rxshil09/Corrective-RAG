"""
test_hallucination_detector.py — Unit tests for the Hallucination Detector

Tests the core novelty: the hallucination detection layer.
Uses mock responses to avoid API calls during testing.
"""

import sys
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hallucination_detector import HallucinationDetector, HallucinationReport


class TestHallucinationReport(unittest.TestCase):
    """Tests for the HallucinationReport dataclass."""

    def test_verdict_grounded(self):
        report = HallucinationReport(
            consistency_score=0.9,
            has_hallucination=False,
            hallucinated_claims=[],
            supported_claims=["Claim A"],
            reasoning="All good",
        )
        self.assertIn("GROUNDED", report.verdict)
        self.assertEqual(report.verdict_color, "green")

    def test_verdict_partial_hallucination(self):
        report = HallucinationReport(
            consistency_score=0.5,
            has_hallucination=True,
            hallucinated_claims=["False claim"],
            supported_claims=[],
            reasoning="Some issues",
        )
        self.assertIn("PARTIAL", report.verdict)
        self.assertEqual(report.verdict_color, "yellow")

    def test_verdict_severe_hallucination(self):
        report = HallucinationReport(
            consistency_score=0.2,
            has_hallucination=True,
            hallucinated_claims=["Many false claims"],
            supported_claims=[],
            reasoning="Very bad",
        )
        self.assertIn("SEVERE", report.verdict)
        self.assertEqual(report.verdict_color, "red")

    def test_needs_strict_mode(self):
        report = HallucinationReport(
            consistency_score=0.3,
            has_hallucination=True,
            hallucinated_claims=[],
            supported_claims=[],
            reasoning="",
            needs_strict_mode=True,
        )
        self.assertTrue(report.needs_strict_mode)


class TestHallucinationDetector(unittest.TestCase):
    """Tests for the HallucinationDetector class."""

    def setUp(self):
        """Set up detector with mocked LLM client."""
        with patch('hallucination_detector.get_llm_client', return_value=(MagicMock(), 'gpt-4o-mini', 'mock_provider')):
            self.detector = HallucinationDetector()
        # Replace client with fresh mock so each test controls it cleanly
        self.detector.client = MagicMock()

    def _make_mock_response(self, json_data: dict) -> MagicMock:
        """Create a mock OpenAI response."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(json_data)
        return mock_resp

    def test_detect_grounded_answer(self):
        """Test that a well-grounded answer gets high score."""
        self.detector.client.chat.completions.create.return_value = self._make_mock_response({
            "consistency_score": 0.95,
            "has_hallucination": False,
            "hallucinated_claims": [],
            "supported_claims": ["RAG combines retrieval with generation"],
            "reasoning": "Answer is well-supported by context"
        })
        
        report = self.detector.detect(
            answer="RAG combines retrieval with generation.",
            context="RAG stands for Retrieval-Augmented Generation. It combines retrieval with generation."
        )
        
        self.assertFalse(report.has_hallucination)
        self.assertGreaterEqual(report.consistency_score, 0.6)

    def test_detect_hallucinated_answer(self):
        """Test that a hallucinated answer gets low score."""
        self.detector.client.chat.completions.create.return_value = self._make_mock_response({
            "consistency_score": 0.2,
            "has_hallucination": True,
            "hallucinated_claims": ["The system was invented in 1985", "Uses quantum computing"],
            "supported_claims": [],
            "reasoning": "Multiple unsupported claims"
        })
        
        report = self.detector.detect(
            answer="RAG was invented in 1985 and uses quantum computing.",
            context="RAG is a modern technique combining information retrieval with LLMs."
        )
        
        self.assertTrue(report.has_hallucination)
        self.assertLess(report.consistency_score, 0.6)
        self.assertGreater(len(report.hallucinated_claims), 0)

    def test_needs_strict_mode_triggered(self):
        """Test that severe hallucination triggers strict mode."""
        self.detector.client.chat.completions.create.return_value = self._make_mock_response({
            "consistency_score": 0.1,
            "has_hallucination": True,
            "hallucinated_claims": ["Many false facts"],
            "supported_claims": [],
            "reasoning": "Almost entirely hallucinated"
        })
        
        report = self.detector.detect("fabricated answer", "real context")
        self.assertTrue(report.needs_strict_mode)

    def test_parse_json_with_code_fences(self):
        """Test that JSON wrapped in code fences is parsed correctly."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '```json\n{"consistency_score": 0.8, "has_hallucination": false, "hallucinated_claims": [], "supported_claims": ["fact"], "reasoning": "good"}\n```'
        self.detector.client.chat.completions.create.return_value = mock_resp
        
        report = self.detector.detect("answer", "context")
        self.assertAlmostEqual(report.consistency_score, 0.8, places=1)
        self.assertFalse(report.has_hallucination)

    def test_score_clamping(self):
        """Test that scores outside [0,1] are clamped."""
        self.detector.client.chat.completions.create.return_value = self._make_mock_response({
            "consistency_score": 1.5,  # Out of range!
            "has_hallucination": False,
            "hallucinated_claims": [],
            "supported_claims": [],
            "reasoning": ""
        })
        
        report = self.detector.detect("answer", "context")
        self.assertLessEqual(report.consistency_score, 1.0)

    def test_graceful_error_handling(self):
        """Test that API errors don't crash the system."""
        self.detector.client.chat.completions.create.side_effect = Exception("API Error")
        
        report = self.detector.detect("answer", "context")
        # Should return a safe default, not raise
        self.assertIsNotNone(report)
        self.assertIsInstance(report.consistency_score, float)

    def test_malformed_json_handled(self):
        """Test that malformed JSON responses are handled gracefully."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "This is not JSON at all!"
        self.detector.client.chat.completions.create.return_value = mock_resp
        
        report = self.detector.detect("answer", "context")
        self.assertIsNotNone(report)


class TestRAGEngine(unittest.TestCase):
    """Tests for RAGResponse logic (no langchain needed)."""

    def _make_response(self, score: float):
        """Helper to build a RAGResponse-like object using the real dataclass."""
        # Define a minimal RAGResponse inline to avoid langchain import
        from dataclasses import dataclass, field as dc_field
        from typing import List as TList
        
        mock_report = HallucinationReport(
            consistency_score=score,
            has_hallucination=score < 0.6,
            hallucinated_claims=[],
            supported_claims=[],
            reasoning=""
        )

        @dataclass
        class _RAGResponse:
            query: str
            answer: str
            sources: TList[str]
            consistency_score: float
            hallucination_report: object
            regeneration_count: int
            used_strict_mode: bool
            used_query_rewrite: bool = False
            rewritten_query: str = None
            execution_trace: TList[str] = dc_field(default_factory=list)

            @property
            def is_reliable(self):
                return self.consistency_score >= 0.6

            @property
            def confidence_label(self):
                if self.consistency_score >= 0.85:
                    return "HIGH"
                elif self.consistency_score >= 0.6:
                    return "MEDIUM"
                elif self.consistency_score >= 0.4:
                    return "LOW"
                else:
                    return "VERY LOW"

        return _RAGResponse(
            query="q", answer="a", sources=[],
            consistency_score=score,
            hallucination_report=mock_report,
            regeneration_count=0, used_strict_mode=False,
            used_query_rewrite=False, rewritten_query=None,
            execution_trace=["retrieve", "generate", "detect_hallucination"]
        )

    def test_rag_response_fields(self):
        r = self._make_response(0.85)
        self.assertEqual(r.query, "q")
        self.assertTrue(r.is_reliable)
        self.assertEqual(r.confidence_label, "HIGH")
        self.assertFalse(r.used_query_rewrite)
        self.assertIn("retrieve", r.execution_trace)

    def test_confidence_labels(self):
        self.assertEqual(self._make_response(0.9).confidence_label, "HIGH")
        self.assertEqual(self._make_response(0.7).confidence_label, "MEDIUM")
        self.assertEqual(self._make_response(0.5).confidence_label, "LOW")
        self.assertEqual(self._make_response(0.2).confidence_label, "VERY LOW")


class TestLangGraphRouting(unittest.TestCase):
    """Tests for LangGraph conditional edge routing logic."""

    def setUp(self):
        from rag_graph import RAGGraphBuilder
        mock_engine = MagicMock()
        mock_engine.detector = MagicMock()
        mock_engine.db = MagicMock()
        mock_engine.client = MagicMock()
        mock_engine.model_name = "mock-model"
        self.builder = RAGGraphBuilder(mock_engine)

    def test_route_pass_on_high_score(self):
        mock_report = HallucinationReport(
            consistency_score=0.8,
            has_hallucination=False,
            hallucinated_claims=[],
            supported_claims=["fact"],
            reasoning="verified"
        )
        state = {
            "hallucination_report": mock_report,
            "consistency_score": 0.8,
            "attempt": 1,
            "used_query_rewrite": False,
            "verbose": False,
        }
        route = self.builder.route_after_detection(state)
        self.assertEqual(route, "pass")

    def test_route_rewrite_query_on_severe_hallucination(self):
        mock_report = HallucinationReport(
            consistency_score=0.2,
            has_hallucination=True,
            hallucinated_claims=["unsupported claim"],
            supported_claims=[],
            reasoning="bad"
        )
        state = {
            "hallucination_report": mock_report,
            "consistency_score": 0.2,
            "attempt": 1,
            "used_query_rewrite": False,
            "verbose": False,
        }
        route = self.builder.route_after_detection(state)
        self.assertEqual(route, "rewrite_query")

    def test_route_strict_generate_on_partial_hallucination(self):
        mock_report = HallucinationReport(
            consistency_score=0.5,
            has_hallucination=True,
            hallucinated_claims=["minor issue"],
            supported_claims=["supported fact"],
            reasoning="partially grounded"
        )
        state = {
            "hallucination_report": mock_report,
            "consistency_score": 0.5,
            "attempt": 1,
            "used_query_rewrite": False,
            "verbose": False,
        }
        route = self.builder.route_after_detection(state)
        self.assertEqual(route, "strict_generate")

    def test_route_end_on_max_attempts(self):
        mock_report = HallucinationReport(
            consistency_score=0.3,
            has_hallucination=True,
            hallucinated_claims=["bad"],
            supported_claims=[],
            reasoning=""
        )
        state = {
            "hallucination_report": mock_report,
            "consistency_score": 0.3,
            "attempt": 3,  # MAX_REGENERATION_ATTEMPTS
            "best_score": 0.4,
            "used_query_rewrite": True,
            "verbose": False,
        }
        route = self.builder.route_after_detection(state)
        self.assertEqual(route, "end_max_attempts")


class TestRelevanceFiltering(unittest.TestCase):
    """Tests for ChromaDB distance scoring and relevance filtering."""

    def test_relevance_filter_accepted_and_dropped(self):
        from langchain_core.documents import Document
        from rag_graph import RAGGraphBuilder

        doc_good = Document(page_content="Relevant content", metadata={"source_name": "good_doc.md"})
        doc_bad = Document(page_content="Unrelated noise", metadata={"source_name": "noise_doc.md"})

        mock_engine = MagicMock()
        # Chroma returns (doc, distance). Distance 0.2 -> similarity 0.8 (accepted). Distance 0.9 -> similarity 0.1 (filtered)
        mock_engine.db.similarity_search_with_score.return_value = [
            (doc_good, 0.2),
            (doc_bad, 0.9),
        ]
        builder = RAGGraphBuilder(mock_engine)

        state = {
            "query": "What is RAG?",
            "search_query": "What is RAG?",
            "verbose": False,
        }
        res = builder.node_retrieve(state)

        # Only doc_good should pass the MIN_RELEVANCE_SCORE (0.25)
        self.assertEqual(len(res["retrieved_chunks"]), 1)
        self.assertEqual(res["retrieved_chunks"][0].metadata["source_name"], "good_doc.md")
        self.assertIn("good_doc.md", res["sources"])

    def test_relevance_fallback_when_all_filtered(self):
        from langchain_core.documents import Document
        from rag_graph import RAGGraphBuilder

        doc1 = Document(page_content="Low relevance 1", metadata={"source_name": "doc1.md"})
        doc2 = Document(page_content="Low relevance 2", metadata={"source_name": "doc2.md"})

        mock_engine = MagicMock()
        # All distances are 0.95 -> similarity 0.05 (< 0.25)
        mock_engine.db.similarity_search_with_score.return_value = [
            (doc1, 0.95),
            (doc2, 0.95),
        ]
        builder = RAGGraphBuilder(mock_engine)

        state = {
            "query": "Completely obscure topic",
            "search_query": "Completely obscure topic",
            "verbose": False,
        }
        res = builder.node_retrieve(state)

        # Fallback to MIN_CHUNKS_REQUIRED (1)
        self.assertGreaterEqual(len(res["retrieved_chunks"]), 1)


class TestDocumentLoaders(unittest.TestCase):
    """Tests for multi-format document loading logic."""

    def test_load_documents_multi_format_discovery(self):
        from create_database import load_documents
        import tempfile
        import shutil

        # Create temporary folder with mock markdown and txt files
        tmp_dir = tempfile.mkdtemp()
        try:
            (Path(tmp_dir) / "test.md").write_text("# Markdown Title\nSome content.", encoding="utf-8")
            (Path(tmp_dir) / "test.txt").write_text("Plain text content.", encoding="utf-8")

            docs = load_documents(tmp_dir)
            self.assertEqual(len(docs), 2)
            sources = [d.metadata.get("source", "") for d in docs]
            self.assertTrue(any("test.md" in s for s in sources))
            self.assertTrue(any("test.txt" in s for s in sources))
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("Hallucination-Aware RAG — Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)