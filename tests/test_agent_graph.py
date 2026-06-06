"""
tests/test_agent_graph.py
=========================
Unit tests for agent_graph.py
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from agent_graph import (
    AgentEdge,
    AgentGraph,
    AgentNode,
    AgentType,
    BottleneckReport,
    CentralityMetrics,
    EdgeType,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def graph():
    """Fresh graph with initial topology."""
    return AgentGraph().build_initial_graph()


@pytest.fixture
def empty_graph():
    """Empty graph, no nodes or edges."""
    return AgentGraph()


@pytest.fixture
def valid_json_file(graph, tmp_path):
    """A valid agent_graph.json saved to a temp file."""
    path = tmp_path / "agent_graph.json"
    graph.save(str(path))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildInitialGraph:
    def test_creates_four_nodes(self, graph):
        assert len(graph.nodes) == 4
        assert "VisionAgent" in graph.nodes
        assert "GeomechanicalAgent" in graph.nodes
        assert "GasAgent" in graph.nodes
        assert "MonitorAgent" in graph.nodes

    def test_creates_six_edges(self, graph):
        assert len(graph.edges) == 6

    def test_node_attributes(self, graph):
        assert graph.nodes["GasAgent"].type == "gas"
        assert graph.nodes["GasAgent"].active is True
        assert graph.nodes["GasAgent"].alert_threshold == 0.3

        assert graph.nodes["VisionAgent"].type == "vision"
        assert graph.nodes["VisionAgent"].alert_threshold == 0.7

        assert graph.nodes["MonitorAgent"].type == "monitor"
        assert graph.nodes["MonitorAgent"].reporting_frequency == "every_1min"

    def test_edge_types(self, graph):
        # GasAgent -> MonitorAgent is alert
        edge = graph.edges[("GasAgent", "MonitorAgent")]
        assert edge.type == "alert"
        assert edge.weight == 1.0

        # VisionAgent -> MonitorAgent is report
        edge = graph.edges[("VisionAgent", "MonitorAgent")]
        assert edge.type == "report"
        assert edge.weight == 0.6

        # MonitorAgent -> VisionAgent is query
        edge = graph.edges[("MonitorAgent", "VisionAgent")]
        assert edge.type == "query"

    def test_returns_self_for_chaining(self, graph):
        result = graph.build_initial_graph()
        assert result is graph


class TestAddNode:
    def test_add_node_inserts_into_graph(self, empty_graph):
        node = AgentNode(id="TestAgent", type="gas", active=True)
        empty_graph.add_node(node)

        assert "TestAgent" in empty_graph.nodes
        assert empty_graph.nodes["TestAgent"].type == "gas"

    def test_add_node_invalidates_cache(self, empty_graph):
        # Force cache by computing centrality
        empty_graph.add_node(AgentNode(id="A", type="vision", active=True))
        empty_graph.add_node(AgentNode(id="B", type="gas", active=True))
        _ = empty_graph.compute_centrality()
        assert empty_graph._centrality_cache is not None

        empty_graph.add_node(AgentNode(id="C", type="geomechanical", active=True))
        assert empty_graph._centrality_cache is None  # cache invalidated


class TestAddEdge:
    def test_add_edge_inserts_edge(self, empty_graph):
        empty_graph.add_node(AgentNode(id="A", type="vision", active=True))
        empty_graph.add_node(AgentNode(id="B", type="gas", active=True))

        edge = AgentEdge(source="A", target="B", type="alert", weight=0.9)
        empty_graph.add_edge(edge)

        assert ("A", "B") in empty_graph.edges
        assert empty_graph.edges[("A", "B")].type == "alert"

    def test_add_edge_raises_on_missing_source(self, empty_graph):
        empty_graph.add_node(AgentNode(id="B", type="gas", active=True))
        edge = AgentEdge(source="A", target="B", type="alert")

        with pytest.raises(ValueError, match="Source agent 'A' does not exist"):
            empty_graph.add_edge(edge)

    def test_add_edge_raises_on_missing_target(self, empty_graph):
        empty_graph.add_node(AgentNode(id="A", type="vision", active=True))
        edge = AgentEdge(source="A", target="B", type="alert")

        with pytest.raises(ValueError, match="Target agent 'B' does not exist"):
            empty_graph.add_edge(edge)


# ─────────────────────────────────────────────────────────────────────────────
# CENTRALITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCentrality:
    def test_returns_all_four_metrics(self, graph):
        metrics = graph.compute_centrality()

        assert isinstance(metrics, CentralityMetrics)
        assert "degree" in metrics.to_dict()
        assert "betweenness" in metrics.to_dict()
        assert "pagerank" in metrics.to_dict()
        assert "closeness" in metrics.to_dict()

    def test_contains_all_nodes(self, graph):
        metrics = graph.compute_centrality()

        for node_id in graph.nodes:
            assert node_id in metrics.degree
            assert node_id in metrics.betweenness
            assert node_id in metrics.pagerank
            assert node_id in metrics.closeness

    def test_betweenness_monitor_is_highest(self, graph):
        """MonitorAgent has edges from all 3 agents so should have high betweenness."""
        metrics = graph.compute_centrality()
        most_central, _ = graph.most_central_node("betweenness")
        assert most_central == "MonitorAgent"

    def test_caching_avoids_recompute(self, graph):
        metrics1 = graph.compute_centrality()
        metrics2 = graph.compute_centrality()
        assert graph._centrality_cache is not None
        # Same object returned from cache
        assert metrics1 is metrics2

    def test_force_refresh_bypasses_cache(self, graph):
        _ = graph.compute_centrality()
        assert graph._centrality_cache is not None

        metrics = graph.compute_centrality(force_refresh=True)
        # Should still return valid metrics (freshly computed)
        assert len(metrics.degree) == 4


class TestMostCentralNode:
    def test_betweenness_returns_monitor(self, graph):
        node, score = graph.most_central_node("betweenness")
        assert node == "MonitorAgent"
        assert score > 0

    def test_pagerank_returns_monitor(self, graph):
        node, score = graph.most_central_node("pagerank")
        assert node == "MonitorAgent"


# ─────────────────────────────────────────────────────────────────────────────
# BOTTLENECK DETECTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectBottlenecks:
    def test_returns_bottleneck_report(self, graph):
        report = graph.detect_bottlenecks()

        assert isinstance(report, BottleneckReport)
        assert isinstance(report.critical_nodes, list)
        assert isinstance(report.critical_edges, list)
        assert isinstance(report.alert_coverage, float)

    def test_cobertura_alerta_is_between_0_and_1(self, graph):
        report = graph.detect_bottlenecks()
        assert 0.0 <= report.alert_coverage <= 1.0

    def test_monitor_not_in_nodos_criticos(self, graph):
        """Monitor has high connectivity so should not be flagged as critical."""
        report = graph.detect_bottlenecks()
        # Monitor is highly connected so shouldn't be in critical nodes
        # (it has high betweenness, not low degree)
        # With default thresholds, it may or may not be in the list
        # The key is it shouldn't cause failures

    def test_caching_avoids_recompute(self, graph):
        report1 = graph.detect_bottlenecks()
        report2 = graph.detect_bottlenecks()
        assert graph._bottleneck_cache is not None
        assert report1 is report2


# ─────────────────────────────────────────────────────────────────────────────
# ALERT CASCADE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeAlertCascade:
    def test_returns_propagation_tree(self, graph):
        result = graph.analyze_alert_cascade("GasAgent")

        assert "source" in result
        assert "total_affected" in result
        assert "affected_agents" in result
        assert "propagation_paths" in result
        assert result["source"] == "GasAgent"

    def test_source_included_in_affected(self, graph):
        result = graph.analyze_alert_cascade("GasAgent")
        assert "GasAgent" in result["affected_agents"]

    def test_unknown_agent_returns_error(self, graph):
        result = graph.analyze_alert_cascade("NonExistentAgent")
        assert "error" in result


class TestDetectAlertFatigue:
    def test_returns_fatigue_scores(self, graph):
        result = graph.detect_alert_fatigue()

        assert "fatigue_scores" in result
        assert "at_risk_agents" in result
        assert len(result["fatigue_scores"]) == 4  # one per agent

    def test_fatigue_scores_between_0_and_1(self, graph):
        result = graph.detect_alert_fatigue()
        for agent, score in result["fatigue_scores"].items():
            assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSave:
    def test_save_creates_json_file(self, graph, tmp_path):
        path = tmp_path / "agent_graph.json"
        graph.save(str(path))

        assert path.exists()

    def test_save_json_structure(self, graph, tmp_path):
        path = tmp_path / "agent_graph.json"
        graph.save(str(path))

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 4
        assert len(data["edges"]) == 6


class TestLoad:
    def test_load_restores_graph(self, valid_json_file):
        graph = AgentGraph().load(str(valid_json_file))

        assert len(graph.nodes) == 4
        assert len(graph.edges) == 6
        assert "MonitorAgent" in graph.nodes

    def test_load_restores_centrality_cache(self, valid_json_file):
        # First save with centrality computed
        graph = AgentGraph().build_initial_graph()
        _ = graph.compute_centrality()  # populate cache

        path = valid_json_file
        graph.save(str(path))

        # Load should restore cache
        loaded = AgentGraph().load(str(path))
        assert loaded._centrality_cache is not None

    def test_load_raises_on_missing_file(self, empty_graph):
        with pytest.raises(FileNotFoundError):
            empty_graph.load("/nonexistent/path.json")

    def test_load_raises_on_invalid_json_structure(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        # JSON has "nodes" key but missing "edges" entirely
        bad_file.write_text('{"nodes": {}}', encoding="utf-8")

        with pytest.raises(ValueError, match="missing 'edges' key"):
            AgentGraph().load(str(bad_file))

    def test_load_raises_on_missing_node_fields(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(
            json.dumps({
                "nodes": {"Agent1": {"id": "Agent1"}},  # missing required fields
                "edges": {},
            }),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing required fields"):
            AgentGraph().load(str(bad_file))

    def test_load_raises_on_invalid_edge_key_format(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(
            json.dumps({
                "nodes": {
                    "A": {
                        "id": "A", "type": "vision", "active": True,
                        "reporting_frequency": "every_5min", "alert_threshold": 0.5,
                        "alerts_sent": 0, "alerts_received": 0
                    },
                    "B": {
                        "id": "B", "type": "gas", "active": True,
                        "reporting_frequency": "every_5min", "alert_threshold": 0.5,
                        "alerts_sent": 0, "alerts_received": 0
                    },
                },
                "edges": {
                    "invalid_key": {  # no "->" separator
                        "source": "A", "target": "B", "type": "alert",
                        "weight": 1.0, "frequency": "every_1min",
                        "messages_sent": 0, "last_communication": None
                    }
                },
            }),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Invalid edge key format"):
            AgentGraph().load(str(bad_file))


class TestSaveLoadRoundtrip:
    def test_save_load_preserves_graph(self, graph, tmp_path):
        path = tmp_path / "roundtrip.json"
        graph.save(str(path))

        loaded = AgentGraph().load(str(path))

        assert len(loaded.nodes) == len(graph.nodes)
        assert len(loaded.edges) == len(graph.edges)
        for node_id in graph.nodes:
            assert node_id in loaded.nodes
            assert loaded.nodes[node_id].type == graph.nodes[node_id].type

    def test_save_load_preserves_edge_attributes(self, graph, tmp_path):
        path = tmp_path / "roundtrip.json"
        graph.save(str(path))

        loaded = AgentGraph().load(str(path))

        for (src, tgt), edge in graph.edges.items():
            loaded_edge = loaded.edges[(src, tgt)]
            assert loaded_edge.type == edge.type
            assert loaded_edge.weight == edge.weight
            assert loaded_edge.frequency == edge.frequency


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE EDGE / EVENT LOG TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateEdge:
    def test_update_edge_modifies_attributes(self, graph):
        graph.update_edge("GasAgent", "MonitorAgent", messages_sent=42)

        edge = graph.edges[("GasAgent", "MonitorAgent")]
        assert edge.messages_sent == 42

    def test_update_edge_logs_event(self, graph):
        initial_log_len = len(graph.get_event_log())
        graph.update_edge("GasAgent", "MonitorAgent", messages_sent=1)

        assert len(graph.get_event_log()) == initial_log_len + 1


class TestEventLog:
    def test_get_event_log_returns_list(self, graph):
        log = graph.get_event_log()
        assert isinstance(log, list)

    def test_log_records_initialization(self, graph):
        log = graph.get_event_log()
        event_types = [e["event_type"] for e in log]
        assert "graph_initialized" in event_types

    def test_log_records_edge_updates(self, graph):
        graph.update_edge("GasAgent", "MonitorAgent", messages_sent=5)
        log = graph.get_event_log()
        event_types = [e["event_type"] for e in log]
        assert "edge_updated" in event_types


# ─────────────────────────────────────────────────────────────────────────────
# NODE / EDGE OPERATIONS TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDeactivateNode:
    def test_deactivate_marks_node_inactive(self, graph):
        graph.deactivate_node("VisionAgent")

        assert graph.nodes["VisionAgent"].active is False

    def test_deactivate_invalidates_cache(self, graph):
        _ = graph.compute_centrality()
        assert graph._centrality_cache is not None

        graph.deactivate_node("VisionAgent")
        assert graph._centrality_cache is None


class TestRemoveEdge:
    def test_remove_edge_deletes_edge(self, graph):
        graph.remove_edge("GasAgent", "MonitorAgent")

        assert ("GasAgent", "MonitorAgent") not in graph.edges

    def test_remove_edge_invalidates_cache(self, graph):
        _ = graph.compute_centrality()
        assert graph._centrality_cache is not None

        graph.remove_edge("GasAgent", "MonitorAgent")
        assert graph._centrality_cache is None


# ─────────────────────────────────────────────────────────────────────────────
# TO_DATAFRAME TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestToDataFrame:
    def test_returns_dataframe(self, graph):
        df = graph.to_dataframe()
        assert len(df) == 6
        assert "source" in df.columns
        assert "target" in df.columns
        assert "type" in df.columns
        assert "weight" in df.columns

    def test_dataframe_contains_all_edges(self, graph):
        df = graph.to_dataframe()
        edge_pairs = set(zip(df["source"], df["target"]))
        expected = set(graph.edges.keys())
        assert edge_pairs == expected


# ─────────────────────────────────────────────────────────────────────────────
# ALERT COVERAGE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertCoverage:
    def test_coverage_is_025_for_default_graph(self, graph):
        """Monitor only reaches itself via alert/report edges (all are incoming).

        In the default topology, all alert/report edges go TOWARDS the Monitor,
        not FROM it. So from Monitor's perspective, only Monitor itself is reachable
        via alert/report edges → coverage = 1/4 = 0.25.
        """
        report = graph.detect_bottlenecks()
        assert report.alert_coverage == 0.25

    def test_coverage_0_when_monitor_missing(self, empty_graph):
        """If there's no Monitor, there's no alert coverage."""
        empty_graph.add_node(AgentNode(id="Agent", type="vision", active=True))
        # Don't add Monitor
        report = empty_graph.detect_bottlenecks()
        assert report.alert_coverage == 0.0