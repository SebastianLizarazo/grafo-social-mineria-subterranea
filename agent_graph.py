"""
agent_graph.py
=============
Agent Interaction Graph — Underground Mining IA
Layer 5: Systemic analysis of agent communication patterns.

This module models the communication graph between the 4 LangGraph agents:
    - VisionAgent        (vision: crack/structural defect detection)
    - GeomechanicalAgent (geomechanical sensor monitoring)
    - GasAgent           (environmental gas monitoring)
    - MonitorAgent       (orchestration and decision hub)

It provides:
    - Graph construction and updates
    - Centrality metrics (degree, betweenness, PageRank, closeness)
    - Bottleneck detection
    - Alert cascade analysis
    - Serialization to JSON/Parquet
    - Visualization (Gravis + Altair)
    - Integration with DuckDB for event log querying

Usage:
    from agent_graph import AgentGraph
    graph = AgentGraph()
    graph.build_initial_graph()
    metrics = graph.compute_centrality()
    bottlenecks = graph.detect_bottlenecks()
    graph.visualize_static("output/agent_graph.png")
    graph.visualize_interactive("output/agent_graph.html")
    graph.save("./data/agent_graph.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optional graph-tool import — falls back to NetworkX if unavailable
# ─────────────────────────────────────────────────────────────────────────────
try:
    import graph_tool.all as gt
    HAS_GRAPHTOOL = True
except ImportError:
    HAS_GRAPHTOOL = False

import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
try:
    import gravis
    HAS_GRAVIS = True
except ImportError:
    HAS_GRAVIS = False

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

# pandas is needed for to_dataframe() regardless of altair
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS & DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class AgentType(Enum):
    VISION = "VisionAgent"
    GEOMECHANICAL = "GeomechanicalAgent"
    GAS = "GasAgent"
    MONITOR = "MonitorAgent"


class EdgeType(Enum):
    ALERT = "alert"
    QUERY = "query"
    REPORT = "report"
    CONFIRMATION = "confirmation"


@dataclass
class AgentNode:
    """Represents a node in the agent communication graph."""
    id: str
    type: str  # "vision" | "geomechanical" | "gas" | "monitor"
    active: bool = True
    reporting_frequency: str = "every_5min"  # reporting frequency
    alert_threshold: float = 0.5  # alert threshold
    alerts_sent: int = 0
    alerts_received: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentEdge:
    """Represents a directed edge (communication) between two agents."""
    source: str
    target: str
    type: str  # "alert" | "query" | "report" | "confirmation"
    weight: float = 1.0
    frequency: str = "every_5min"
    messages_sent: int = 0
    last_communication: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CentralityMetrics:
    """Centrality metrics for all nodes in the graph."""
    degree: dict[str, float]
    betweenness: dict[str, float]
    pagerank: dict[str, float]
    closeness: dict[str, float]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "degree": self.degree,
            "betweenness": self.betweenness,
            "pagerank": self.pagerank,
            "closeness": self.closeness,
            "timestamp": self.timestamp,
        }


@dataclass
class BottleneckReport:
    """Identified bottlenecks in the agent communication graph."""
    critical_nodes: list[str]
    critical_edges: list[tuple[str, str]]
    alert_coverage: float  # 0-1, how well alerts propagate
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "critical_nodes": self.critical_nodes,
            "critical_edges": [list(a) for a in self.critical_edges],
            "alert_coverage": self.alert_coverage,
            "timestamp": self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT GRAPH CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AgentGraph:
    """
    Models and analyzes the communication graph between the 4 mining safety agents.

    Parameters
    ----------
    use_graphtool : bool
        If True and graph-tool is available, use it for analysis (faster).
        Defaults to False (uses NetworkX) if not specified.
    """

    def __init__(self, use_graphtool: bool = False):
        self.use_graphtool = HAS_GRAPHTOOL and use_graphtool
        self.G: nx.DiGraph = nx.DiGraph()
        self.nodes: dict[str, AgentNode] = {}
        self.edges: dict[tuple[str, str], AgentEdge] = {}
        self._centrality_cache: Optional[CentralityMetrics] = None
        self._bottleneck_cache: Optional[BottleneckReport] = None
        self._event_log: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────

    def build_initial_graph(self) -> "AgentGraph":
        """
        Build the initial 4-agent communication graph based on the
        LangGraph orchestration described in the project spec.

        Default topology:
            GasAgent → MonitorAgent (alert, high frequency)
            GeomechanicalAgent → MonitorAgent (alert, medium frequency)
            VisionAgent → MonitorAgent (report, low frequency)
            MonitorAgent → VisionAgent (status query)
            MonitorAgent → GasAgent (status query)
            MonitorAgent → GeomechanicalAgent (status query)
        """
        # ── Nodes ────────────────────────────────────────────────────────
        agents = [
            AgentNode(
                id="VisionAgent",
                type="vision",
                active=True,
                reporting_frequency="every_10min",
                alert_threshold=0.7,
            ),
            AgentNode(
                id="GeomechanicalAgent",
                type="geomechanical",
                active=True,
                reporting_frequency="every_5min",
                alert_threshold=0.5,
            ),
            AgentNode(
                id="GasAgent",
                type="gas",
                active=True,
                reporting_frequency="every_1min",
                alert_threshold=0.3,
            ),
            AgentNode(
                id="MonitorAgent",
                type="monitor",
                active=True,
                reporting_frequency="every_1min",
                alert_threshold=0.5,
            ),
        ]

        for agent in agents:
            self.add_node(agent)

        # ── Edges ────────────────────────────────────────────────────────
        edges_config = [
            # (source, target, type, weight, frequency)
            ("GasAgent", "MonitorAgent", "alert", 1.0, "every_1min"),
            ("GeomechanicalAgent", "MonitorAgent", "alert", 0.8, "every_5min"),
            ("VisionAgent", "MonitorAgent", "report", 0.6, "every_10min"),
            ("MonitorAgent", "VisionAgent", "query", 0.5, "every_10min"),
            ("MonitorAgent", "GasAgent", "query", 0.5, "every_10min"),
            ("MonitorAgent", "GeomechanicalAgent", "query", 0.5, "every_10min"),
        ]

        for src, tgt, edge_type, weight, freq in edges_config:
            self.add_edge(
                AgentEdge(
                    source=src,
                    target=tgt,
                    type=edge_type,
                    weight=weight,
                    frequency=freq,
                )
            )

        self._log_event("graph_initialized", {"n_nodes": 4, "n_edges": len(edges_config)})
        return self

    def add_node(self, agent: AgentNode) -> "AgentGraph":
        """Add an agent node to the graph."""
        self.nodes[agent.id] = agent
        self.G.add_node(agent.id, **agent.to_dict())
        self._invalidate_cache()
        return self

    def add_edge(self, edge: AgentEdge) -> "AgentGraph":
        """Add a directed communication edge between two agents.

        Raises
        ------
        ValueError
            If source or target agent does not exist in the graph.
        """
        if edge.source not in self.nodes:
            raise ValueError(f"Source agent '{edge.source}' does not exist in graph")
        if edge.target not in self.nodes:
            raise ValueError(f"Target agent '{edge.target}' does not exist in graph")
        key = (edge.source, edge.target)
        self.edges[key] = edge
        self.G.add_edge(
            edge.source,
            edge.target,
            **edge.to_dict(),
        )
        self._invalidate_cache()
        return self

    def update_edge(self, source: str, target: str, **kwargs) -> "AgentGraph":
        """Update edge attributes (e.g., increment message count)."""
        key = (source, target)
        if key in self.edges:
            for k, v in kwargs.items():
                setattr(self.edges[key], k, v)
        if self.G.has_edge(source, target):
            self.G[source][target].update(kwargs)
        self._log_event("edge_updated", {
            "source": source,
            "target": target,
            **kwargs,
        })
        return self

    def remove_edge(self, source: str, target: str) -> "AgentGraph":
        """Remove an edge (e.g., disable an alert channel)."""
        key = (source, target)
        self.edges.pop(key, None)
        self.G.remove_edge(source, target)
        self._invalidate_cache()
        self._log_event("edge_removed", {"source": source, "target": target})
        return self

    def deactivate_node(self, node_id: str) -> "AgentGraph":
        """Mark an agent as inactive (e.g., during maintenance)."""
        if node_id in self.nodes:
            self.nodes[node_id].active = False
            self.G.nodes[node_id]["active"] = False
            self._invalidate_cache()
            self._log_event("node_deactivated", {"node_id": node_id})
        return self

    def _invalidate_cache(self):
        """Clear cached metrics when graph structure changes."""
        self._centrality_cache = None
        self._bottleneck_cache = None

    # ─────────────────────────────────────────────────────────────────────────
    # CENTRALITY METRICS
    # ─────────────────────────────────────────────────────────────────────────

    def compute_centrality(self, force_refresh: bool = False) -> CentralityMetrics:
        """
        Compute all centrality metrics for the current graph.

        Uses graph-tool if available and requested, otherwise NetworkX.

        Returns
        -------
        CentralityMetrics
            degree, betweenness, PageRank, closeness per node.
        """
        if self._centrality_cache and not force_refresh:
            return self._centrality_cache

        if self.use_graphtool and HAS_GRAPHTOOL:
            metrics = self._compute_centrality_gt()
        else:
            metrics = self._compute_centrality_nx()

        self._centrality_cache = metrics
        return metrics

    def _compute_centrality_nx(self) -> CentralityMetrics:
        """Compute centrality using NetworkX."""
        degree = nx.degree_centrality(self.G)
        betweenness = nx.betweenness_centrality(self.G)
        pagerank = nx.pagerank(self.G)
        closeness = nx.closeness_centrality(self.G)

        return CentralityMetrics(
            degree=dict(degree),
            betweenness=dict(betweenness),
            pagerank=dict(pagerank),
            closeness=dict(closeness),
        )

    def _compute_centrality_gt(self) -> CentralityMetrics:
        """Compute centrality using graph-tool (faster for large graphs)."""
        g = gt.Graph(directed=True)

        # Add vertices
        name_to_vprop = g.new_vertex_property("string")
        v_index = {node_id: i for i, node_id in enumerate(self.G.nodes())}

        for node_id, idx in v_index.items():
            name_to_vprop[idx] = node_id

        # Add edges
        for u, v in self.G.edges():
            g.add_edge(v_index[u], v_index[v])

        # Degree centrality — use graph_tool's built-in degree property maps
        in_deg = g.degree_property_map("in", dtype="int")
        out_deg = g.degree_property_map("out", dtype="int")
        n = g.num_vertices()
        total_deg = {
            name_to_vprop[i]: (in_deg[g.vertex(i)] + out_deg[g.vertex(i)]) / (2 * (n - 1))
            if n > 1 else 0.0
            for i in range(n)
        }

        # Betweenness
        betweenness_gt = gt.betweenness(g)
        betweenness = {name_to_vprop[i]: abs(betweenness_gt[g.vertex(i)])
                       for i in range(g.num_vertices())}

        # PageRank
        pagerank_gt = gt.pagerank(g)
        pagerank = {name_to_vprop[i]: pagerank_gt[g.vertex(i)]
                    for i in range(g.num_vertices())}

        # Closeness
        closeness = gt.closeness(g)
        closeness_dict = {name_to_vprop[i]: max(0, closeness[g.vertex(i)])
                         for i in range(g.num_vertices())}

        return CentralityMetrics(
            degree=total_deg,
            betweenness=betweenness,
            pagerank=pagerank,
            closeness=closeness_dict,
        )

    def most_central_node(self, metric: str = "betweenness") -> tuple[str, float]:
        """
        Return the most central node by a given centrality metric.

        Parameters
        ----------
        metric : str
            One of "degree", "betweenness", "pagerank", "closeness".
        """
        metrics = self.compute_centrality()
        metric_dict = getattr(metrics, metric)
        return max(metric_dict.items(), key=lambda x: x[1])

    def least_central_node(self, metric: str = "degree") -> tuple[str, float]:
        """Return the least central node by a given centrality metric."""
        metrics = self.compute_centrality()
        metric_dict = getattr(metrics, metric)
        return min(metric_dict.items(), key=lambda x: x[1])

    # ─────────────────────────────────────────────────────────────────────────
    # BOTTLENECK DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def detect_bottlenecks(
        self,
        threshold_degree: float = 0.1,
        threshold_betweenness: float = 0.3,
        force_refresh: bool = False,
    ) -> BottleneckReport:
        """
        Identify critical nodes and edges that are potential bottlenecks
        in the agent communication network.

        Parameters
        ----------
        threshold_degree : float
            Nodes with degree centrality below this are considered critical.
        threshold_betweenness : float
            Nodes with betweenness centrality above this are critical bottlenecks.

        Returns
        -------
        BottleneckReport
            List of critical nodes, critical edges, and alert coverage score.
        """
        if self._bottleneck_cache and not force_refresh:
            return self._bottleneck_cache

        metrics = self.compute_centrality()

        # Critical nodes: low degree OR high betweenness
        critical_nodes = []
        for node_id in self.G.nodes():
            deg = metrics.degree.get(node_id, 0)
            bet = metrics.betweenness.get(node_id, 0)
            if deg < threshold_degree or bet > threshold_betweenness:
                critical_nodes.append(node_id)

        # Critical edges: edges where source has low out-degree or target has low in-degree
        critical_edges = []
        for u, v in self.G.edges():
            out_deg = self.G.out_degree(u)
            in_deg = self.G.in_degree(v)
            if out_deg == 1 or in_deg == 1:
                critical_edges.append((u, v))

        # Alert coverage: fraction of nodes reachable from monitor via alerts
        alert_coverage = self._compute_alert_coverage()

        report = BottleneckReport(
            critical_nodes=critical_nodes,
            critical_edges=critical_edges,
            alert_coverage=alert_coverage,
        )

        self._bottleneck_cache = report
        return report

    def _compute_alert_coverage(self) -> float:
        """
        Compute what fraction of agents can receive an alert
        starting from the MonitorAgent.

        An alert travels along edges of type 'alert' or 'report'.
        Uses NetworkX descendants on a filtered subgraph for clarity.
        """
        if not self.G.has_node("MonitorAgent"):
            return 0.0

        # Build subgraph with only alert/report edges
        alert_edges = [
            (u, v) for u, v in self.G.edges()
            if self.G[u][v].get("type") in ("alert", "report")
        ]
        alert_subgraph = self.G.edge_subgraph(alert_edges).copy()

        # Get all nodes reachable from MonitorAgent via alert edges
        reachable = nx.descendants(alert_subgraph, "MonitorAgent")
        reachable.add("MonitorAgent")  # include the monitor itself

        total_agents = len(self.nodes)
        return len(reachable) / total_agents if total_agents > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # ALERT CASCADE ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_alert_cascade(self, source_agent: str) -> dict:
        """
        Trace the path of an alert from source_agent through the network.

        Returns the propagation tree and total nodes affected.
        """
        if not self.G.has_node(source_agent):
            return {"error": f"Agent {source_agent} not found"}

        affected = set()
        paths = {}
        stack = [(source_agent, [source_agent])]
        visited = set()

        while stack:
            current, path = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            affected.add(current)
            paths[current] = path

            for neighbor in self.G.successors(current):
                edge_data = self.G[current][neighbor]
                if edge_data.get("type") in ("alert", "report") and neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        return {
            "source": source_agent,
            "total_affected": len(affected),
            "affected_agents": list(affected),
            "propagation_paths": paths,
        }

    def detect_alert_fatigue(self) -> dict:
        """
        Detect potential alert fatigue conditions:
        - Agents receiving too many alerts from multiple sources
        - Monitor with excessive incoming alert volume

        Returns a dict with fatigue_score per agent (0=healthy, 1=exhausted).
        """
        fatigue_scores = {}

        for node_id in self.G.nodes():
            incoming_edges = list(self.G.in_edges(node_id, data=True))
            alert_edges = [e for e in incoming_edges if e[2].get("type") == "alert"]

            # Normalize: more than 3 incoming alerts per source = fatigue risk
            fatigue_scores[node_id] = min(1.0, len(alert_edges) / 3.0)

        return {
            "fatigue_scores": fatigue_scores,
            "at_risk_agents": [a for a, s in fatigue_scores.items() if s > 0.6],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> "AgentGraph":
        """
        Serialize graph to JSON file.

        Parameters
        ----------
        path : str | Path
            Output path (typically .json).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": {f"{k[0]}->{k[1]}": edge.to_dict() for k, edge in self.edges.items()},
            "centrality": self.compute_centrality().to_dict() if self._centrality_cache else None,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return self

    def load(self, path: str | Path) -> "AgentGraph":
        """
        Load graph from JSON file.

        Parameters
        ----------
        path : str | Path
            Input path (from a previous save() call).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the JSON structure is invalid or missing required fields.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Validate top-level structure
        if "nodes" not in data:
            raise ValueError("Invalid graph JSON: missing 'nodes' key")
        if "edges" not in data:
            raise ValueError("Invalid graph JSON: missing 'edges' key")

        # Required AgentNode fields
        required_node_fields = {"id", "type", "active", "reporting_frequency",
                                "alert_threshold", "alerts_sent", "alerts_received"}
        # Required AgentEdge fields
        required_edge_fields = {"source", "target", "type", "weight",
                                "frequency", "messages_sent", "last_communication"}

        # Rebuild graph from JSON
        self.G = nx.DiGraph()
        self.nodes = {}
        self.edges = {}

        for node_id, node_data in data["nodes"].items():
            # Validate node has required fields
            missing = required_node_fields - set(node_data.keys())
            if missing:
                raise ValueError(f"Node '{node_id}' missing required fields: {missing}")
            agent = AgentNode(**node_data)
            self.nodes[node_id] = agent
            self.G.add_node(node_id, **node_data)

        for edge_key, edge_data in data["edges"].items():
            # Validate edge key format
            if "->" not in edge_key:
                raise ValueError(f"Invalid edge key format '{edge_key}' — expected 'source->target'")
            src, tgt = edge_key.split("->")

            # Validate edge has required fields
            missing = required_edge_fields - set(edge_data.keys())
            if missing:
                raise ValueError(f"Edge '{edge_key}' missing required fields: {missing}")

            edge = AgentEdge(source=src, target=tgt, **{
                k: v for k, v in edge_data.items()
                if k not in ("source", "target")
            })
            self.edges[(src, tgt)] = edge
            self.G.add_edge(src, tgt, **edge_data)

        # Restore cached metrics (validate structure if present)
        if data.get("centrality"):
            c = data["centrality"]
            required_centrality = {"degree", "betweenness", "pagerank", "closeness"}
            missing = required_centrality - set(c.keys())
            if missing:
                raise ValueError(f"Centrality data missing required fields: {missing}")
            self._centrality_cache = CentralityMetrics(
                degree=c["degree"],
                betweenness=c["betweenness"],
                pagerank=c["pagerank"],
                closeness=c["closeness"],
                timestamp=c.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
            )

        self._log_event("graph_loaded", {"path": str(path)})
        return self

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Export edge list as a pandas DataFrame for analysis in DuckDB.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )
        rows = []
        for (src, tgt), edge in self.edges.items():
            row = {
                "source": src,
                "target": tgt,
                "type": edge.type,
                "weight": edge.weight,
                "frequency": edge.frequency,
                "messages_sent": edge.messages_sent,
                "last_communication": edge.last_communication,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZATION — GRAVIS (static)
    # ─────────────────────────────────────────────────────────────────────────

    def visualize_static(
        self,
        output_path: str | Path,
        figsize: tuple[int, int] = (12, 8),
    ) -> "AgentGraph":
        """
        Generate a static visualization using Gravis.

        Parameters
        ----------
        output_path : str | Path
            Path to save the figure (e.g., .png, .svg).
        """
        if not HAS_GRAVIS:
            print("[WARNING] gravis not installed. Install with: pip install gravis")
            print("Static visualization skipped.")
            return self

        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build Gravis-compatible data
        nodes_data = [
            {
                "id": node_id,
                "label": node_id,
                "color": self._node_color(node.type),
                "size": self._node_size(node_id),
            }
            for node_id, node in self.nodes.items()
        ]

        edges_data = [
            {
                "source": src,
                "target": tgt,
                "color": self._edge_color(edge.type),
                "size": edge.weight * 3,
                "label": edge.type,
            }
            for (src, tgt), edge in self.edges.items()
        ]

        graph_data = {"nodes": nodes_data, "edges": edges_data}
        fig = gravis.vis(graph_data, figsize=figsize, directed=True)

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[OK] Static visualization saved to: {output_path}")
        return self

    def _node_color(self, node_type: str) -> str:
        """Map agent type to color."""
        colors = {
            "vision": "#4CAF50",
            "geomechanical": "#2196F3",
            "gas": "#FF5722",
            "monitor": "#9C27B0",
        }
        return colors.get(node_type, "#9E9E9E")

    def _node_size(self, node_id: str) -> float:
        """Size node by betweenness centrality."""
        if self._centrality_cache:
            bet = self._centrality_cache.betweenness.get(node_id, 0)
            return 0.5 + bet * 5
        return 1.0

    def _edge_color(self, edge_type: str) -> str:
        """Map edge type to color."""
        colors = {
            "alert": "#F44336",
            "query": "#2196F3",
            "report": "#FFC107",
            "confirmation": "#4CAF50",
        }
        return colors.get(edge_type, "#9E9E9E")

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZATION — ALTAIR (interactive web)
    # ─────────────────────────────────────────────────────────────────────────

    def visualize_interactive(self, output_path: str | Path) -> "AgentGraph":
        """
        Generate an interactive HTML visualization using Altair + Vega-Lite.

        Parameters
        ----------
        output_path : str | Path
            Path to save the HTML file.
        """
        if not HAS_ALTAIR:
            print("[WARNING] altair not installed. Install with: pip install altair")
            print("Interactive visualization skipped.")
            return self

        if not HAS_GRAPHTOOL and not HAS_ALTAIR:
            print("[WARNING] altair and pandas are required for interactive visualization.")
            return self

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.compute_centrality()

        # Compute node positions from NetworkX spring layout
        pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
        pos_df = pd.DataFrame([
            {"id": node_id, "x": float(x), "y": float(y)}
            for node_id, (x, y) in pos.items()
        ])

        # Nodes DataFrame — with metrics + positions merged
        nodes_df = pd.DataFrame([
            {
                "id": node_id,
                "type": node.type,
                "active": node.active,
                "degree": metrics.degree.get(node_id, 0),
                "betweenness": metrics.betweenness.get(node_id, 0),
                "pagerank": metrics.pagerank.get(node_id, 0),
                "closeness": metrics.closeness.get(node_id, 0),
            }
            for node_id, node in self.nodes.items()
        ]).merge(pos_df, on="id", how="left")

        # Edges as midpoint markers between source and target positions
        # (x2/y2 encoding doesn't work reliably in Altair 6.x for separate datasets)
        edge_midpoints = []
        for (src, tgt), edge in self.edges.items():
            src_row = pos_df[pos_df["id"] == src]
            tgt_row = pos_df[pos_df["id"] == tgt]
            if len(src_row) > 0 and len(tgt_row) > 0:
                edge_midpoints.append({
                    "source": src,
                    "target": tgt,
                    "type": edge.type,
                    "weight": edge.weight,
                    "x": (src_row["x"].values[0] + tgt_row["x"].values[0]) / 2,
                    "y": (src_row["y"].values[0] + tgt_row["y"].values[0]) / 2,
                })
        edges_df = pd.DataFrame(edge_midpoints)

        # Base chart: agent nodes as circles
        base = alt.Chart(nodes_df).mark_circle(
            size=300,
            stroke="#333",
            strokeWidth=1.5,
        ).encode(
            x=alt.X("x:Q", title="", axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
            y=alt.Y("y:Q", title="", axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
            color=alt.Color("type:N", legend=alt.Legend(title="Agent Type")),
            size=alt.Size("betweenness:Q", legend=alt.Legend(title="Betweenness")),
            tooltip=["id:N", "type:N", "degree:Q", "pagerank:Q"],
        ).properties(
            width=700,
            height=500,
            title="Agent Communication Graph — Underground Mining",
        )

        # Edge midpoints as small squares between nodes
        edge_marks = (
            alt.Chart(edges_df)
            .mark_square(size=80, strokeWidth=1)
            .encode(
                x="x:Q",
                y="y:Q",
                color=alt.Color("type:N", legend=alt.Legend(title="Communication Type")),
                size=alt.Size("weight:Q", legend=alt.Legend(title="Weight")),
                tooltip=["source:N", "target:N", "type:N", "weight:Q"],
            )
        )

        chart = alt.layer(edge_marks, base).configure_view(
            stroke="#1a1a2e",
        ).configure_title(
            fontSize=16,
            font="Roboto",
        )

        chart.save(str(output_path), inline=True)
        print(f"[OK] Interactive visualization saved to: {output_path}")
        return self

    def visualize_d3(self, output_path: str | Path) -> "AgentGraph":
        """
        Generate an interactive D3.js-based force-directed graph visualization.

        Renders nodes as circles, edges as arrows with type-based coloring.
        Fully self-contained HTML (no external dependencies except CDN D3).

        Parameters
        ----------
        output_path : str | Path
            Path to save the HTML file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.compute_centrality()

        # Build nodes JSON
        nodes_json = [
            {
                "id": node_id,
                "tipo": node.type,
                "active": node.active,
                "degree": round(metrics.degree.get(node_id, 0), 4),
                "betweenness": round(metrics.betweenness.get(node_id, 0), 4),
                "pagerank": round(metrics.pagerank.get(node_id, 0), 4),
                "closeness": round(metrics.closeness.get(node_id, 0), 4),
            }
            for node_id, node in self.nodes.items()
        ]

        # Build links JSON
        links_json = [
            {
                "source": src,
                "target": tgt,
                "type": edge.type,
                "weight": edge.weight,
                "frequency": edge.frequency,
                "messages": edge.messages_sent,
            }
            for (src, tgt), edge in self.edges.items()
        ]

        node_color_map = {
            "vision": "#4CAF50",
            "geomechanical": "#2196F3",
            "gas": "#FF5722",
            "monitor": "#9C27B0",
        }
        edge_color_map = {
            "alert": "#F44336",
            "query": "#2196F3",
            "report": "#FFC107",
        }

        nodes_json_str = json.dumps(nodes_json, ensure_ascii=False)
        links_json_str = json.dumps(links_json, ensure_ascii=False)
        node_color_map_str = json.dumps(node_color_map)
        edge_color_map_str = json.dumps(edge_color_map)

        html_content = (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '<head>\n'
            '  <meta charset="UTF-8">\n'
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            '  <title>Agent Communication Graph — Underground Mining</title>\n'
            '  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>\n'
            '  <style>\n'
            '    * { box-sizing: border-box; margin: 0; padding: 0; }\n'
            '    body { font-family: Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; }\n'
            '    #graph-container { width: 100vw; height: 100vh; display: flex; flex-direction: column; }\n'
            '    header { padding: 14px 20px; background: #16213e; border-bottom: 1px solid #0f3460; '
            'display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }\n'
            '    header h1 { font-size: 15px; font-weight: 500; color: #e0e0e0; }\n'
            '    .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; }\n'
            '    .legend-item { display: flex; align-items: center; gap: 5px; }\n'
            '    .legend-dot { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }\n'
            '    .legend-line { width: 18px; height: 3px; border-radius: 2px; flex-shrink: 0; }\n'
            '    #graph { flex: 1; overflow: hidden; }\n'
            '    .node-label { font-size: 12px; fill: #e0e0e0; text-anchor: middle; '
            'pointer-events: none; font-family: Arial, sans-serif; }\n'
            '    .tooltip { position: absolute; background: #16213e; border: 1px solid #0f3460; '
            'border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #e0e0e0; '
            'pointer-events: none; display: none; line-height: 1.6; z-index: 100; }\n'
            '    .tooltip strong { color: #fff; }\n'
            '  </style>\n'
            '</head>\n'
            '<body>\n'
            '<div id="graph-container">\n'
            '  <header>\n'
            '    <h1>Agent Communication Graph — Underground Mining</h1>\n'
            '    <div class="legend">\n'
            '      <div class="legend-item"><div class="legend-dot" style="background:#4CAF50"></div>Vision</div>\n'
            '      <div class="legend-item"><div class="legend-dot" style="background:#2196F3"></div>Geomechanical</div>\n'
            '      <div class="legend-item"><div class="legend-dot" style="background:#FF5722"></div>Gas</div>\n'
            '      <div class="legend-item"><div class="legend-dot" style="background:#9C27B0"></div>Monitor</div>\n'
            '      <div class="legend-item"><div class="legend-line" style="background:#F44336"></div>Alert</div>\n'
            '      <div class="legend-item"><div class="legend-line" style="background:#2196F3"></div>Query</div>\n'
            '      <div class="legend-item"><div class="legend-line" style="background:#FFC107"></div>Report</div>\n'
            '    </div>\n'
            '  </header>\n'
            '  <div id="graph"></div>\n'
            '</div>\n'
            '<div class="tooltip" id="tooltip"></div>\n'
            '<script>\n'
            'const NODES = ' + nodes_json_str + ';\n'
            'const LINKS = ' + links_json_str + ';\n'
            'const NODE_COLORS = ' + node_color_map_str + ';\n'
            'const EDGE_COLORS = ' + edge_color_map_str + ';\n'
            '\n'
            'const container = document.getElementById("graph");\n'
            'const tooltip   = document.getElementById("tooltip");\n'
            '\n'
            'function getSize() {\n'
            '  return { w: container.clientWidth, h: container.clientHeight };\n'
            '}\n'
            '\n'
            'let { w: width, h: height } = getSize();\n'
            '\n'
            'const svg = d3.select("#graph")\n'
            '  .append("svg")\n'
            '  .attr("width", width)\n'
            '  .attr("height", height);\n'
            '\n'
            'const zoomG = svg.append("g");\n'
            '\n'
            'svg.call(\n'
            '  d3.zoom()\n'
            '    .scaleExtent([0.3, 5])\n'
            '    .on("zoom", ev => zoomG.attr("transform", ev.transform))\n'
            ');\n'
            '\n'
            '// One marker per edge type\n'
            'const defs = svg.append("defs");\n'
            '\n'
            'Object.entries(EDGE_COLORS).forEach(([edge_type, color]) => {\n'
            '  defs.append("marker")\n'
            '    .attr("id", `arrow-${edge_type}`)\n'
            '    .attr("viewBox", "0 -5 10 10")\n'
            '    .attr("refX", 10)\n'
            '    .attr("refY", 0)\n'
            '    .attr("markerWidth",  6)\n'
            '    .attr("markerHeight", 6)\n'
            '    .attr("orient", "auto")\n'
            '    .append("path")\n'
            '    .attr("fill", color)\n'
            '    .attr("d", "M0,-5L10,0L0,5");\n'
            '});\n'
            '\n'
            '// Node size scale\n'
            'const maxB      = Math.max(...NODES.map(n => n.betweenness), 0.001);\n'
            'const sizeScale = d3.scaleSqrt().domain([0, maxB]).range([12, 32]);\n'
            '\n'
            '// Force simulation\n'
            'const simulation = d3.forceSimulation(NODES)\n'
            '  .force("link",      d3.forceLink(LINKS).id(d => d.id).distance(160).strength(0.8))\n'
            '  .force("charge",    d3.forceManyBody().strength(-600))\n'
            '  .force("center",    d3.forceCenter(width / 2, height / 2))\n'
            '  .force("collision", d3.forceCollide().radius(d => sizeScale(d.betweenness) + 30));\n'
            '\n'
            '// Edges\n'
            'const link = zoomG.append("g")\n'
            '  .selectAll("path")\n'
            '  .data(LINKS)\n'
            '  .join("path")\n'
            '  .attr("stroke",         d => EDGE_COLORS[d.type] || "#666")\n'
            '  .attr("stroke-opacity", 0.75)\n'
            '  .attr("stroke-width",   d => Math.max(1.5, d.weight * 3))\n'
            '  .attr("fill",           "none")\n'
            '  .attr("marker-end",     d => `url(#arrow-${d.type})`);\n'
            '\n'
            '// Nodes\n'
            'const node = zoomG.append("g")\n'
            '  .selectAll("circle")\n'
            '  .data(NODES)\n'
            '  .join("circle")\n'
            '  .attr("r",            d => sizeScale(d.betweenness))\n'
            '  .attr("fill",         d => NODE_COLORS[d.tipo] || "#9E9E9E")\n'
            '  .attr("stroke",       "#fff")\n'
            '  .attr("stroke-width", 2)\n'
            '  .style("cursor",      "pointer")\n'
            '  .call(\n'
            '    d3.drag()\n'
            '      .on("start", (ev, d) => { if (!ev.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })\n'
            '      .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })\n'
            '      .on("end",   (ev, d) => { if (!ev.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })\n'
            '  );\n'
            '\n'
            '// Labels\n'
            'const label = zoomG.append("g")\n'
            '  .selectAll("text")\n'
            '  .data(NODES)\n'
            '  .join("text")\n'
            '  .attr("class", "node-label")\n'
            '  .text(d => d.id.replace("Agent", ""));\n'
            '\n'
            '// Link path with bidirectional detection and curvature\n'
            'function linkPath(d) {\n'
            '  const src = d.source, tgt = d.target;\n'
            '  const dx  = tgt.x - src.x;\n'
            '  const dy  = tgt.y - src.y;\n'
            '  const len = Math.sqrt(dx * dx + dy * dy) || 1;\n'
            '\n'
            '  const r_src = sizeScale(src.betweenness);\n'
            '  const r_tgt = sizeScale(tgt.betweenness);\n'
            '\n'
            '  // Start and end at node border\n'
            '  const sx = src.x + (dx / len) * r_src;\n'
            '  const sy = src.y + (dy / len) * r_src;\n'
            '  const ex = tgt.x - (dx / len) * (r_tgt + 6);\n'
            '  const ey = tgt.y - (dy / len) * (r_tgt + 6);\n'
            '\n'
            '  // Perpendicular normal\n'
            '  const nx = -dy / len;\n'
            '  const ny =  dx / len;\n'
            '\n'
            '  // Detect bidirectional edge\n'
            '  const isPair = LINKS.some(l =>\n'
            '    (l.source.id || l.source) === (d.target.id || d.target) &&\n'
            '    (l.target.id || l.target) === (d.source.id || d.source)\n'
            '  );\n'
            '  const curvature = isPair ? 40 : 0;\n'
            '\n'
            '  const mx = (sx + ex) / 2 + nx * curvature;\n'
            '  const my = (sy + ey) / 2 + ny * curvature;\n'
            '\n'
            '  return curvature === 0\n'
            '    ? `M${sx},${sy} L${ex},${ey}`\n'
            '    : `M${sx},${sy} Q${mx},${my} ${ex},${ey}`;\n'
            '}\n'
            '\n'
            '// Tick\n'
            'simulation.on("tick", () => {\n'
            '  link.attr("d", linkPath);\n'
            '  node.attr("cx", d => d.x).attr("cy", d => d.y);\n'
            '  label\n'
            '    .attr("x", d => d.x)\n'
            '    .attr("y", d => d.y + sizeScale(d.betweenness) + 16);\n'
            '});\n'
            '\n'
            '// Tooltips — nodes\n'
            'node\n'
            '  .on("mouseover", (ev, d) => {\n'
            '    tooltip.style.display = "block";\n'
            '    tooltip.innerHTML = `<strong>${d.id}</strong><br>Type: ${d.type}<br>Active: ${d.active}<br>Degree: ${d.degree.toFixed(3)}<br>Betweenness: ${d.betweenness.toFixed(3)}<br>PageRank: ${d.pagerank.toFixed(4)}<br>Closeness: ${d.closeness.toFixed(3)}`;\n'
            '    d3.select(ev.currentTarget).attr("stroke-width", 3);\n'
            '  })\n'
            '  .on("mousemove", ev => {\n'
            '    tooltip.style.left = (ev.pageX + 14) + "px";\n'
            '    tooltip.style.top  = (ev.pageY - 28) + "px";\n'
            '  })\n'
            '  .on("mouseout", ev => {\n'
            '    tooltip.style.display = "none";\n'
            '    d3.select(ev.currentTarget).attr("stroke-width", 2);\n'
            '  });\n'
            '\n'
            '// Tooltips — edges\n'
            'link\n'
            '  .on("mouseover", (ev, d) => {\n'
            '    tooltip.style.display = "block";\n'
            '    const sid = d.source.id || d.source;\n'
            '    const tid = d.target.id || d.target;\n'
            '    tooltip.innerHTML = `<strong>${sid} → ${tid}</strong><br>Type: ${d.type}<br>Weight: ${d.weight}<br>Frequency: ${d.frequency}<br>Messages: ${d.messages}`;\n'
            '    d3.select(ev.currentTarget).attr("stroke-opacity", 1).attr("stroke-width", d.weight * 4 + 1);\n'
            '  })\n'
            '  .on("mousemove", ev => {\n'
            '    tooltip.style.left = (ev.pageX + 14) + "px";\n'
            '    tooltip.style.top  = (ev.pageY - 28) + "px";\n'
            '  })\n'
            '  .on("mouseout", (ev, d) => {\n'
            '    tooltip.style.display = "none";\n'
            '    d3.select(ev.currentTarget).attr("stroke-opacity", 0.75).attr("stroke-width", Math.max(1.5, d.weight * 3));\n'
            '  });\n'
            '\n'
            '// Resize\n'
            'window.addEventListener("resize", () => {\n'
            '  const { w, h } = getSize();\n'
            '  svg.attr("width", w).attr("height", h);\n'
            '  simulation.force("center", d3.forceCenter(w / 2, h / 2)).alpha(0.3).restart();\n'
            '});\n'
            '</script>\n'
            '</body>\n'
            '</html>'
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[OK] D3.js visualization saved to: {output_path}")
        return self

    def _compute_force_directed_positions(self) -> list[dict]:
        """Kept for backward compatibility. Positions are now computed inline."""
        pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
        return [
            {"id": node_id, "x": float(x), "y": float(y)}
            for node_id, (x, y) in pos.items()
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # EVENT LOGGING
    # ─────────────────────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, data: dict):
        """Log an event to the in-memory event log."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event_type": event_type,
            "data": data,
        }
        self._event_log.append(entry)

    def get_event_log(self) -> list[dict]:
        """Return the in-memory event log."""
        return self._event_log.copy()

    def export_event_log_parquet(self, output_path: str | Path):
        """Export event log to Parquet for DuckDB querying."""
        if not HAS_ALTAIR:
            print("[WARNING] pandas required for Parquet export. Install altair.")
            return self

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self._event_log)
        df.to_parquet(output_path, index=False)
        print(f"[OK] Event log saved to: {output_path}")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # DUCKDB QUERIES
    # ─────────────────────────────────────────────────────────────────────────

    def query_events(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query on the event log using DuckDB.

        Parameters
        ----------
        sql : str
            DuckDB SQL query on the 'event_log' table.

        Example
        -------
        >>> graph.query_events(
        ...     "SELECT event_type, COUNT(*) as count "
        ...     "FROM event_log GROUP BY event_type ORDER BY count DESC"
        ... )
        """
        if not HAS_DUCKDB:
            raise ImportError("DuckDB is required for queries. Install with: pip install duckdb")

        df = pd.DataFrame(self._event_log)
        conn = duckdb.connect()
        conn.register("event_log", df)
        result = conn.sql(sql).fetchdf()
        conn.close()
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the graph state."""
        metrics = self.compute_centrality()
        bottlenecks = self.detect_bottlenecks()

        lines = [
            "=" * 60,
            "[AGENT GRAPH] Resumen del Grafo de Comunicación",
            "=" * 60,
            f"Nodos activos  : {sum(1 for n in self.nodes.values() if n.activo)} / {len(self.nodes)}",
            f"Aristas totales: {len(self.edges)}",
            "",
            "── Métricas de Centralidad ──",
        ]

        for metric_name in ("degree", "betweenness", "pagerank", "closeness"):
            metric_dict = getattr(metrics, metric_name)
            top_node, top_val = max(metric_dict.items(), key=lambda x: x[1])
            lines.append(f"  {metric_name:12s}: más central = {top_node} ({top_val:.4f})")

        lines.extend([
            "",
            "── Detección de Cuellos de Botella ──",
            f"  Nodos críticos : {bottlenecks.nodos_criticos or 'ninguno'}",
            f"  Aristas críticas: {bottlenecks.aristas_criticas or 'ninguna'}",
            f"  Cobertura alerta: {bottlenecks.cobertura_alerta:.2%}",
        ])

        lines.append("=" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Agent Interaction Graph — Minería Subterránea IA"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build and display the initial agent graph",
    )
    parser.add_argument(
        "--load", type=str,
        help="Load graph from JSON file",
    )
    parser.add_argument(
        "--save", type=str,
        help="Save graph to JSON file",
    )
    parser.add_argument(
        "--visualize-static", type=str,
        help="Save static visualization to path",
    )
    parser.add_argument(
        "--visualize-interactive", type=str,
        help="Save interactive HTML visualization to path (D3.js-based)",
    )
    parser.add_argument(
        "--d3", type=str,
        help="Save D3.js force-directed graph to path",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print graph summary",
    )

    args = parser.parse_args()

    graph = AgentGraph()

    if args.load:
        graph.load(args.load)
    else:
        graph.build_initial_graph()

    if args.save:
        graph.save(args.save)

    if args.visualize_static:
        graph.visualize_static(args.visualize_static)

    if args.visualize_interactive:
        graph.visualize_interactive(args.visualize_interactive)

    if args.d3:
        graph.visualize_d3(args.d3)

    if args.summary:
        print(graph.summary())

    if not any(vars(args).values()):
        parser.print_help()
