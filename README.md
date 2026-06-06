# Grafo Social — Minería Subterránea IA

Módulo de análisis sistémico de patrones de comunicación entre agentes LangGraph para monitoreo de seguridad en minería subterránea.

---

## Estructura del Proyecto

```
grafo-social-mineria-subterranea/
├── agent_graph.py          # Módulo principal (~1380 líneas)
├── data/
│   └── agent_graph.json    # Grafo serializado en JSON
├── output/
│   ├── agent_graph_d3.html # Visualización D3 interactiva
│   └── README.md           # Instrucciones de la visualización
├── tests/
│   └── test_agent_graph.py  # Suite de tests (49 tests)
├── venv/                    # Entorno virtual Python
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Este archivo
```

---

## Requisitos

- Python 3.10+
- Entorno virtual (venv)

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/SebastianLizarazo/grafo-social-mineria-subterranea.git
cd grafo-social-mineria-subterranea

# Crear entorno virtual
python -m venv venv

# Instalar dependencias
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Uso básico

```python
from agent_graph import AgentGraph

# Construir el grafo inicial con la topología por defecto
g = AgentGraph()
g.build_initial_graph()

# Calcular métricas de centralidad
metrics = g.compute_centrality()
print(metrics.betweenness)  # {'VisionAgent': 0.0, 'GeomechanicalAgent': 0.0, ...}

# Detectar cuellos de botella
report = g.detect_bottlenecks()
print(report.critical_nodes)    # nodos críticos
print(report.alert_coverage)     # cobertura de alertas

# Analizar cascada de alertas desde un agente
result = g.analyze_alert_cascade("GasAgent")
print(result["total_affected"])

# Serializar a JSON
g.save("data/agent_graph.json")

# Cargar desde JSON
g2 = AgentGraph().load("data/agent_graph.json")

# Exportar a DataFrame (para análisis con DuckDB/SQL)
df = g.to_dataframe()
print(df.head())
```

---

## Los 4 Agentes

| Agente | Rol | Frecuencia | Umbral |
|--------|-----|------------|--------|
| `VisionAgent` | Detección de grietas/defectos estructurales | cada 10 min | 0.7 |
| `GeomechanicalAgent` | Monitoreo de sensores geomecánicos | cada 5 min | 0.5 |
| `GasAgent` | Monitoreo ambiental de gases | cada 1 min | 0.3 |
| `MonitorAgent` | Orquestación y decisión central | cada 1 min | 0.5 |

### Topología por defecto

```
GasAgent         → MonitorAgent  (alert, peso 1.0)
GeomechanicalAgent → MonitorAgent  (alert, peso 0.8)
VisionAgent      → MonitorAgent  (report, peso 0.6)
MonitorAgent     → VisionAgent  (query)
MonitorAgent     → GasAgent     (query)
MonitorAgent     → GeomechanicalAgent (query)
```

---

## Tests

Ejecutar la suite completa:

```bash
.\venv\Scripts\python.exe -m pytest tests/test_agent_graph.py -v
```

**49 tests** cubriendo:
- Construcción del grafo
- Métricas de centralidad (degree, betweenness, PageRank, closeness)
- Detección de cuellos de botella
- Análisis de cascada de alertas
- Serialización (save/load roundtrip)
- Validación de errores
- Event logging
- Exportación a DataFrame

---

## Visualización D3 Interactiva

El archivo `output/agent_graph_d3.html` es una visualización force-directed auto-contenida. Requiere servirse por HTTP para funcionar (D3 se carga desde CDN y el navegador bloquea módulos con `file://`).

### Cómo levantarlo

```bash
# Desde la raíz del proyecto:
.\venv\Scripts\python.exe -m http.server 8080
```

Abrir: **[http://localhost:8080/output/agent_graph_d3.html](http://localhost:8080/output/agent_graph_d3.html)**

### Regenerar la visualización

Si hiciste cambios en el grafo:

```bash
.\venv\Scripts\python.exe -c "
from agent_graph import AgentGraph
g = AgentGraph()
g.build_initial_graph()
g.compute_centrality()
g.visualize_d3('output/agent_graph_d3.html')
print('Listo: output/agent_graph_d3.html')
"
```

### Interacciones

- **Drag** — arrastrar nodos para reubicarlos
- **Zoom/Pan** — scroll del mouse para zoom, arrastrar el fondo para panear
- **Tooltip** — pasar el mouse sobre un nodo o arista para ver métricas detalladas
- **Nodos** — el tamaño refleja centralidad de intermediación (más grande = más crítico)
- **Colores de nodos** — verde (visión), azul (geomecánico), naranja (gas), violeta (monitor)
- **Colores de aristas** — rojo (alert), azul (query), amarillo (report)

---

## API Referencia

### Clases principales

| Clase | Descripción |
|-------|-------------|
| `AgentGraph` | Grafo de comunicación entre agentes |
| `AgentNode` | Nodo representando un agente |
| `AgentEdge` | Arista dirigida de comunicación |
| `CentralityMetrics` | Métricas de centralidad (degree, betweenness, PageRank, closeness) |
| `BottleneckReport` | Reporte de cuellos de botella detectados |

### Métodos de AgentGraph

| Método | Descripción |
|--------|-------------|
| `build_initial_graph()` | Construye la topología por defecto de 4 agentes y 6 aristas |
| `compute_centrality(force_refresh=False)` | Calcula métricas de centralidad (cacheadas) |
| `detect_bottlenecks()` | Identifica nodos y aristas críticas |
| `analyze_alert_cascade(source_agent)` | Traza la propagación de una alerta |
| `detect_alert_fatigue()` | Detecta condiciones de fatiga de alertas |
| `save(path)` | Serializa el grafo a JSON |
| `load(path)` | Carga el grafo desde JSON |
| `to_dataframe()` | Exporta las aristas como pandas DataFrame |
| `visualize_d3(path)` | Genera visualización HTML interactiva con D3.js |
| `update_edge(source, target, **kwargs)` | Actualiza atributos de una arista |
| `deactivate_node(node_id)` | Marca un agente como inactivo |
| `remove_edge(source, target)` | Elimina una arista del grafo |

---

## Notas técnicas

- **Caching**: las métricas de centralidad y el reporte de cuellos de botella se cachean automáticamente. Usá `force_refresh=True` para recomputar.
- **Fallbacks**: graph-tool está soportado como backend opcional para grafos grandes (más rápido). NetworkX es el default.
- **Validación**: `load()` valida la estructura del JSON y lanza `ValueError` con mensajes claros si falta algo.
- **Event log**: todas las operaciones se registran en `_event_log` en memoria.