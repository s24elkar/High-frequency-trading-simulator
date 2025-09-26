# Visual Assets Index

This index collects entry points for exploring the simulator visually—whether
through dependency graphs, simulation dashboards, or neural Hawkes notebooks.

## Dependency Graphs
- **CMake targets**: `docs/graphs/cmake_graph.dot` (render with `dot -Tpng`)
- **Python imports**: `docs/graphs/python_imports.dot`
- Generate fresh graphs via `python scripts/generate_dependency_graphs.py`

## Simulation Dashboards
- `python/timeline_dashboard.py` — produce timeline plots combining events,
  intensities, and binned counts.
- `python/viz.py` — standalone helpers for intensity and autocorrelation plots.
- Demo notebook: `docs/neural_hawkes_tutorial.ipynb`

## Interactive Tools
- Streamlit app (`python/streamlit_app.py`) for parameter sweeps and live plots.
- Command-line demo (`python/demo.py`) emits figures under `docs/images/`.

## Architecture Overview
- Mermaid diagram: `docs/architecture.mmd`
- The primer (`docs/research_primer.md`) documents data flow and design choices.

Keep this index updated as you add new visual artefacts or dashboards so
teammates can quickly discover them.
