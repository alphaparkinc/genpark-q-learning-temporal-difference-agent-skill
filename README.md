# genpark-q-learning-temporal-difference-agent-skill

> Tabular Q-learning temporal difference agent with epsilon-greedy exploration and Bellman state-action updates.

Part of the **GenPark AI Agent Skills Matrix**. Production-ready, zero external dependencies, native Python 3.9+ standard library.

## Architecture

```mermaid
flowchart TD
    A[Environment State / Reward] --> B[RL Policy & Value Estimator]
    B --> C[Bellman / UCT Tree Search Kernel]
    C --> D[Optimal Action Selection]
    D --> E[MCP Protocol Endpoint]
```

## Features
- **Zero Third-Party Dependencies**: Pure Python standard library (`math`, `random`).
- **Optimal Decision Making**: Value iteration convergence, Q-learning TD updates, and UCT search.
- **Native MCP Protocol Support**: Integrated JSON-RPC 2.0 stdio server ready for Claude Desktop, Cursor, and Windsurf.

## Installation

```bash
pip install genpark-q-learning-temporal-difference-agent-skill
```

Or clone directly:

```bash
git clone https://github.com/alphaparkinc/genpark-q-learning-temporal-difference-agent-skill.git
cd genpark-q-learning-temporal-difference-agent-skill
python example_usage.py
```

## Quick Start

```python
from client import *
# Refer to example_usage.py for end-to-end execution
```

## Model Context Protocol (MCP) Setup

Add to your `claude_desktop_config.json` or `cursor.json`:

```json
{
  "mcpServers": {
    "genpark-q-learning-temporal-difference-agent-skill": {
      "command": "python",
      "args": ["-m", "genpark-q-learning-temporal-difference-agent-skill.mcp_server"]
    }
  }
}
```

## License
MIT License. Copyright (c) 2026 AlphaPark Inc. & Alpha-Park.
