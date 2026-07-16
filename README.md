# GenAI-Based EdTech Platform

A modular educational AI platform developed in independently testable phases. This repository is organized as a platform root so later capabilities can be added without mixing their code, dependencies, data, or evaluation artifacts with Phase 1.

## Implemented phase

| Module | Status | Purpose |
|---|---|---|
| [`computer-science-rag`](computer-science-rag/README.md) | Phase 1 implemented | Grounded Cambridge O Level 2210 and A Level 9618 Computer Science question answering |

The Phase 1 module contains its own source code, configuration, tests, evaluation framework, application entry points, dependency file, and detailed architecture documentation.

## Repository layout

```text
GenAI-Based-Edtech-Platform/
├── README.md
├── .gitignore
└── computer-science-rag/
    ├── README.md
    ├── configs/
    ├── evaluation/
    ├── scripts/
    ├── src/
    ├── streamlit_app/
    └── tests/
```

Raw source documents, generated indexes, model caches, evaluation results, local environments, and secrets are intentionally excluded from the public repository.

## Phase 1 quick start

```powershell
git clone https://github.com/BasitSol/GenAI-Based-Edtech-Platform.git
cd GenAI-Based-Edtech-Platform\computer-science-rag
```

Continue with the complete [Phase 1 installation, architecture, build, test, and evaluation guide](computer-science-rag/README.md).

## Development convention

Each future phase should be added as a separate top-level module with its own README, runtime configuration, dependency boundaries, tests, and validation procedure. Repository-wide exclusions remain in the root `.gitignore`.
