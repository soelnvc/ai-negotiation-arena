<div align="center">

# 🤝 AI Negotiation Arena

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-success.svg?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Phase-1_Complete-purple.svg?style=for-the-badge)

*A deterministic, multi-agent reinforcement learning environment evaluating the social intelligence, long-term planning, and negotiation capabilities of LLMs.*

</div>

---

## 🏗️ Architecture & Engineering Highlights



This environment was designed with strict MLOps and AI engineering principles to ensure reproducibility and stability during evaluation:

* 🛡️ **Pydantic Data Contracts:** Action and Observation payloads are strictly validated to prevent malformed agent outputs from crashing the simulation.
* 🧮 **Deterministic Economy:** Floating-point math errors are eliminated using integer-based value creation mechanics (e.g., successful trades generate a strict 20% systemic resource gain).
* 📊 **Telemetry-First Grading:** Evaluation is handled by deterministic graders that rely on hard event counters (telemetry) rather than fragile heuristics.
* 🔌 **Graceful Degradation:** The task registry includes dynamic `getattr` fallbacks to ensure compatibility across different versions of the OpenEnv framework.

## ⚖️ The Rules of the Arena

Agents must survive for 100 steps while managing two distinct matrices: **Capital** (Resources) and **Trust** (Reputation). The positive-sum economy rewards cooperation but leaves agents vulnerable to exploitation.

| Action | Execution Cost | Economic Impact | Social Impact (Trust) |
| :--- | :--- | :--- | :--- |
| 🌾 **Gather** | 1 Turn | +10 Resources | Neutral |
| 🤝 **Trade** | 1 Turn | Mutually beneficial (+20%) | +0.05 (Increase) |
| 🕊️ **Ally** | 1 Turn | Builds Alliance Streak | +0.10 (Symmetric Increase) |
| 🗡️ **Betray** | 1 Turn | Steal up to 15 Resources | -0.15 (Catastrophic Drop) |

## 🏆 Evaluation Tasks

The environment exposes three officially registered tasks for OpenEnv benchmarking, featuring a meaningful difficulty progression:

> **🟢 Task 1: Resource Scavenger (Easy)**
> * **Goal:** Survive 50 steps and gather a baseline of 50 resources.
> * **Evaluates:** Basic environment comprehension and action-space navigation.

> **🟡 Task 2: Honest Trader (Medium)**
> * **Goal:** Successfully complete 3 trades without ever initiating a betrayal.
> * **Evaluates:** Cooperation and resisting the temptation of short-term defection rewards.

> **🔴 Task 3: Master Negotiator (Hard)**
> * **Goal:** Maintain alliance stability (30+ turn streak) under systemic resource pressure while accumulating wealth.
> * **Evaluates:** Advanced social intelligence, long-term multi-variable optimization, and scarcity adaptation.

## 🚀 Project Roadmap

- [x] **Phase 1: Environment Definition** (State models, step logic, economy, trust matrix)
- [x] **Phase 1b: Task Registration & Graders** (Telemetry trackers, 0.0-1.0 scoring)
- [ ] **Phase 2: Baseline Agent** (LLM integration and prompt engineering)
- [ ] **Phase 3: Multi-Agent Evaluation**

## 🧪 Testing

The environment logic and grader determinism are fully covered by a robust test suite.

```bash
# Run the test suite to verify the Gym's integrity
python -m pytest
