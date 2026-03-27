# Project Plan: AI Negotiation Arena (OpenEnv)

## 1. Project Overview
**Goal:** Build a multi-agent Reinforcement Learning (RL) environment where agents must negotiate, cooperate, and strategically manage trust to maximize rewards. This models real-world social intelligence and resource management.

## 2. OpenEnv Spec Compliance
- **Architecture:** 3-Component Pattern (Models, Server/Environment, Client).
- **API:** Implement full `reset()`, `step(action)`, and `state()` methods.
- **Types:** Use Pydantic for `ArenaAction`, `ArenaObservation`, and `ArenaState`.

## 3. Core Mechanics & Reward System
| Action | Reward Signal | Description |
| :--- | :--- | :--- |
| Survival | +50 | Awarded if the agent remains active until the end of the episode. |
| Resource Gain | +10 | Small reward for acquiring currency/energy. |
| Successful Trade| +20 | Reward for mutually beneficial exchanges. |
| Alliance | +30 | Forming a stable partnership. |
| Betrayal | +40 | High-risk, high-reward strategic backstabbing. |
| Betrayed | -30 | Penalty for failing to detect deception. |
| Conflict Loss | -50 | Penalty for losing a direct resource battle. |

## 4. Required Tasks & Graders
Each task must be scored 0.0 to 1.0 by a deterministic programmatic grader.
1. **Task 1 (Easy) - Resource Scavenger:** Agent must survive 50 steps solo by gathering resources.
2. **Task 2 (Medium) - Honest Trader:** Agent must successfully complete 3 trades without initiating betrayal.
3. **Task 3 (Hard) - Master Negotiator:** Agent must maintain an alliance for 30+ steps despite declining global resources.

## 5. Technical Requirements
- **Containerization:** Must include a `Dockerfile` that builds and runs locally with `docker build`.
- **Deployment:** Target platform is Hugging Face Spaces.
- **Baseline:** Include a `baseline.py` script using OpenAI API client to generate reproducible scores.
- **Validation:** Must pass `openenv validate` before submission.

## 6. Development Timeline
- **Phase 1:** Define `models.py` (Data Contracts).
- **Phase 2:** Implement `server/environment.py` logic and Reward System.
- **Phase 3:** Create `client.py` for WebSocket communication.
- **Phase 4:** Write 3 Grader scripts and the Baseline Inference script.
- **Phase 5:** Final documentation (README.md) and HF Space deployment.