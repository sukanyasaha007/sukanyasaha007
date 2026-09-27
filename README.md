# Sukanya Saha

ML engineer working on LLM post-training with verifiable rewards, inference systems, and GPU performance.
Currently applying RL and fine-tuning to hardware verification at Siemens; previously ML infrastructure at Intel.

## Current work — Siemens (2025–present)

- **RL post-training with verifiable rewards.** RLVR, DPO, and online GRPO (TRL) with vLLM-served rollouts,
  using staged simulator compile and test outcomes as the reward signal for RTL verification and debug tasks.
- **Domain fine-tuning at 32B scale.** QLoRA SFT of Qwen2.5-Coder-32B for SystemVerilog RTL generation:
  first-pass accuracy 62% → 86% at half the average tool calls on internal evals.
- **Agent evaluation.** Built the evaluation harness for an LLM auto-fix agent on NVIDIA's CVDP benchmark;
  ablations isolated self-verification as the change that drove the accuracy gain.
- **Agent infrastructure.** MCP platform exposing 30+ EDA tools (compile, simulate, lint, coverage) to LLM agents,
  plus a hybrid RAG pipeline with 3.1x lower retrieval latency.

## Previously

- **Intel (2022–2025)** — Multi-GPU fine-tuning of Qwen (LoRA, QLoRA, MoE) using model and data parallelism,
  tensor sharding, and gradient checkpointing. Multi-node vLLM serving with KV caching.
  Computer-vision anomaly detection for chip design blocks, cutting QA test time from 6 months to 5 days.
- **Google (2021–2022)** — Research apprentice: GAN-based dual-network apparel recommendation (DeepFashion).
- **HSBC, Cognizant (2016–2020)** — Applied ML: segmentation, churn and attrition forecasting, NLP.

## Learning in public

Reimplementing core papers from scratch, each verified numerically against a reference implementation.

| Project | Description |
|---|---|
| [paper-implementations](https://github.com/sukanyasaha007/paper-implementations) | GPT-2, Transformer, ViT, FlashAttention (Triton) |

<!-- Add as shipped: grpo-jax (flagship), LoRA/DPO rebuilds -->

## Tools

**Training and post-training:** PyTorch, JAX, TRL, PEFT, Hugging Face Transformers
**Inference and serving:** vLLM, TensorRT, ONNX Runtime
**GPU and performance:** CUDA C++, Triton, NCCL, ROCm, Nsight Systems, PyTorch Profiler

## Writing

<!-- Link technical posts here once published -->

## Education

MS Computer Science, University of Colorado Boulder

## Contact

[LinkedIn](https://www.linkedin.com/in/sukanyasaha007/) · sukanyasaha007@gmail.com
