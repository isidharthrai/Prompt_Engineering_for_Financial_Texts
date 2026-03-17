# Adaptive Prompt Engineering for Financial Text Understanding

**MSc AI & ML Thesis — Liverpool John Moores University, UK · March 2026**  
**Author:** Sidharth Rai &nbsp;|&nbsp; **Student ID:** 1187305

---

This repository contains the implementation code and results for my MSc thesis on evaluating prompt engineering strategies — Zero-Shot, Few-Shot, Chain-of-Thought, and Tree-of-Thought — applied to three financial NLP tasks: Sentiment Analysis, Risk Assessment, and Insight Generation. Experiments were conducted across three open-source language models (Llama 3.1:8B, Qwen3:8B, DeepSeek-R1:8B) using the Financial PhraseBank dataset.

---

## Dataset

**Financial PhraseBank v1.0** — Malo et al. (2014)  
Download: [https://huggingface.co/datasets/takala/financial_phrasebank/blob/main/data/FinancialPhraseBank-v1.0.zip](https://huggingface.co/datasets/takala/financial_phrasebank/blob/main/data/FinancialPhraseBank-v1.0.zip)  
Place the extracted files inside `DatasetAnalysis_FinancialPhraseBank/FinancialPhraseBank-v1.0/` before running any notebooks.

## Requirements

```bash
pip install -r requirements.txt
```

> **Note:** Experiments use [Ollama](https://ollama.com/) for local LLM inference. Run `ollama pull llama3.1:8b`, `ollama pull qwen3:8b`, and `ollama pull deepseek-r1:8b` before executing the notebooks.