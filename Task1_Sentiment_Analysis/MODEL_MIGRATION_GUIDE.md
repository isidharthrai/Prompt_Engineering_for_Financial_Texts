# Model Migration Guide: GPT OSS/Llama-3.3 → Mixtral/Llama-3.1/FinBERT

## Overview

All experiments (E1-E12) have been updated to use:

- **E1, E4, E7, E10**: Mixtral-8x7B-32768 (via Groq)
- **E2, E5, E8, E11**: Llama-3.1-70B-Versatile (via Groq)
- **E3, E6, E9, E12**: FinBERT (ProsusAI/finbert via HuggingFace)

## Changes Made

### ✅ Completed Updates

#### 1. **Zero-Shot Notebook** (`E1_E2_E3_zero_shot_sentiment_All_agree.ipynb`)

- E1: Mixtral-8x7B (replaces GPT OSS 20B)
- E2: Llama-3.1-70B (replaces GPT OSS 120B)
- E3: FinBERT (replaces Llama-3.3-70B)
- Added `call_finbert()` function
- Added FinBERT model loading in setup
- Updated all CSV file names

#### 2. **Few-Shot Notebook** (`E4_E5_E6_few_shot_sentiment.ipynb`)

- E4: Mixtral-8x7B
- E5: Llama-3.1-70B
- E6: FinBERT
- Model names updated throughout
- **Note**: FinBERT doesn't use few-shot examples (pre-trained only)

#### 3. **Chain-of-Thought Notebook** (`E7_E8_E9_cot_sentiment.ipynb`)

- E7: Mixtral-8x7B
- E8: Llama-3.1-70B
- E9: FinBERT
- Model names updated throughout
- **Note**: FinBERT cannot do step-by-step reasoning (outputs sentiment directly)

#### 4. **Tree-of-Thought Notebook** (`E10_E11_E12_tot_sentiment.ipynb`)

- E10: Mixtral-8x7B
- E11: Llama-3.1-70B
- E12: FinBERT
- Model names updated throughout
- **Note**: FinBERT cannot do multi-path exploration (single forward pass)

## ⚠️ Important Considerations

### FinBERT Limitations Across Prompting Strategies

**Key Point**: FinBERT is a **fine-tuned BERT model** for financial sentiment, NOT a generative LLM. It:

- ✅ Provides sentiment classification (positive/negative/neutral)
- ✅ Returns confidence scores
- ❌ Cannot follow prompts or instructions
- ❌ Cannot do reasoning (CoT/ToT)
- ❌ Cannot use few-shot examples

### How FinBERT Works in Each Approach

#### **Zero-Shot (E3)** ✅ **VALID**

FinBERT's natural mode - no prompts needed, just raw classification:

```python
def call_finbert(sentence):
    result = finbert_pipeline(sentence[:512])
    return {
        "sentiment": result[0]["label"].lower(),
        "confidence": result[0]["score"],
        "rationale": f"FinBERT classification: {result[0]['label']}"
    }
```

#### **Few-Shot (E6)** ⚠️ **CONCEPTUALLY INVALID**

FinBERT cannot use few-shot examples, but we include it for comparison:

- Still just runs raw classification
- Examples in prompt are **ignored** (FinBERT doesn't read prompts)
- Performance = same as Zero-Shot
- **Academic Value**: Shows "pre-trained fine-tuned model vs prompt engineering"

#### **Chain-of-Thought (E9)** ❌ **INVALID APPROACH**

FinBERT cannot do step-by-step reasoning:

- CoT prompt completely ignored
- No reasoning steps generated
- Just returns sentiment label
- **Alternative**: Report FinBERT as "N/A - not applicable" or omit

#### **Tree-of-Thought (E12)** ❌ **INVALID APPROACH**

FinBERT cannot explore multiple hypothesis paths:

- Multi-path reasoning impossible
- Path scores = N/A
- **Alternative**: Report as "N/A" or omit

### Recommended Approach for Assignment

**Option 1: Full Comparison (Recommended)**
Include FinBERT in all experiments with clear disclaimers:

- **E3 (Zero-Shot)**: Valid - baseline FinBERT performance
- **E6 (Few-Shot)**: Note "FinBERT ignores examples, shows pre-training advantage"
- **E9 (CoT)**: Note "FinBERT cannot reason, shows raw classification limit"
- **E12 (ToT)**: Note "FinBERT single-path only, baseline comparison"

**Insight**: Shows that domain-specific fine-tuning (FinBERT) may outperform prompt engineering on general models.

**Option 2: Hybrid Approach**

- Run FinBERT only for E3 (Zero-Shot)
- For E6, E9, E12: Use alternative models
  - **Option A**: Add `gpt-4o-mini` (best JSON compliance)
  - **Option B**: Add `gemma2-9b-it` (good instruction following)
  - **Option C**: Use same FinBERT results as E3 (mark as duplicate)

**Option 3: Replace FinBERT Entirely**
Use three LLM models that support all prompting strategies:

- E1/E4/E7/E10: Mixtral-8x7B
- E2/E5/E8/E11: Llama-3.1-70B
- E3/E6/E9/E12: **gpt-4o-mini** or **claude-3-5-haiku**

## Implementation Details

### Required Changes for FinBERT Integration

#### 1. Add FinBERT Setup to All Notebooks

```python
from transformers import pipeline
import torch

# Load FinBERT
print("Loading FinBERT model...")
device = 0 if torch.cuda.is_available() else -1
finbert_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    device=device
)
print(f"✓ FinBERT loaded on {'GPU' if device == 0 else 'CPU'}")
```

#### 2. Add FinBERT Inference Function

```python
def call_finbert(sentence):
    """FinBERT classification - no prompt needed"""
    try:
        result = finbert_pipeline(sentence[:512])
        label_map = {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral"
        }
        return {
            "sentiment": label_map.get(result[0]["label"].lower(), "neutral"),
            "confidence": result[0]["score"],
            "rationale": f"FinBERT: {result[0]['label']}"
        }
    except Exception as e:
        print(f"FinBERT error: {str(e)[:100]}")
        return None
```

#### 3. Update Experiment Loops

Replace API call with FinBERT call:

```python
# OLD (for Mixtral/Llama)
prompt = create_few_shot_prompt(row["sentence"])
response = call_llama(prompt, model_name="mixtral-8x7b-32768")
parsed = parse_response(response)

# NEW (for FinBERT)
result = call_finbert(row["sentence"])  # No prompt needed
```

### Expected Performance Comparison

| Model | Strengths | Weaknesses | Best Use Case |
|-------|-----------|------------|---------------|
| **Mixtral-8x7B** | Balanced, good JSON, cost-efficient | Not financial-specific | General prompt engineering |
| **Llama-3.1-70B** | Large, capable, better than 3.3 | Higher cost | Complex reasoning tasks |
| **FinBERT** | Domain-specific, fast, free | No reasoning, no prompts | Production baseline |

**Expected Negative F1 (Critical Metric)**:

- Mixtral: 0.55-0.70 (with good prompts)
- Llama-3.1: 0.50-0.65
- **FinBERT: 0.75-0.85** ⭐ (trained on financial data)

## Comparative Analysis Structure

### Experiment Matrix (12 Total)

| Approach | E1 | E2 | E3 |
|----------|----|----|---- |
| **Zero-Shot** | Mixtral | Llama-3.1 | FinBERT* |
| **Few-Shot** (E4-E6) | Mixtral | Llama-3.1 | FinBERT* |
| **CoT** (E7-E9) | Mixtral | Llama-3.1 | FinBERT* |
| **ToT** (E10-E12) | Mixtral | Llama-3.1 | FinBERT* |

\* FinBERT results may be identical across approaches (no prompt awareness)

### Key Comparison Dimensions

1. **Model Comparison** (within each approach)
   - Which model best for each prompting strategy?
   - Does FinBERT beat prompt-engineered LLMs?

2. **Approach Comparison** (within each model)
   - Does complexity help? (Zero → Few → CoT → ToT)
   - Diminishing returns analysis

3. **Cross-Analysis**
   - Best overall: E? (model + approach combination)
   - Cost-benefit: Performance vs API costs
   - Production recommendation

4. **Negative Class Focus**
   - Critical metric: Negative F1 scores
   - False negative rate (missing bad financial news)
   - Confusion patterns (negative→neutral vs negative→positive)

## Final Recommendations

### For Your Assignment

**Include FinBERT with caveats**:

- Clearly state limitations in each section
- Compare "domain fine-tuning vs prompt engineering"
- Highlight that FinBERT E3 = E6 = E9 = E12 (same model)
- Analyze: "When does fine-tuning beat prompting?"

**Analysis Focus**:

1. If FinBERT wins: "Fine-tuning > Prompt Engineering for this task"
2. If Mixtral/Llama win: "General models + good prompts > narrow fine-tuning"
3. Hybrid insight: "Use FinBERT for production, LLMs for edge cases"

### Next Steps

1. Run all 12 experiments (expect ~8-10 hours total)
2. Create comprehensive comparison notebook
3. Statistical significance testing (McNemar's test)
4. Cost-benefit analysis
5. Production deployment recommendation
