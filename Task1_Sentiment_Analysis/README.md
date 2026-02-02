# Task 1: Sentiment Analysis - Complete Experimental Framework

## Overview

This folder contains a systematic experimental framework for evaluating Large Language Models on **Financial Sentiment Analysis** using the FinancialPhraseBank dataset.

## Experiments Structure

### 📂 Folder Organization

```
Task1_Sentiment_Analysis/
├── Zero_Shot/                  # Experiments E1, E2, E3
│   └── E1_E2_E3_zero_shot_sentiment.ipynb
├── Few_Shot/                   # Experiments E4, E5, E6
│   └── E4_E5_E6_few_shot_sentiment.ipynb
├── Chain_of_Thought/           # Experiments E7, E8, E9
│   └── E7_E8_E9_cot_sentiment.ipynb
├── Tree_of_Thought/            # Experiment E10 (+ variations)
│   └── E10_tot_sentiment.ipynb
└── Results/                    # Comprehensive analysis
    └── comprehensive_results_comparison.ipynb
```

## Experiments Summary

| Exp ID | Model | Prompting Strategy | Key Features |
|--------|-------|-------------------|--------------|
| **E1** | Gemini 2.5 Pro | Zero-Shot | Baseline - no examples |
| **E2** | Gemini 2.5 Flash | Zero-Shot | Fast model baseline |
| **E3** | Llama-3.3-70B | Zero-Shot | Open-source baseline |
| **E4** | Gemini 2.5 Pro | Few-Shot | 5 labeled examples |
| **E5** | Gemini 2.5 Flash | Few-Shot | 5 labeled examples |
| **E6** | Llama-3.3-70B | Few-Shot | 5 labeled examples |
| **E7** | Gemini 2.5 Pro | Chain-of-Thought | Step-by-step reasoning |
| **E8** | Gemini 2.5 Flash | Chain-of-Thought | Step-by-step reasoning |
| **E9** | Llama-3.3-70B | Chain-of-Thought | Step-by-step reasoning |
| **E10** | Gemini 2.5 Pro | Tree-of-Thought | Multi-path exploration |

## Dataset

**FinancialPhraseBank v1.0** - 100% Agreement Subset

- **Size**: 2,264 sentences
- **Classes**: Positive (25%), Negative (13%), Neutral (61%)
- **Source**: Aalto University
- **Quality**: All sentences have 100% annotator agreement

## Evaluation Metrics

All experiments are evaluated using:

- ✅ **Accuracy**: Overall classification correctness
- ✅ **Macro-F1**: Average F1 across all classes (handles class imbalance)
- ✅ **Precision**: Correctness of positive predictions
- ✅ **Recall**: Coverage of actual positives
- ✅ **Confusion Matrix**: Detailed error analysis
- ✅ **Per-class Recall Score (PRS)**: Individual class performance

## Setup Instructions

### 1. Environment Setup

Create a `.env` file in the root directory:

```bash
# API Keys
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn google-generativeai groq python-dotenv tqdm
```

### 3. Run Experiments

Execute notebooks in order:

1. **Zero-Shot** experiments first (baseline)
2. **Few-Shot** experiments
3. **Chain-of-Thought** experiments  
4. **Tree-of-Thought** experiments
5. **Results Comparison** (aggregates all results)

## Usage

### Running Individual Experiments

```python
# Example: Run Zero-Shot experiments
jupyter notebook Zero_Shot/E1_E2_E3_zero_shot_sentiment.ipynb
```

Each notebook includes:

- Data loading from FinancialPhraseBank
- Prompt engineering for the specific strategy
- Model inference with retry logic
- Metrics calculation
- Visualizations (confusion matrices, performance charts)
- Results export to CSV

### Comparing Results

```python
# Run comprehensive comparison
jupyter notebook Results/comprehensive_results_comparison.ipynb
```

This notebook provides:

- Strategy-wise performance comparison
- Model-wise performance comparison
- Cost-performance trade-off analysis
- Best configuration recommendations

## Expected Outputs

### Per Experiment

- CSV files with predictions
- Confusion matrices (PNG)
- Performance comparison charts (PNG)
- Metrics summary (CSV)

### Comprehensive Analysis

- Complete results table
- Heatmaps across all experiments
- Radar charts
- Cost-efficiency analysis
- Statistical comparisons

## Key Research Questions

1. **Which prompting strategy performs best?**
   - Zero-shot vs Few-shot vs CoT vs ToT

2. **How do proprietary models compare to open-source?**
   - Gemini (Pro & Flash) vs Llama-3.3-70B

3. **What is the cost-performance trade-off?**
   - Accuracy gains vs computational costs

4. **Which configuration is deployment-ready?**
   - Balance of accuracy, speed, and cost

## Notes

- **Test Mode**: Notebooks default to 100 samples for quick testing
  - Remove `.head(100)` for full dataset runs
  
- **Rate Limiting**: Built-in delays prevent API throttling
  - Adjust `time.sleep()` values if needed

- **Error Handling**: Robust retry logic for API calls
  - Exponential backoff on failures

- **Reproducibility**: Set `random_state=42` throughout
  - Ensures consistent results

## Results Interpretation

After running all experiments, consult the comprehensive results notebook for:

1. **Best Overall**: Highest Macro-F1 score
2. **Most Cost-Efficient**: Best F1/Cost ratio
3. **Fastest**: Lowest latency with acceptable performance
4. **Production Recommendation**: Based on your constraints

## Citation

If using this experimental framework, please cite:

```
Malo, P., Sinha, A., Takala, P., Korhonen, P. and Wallenius, J. (2013): 
"Good debt or bad debt: Detecting semantic orientations in economic texts." 
Journal of the American Society for Information Science and Technology.
```

## License

This experimental framework is for academic research purposes.

---

**Author**: Sidharth Rai  
**Institution**: Liverpool John Moores University (LJMU)  
**Date**: January 2026
