# Complete Assignment Setup - Financial Sentiment Analysis

## ✅ MIGRATION COMPLETE

All 12 experiments have been updated from old models (GPT OSS 20B/120B, Llama-3.3-70B) to new models:

- **Mixtral-8x7B-32768** (E1, E4, E7, E10)
- **Llama-3.1-70B-Versatile** (E2, E5, E8, E11)
- **FinBERT (ProsusAI/finbert)** (E3, E6, E9, E12)

---

## 📊 Experiment Matrix

| Approach | E1-E3 | E4-E6 | E7-E9 | E10-E12 |
|----------|--------|--------|--------|---------|
| **Zero-Shot** | [E1_E2_E3_zero_shot_sentiment_All_agree.ipynb](Zero_Shot/E1_E2_E3_zero_shot_sentiment_All_agree.ipynb) | - | - | - |
| **Few-Shot** | - | [E4_E5_E6_few_shot_sentiment.ipynb](Few_Shot/E4_E5_E6_few_shot_sentiment.ipynb) | - | - |
| **Chain-of-Thought** | - | - | [E7_E8_E9_cot_sentiment.ipynb](Chain_of_Thought/E7_E8_E9_cot_sentiment.ipynb) | - |
| **Tree-of-Thought** | - | - | - | [E10_tot_sentiment.ipynb](Tree_of_Thought/E10_tot_sentiment.ipynb) |

### Model Distribution

- **E1, E4, E7, E10**: Mixtral-8x7B (via Groq API)
- **E2, E5, E8, E11**: Llama-3.1-70B (via Groq API)
- **E3, E6, E9, E12**: FinBERT (local inference)

---

## 🚀 Running the Assignment

### Prerequisites

1. **Groq API Key**: Add to `.env` file

   ```
   GROQ_API_KEY=your_key_here
   ```

2. **Install Dependencies**:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn groq python-dotenv tqdm transformers torch
   ```

### Execution Order

#### Step 1: Run Individual Experiments (8-10 hours total)

```bash
# Zero-Shot (E1-E3) - ~2 hours
jupyter notebook Zero_Shot/E1_E2_E3_zero_shot_sentiment_All_agree.ipynb

# Few-Shot (E4-E6) - ~2.5 hours  
jupyter notebook Few_Shot/E4_E5_E6_few_shot_sentiment.ipynb

# Chain-of-Thought (E7-E9) - ~3 hours
jupyter notebook Chain_of_Thought/E7_E8_E9_cot_sentiment.ipynb

# Tree-of-Thought (E10-E12) - ~3.5 hours
jupyter notebook Tree_of_Thought/E10_tot_sentiment.ipynb
```

**Important Notes**:

- Run notebooks sequentially (not parallel) to respect API rate limits
- Each notebook saves timestamped CSV results
- Monitor for parsing errors (especially E9, E12 with FinBERT)
- Expected total cost: ~$1-3 USD for Mixtral + Llama inference

#### Step 2: Comprehensive Comparative Analysis (30 minutes)

```bash
jupyter notebook Results/comprehensive_comparative_analysis.ipynb
```

This notebook:

- Loads all 12 experiment results
- Calculates comprehensive metrics
- Generates comparison visualizations
- Performs statistical significance tests
- Analyzes cost-benefit trade-offs
- Provides production recommendations

---

## 📁 Files Created

### Notebooks (Updated)

- ✅ `Zero_Shot/E1_E2_E3_zero_shot_sentiment_All_agree.ipynb`
- ✅ `Few_Shot/E4_E5_E6_few_shot_sentiment.ipynb`
- ✅ `Chain_of_Thought/E7_E8_E9_cot_sentiment.ipynb`
- ✅ `Tree_of_Thought/E10_tot_sentiment.ipynb`
- ✅ `Results/comprehensive_comparative_analysis.ipynb` (NEW)

### Documentation

- ✅ `MODEL_MIGRATION_GUIDE.md` - Detailed migration guide and FinBERT integration notes

### Backups (Created)

- `Zero_Shot/E1_E2_E3_zero_shot_sentiment_All_agree_BACKUP.ipynb`
- `Few_Shot/E4_E5_E6_few_shot_sentiment_BACKUP.ipynb`
- `Chain_of_Thought/E7_E8_E9_cot_sentiment_BACKUP.ipynb`
- `Tree_of_Thought/E10_tot_sentiment_BACKUP.ipynb`

---

## 🎯 Assignment Deliverables

After running all experiments, you will have:

### 1. Experiment Results (12 CSV files)

- `e1_mixtral_8x7b_zero_shot_YYYYMMDD_HHMMSS.csv`
- `e2_llama_3_1_70b_zero_shot_YYYYMMDD_HHMMSS.csv`
- `e3_finbert_zero_shot_YYYYMMDD_HHMMSS.csv`
- `e4_mixtral_8x7b_few_shot_YYYYMMDD_HHMMSS.csv`
- `e5_llama_3_1_70b_few_shot_YYYYMMDD_HHMMSS.csv`
- `e6_finbert_few_shot_YYYYMMDD_HHMMSS.csv`
- `e7_Mixtral_8x7B_cot_YYYYMMDD_HHMMSS.csv`
- `e8_Llama_3_1_70B_cot_YYYYMMDD_HHMMSS.csv`
- `e9_FinBERT_cot_YYYYMMDD_HHMMSS.csv`
- `e10_Mixtral_8x7B_tot_YYYYMMDD_HHMMSS.csv`
- `e11_Llama_3_1_70B_tot_YYYYMMDD_HHMMSS.csv`
- `e12_FinBERT_tot_YYYYMMDD_HHMMSS.csv`

### 2. Metrics Summaries (4 CSV files)

- `zero_shot_metrics_summary_YYYYMMDD_HHMMSS.csv`
- `few_shot_metrics_summary_YYYYMMDD_HHMMSS.csv`
- `cot_metrics_summary_YYYYMMDD_HHMMSS.csv`
- `tot_metrics_summary_YYYYMMDD_HHMMSS.csv`

### 3. Visualizations (16+ PNG files)

- `zero_shot_performance_comparison.png`
- `zero_shot_confusion_matrices.png`
- `zero_shot_confidence_analysis.png`
- `few_shot_performance_comparison.png`
- `few_shot_confusion_matrices.png`
- `few_shot_confidence_analysis.png`
- `cot_performance_comparison.png`
- `cot_confusion_matrices.png`
- `cot_confidence_analysis.png`
- `tot_performance_comparison.png` (if applicable)
- `tot_confusion_matrices.png` (if applicable)
- `model_comparison.png` ⭐
- `approach_comparison.png` ⭐
- `model_approach_heatmap.png` ⭐
- `cost_benefit_analysis.png` ⭐

### 4. Comprehensive Analysis (3 CSV files)

- `comprehensive_comparative_analysis.csv` ⭐
- `cost_benefit_analysis.csv` ⭐
- `production_readiness_evaluation.csv` ⭐

---

## 📊 Expected Results Preview

### Critical Metrics to Watch

**1. Negative F1 Score** (Most Important)

- **Target**: > 0.50
- **Why Critical**: Can't miss bad financial news in production
- **Expected Best**: FinBERT (~0.70-0.85) or Mixtral with CoT (~0.60-0.75)

**2. Macro F1 Score**

- **Target**: > 0.75 for production
- **Expected Range**: 0.65-0.80 depending on model/approach

**3. Parsing Error Rate**

- **Target**: < 5%
- **Risk**: High for FinBERT in CoT/ToT (cannot follow complex formats)

### Model Comparison Hypothesis

| Model | Expected Strengths | Expected Weaknesses | Best Approach |
|-------|-------------------|---------------------|---------------|
| **Mixtral-8x7B** | Good balance, JSON compliance | Not financial-specific | CoT or Few-Shot |
| **Llama-3.1-70B** | Large, capable | Higher cost, slower | CoT |
| **FinBERT** | Domain-trained, fast, free | No reasoning | Zero-Shot only |

### Approach Comparison Hypothesis

| Approach | Expected Performance | Cost | Best Model |
|----------|---------------------|------|------------|
| **Zero-Shot** | Baseline (lowest) | Low | FinBERT |
| **Few-Shot** | +10-15% over Zero | Medium | Mixtral |
| **Chain-of-Thought** | +5-10% over Few | High | Llama-3.1 |
| **Tree-of-Thought** | +0-5% over CoT | Very High | Uncertain |

---

## ⚠️ Important FinBERT Caveats

### FinBERT Cannot

- ❌ Follow prompts or instructions
- ❌ Use few-shot examples
- ❌ Perform step-by-step reasoning (CoT)
- ❌ Explore multiple hypotheses (ToT)

### FinBERT Integration Strategy

**E3 (Zero-Shot)**: ✅ **Valid** - Natural FinBERT usage
**E6 (Few-Shot)**: ⚠️ **Conceptually invalid** - Examples ignored, but included for comparison
**E9 (CoT)**: ❌ **Invalid** - No reasoning capability, results = same as E3
**E12 (ToT)**: ❌ **Invalid** - No multi-path exploration, results = same as E3

**Recommendation**: Note in your analysis that FinBERT E6 = E9 = E12 = E3 (all identical), demonstrating the difference between:

- **Prompt Engineering** (Mixtral/Llama): Flexible, instruction-following
- **Fine-Tuning** (FinBERT): Domain-specific, no flexibility

---

## 🎓 Assignment Write-Up Outline

### 1. Introduction

- Research question: Prompt engineering vs fine-tuning for financial sentiment
- Dataset: FinancialPhraseBank (2,217 sentences, 100% agreement)
- Experimental design: 3 models × 4 approaches = 12 experiments

### 2. Methodology

- **Models**: Mixtral-8x7B, Llama-3.1-70B, FinBERT
- **Approaches**: Zero-Shot, Few-Shot, CoT, ToT
- **Evaluation**: Accuracy, Macro-F1, Negative-F1, confusion matrices

### 3. Results

- Model comparison (which model wins overall?)
- Approach comparison (does complexity help?)
- Model × Approach interaction (heatmaps)
- Statistical significance (McNemar's tests)

### 4. Cost-Benefit Analysis

- API costs for Mixtral/Llama
- Performance per dollar
- Free FinBERT alternative

### 5. Discussion

- **Key Finding 1**: Best model for this task
- **Key Finding 2**: Optimal prompting complexity
- **Key Finding 3**: Fine-tuning vs prompt engineering verdict
- **Limitations**: FinBERT cannot reason, dataset imbalance

### 6. Conclusion & Recommendations

- Production deployment recommendation
- Trade-offs: performance vs cost vs flexibility
- Future work: ensemble, fine-tuning LLMs, active learning

---

## 🔧 Troubleshooting

### Issue: Groq API Rate Limits

**Solution**: Notebooks have 0.5s delays built-in. If still hitting limits, increase to 1.0s.

### Issue: FinBERT Not Detecting Negatives

**Expected**: FinBERT should perform better than LLMs on negatives (trained on financial data)

### Issue: High Parsing Errors for FinBERT in CoT/ToT

**Expected**: FinBERT doesn't output JSON, only sentiment labels. Parsing errors = 100% for structured outputs.

### Issue: Long Runtime

**Expected**: 2,217 samples × 0.5s = ~18 minutes per model per notebook = ~2-3 hours per notebook.

---

## 📌 Next Steps

1. ✅ All notebooks updated with new models
2. ✅ Comparative analysis notebook created
3. ⏳ **Your Turn**: Run the 4 experiment notebooks sequentially
4. ⏳ Run the comprehensive comparative analysis
5. ⏳ Write up your findings
6. ⏳ Submit assignment with all CSVs and visualizations

Good luck! 🚀

---

**Questions?** See [MODEL_MIGRATION_GUIDE.md](MODEL_MIGRATION_GUIDE.md) for detailed technical notes on FinBERT integration and model differences.
