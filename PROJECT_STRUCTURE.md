# LJMU Track Thesis - Complete Project Structure

## Overview
This workspace contains all experimental notebooks for the LJMU thesis on Large Language Model prompting strategies for financial text analysis.

## Project Organization

### Task 1: Sentiment Analysis (E1-E10) ✅ Complete
**Objective**: Classify financial statements as positive, negative, or neutral

```
Task1_Sentiment_Analysis/
├── Zero_Shot/
│   └── E1_E2_E3_zero_shot_sentiment.ipynb
├── Few_Shot/
│   └── E4_E5_E6_few_shot_sentiment.ipynb
├── Chain_of_Thought/
│   └── E7_E8_E9_cot_sentiment.ipynb
├── Tree_of_Thought/
│   └── E10_tot_sentiment.ipynb
├── Results/
│   └── comprehensive_results_comparison.ipynb
├── README.md
├── IMPLEMENTATION_SUMMARY.md
└── env_template.txt
```

**Experiments**:
- E1-E3: Zero-Shot (Gemini Pro, Flash, Llama)
- E4-E6: Few-Shot with 5 examples
- E7-E9: Chain-of-Thought reasoning
- E10: Tree-of-Thought multi-path exploration

---

### Task 2: Risk Assessment (E11-E22) ✅ Ready
**Objective**: Assess financial risk levels (Low/Medium/High/Critical)

```
Task2_Risk_Assessment/
├── Zero_Shot/
│   └── E11_E12_E13_zero_shot_risk.ipynb
├── Few_Shot/
│   └── (Template: E14_E15_E16_few_shot_risk.ipynb)
├── Chain_of_Thought/
│   └── (Template: E17_E18_E19_cot_risk.ipynb)
├── Tree_of_Thought/
│   └── (Template: E20_E21_E22_tot_risk.ipynb)
├── Results/
│   └── (Template: comprehensive_results_comparison.ipynb)
└── README.md
```

**Key Features**:
- Sentiment-to-Risk mapping analysis
- Risk distribution visualization
- Inter-model agreement metrics
- Correlation heatmaps

---

### Task 3: Insight Generation (E23-E34) ✅ Ready
**Objective**: Generate actionable financial insights from statements

```
Task3_Insight_Generation/
├── Zero_Shot/
│   └── E23_E24_E25_zero_shot_insights.ipynb
├── Few_Shot/
│   └── (Template: E26_E27_E28_few_shot_insights.ipynb)
├── Chain_of_Thought/
│   └── (Template: E29_E30_E31_cot_insights.ipynb)
├── Tree_of_Thought/
│   └── (Template: E32_E33_E34_tot_insights.ipynb)
├── Results/
│   └── (Template: comprehensive_results_comparison.ipynb)
└── README.md
```

**Insight Categories**:
1. Financial Trends
2. Business Impact
3. Stakeholder Effects
4. Opportunities & Risks
5. Recommended Actions

---

### Dataset
```
DatasetAnalysis_FinancialPhraseBank/
├── FinancialPhraseBank-v1.0/
│   ├── Sentences_50Agree.txt
│   ├── Sentences_66Agree.txt
│   ├── Sentences_75Agree.txt
│   └── Sentences_AllAgree.txt (Primary dataset)
├── financial_phrasebank_analysis.ipynb
└── processed_*.csv files
```

---

## Experimental Design

### Models Tested (3)
1. **Gemini 2.5 Pro** - High-capacity reasoning
2. **Gemini 2.5 Flash** - Fast inference
3. **Llama-3.3-70B** - Open-source alternative

### Prompting Strategies (4)
1. **Zero-Shot**: No examples, direct instruction
2. **Few-Shot**: 5 labeled examples provided
3. **Chain-of-Thought**: Step-by-step reasoning
4. **Tree-of-Thought**: Multi-path exploration

### Total Experiments: 34
- Task 1: 10 experiments (E1-E10)
- Task 2: 12 experiments (E11-E22)
- Task 3: 12 experiments (E23-E34)

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn google-generativeai groq python-dotenv tqdm
```

### 2. Configure API Keys
Create `.env` file in project root:
```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run Experiments
Navigate to each task folder and run notebooks sequentially:
```bash
cd Task1_Sentiment_Analysis/Zero_Shot/
jupyter notebook E1_E2_E3_zero_shot_sentiment.ipynb
```

---

## Evaluation Metrics

### Task 1 (Sentiment)
- Accuracy, Macro-F1, Precision, Recall
- Confusion matrices
- Per-class performance
- Confidence calibration

### Task 2 (Risk)
- Risk distribution analysis
- Sentiment-risk correlation
- Inter-model agreement
- Confidence scores

### Task 3 (Insights)
- Insight quality scores
- Factual accuracy
- Actionability rating
- Comprehensiveness
- Stakeholder relevance

---

## Next Steps

### Immediate Actions
1. ✅ Complete Task 1 full dataset runs (remove .head(100))
2. ✅ Execute E11-E13 (Task 2 Zero-Shot)
3. ✅ Execute E23-E25 (Task 3 Zero-Shot)

### Development Tasks
1. Create Few-Shot notebooks for Task 2 & 3
2. Create Chain-of-Thought notebooks for Task 2 & 3
3. Create Tree-of-Thought notebooks for Task 2 & 3
4. Create comprehensive results comparison notebooks
5. Generate final thesis visualizations

### Analysis Tasks
1. Cross-task performance comparison
2. Prompting strategy effectiveness analysis
3. Model capability assessment
4. Cost-performance trade-off analysis
5. Statistical significance testing

---

## File Naming Conventions

- **Notebooks**: `E{num}_{model}_{strategy}_{task}.ipynb`
- **Results CSV**: `e{num}_{model}_{strategy}_{task}_{timestamp}.csv`
- **Visualizations**: `{strategy}_{metric}_{plot_type}.png`
- **Summaries**: `{strategy}_{task}_summary_{timestamp}.csv`

---

## Notes

- All notebooks include SSL fix for macOS
- API rate limiting: 0.5-1.0 second delays
- Test runs use `.head(100)` - remove for full dataset
- Results auto-saved with timestamps
- Visualizations exported as 300 DPI PNG files

---

**Last Updated**: January 18, 2026
**Status**: Task 1 Complete, Task 2 & 3 Framework Ready
