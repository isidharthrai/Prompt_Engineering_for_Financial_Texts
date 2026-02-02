# Task 2: Financial Risk Assessment

## Overview
This task focuses on assessing financial risk levels from financial statements using Large Language Models (LLMs) with different prompting strategies.

## Objective
Evaluate the ability of LLMs to assess financial risk and map sentiment to appropriate risk levels:
- **Low Risk**: Strong performance, growth indicators
- **Medium Risk**: Stable or neutral conditions
- **High Risk**: Declining performance, negative indicators  
- **Critical Risk**: Severe distress, existential threats

## Experiments (E11-E22)

### Zero-Shot (E11-E13)
- **E11**: Gemini 2.5 Pro - Direct risk assessment without examples
- **E12**: Gemini 2.5 Flash - Faster risk assessment
- **E13**: Llama-3.3-70B - Open-source model risk assessment

### Few-Shot (E14-E16)
- **E14**: Gemini 2.5 Pro - With 5 example risk assessments
- **E15**: Gemini 2.5 Flash - With examples
- **E16**: Llama-3.3-70B - With examples

### Chain-of-Thought (E17-E19)
- **E17**: Gemini 2.5 Pro - Step-by-step risk analysis
- **E18**: Gemini 2.5 Flash - Reasoning-based assessment
- **E19**: Llama-3.3-70B - Detailed risk evaluation

### Tree-of-Thought (E20-E22)
- **E20**: Gemini 2.5 Pro - Multi-path risk exploration
- **E21**: Gemini 2.5 Flash - Alternative risk scenarios
- **E22**: Llama-3.3-70B - Comprehensive risk analysis

## Dataset
Uses FinancialPhraseBank v1.0 (100% agreement subset) with sentiment labels that map to risk levels.

## Evaluation Metrics
- Risk level distribution (Low/Medium/High/Critical)
- Sentiment-to-risk correlation accuracy
- Model confidence scores
- Inter-model agreement rates
- Risk assessment consistency

## Expected Mappings
| Sentiment | Expected Risk Level |
|-----------|-------------------|
| Positive  | Low/Medium        |
| Neutral   | Medium            |
| Negative  | High/Critical     |

## Setup
1. Ensure API keys are configured in `.env` file
2. Install required packages: `pip install -r requirements.txt`
3. Run notebooks sequentially by prompting strategy

## Results Location
All results, visualizations, and analyses are saved in the `Results/` folder.
