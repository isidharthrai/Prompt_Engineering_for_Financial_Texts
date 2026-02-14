#!/usr/bin/env python3
"""
Generate COMPLETE Chain-of-Thought Risk Assessment Notebook
Aggressively adapted from E7 sentiment notebook with fixed prompt formatting
"""

import json
import re

# Read E7 notebook
e7_path = "../../Task1_Sentiment_Analysis/Chain_of_Thought/E7_E8_E9_cot_sentiment.ipynb"
with open(e7_path, 'r') as f:
    e7_nb = json.load(f)

# Deep copy and adapt
risk_nb = {
    "cells": [],
    "metadata": e7_nb["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5
}

def replace_text(text):
    """Aggressively replace sentiment terms with risk terms"""
    if not isinstance(text, str):
        return text
    
    # Pre-calculated replacements for prompt guidelines to keep them clean
    if 'Positive: Financial improvements' in text:
        return '- Positive: Opportunity signals that reduce investment risk (revenue growth, profit increase, market expansion, strong performance)\n'
    if 'Negative: Financial declines' in text:
        return '- Negative: Threat signals that increase investment risk (losses, declining sales, operational challenges, market difficulties)\n'
    if 'Neutral: Factual statements' in text:
        return '- Neutral: Informational content with no direct risk implications, routine announcements, balanced statements\n'
    
    # Model name mappings
    text = text.replace('E7:', 'R7:')
    text = text.replace('E8:', 'R8:')
    text = text.replace('E9:', 'R9:')
    text = text.replace('e7_', 'r7_')
    text = text.replace('e8_', 'r8_')
    text = text.replace('e9_', 'r9_')
    
    # Specific sentiment metric/variable names in code
    text = text.replace('true_sentiment', 'true_risk')
    text = text.replace('predicted_sentiment', 'predicted_risk')
    text = text.replace('"sentiment"', '"risk"')
    text = text.replace("'sentiment'", "'risk'")
    text = text.replace('.sentiment', '.risk')
    text = text.replace('sentiment_df', 'risk_df')
    text = text.replace('sentiment_results', 'risk_results')
    
    # General word replacements (Case-aware)
    text = text.replace('Sentiment Analysis', 'Risk Assessment')
    text = text.replace('Sentiment analysis', 'Risk assessment')
    text = text.replace('SENTIMENT ANALYSIS', 'RISK ASSESSMENT')
    text = text.replace('sentiment analysis', 'risk assessment')
    
    text = text.replace('Sentiment Class', 'Risk Class')
    text = text.replace('Sentiment Distribution', 'Risk Distribution')
    
    # Detection
    text = text.replace('sentiment detection', 'risk detection')
    text = text.replace('detect sentiment', 'assess risk')
    
    # Final cleanup replacements for remaining "sentiment" words
    text = text.replace('Sentiment', 'Risk')
    text = text.replace('sentiment', 'risk')
    text = text.replace('SENTIMENT', 'RISK')
    
    # Specific task descriptions / paths
    text = text.replace('Task1_Sentiment_Analysis', 'Task2_Risk_Assessment')
    text = text.replace('cot_sentiment', 'cot_risk')
    text = text.replace('zero_shot_metrics', 'zero_shot_risk_metrics')
    text = text.replace('few_shot_metrics', 'few_shot_risk_metrics')
    
    return text

# Process each cell
for cell in e7_nb['cells']:
    new_cell = {
        'cell_type': cell['cell_type'],
        'id': cell.get('id', ''),
        'metadata': cell.get('metadata', {})
    }
    
    # Process source
    if 'source' in cell:
        if isinstance(cell['source'], list):
            new_source = []
            for line in cell['source']:
                new_source.append(replace_text(line))
            new_cell['source'] = new_source
        else:
            new_cell['source'] = replace_text(cell['source'])
    
    # Clear outputs for code cells
    if cell['cell_type'] == 'code':
        new_cell['outputs'] = []
        new_cell['execution_count'] = None
    
    risk_nb['cells'].append(new_cell)

# Save
output_path = "R7_R8_R9_cot_risk.ipynb"
with open(output_path, 'w') as f:
    json.dump(risk_nb, f, indent=1, ensure_ascii=False)

print(f"✅ Re-generated: {output_path}")
print("📊 Comprehensive 'sentiment' → 'risk' replacements applied to all 12 sections.")
