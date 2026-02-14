#!/usr/bin/env python3
"""
Generate Chain-of-Thought Risk Assessment Notebook
R7: Llama3.1:8b, R8: Qwen3:8b, R9: DeepSeek-R1:8b
"""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Chain-of-Thought Financial Risk Assessment\n",
                "## Experiments R7, R8, R9 - Ollama Models\n",
                "\n",
                "**Models:**\n",
                "- R7: Llama3.1:8b (Ollama)\n",
                "- R8: Qwen3:8b (Ollama)\n",
                "- R9: DeepSeek-R1:8b (Ollama)\n",
                "\n",
                "**Approach:** Chain-of-Thought reasoning with step-by-step analysis\n",
                "\n",
                "**Task:** Classify financial statements as positive (opportunity/low risk), negative (threat/high risk), or neutral"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import requests\n",
                "import pandas as pd\n",
                "import json\n",
                "import time\n",
                "from datetime import datetime\n",
                "from tqdm import tqdm\n",
                "from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix, classification_report\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "print(\"✓ Libraries imported\")\n",
                "print(\"\\n\" + \"=\"*80)\n",
                "print(\"CHAIN-OF-THOUGHT RISK ASSESSMENT\")\n",
                "print(\"=\"*80)\n",
                "print(\"Models: R7: Llama3.1:8b, R8: Qwen3:8b, R9: DeepSeek-R1:8b\")\n",
                "print(\"=\"*80)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Load Dataset"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_path = \"../../DatasetAnalysis_FinancialPhraseBank/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt\"\n",
                "\n",
                "sentences, risks = [], []\n",
                "with open(data_path, \"r\", encoding=\"utf-8\", errors=\"ignore\") as f:\n",
                "    for line in f:\n",
                "        if \"@\" in line.strip():\n",
                "            parts = line.strip().rsplit(\"@\", 1)\n",
                "            if len(parts) == 2:\n",
                "                sentences.append(parts[0])\n",
                "                risks.append(parts[1])\n",
                "\n",
                "df = pd.DataFrame({\"sentence\": sentences, \"true_risk\": risks})\n",
                "print(f\"✓ Dataset loaded: {len(df)} sentences\")\n",
                "print(f\"\\nRisk distribution:\\n{df['true_risk'].value_counts()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Chain-of-Thought Prompt Design"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def create_cot_prompt(sentence):\n",
                "    prompt = f\"\"\"You are a financial risk assessment expert. Analyze the following financial statement step-by-step.\n",
                "\n",
                "Statement: \"{sentence}\"\n",
                "\n",
                "Follow this reasoning process:\n",
                "\n",
                "Step 1: Identify key financial indicators (revenue, profit, growth, losses, challenges)\n",
                "Step 2: Determine the direction of change (increasing, decreasing, stable)\n",
                "Step 3: Assess impact on business risk (does this reduce or increase investment risk?)\n",
                "Step 4: Consider investor perspective (opportunity vs threat)\n",
                "Step 5: Final classification\n",
                "\n",
                "Guidelines:\n",
                "- Positive: Signals that REDUCE investment risk (growth, profits, expansion)\n",
                "- Negative: Signals that INCREASE investment risk (losses, declines, challenges)\n",
                "- Neutral: Informational with no direct risk impact\n",
                "\n",
                "Provide your response in JSON format:\n",
                "{{\n",
                "    \"step1_indicators\": \"identified indicators\",\n",
                "    \"step2_direction\": \"direction of change\",\n",
                "    \"step3_risk_impact\": \"impact on risk\",\n",
                "    \"step4_investor_view\": \"opportunity or threat\",\n",
                "    \"step5_classification\": \"positive/negative/neutral\",\n",
                "    \"confidence\": 0.0-1.0,\n",
                "    \"final_rationale\": \"one sentence summary\"\n",
                "}}\"\"\"\n",
                "    return prompt\n",
                "\n",
                "print(\"✓ CoT prompt function defined\")\n",
                "print(\"\\nExample prompt:\")\n",
                "print(create_cot_prompt(\"Revenue increased by 15%\")[:500] + \"...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Ollama API Functions"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "OLLAMA_URL = \"http://localhost:11434/api/generate\"\n",
                "\n",
                "def call_ollama(model, prompt, temp=0.0):\n",
                "    for attempt in range(3):\n",
                "        try:\n",
                "            r = requests.post(OLLAMA_URL, json={\"model\": model, \"prompt\": prompt, \"temperature\": temp, \"stream\": False}, timeout=120)\n",
                "            if r.status_code == 200:\n",
                "                return r.json().get(\"response\", \"\")\n",
                "        except Exception as e:\n",
                "            if attempt < 2:\n",
                "                time.sleep(2**attempt)\n",
                "    return None\n",
                "\n",
                "def parse_response(text):\n",
                "    try:\n",
                "        if \"```json\" in text:\n",
                "            text = text.split(\"```json\")[1].split(\"```\")[0]\n",
                "        elif \"```\" in text:\n",
                "            text = text.split(\"```\")[1]\n",
                "        result = json.loads(text.strip())\n",
                "        return result\n",
                "    except:\n",
                "        lower = text.lower()\n",
                "        if \"positive\" in lower and \"negative\" not in lower:\n",
                "            return {\"step5_classification\": \"positive\", \"confidence\": 0.5, \"final_rationale\": \"Parsed\"}\n",
                "        elif \"negative\" in lower:\n",
                "            return {\"step5_classification\": \"negative\", \"confidence\": 0.5, \"final_rationale\": \"Parsed\"}\n",
                "        return {\"step5_classification\": \"neutral\", \"confidence\": 0.5, \"final_rationale\": \"Parsed\"}\n",
                "\n",
                "print(\"✓ Ollama functions defined\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Run Experiments\n### R7: Llama3.1:8b (CoT)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "test_df = df.head(200).copy()\n",
                "\n",
                "print(\"=\"*80)\n",
                "print(\"R7: Llama3.1:8b (CoT)\")\n",
                "print(\"=\"*80)\n",
                "\n",
                "r7_results = []\n",
                "for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=\"R7\"):\n",
                "    response = call_ollama(\"llama3.1:8b\", create_cot_prompt(row[\"sentence\"]))\n",
                "    if response:\n",
                "        parsed = parse_response(response)\n",
                "        r7_results.append({\n",
                "            \"sentence\": row[\"sentence\"],\n",
                "            \"true_risk\": row[\"true_risk\"],\n",
                "            \"predicted_risk\": parsed.get(\"step5_classification\", \"error\"),\n",
                "            \"confidence\": parsed.get(\"confidence\", 0),\n",
                "            \"rationale\": parsed.get(\"final_rationale\", \"\")\n",
                "        })\n",
                "    else:\n",
                "        r7_results.append({\"sentence\": row[\"sentence\"], \"true_risk\": row[\"true_risk\"], \"predicted_risk\": \"error\", \"confidence\": 0, \"rationale\": \"API failed\"})\n",
                "    time.sleep(0.5)\n",
                "\n",
                "r7_df = pd.DataFrame(r7_results)\n",
                "print(f\"\\n✓ R7 completed: {len(r7_df)} predictions\")\n",
                "display(r7_df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### R8: Qwen3:8b (CoT)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"\\n\" + \"=\"*80)\n",
                "print(\"R8: Qwen3:8b (CoT)\")\n",
                "print(\"=\"*80)\n",
                "\n",
                "r8_results = []\n",
                "for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=\"R8\"):\n",
                "    response = call_ollama(\"qwen3:8b\", create_cot_prompt(row[\"sentence\"]))\n",
                "    if response:\n",
                "        parsed = parse_response(response)\n",
                "        r8_results.append({\n",
                "            \"sentence\": row[\"sentence\"],\n",
                "            \"true_risk\": row[\"true_risk\"],\n",
                "            \"predicted_risk\": parsed.get(\"step5_classification\", \"error\"),\n",
                "            \"confidence\": parsed.get(\"confidence\", 0),\n",
                "            \"rationale\": parsed.get(\"final_rationale\", \"\")\n",
                "        })\n",
                "    else:\n",
                "        r8_results.append({\"sentence\": row[\"sentence\"], \"true_risk\": row[\"true_risk\"], \"predicted_risk\": \"error\", \"confidence\": 0, \"rationale\": \"API failed\"})\n",
                "    time.sleep(0.5)\n",
                "\n",
                "r8_df = pd.DataFrame(r8_results)\n",
                "print(f\"\\n✓ R8 completed: {len(r8_df)} predictions\")\n",
                "display(r8_df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### R9: DeepSeek-R1:8b (CoT)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"\\n\" + \"=\"*80)\n",
                "print(\"R9: DeepSeek-R1:8b (CoT)\")\n",
                "print(\"=\"*80)\n",
                "\n",
                "r9_results = []\n",
                "for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=\"R9\"):\n",
                "    response = call_ollama(\"deepseek-r1:8b\", create_cot_prompt(row[\"sentence\"]))\n",
                "    if response:\n",
                "        parsed = parse_response(response)\n",
                "        r9_results.append({\n",
                "            \"sentence\": row[\"sentence\"],\n",
                "            \"true_risk\": row[\"true_risk\"],\n",
                "            \"predicted_risk\": parsed.get(\"step5_classification\", \"error\"),\n",
                "            \"confidence\": parsed.get(\"confidence\", 0),\n",
                "            \"rationale\": parsed.get(\"final_rationale\", \"\")\n",
                "        })\n",
                "    else:\n",
                "        r9_results.append({\"sentence\": row[\"sentence\"], \"true_risk\": row[\"true_risk\"], \"predicted_risk\": \"error\", \"confidence\": 0, \"rationale\": \"API failed\"})\n",
                "    time.sleep(0.5)\n",
                "\n",
                "r9_df = pd.DataFrame(r9_results)\n",
                "print(f\"\\n✓ R9 completed: {len(r9_df)} predictions\")\n",
                "display(r9_df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Calculate Metrics"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def calc_metrics(df, name):\n",
                "    valid = df[df[\"predicted_risk\"].isin([\"positive\",\"negative\",\"neutral\"])].copy()\n",
                "    if valid.empty:\n",
                "        return {\"Experiment\": name, \"Accuracy\": 0, \"Macro-F1\": 0}, np.zeros((3,3)), valid\n",
                "    \n",
                "    y_true, y_pred = valid[\"true_risk\"], valid[\"predicted_risk\"]\n",
                "    labels = [\"positive\",\"negative\",\"neutral\"]\n",
                "    \n",
                "    metrics = {\n",
                "        \"Experiment\": name,\n",
                "        \"Total Samples\": len(df),\n",
                "        \"Valid Predictions\": len(valid),\n",
                "        \"Accuracy\": accuracy_score(y_true, y_pred),\n",
                "        \"Macro-F1\": f1_score(y_true, y_pred, average=\"macro\"),\n",
                "        \"Weighted-F1\": f1_score(y_true, y_pred, average=\"weighted\"),\n",
                "        \"Macro-Precision\": precision_score(y_true, y_pred, average=\"macro\"),\n",
                "        \"Macro-Recall\": recall_score(y_true, y_pred, average=\"macro\"),\n",
                "        \"MCC\": matthews_corrcoef(y_true, y_pred)\n",
                "    }\n",
                "    \n",
                "    for i, lbl in enumerate(labels):\n",
                "        p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)\n",
                "        r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)\n",
                "        f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)\n",
                "        metrics[f\"{lbl.capitalize()}_Precision\"] = p[i]\n",
                "        metrics[f\"{lbl.capitalize()}_Recall\"] = r[i]\n",
                "        metrics[f\"{lbl.capitalize()}_F1\"] = f[i]\n",
                "    \n",
                "    cm = confusion_matrix(y_true, y_pred, labels=labels)\n",
                "    return metrics, cm, valid\n",
                "\n",
                "r7_m, r7_cm, r7_v = calc_metrics(r7_df, \"R7: Llama3.1:8b (CoT)\")\n",
                "r8_m, r8_cm, r8_v = calc_metrics(r8_df, \"R8: Qwen3:8b (CoT)\")\n",
                "r9_m, r9_cm, r9_v = calc_metrics(r9_df, \"R9: DeepSeek-R1:8b (CoT)\")\n",
                "\n",
                "metrics_df = pd.DataFrame([r7_m, r8_m, r9_m])\n",
                "print(\"\\n\" + \"=\"*80)\n",
                "print(\"CHAIN-OF-THOUGHT RISK ASSESSMENT RESULTS\")\n",
                "print(\"=\"*80)\n",
                "display(metrics_df[[\"Experiment\",\"Accuracy\",\"Macro-F1\",\"MCC\"]].round(4))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Visualizations"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "# Performance comparison\n",
                "metrics_plot = [\"Accuracy\", \"Macro-F1\", \"Macro-Precision\", \"Macro-Recall\"]\n",
                "x = np.arange(len(metrics_plot))\n",
                "width = 0.25\n",
                "\n",
                "for i, (m, lbl) in enumerate([(r7_m, \"Llama3.1:8b\"), (r8_m, \"Qwen3:8b\"), (r9_m, \"DeepSeek-R1:8b\")]):\n",
                "    vals = [m[k] for k in metrics_plot]\n",
                "    axes[0].bar(x + i*width, vals, width, label=lbl, alpha=0.8)\n",
                "\n",
                "axes[0].set_xlabel(\"Metrics\", fontsize=12, weight=\"bold\")\n",
                "axes[0].set_ylabel(\"Score\", fontsize=12, weight=\"bold\")\n",
                "axes[0].set_title(\"CoT Risk Assessment Performance\", fontsize=14, weight=\"bold\")\n",
                "axes[0].set_xticks(x + width)\n",
                "axes[0].set_xticklabels(metrics_plot)\n",
                "axes[0].legend()\n",
                "axes[0].set_ylim([0, 1])\n",
                "axes[0].grid(axis=\"y\", alpha=0.3)\n",
                "\n",
                "# Confusion matrices\n",
                "fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))\n",
                "labels = [\"Positive\", \"Negative\", \"Neutral\"]\n",
                "\n",
                "for idx, (cm, title) in enumerate([(r7_cm, \"R7: Llama3.1:8b\"), (r8_cm, \"R8: Qwen3:8b\"), (r9_cm, \"R9: DeepSeek-R1:8b\")]):\n",
                "    sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", xticklabels=labels, yticklabels=labels, ax=axes2[idx])\n",
                "    axes2[idx].set_title(title, fontsize=12, weight=\"bold\")\n",
                "    axes2[idx].set_ylabel(\"True\", fontsize=11, weight=\"bold\")\n",
                "    axes2[idx].set_xlabel(\"Predicted\", fontsize=11, weight=\"bold\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"cot_risk_confusion_matrices.png\", dpi=300, bbox_inches=\"tight\")\n",
                "plt.show()\n",
                "print(\"✓ Visualizations saved\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Save Results"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ts = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                "\n",
                "r7_df.to_csv(f\"r7_llama3_1_8b_cot_risk_{ts}.csv\", index=False)\n",
                "r8_df.to_csv(f\"r8_qwen3_8b_cot_risk_{ts}.csv\", index=False)\n",
                "r9_df.to_csv(f\"r9_deepseek_r1_8b_cot_risk_{ts}.csv\", index=False)\n",
                "metrics_df.to_csv(f\"cot_risk_metrics_summary_{ts}.csv\", index=False)\n",
                "\n",
                "print(f\"✓ All results saved with timestamp: {ts}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Conclusions\n",
                "\n",
                "**Chain-of-Thought for Risk Assessment:**\n",
                "\n",
                "CoT prompting guides models through explicit reasoning steps:\n",
                "1. Identify financial indicators\n",
                "2. Determine direction of change\n",
                "3. Assess risk impact\n",
                "4. Consider investor perspective\n",
                "5. Final classification\n",
                "\n",
                "This structured approach improves transparency and reasoning quality for financial risk assessment."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "pmp", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Save notebook
output_path = "R7_R8_R9_cot_risk.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"✅ Created: {output_path}")
print("📊 Chain-of-Thought Risk Assessment Notebook")
print("   • R7: Llama3.1:8b")
print("   • R8: Qwen3:8b")
print("   • R9: DeepSeek-R1:8b")
