# Explainable-AI - Social Bias in BERT - Attention based explanations vs post-hoc methods
This repository contains methods to detect and interpret social biases in the BERT (Bidirectional Encoder Representations from Transformers) model. Using the CrowS-Pairs dataset, we analyze how the model's  predictions favor stereotypical sentences over anti-stereotypical sentences.

## Data set
The study uses the CrowS-Pairs dataset, which consists of 1,508 sentence pairs covering nine bias categories, including race, gender, religion, and socioeconomic status. Each pair contrasts a historically disadvantaged group with an advantaged group for testing the model.

## Structure of notebooks
01_dataset_analysis.ipynb:
Analyzing the data set in general, to see what variables are in there and how the data works.

02_attention_rollout.ipynb:
Attention based explanation proposed by Abnar & Zuidema (2020). This method treats BERT as a Directed Acyclic Graph (DAG) and recursively multiplies attention matrices across layers while accounting for residual connections. It traces the flow of information from the raw input tokens to the final hidden representations.

03_shap_explanations.ipynb:
Post-hoc method where we calculate Shapley values to see what tokens contribute to the model's output. We do this by appyling SHAP to Pseudo-Log-Likelihood PLL. Pll works by masking tokens one by one and summing their conditional log probabilities. Then this score shows how "acceptable" the models finds a biased vs unbiased sentence.

These functions are made and can be used by installing the necessary packages in the requirements.txt file.
