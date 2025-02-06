# Hidden-Bias-Benchmark

Repository for the Hidden Bias Benchmark dataset.

## Concept Lists

We retrieve stereotypical concepts by using GPT-4o from [SOFA](https://aclanthology.org/2024.emnlp-main.812.pdf), [BBQ](https://aclanthology.org/2022.findings-acl.165/), [StereoSets](https://aclanthology.org/2021.acl-long.416/), and [Crows-Pairs](https://aclanthology.org/2020.emnlp-main.154/). \
And pairing with anti-stereotypical concepts correspondingly. \
Concepts are in 📂 concept_lists/📄modified_all_concepts074.csv

## Dataset

Our HBB dataset is in 📂 data/📄 Hidden-Bias-Benchmark.csv

---
Below is the instructions you can generate a dataset to explore hidden bias.

## Extract Concepts

`python concept_analysis.py --model_name=gpt-4o --dataset=bbq --all`

Can use any datasets you want. NOT Only limited to the datasets mentioned before.

## Generate Raw Questions

`python q_generate.py --model_name=gpt-4o --all_q`

## Final Questions
Use `questions_final.ipynb` to replace [[X]] to finish up question generation.
