# Description-Based Bias Benchmark (DBB)

Repository for the Description-based Bias Benchmark dataset.

[//]: # (**Paper:** _What’s Not Said Still Hurts: A Description-Based Evaluation Framework for Measuring Social Bias in LLMs_ &#40;ARR 2025 Submission&#41;  )

[//]: # (**Repository:** [DBB on OpenScience]&#40;https://anonymous.4open.science/r/Hidden-Bias-Benchmark-A84F/&#41;)

---

## Overview

The **Description-Based Bias Benchmark (DBB)** is a large-scale dataset designed to systematically evaluate social bias in large language models (LLMs) at the semantic, contextual, and descriptive levels. 

---

## Dataset Summary

- **Instances:** ~103,649 pairs of questions/options
- **Categories:** Age, Gender, Race/Ethnicity, Socioeconomic Status (SES), Religions

---

## Intended Use

- **Purpose:** Benchmark and analyze social bias in LLMs at description level.
- **Users:** NLP researchers, LLM developers, fairness auditors.

---

## Dataset Structure

Each item contains:
- A scenario with demographic identity (explicitly or implicitly) (context)
- Two answer options reflecting opposing concepts
- Concept pairs
- Traditional stereotype explanation
- Category label (e.g., gender, SES, etc.)
- Biased Target (e.g. male, young, etc.)

---

## Data Generation & Quality Control

- Bias concepts adapted from [SOFA](https://aclanthology.org/2024.emnlp-main.812.pdf), [BBQ](https://aclanthology.org/2022.findings-acl.165/), [StereoSets](https://aclanthology.org/2021.acl-long.416/), and [Crows-Pairs](https://aclanthology.org/2020.emnlp-main.154/).
- Contexts and options generated using GPT-4o, then refined.
- **Manual Review:** Every sample included has been individually reviewed and confirmed to meet success criteria for fluency, coherence, and semantic alignment.

---

# _Important Files and Codes_

## Concept Lists

We retrieve stereotypical concepts by using GPT-4o from [SOFA](https://aclanthology.org/2024.emnlp-main.812.pdf), [BBQ](https://aclanthology.org/2022.findings-acl.165/), [StereoSets](https://aclanthology.org/2021.acl-long.416/), and [Crows-Pairs](https://aclanthology.org/2020.emnlp-main.154/). \
And pairing with anti-stereotypical concepts correspondingly. \
Concepts are in 📂 concept_lists/📄modified_all_concepts074.csv

## Dataset

Our DBB dataset is in 📂 data/📄 Bias-Dataset.csv

`Bias-Dataset-More-Samples.zip` has more samples for the dataset.

---
Below is the instructions you can generate a dataset to explore bias via the description-based method. Codes are in 📂 src/

## Extract Concepts

`python concept_analysis.py --model_name=gpt-4o --dataset=bbq --all`

Can use any datasets you want. NOT Only limited to the datasets mentioned before.

## Generate Raw Questions

`python q_generate.py --model_name=gpt-4o --all_q`

## Final Questions
Use `questions_final.ipynb` to replace [[X]] to finish up question generation.

## Results
`GPT-4o-results.zip` contains results of each question for GPT-4o in DBB.


[//]: # (## Citation)

[//]: # ()
[//]: # (If you use DBB in your work, please cite:)

[//]: # ()
[//]: # (```bibtex)