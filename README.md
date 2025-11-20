# SAND Challenge Apply from the Parthenope Computational Intelligence & Smart Systems Lab (PCL) team

This repository contains our submission for the SAND (Speech Analysis for Neurodegenerative Diseases) Challenge, part of the IEEE ICASSP 2026 international conference.

The goal of this project is to develop an Artificial Intelligence system for the diagnosis and classification of dysarthria severity in ALS patients, using only vocal signals.

Dataset is been given by the SAND Challenge Team

## Task 1: Multi-Class Classification (T0)
As per the challenge specifications, our primary focus is Task 1.

Objective: To identify the most reliable approach for correctly detecting and classifying the severity of voice disorders (dysarthria) by analyzing audio signals from the first assessment (T0).

The model must classify each subject into one of the following five classes:

- ALS with Severe dysarthria (Class 1)

- ALS with Moderate dysarthria (Class 2)

- ALS with Mild dysarthria (Class 3)

- ALS with No dysarthria (Class 4)

- Healthy Subject (Class 5)

## Our Approach - Hyb-DysNet

To address the critical challenges of this dataset (**small sample size** and **extreme class imbalance**), we developed **Hyb-DysNet**, a hybrid framework that fuses:

1.  **Hand-Crafted Features (OpenSMILE):** Extracts 88 robust clinical features (eGeMAPSv02) like jitter, shimmer, and formants.
2.  **Deep Embeddings (Wav2Vec2-XLS-R):** Uses a pre-trained 300M parameter transformer to capture high-level semantic context from raw audio.
3.  **Feature Selection:** Reduces noise by selecting only the most discriminative features.
4.  **Ensemble Learning:** A soft-voting classifier combining **XGBoost**, **LightGBM**, and **Logistic Regression**, trained on **SMOTE-balanced** data.

This notebook implements the full pipeline: **Feature Extraction**, **Training**, **Evaluation**, and **Submission Generation**.

