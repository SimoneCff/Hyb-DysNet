# Hyb-DysNet: A Hybrid Feature Fusion Ensemble for Dysarthria Severity Classification

> **SAND Challenge Task 1 Submission** (IEEE ICASSP 2026)

**Authors:** PCL Team (Parthenope Computational Intelligence & Smart Systems Lab)
**Task:** Task 1 - Multi-class classification at T0  
**Objective:** Classify ALS patients into 5 severity levels of dysarthria using audio signals.

---

## Overview

Amyotrophic Lateral Sclerosis (ALS) leads to progressive motor neuron degeneration, often resulting in dysarthria. Early and accurate assessment of dysarthria severity is crucial for patient monitoring. 

**Hyb-DysNet** is a robust machine learning framework designed to tackle the critical challenges of the SAND dataset: **limited sample size** and **extreme class imbalance**. 

Our approach fuses **clinically interpretable acoustic features** (OpenSMILE) with **deep learned representations** (Wav2Vec2-XLS-R) and employs an ensemble of classifiers to achieve superior generalization.

---

## Methodology

Our pipeline consists of four main stages:

### 1. Hybrid Feature Extraction
We employ a dual-stream feature extraction strategy:
* **Hand-Crafted Features (OpenSMILE):** We extract the **eGeMAPSv02** feature set (88 features), which captures essential paralinguistic characteristics such as jitter, shimmer, loudness, and formants. This provides the model with expert clinical knowledge.
* **Deep Embeddings (Wav2Vec2-XLS-R):** We leverage `facebook/wav2vec2-xls-r-300m`, a massive pre-trained transformer, as a fixed feature extractor. We process raw audio waveforms and apply mean pooling to capture high-level semantic and acoustic context (1024 dimensions).

### 2. Data Preprocessing & Balancing
To address the scarcity of data for minority classes (Severe/Moderate):
* **Data Augmentation:** We apply offline augmentation (noise injection, pitch shifting, time stretching) specifically to the minority classes in the training set.
* **SMOTE:** We use Synthetic Minority Over-sampling Technique to mathematically balance the class distribution in the feature space.
* **Scaling:** All features are standardized (Z-score normalization).

### 3. Feature Selection
Given the high dimensionality of our hybrid feature vector (>1100 features) relative to the sample size, we apply **Feature Selection** (`SelectKBest` with ANOVA F-value) to retain only the top 400 most discriminative features, reducing noise and overfitting.

### 4. Ensemble Classification
The final classification is performed by a **Soft-Voting Ensemble** combining three diverse models:
* **XGBoost (GPU):** A gradient boosting decision tree tuned with high regularization (`max_depth=6`, `subsample=0.8`) to handle non-linear relationships.
* **LightGBM (GPU):** An efficient gradient boosting framework optimized for dense feature sets, trained with `class_weight='balanced'`.
* **Logistic Regression:** A linear model added to the ensemble to provide stability and generalization capabilities.

The final prediction for each subject is obtained by aggregating the predictions of all their individual audio files using a **Majority Voting** strategy.

---

## Results

Our approach was evaluated on the official **Validation Baseline** split provided by the organizers.

**Key Findings:**
* The inclusion of **OpenSMILE** features significantly improved the detection of "Moderate" dysarthria.
* **Wav2Vec2** embeddings helped distinguish between "Healthy" and "No Dysarthria" subjects.
* The **Ensemble** strategy proved more robust than any single model.

---

##  Repository Structure
```
.
├── notebooks/                       # Source code and experiments
│   └──  sand_challenge_submission.ipynb  # Main notebook (Extraction, Training, Inference)
├── submission/                      # Final output files
│   └── submission_task1.csv         # The final CSV file for the challenge
```

---

## Installation & Requirements

To reproduce our results, ensure you have Python 3.8+ and the following libraries installed. We recommend using a GPU-enabled environment (e.g., Kaggle, Colab).

```bash
pip install opensmile transformers torchaudio xgboost lightgbm imbalanced-learn pandas numpy scikit-learn tqdm librosa
```
---

## Usage
#### 1. Data Setup: Place the SAND dataset files (sand_task_1.xlsx, training/ folder) in the input directory.

#### 2. Run the Notebook: Execute the sand_challenge_submission.ipynb notebook. It will automatically:

* Extract hybrid features from the audio.

* Train the ensemble classifier using the official baseline split.

* Evaluate performance on the validation set.

* Generate predictions for the test set.

---
Submitted to IEEE ICASSP 2026 - SAND Challenge.
