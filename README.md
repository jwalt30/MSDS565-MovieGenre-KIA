# MSDS 565 — Movie Genre Classification with NLP, LLM Distillation, and Explainable AI

**Course:** MSDS 565  
**Project:** Project 3 — Movie Genre Classification  
**Team:** KIA (KIAGJ)  
**Repository:** `MSDS565-MovieGenre-KIA`

## Project Overview

This project investigates how well movie genres can be inferred from written plot synopses using Natural Language Processing (NLP) and classical machine learning. Using the MPST/IMDb movie synopsis dataset, the team develops and evaluates genre-specific binary classifiers for **Crime, Action, and Comedy**.

The project goes beyond building a classifier. It compares two text representations—**TF-IDF features** and **Sentence Transformer embeddings**—while holding the classifier family constant (logistic regression). It then explores a knowledge-distillation workflow in which a large language model (LLM) labels a small sample of movie plots and logistic regression models are trained on those pseudo-labels. Finally, Explainable AI (xAI) methods are used to examine which words influence the models' predictions.

The central questions are:

- How much predictive information about movie genre is contained in a plot synopsis?
- Do sparse, word-level TF-IDF features or dense sentence embeddings produce stronger genre classifiers?
- What performance changes when training labels come from an LLM rather than the dataset's IMDb genre labels?
- Which words drive model predictions, and how can those contributions be interpreted?

## Project Objectives

1. Explore the dataset's movie characteristics, genre-label distributions, and synopsis text.
2. Prepare a consistent train/test evaluation using the dataset's predefined splits.
3. Build separate binary logistic regression classifiers for Crime, Action, and Comedy.
4. Compare TF-IDF and Sentence Transformer representations under a consistent modeling approach.
5. Evaluate the effect of LLM-generated pseudo-labels through a teacher–student experiment.
6. Interpret model behavior using logistic regression coefficients and SHAP explanations.
7. Document the workflow, saved model artifacts, and steps needed to reproduce the analysis.

## Dataset

The project uses **`mpst_imdb_movies.csv`**, containing movie plot synopses, IMDb-related movie attributes, predefined data splits, and multi-label genre indicators.

The exploratory analysis begins with **11,824 movies**, 33 columns, and 24 genre-label columns. The dataset's genre labels are multi-label: a movie may belong to more than one genre.

A minimum-vote filter of **`numVotes >= 100`** is applied consistently in the modeling and xAI notebooks. This retains 11,695 movies (98.9% of the original rows). After combining the predefined training and validation splits, the modeling workflow uses:

| Partition | Rows |
|---|---:|
| Training (train + validation) | 9,365 |
| Held-out test | 2,330 |
| Total after vote filter | 11,695 |

The selected target genres are **Crime, Action, and Comedy**. Each genre is modeled as an independent yes/no classification task, allowing a movie to receive more than one positive genre prediction.

## Methodology and Workflow

### 1. Exploratory Data Analysis (EDA)

Notebook 1 examines the dataset before modeling. The analysis includes:

- Distributions of release year, runtime, IMDb average rating, and vote counts (with vote counts examined on a log scale).
- Average-rating comparisons across genres.
- A review of possible minimum-vote thresholds and the number of rows removed from each predefined split.
- Positive counts and positive rates for all 24 genre labels.
- A correlation heatmap of the genre indicator columns.
- Synopsis-length statistics, including character counts and approximate token counts.
- TF-IDF vocabulary analysis to identify terms that are unusually prominent within selected genres compared with the overall corpus.

The EDA supports the choice of Crime, Action, and Comedy as modeling targets and establishes the 100-vote threshold used in subsequent notebooks.

### 2. Text Representation and Model Development

Notebook 2 converts plot synopses into numerical features and trains one logistic regression classifier for each selected genre.

**Text representations**

- **TF-IDF:** A sparse, word-level representation using English stop-word removal, a minimum document frequency of 5, a maximum of 10,000 features, and sublinear term frequency.
- **Sentence embeddings:** The `sentence-transformers/all-MiniLM-L6-v2` model produces 384-dimensional embeddings for each synopsis.

**Classifier and evaluation design**

- Logistic regression is used for every representation/genre combination so that the comparison focuses on the text representation rather than changing classifier families.
- The regularization setting is held at `C=0.1`.
- Class-weight settings are compared using five-fold stratified cross-validation and macro-F1.
- Vectorizers and scalers are fitted on training data only; the held-out test set is transformed and evaluated without fitting on test data.
- Performance is reported using classification metrics, confusion matrices, and test-set macro-F1.

**LLM teacher–student experiment**

The notebook also evaluates knowledge distillation through pseudo-labeling:

1. A sample of training-set movie plots is selected from the three target genres.
2. Google's Gemini model (`gemini-3.5-flash-lite`) acts as the teacher and assigns Crime, Action, and Comedy labels using plot text only.
3. Logistic regression student models are trained using the teacher-generated labels.
4. The students are evaluated against the true IMDb labels on the same held-out test set.
5. A control experiment trains models on the same small sample using the original IMDb labels. This helps distinguish the effect of having fewer training examples from the effect of noisy teacher labels.

The teacher is prompted without movie titles to reduce the chance of title recognition influencing its genre decisions. Structured JSON output is used for the annotations.

### 3. Explainable AI (xAI)

Notebook 3 focuses on the TF-IDF logistic regression models because their features correspond to individual words and can be interpreted directly.

The analysis includes:

- **Coefficient interpretation:** Identifies words with the strongest positive and negative associations for Crime, Action, and Comedy.
- **SHAP:** Uses `shap.LinearExplainer` to estimate how individual word features contribute to Crime predictions relative to a training-set baseline.
- **Global explanations:** Compares mean absolute SHAP values with coefficient rankings to show the difference between a word's model weight and its contribution across actual test examples.
- **Local explanations:** Uses SHAP waterfall plots to examine a confident correct Crime prediction, the model's most confident false positive, and its most confident false negative.

This notebook also checks that the saved TF-IDF artifacts reproduce the test-set macro-F1 results from Notebook 2.

## Results

### Held-out test performance

The table below reports **macro-F1** on the 2,330-row held-out test set. Macro-F1 gives equal weight to the positive and negative classes.

| Representation | Crime | Action | Comedy | Mean |
|---|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.7519 | 0.7867 | 0.7377 | **0.7587** |
| MiniLM embeddings + logistic regression | 0.7353 | 0.7676 | 0.6928 | **0.7319** |

In this experiment, TF-IDF outperformed the MiniLM embedding representation for all three selected genres. Action had the strongest macro-F1 under both representations, while Comedy was the most challenging target for the embedding model.

### LLM pseudo-labeling findings

The teacher labeled a sample of 297 unique training-set plots. The teacher's macro-F1 on that labeled sample was 0.778 for Crime, 0.761 for Action, and 0.825 for Comedy. These are sample-level teacher results, not held-out test results.

The student-model comparison indicates that reducing the training sample from 9,365 rows to 297 rows has a substantial effect on performance. The notebook's control experiment separates this sample-size effect from the additional effect of using teacher-generated rather than IMDb labels. The reported mean macro-F1 loss was:

- **TF-IDF:** 0.161 total loss, decomposed into 0.089 associated with fewer training rows and 0.072 associated with noisier labels.
- **MiniLM embeddings:** 0.129 total loss, with the control comparison attributing 0.134 to fewer training rows and -0.005 to the label difference.

These findings are specific to the sample, teacher model, prompts, and evaluation design used in this project; they should not be interpreted as a general measure of LLM labeling quality.

### Interpretability findings

The coefficient and SHAP analyses show that words such as *police*, *murder*, *gun*, and *crime* are influential signals for the Crime classifier. The project also identifies limitations of word-based models, including the influence of recurring character names and the difference between a feature's coefficient size and its actual contribution across the test set.

## Repository Structure

```text
MSDS565-MovieGenre-KIA/
│
├── README.md
├── environment.yml
│
├── models/
│   ├── P3_N2_emb-minilm_logreg_action.pkl
│   ├── P3_N2_emb-minilm_logreg_comedy.pkl
│   ├── P3_N2_emb-minilm_logreg_crime.pkl
│   ├── P3_N2_tfidf_logreg_action.pkl
│   ├── P3_N2_tfidf_logreg_comedy.pkl
│   ├── P3_N2_tfidf_logreg_crime.pkl
│   ├── P3_N2_tfidf_scaler.pkl
│   └── P3_N2_tfidf_vectorizer.pkl
│
└── notebooks/
    ├── P3_MovieGenre-1_EDA_KIAGJ.ipynb
    ├── P3_MovieGenre-2_Modeling_KIAGJ.ipynb
    └── P3_MovieGenre-3_xAI_KIAGJ.ipynb
```

### Notebook Guide

| Notebook | Purpose |
|---|---|
| `P3_MovieGenre-1_EDA_KIAGJ.ipynb` | Dataset exploration, genre balance, synopsis characteristics, and vocabulary analysis. |
| `P3_MovieGenre-2_Modeling_KIAGJ.ipynb` | TF-IDF and MiniLM feature construction, logistic regression, held-out evaluation, and LLM teacher–student experiment. |
| `P3_MovieGenre-3_xAI_KIAGJ.ipynb` | Model verification, coefficient interpretation, SHAP global analysis, and individual prediction explanations. |

### Saved Model Artifacts

The `models/` directory contains:

- Three TF-IDF logistic regression classifiers (Crime, Action, and Comedy).
- Three MiniLM-embedding logistic regression classifiers (Crime, Action, and Comedy).
- The fitted TF-IDF vectorizer.
- The fitted TF-IDF scaler.

The xAI notebook loads the TF-IDF vectorizer, scaler, and genre-specific TF-IDF classifiers saved by Notebook 2. Keep these artifacts together so the explanations use the same fitted feature transformation as the trained models.

## Tools and Technologies

- **Language:** Python
- **Development environment:** Jupyter Notebook, Conda
- **Data analysis:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Text representation:** scikit-learn TF-IDF, Sentence Transformers
- **Machine learning:** scikit-learn logistic regression, GridSearchCV, stratified cross-validation
- **LLM annotation:** Google Gemini API through the `google-genai` SDK
- **Explainability:** SHAP
- **Model persistence:** joblib / Python pickle artifacts
- **Version control:** Git and GitHub

## Reproducibility and Setup

### 1. Clone the repository

```bash
git clone https://github.com/jwalt30/MSDS565-MovieGenre-KIA.git
cd MSDS565-MovieGenre-KIA
```

### 2. Create and activate the Conda environment

The included `environment.yml` defines the project environment named `msds565`.

```bash
conda env create -f environment.yml
conda activate msds565
```

If the environment already exists, update it with:

```bash
conda env update -f environment.yml --prune
```

### 3. Add the dataset

Obtain `mpst_imdb_movies.csv` separately and place it in one of the locations supported by the modeling notebook. The dataset is intentionally not stored in this repository.

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open the notebooks in order:

1. Run Notebook 1 to review the data and EDA.
2. Run Notebook 2 to build features, train/evaluate models, and save model artifacts.
3. Run Notebook 3 to load the saved TF-IDF artifacts and generate explanations.

Notebook 2 may create embedding caches, teacher-response caches, and pseudo-labeled data files outside the repository alongside the dataset. These are generated working files and are not required as committed repository contents.

### 5. Gemini API key (Notebook 2)

The LLM teacher-annotation portion of Notebook 2 requires a Google Gemini API key. Set it as an environment variable named `GEMINI_API_KEY` before running that portion of the notebook. The notebook also supports entering the key at a hidden prompt if the variable is not set.

**Never commit API keys, passwords, or other credentials to GitHub.** The TF-IDF/MiniLM modeling and xAI workflow should be treated separately from the optional API-dependent teacher-annotation step.

## Limitations and Considerations

- Genre labels are multi-label, and plot synopses do not always communicate tone or genre explicitly.
- The 100-vote threshold removes a small portion of the dataset but changes the population represented by the analysis.
- Sentence embeddings from `all-MiniLM-L6-v2` are limited by the model's input length; long synopses may be truncated.
- The LLM teacher experiment uses a small, genre-enriched sample rather than a random sample of all movies. Its sample-level metrics should be interpreted with that design in mind.
- Model performance is evaluated on the provided held-out split; results may differ on new movies or different data.
- SHAP explanations describe how the fitted model uses features. They do not establish causal relationships between words and movie genres.

## Team

**Team KIA (KIAGJ)**  
MSDS 565 — Predictive Modeling

---

*This repository contains the notebooks and model artifacts for the MSDS 565 Movie Genre Classification project. See the notebooks for the full analysis, implementation details, evaluation outputs, and visualizations.*
