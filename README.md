# Scalable Credit Card Fraud Detection with Distributed Machine Learning

## Overview

This project addresses the critical challenge of credit card fraud detection in a big data context, focusing on extreme class imbalance and system scalability. It develops, evaluates, and analyzes machine learning models - both supervised and unsupervised - on a large-scale, synthetically augmented transaction dataset using Apache Spark on Google Cloud Platform (GCP). The goal is to build a robust, efficient, and adaptable fraud detection pipeline capable of processing high-volume financial data while minimizing false positives and detecting rare fraud instances.

## Core Components & Pipeline

The project implements a comprehensive, end-to-end machine learning pipeline for fraud detection, featuring:

1.  **Data Preparation:**
    * Initial loading and stratified splitting of an anonymized credit card transaction dataset (284,807 transactions, ~0.17% fraud). 
    * Creation of separate training (~4M rows) and untouched test (~56K rows) sets to avoid data leakage and ensure realistic evaluation. 
2.  **Class Imbalance Handling:**
    * Application of **SMOTE** (Synthetic Minority Over-sampling Technique) to oversample the minority (fraud) class in the training set to 25%. 
    * Further augmentation of the training set with **bootstrapping and jitter** to enhance model robustness and simulate real-world variability. 
3.  **Model Implementation & Evaluation:**
    * Development of **Apache Spark ML pipelines** for four diverse models: Logistic Regression, LightGBM, Isolation Forest, and Random Forest. 
    * Rigorous evaluation using metrics tailored for imbalanced datasets: **Precision, Recall, F1-Score, ROC-AUC, and PR-AUC**. 
4.  **Scalability Analysis:**
    * Systematic assessment of model training time and predictive performance across varying dataset sizes (5%, 25%, 100% of the augmented training data). 
    * Comparison of model behavior on 2-node and 3-node **GCP Dataproc clusters**. 
5.  **Compute Resource Monitoring:**
    * Detailed analysis of **CPU utilization, Disk I/O operations (read/write), and YARN memory allocation** during model training, collected from GCP Monitoring UI. 
    * Identified resource bottlenecks and evaluated the efficiency of Spark task scheduling and partitioning strategies. 

## Data

he project uses a benchmark credit card fraud detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Total Transactions:** 284,807. 
* **Fraud Rate:** ~0.17% (492 fraudulent transactions). 
* **Features:** 28 anonymized PCA-derived features (V1-V28), plus `Time` and `Amount`. 
* **Target:** `Class` (0 for legitimate, 1 for fraudulent). 
* **Augmented Training Data:** Expanded to ~4 million rows using SMOTE and bootstrapping. 

## Methodology & Key Techniques

* **Distributed Processing:** All data preprocessing and modeling pipelines run on **Apache Spark DataFrames** across **Google Cloud Dataproc** clusters to handle large data volumes efficiently. 
* **Imbalance Handling:**
    * **SMOTE:** Generates synthetic minority samples by interpolating between existing fraud examples. 
    * **Bootstrapping with Jitter:** Augments training data by resampling with replacement and adding controlled random noise to numeric features, increasing diversity. 
* **Model Selection:**
    * **Supervised:** Logistic Regression (interpretable baseline), LightGBM (efficient gradient boosting), Random Forest (robust ensemble). 
    * **Unsupervised:** Isolation Forest (anomaly detection). 
* **Evaluation Philosophy:** Prioritizes **PR-AUC, Precision, and Recall** over Accuracy for highly imbalanced datasets, reflecting true operational performance. 

## Results & Insights

* **Model Performance:**
    * **Logistic Regression:** Strong baseline (PR-AUC: 0.7095, Accuracy: 0.9992), with fastest training times and high interpretability. 
    * **LightGBM:** Achieved high PR-AUC (0.6942) and Recall (0.7957), balancing false negatives effectively, but with higher training costs. 
    * **Random Forest:** High precision (0.8219), but recall (0.6452) indicated missed frauds. 
    * **Isolation Forest:** Underperformed (PR-AUC: 0.0021, ROC-AUC: ~0.5052), as SMOTE-generated samples reduced fraud's "rarity," undermining its core anomaly detection logic. 
* **Scalability & Resource Utilization:**
    * Training time generally scaled linearly with dataset size up to a threshold, after which resource contention (I/O, memory) degraded throughput. 
    * **Logistic Regression** maintained the fastest and most predictable training times across configurations. 
    * Adding more worker nodes did not always lead to faster training due to Spark task scheduling and executor coordination overhead. 
    * **2-node cluster** showed higher average CPU utilization (53.5%) and more sustained disk activity, while the **3-node cluster** exhibited lower average CPU (44.1%) but higher peak disk operations and more volatile memory usage. 
    * Proper repartitioning (e.g., 4x worker count) improved task distribution and minimized straggler effects. 

## Tools & Technologies

* **Python:** Primary programming language.
    * `pyspark.ml`: For distributed machine learning models (Logistic Regression, RandomForest, VectorAssembler).
    * `synapse.ml` (Microsoft SynapseML): For LightGBMClassifier, IsolationForest.
    * `pandas`, `numpy`: For data manipulation and numerical operations.
    * `imblearn.over_sampling.SMOTE`: For synthetic minority oversampling.
    * `sklearn.metrics`: For model evaluation (accuracy, precision, recall, F1, ROC-AUC, PR-AUC).
    * `matplotlib`, `seaborn`, `sklearn.manifold.TSNE`: For data visualization.
* **Apache Spark:** Distributed computing framework for large-scale data processing and ML.
* **Google Cloud Platform (GCP):**
    * **Dataproc:** Managed Spark clusters for scalable execution. 
    * **Google Cloud Storage (GCS):** For data loading and saving results.
    * **GCP Monitoring UI:** For collecting system resource metrics. 
* **Jupyter Notebook:** Interactive development environment for code execution and analysis.
* **Git:** Version control.

## How to View This Project

1.  **Read the Report:** For a comprehensive overview of the project's objectives, methodology, results, and insights, please refer to the `final_report.pdf`.
2.  **Explore the Code:** The detailed implementation of data preprocessing, imbalance handling, model training, and scalability analysis is available in the `.ipynb` notebooks (`smote_bootstrap.ipynb` and `fraud-ML-pipeline-scalability.ipynb`). GitHub natively renders Jupyter notebooks, allowing you to view the code and outputs directly in your browser.

## Limitations & Future Work

* **Dataset Specificity:** Uses a single, anonymized historical dataset, which may not generalize to all fraud behaviors or platforms. 
* **Interpretability:** Some high-performing models (e.g., Random Forest) offer limited insight into individual predictions, a critical requirement in finance. 
* **Static Evaluation:** All evaluations are offline; real-time fraud detection requires handling high throughput with minimal latency, not explicitly modeled here. 
* **XGBoost Integration:** Unable to effectively scale XGBoost due to PySpark compatibility issues with dynamic resource allocation on Dataproc. 

Future work could involve integrating multi-institutional data, using temporal models (RNNs, Transformers), incorporating explainability techniques (LIME), exploring streaming/online learning, and investigating privacy-preserving learning frameworks. 

## Acknowledgements

This project was completed as part of the ST446 course at the London School of Economics and Political Science. Special thanks to the original data providers for the credit card transaction dataset.