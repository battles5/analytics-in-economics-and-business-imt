# The Causal Effect of R&D Expenditure on Productivity Growth: A Double Machine Learning Approach

## Overview

This repository contains the Python code and analytical pipeline used for the empirical analysis in the paper *"Course project: The causal effect of R&D expenditure on productivity growth: A double machine learning approach"*. The study applies Double Machine Learning (DML) to cross-country data to estimate the causal impact of research and development on total factor productivity (TFP). 

Note that the raw and processed datasets, original LaTeX paper files, and other course materials are **not** included in this repository and must be sourced independently if attempting to replicate the dataset.

> **Disclaimer:** This work represents the final project for the MD2SL Master's course in Analytics in Economics and Business (University of Florence & IMT Lucca). There is no claim to originality or academic value; it is purely a didactic exercise within the context of the course. The analysis code and the LaTeX manuscript were generated with the assistance of Claude Opus 4.6 High and Gemini 3.1 Pro (Preview).

## Repository Structure & Code Functionality

The analysis is divided into five modular Python scripts found in the `code/` directory:

1. **`data_preparation.py`**  
   Responsible for downloading/loading the raw datasets (Penn World Table, World Development Indicators, and Worldwide Governance Indicators), applying primary cleaning steps, calculating 2000-2019 country averages, performing Principal Component Analysis (PCA) for the highly collinear governance indicators, and exporting the final estimation dataset.
   
2. **`dml_analysis.py`**  
   Implements the core Partially Linear Regression (PLR) model using `DoubleML`. Runs the estimation loop over nine different machine learning nuisance models (including LASSO, Regularized linear regressions, Random Forest, Gradient Boosting, BART, and Neural Networks). Incorporates Bayesian hyperparameter tuning via `Optuna` and extracts the cross-validated $R^2$ scores for both treatment and outcome predictions (a crucial diagnostic step).

3. **`causal_forest.py`**  
   Extends the average analysis by estimating Conditional Average Treatment Effects (CATEs) using the `econml` Causal Forest implementation to analyze treatment heterogeneity across different levels of country development.

4. **`shap_analysis.py`**  
   Uses SHAP (SHapley Additive exPlanations) values to interpret feature importance in the first-stage nuisance models, helping identify the primary drivers of R&D investment and confounding productivity effects.

5. **`experimental_extensions.py`**  
   Explores robustness and additional model structures through advanced modeling techniques like Super Learner ensembling, spatial/peer-group lags, Gaussian Mixture Models (Latent Regimes), and Gaussian Process regressions.

## Usage Guide

1. Clone the repository and navigate into the folder:
   ```bash
   git clone https://github.com/battles5/analytics-in-economics-and-business-imt.git
   cd analytics-in-economics-and-business-imt
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the data:
   The scripts expect the dataset to be placed correctly relative to the project root. Since `data/` is excluded, you will need to execute the first script (`python code/data_preparation.py`) after obtaining or replacing the source inputs to generate the `final_data.csv`.

4. Run the analytical pipeline:
   Execute the remaining scripts sequentially to reproduce the models and visual outputs discussed in the study:
   ```bash
   python code/dml_analysis.py
   python code/causal_forest.py
   python code/shap_analysis.py
   python code/experimental_extensions.py
   ```

## Included References & Data Sources

The underlying theoretical frameworks and sources coded in this project include:

* **Optuna:** Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *Proceedings of the 25th ACM SIGKDD*.
* **DoubleML:** Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML: An Object-Oriented Implementation of Double Machine Learning in Python. *Journal of Machine Learning Research*.
* **Causal Machine Learning Value-add:** Baiardi, D., & Naghi, A. A. (2024). The value-added of machine learning to causal inference: Evidence from revisited studies. *European Economic Review*.
* **Methodological Core (DML):** Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1).
* **Penn World Table:** Feenstra, R. C., Inklaar, R., & Timmer, M. P. (2015). The Next Generation of the Penn World Table. *American Economic Review*, 105(10).
* **Super Learner:** van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super Learner. *Statistical Applications in Genetics and Molecular Biology*.
* **SHAP:** Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*.
* **World Development Indicators:** World Bank. (2024). *World Development Indicators*. Washington, D.C.: World Bank.
* **Worldwide Governance Indicators:** World Bank. (2025). *Worldwide Governance Indicators*. Washington, D.C.: World Bank.
* **Causal Forests:** Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *Journal of the American Statistical Association*.