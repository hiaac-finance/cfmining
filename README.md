# P²CE: Model-Agnostic Plausible Pareto-Optimal Counterfactual Explanations

The increasing use of machine learning algorithms in social applications has raised concerns about fairness and transparency, leading to the development of counterfactual explanations. These explanations supports individuals to understand and potentially alter unfavorable decisions in areas such as loan applications, job selections, and more, by providing actionable changes to input features that would lead to a desired outcome. Existing methods often struggle to balance feasibility, plausibility, and computational efficiency. To address this, we introduce P²CE, an algorithm for generating plausible Pareto-optimal counterfactual explanations, offering users a diverse set of optimal trade-offs between different notions of feasibility. P²CE employs an auxiliary isolation forest outlier detector to ensure that explanations are in accordance with the data distribution and leverages SHAP values to obtain optimal results with short computing times, regardless of the underlying model.  Our algorithm was empirically evaluated on three datasets, demonstrating superior performance in terms of both solution quality and computational efficiency compared to related techniques.

## Overview

This repository contains the implementation of P²CE and executed experiments. The `cfmining` directory contains Python scripts for data processing, model training, and evaluation, while the `notebooks` directory contains Jupyter notebooks with experiments and visualization.

## Installation

The recommend way to run the code is to set a Docker container. The file `Dockerfile`contains the configuration of the container utilized during development. Another way is to have Python installed on your machine. It is recommended to use a virtual environment to manage dependencies. Follow the steps below to set up your environment:

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment. On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
    
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package:
    ```bash
    pip install .
    ```

## Directory Structure

- `cfmining/`: Contains Python scripts with algorithm implementation (`algorithms.py`), data preprocessing (`data_preprocessing.py`and `datasets.py`), and some other utilities for P²CE functioning.
- `notebooks/`: Contains Jupyter notebooks with experiments. `experiments_[model_name].ipynb` presents the execution of experiments with the "model_name" classifier.
- `data/`: A directory with preprocessed datasets.
- This reposity was built upon MAPOCAM repository, and the implementation of the algorithm is also present.

## Usage

1. **Experiments**:

    - Run all cells of Jupyter notebooks.