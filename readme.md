# ***Evaluating Spatiotemporal Graph Neural Network Architectures for Air Quality Forecasting***

This repository supports the doctoral dissertation titled **Evaluating Spatiotemporal Graph Neural Network Architectures for Air Quality Forecasting**, available at **https://github.com/moses-b-alexander/stgnn-aqf-praxis**.

This repository contains the full experimental setup for reproducing the results of my research on spatiotemporal graph neural networks for air quality forecasting. It includes source code, dataset, and a zipped archive containing the experimental results from three trials.

This repository includes:

* `src/`: source code for running and evaluating experiment
* `weather/`: covariate climatology data used in dataset creation
* `experiment.zip`: archive containing a full copy of the repository structure from the experimental run, including `src/` and `weather/` as well as additional components listed below
* `requirements.txt`: dependency list for installation via Python 3.12 pip
* `README.md`: full experiment documentation and instructions

---

## 🔍 Objective

Spatiotemporal graph neural networks, augmented with spatiotemporal positional encoding and exogenous covariates capturing long-term climate trends, predict future air pollution values based on historical data. By modelling spatial and temporal relationships between air quality sensors, areas of concentration and patterns of dispersion can be more effectively understood by policymakers (Zheng et al, 2015). Thus, they can make more optimal decisions to improve air quality via various environmental measures aimed to mitigate the concerning pollution-related factors identified via spatiotemporal analysis of urban air quality.

---

## 🧠 Model Architecture Overview

Each model architecture consists of four stages:

* **Positional Encoding**: Applied to raw input features using either additive or concatenative fusion to enhance input representation
* **Spatial Component**: A graph convolutional network layer, one of GraphConv, DiffConv, or GATConv
* **Temporal Component**: A recurrent neural network layer, one of RNN, LSTM, or GRU
* **Bidimensional Attention Mechanism**: Applied at the end to refine joint spatiotemporal representation; decoded by a linear layer

Model variants differ in positional encoding strategy, spatial-temporal module combinations, and whether attention is applied. The hyperparameter definitions remain the same throughout. 

---

## ⚙️ Methodology

The air quality data was obtained from the Urban Air Project by Microsoft Research (2014–2015) *https://www.microsoft.com/en-us/research/project/urban-air/*, and exogenous climatology covariates for the relevant timeframe were sourced from the World Bank Group's Climate Knowledge Portal *https://climateknowledgeportal.worldbank.org/country/china/climate-data-historical*. The models were developed using Torch Spatiotemporal library (tsl): https://torch-spatiotemporal.readthedocs.io/en/latest/index.html.

1. **Load Datasets**
2. **Define Spatiotemporal Graph Data**

   * Construct edges using similarity threshold value of *0.4*
   * Apply *masking* for missing graph features
   * Specify forecasting horizon of *4* and training window length of *20*
3. **Preprocess Spatiotemporal Graph Data**

   * Standardize graph features by subtracting the mean and dividing by the standard deviation
4. **Preprocess Covariate Climatology Data**

   * Align covariate feature sequence lengths through duplication by hour
5. **Merge datasets**
6. **Prepare Model for Training**

   * Split data into training, validation, and test sets in the ratio of *0.80* / *0.05* / *0.15*
   * Set batch size to *64*
   * Define loss metrics *MAE* and *MAPE* for evaluation and monitoring
   * Configure optimization strategy with learning rate *0.001*, optimization algorithm *Adam*, and weight decay *0.0001*
7. **Establish logging and model checkpointing**
8. **Train Model**
9. **Test Model**
10. **Evaluate Output**

   * Analyze prediction error and diagnostic metrics
   * Review training and evaluation logs

---

## 📁 Directory Structure (inside experiment.zip)

```
experiment.zip
├── src/                 # Source code (duplicate of root src/)
├── weather/             # Covariate climatology data (duplicate of root weather/)
├── data/                # Spatiotemporal graph data downloaded using tsl
├── dataset/             # Final processed dataset used for training (merged from files within data/ and weather/)
├── table/               # CSV of results averaged over 3 trials (table_final.csv)
├── plots/               # 52 summary visualizations generated using table/
├── requirements.txt     # Python 3.12 dependencies
├── logs/ (excluded)     # Training logs and model checkpoints for all 3 trials of 45 models each
```

---

## 🧪 Running the Experiment

Ensure you are using **Python 3.12**.

### 📦 Installation

```bash
pip install -r requirements.txt
```

### ▶️ Run the Experiment

1. Open `src/main.py`
2. **Uncomment lines 1660–1681**
3. Run the following command:

```bash
python src/main.py > experiment.txt
```

This will train all 45 models once, while producing training logs and diagnostic metrics.

### 📊 Generate Table and Plots

1. Open `src/main.py`
2. **Comment lines 1660–1681 (to avoid rerunning the experiment)**
3. **Uncomment lines 1683–1686**
4. Run the following command:

```bash
python src/main.py > visualization.txt
```

This will produce:

* `table/table_final.csv` (table of prediction errors and diagnostic metrics from the trial)
* `plots/` (52 `.png` summary visualizations of prediction errors, parameter counts, and training times)

---

## 📈 Evaluation Metrics

Each experimental run evaluates:

* Loss: `test_mae`
* Monitoring metrics: `test_mae_lag_01`, `test_mae_lag_02`, `test_mae_lag_03`
* Reference: `test_mape`
* Model architecture displaying structure of layers
* Model parameter count
* Model training time

Metrics for one full trial are saved in `experiment.txt`.

---

## 📦 Archive Notes

The `experiment.zip` archive contains:

* `src/`: duplicated from repository root
* `weather/`: duplicated from repository root
* `data/`: spatiotemporal graph dataset provided by `tsl`
* `dataset/`: merged dataset created from files within `data/` and `weather/`
* `table/`: summative `table_final.csv` recording average performance over 3 trials for all 45 models
* `plots/`: 52 summary chart images (.png) visualizing `table_final.csv`
* `requirements.txt`: dependency list for installation via Python 3.12 pip

> ⚠️ `logs/` is excluded due to large folder size (checkpoints for each of 45 models for all 3 trials)

---

## 📜 Results

This experiment evaluated 45 spatiotemporal graph neural network models over 3 trials. Key findings include:

* **DiffConv + LSTM** combinations achieved the highest overall predictive accuracy.
* **Positional encoding** without attention, either additive or concatenative mode, provided consistent but marginal improvements in MAE across most architectures.
* **Attention combined with either positional encoding mode** significantly and consistently improved predictive accuracy when used alongside LSTM or GRU temporal components.
* **Concatenative positional encoding with attention** led to catastrophic degradation when paired with RNNs as the temporal component. The combination of concatenative positional encoding, weak temporal modeling, and attention severely undermined the representational capacity of the model, resulting in systematically poor and considerably degraded performance across all trials.

---

## 📚 Citation

Alexander, M. B. (2025). *Evaluating Spatiotemporal Graph Neural Network Architectures for Air Quality Forecasting* (Doctoral dissertation). The George Washington University. https://github.com/moses-b-alexander/stgnn-aqf-praxis

---

## 👤 Author

***Moses B. Alexander***

Email: **mosesalexander209@gmail.com**
GitHub: **https://github.com/moses-b-alexander/**

---

## ⚖️ License

No license — all rights reserved.
