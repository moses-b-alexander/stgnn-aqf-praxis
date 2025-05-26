# ***Evaluating Spatiotemporal Graph Neural Network Architectures for Air Quality Forecasting***

This codebase supports the doctoral dissertation titled **Evaluating Spatiotemporal Graph Neural Network Architectures for Air Quality Forecasting**, available at *https://github.com/moses-b-alexander/stgnn-aqf-praxis/* .

This repository contains the full experimental setup for reproducing the results of my research on spatiotemporal graph neural networks. It includes source code, covariate data, and an archive containing the experimental code, data, and results from three trials.

The repository includes:

* `src/`: main experiment source code
* `weather/`: covariate data used in dataset creation
* `requirements.txt`: dependency list for installation via Python 3.12 pip
* `README.md`: full experiment documentation and instructions
* `experiment.zip`: archive containing a full copy of the repository structure from the experimental results, including `src/` and `weather/` as well as additional components listed below

---

## 🔍 Objective

STGNNs augmented with spatiotemporal positional encoding and exogenous covariates representing long-term climate trends predict future air pollution values based on historical data. By modelling spatial and temporal relationships between sensors, areas of concentration and patterns of dispersion can be better understood by policymakers (Zheng et al, 2015). Thus, they can make more optimal decisions to improve air quality via various environmental measures aimed to mitigate the concerning factors identified via spatiotemporal analysis of urban air quality.

---

## 🧠 Model Architecture Overview

Each model architecture consists of four stages:

* **Positional Encoding (optional)**: Applied to raw features using either additive or concatenative fusion to enhance input representation, prior to the spatial component
* **Spatial Component**: One of GraphConv, DiffusionConv, or GATConv
* **Temporal Component**: One of RNN, GRU, or LSTM
* **Attention (optional)**: Applied after the temporal component to refine joint spatiotemporal representation; followed by a linear decoder

Model variants differ in positional encoding strategy, spatial-temporal combinations, and whether attention is applied. The hyperparameter definitions remain the same throughout. 

---

## ⚙️ Methodology

The air quality data was obtained from the Urban Air Project by Microsoft Research (2014–2015) https://www.microsoft.com/en-us/research/project/urban-air/, and exogenous climatology covariates for the relevant timeframe were sourced from the World Bank Group's Climate Knowledge Portal https://climateknowledgeportal.worldbank.org/country/china/climate-data-historical. The models were developed using Torch Spatiotemporal (tsl): https://torch-spatiotemporal.readthedocs.io/en/latest/index.html.

1. **Load and Save Datasets**
2. **Define Spatiotemporal Graph Data**

   * Construct edges based on similarity threshold
   * Apply masking for missing values
   * Specify forecasting horizon and temporal window length
3. **Preprocess Features**

   * Standardize feature values by subtracting the mean and dividing by the standard deviation
4. **Load Exogenous Climatology Data**

   * Align sequence lengths with graph features
   * Merge covariate features into the graph dataset
5. **Prepare Model**

   * Split data into training, validation, and test sets in the ratio of 0.6 / 0.2 / 0.2
   * Set batch size
   * Define loss metrics for evaluation and monitoring
   * Configure optimization strategy with learning rate, optimization algorithm, and weight decay
   * Establish logging and model checkpointing
6. **Train Model**
7. **Test Model**
8. **Evaluate Model**

   * Analyze forecasted predictions and error metrics
   * Review training and evaluation logs

---

## 📁 Directory Structure (inside experiment.zip)

```
experiment.zip
├── src/                 # Main experiment source code (duplicate of root src/)
├── weather/             # Covariate data (duplicate of root weather/)
├── dataset/             # Final dataset used for training (merged from data/ and weather/)
├── data/                # Raw graph data downloaded by tsl
├── results/             # CSVs from three trials (results.csv)
├── plots/               # 52 summary plots generated from results
├── logs/ (excluded)     # Large training logs and model checkpoints (45 models)
├── requirements.txt     # Python dependencies
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
2. **Uncomment lines 1657–1678**
3. Run the following command:

```bash
python src/main.py > experiment.txt
```

This will train all models and generate logs and metrics.

### 📊 Generate Results and Plots

1. Open `src/main.py`
2. **Comment lines 1657–1678 (to avoid rerunning the experiment)**
3. **Uncomment lines 1680–1683**
4. Run the following command:

```bash
python src/main.py > visualization.txt
```

This will produce:

* `results/results.csv` (aggregated results from the trial)
* `plots/` (52 `.png` summary visualizations of model errors, parameter counts, and training times)

---

## 📈 Evaluation Metrics

Each experimental run evaluates:

* Loss: `test_mae`
* Monitoring metrics: `test_mae_lag_01`, `test_mae_lag_02`, `test_mae_lag_03`
* Reference: `test_mape`
* Model architecture displaying layer structure
* Model parameter count
* Model training time

Metrics for one full run are saved in `experiment.txt`.

---

## 📦 Archive Notes

The `experiment.zip` archive contains:

* `src/` and `weather/`: duplicated from repository root
* `requirements.txt`: dependency list for installation
* `results/`: final `results.csv` recording performance from three trials
* `plots/`: 52 summary chart images (.png) generated from `results.csv`
* `data/`: raw graph dataset downloaded by `tsl`
* `dataset/`: merged dataset created from `data/` and `weather/`

> ⚠️ `logs/` is excluded due to large folder size (checkpoints for 45 models)

---

## 📜 Results

This experiment evaluated 45 spatiotemporal graph neural network models across three trials. Key findings include:

* **DiffConv + LSTM** combinations achieved the highest overall predictive accuracy.
* **Positional encoding** without attention provided consistent but marginal improvements in MAE across most architectures.
* **Attention combined with positional encoding** significantly improved predictive accuracy when used alongside LSTM or GRU temporal components.
* **Concatenative positional encoding with attention** led to catastrophic degradation when paired with vanilla RNNs. The synthesis of concatenative positional encoding, attention, and weak temporal modeling disrupted the representational capacity of the architecture, resulting in catastrophic and consistently worst-case performance across trials.

---

## 📚 Citation

Alexander, M. B. (2025). ***Evaluating Spatiotemporal Graph Neural Network Architectures for Air Quality Forecasting*** (Doctoral dissertation). The George Washington University. https://github.com/moses-b-alexander/stgnn-aqf-praxis/

---

## 👤 Author

***Moses Alexander***

Email: **m.alexander1@gwmail.gwu.edu**
GitHub: **https://github.com/moses-b-alexander/**

---

## ⚖️ License

No license — all rights reserved.
