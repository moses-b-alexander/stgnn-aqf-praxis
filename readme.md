# ***Evaluating Spatiotemporal Graph Neural Network Architectural Design for Urban Air Pollution Forecasting***

This repository supports the doctoral dissertation titled **Evaluating Spatiotemporal Graph Neural Network Architectural Design for Urban Air Pollution Forecasting**, available at **https://github.com/moses-b-alexander/stgnn-aqf-praxis**.

This repository contains the full experimental setup for reproducing the results of my research on spatiotemporal graph neural networks for air pollution forecasting. It includes source code files and a zipped folder containing the experimental archive.

This repository includes:

* `src/`: source code for running and evaluating experiment
* `experiment.zip`: archive containing a full copy of the repository structure from the experiment
* `requirements.txt`: dependency list for installation via Python 3.12 pip
* `README.md`: document describing purpose and content of repository

---

## Objective

Air pollution, exacerbated by urban activity and layout, harms human health. The World Health Organization reported that 4.3 million deaths in 2012 could be attributed to reduced life expectancy and long-term medical conditions resulting from PM2.5 pollution (Jiang et al, 2016).

STGNNs forecast PM2.5 concentration values from historical data recorded by air quality sensor networks. By modeling spatiotemporal patterns using air quality sensor network data, STGNNs support policymakers in more precise understanding of urban air pollution dynamics (Zheng et al, 2015).

* Develop a STGNN model to forecast PM2.5 pollution concentration values in PPM
* Enhance input representations with additive and concatenative positional encoding
* Apply bidimensional attention mechanism to refine spatiotemporal representations
* Investigate the accuracy and utility of STGNNs for forecasting air pollution

---

## Design

Each model architecture consists of four components:

* **Positional Encoding**: Applied to raw input features using either additive or concatenative fusion to enhance input representation
* **Spatial Component**: A graph convolutional network layer: one of GraphConv, DiffConv, or GATConv
* **Temporal Component**: A recurrent neural network layer: one of RNN, LSTM, or GRU
* **Bidimensional Attention Mechanism**: Applied at the end to refine joint spatiotemporal representation; decoded by a linear layer

Model architectural configurations differ in positional encoding strategy, spatial-temporal module combination, and whether attention is applied. The hyperparameter definitions and hardware setup were constant throughout the experiment.

---

## Methodology

The air quality sensor network data were obtained from the Urban Air Project by Microsoft Research (2014–2015) *https://www.microsoft.com/en-us/research/project/urban-air/*, and exogenous covariate climatology data for the relevant timeframe were retrieved from the World Bank Group's Climate Knowledge Portal *https://climateknowledgeportal.worldbank.org/country/china/climate-data-historical*. The models were developed using Torch Spatiotemporal library (tsl): https://torch-spatiotemporal.readthedocs.io/en/latest/index.html. Several common Python machine learning-related frameworks were also employed: PyTorch, Pandas, NumPy, Einops, Matplotlib, and others.

1. **Load datasets**
2. **Define air quality sensor network data**

   * Construct edges using distance threshold
   * Apply masking for missing graph features
   * Specify forecasting horizon and training window length
3. **Preprocess air quality sensor network data**

   * Standardize PM2.5 concentration values
4. **Preprocess exogenous covariate climatology data**

   * Align covariate feature sequence lengths through duplication by hour
5. **Merge datasets**
6. **Prepare model**

   * Split data into training, validation, and test datasets
   * Set batch size
   * Define loss metric and diagnostic metrics
   * Configure optimization strategy
7. **Establish logging and checkpointing for model**
8. **Train and test Model**
9. **Evaluate Output**

   * Analyze forecasting error and diagnostic metrics
   * Review training and evaluation logs

---

## Archive

```
experiment.zip
├── src/                 # Source code (1) file in .PY format (duplicate of root src/)
├── weather/             # Exogenous covariate climatology data (2) files in .CSV format (duplicate of root weather/)
├── data/                # Air quality sensor network data (2) files in .H5 and .NPY formats and downloaded using tsl module
├── dataset/             # Final processed dataset (1) file used for training in .PT format (merged from data in files within data/ and weather/)
├── table/               # Table of forecasting error, diagnostic metrics, model parameter count, and training duration (1) file in CSV format, averaged over 3 trials for each of 45 models
├── plots/               # Summary visualization (52) files in PNG format of experimental data recorded in table/
├── requirements.txt     # Python 3.12 pip dependency list
├── logs/ (excluded)     # Training logs and model checkpoints for all 3 trials of 45 models each
```

---

## Usage

Ensure you are using **Python 3.12**.

### Installation

```bash
pip install -r requirements.txt
```

### Run the experiment

1. Open `src/main.py`
2. **Uncomment lines 1660–1681**
3. Run the following command:

```bash
python src/main.py > experiment.txt
```

This will train all 45 models once, with the defined hyperparameters.

### Generate table of results and summary plots

1. Open `src/main.py`
2. **Comment lines 1660–1681 (to avoid rerunning the experiment)**
3. **Uncomment lines 1683–1686**
4. Run the following command:

```bash
python src/main.py > visualization.txt
```

This will produce the table of prediction errors and diagnostic metrics from the trial and 52 summary visualizations of forecasting mean absolute errors, model parameter counts, and training durations.

---

## Evaluation

Each experimental run evaluates:

* Loss: `test_mae`
* Model architectural structure of component layers
* Model parameter count
* Model training time
* Diagnostic metrics: `test_mae_lag_01`, `test_mae_lag_02`, `test_mae_lag_03`
* Reference: `test_mape`

---

## Results

Key findings include:

* **DiffConv + LSTM** combinations achieved the highest overall predictive accuracy, with **DiffConv + GRU** combinations following closely behind. Both of these architectural configurations included both positional encoding and attention.
* **Positional encoding** without attention, either additive or concatenative mode, provided consistent but marginal improvements in MAE across most architectural configurations.
* **Attention combined with either positional encoding mode** significantly and consistently improved predictive accuracy when used in combination with LSTM or GRU temporal components.
* **Concatenative positional encoding with attention** led to catastrophic degradation in forecasting accuracy when combined with RNNs as the temporal component. The combination of concatenative positional encoding, weak temporal modeling, and attention severely undermined the representational quality of the learned spatiotemporal embeddings. This resulted in systematically poor and considerably diminished performance overall, due to a poorly structured and miscalibrated embedding space.

---

## Citation

Alexander, M. B. (2025). *Evaluating Spatiotemporal Graph Neural Network Architectural Design for Urban Air Pollution Forecasting* (Doctoral dissertation). The George Washington University. https://github.com/moses-b-alexander/stgnn-aqf-praxis/

---

## License

No license — all rights reserved.











