# Deep Residual Time Series Forecasting

The repository for Deep Residual Time Series Forecasting model, developed for the 2020 OhioT1D competition. This includes the code, pre-trained weights, and data conversion tools.

## Overview
Our approach builds upon the AA-Forecast architecture, enhancing it with several key modifications:

•Integration of RNN blocks to capture temporal dependencies.
Use of shared output layers to streamline the forecasting process.
Incorporation of additional variables for richer input features.
Implementation of enhanced loss functions for better performance..

## Repository Contents
```
drtf.py: Core implementation of the model, including architecture, training routines, and evaluation metrics.
convert_data.py: Converts raw competition data into a suitable format for analysis.
PRETRAINS.txt: Links to pre-trained weights from the Tidepool dataset (Tidepool).
Getting Started
Clone the repository, install dependencies, convert the data, download pre-trained weights, and run the model. Here are the basic steps:
```
## Clone the repository:
```
bash
Copy code
git clone https://github.com/yourusername/deep-residual-time-series-forecasting.git
cd deep-residual-time-series-forecasting
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Convert the data:

bash
Copy code
python convert_data.py --input_path raw_data/ --output_path processed_data/
Download pre-trained weights from PRETRAINS.txt and place them in the specified directory.
```
Run the model:

bash
Copy code
python drtf.py --data_path processed_data/ --weights_path pre_trained_weights/
Contribution
We welcome contributions. Open an issue or submit a pull request for suggestions or new features.

