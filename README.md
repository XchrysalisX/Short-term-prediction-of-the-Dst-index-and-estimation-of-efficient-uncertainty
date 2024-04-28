# Short-term-prediction-of-the-Dst-index-and-estimation-of-efficient-uncertainty
## Prerequisites

- Windows(11) or Linux(Ubuntu 22.04)
- python >= 3.8
- NVIDIA GPU(RTX 4090)
- CUDA 11.3
- cuDNN 8.2.1
- TensorFlow 2.9.0

## Getting Started

### Installation

- Install TensorFlow from https://pypi.org/project/tensorflow

- Install CUDA from https://developer.nvidia.com/cuda-downloads

- Install cuDNN from https://developer.nvidia.com/cudnn-downloads

- Download this repository

	Linux:

	>git clone https://github.com/XchrysalisX/Short-term-prediction-of-the-Dst-index-and-estimation-of-efficient-uncertainty
	>cd Short-term-prediction-of-the-Dst-index-and-estimation-of-efficient-uncertainty

- Unzip the data files in the dataset folder

## Run

- prediction_using_bayesian_LSTM_att.py predicts geomagnetic storms with an LSTM autocoder and an LSTM predictive model, including preprocessing of the data, model training, and evaluating the uncertainty of the model predictions.
- plotma.py mainly shows the error analysis for different prediction durations, and compares the performance of the "Miscalibration Area" and "RMSCE" metrics under different settings through line graphs.
- plotstorm14_un.py extended the data visualization further, especially in plotting predicted versus actual results, as well as calculating and displaying cases where prediction intervals exceeded the actual values.
