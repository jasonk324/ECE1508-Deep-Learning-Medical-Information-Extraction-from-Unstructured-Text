# Project Code Overview

There are 5 main script used within this project: 
1. `data_processing/experiement.py`: Output experimentation output of using different LLMs
2. `data_processing/label.py`: Label all of the data and save them to `data/processed/sim_sum_gem3_processed.jsonl`
3. `model_processing/vocabulary.py`: Analyze the vocabulary generated from preprocessing
4. `model_processing/train.py`: Train a model for specified parameters
5. `model_processing/inference.py`: Inference on the model and genreate test accuracy

Within the "Quick Start" section, it will be mentioned how to run these piece of code however,to simplify viewing each of these at a time, `Exmaple Notebook.ipynb` was created to give an overview of each of these files pasted all within one notebook file.

```

Project Code/
├── data/
│ ├── processed/
│ │ ├── demo_experiment/                   # Duplicate of experiement folder for the Example Notebook.ipynb
│ │ │ ├── ...                                 
│ │ ├── experiment/                        # Experiment resuls of trying to label data with different LLMs
│ │ │ ├── gemma_processed.jsonl            # Results from Gemma3   12B
│ │ │ ├── llama_processed.jsonl            # Results from Llama3.1 8B
│ │ │ └── qwen_processed.jsonl             # Results from Qwen3    8B
│ │ ├── demo_sim_sum_gem3_processed.jsonl  # Duplicate processed file for Example Notebook.ipynb
│ │ ├── sim_sum_gem3_processed.jsonl       # Full processed data from using Gemma3 12B
│ │ └── ...
│ ├── raw/
│ │ └── SimSUM.csv               # Raw data for SimSUM dataset installed
│
├── data_processing/           
│ ├── pycache/           
│ ├── init.py           
│ ├── functions.py               # Helper functions used throughout this folder
│ ├── experiment.py              # Script for experimenting results of various LLMs
│ ├── label.py                   # Script for labelling information
│ └── prompt.py                  # System prompt for LLM agent
│
├── env/
│ └── ...                        # Virtual enviroment
│
├── model_processing/  
│ ├── pycache/ 
│ ├── init.py
│ ├── classes.py                 # Contains data and model classes used
│ ├── data_functions.py          # Helper functions 
│ ├── inference_functions.py     # Helper functions 
│ ├── model_functions.py         # Helper functions 
│ ├── inference.py               # Script to inference on an existing model
│ ├── train.py                   # Script to train a model
│ └── vocabulary.py              # Script to investigate the generated vocabulary
│
├── models/
│ ├── trial_1
│ │ │ ├── model.pt               # Saved weights from outputed model
│ │ │ ├── training_data.csv      # Lose curve training changes
│ │ │ ├── training_info.txt      # Notes captured about parameters used to train that model
│ │ │ └── training_plot.png      # png image of the lose curve graph genereated from training
│ └── ...
│
├── Example Notebook.ipynb       # Example notebook showing all the code work at once
├── README.md                    # Code Project information file
└── requirements.txt             # Python library requirements for this project
```
---

## Quickstart

### 1) Install dependencies

Navigate into the `/Project Code/` directory and use the following:

**Windows (PowerShell)**
```powershell
python -m env venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

This project was written using PyTorch with CUDA support so if your system does not have CUDA then a separate version of PyTorch install might need to be considered but can be installed from [PyTorch Official Website](https://pytorch.org/).

### 2) Running Jupyter Notebook 

Open `Example Notebook.ipynb` with an method of your choice and ensure that the virtual enviroment is selected. At which point you should be able to succesfully run the entire notebook.

### 3) Running Scripts

All scripts were written so that they can be called from the parent folder `/Project Code/` using the following command format:

```
python -m {folder_name}.{file_name}
```

## Specific notes on some results

### LLM Experimentation Script

File: `data_processing/experiement.py`

Recieved the following output result:
```
Processing with model llama completed in 96.63 seconds.
Processing with model gemma completed in 203.33 seconds.
Processing with model qwen completed in 569.33 seconds.

Model: llama, Success Count: 1, Failure Count: 19
Model: gemma, Success Count: 13, Failure Count: 7
Model: qwen, Success Count: 10, Failure Count: 10
```
### Training Script

File: `model_processing/train.py` 

Each time a model was trained it would be outputted to the `models/` folder with 4 files generated:
1. `model.pt`: Saved weights of the model
2. `training_data.csv`: Loss curve calculated during trianing
3. `training.txt`: Parameters used in training
4. `training_plot.png`: Image of the loss curve generated

The output file `training.txt` was updated a few time when training different models so not all the files are consistent. 

The report mentioned 4 different loss graphs for 4 different models selected. These models exist in the following trial numbers

* Default: `trial_2`
* Regularization: `trial_8`
* Min Frequency set to 5: `trial_9`
* Early stop: `trial_18`