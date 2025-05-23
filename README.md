# EELo-COT

This repository contains code for performing EELo-COT in a language model during text generation. The system allows configuring different types of intervention functions through a JSON configuration file.

## Environment
install Python 3.10.16
install Transformer 4.51.3
install Pytorch 2.7.0
```python
pip install transformers==4.51.3
```

## Repository Structure
```
├── inference_probing.py     # Main code for running inference with interventions
├── intervene_functions.py   # Implementation of intervention functions
├── model.py                 # Model definition with intervention support
├── intervene_config.json    # Configuration file for interventions
└── results/                 # Directory for output results
```


