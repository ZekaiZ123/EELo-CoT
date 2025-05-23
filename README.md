# EELo-COT

This repository contains code for performing EELo-COT in a language model during text generation. The system allows configuring different types of intervention functions through a JSON configuration file.

## Environment
install Python 3.10.16  
install Pytorch 2.7.0  
install Transformer 4.51.3  
```python
pip install torch==2.7.0 torchvision==0.18.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu121
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

## Running the Code
First create a grid1 folder under the configs folder.
Then, run 
```python
python EELo-CoT/gpqa/raasoning_probing_vllm-main_multi_layer/create_grid.py
```

Finally, run 
```python
python EELo-CoT/gpqa/raasoning_probing_vllm-main_multi_layer/inference_probing.py \
  --config EELo-CoT/gpqa/raasoning_probing_vllm-main_multi_layer/configs/grid1 \
  --num_gpus 7 \
  --output_dir EELo-CoT/gpqa/raasoning_probing_vllm-main_multi_layer/results
```


