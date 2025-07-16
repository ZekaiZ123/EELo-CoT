import json
import copy
acc_len = [40]
ref_factor = [2]


original_config = """{
  "model": "Qwen/Qwen2.5-7B",
  "layer_idx": 27,
  "intervene_functions": [
    {
      "type": "KeywordDecayIntervene",
      "amp": 1.0,
      "top_neurons_file": "/home/zekai/EELo-CoT/Activation_data/Qwen2.5_7b_Activations_index.txt",
      "layer_list_file": "/home/zekai/EELo-CoT/Activation_data/Qwen2.5_7b_Activations_layer_index.txt",
      "keywords": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 
      "t_max": 100, 
      "t_initial": 5, 
      "n_neurons": 100, 
      "cool_down": 10 
    }
  ],
  "generation_params": {
    "max_new_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.9,
    "_sentence_cooldown": 4,
    "do_sample": true
  }
}"""

json_config = json.loads(original_config)

for num_acc in acc_len:
    for ref_fac in ref_factor:
        json_config_copy = copy.deepcopy(json_config)
        json_config_copy["intervene_functions"][0]["amp"] = ref_fac
        json_config_copy["intervene_functions"][0]["n_neurons"] = num_acc
        json.dump(json_config_copy, open(f"/home/zekai/EELo-CoT/configs/Qwen2.5-7B_self_neuron_neuron{num_acc}_factor_{ref_fac}_cold4.json", "w"))
