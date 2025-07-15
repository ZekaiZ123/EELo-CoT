import json
import copy
acc_len = [40]
ref_factor = [2]


original_config = """{
  "model": "Qwen/Qwen2.5-7B",
  "intervene_functions": [
    {
      "type": "DeltaPatchIntervene",
      "target_layers": [22, 23, 24, 25, 26, 27],
      "delta_files": {
        "22": "/data/zekai/EELo-CoT_full_activations_extraction/activation_deltas/delta_layer22.pt",
        "23": "/data/zekai/EELo-CoT_full_activations_extraction/activation_deltas/delta_layer23.pt",
        "24": "/data/zekai/EELo-CoT_full_activations_extraction/activation_deltas/delta_layer24.pt",
        "25": "/data/zekai/EELo-CoT_full_activations_extraction/activation_deltas/delta_layer25.pt",
        "26": "/data/zekai/EELo-CoT_full_activations_extraction/activation_deltas/delta_layer26.pt",
        "27": "/data/zekai/EELo-CoT_full_activations_extraction/activation_deltas/delta_layer27.pt"
      }
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
