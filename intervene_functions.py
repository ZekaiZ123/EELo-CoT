import torch
import math
from transformers import PreTrainedTokenizer
    
class ConstantInterveneFunction:
    def __init__(
        self,
        amp: float,
        neuron_list: list[int],
        layer_list: list[int],
        n_neurons:int,
        modules = None
    ):
        assert len(neuron_list) == len(layer_list)
        self.amp = amp
        self.neuron_list = neuron_list[:n_neurons]
        self.layer_list = layer_list[:n_neurons]
        self.modules = modules
        self.handles: list[torch.utils.hooks.RemovableHandle] = []


    def __call__(self, last_token_id: int):
        for h in self.handles:
            h.remove()
        self.handles.clear()

        for neuron, layer in zip(self.neuron_list, self.layer_list): 
            if layer < 0 or layer >= len(self.modules):
                continue
            module = self.modules[layer]

            def _hook(mod, inputs, output, neuron=neuron):
                output[..., neuron] *= self.amp
                return output

            handle = module.register_forward_hook(_hook)
            self.handles.append(handle)

        return self.handles

class KeywordInterveneFunction:
    def __init__(self, amp:float,
                 top_neurons:list[int],
                 n_neurons:int,
                 keywords:list[int],
                 module = None):
        self.amp = amp
        self.handle = None
        self.module = module
        self.n_neurons = n_neurons
        self.top_neurons = top_neurons[:n_neurons]
        self.keywords = set(keywords)
    
    def __call__(self, last_token_id:int):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        def _hook(module, inputs, output):
            output[..., self.top_neurons] *= self.amp
            return output
        if last_token_id in self.keywords:
            self.handle = self.module.register_forward_hook(_hook)
        return self.handle

class ConstantDecayInterveneFunction:
    def __init__(self, amp:float,
                 top_neurons:list[int],
                 n_neurons:int,
                 t_max:int,
                 module = None):
        self.amp = amp
        self.handle = None
        self.module = module
        self.n_neurons = n_neurons
        self.top_neurons = top_neurons[:n_neurons]
        self.t = 0
        self.t_max = t_max
    
    def __call__(self, last_token_id:int):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        f = max(0.0, -self.t / self.t_max + 1.0)
        def _hook(module, inputs, output):
            output[..., self.top_neurons] *= (f * self.amp + 1.0)
            return output
        self.t += 1
        self.handle = self.module.register_forward_hook(_hook)
        return self.handle

class KeywordDecayInterveneFunction:
    def __init__(self, amp:float,
                 top_neurons:list[int],
                 layer_list: list[int],
                 n_neurons:int,
                 t_max:int,
                 keywords:list[int],
                 t_initial:int,
                 cool_down:int,
                 modules: list[torch.nn.Module],
                 tokenizer: PreTrainedTokenizer):
                 #modules: list[torch.nn.Module] = None):
        assert len(top_neurons) == len(layer_list), "neuron_list 与 layer_list same length"
        self.amp = amp
        self.top_neurons = top_neurons[:n_neurons]
        self.layer_list = layer_list[:n_neurons]
        self.modules = modules or [] 
        self.tokenizer = tokenizer
        self.handles = []
        self.handle = None
        self.n_neurons = n_neurons
        self.t_max = t_max
        self.t_initial = t_initial
        self.cool_down = cool_down
        self.t = 0
        self.current_pin = 0
        self.keywords = keywords
        
    def _schedule_factor(self, t: int) -> float:
        #a, b, c = 0.3170, 0.030, -0.9997
        a, b, c = 0.170, 0.033, -0.9997
        denom = t + c
        if denom <= 0:
            denom = 1e-3
        return a - b * math.log(denom)
    
    def __call__(self, last_token_id: int):
        # 1) 清理旧的 hook
        for h in self.handles:
            h.remove()
        self.handles.clear()

        # 2) decode token 并检测数字
        token_str = self.tokenizer.decode([last_token_id], skip_special_tokens=False)
        saw_number = any(ch.isdigit() for ch in token_str)

        # 3) 如果满足 t_initial、cool_down 且看到了数字，就重置 trigger
        if (
            self.t > self.t_initial
            and self.t > self.current_pin + self.cool_down
            and saw_number
        ):
            self.current_pin = self.t

        # 4) 在 t_max 步内插入干预
        elapsed = self.t - self.current_pin
        if self.current_pin != 0 and elapsed <= self.t_max:
            f = self._schedule_factor(elapsed)
            for neuron, layer in zip(self.top_neurons, self.layer_list):
                if 0 <= layer < len(self.modules):
                    submod = self.modules[layer]
                    def _hook(module, inputs, output, neuron=neuron, factor=f):
                        output[..., neuron] *= (factor * self.amp + 1.0)
                        return output
                    handle = submod.register_forward_hook(_hook)
                    self.handles.append(handle)

        # 5) 时间步推进
        self.t += 1
        return self.handles
    
    
            