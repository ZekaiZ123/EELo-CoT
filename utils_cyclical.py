# --- utils_cyclical.py  (new file) ------------------------------------------
import math
from transformers import LogitsProcessor

class CyclicalWaitSchedule:
    """
    Lightweight helper that returns a +/- penalty for the current position.
    Positive → we want to encourage 'Wait', negative → discourage.
    """
    def __init__(self, amplitude=3.0, period=600, shift=0.0, phi=None):
        self.A   = float(amplitude)
        self.T   = float(period)
        self.phi = phi or [(0, 10_000)]   # list of (start, end) windows
        self.shift = shift

    # ---------------------------------------------------------------------- #
    def __call__(self, pos):
        if not any(s <= pos < e for s, e in self.phi):
            return 0.0                         # outside the active window
        # map position to [0, 1)
        x = ((pos + self.shift * self.T) % self.T) / self.T
        if x <= 0.25:          p =  (x / .25) * self.A
        elif x <= 0.75:        p =  self.A - (x - .25) / .5 * 2 * self.A
        else:                  p = -self.A + (x - .75) / .25 * self.A
        return p


class CyclicalWaitLogitsProcessor(LogitsProcessor):
    """
    Plug-compatible with HF generation.  Adds the schedule value to the
    logits of the chosen wait tokens.
    """
    def __init__(self, tokenizer, schedule: CyclicalWaitSchedule,
                 wait_strs=("wait", "Wait", "but", "But", "Alternatively")):
        self.ids = [tokenizer.convert_tokens_to_ids(s) for s in wait_strs]
        self.end_think = tokenizer.convert_tokens_to_ids("</think>")
        self.schedule  = schedule

    # ---------------------------------------------------------------------- #
    def __call__(self, input_ids, scores):
        if self.end_think in input_ids[0].tolist():     # stop once </think>
            return scores
        pos = input_ids.shape[1]                        # current length
        penalty = self.schedule(pos)
        if penalty != 0.0:
            scores[:, self.ids] += penalty
        return scores
