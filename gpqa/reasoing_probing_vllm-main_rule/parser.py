
# ============================================================
# 1. GRADER CODE
# ============================================================
import re
import regex
import multiprocessing
from math import isclose
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex

try:
    from latex2sympy2 import latex2sympy
except ImportError:
    def latex2sympy(s, variable_values={}):
        return parse_expr(s)

def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None

def is_digit(num):
    return parse_digits(num) is not None

def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []
    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)
    return ", ".join(pmatrix_list)

def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, rel_tol=1e-4)

def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s
    a = _parse(a)
    b = _parse(b)
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass
    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass
    try:
        if hasattr(a, 'shape') and hasattr(b, 'shape'):
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass
    return False

def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)

def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    return output_queue.get()

def math_equal(prediction, reference, include_percentage=True, is_close=True, timeout=False) -> bool:
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    try:
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass
    if not prediction and prediction not in [0, False]:
        return False
    reference = str(reference).strip()
    prediction = str(prediction).strip()
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)
    pred_str, ref_str = prediction, reference
    if ((prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or
        (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("["))):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True
    if (regex.match(r"(\(|\[).+(\)|\])", prediction) is not None and
        regex.match(r"(\(|\[).+(\)|\])", reference) is not None):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all([math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                    for i in range(len(pred_parts))]):
                return True
    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference):
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close):
            return True
    elif (reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and "=" not in prediction):
        if math_equal(prediction, reference.split("=")[1], include_percentage, is_close):
            return True
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True
    return False

def math_equal_process_wrapper(param):
    # Expecting param = (idx, predicted_answer, gold_answer)
    return math_equal(param[-2], param[-1])


# ============================================================
# 2. PARSER CODE
# ============================================================
import random
import sympy
from typing import Any, Dict
from word2number import w2n

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string

def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text

unit_texts = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m", "deg",
    "mile", "q .", "monkey", "prime", "ratio", "profit of rs", "rd", "o", "gm",
    "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab", "way", "west", "a .",
    "b .", "c .", "d .", "e .", "f .", "g .", "h .", "t", "a", "h", "no change",
    "men", "soldier", "pie", "bc", "excess", "st", "inches", "noon", "percent", "by",
    "gal", "kmh", "c", "acre", "rise", "a . m", "th", "π r 2", "sq", "mark", "l",
    "toy", "coin", "sq . m", "gallon", "° f", "profit", "minw", "yr", "women",
    "feet", "am", "pm", "hr", "cu cm", "square", "v â € ™", "are", "rupee", "rounds",
    "cubic", "cc", "mtr", "s", "ohm", "number", "kmph", "day", "hour", "minute",
    "min", "second", "man", "woman", "sec", "cube", "mt", "sq inch", "mp", "∏ cm ³",
    "hectare", "more", "sec", "unit", "cu . m", "cm 2", "rs .", "rs", "kg", "g",
    "month", "km", "m", "cm", "mm", "apple", "liter", "loss", "yard", "pure", "year",
    "increase", "decrease", "d", "less", "Surface", "litre", "pi sq m", "s .",
    "metre", "meter", "inch",
]
unit_texts.extend([t + "s" for t in unit_texts])

def strip_string(string, skip_unit=False):
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string
    if not skip_unit:
        for _ in range(2):
            for unit_text in unit_texts:
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")
    string = convert_word_number(string)
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if (string.startswith("{") and string.endswith("}") and string.isalnum()) or \
       (string.startswith("(") and string.endswith(")") and string.isalnum()) or \
       (string.startswith("[") and string.endswith("]") and string.isalnum()):
        string = string[1:-1]
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = re.sub(r"\\mbox{.*?}", "", string)
    string = string.replace("'", "")
    string = string.replace('"', "")
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string

def extract_answer(pred_str, data_name):
    if not isinstance(pred_str, str):
        pred_str = str(pred_str)
    pred_str = pred_str.replace("\u043a\u0438", "")
    # In our case we assume math problems use LaTeX boxing.
    pred_str = re.split(r"\n\s*(?:Human|Assistant)\s*:", pred_str)[0]

    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "answer is" in pred_str.lower():
        pred = pred_str.split("answer is")[-1].strip()
    else:
        pattern = "-?\d*\.?\d+"
        pred = re.findall(pattern, pred_str.replace(",", ""))
        if len(pred) >= 1:
            pred = pred[-1]
        else:
            pred = ""
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred

def parse_ground_truth(example: Dict[str, Any], data_name):
    if "gt_cot" in example and "gt" in example:
        if data_name in ["math", "math-oai", "minerva_math", "amps", "hungarian_exam"]:
            gt_ans = extract_answer(example["gt_cot"], data_name)
        elif data_name in ["carp_en", "minerva_math"]:
            gt_ans = example["gt"]
        else:
            gt_ans = strip_string(example["gt"])
        return example["gt_cot"], gt_ans
    if data_name in ["math", "math-oai", "minerva_math", "amps", "hungarian_exam"]:
        gt_cot = example["solution"]
        gt_ans = extract_answer(gt_cot, data_name)
    elif data_name in ["mathqa"]:
        gt_cot = example["rationale"]
        gt_ans = example["correct"].upper()
    elif data_name == "gsm8k":
        gt_cot, gt_ans = example["answer"].split("####")
    elif data_name == "gsm-hard":
        gt_cot, gt_ans = example["code"], example["target"]
    elif data_name == "svamp":
        gt_cot, gt_ans = example["Equation"], example["Answer"]
    elif data_name == "asdiv":
        gt_cot = example["formula"]
        gt_ans = re.sub(r"\(.*?\)", "", example["answer"])
    elif data_name == "mawps":
        gt_cot, gt_ans = None, example["target"]
    elif data_name == "tabmwp":
        gt_cot = example["solution"]
        gt_ans = example["answer"]
    elif data_name == "bbh":
        gt_cot, gt_ans = None, example["answer"].replace("(", "").replace(")", "")
    elif data_name in ["theorem-qa", "math_collection", "arc"]:
        gt_cot, gt_ans = None, example["answer"]
    elif data_name == "carp_en":
        gt_cot, gt_ans = example["steps"], example["answer"]
    elif data_name == "tal_sc_en":
        gt_cot, gt_ans = example["answer_analysis"][0].strip()
    elif data_name == "mmlu_stem":
        abcd = "ABCD"
        gt_cot, gt_ans = None, abcd[example["answer"]]
    elif data_name == "sat_math":
        gt_cot, gt_ans = None, example["Answer"]
    elif data_name == "aqua":
        gt_cot, gt_ans = None, example["correct"]
    else:
        raise NotImplementedError(f"`{data_name}`")
    gt_cot = str(gt_cot).strip()
    if data_name not in ["minerva_math", "carp_en"]:
        gt_ans = strip_string(gt_ans, skip_unit=data_name == "carp_en")
    else:
        gt_ans = gt_ans.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    return gt_cot, gt_ans

# ============================================================
# 3. LATEX2SYMPY2 CODE
# ============================================================
import sympy
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener
try:
    from gen.PSParser import PSParser
    from gen.PSLexer import PSLexer
except Exception:
    pass
from sympy.parsing.sympy_parser import parse_expr

def latex2sympy(sympy_str: str, variable_values={}):
    sympy_str = sympy_str.replace(r'\dfrac', r'\frac')
    sympy_str = sympy_str.replace(r'\tfrac', r'\frac')
    sympy_str = sympy_str.replace(r'\left', '')
    sympy_str = sympy_str.replace(r'\right', '')
    try:
        return parse_expr(sympy_str)
    except Exception as e:
        return sympy.sympify(sympy_str)


