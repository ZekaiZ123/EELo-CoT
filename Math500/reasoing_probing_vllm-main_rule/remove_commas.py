file_name = "./data/math_symbols.txt"
with open(file_name, "r") as f:
    neurons = f.readlines()

neurons = [neuron.split(",")[0] for neuron in neurons]

with open(file_name, "w") as f:
    for neuron in neurons:
        f.write(f"{neuron}\n")
