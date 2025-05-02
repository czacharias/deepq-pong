import numpy as np
import random
from dense import Dense
from activations import Tanh, Sigmoid, ReLU, LeakyReLU
from losses import mse, mse_prime
from network import train, predict
from pong import PongGame, Ball, Paddle
from collections import deque
import copy
import ast

new_model = [
    Dense(6, 16),
    LeakyReLU(),
    Dense(16, 32),
    LeakyReLU(),
    Dense(32, 64),
    LeakyReLU(),
    Dense(64, 32),
    LeakyReLU(),
    Dense(32, 16),
    LeakyReLU(),
    Dense(16, 3)
]

game = PongGame()


def train_model(model, rounds):
    target_model = [copy.deepcopy(layer) for layer in model]
    target_update_freq = 10
    results = deque(maxlen=100)

    for _ in range(rounds):
        output, win = game.play_round(model)
        results.append(win)
        if(len(output) >= 256):
            batch= random.sample(output, 256)

            states = []
            targets = []

            for state, action, reward, next_state, done in batch:
                q_values= predict(model, state)
                next_q_values= predict(target_model, next_state)

                target = reward + (0 if done else 0.98 * np.max(next_q_values))

                q_values[action] = target

                states.append(state)
                targets.append(q_values)

            if _ % target_update_freq == 0:
                target_model = [copy.deepcopy(layer) for layer in model]

            train(model, mse, mse_prime, states, targets, epochs=12, learning_rate=0.001, verbose=False)
    
    return results

def get_rate(results):
    if results:
        return str((sum(results)/len(results)) * 100)
    
def save_model(model, results, filename=None):
    model_accuracy = get_rate(results)
    architecture = []
    architecture.append(model[0].input_size)
    for layer in model:
        if isinstance(layer, Dense):
            architecture.append(layer.output_size)
    
    if filename == None:
        filename = model_accuracy + " " + str(len(model)) + "x" + str(max(architecture))

    with open(filename, "w") as f:
        
        f.write("Architecture: " + str(architecture) + "\n")
        f.write(results + "% Accuracy\n")

        for i, layer in enumerate(model):
            if isinstance(layer, Dense):
                f.write(f"\nLayer {i} (Dense):\n")
                weights = layer.weights.flatten()
                biases = layer.bias.flatten()
                
                f.write(f"Weights: {weights.tolist()}\n")
                f.write(f"Biases: {biases.tolist()}\n")
            else:
                f.write(f"\nLayer {i} ({layer.__class__.__name__})\n")


def load_model(filename):
    new_model = []

    activations = {
        "Sigmoid": Sigmoid,
        "Tanh": Tanh,
        "ReLU": ReLU,
        "LeakyReLU": LeakyReLU,
    }

    with open(filename, "r") as f:
        architecture_line = f.readline()
        architecture = ast.literal_eval(architecture_line[architecture_line.find('['):])
        architecture = np.array(architecture, dtype=int)

        curr_layer = 0

        f.readline()

        while True:
            line = f.readline()
            if not line:
                break 

            line = line.strip()
            if line.startswith("Layer"):
                layer_type = line[line.find('(')+1:line.find(')')]

                if layer_type != "Dense":
                    activation_layer = activations[layer_type]
                    if activation_layer is None:
                        raise ValueError(f"Unknown activation: {layer_type}")
                    new_model.append(activation_layer())
                    continue

                weights_line = f.readline().strip()
                biases_line = f.readline().strip()

                weights = ast.literal_eval(weights_line[weights_line.find('['):])
                biases = ast.literal_eval(biases_line[biases_line.find('['):])

                input_size = architecture[curr_layer]
                output_size = architecture[curr_layer + 1]

                weights = np.array(weights).reshape((output_size, input_size))
                biases = np.array(biases).reshape((output_size, 1))

                new_model.append(Dense(input_size, output_size, weights, biases))

                curr_layer += 1

    return new_model

        
if __name__ == "__main__":
    #model_results = train_model(new_model, 20)
    #save_model(new_model, model_results)

    print('loading model')
    loaded_model = load_model("80.0 11x64")
    print("model loaded")
    game.main_loop(loaded_model)