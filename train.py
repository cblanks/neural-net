# dependencies
import math
import pickle
from lib.neural_net import NeuralNet
import data

# helper functions


def rms(X):
    sum_xx = 0.0
    for x in X:
        sum_xx += x*x

    return math.sqrt(sum_xx)


# options
nets_per_step = 100
top_n_nets = 10

# main
best_nets = []
minimal_loss = math.inf

n = 0
while n < 100:
    n += 1

    # 1. get outputs from a set of randomly generated networks
    trials = []
    for i in range(nets_per_step):
        nn = NeuralNet(9, [5, 3, 1])
        if len(best_nets) > 0:
            nn.deriveFrom(
                list(map(lambda x: x["net"], best_nets))
            )

        checks = [
            {"result": nn.getOutput(data.cross), "target": -1},
            {"result": nn.getOutput(data.plus), "target": 1},
            {"result": nn.getOutput(data.zero), "target": 0}
        ]

        trials.append({"net": nn, "score": rms(
            list(map(lambda x: (x["target"] - x["result"]), checks))
        )})

    # 2. rank nets by output score, update record of top 10 nets
    for trials in trials:
        best_nets.append(trials)

    # print(best_nets)
    sorted_nets = sorted(best_nets, key=lambda x: x["score"])
    best_nets = sorted_nets[0:10]

    print(f'Loss: {best_nets[0]["score"]}')
    if best_nets[0]["score"] < 1e-9:
        break

# finally save best nn in a pkl file to be used for testing
with open("trained_network.pkl", "wb") as nn_file:
    pickle.dump(best_nets[0]["net"], nn_file)
