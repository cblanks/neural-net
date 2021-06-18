# dependencies
import pickle
import data

# main
with open("trained_network.pkl", "rb") as nn_file:
    nn = pickle.load(nn_file)
    print(f'Plus: {nn.getOutput(data.plus)} (expect +1)')
    print(f'Zero: {nn.getOutput(data.zero)} (expect 0)')
    print(f'Cross: {nn.getOutput(data.cross)} (expect -1)')
