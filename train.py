"""
Plan
----
First layer neurons each receive one input for each array value, from 'cross' or 'plus'.
  We will start with three neurons in this layer.
  Each of these neurons has been initialized with random parameters.
  Each neuron produces a result.
Hidden layer neurons (fewer than in first layer) each receive one input from each first layer neuron.
  We will start with two neurons in this layer.
  Each of these neurons has been initialized with random parameters.
  Each neuron produces a result.
Final layer contains one single neuron that will receive one input from each hidden layer neuron.
  This neuron has been initialized with random parameters.
  This neuron will produce a result, 1 for 'cross' and 0 for 'plus'.

We have a lot of parameters to optimize here:
  9 + 1 on each of the 3 first layer neurons = 30
  3 + 1 on each of the 2 hidden layer neurons = 8
  2 + 1 on the final layer neuron = 3
  TOTAL = 41

> Don't think neuron outputs can be binary otherwise what are we optimizing towards?

Can we try with one single neuron first?

Plan 0
------
Neuron receives 9 inputs, so 9 weights plus one bias = 10 free parameters, each 0 <= x <= 1.
Each tanh neuron emits a result in the range -1 to 1.
Aim that -1 corresponds to 'cross' and 1 to 'plus'.

Try 6 independent runs and compare the final results:
- score = RMS( score_for_cross + 1, 1 - score_for_plus )

From the top 4 (ranked by minimum score) calculate averages of each parameter from all unique pairings:
- 43, 42, 41, 32, 31, 21

Use these values to run 6 more trials.
Repeat.

Back to Plan 1. This process means it doesn't matter how many parameters I have, I will just average each pair. OK!
"""
from lib.layer import NeuronLayer
import layer

network_states = []

state0 = [
  NeuronLayer(1)
]

