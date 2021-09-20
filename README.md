# neural-net

A project to help me think about the fundamentals of neural nets.  Aim to build one from scratch to differentiate between two simple b/w images.

## Plan

1. First layer neurons each receive one input for each array value, from 'cross' or 'plus'.
   - We will start with three neurons in this layer.
   - Each of these neurons has been initialized with random parameters.
   - Each neuron produces a result.
2. Hidden layer neurons (fewer than in first layer) each receive one input from each first layer neuron.
   - We will start with two neurons in this layer.
   - Each of these neurons has been initialized with random parameters.
   - Each neuron produces a result.
3. Final layer contains one single neuron that will receive one input from each hidden layer neuron.
   - This neuron has been initialized with random parameters.
   - This neuron will produce a result, 1 for 'cross' and 0 for 'plus'.

> Don't think neuron outputs can be binary otherwise what are we optimizing towards?
> Some reading suggests a tanh function to give a tight s-curve output between -1 and 1.

We have a lot of parameters to optimize here:
- 9 + 1 on each of the 3 first layer neurons = 30
- 3 + 1 on each of the 2 hidden layer neurons = 8
- 2 + 1 on the final layer neuron = 3
- TOTAL = 41


## Plan 0

Can we try with one single neuron first?

1. Neuron receives 9 inputs, so 9 weights plus one bias = 10 free parameters, each 0 <= x <= 1.
   - Each tanh neuron emits a result in the range -1 to 1.
   - Aim that -1 corresponds to 'cross' and 1 to 'plus'.
2. Try 6 independent runs and compare the final results:
   - score = RMS( score_for_cross + 1, 1 - score_for_plus )
3. From the top 4 (ranked by minimum score) calculate average of each parameter
4. Use these values to see parameters for new neurons and run 6 more trials.
5. Repeat.

> Back to Plan 1. This process means it doesn't matter how many parameters I have, I will just average each parameter to seed new iterations of the network. OK!

> Am I really training a network or simply finding a network configuration which works? Is there a difference?

## TODO

- [ ] Try ReLU vsatan vas sigmoid.
