import random
from neuron_factory import NeuronFactory

class NeuralNet():
  count = 0

  def __init__(self, n_inputs, neurons_per_layer):
    """
    Define the layers of neurons:
    - first layer includes one neuron per data input, each of which having the same number of inputs
    - hidden layers TBC
    - final layer includes a single neuron (TODO: generalize for larger numbers of classes)
    """
    NeuralNet.count += 1
    self.neuron_factory = NeuronFactory()
    self.input_count = n_inputs
    self._generateRandomNeurons(neurons_per_layer)

  def __del__(self):
    NeuralNet.count -= 1

  def _generateRandomNeurons(self, neurons_per_layer):
    self.layers = []
    for i in range(len(neurons_per_layer)):
      self.layers.append([])

      inputs_per_neuron = self.input_count
      if i > 0: inputs_per_neuron = self.neuronCount(i-1)

      for j in range(neurons_per_layer[i]):
        self.layers[i].append(
          self.neuron_factory.new(inputs_per_neuron)
        )

  def neuronCount(self, i):
    return len(self.layers[i])

  def layerCount(self):
    return len(self.layers)

  def inherit(self, neural_net_list):
    """
    Create a new Neural Net by reference to a set of Neural Nets of the same dimensions.
    """
    for n in neural_net_list:
      if not self.layerCount() == n.layerCount():
        raise IndexError(f'This Neural Net expects to inherit from Neural Nets with the same dimensions. {self.layerCount()} layers were expected but {n.layerCount()} were received.')

      for i in range(self.layerCount()):
        if not self.neuronCount(i) == n.neuronCount(i):
          raise IndexError(f'This Neural Net expects to inherit from Neural Nets with the same dimensions. {self.neuronCount(i)} neurons were expected in layer {i} but {n.neuronCount(i)} were received.')
          
    for i in range(len(self.layers)):
      for j in range(len(layer)):
        self.layers[i][j] = self.neuron_factory.combine(
          list(map(lambda x : x.layers[i][j], neural_net_list))
        )

if __name__ == "__main__":

  import unittest
  
  class NeuralNetTestMethods(unittest.TestCase):

    def test_count_neurons_created(self):
      NeuralNet.count = 0
      nn1 = NeuralNet(9, [5, 3, 1])
      self.assertEqual(NeuralNet.count, 1)
      nn2 = NeuralNet(9, [5, 3, 1])
      self.assertEqual(NeuralNet.count, 2)
      del nn1, nn2
      self.assertEqual(NeuralNet.count, 0)

    def test_requested_layers_are_filled(self):
      nn = NeuralNet(9, [5, 3, 1])
      self.assertEqual(nn.layerCount(), 3)
      self.assertEqual(nn.neuronCount(0), 5)
      self.assertEqual(nn.neuronCount(1), 3)
      self.assertEqual(nn.neuronCount(2), 1)

    def test_combined_nets(self):
      random.seed(23)
      nn1 = NeuralNet(9, [5, 3, 1])
      nn2 = NeuralNet(9, [5, 3, 1])
      nn3 = NeuralNet(9, [5, 3, 1])
      nn3.inherit([nn1, nn2])
      self.assertEqual(nn3.layerCount(), 3)

    # def test_count_is_not_int(self):
    #   with self.assertRaises(TypeError):
    #     l = NeuronLayer(4.5)

  unittest.main()
