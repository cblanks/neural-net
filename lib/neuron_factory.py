import random
from neuron import Neuron

class NeuronFactory():

  def new(self, input_count):
    return Neuron(
      self._bias(),
      self._newWeights(input_count)
    )

  def combine(self, neuron_list):
    if len(neuron_list) < 2:
      raise IndexError(f'This Neuron Factory expects to combine two or more neurons but received {len(neuron_list)}.')

    return Neuron(
      self._bias(self._average(
        list(map(lambda x : x.bias, neuron_list))
      )),
      self._combineWeights(
        list(map(lambda x : x.input_weights, neuron_list))
      )
    )

  def count(self):
    return Neuron.count

  def _average(self, number_list):
    sum = 0.0
    for n in number_list:
      sum += n

    return sum / float(len(number_list))

  def _bias(self, mode=0.5):
    return random.triangular(0.0, 1.0, mode)

  def _newWeights(self, n, mode=0.5):
    if n < 1:
      raise ValueError(f'This Neuron Factory can make neurons with one or more inputs but {n} were requested.')

    new_weights = []
    for i in range(n):
      new_weights.append(random.triangular(0.0, 1.0, mode))

    return new_weights

  def _combineWeights(self, weights_list):
    for i in range(1, len(weights_list)):
      if not len(weights_list[i]) == len(weights_list[0]):
        raise IndexError(f'This Neuron Factory expects to combine neurons with the same number of inputs but received neurons with {len(weights_list[i])} and {len(weights_list[0])} inputs respectively.')

    new_weights = []
    for i in range(len(weights_list[0])):
      new_weights.append(
        random.triangular(0.0, 1.0, self._average(
          list(map(lambda x : x[i], weights_list))
        ))
      )

    return new_weights

if __name__ == "__main__":

  import unittest
  
  class NeuronFactoryTestMethods(unittest.TestCase):

    def test_count_neurons_created(self):
      Neuron.count = 0
      f = NeuronFactory()
      n1 = f.new(3)
      self.assertEqual(Neuron.count, 1)
      n2 = f.new(5)
      self.assertEqual(Neuron.count, 2)
      self.assertEqual(f.count(), 2)
      del n1, n2
      self.assertEqual(f.count(), 0)

    def test_too_few_inputs_throws(self):
      f = NeuronFactory()
      with self.assertRaises(ValueError):
        n = f.new(-1)
  
    def test_bias_is_combined(self):
      f = NeuronFactory()
      random.seed(23)
      n1 = f.new(3)
      n2 = f.new(3)
      n3 = f.combine([n1, n2])
      self.assertEqual(n3.bias, 0.3606006159355911)

    def test_weights_are_combined(self):
      f = NeuronFactory()
      random.seed(23)
      n1 = f.new(2)
      n2 = f.new(2)
      n3 = f.combine([n1, n2])
      self.assertEqual(n3.input_weights[0], 0.3007214453148091)
      self.assertEqual(n3.input_weights[1], 0.3433995076781547)
  
    def test_neurons_with_different_input_count_cannot_be_combined(self):
      f = NeuronFactory()
      n1 = f.new(2)
      n2 = f.new(3)
      with self.assertRaises(IndexError):
        n3 = f.combine([n1, n2])

  unittest.main()
