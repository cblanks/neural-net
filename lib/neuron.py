import math


class Neuron():
    count = 0

    def __init__(self, bias, input_weights=[0.1, 0.1]):
        self._setBias(bias)
        self._setWeights(input_weights)
        Neuron.count += 1

    def __del__(self):
        Neuron.count -= 1

    def _combineInputs(self, input_list):
        sum_xw = 0.0
        sum_w = 0.0
        for i in range(len(input_list)):
            sum_xw += input_list[i] * self.input_weights[i]
            sum_w += self.input_weights[i]

        return sum_xw / sum_w

    def _setBias(self, bias):
        if bias < -1.0 or bias > 1.0:
            raise ValueError(
                f'This neuron can have a bias no less than -1.0 and no more than 1.0 but {bias} was received.')

        self.bias = bias

    def _setWeights(self, weights):
        if len(weights) < 1:
            raise IndexError(
                f'This neuron can have must have at least 1 input but {len(weights)} were received.')

        for w in weights:
            if w < -1.0 or w > 1.0:
                raise ValueError(
                    f'This neuron can apply input weights of no less than 0.0 and no more than 1.0 but a weight of {w} was received.')

        self.input_weights = weights

    def numberOfInputs(self):
        return len(self.input_weights)

    def getOutput(self, input_list):
        if not len(input_list) == self.numberOfInputs():
            raise IndexError(
                f'This neuron expects {self.numberOfInputs()} inputs but received {len(input_list)}.')

        return math.tanh(self._combineInputs(input_list) + self.bias)


if __name__ == "__main__":

    import unittest

    class NeuronTestMethods(unittest.TestCase):

        def test_count_neurons_created(self):
            Neuron.count = 0
            n1 = Neuron(0.1)
            self.assertEqual(Neuron.count, 1)
            n2 = Neuron(0.2)
            self.assertEqual(Neuron.count, 2)
            del n1, n2
            self.assertEqual(Neuron.count, 0)

        def test_bias_is_set(self):
            n = Neuron(0.1)
            self.assertEqual(n.bias, 0.1)

        def test_bias_is_too_large(self):
            with self.assertRaises(ValueError):
                n = Neuron(1.1)

        def test_weights_are_set(self):
            n = Neuron(0.1, [0.1, 0.2, 0.3, 0.4, 0.5])
            self.assertEqual(len(n.input_weights), 5)

        def test_too_few_weight_throws(self):
            with self.assertRaises(IndexError):
                n = Neuron(0.1, [])

        def test_a_weight_is_too_large(self):
            with self.assertRaises(ValueError):
                n = Neuron(0.1, [0.1, 0.2, 0.3, 1.4, 0.5])

        def test_too_many_inputs_throws(self):
            n = Neuron(0.1, [0.1, 0.2])
            with self.assertRaises(IndexError):
                o = n.getOutput([0.3, 0.4, 0.5])

        def test_output_returned(self):
            n = Neuron(0.3, [0.1, 0.1])
            o = n.getOutput([0.3, 0.5])
            self.assertEqual(o, 0.6043677771171635)

    unittest.main()
