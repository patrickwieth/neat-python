import math


def sigmoid_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    z = max(-60.0, min(60.0, s))
    return 1.0 / (1.0 + math.exp(-z))


def tanh_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    z = max(-60.0, min(60.0, s))
    return math.tanh(z)


def sin_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    z = max(-60.0, min(60.0, s))
    return math.sin(z)


def gauss_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    z = max(-60.0, min(60.0, s))
    return math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)


def relu_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    return s if s > 0.0 else 0


def identity_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    return s


def clamped_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    return max(-1.0, min(1.0, s))


def inv_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    if s == 0:
        return 0.0

    return 1.0 / s


def log_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    z = max(1e-7, s)
    return math.log(z)


def exp_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w
    
    z = max(-60.0, min(60.0, s))
    return math.exp(z)


def abs_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    return abs(s)


def hat_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    return max(0.0, 1 - abs(s))


def square_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    return s ** 2


def cube_activation(bias, links, ivalues):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    return s ** 3


def maxout_activation(bias, links, ivalues):
    inputs = []
    for i, w in links:
        inputs.append(ivalues[i] * w) 

    return max(inputs)    


def elu(x):
    s = 0.0
    for i, w in links:
        s += ivalues[i] * w

    return s if s > 0 else math.exp(x) - 1


class InvalidActivationFunction(Exception):
    pass


class ActivationFunctionSet(object):
    def __init__(self):
        self.functions = {}

    def add(self, config_name, function):
        # TODO: Verify that the given function has the correct signature.
        self.functions[config_name] = function

    def get(self, config_name):
        f = self.functions.get(config_name)
        if f is None:
            raise InvalidActivationFunction("No such function: {0!r}".format(config_name))

        return f

    def is_valid(self, config_name):
        return config_name in self.functions


