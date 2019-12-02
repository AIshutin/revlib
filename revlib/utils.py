import torch
from torch import nn

class IOdataset(torch.utils.data.Dataset):
    def __init__(self, input, output):
        super().__init__()
        assert(type(input) == type(output))
        if isinstance(input, torch.Tensor):
            self.len = input.shape[0]
            assert(self.len == output.shape[0])
        else:
            self.len = len(input)
            assert(self.len == len(output))

        self.input = input
        self.output = output

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return (self.input[ind], self.output[ind])

    def apply_transform(self, transform, where="output"):
        if where == "output":
            for i in range(len(self.output)):
                output[i] = transform(output[i])
        else:
            for i in range(len(self.input)):
                input[i] = transform(input[i])

class NoLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind][0]

def calc_parameters(lay):
    total = 0
    try:
        for tensor in lay.parameters():
            curr = 1
            for el in tensor.shape:
                curr *= el
            total += curr
        return total
    except AttributeError as exp:
        return 0

def extract_layers(net):
    lays = []
    for lay in net.modules():
        if not isinstance(lay, nn.Sequential) and not isinstance(lay, LayBlock):
            lays.append(lay)
    return lays

ACTIVATIONS = [nn.ReLU, nn.ReLU6, nn.ELU, nn.SELU, nn.PReLU, nn.LeakyReLU,
            nn.Threshold, nn.Sigmoid, nn.Tanh, nn.LogSigmoid,
            nn.Softplus,  nn.Softsign, nn.Softmin,
            nn.Softmax, nn.Softmax2d, nn.LogSoftmax]


def check_if_activation(lay):
    for act in ACTIVATIONS:
        if isinstance(lay, act):
            #print(lay, True)
            return True
    #print(lay, False)
    return False


class LayBlock(torch.nn.Module):
    def __init__(self, lays, ind, ishapes, oshapes):
        super().__init__()
        self.lays = nn.Sequential(*lays)
        self.ind = ind
        self.reverted = False

        self.ishapes = ishapes
        self.oshapes = oshapes

    def revert(self):
        assert(self.reverted is False)
        self.reverted = True

        actf = nn.ReLU
        layers = extract_layers(self.lays)
        self.ind += len(layers) - 1
        for lay in layers:
            if check_if_activation(lay):
                actf = type(lay)

        import reverse_layers as rl
        lays = []
        lays = [rl.get_reversed(lay=layers[i], input_shape=self.ishapes[i],   \
                    output_shape=self.oshapes[i]) for i in range(len(layers)) \
                    if not check_if_activation(layers[i])][::-1]

        lays.append(actf())
        self.lays = nn.Sequential(*lays)
        return self

    def forward(self, X):
        return self.lays(X)

    def __str__(self):
        return str(self.lays)

    def __repr__(self):
        return self.lays.__repr__()

def apply_net(net, data, device=torch.device('cpu'), batch_size=1):
    assert(batch_size == 1)
    default = torch.device('cpu')
    return [net(el.to(device).unsqueeze(0)).squeeze(0).to(default) for el in data]
