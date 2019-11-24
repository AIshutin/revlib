import torch

class IOdataset(torch.utils.data.Dataset):
    def __init__(self, input, output):
        super().__init__()
        assert(type(input) == type(output))
        if isinstance(input, type(torch.tensor)):
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
