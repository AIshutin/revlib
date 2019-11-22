import torch
ConvTranspose2d_DEFAULT_KERNEL_DIM_SIZE = 5
ConvTranspose2d_DEFAULT_MAX_STRIDE = 4
ConvTranspose2d_DEFAULT_MAX_DILATION = 4
ConvTranspose2d_DEFAULT_MAX_INP_PADDING = 4
ConvTranspose2d_DEFAULT_MAX_OUT_PADDING = 4

class ReversedLayerClassNotDeclared(Exception):
    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def __str__(self):
        return f"There is not any layer being reversed one for {self.name}"
    __repr__ = __str__

class ReversedLayerParamsNotFound(Exception):
    def __init__(self, name="", input_shape="", output_shape=""):
        super().__init__()
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __str__(self):
        return f"Parameters for reverting {self.name} with input_shape {self.output_shape}" \
        f" and output_shape {self.input_shape} not found"
    __repr__ = __str__

def calc_parameters(lay):
    total = 0
    for tensor in lay.parameters():
        curr = 1
        for el in tensor.shape:
            curr *= el
        total += curr
    return total

def choose_parameters_in_ConvTranspose2d_space(input_shape, output_shape, layer, symmetric=True):
    batch_size, cin, hin, win = input_shape
    batch_size, cout, hout, wout = output_shape

    input_ex = torch.randn((1, cin, hin, win))
    inp_pads = range(0, ConvTranspose2d_DEFAULT_MAX_INP_PADDING + 1)
    out_pads = range(0, ConvTranspose2d_DEFAULT_MAX_OUT_PADDING + 1)

    configurations = []

    stride1, stride2 = layer.stride
    dil1, dil2 = layer.dilation
    kdim1, kdim2 = layer.kernel_size

    for ipad1 in inp_pads:
        for ipad2 in inp_pads:
            if ipad1 != ipad2 and symmetric:
                continue
            for opad1 in out_pads:
                for opad2 in out_pads:
                    if opad1 != opad2 and symmetric:
                        continue
                    params =  { 'in_channels': cin,
                                'out_channels': cout,
                                'kernel_size': (kdim1, kdim2),
                                'stride': (stride1, stride2),
                                'dilation': (dil1, dil2),
                                'padding': (ipad1, ipad2),
                                'output_padding': (opad1, opad2)}

                    try:
                        lay = torch.nn.ConvTranspose2d(**params)
                        if lay(input_ex).shape != (1, cout, hout, wout):
                            continue
                    except Exception as exp:
                        continue
                    configurations.append((calc_parameters(lay), ipad1 + ipad2 + opad1 + opad2, params))

    configurations.sort(key=lambda x: x[:-1])
    return [el[-1] for el in configurations]

def choose_parameters_in_Linear_space(input_shape, output_shape, lay=None):
    return [{'in_features': input_shape[-1], 'out_features': output_shape[-1]}]

reverse_layer = {torch.nn.Linear: torch.nn.Linear,
                 torch.nn.Conv2d: torch.nn.ConvTranspose2d}
#                 type(torch.nn.modules.conv.Conv2d): torch.nn.ConvTranspose2d}

def get_reversed(input_shape, output_shape, lay, verbose=False):
    if type(lay) not in reverse_layer:
        raise ReversedLayerClassNotDeclared(type(lay))
    rclass = reverse_layer[type(lay)]
    print(lay)
    params = globals()[f'choose_parameters_in_{rclass.__name__}_space'](output_shape, input_shape, lay)
    if len(params) == 0:
        raise ReversedLayerParamsNotFound(type(rclass), input_shape, output_shape)
    if verbose:
        print(rclass, params)
    return rclass(**params[0])
