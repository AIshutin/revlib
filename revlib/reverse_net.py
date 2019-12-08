import torch
from torch import nn

import reverse_layers as rl
import utils
from utils import extract_layers, check_if_activation

def group_by_blocks(encoder_lays, input_shapes=None, output_shapes=None):
    input_shapes = input_shapes if input_shapes is not None else [None] * len(encoder_lays)
    output_shapes = output_shapes if output_shapes is not None else [None] * len(encoder_lays)

    blocks = []
    i = 0
    while i < len(encoder_lays):
        mx_len = 1
        while i + mx_len < len(encoder_lays) and utils.calc_parameters(encoder_lays[i + mx_len]) == 0:
            mx_len += 1
        blocks.append(utils.LayBlock(encoder_lays[i:i + mx_len], ind=i, \
                                    ishapes=input_shapes[i:i + mx_len],
                                    oshapes=output_shapes[i:i + mx_len]))
        i += mx_len
    return blocks

def join_blocks(blocks):
    decoder_lays = []
    for el in blocks:
        decoder_lays += extract_layers(el.lays)
    return nn.Sequential(*decoder_lays)

def reverse_net(encoder, input=None):
    """
    Input example, not in batch
    """
    lays = extract_layers(encoder)
    if input is not None:
        input = input.squeeze(0)
        input_shapes = []
        output_shapes = []
        for lay in lays:
            input_shapes.append(input.shape)
            input = lay(input)
            output_shapes.append(input.shape)
        blocks = group_by_blocks(lays, input_shapes, output_shapes)
    else:
        blocks = group_by_blocks(lays)
    return [el.revert() for el in blocks]