import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import reverse_layers
import train
import utils
import sys
import os

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD  = [0.229, 0.224, 0.225]

DEFAULT_ANTISTD = [1 / el for el in DEFAULT_STD]
DEFAULT_ANTIMEAN = [-el for el in DEFAULT_MEAN]

DEFAULT_ONES = [1] * 3
DEFAULT_ZEROS = [0] * 3

img2tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)])

tensor2img = transforms.Compose([
        transforms.Normalize([0] * 3, DEFAULT_ANTISTD),
        transforms.Normalize(DEFAULT_ANTIMEAN, [1] * 3),
        transforms.ToPILImage()])

def test_lay(lay, input, output, verbose=False, examples_num=5):
    """
    @ input is an input of the lay
    @ output is an output of the lay
    @ if examples_num != 0 input must be images
    """

    ishape = input[0].unsqueeze(0).shape
    oshape = output[0].unsqueeze(0).shape

    reverted_lay = reverse_layers.get_reversed(ishape, oshape, lay, verbose=True)
    reverted_lay, loss = train.train_lay(reverted_lay, utils.IOdataset(input, output), verbose=True)

    if examples_num == 0:
        return (reverted_lay, loss, [], [])

    edataset = utils.IOdataset([output[i] for i in range(examples_num)],
                                [input[i] for i in range(examples_num)],)


    loader = torch.utils.data.DataLoader(edataset, batch_size=1)
    input_images = []
    output_images = []
    for (x, y) in loader:
        input_images.append(tensor2img(y[0]))
        img_rev = reverted_lay(x)[0]
        output_images.append(tensor2img(img_rev))

    return (reverted_lay, loss, input_images, output_images)

class DebugNet(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        print(x.shape)
        return self.net(x)


if __name__ == "__main__":
    vgg = models.vgg16(pretrained=True)
    lay = vgg.features[0]
    lay.eval()
    imagefolder = '../../flower_photos'

    '''inputs = utils.NoLabelsDataset(torchvision.datasets.ImageFolder(imagefolder, transform=transforms.Resize((224, 224))))
    inputs[0].save('0.png')
    tensor2img(img2tensor(inputs[0])).save('0ch.png')'''

    inputs = utils.NoLabelsDataset(torchvision.datasets.ImageFolder(imagefolder, transform=img2tensor))
    lay = DebugNet(lay)
    with torch.no_grad():
        outputs = utils.NoLabelsDataset(torchvision.datasets.ImageFolder(imagefolder,
                                        transform=transforms.Compose([img2tensor, lambda x: lay(x.unsqueeze(0)).squeeze(0)])))
    lay = lay.net
    _, loss, original, reverted = test_lay(lay, inputs, outputs, verbose=True)

    print(f"Loss: {loss:9.4f}")
    try:
        os.mkdir('examples')
    except FileExistsError as exp:
        pass

    for i in range(len(original)):
        original[i].save(f'examples/original{i}.png')
        reverted[i].save(f'examples/reverted{i}.png')
