import torch
import copy
import utils
from torch import nn
import torchvision.models as models
import tqdm

LAY_BATCHSIZE = 32
ALLOW_GPU = True
EPOCHS = 1#20
RANDOM_SEED = 1791791791
EVAL_PART = 0.33
LAY_LR = 1e-4
NET_LR = 1e-4
LAY_MOMENTUM = 0.9
NET_MOMENTUM = 0.9
LR_DECAY = 0.1
LR_THR = 1e-6
torch.manual_seed(RANDOM_SEED)

criterion = torch.nn.MSELoss()

DEVICE = None

class VGG16_MLoss(nn.Module):
	def __init__(self, ind=[1, 3, 6], device=None):
		super().__init__()
		if device is None:
			device = get_device()
		self.vgg = utils.extract_layers(models.vgg16(pretrained=True).eval().to(device).features)[:max(ind)]
		self.ind = ind

	def forward(self, X, target):
		loss = 0
		for i, lay in enumerate(self.vgg):
			X = lay(X)
			target = lay(target)
			if i in self.ind:
				loss += criterion(X, target)
		return loss

def get_device():
    global DEVICE
    if DEVICE is None:
        if ALLOW_GPU and torch.cuda.is_available():
            DEVICE = torch.device('cuda:0')
        else:
            DEVICE = torch.device('cpu')
    return DEVICE

def train_lay(reverted_lay, io_data, verbose=False, optimizer=None, device=None, criterion=None):
    """
    Trains decoder layer which restores input_data of normal layer from output_data of normal layer

    @ io_data[i][0] is the input of normal layer, thus it should be the ouput of reverted layer. \
    And vice versa for io_data[i][1].
    """

    if utils.calc_parameters(reverted_lay) == 0:
        return (reverted_lay, 0)

    if optimizer is None:
        device = get_device()
        reverted_lay.to(device)
        optimizer = torch.optim.SGD(reverted_lay.parameters(), lr=LAY_LR, momentum=LAY_MOMENTUM)
    else:
    	assert(device is not None)
    test_size = int(len(io_data) * EVAL_PART)
    train_size = len(io_data) - test_size
    traindata, testdata = torch.utils.data.random_split(io_data, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=LAY_BATCHSIZE)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=LAY_BATCHSIZE)

    best = None
    best_loss = None
    
    print('DECODER')
    print(reverted_lay)

    if criterion is None:
    	del criterion

    for epoch in range(EPOCHS):
        iter = trainloader
        if verbose:
            import tqdm
            iter = tqdm.tqdm(iter, desc=f"Epoch {epoch} from {EPOCHS}: ")
        for (Y, X) in iter:
            optimizer.zero_grad()
            X = X.to(device)
            Y = Y.to(device)
            print(Y.shape, X.shape)
            predicted = reverted_lay(X)
            print('predicted', predicted.shape)
            loss = criterion(predicted, Y)
            loss.backward()
            optimizer.step()

        total_loss = 0
        with torch.no_grad():
            for (Y, X) in testloader:
                total_loss += criterion(reverted_lay(X.to(device)), Y.to(device)).item()

        if verbose:
            print(f"Total val loss: {total_loss:9.4f}")
        if best_loss is None or best_loss > total_loss:
            best = copy.deepcopy(reverted_lay).to('cpu')
            best_loss = total_loss

    if optimizer is None:
        return (best, best_loss)
    else:
        return (best, best_loss, optimizer)

def split_encoder(encoder, blocks):
    #print(encoder)
    #print(blocks)
    enc_blocks = []
    lays = utils.extract_layers(encoder)
    last = 0
    for block in blocks:
        enc_blocks.append(nn.Sequential(*lays[last:block.ind + 1]))
        last = block.ind + 1
    return enc_blocks

def train_net(input, encoder, blocks, verbose=False):
    encoder = split_encoder(encoder, blocks)
    device = get_device()
    dataset = utils.IOdataset(input, copy.deepcopy(input))
    optimizer = None
    decoder = nn.Sequential()
    #torch.optim.SGD(reverted_lay.parameters(), lr=LAY_LR, momentum=LAY_MOMENTUM)

    iterator = range(len(blocks))
    if verbose:
        iterator = tqdm.tqdm(iterator, desc="Blocks training")

    criterion = VGG16_MLoss()
    enc = None
    for i in iterator:
        if enc is None:
            enc = nn.Sequential(encoder[0])
            enc.to(device)
            enc.eval()
        else:
            enc = nn.Sequential(*(utils.extract_layers(enc) + [encoder[i].to(device)]))
            enc.eval()

        print('ENCODER:')
        print(enc)
        dataset.output = utils.apply_net(enc, dataset.input, device, batch_size=LAY_BATCHSIZE)

        net = blocks[i]
        net.to(device)
        print('DEC_HEAD', net)
        print(utils.extract_layers(decoder))
        decoder = nn.Sequential(*([net] + utils.extract_layers(decoder)))

        if i == 0:
            optimizer = torch.optim.SGD(decoder.parameters(), lr=NET_LR, momentum=NET_MOMENTUM)
        else:
            groups = []
            for param_group in optimizer.param_groups:
                param_group['lr'] *= LR_DECAY
                if param_group['lr'] >= LR_THR:
                    groups.append(param_group)
            optimizer.param_groups = groups

            optimizer.add_param_group({'params': net.parameters(),
                                            'lr': NET_LR,
                                            'momentum': NET_MOMENTUM})

        decoder, loss, optimizer = train_lay(decoder, dataset, verbose, optimizer, device, 
        									criterion=criterion)

    decoder = nn.Sequential(*utils.extract_layers(decoder))
    return decoder
