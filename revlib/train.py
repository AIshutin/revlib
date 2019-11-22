import torch
import copy

LAY_BATCHSIZE = 32
ALLOW_GPU = True
EPOCHS = 1#20
RANDOM_SEED = 1791791791
EVAL_PART = 0.33
LAY_LR = 1e-4
LAY_MOMENTUM = 0.9
torch.manual_seed(RANDOM_SEED)

criterion = torch.nn.MSELoss()

def train_lay(reverted_lay, io_data, device=None, verbose=False):
    """
    Trains decoder layer which restores input_data of normal layer from output_data of normal layer

    @ io_data[i][0] is the input of normal layer, thus it should be the ouput of reverted layer. \
    And vice versa for io_data[i][1].
    """

    if device is None:
        if ALLOW_GPU and torch.cuda.is_available():
            device = torch.device('cuda:0')
            reverted_lay.to(device)
        else:
            device = torch.device('cpu')

    test_size = int(len(io_data) * EVAL_PART)
    train_size = len(io_data) - test_size
    traindata, testdata = torch.utils.data.random_split(io_data, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=LAY_BATCHSIZE)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=LAY_BATCHSIZE)

    best = None
    best_loss = None

    optimizer = torch.optim.SGD(reverted_lay.parameters(), lr=LAY_LR, momentum=LAY_MOMENTUM)

    for epoch in range(EPOCHS):
        iter = trainloader
        if verbose:
            import tqdm
            iter = tqdm.tqdm(iter, desc=f"Epoch {epoch} from {EPOCHS}: ")
        for (Y, X) in iter:
            optimizer.zero_grad()
            X = X.to(device)
            Y = Y.to(device)
            predicted = reverted_lay(X)
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
    return (best, best_loss)
