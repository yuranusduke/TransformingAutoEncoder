"""
Main file

Created by Kunhong Yu
Date: 2021/07/01
"""
import numpy as np
import torch as t
import torchvision as tv
from utils import Config, BatchShift, show_batch, show_weights
from capsules import transAE
import tqdm
import matplotlib.pyplot as plt
import sys
import fire


# Step 0 Decide the structure of the model#
opt = Config()
device = t.device(opt.device)

# Step 1 Load the data set#
def get_data_loader(dataset : str, train : bool, batch_size : int) -> t.utils.data.DataLoader:
    """Get data loader
    Args :
        --dataset: string format
        --train: True for train, False for test
        --batch_size: training or testing batch size
    return :
        --data_loader
    """
    if train:
        if dataset == 'mnist':
            train_data = tv.datasets.MNIST(root = './data/',
                                           train = True,
                                           download = True,
                                           transform = tv.transforms.ToTensor())
        elif dataset == 'fashion_mnist':
            train_data = tv.datasets.FashionMNIST(root = './data/',
                                                  train = True,
                                                  download = True,
                                                  transform = tv.transforms.ToTensor())
        else:
            raise Exception('No other data sets!')
        data_loader = t.utils.data.DataLoader(train_data,
                                              batch_size = batch_size,
                                              shuffle = True,
                                              drop_last = False)
    else:
        if dataset == 'mnist':
            test_data = tv.datasets.MNIST(root = './data/',
                                          train = False,
                                          download = True,
                                          transform = tv.transforms.ToTensor())
        elif dataset == 'fashion_mnist':
            test_data = tv.datasets.FashionMNIST(root = './data/',
                                                 train = False,
                                                 download = True,
                                                 transform = tv.transforms.ToTensor())
        else:
            raise Exception('No other data sets!')
        data_loader = t.utils.data.DataLoader(test_data,
                                              batch_size = batch_size,
                                              shuffle = False,
                                              drop_last = False)

    return data_loader


# Step 2 Reshape the inputs#
# Step 3 Normalize the inputs#
# Step 4 Initialize parameters#
# Step 5 Forward pass#
def get_model(cap_dim : int, out_dim : int, num_caps : int):
    """Get model instance
    Args :
        --cap_dim: capsule hidden dimension
        --out_dim: output dimension
        --num_caps: number of capsules
    return :
        --model
    """
    model = transAE(input_dim = 784, cap_dim = cap_dim, out_dim = out_dim, num_caps = num_caps)
    model.to(device)
    print('Model : \n', model)

    return model


def train(dataset : str, cap_dim : int, out_dim : int, num_caps : int, epochs : int, batch_size : int):
    """train the model
    Args :
        --dataset: string format
        --cap_dim: capsule hidden dimension
        --out_dim: output dimension
        --num_caps: number of capsules
        --epochs: training epochs
        --batch_size
    """
    data_loader = get_data_loader(dataset, True, batch_size = batch_size)
    model = get_model(cap_dim = cap_dim, out_dim = out_dim, num_caps = num_caps)
    # Step 6 Compute cost#
    cost = t.nn.BCELoss().to(device)

    # Step 7 Backward pass#
    optimizer = t.optim.Adam(filter(lambda x : x.requires_grad, model.parameters()),
                             amsgrad = True, weight_decay = 0)

    # Step 8 Update parameters#
    losses = []
    for epoch in tqdm.tqdm(range(epochs)):
        print('Epoch : %d / %d.' % (epoch + 1, epochs))
        for i, (batch_x, _) in enumerate(data_loader):
            batch_x = batch_x.view(-1, 784)
            batch_x = batch_x.to(device)
            batch_y, R = BatchShift(batch_x.view(-1, 28, 28).cpu().numpy().copy(), [-4, 4])
            R = t.from_numpy(R).to(device)
            batch_y = t.from_numpy(batch_y).to(device)

            optimizer.zero_grad()
            out = model(batch_x, R)
            batch_cost = cost(out, batch_y.view(-1, 784))

            batch_cost.backward()
            optimizer.step()

            if i % batch_size == 0:
                print('\tBatch %d / %d has cost : %.2f.' % (i + 1, len(data_loader), batch_cost.item()))
                losses.append(batch_cost.item())

    print('Training is done!')

    plt.plot(range(len(losses)), losses, '-^g', label = 'loss')
    plt.xlabel('Steps')
    plt.ylabel('Values')
    plt.title('Training loss')
    plt.grid(True)
    plt.legend(loc = 'best')
    plt.savefig(f'./results/training_loss_{dataset}.png')

    t.save(model, f'./results/model_{dataset}.pth')


def test(dataset : str, batch_size : int):
    """test and saving results
    Args :
        --dataset
        --batch_size
    """
    model = t.load(f'./results/model_{dataset}.pth')
    model.to(device)

    data_loader = get_data_loader(dataset, train = False, batch_size = batch_size)
    model.eval()

    # According to paper, we visualize first 20 generation units of first 7 capsules
    capsules = model.transae_layer
    for i, capsule in enumerate(capsules[:7]):
        weight = capsule.output_layer[0].weight[:, :20].transpose(1, 0).view(20, 28, 28).data.cpu().numpy()
        for j in range(20):
            if j == 0:
                cap_weight = weight[j, ...]
            else:
                cap_weight = np.concatenate([cap_weight, weight[j, ...]], axis = -1)

        if i == 0:
            weights = cap_weight
        else:
            weights = np.concatenate([weights, cap_weight], axis = 0)

    show_weights(weights, dataset)

    with t.no_grad():
        for i, (batch_x, _) in enumerate(data_loader):
            sys.stdout.write('\r>>Testing batch %d / %d.' % (i + 1, len(data_loader)))
            sys.stdout.flush()

            batch_x = batch_x.view(-1, 784)
            batch_x = batch_x.to(device)
            batch_y, R = BatchShift(batch_x.view(-1, 28, 28).cpu().numpy().copy(), [-4, 4])
            R = t.from_numpy(R).to(device)
            batch_y = t.from_numpy(batch_y).to(device)

            out = model(batch_x, R)
            out = out.view(-1, 28, 28)

            input = batch_x[:10, ...].view(-1, 28, 28).cpu() # input
            out = out[:10, ...].cpu() # generated
            batch_y = batch_y[:10, ...].cpu() # target

            tensor = t.cat((input, out, batch_y), dim = -1)
            show_batch(tensor, i + 1, dataset)

    print('\nTesting is done!')


def main(**kwargs):
    """Main function"""
    opt = Config()
    opt.parse(**kwargs)

    if 'only_test' in kwargs and kwargs['only_test']:
        test(opt.dataset, opt.batch_size)

    else:
        train(cap_dim = opt.cap_dim,
              out_dim = opt.out_dim,
              num_caps = opt.num_caps,
              epochs = opt.epochs,
              batch_size = opt.batch_size,
              dataset = opt.dataset)

        test(opt.dataset, opt.batch_size)



if __name__ == '__main__':
    fire.Fire()

    """
    Usage:
    python main.py main --cap_dim=120 --out_dim=300 --num_caps=7 --epochs=20 --batch_size=32 --only_test=False --dataset='mnist'
    """

    print('\nDone!\n')