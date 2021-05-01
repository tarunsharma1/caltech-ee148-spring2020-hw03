from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import torchvision
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    Linear, Conv2d, MaxPool2d, AvgPool2d, ReLU, Softmax, BatchNorm2d, Dropout, Flatten, Sequential.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        #self.conv3 = nn.Conv2d(16, 16, 3, 1)
        self.BatchNorm1 = nn.BatchNorm2d(8)
        self.BatchNorm2 = nn.BatchNorm2d(16)
        self.BatchNorm3 = nn.BatchNorm1d(64)

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(400, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout2(x)

        #x = self.conv3(x)
        #x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        #x = self.dropout3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.BatchNorm3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    train_loss = 0
    test_num = 0
    correct = 0

    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))
        # sum up loss for every batch
        train_loss += loss.item()
        test_num += len(data)
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
            
    # return the loss
    train_loss /= test_num
    print ('Training accuracy: {}'.format(100. * correct/test_num))
    return train_loss


activations = np.zeros((10000, 64))
count = 0
def store_activations(self, input, output):
    global count
    global activations
    # question says "a feature vector (taken from just before the final linear layer), hence taking input of the last linear layer"
    activations[count*1000:(count*1000) + 1000,:] = input[0].data.cpu().numpy()
    count += 1



def tsne_viz(total_targets):
    global activations
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(activations)
    # plot each class at a time so that we get different colors
    for i in range(10):
        indices = np.where(total_targets==i)[0]
        #import ipdb;ipdb.set_trace()
        plt.scatter(tsne_results[indices,0],tsne_results[indices,1], label=str(i))
    
    plt.legend()
    plt.show()



def test(model, device, test_loader):
    global activations
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    
    ###### visualize filters (weights) in first layer - UNCOMMENT TO RUN ######
    # filters = model.conv1.weight.data.cpu().numpy()
    # fig = plt.figure(figsize = (2,4))
    # grid = ImageGrid(fig, 111, nrows_ncols=(2,4), axes_pad=0.1, cbar_location="right",
    #                   cbar_mode="single",
    #                   cbar_size="10%",
    #                   cbar_pad=0.05)
    # norm = matplotlib.colors.Normalize(vmax=np.max(filters), vmin=np.min(filters))
    # for ax, idx in zip(grid, range(0,8)):
    #     im = ax.imshow(filters[idx,0,:,:], norm=norm)
    
    # ax.cax.colorbar(im)
    # ax.cax.toggle_label(True)
    # plt.show()

    ##########################################################

    
    model.fc2.register_forward_hook(store_activations)

    euclidean_dist = np.zeros((4, 10000))

    #size of test set
    total_preds = []
    total_targets = []

    with torch.no_grad():   # For the inference step, gradient is not computed
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            total_preds.extend(pred.cpu().numpy().reshape(len(data)))
            total_targets.extend(target.cpu().numpy().reshape(len(data)))

            ######## euclidean distance - UNCOMMENT TO RUN ###########
            # # calculate distance between one image and every other image
            # for k in range(4):
            #     for j in range(0, len(data)):
            #         # compute euclidean distance between feature vector of that image (img0 to img4) and all other images
            #         euclidean_dist[k,batch_id*len(data) + j] = np.linalg.norm(activations[k] - activations[batch_id*len(data) + j])


            ##########################################################





            ########### visualize incorrect predictions - UNCOMMENT TO RUN ##############
            # incorrect_preds_indices_tensor = torch.where(~pred.eq(target.view_as(pred)))[0]
            # incorrect_preds_images = data[incorrect_preds_indices_tensor]

            # print ('correct label:', target[incorrect_preds_indices_tensor])
            # print ('predicted labels:', pred[incorrect_preds_indices_tensor])

            # img_grid = torchvision.utils.make_grid(data[torch.where(~pred.eq(target.view_as(pred)))[0]].cpu(),nrow=3)
            # plt.imshow(img_grid.permute(1,2,0))
            # plt.show()
            # ###############################################################################


            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))



    ################# euclidean distance - grid - UNCOMMENT TO RUN  ##################
    # # make a grid with those images ..need to somehow access the image data from here - have to go through data loader each time i guess
    # images_euclid = np.zeros((4, 9, 28, 28))


    # for ind in range(4): # this loop is for the 4 images that we want to find the closest euclidean distance from
    #     # reset iterator to start - this whole part of the code will end up going through the entire test set 4 times in total..not the best but better than going through it 4x8 times
    #     test_set_iterator = iter(test_loader)
    #     top_idxs = euclidean_dist[ind,:].argsort()[0:9] # smallest euclidean distances from that image.. 9 because first one will be the image itself after sorting (i confirmed this)
    #     # sort those so that its easier for accessing the data from dataloader
    #     top_idxs.sort()
    #     print ('indices are: ', top_idxs)
    #     print ('euclidean distances are: ', euclidean_dist[ind, top_idxs])
        
    #     index_count = 0
    #     batch = 0

    #     while (True):
    #         data, target = next(test_set_iterator)

    #         # check if the index to be found is in the current batch (i am going to loop through batches with next(iter))
            
    #         while(True):
    #             if index_count == len(top_idxs):
    #                 break

    #             # current item to search for
    #             key = top_idxs[index_count]

    #             if key >= batch*len(data) and key < (batch+1)*len(data):
    #                 images_euclid[ind, index_count, :,:] = data.numpy()[key - (batch*len(data)),0,:,:]
    #                 # go to the next top_idx and look in the same batch before proceeding to the next batch outside the inner while loop
    #                 index_count += 1
    #             else:
    #                 # proceed to the next batch of data
    #                 break

            
    #         batch += 1

    #         # reached the end, now break out and move to the next image (out of the 4 images)
    #         if index_count == len(top_idxs):
    #             break

    # # make a grid of the 4 x 9 images
    # images_euclid = images_euclid.reshape(4*len(top_idxs),28,28)

    # fig = plt.figure(figsize = (4,9))
    # grid = ImageGrid(fig, 111, nrows_ncols=(4,9), axes_pad=0.1)
    
    # for ax, o in zip(grid, range(4*len(top_idxs))):    
    #     ax.imshow(images_euclid[o,:,:])
    # plt.show()
    



    #######################################################################
    
    
    ######### tSNE visualization - UNCOMMENT TO RUN ##########
    # total_targets = np.array(total_targets)
    # tsne_viz(total_targets)

    #########################################################



    ######## generate confusion matrix -UNCOMMENT TO RUN ################
    # cm = confusion_matrix(total_targets, total_preds)
    # plt.matshow(cm)
    # plt.yticks(range(10))
    # plt.xticks(range(10))
    # plt.show()

    ###################################################

    return test_loss


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        test(model, device, test_loader)

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    #transforms.GaussianBlur(3),
                    #transforms.RandomResizedCrop(28,scale=(0.7,1.0)),
                    #transforms.RandomHorizontalFlip(),  
                    #transforms.RandomRotation(10), 
                    #transforms.RandomAffine(degrees=5, translate= (0.1,0.1), scale=(0.9,1.1), shear=5),
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    
    # these should be indices such that it is 15% of each class
    subset_indices_valid = []

    labels = train_dataset.targets.numpy()
    
    for label in np.unique(labels):
        val_class_indices = np.where(labels==label)[0]
        # different seed per class
        random.seed(label)
        random.shuffle(val_class_indices)
        # take first 15% of this
        subset_indices_valid += list(val_class_indices[0:int(0.15*val_class_indices.shape[0])])

    
    # this should everything else other than what is in subset_indices_val
    subset_indices_train = [x for x in range(len(train_dataset)) if x not in subset_indices_valid]
    random.shuffle(subset_indices_train)

    ##### train on half, quarter, eighth and sixteenth using next line ##################
    #subset_indices_train = subset_indices_train[0:int(len(subset_indices_train)/2)]
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    train_loss_list = []
    val_loss_list = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        train_loss_list.append(train_loss)
        val_loss = test(model, device, val_loader)
        val_loss_list.append(val_loss)
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    plt.plot(range(1, args.epochs + 1), np.array(train_loss_list), label='train loss')
    plt.plot(range(1, args.epochs + 1), np.array(val_loss_list), label='val loss')
    plt.show()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()
