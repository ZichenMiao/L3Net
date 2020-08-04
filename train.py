import numpy as np 
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch import optim
import argparse

from gen_data import gen_data
from model import Model, Model_1layer

parser = argparse.ArgumentParser()
parser.add_argument('num_layer_model', default='2', help='2-layer model or 1-layer model')
parser.add_argument('graph_type', help='type of graph and data')
parser.add_argument('--gcn_type', default='ChebNet', choices=['ChebNet', 'GCN_Bases'], help='which gcn')
parser.add_argument('--order_list', type=int, default=[], nargs='+', help='like "patch_size"')
parser.add_argument('--L', type=int, default=3, help='"L" in ChebNet')
parser.add_argument('--use_shared_bases', action='store_true')
args = parser.parse_args()

TEST_TIMES = 5


graph_type = args.graph_type

## model args
gcn_type = args.gcn_type # 'ChebNet', 'GCN_Bases'
use_shared_bases = args.use_shared_bases
if gcn_type == 'GCN_Bases':
    order_list = args.order_list
    num_bases = len(order_list)
elif gcn_type == 'ChebNet':
    num_bases = args.L
    order_list = list(range(num_bases))

## define dataset
class Toy_Dataset(Dataset):
    def __init__(self, partition='train', graph_type='ring'):
        data_path = '{}_datas_{}.npy'.format(partition, graph_type)
        label_path = '{}_labels_{}.npy'.format(partition, graph_type)
        self.datas = np.load(data_path)
        self.labels = np.load(label_path)
        print("loading data from {}".format(data_path))
        print("loading label from {}".format(label_path))
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]

print(args)
res_list = []
for time in range(TEST_TIMES):
    if args.num_layer_model == '2':
        ### two-layer model
        model = Model(gcn_type=gcn_type, num_bases=num_bases, order_list=order_list,
                        graph_type=graph_type, use_shared_bases=use_shared_bases)
    elif args.num_layer_model == '1':
        ### one-layer model
        model = Model_1layer(gcn_type=gcn_type, num_bases=num_bases, order_list=order_list,
                        graph_type=graph_type, use_shared_bases=use_shared_bases)
    else:
        raise ValueError('Wrong num of layers')

    # model.cuda()
    print(model)
    ## show total params
    print('\nTotal params: {}k\n'.format(model.total_params/1e3))

    ## train 
    epoch = 100
    batch_size = 100
    lr = 1e-3
    decay_times = [60]

    train_loader = DataLoader(Toy_Dataset(partition='train', graph_type=graph_type), batch_size=batch_size, 
                                shuffle=True, drop_last=True)
    val_loader = DataLoader(Toy_Dataset(partition='val', graph_type=graph_type), batch_size=batch_size)
    # test_loader = DataLoader(Toy_Dataset(partition='test'), batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epo in range(epoch):
        
        ## adjust learning rate
        if epo+1 in decay_times:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        ## train an epoch
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct_step = pred.eq(target.view_as(pred)).sum().item()
            correct += correct_step
            loss = F.nll_loss(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        # report this epoch's results
        train_loss /= len(train_loader.dataset)
        print('Epoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) \r'.format(
                    epo, train_loss, correct, len(train_loader.dataset),
                    100. * correct / len(train_loader.dataset)))
        
        ## validate
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                # data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = 100. * correct / len(val_loader.dataset)
        print('Epoch: {}, Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) \r'.format(
                    epo, val_loss, correct, len(val_loader.dataset), val_acc))

        ## update best val acc
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            # if args.gcn_type == 'GCN_Bases':
            #     torch.save(model.state_dict(), 'checkpoints/GCN_Bases1_best.pth')
            # elif args.gcn_type == 'ChebNet':
            #     torch.save(model.state_dict(), 'checkpoints/ChebNet_K{}_best.pth'.format(args.num_bases))
            # else:
            #     raise ValueError('wrong gcn type')
            # print('save best')

        print('best Val Acc: {:.4f}'.format(best_val_acc))
    
    ## store this time's best result
    res_list.append(best_val_acc)

## report mean and acc
print("Model: {}, {}".format(gcn_type, order_list))
print("Mean Acc: {}, Std Acc: {}".format(np.mean(res_list), np.std(res_list)))