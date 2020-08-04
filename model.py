import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from graph import gen_graph

# function used to adjust adjacency and produce bases_template
def normalize_undigraph(A):
    Dl = torch.sum(A, 0)
    num_node = A.shape[0]
    Dn = torch.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = torch.matmul(torch.matmul(Dn, A), Dn)
    return DAD

def A2normed_scaled_L(A):
    """
    From original A(not A+I) to normalized, rescaled Graph Laplacian
    """
    ## L = I - D^-0.5 * A * D^-0.5
    normed_L = torch.eye(A.shape[0]) - normalize_undigraph(A)
    ## L_ = L * (2/eigval_max) - I
    # eigenvalue: fetch the real part
    lambda_max = 1.02 * torch.max(torch.eig(normed_L, eigenvectors=False)[0][:, 0])
    normed_rescaled_L = normed_L * (2/lambda_max) - torch.eye(A.shape[0])

    return normed_rescaled_L

def k_th_order_A(order, A):
    """
    modify A to incorporate the right order of neighbors
    :param: A
    :return: new A
    """
    if order == 0:
        return torch.eye(A.shape[1]).float()
    A_total = torch.zeros_like(A)
    for i in range(1, order + 1):
        A_total += A.matrix_power(i)

    return (A_total != 0).float()


class GraphConv_Bases_Shared(nn.Module):
    def __init__(self, in_channels, out_channels, level, bias=True, 
                    order_list=[1], num_bases=1, gcn_type='GCN_Bases', graph_type='ring'):
        super(GraphConv_Bases_Shared, self).__init__()
        # import pdb; pdb.set_trace()
        # bases hyper-parameter
        assert gcn_type == 'GCN_Bases'
        assert order_list == [1]
        assert num_bases == 1
        # assert graph_type == 'chain'

        self.gcn_type = gcn_type
        self.num_bases = num_bases
        self.order_list = order_list
        self.graph_type = graph_type

        ## get graph
        print('using {} graph'.format(self.graph_type))
        A_ = gen_graph(level=level, graph_type=self.graph_type).float()
        self.num_nodes = A_.shape[0]

        ## bases with shape [3], used for chain graph
        self.bases = nn.Parameter(torch.Tensor(3))
        in_size = 3 * self.num_nodes * 1.0
        std_ = math.sqrt(1. / in_size)
        nn.init.normal_(self.bases, std=std_)
        
        ## define coeff operation, 
        ## for three types of GCN, this is the same
        self.coeff_conv = nn.Conv1d(
            in_channels=self.num_bases*in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

        ## bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            n = in_channels * self.num_bases
            stdv = 1. / math.sqrt(n)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def build_local_filter(self,):
        ## from [3] to tri-diagonal matrix
        bases_matrix = []
        for i in range(self.num_nodes):
            if i == 0:
                bases_matrix.append(F.pad(self.bases[1:], (0,self.num_nodes-2)))
            elif i == self.num_nodes-1:
                bases_matrix.append(F.pad(self.bases[:-1], (self.num_nodes-2,0)))
            else:
                bases_matrix.append(F.pad(self.bases, (i-1, self.num_nodes-2-i)))

        """
            Remember: each bases is a column vector of whole matrix
        """
        # pdb.set_trace()
        bases_matrix = torch.stack(bases_matrix, dim=1)
        return bases_matrix

    def forward(self, input):
        N, in_channels, num_nodes = input.shape

        ## first step in dcf 
        bases_matrix = self.build_local_filter()
        features_bases = torch.matmul(input, bases_matrix)
        
        ## second step, with shape [N, out_channels, num_nodes]
        features_bases = self.coeff_conv(features_bases)
        
        ## add bias
        features_bases += self.bias.unsqueeze(-1)
        
        return features_bases


class GraphConv_Bases(nn.Module):
    def __init__(self, in_channels, out_channels, level, bias=True, 
                    order_list=(1), num_bases=1, gcn_type='GCN_Bases', graph_type='ring'):
        super(GraphConv_Bases, self).__init__()
        # bases hyper-parameter
        assert gcn_type in ('GCN_Bases', 'GCN', 'ChebNet')
        assert len(order_list) == num_bases
        
        self.gcn_type = gcn_type
        self.num_bases = num_bases
        self.order_list = order_list
        self.graph_type = graph_type

        ## get graph
        print('using {} graph'.format(self.graph_type))
        A_ = gen_graph(level=level, graph_type=self.graph_type).float()

        # get bases_template from A' and order_list, with shape [num_bases, V, V]
        bases_template = self.get_bases_template(A_)
        self.register_buffer("bases_template", bases_template)
        
        ## create bases_mask, with shape [num_bases, V, V]
        if self.gcn_type == 'GCN_Bases':
            self.bases_mask = nn.Parameter(torch.Tensor(*(self.bases_template.shape)))
            # init bases, 3: avg. support size
            in_size = self.num_bases * 3 * self.bases_template.shape[0]
            std_ = math.sqrt(1. / in_size)
            nn.init.normal_(self.bases_mask, std=std_)
        else: # GCN and ChebNet have no trainable bases, thus no bases_mask
            self.bases_mask = torch.tensor(1.0)
       
        ## define coeff operation, 
        ## for three types of GCN, this is the same
        self.coeff_conv = nn.Conv1d(
            in_channels=self.num_bases*in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

        ## bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            n = in_channels * self.num_bases
            stdv = 1. / math.sqrt(n)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def get_bases_template(self, A_):
        """
        Get bases_template from adjusted A' = A + I
        Args:
            A_: A + I
        """
        bases_template = []

        for order in self.order_list:
            if self.gcn_type == 'GCN':
                assert (len(self.order_list) == 1) and (self.order_list[0]==1)
                assert self.num_bases == 1
                bases_template += [normalize_undigraph(A_)]

            elif self.gcn_type == 'ChebNet':
                assert self.order_list == list(np.arange(self.num_bases))
                # build identity martix
                I = torch.eye(A_.shape[0])
                # get original A: A'-I
                A = A_ - I
                # get normalized_rescaled_laplacian L_
                L_ = A2normed_scaled_L(A)

                # get L0, L1, ..., Lk
                if order == 0:
                    L0 = I
                    bases_template += [L0]
                
                elif order == 1:
                    L1 = L_
                    bases_template += [L1]

                else: # order >= 2
                    # L2 = 2L*L1 - L0
                    L2 = 2 * torch.matmul(L_, L1) - L0
                    bases_template += [L2]
                    # update L1, L0 and compute L2 recursively
                    L1, L0 = L2, L1 
            
            else: # 'GCN_Bases'
                # assert not (0 in self.order_list)
                bases_template += [k_th_order_A(order, A_)]
            
        bases_template = torch.stack(bases_template, dim=0)
        bases_template.requires_grad = False

        return bases_template

    def forward(self, input):
        N, in_channels, num_nodes = input.shape

        ## first step in dcf 
        features_bases = []
        rec_kernel = self.bases_template * self.bases_mask
        for kernel in rec_kernel:
            # each with shape [N, in_channels, num_nodes]
            features_bases += [torch.matmul(input, kernel)]
            
        # with shape [N, in_channels*num_bases, num_nodes]
        features_bases = torch.cat(features_bases, dim=1)
        
        ## second step, with shape [N, out_channels, num_nodes]
        features_bases = self.coeff_conv(features_bases)
        
        ## add bias
        features_bases += self.bias.unsqueeze(-1)
        
        return features_bases

class Model(nn.Module):
    def __init__(self, gcn_type, num_bases, order_list, graph_type='ring',
                    use_shared_bases=False):
        super(Model, self).__init__()
        if use_shared_bases:
            assert gcn_type == 'GCN_Bases'
        
        self.gcn_type = gcn_type
        self.num_bases = num_bases
        self.order_list = order_list
        self.graph_type = graph_type
        self.use_shared_bases = use_shared_bases

        ## 2 gc layers
        if self.use_shared_bases:
            self.gconv1 = GraphConv_Bases_Shared(1, 32, level=1)
            self.gconv2 = GraphConv_Bases_Shared(32, 64, level=2)
        else:
            self.gconv1 = GraphConv_Bases(1, 32, level=1, bias=True, gcn_type=gcn_type,
                                        num_bases=num_bases, order_list=order_list,
                                        graph_type=self.graph_type)
            self.gconv2 = GraphConv_Bases(32, 64, level=2, bias=True, gcn_type=gcn_type,
                                        num_bases=num_bases, order_list=order_list,
                                        graph_type=self.graph_type)
        
        ## bn layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        
        # calculate params for GCN layers
        self.total_params = np.sum([param.numel() for param in self.parameters()])
        if self.gcn_type == 'GCN_Bases' and not self.use_shared_bases:
            # import pdb; pdb.set_trace()
            self.total_params -= np.sum([conv.bases_template.numel() -
                    torch.sum(conv.bases_template) for conv in [self.gconv1, self.gconv2]])


        ## relu
        self.relu = nn.ReLU(inplace=True)

        ## pooling layers
        self.max_pool1 = nn.MaxPool1d(kernel_size=2)
        # pool to 1 node finally
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.gconv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.bn1(x)
        # print(x.shape)

        x = self.gconv2(x)
        x = self.relu(x)
        # print(x.shape)
        
        x = self.global_avg_pool(x)
        # x = self.max_pool2(x)

        # print(x.shape)
        x = self.bn2(x)
        # print(x.shape)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


class Model_1layer(nn.Module):
    def __init__(self, gcn_type, num_bases, order_list, graph_type='ring', 
                    use_shared_bases=False):
        super(Model_1layer, self).__init__()
        if use_shared_bases:
            assert gcn_type == 'GCN_Bases'
        self.gcn_type = gcn_type
        self.num_bases = num_bases
        self.order_list = order_list
        self.graph_type = graph_type
        self.use_shared_bases = use_shared_bases

        ## gc layer1, bn1
        if self.use_shared_bases:
            self.gconv1 = GraphConv_Bases_Shared(1, 32, level=1)
        else:
            self.gconv1 = GraphConv_Bases(1, 32, level=1, bias=True, gcn_type=gcn_type,
                                        num_bases=num_bases, order_list=order_list,
                                        graph_type=self.graph_type)
        self.bn1 = nn.BatchNorm1d(32)
        
        # calculate params for GCN layers
        self.total_params = np.sum([param.numel() for param in self.parameters()])
        if self.gcn_type == 'GCN_Bases' and not self.use_shared_bases:
            # import pdb; pdb.set_trace()
            self.total_params -= self.gconv1.bases_template.numel() - \
                                    torch.sum(self.gconv1.bases_template)

        ## relu
        self.relu = nn.ReLU(inplace=True)

        ## pooling layers
        # pool to 1 node finally
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        ## one fc layers
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.gconv1(x)
        x = self.relu(x)
        x = self.global_avg_pool(x)
        x = self.bn1(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)