import torch
import numpy as np
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import math
seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)
class GraphConvolution_1(Module):
    def __init__(self,input,dm,output,bias = True):
        super(GraphConvolution_1,self).__init__()
        self.input = input
        self.output  = output
        self.weight = Parameter(torch.FloatTensor(input,7))
        self.theta = Parameter(torch.FloatTensor(7,output))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = np.sqrt(6.0 / (self.weight.shape[0] + self.weight.shape[1]))
        # stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        self.theta.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)


    def forward(self,input,adj):

        support = torch.mm(input,self.weight)
        output = torch.spmm(adj, support)
        output = torch.mm(output,self.theta)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphConvolution_2(Module):
    def __init__(self, input, output, bias=True):
        super(GraphConvolution_2, self).__init__()
        self.input = input
        self.output = output
        self.weight = Parameter(torch.FloatTensor(input, output))

        if bias:
            self.bias = Parameter(torch.FloatTensor(output))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = np.sqrt(6.0 / (self.weight.shape[0] + self.weight.shape[1]))
        # stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

