import torch.nn as nn

from layers import GraphConvolution_1,GraphConvolution_2
import torch.nn.functional as F

class GCN_FS(nn.Module):
    def __init__(self,inputfeature,hidden_units,middle_output,nclass,dropout):
        super(GCN_FS, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution_1(inputfeature,hidden_units,middle_output)
        #self.gc1 = GraphConvolution_2(inputfeature, hidden_units)
        self.gc2 = GraphConvolution_2(middle_output,nclass)

    def forward(self,x,adj ):
        x = self.gc1(x,adj)
        x = F.dropout(x,self.dropout)
        x = self.gc2(x,adj)
        x = F.dropout(x,self.dropout)
        return F.log_softmax(x, dim=1)


class GCN_Class(nn.Module):
    def __init__(self,inputfeature,hidden_units,nclass,dropout):
        super(GCN_Class, self).__init__()
        self.dropout = dropout
        #self.gc1 = GraphConvolution_1(inputfeature,hidden_units,middle_output)
        self.gc1 = GraphConvolution_2(inputfeature, hidden_units)
        self.gc2 = GraphConvolution_2(hidden_units,nclass)

    def forward(self,x,adj ):
        x = self.gc1(x,adj)
        x = F.dropout(x,self.dropout)
        x = self.gc2(x,adj)
        x = F.dropout(x,self.dropout)
        return F.log_softmax(x, dim=1)
