from __future__ import print_function
import torch
from util import *
import torch.nn.functional as F
import torch.optim as optim
import time
from models import GCN_FS,GCN_Class
import heapq
import scipy.sparse as scipy
import argparse
Content_path = 'cora/cora.content'
Cites_path = 'cora/cora.cites'

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)
# adj,features,labels,train_y,test_y,val_y = load_data_repeat("cora")
# print(adj)


adj,features,labels,train_y,test_y,val_y = load_data(Content_path,Cites_path)

print(train_y,test_y,val_y)
labels = torch.LongTensor(np.where(labels)[1])
features = preprocess_features(features)
# features = features.todense()
features = torch.Tensor(features)
adj = preprocess_adj(adj)
adj = torch.Tensor(adj)
# print(adj)
# print(features)
print(labels)
#models
alpha = 0
beta = 1000
CLASS_hid = 16
FS_hid = 7
FS_mid = 32
model = GCN_FS(inputfeature=features.shape[1],
            hidden_units= FS_hid,
            middle_output = FS_mid,
            nclass = labels.max().item()+1,
            dropout= 0.5)
optimizer = optim.Adam(model.parameters(),lr = 0.01,weight_decay= 5e-4)

def train(epoch,features,model,alpha,beta):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    outputs = model(features,adj)

    l21_reg = 0
    temp = []
    params = model.parameters()
    for param in params:
        for i in range(param.shape[0]):
            temp.append(torch.norm(param[i]))
        temp = torch.Tensor(temp)
        l21_reg = torch.norm(temp,p=1)
        W = param
        break
    # # XW-Z
    Z = outputs
    Q = features.mm(W)
    normF = torch.norm(Q[train_y] - Z[train_y])

    loss_train = F.nll_loss(outputs[train_y],labels[train_y]) + alpha* normF * normF  + beta * l21_reg     # W的

    acc_train  = accuracy(outputs[train_y],labels[train_y])
    loss_train.backward()
    optimizer.step()
    loss_val = F.nll_loss(outputs[val_y], labels[val_y])
    acc_val = accuracy(outputs[val_y], labels[val_y])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return W
def test(features,model):
    model.eval()
    outputs = model(features,adj)
    loss_test = F.nll_loss(outputs[test_y],labels[test_y])
    acc_test = accuracy(outputs[test_y],labels[test_y])
    print("Test set results:",
         "loss= {:.4f}".format(loss_test.item()),
         "accuracy= {:.4f}".format(acc_test.item()))
# def eval(features,model):
#     model.eval()
#     outputs = model(features, adj)
#     loss_val = F.nll_loss(outputs[val_y], labels[val_y])
#
#     acc_val = accuracy(outputs[val_y], labels[val_y])
#     print("Val set results:",
#           "loss= {:.4f}".format(loss_val.item()),
#           "accuracy= {:.4f}".format(acc_val.item()))
#train

for epoch in range(200):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    outputs = model(features, adj)

    l21_reg = 0
    temp = []
    params = model.parameters()
    for param in params:
        for i in range(param.shape[0]):
            l21_reg += torch.norm(param[i])


        W = param
        break
    #     # # XW-Z
    Z = outputs
    # Q = features[:,train_y].mm(W[train_y])
    Q = features.mm(W)
    # normF = torch.norm(Q[train_y] - Z[train_y])
    normF = torch.norm(Q - Z)
    loss_train = F.nll_loss(outputs[train_y], labels[train_y])  ++ alpha* normF * normF+  beta * l21_reg     # W的

    loss_val = F.nll_loss(outputs[val_y], labels[val_y])
    acc_train = accuracy(outputs[train_y], labels[train_y])
    acc_val = accuracy(outputs[val_y], labels[val_y])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    acc_train = accuracy(outputs[train_y], labels[train_y])
    loss_train.backward()
    optimizer.step()

    # if (epoch+1) % 20 == 0:
    #     eval(features,model)
print('train_finished')

test(features,model)



scores = torch.norm(W,p=2,dim=1)
scores = scores.detach().numpy().tolist()

select_num = [int(features.shape[1] * 0.1),int(features.shape[1] * 0.2),int(features.shape[1] * 0.3),int(features.shape[1] * 0.4),int(features.shape[1] * 0.5),int(features.shape[1] * 0.6),int(features.shape[1] * 0.7),int(features.shape[1] * 0.8),features.shape[1]]
for n in select_num:

    import heapq
    idx = map(scores.index, heapq.nlargest(n, scores))
    indices = []
    for i in idx:
        indices.append(i)

    indices.sort()
    # print(indices)
    print("n=",n)


    features_new = features[:,indices]
    eval_model = GCN_Class(inputfeature=features_new.shape[1],
                hidden_units= CLASS_hid,
                nclass = labels.max().item()+1,
                dropout= 0.5)
    optimizer = optim.Adam(eval_model.parameters(),lr = 0.01,weight_decay= 5e-4)
    for epoch in range(200):
        eval_model.train()
        optimizer.zero_grad()
        outputs = eval_model(features_new, adj)
        loss_train = F.nll_loss(outputs[train_y], labels[train_y])
        acc_train = accuracy(outputs[train_y], labels[train_y])
        loss_train.backward()
        optimizer.step()
        # if (epoch + 1) % 20 == 0:
        #     eval(features_new,model)
    test(features_new,eval_model)