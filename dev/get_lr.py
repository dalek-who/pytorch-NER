import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

class Mongo(nn.Module):
    def __init__(self):
        super(Mongo, self).__init__()
        self.linear = nn.Linear(10, 20)
    def forward(self, x):
        return self.linear(x)

lr = 0.015
decay = 0.05
momentum: float = 0.9
lr_lambda = lambda epoch: (1 / (1 + decay * epoch))
# lr_lambda = lambda epoch: 0.1 ** epoch

model: nn.Module = Mongo()
optimizer = SGD(model.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

x = torch.zeros(10)
sc_lr_list = []
my_lr_list = []
my_lr = lr
for epoch in range(60):
    sc_lr = scheduler.get_lr()[0]
    my_lr = lr * lr_lambda(epoch)

    sc_lr_list.append(sc_lr)
    my_lr_list.append(my_lr)

    print(my_lr_list[-1] == sc_lr_list[-1])

    scheduler.step()
print(my_lr_list == sc_lr_list)