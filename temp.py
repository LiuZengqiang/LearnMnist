import torch
import torch.nn.functional as nnf

target = torch.tensor([1, 2, 3])
x = torch.randn(2,3)
print(x)

print(x.argmax())
print(x.argmax(axis=0))
y = torch.sum(x, dim=0)
print("y", y)
print(x.argmax(axis=1))

# x = torch.zeros(3, 5)

# x[0][0] = 1
# x[1][0] = 1
# x[2][0] = 1

# loss_fn = torch.nn.CrossEntropyLoss()

# print(loss_fn(x, target.long()))
