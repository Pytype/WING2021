import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt


dataset = datasets.load_boston()

x_data = torch.from_numpy(dataset.data).float()
y_data = torch.from_numpy(dataset.target).float()

x = torch.zeros(506, 1, dtype=torch.float)
y = torch.zeros(506, 1, dtype=torch.float)


k=-1
while(k<0 or k>13):
    print("'0:CRIM' '1:ZN' '2:INDUS' '3:CHAS' '4:NOX' '5:RM' '6:AGE' '7:DIS' '8:RAD' '9:TAX' '10:PTRATIO' '11:B' '12:LSTAT'")
    print()
    k = int(input("0 ~ 12의 수를 입력 : "))
    if (k<0 or k>13):
        print("잘못된 입력입니다.")
        print()

print(dataset['feature_names'][k], end="")
print("과 보스턴 집값의 선형 관계를 분석합니다.")
print()

for i in range(506):
    x[i]=x_data[i][k]
    y[i]=y_data[i]

plt.scatter(x, y, s=7, c="blue")
plt.xlabel(dataset['feature_names'][k])
plt.ylabel('Price')
plt.show()


model = nn.Linear(1, 1)
costf = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 10000
costarr = []

for i in range(epochs + 1):
    optimizer.zero_grad()
    prediction = model(x)

    cost = costf(prediction, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    costarr.append(cost.item())

    if i % 500 == 0:
        print('Epoch {:4d}/{}'.format(i, epochs))
        print('\tCost : %f' % cost.item())
        result = list(model.parameters())
        print('\tW = %f, \t b = %f' % (result[0].item(), result[1].item()))

        if i % 2000 == 0:
            plt.scatter(x.numpy(), y.numpy(), s=7, c="gray")
            plt.plot(x.detach().numpy(), prediction.detach().numpy(), c="red")
            plt.xlabel(dataset['feature_names'][k])
            plt.ylabel('Price')
            plt.show()

print()
print()
print('W : ' + str(round(result[0].item(),6)))
print('b : ' + str(round(result[1].item(),6)))
print('Cost : ' + str(round(cost.item(), 6)))
print()

print("Price = %.6f * %s + %.6f" % (result[0].item(), dataset['feature_names'][k], result[1].item()))
print()

x2 = [n for n in range(0,10001)]
plt.plot(x2,costarr, c="orange")
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()