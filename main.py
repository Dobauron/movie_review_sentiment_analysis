import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


# torch.manual_seed(41) ustawia ziarno losowości (seed) w PyTorchu, co powoduje, że operacje losowe (np. inicjalizacja wag, shuffling danych) będą powtarzalne między różnymi uruchomieniami kodu.
torch.manual_seed(41)

model = Model()

import matplotlib.pyplot as plt
import pandas as pd

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
iris_DataFrame = pd.read_csv(url)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# transformacja tabeli klasyfikacyjnej z tekstowej na liczbowa
le = LabelEncoder()
iris_DataFrame['species_numeric'] = le.fit_transform(iris_DataFrame.iloc[:, -1])

# test train split
X = iris_DataFrame.drop(["species", "species_numeric"], axis=1)
y = iris_DataFrame['species_numeric']

# convert to numpy array
X = X.values
y = y.values

# train, test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())

    if i % 10 == 0:
        print(f'epoch {i} and loss: {loss}')

    optimizer.zero_grad()  # zeruje gradienty
    loss.backward()  # propagacja wsteczna błędu - implementacja w pytorch
    optimizer.step()  # aktualizuje wagi

#
# plt.plot(range(epochs), losses)
# plt.ylabel("loss/error")
# plt.xlabel("Epoch")
# plt.show()

with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)
    print(loss)

correct = 0

with torch.no_grad():
    for i, data in enumerate(X_test):
        if y_test[i] == 0:
            x = "Setosa"
        elif y_test[i] == 1:
            x = "Versicolor"
        else:
            x = "Virginica"

        y_val = model.forward(data)  # przepuszcza dane testowe przez funkcj forward raz
        print(f'{i + 1}.)  {str(y_val)} \t {x} \t {y_val.argmax().item()}')

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f"We got {correct}, correct")

# adding new data from outside

new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])


with torch.no_grad():
    print(model(new_iris))


newer_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])

with torch.no_grad():
    print(model(newer_iris))


# Save nn model
torch.save(model.state_dict(), 'my_really_Awsome_iris_model.pt')

#load saved model

new_model = Model()
new_model.load_state_dict(torch.load('my_really_Awsome_iris_model.pt'))

print(new_model.eval())