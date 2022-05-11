import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y
from torch import nn, optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset


class PhIPSeqDataset(Dataset):
    """Phage IP Sequencing dataset."""

    def __init__(self, x, y=None, transform=None):
        self.transform = transform
        self.x = x.values if type(x) == pd.DataFrame else x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.y is not None:
            output_group = self.y[idx]
        else:
            output_group = 0.5
        output_vector = np.array([1 - output_group, output_group])

        sample = {'input': torch.from_numpy(self.x[idx]), 'output': torch.from_numpy(output_vector)}
        if self.transform:
            sample = self.transform(sample)

        return sample


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        trainloader = DataLoader(PhIPSeqDataset(X, y), batch_size=4, shuffle=True)
        net = NeuralNetwork(X.shape[1]).to('cpu')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(20):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data['input'], data['output']

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs.float())
                loss = criterion(outputs.float(), labels.float())
                loss.backward()
                optimizer.step()
            self.net_ = net
            self.criterion_ = criterion
            self.optimizer = optimizer

    def predict_proba(self, X):
        # Check that fit had been called
        check_is_fitted(self)
        testloader = DataLoader(PhIPSeqDataset(X), batch_size=1, shuffle=True)
        pred_y = []
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data['input'], data['output']
                print(inputs.shape)
                # forward + backward + optimize
                outputs = self.net_(inputs.float())
                pred_y.append(outputs)
        ret = list(map(lambda v: softmax(v, dim=len(v)).cpu().detach().numpy()[0], pred_y))
        return ret

    def predict(self, X):
        probas = self.predict_proba(X)
        return list(map(np.argmax, probas))

