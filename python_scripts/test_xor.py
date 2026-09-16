import json
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader 

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Use square brackets to create lists instead of sets.
X = [
    [0.02, 0.08], [0.95, 0.03], [0.01, 0.92], [0.98, 0.91], [0.05, 0.04], [0.89, 0.01], [0.03, 0.96],
    [0.92, 0.99], [0.07, 0.01], [0.96, 0.05], [0.02, 0.88], [0.88, 0.94], [0.04, 0.06], [0.91, 0.02],
    [0.06, 0.95], [0.97, 0.93], [0.01, 0.03], [0.94, 0.08], [0.08, 0.97], [0.91, 0.89], [0.03, 0.02],
    [0.99, 0.04], [0.05, 0.92], [0.95, 0.98], [0.02, 0.01], [0.91, 0.06], [0.07, 0.94], [0.93, 0.96],
    [0.04, 0.05], [0.98, 0.01]  
]

Y_labels = [
    0, 1, 1, 0, 0, 1, 1,
    0, 0, 1, 1, 0, 0, 1,
    1, 0, 0, 1, 1, 0, 0,
    1, 1, 0, 0, 1, 1, 0, 0, 1
]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        
        self.layers = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
               
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 

    def forward(self, x):
        return self.layers(x)


def training_loop(train_dataloader, test_dataloader, model, optimizer, loss_function):
    epochs = 120

    for i in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        test_loss = 0.0
        test_accuracy = 0.0

        n_batch = 0
        model.train()
        for data, target in train_dataloader:
            optimizer.zero_grad()
            prediction = model(data)
            loss = loss_function(prediction, target)

            _, indices = torch.max(prediction, 1)
            train_accuracy += (indices == target).sum().item()
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            n_batch += 1

        train_loss /= n_batch
        # Divide by the total dataset size to calculate the accuracy.
        train_accuracy /= len(train_dataloader.dataset)

        n_batch = 0
        model.eval()
        with torch.no_grad():
            for data, target in test_dataloader:
                prediction = model(data)
                loss = loss_function(prediction, target)
                _, indices = torch.max(prediction, 1)
                test_accuracy += (indices == target).sum().item()
                test_loss += loss.item()
                n_batch += 1

        test_loss /= n_batch
        test_accuracy /= len(test_dataloader.dataset)

        if((i + 1) % 10 == 0 or i == 0):
            print(f'Epoch {i+1}/{epochs} | loss: {train_loss:.4f} | train_acc: {train_accuracy:.4f} | val_loss: {test_loss:.4f} | val_acc: {test_accuracy:.4f}')


def load_cpp_weights(model, filename):
    """Loads flattened C++ weights into the matching PyTorch parameters."""
    values = torch.tensor(pd.read_csv(filename, header=None).to_numpy().flatten(), dtype=torch.float32)
    offset = 0

    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            if offset + size > values.numel():
                raise ValueError("The CSV contains fewer weights than the PyTorch model")

            if parameter.ndim == 2:
                cpp_shape = (parameter.shape[1], parameter.shape[0])
                # C++ stores Linear weights as [input, output], while PyTorch expects [output, input].
                parameter_values = values[offset:offset + size].reshape(cpp_shape).transpose(0, 1)
            else:
                parameter_values = values[offset:offset + size].reshape(parameter.shape)

            parameter.copy_(parameter_values)
            offset += size

    if offset != values.numel():
        raise ValueError("The CSV contains more weights than the PyTorch model")


def save_gradients(model, filename):
    """Saves the current PyTorch gradients to a JSON file."""
    gradients = {}

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            gradients[name] = None
            continue

        gradients[name] = {
            "shape": list(parameter.grad.shape),
            "values": parameter.grad.detach().cpu().tolist(),
        }

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump({"gradients": gradients}, output, indent=2)


def main():
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_labels, dtype=torch.long)

    X_train, X_test = X_tensor[:21], X_tensor[21:]
    Y_train, Y_test = Y_tensor[:21], Y_tensor[21:]

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=7)
    test_dataloader = DataLoader(test_dataset, batch_size=9)
    model = Model()
    load_cpp_weights(model, PROJECT_ROOT / "weights" / "xor_weights.csv")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    loss_function = nn.CrossEntropyLoss()

    training_loop(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function
    )
    save_gradients(model, PROJECT_ROOT / "gradients" / "gradients_python.json")


# Run the training script when this file is executed directly.
if __name__ == '__main__':
    main()