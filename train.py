import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import LeNet

def main(args):
    data = torch.load(args.data_path)
    train_X, train_y = data["train_X"], data["train_y"]

    dataset = TensorDataset(train_X, train_y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(args.epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Эпоха {epoch + 1}/{args.epochs}, Потери: {loss.item()}")

    torch.save(model.state_dict(), args.model_save_path)
    print(f"Модель сохранена в {args.model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_save_path", type=str, required=True)

    args = parser.parse_args()
    main(args)

# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np

# from model import LeNet

# def main(args):
#     # Загружаем данные
#     data = np.load(args.data_path)
#     X_train, y_train = data["X"], data["y"]

#     # Преобразуем к PyTorch-формату
#     X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
#     y_train = torch.tensor(y_train, dtype=torch.long)

#     dataset = TensorDataset(X_train, y_train)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

#     # После загрузки данных из DataLoader:
#     for batch_X, batch_y in dataloader:
#         # Преобразование размера батча для работы с LeNet
#         batch_X = batch_X.view(batch_X.size(0), 1, 28, 28).to(device)  # Преобразование в [batch_size, 1, 28, 28]
#         batch_y = batch_y.to(device)

#     # Forward pass
#     outputs = model(batch_X)
#     model = LeNet()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Обучение
#     model.train()
#     for epoch in range(args.epochs):
#         for batch_X, batch_y in dataloader:
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()

#         print(f"Эпоха {epoch + 1}/{args.epochs}, Потери: {loss.item()}")

#     # Сохраняем модель
#     torch.save(model.state_dict(), args.model_save_path)
#     print(f"Модель сохранена в {args.model_save_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", type=str, required=True)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--model_save_path", type=str, required=True)

#     args = parser.parse_args()
#     main(args)

