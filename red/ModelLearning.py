import torch

X_train = torch.load("X_train.pt")
X_val = torch.load("X_val.pt")
y_train = torch.load("y_train.pt")
y_val = torch.load("y_val.pt")

W = torch.randn(11, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

alpha = 0.01

for _ in range(1000):
    y_hat = X_train @ W + b
    loss = ((y_hat - y_train)**2).mean()
    loss.backward()
    with torch.no_grad():
        W -= alpha*W.grad
        b -= alpha*b.grad
    W.grad = None
    b.grad = None
    print(f"Ошибка: {loss:.5f}")
with torch.no_grad():
    y_hat = X_val @ W + b
score = (y_hat - y_val).abs().mean()
print(f"Среднее отклонение: {score:.2f} балла")
