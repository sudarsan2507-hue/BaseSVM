import csv
from typing import List, Tuple


def load_csv(path: str) -> Tuple[List[float], List[int]]:
    xs: List[float] = []
    ys: List[int] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(int(row["y"]))
    return xs, ys


def train_linear_svm(
    xs: List[float],
    ys: List[int],
    lr: float = 0.01,
    reg: float = 0.01,
    epochs: int = 2000,
) -> Tuple[float, float]:
    w = 0.0
    b = 0.0
    n = len(xs)
    for _ in range(epochs):
        for x, y in zip(xs, ys):
            margin = y * (w * x + b)
            if margin >= 1:
                w = w - lr * (2 * reg * w)
            else:
                w = w - lr * (2 * reg * w - y * x)
                b = b + lr * y
    return w, b


def predict(w: float, b: float, x: float) -> int:
    return 1 if (w * x + b) >= 0 else -1


def main():
    train_xs, train_ys = load_csv("data/svm_hours_train.csv")
    test_xs, test_ys = load_csv("data/svm_hours_test.csv")

    w, b = train_linear_svm(train_xs, train_ys)
    print(f"Weight: {w:.6f}")
    print(f"Bias: {b:.6f}")

    train_preds = [predict(w, b, x) for x in train_xs]
    train_acc = sum(1 for p, y in zip(train_preds, train_ys) if p == y) / len(train_ys)
    print(f"Training accuracy: {train_acc:.2f}")

    test_preds = [predict(w, b, x) for x in test_xs]
    test_acc = sum(1 for p, y in zip(test_preds, test_ys) if p == y) / len(test_ys)
    print(f"Test accuracy: {test_acc:.2f}")

    print("\nTest samples:")
    for x, y, p in zip(test_xs, test_ys, test_preds):
        decision = w * x + b
        print(f"x={x}\ttrue={y}\tpred={p}\tdecision={decision:.4f}")


if __name__ == "__main__":
    main()
