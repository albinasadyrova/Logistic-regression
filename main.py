def main():
    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Загрузка данных
    digits = load_digits()
    X, y = digits.data, digits.target

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Преобразование меток для логистической регрессии (бинарная классификация)
    y_train_binary = (y_train == 5).astype(int)
    y_test_binary = (y_test == 5).astype(int)

    # Метод k ближайших соседей (KNN)
    class KNNClassifier:
        def __init__(self, k=3):
            self.k = k

        def fit(self, X, y):
            self.X_train = X
            self.y_train = y

        def predict(self, X_test):
            y_pred = [self._predict(x) for x in X_test]
            return np.array(y_pred)

        def _predict(self, x):
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            return most_common

        def _euclidean_distance(self, x1, x2):
            return np.sqrt(np.sum((x1 - x2)**2))

    # Значения k для тестирования
    k_values = [3, 5, 7, 9]

    for k in k_values:
        # KNN с определенным значением k
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        print(f"KNN (k={k}): Точность на тестовом наборе данных: {accuracy_knn:.2f}")

    # Логистическая регрессия
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    class LogisticRegression:
        def __init__(self, learning_rate=0.01, n_iterations=1000):
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations

        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
            self.weights = np.zeros(X.shape[1])
            self.bias = 0

            for _ in range(self.n_iterations):
                linear_model = np.dot(self.X_train, self.weights) + self.bias
                y_pred = sigmoid(linear_model)

                dw = (1 / len(self.X_train)) * np.dot(self.X_train.T, (y_pred - self.y_train))
                db = (1 / len(self.X_train)) * np.sum(y_pred - self.y_train)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        def predict(self, X_test):
            linear_model = np.dot(X_test, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
            return np.array(y_pred_class)

    # Значения скорости обучения для тестирования
    learning_rates = [0.001, 0.01, 0.1]

    for learning_rate in learning_rates:
        # Логистическая регрессия с определенной скоростью обучения
        logreg = LogisticRegression(learning_rate=learning_rate)
        logreg.fit(X_train, y_train_binary)
        y_pred_logreg = logreg.predict(X_test)
        accuracy_logreg = accuracy_score(y_test_binary, y_pred_logreg)
        print(f"Логистическая регрессия (learning_rate={learning_rate}): Точность на тестовом наборе данных: {accuracy_logreg:.2f}")

if __name__ == "__main__":
    main()
