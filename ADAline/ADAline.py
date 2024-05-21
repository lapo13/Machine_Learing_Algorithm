import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Perceptron class, il percettrone è un classificatore binario lineare
# che permette di classificare un insieme di dati in due classi distinte da una retta
# (o iperpiano) definita da un vettore di pesi w e un termine di bias b. Se il prodotto
# scalare tra il vettore di pesi e il vettore di input è maggiore di una soglia, il percettrone
# classifica l'input come appartenente alla classe 1, altrimenti come appartenente alla classe -1.

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=[cmap(idx)],
                    marker=markers[idx], label=cl)

class Adaline(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        #viene usando il metodo zip per iterare su due liste contemporaneamente, questo permette di
        #implementare la backpropagation in modo più efficiente e veloce, in quanto si aggiornano i pesi (la back propagation consisterà nel calcolare l'errore rispetto al target e aggiornare i pesi)
        #dopo ogni iterazione.
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output) #calcolo dell'errore, può divergere se il tasso di apprendimento è troppo alto, poichè la funzione di costo è continua e differenziabile,
            # se il passo verso il minimo è troppo grande, si può scivolare oltre il minimo e risultare in un errore negativo che farà aumentare la funzione di costo.
            self.w_[1:] += self.eta * X.T.dot(errors) #aggiornamento dei pesi
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Load Iris dataset

s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')
print('URL:', s)
df = pd.read_csv(s, header=None, encoding='utf-8')
df.tail()

# Plot Iris dataset
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

#plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
#plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
#plt.xlabel('sepal length [cm]')
#plt.ylabel('petal length [cm]')
#plt.legend(loc='upper left')
#plt.show()

# Train Adaline model
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = Adaline(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = Adaline(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()
# In questo caso, si può osservare come la funzione di costo decresca in modo più lento
# rispetto al percettrone, questo è dovuto al fatto che la funzione di costo è continua e differenziabile
# rispetto alla funzione di costo del percettrone, che è discontinua e non differenziabile.
# Inoltre, si può osservare come il tasso di apprendimento influenzi la convergenza del modello:
# un tasso di apprendimento troppo alto può causare oscillazioni e divergenza, mentre un tasso di apprendimento
# troppo basso può causare una convergenza lenta e inefficiente.


# Standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = Adaline(n_iter=15, eta=0.01)
ada.fit(X_std, y)

# Plot decision regions
plot_decision_regions(X_std, y, classifier=ada)
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.tight_layout()
plt.show()