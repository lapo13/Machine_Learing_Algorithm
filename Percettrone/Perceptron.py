import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Perceptron class, il percettrone è un classificatore binario lineare
# che permette di classificare un insieme di dati in due classi distinte da una retta
# (o iperpiano) definita da un vettore di pesi w e un termine di bias b. Se il prodotto
# scalare tra il vettore di pesi e il vettore di input è maggiore di una soglia, il percettrone
# classifica l'input come appartenente alla classe 1, altrimenti come appartenente alla classe -1.

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        #viene usando il metodo zip per iterare su due liste contemporaneamente, questo permette di
        #implementare la backpropagation in modo più efficiente e veloce, in quanto si aggiornano i pesi (la back propagation consisterà nel calcolare l'errore rispetto al target e aggiornare i pesi)
        #dopo ogni iterazione.
        for _ in range(self.n_iter):
            errors = 0
            print('epoch:', _)
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

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

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# Train Perceptron model
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

#lo scopo del progetto è quello di implementare un percettrone e studiarne il comportamento su un dataset di esempio.
#Il dataset scelto è il dataset Iris, che contiene 150 esempi di fiori, divisi in 3 classi.
#Per semplicità, si è scelto di considerare solo i primi 100 esempi, relativi alle prime due classi.
#Il dataset contiene 4 features, ma per semplicità si è scelto di considerare solo le prime due.
#I grafici mostrano la distribuzione dei dati e il numero di aggiornamenti dei pesi del percettrone per ogni epoca.