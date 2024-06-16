import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
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

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolors='black',
                    alpha=1.0, linewidths=1.5, marker='o', s=100, label='test set')


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y)) #la funzione np.unique restituisce le tre etichette attribuite ai fiori

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
#la funzione train_test_split divide il dataset in due parti, una per il training e una per il test.
# Il parametro stratify=y garantisce che le etichette siano distribuite in modo uniforme nei due set
#inoltre il parametro random_state=1 garantisce che la divisione sia sempre la stessa

print('Labels counts in y:', np.bincount(y)) #la funzione np.bincount conta il numero di occorrenze di ciascuna etichetta
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

sc = StandardScaler()#l' oggetto StandardScaler ci consente di standardizzare le features
sc.fit(X_train) #calcola la media e la deviazione standard per ogni feature, usando l'oggetto sc e il metodo fit

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#usa il metodo transform dell'oggetto standardscaler per standardizzare le features del training e del test set

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
print('Accuracy: %.2f' % ppn.score(X_test_std, y_test)) #il metodo score calcola l'accuratezza del modello
# e restituisce il rapporto tra il numero di previsioni corrette e il numero totale di previsioni

#stampiamo le zone di decisione del modello

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
