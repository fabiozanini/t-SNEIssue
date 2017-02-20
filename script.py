import matplotlib.pyplot as plt # matplotlib 1.4.3 
from sklearn.manifold import TSNE # scikit-learn 0.17
import pandas # pandas 0.16.2

# Read data
data = pandas.read_csv("data.csv", sep=",")

# Create figure
hFig, hAxs = plt.subplots(2, 1, figsize=(7, 16))
for hAx, eta in zip(hAxs, [1000, 200]):
    # Fit model
    model = TSNE(n_components=2, perplexity=10, verbose=2, method='barnes_hut',
                 init='pca', n_iter=1000,
                 learning_rate=eta)
    model.fit(data.values.T)
    # Plot results
    hAx.scatter(model.embedding_[:, 0], model.embedding_[:, 1], 20, color="grey")
    for i, txt in enumerate(data.keys()):
        hAx.annotate(txt, (model.embedding_[i, 0], model.embedding_[i, 1]))
    hAx.set_title('Learning rate: '+str(eta))


plt.tight_layout()
plt.savefig('ResultScriptPy.png')

plt.ion()
plt.show()
