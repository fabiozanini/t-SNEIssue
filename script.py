import matplotlib.pyplot as plt # matplotlib 1.4.3 
from sklearn.manifold import TSNE # scikit-learn 0.17
import pandas # pandas 0.16.2

# Read data
data = pandas.read_csv("data.csv", sep=",")
# Fit model
model = TSNE(n_components=2, perplexity=10, verbose=2, method='barnes_hut', init='pca', n_iter=1000)
model.fit(data.values.T) 
# Plot results
hFig, hAx = plt.subplots()
hAx.scatter(model.embedding_[:, 0], model.embedding_[:, 1], 20, color="grey")
for i, txt in enumerate(data.keys()):
    hAx.annotate(txt, (model.embedding_[i, 0], model.embedding_[i, 1]))