library(Rtsne) # load t-SNE library

# Read data
data <- read.table(file="data.csv", header=T, sep=",", skip=0) # read data
# Fit model
model <- Rtsne(t(data), dims=2, perplexity=10, verbose=T, pca=TRUE, max_iter = 1000) # Run TSNE method=Barnes-Hut
# Plot results
plot(model$Y, type="n")
text(model$Y[, 1], model$Y[, 2], labels=colnames(data), cex=1, col="black")
