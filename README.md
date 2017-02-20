# Minimal example to inconsistency issue with sklearn.manifold.TSNE

The files needed to reproduce the issue are as follows (versions of python packages are commented in the script file):

1. `data.csv`: It's a 3000 x 49 data matrix
2. `script.py`: Use t-SNE to project the data matrix onto 2D (note we're projecting the transponse of data.csv i.e we have 49 values to project) (barnes_hut method)
3. `script.R`: Same as `script.py` but in R (barnes_hut method)
4. `Maaten_script.py`: Same as `script.py` but uses the exact method. The core of the script was taken from [Van der Maaten](https://lvdmaaten.github.io/tsne/code/tsne_python.zip) 

The issue is that the results obtained from `script.py` using `sklearn.manifold.TSNE` are inconsistent with those obtained from `script.R` and `Maaten_script.py`.
The data points named A, B, C,... should cluster together. That is, A is close to B, B is close to C,..., AV is close to AW etc..The expected result is
obtained when using `script.R` or `Maaten_script.py`, even when multiple starting points are used. But for `script.py`, data points aren't separated.

Expected results obtained by: 

`script.R`

![alt text][ScriptR]
[ScriptR]: https://raw.githubusercontent.com/jjvalletta/t-SNEIssue/master/ResultScriptR.png "Result script.R"

and `Maaten_script.py`

![alt text][MaatenScript]
[MaatenScript]: https://raw.githubusercontent.com/jjvalletta/t-SNEIssue/master/ResultMaaten_scripty.png "Result Maaten_script.py"

Unexpected result obtained by `script.py` using `sklearn.manifold.TSNE`

![alt text][ScriptPy]
[ScriptPy]: https://raw.githubusercontent.com/iosonofabio/t-SNEIssue/master/ResultScriptPy.png "Result script.py"

Observations for `script.py`:

* Irrespective of starting point (`random_state`) optimiser seems to arrive to the same solution every time (iteration=175, error=1.820793, gradient norm = 0.0009956)
* Changing `n_iter` has no effect as the optimiser is exiting early because of `min_grad_norm`
* Inspecting the source code at `python2.7/site-packages/sklearn/mainfold/t_sne.py` I found that:
  1. Line 812: if `method == 'barnes_hut'` then `min_grad_norm` gets overwritten by a hard-coded value of 1e-3. 
  2. Changing this does make the optimiser run for longer, the result doesn't change much though
  3. I also tried using `method == 'exact'` Same though, optimiser runs for longer, but result doesn't change much. The `error` is still high around 2.7. as opposed to the other scripts where it drops to <0.1
* My guess is that the issue is in `_gradient_descent` but can't figure it out

Thanks for anyone looking into this.
   
