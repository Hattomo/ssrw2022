import nbclean
c = nbclean.clean.NotebookCleaner('./train.ipynb')
c.clear(kind='output')
c.save('./train.ipynb')