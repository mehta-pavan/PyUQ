.. specPy documentation master file, created by
   sphinx-quickstart on Sat Sep  2 01:27:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyUQ : Uncertainty Quantification and Data Assimilation - A Python library
==========================================================================


PyUQ is a python library for uncertainty quantifcation and data assimilation, combining the features of c-ANOVA, POD and Krigging. 

Sensitivty analysis is performed using Sobol indices, and sampling via Sobal sequence

This is library was developed during the author's time at M2P2 Lab, France under supervision of Prof. Pierre Sagaut.  



Author
======

Pavan Pranjivan Mehta
SISSA mathLab, Italy
Email : pavan.mehta@sissa.it
Web : https://www.pavanpmehta.com/


Documentation
=============


Please read the PDF provided within the directory `capk`


https://github.com/mehta-pavan/PyUQ/tree/master/capk




.. toctree::
   :maxdepth: 1
   :caption: Tutorial:
   
   test_file



.. toctree::
   :maxdepth: 1
   :caption: API Reference:
   
   capk_v2
   cross_validation
   cv
   anova_4D
   anova_adaptive_creteria
   anova_decomposition
   anova_terms
   kernel
   krig_for_anova
   krigging
   pod
   sobol
   sample
   loo
   sobol_indices_3
   reinforce
   stat_var_expect
   
   
   
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
