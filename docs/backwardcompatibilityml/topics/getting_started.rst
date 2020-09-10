.. _getting_started:

Getting Started
===============

Backward Compatibility ML library requirements
----------------------------------------------

The requirements for installing and running the Backward Compatibility ML library are:

    - Windows 10 / Linux OS (tested on Ubuntu 18.04 LTS)
    - Python 3.6

Installing the Backward Compatibility ML library
------------------------------------------------

Follow these steps to install the Backward Compatibility ML library on your computer. 
You may want to `install Anaconda <https://www.anaconda.com/distribution/>`_ 
(or other virtual environment) on your system for convenience, then follow these steps:

    **1. (optional) Prepare a conda virtual environment:**
      
        .. code-block:: bash

            conda create -n bcml python=3.6
            conda activate bcml

    **2. (optional) Ensure you have the latest pip**
  
        .. code-block:: bash

            python -m pip install --upgrade pip

    **3. Install the Backward Compatibility ML library:**

        .. code-block:: bash

            pip install -U backwardcompatibilityml

    **4. Import the `backwardcompatibilityml` package in your code. For example:**

        .. code-block:: python

            import backwardcompatibilityml.loss as bcloss
            import backwardcompatibilityml.scores as scores

Running the Backward Compatibility ML library examples
------------------------------------------------------

.. note::
    The Backward Compatibility ML library examples were developed as Jupyter Notebooks
    and require the `Jupyter Software <https://jupyter.org/install>`_ to be installed.
    The steps below assume that you have `git <https://git-scm.com/downloads>`_ installed
    on your system. 

The Backward Compatibility ML library includes several examples so you can quickly 
get an idea of its benefits and learn how to integrate it into your existing ML training workflow.

To download and run the examples, follow these steps:

**1. Clone the BackwardCompatibilityML repository:**
      
        .. code-block:: bash

            git clone https://github.com/microsoft/BackwardCompatibilityML.git

**2. Start your Jupyter Notebooks server and load an example notebook under the `examples` folder:**
      
        .. code-block:: bash

            cd BackwardCompatibilityML
            cd examples
            jupyter notebook

Backward Compatibility ML library examples included
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :file: examples.csv

Next steps
----------

Do you want to learn how to integrate the Backward Compatibility ML Loss Function in your new or existing ML training workflows? :ref:`Follow this tutorial. <integrating_loss_functions>`

If you want to ask us a question, suggest a feature or report a bug, please contact the team by filing an issue in our repository on `GitHub. <https://github.com/microsoft/BackwardCompatibilityML/issues>`_ We look forward to hearing from you!

