# Introduction

Updates that may improve an AI system’s accuracy can also introduce new
and unanticipated errors that damage user trust. Updates that introduce
new errors can also break trust between software components and machine
learning models, as these errors are propagated and compounded
throughout larger integrated AI systems. The Backward Compatibility ML
library is an open-source project for evaluating AI system updates in a
new way for increasing system reliability and human trust in AI
predictions for actions.

The Backward Compatibility ML project has two components:

- **A series of loss functions** in which users can vary the weight
  assigned to the dissonance factor and explore performance/capability
  tradeoffs during machine learning optimization.

- **Visualization widgets** that help users examine metrics and error
  data in detail. They provide a view of error intersections between
  models and incompatibility distribution across classes.

# Getting Started

1. Setup a Python virtual environment or Conda environment and activate it.
2. From within the root folder of this project do `pip install -r requirements.txt`
3. From within the root folder do `npm install`
4. From within the root folder of this project do `npx webpack && pip install -e .`
5. You should now be able to import the `backwardcompatibilityml` module and use it.

# Examples

Start your Jupyter Notebooks server and load in the example notebook under the `examples` folder
to see how the `backwardcompatibilityml` module is used.

To demo the widget, open the notebook `compatibility-analysis.ipynb`.

# Tests

To run tests, make sure that you are in the project root folder and do:

1. `pip install -r dev-requirements.txt`
2. `pytest tests/`

# Contributing

Check [CONTRIBUTING](CONTRIBUTING.md) page.

# Microsoft Open Source Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# License

This project is licensed under the terms of the MIT license. See [LICENSE.txt](LICENSE.txt) for additional details.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
