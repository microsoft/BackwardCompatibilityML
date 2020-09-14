# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import pkg_resources
from jinja2 import Template
from IPython.display import (
    display,
    HTML
)
import torch.optim as optim
from flask import Response
from backwardcompatibilityml import loss
from backwardcompatibilityml.sweep_management import SweepManager
from rai_core_flask.flask_helper import FlaskHelper
from rai_core_flask.environments import (
    AzureNBEnvironment,
    DatabricksEnvironment,
    LocalIPythonEnvironment)


def build_environment_params(flask_service_env):
    """
    A small helper function to return a dictionary of
    the environment type and the base url of the
    Flask service for the environment type.

    Args:
        flask_service_env: An instance of an environment from
            rai_core_flask.environments.

    Returns:
        A dictionary of the environment type specified as a string,
        and the base url to be used when accessing the Flask
        service for this environment type.
    """

    if isinstance(flask_service_env, LocalIPythonEnvironment):
        return {
            "environment_type": "local",
            "base_url": ""
        }
    elif isinstance(flask_service_env, AzureNBEnvironment):
        return {
            "environment_type": "azureml",
            "base_url": flask_service_env.base_url
        }
    elif isinstance(flask_service_env, DatabricksEnvironment):
        return {
            "environment_type": "databricks",
            "base_url": flask_service_env.base_url
        }
    else:
        return {
            "environment_type": "unknown",
            "base_url": ""
        }


class CompatibilityAnalysis(object):
    """
    The CompatibilityAnalysis class is an interactive widget intended for use
    within a Jupyter Notebook. It provides an interactive UI for the user
    to interact with for:
    
        1. Initiating a sweep of the lambda_c parameter space while performing
           compatibility training / updating of a model h2 with respect to a
           reference model h1.
        2. Checking on the status of the sweep being performed.
        3. Interacting with the data generated during the sweep, once the sweep
           is completed.

    Note that this class may only be instantiated once within the same Notebook
    at this time.

    This class works by instantiating a Flask server listening on a free port
    in the 5000 - 5099 range, or a port explicitly specified by the user.

    It then registers a few REST api endpoints on this Flask server.
    The UI for the widget which is displayed within the Jupyter Notebook,
    interacts with these REST api endpoints over HTTP requests.
    It dynamically loads data and uses it to render visualizations
    within the widget UI.

    Args:
        folder_name: A string value representing the full path of the
            folder wehre the result of the compatibility sweep is to be stored.
        number_of_epochs: The number of training epochs to use on each sweep.
        h1: The reference model being used.
        h2: The new model being traind / updated.
        training_set: The list of training samples as (input, target) pairs.
        test_set: The list of testing samples as (input, target) pairs.
        batch_size_train: An integer representing batch size of the training set.
        batch_size_test: An integer representing the batch size of the test set.
        lambda_c_stepsize: The increments of lambda_c to use as we sweep the parameter
            space between 0.0 and 1.0.
        OptimizerClass: The class to instantiate an optimizer from for training.
        optimizer_kwargs: A dictionary of the keyword arguments to be used to
            instantiate the optimizer.
        NewErrorLossClass: The class of the New Error style loss function to
            be instantiated and used to perform compatibility constrained
            training of our model h2.
        StrictImitationLossClass: The class of the Strict Imitation style loss
            function to be instantiated and used to perform compatibility
            constrained training of our model h2.
        port: An integer value to indicate the port to which the Flask service
            should bind.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    """

    def __init__(self, folder_name, number_of_epochs, h1, h2, training_set, test_set,
                 batch_size_train, batch_size_test, lambda_c_stepsize=0.25,
                 OptimizerClass=None, optimizer_kwargs=None,
                 NewErrorLossClass=None, StrictImitationLossClass=None,
                 port=None, new_error_loss_kwargs=None,
                 strict_imitation_loss_kwargs=None, device="cpu"):
        self.flask_service = FlaskHelper(ip="0.0.0.0", port=port)

        if OptimizerClass is None:
            OptimizerClass = optim.SGD

        if optimizer_kwargs is None:
            optimizer_kwargs = dict()

        if NewErrorLossClass is None:
            NewErrorLossClass = loss.BCNLLLoss

        if StrictImitationLossClass is None:
            StrictImitationLossClass = loss.StrictImitationCrossEntropyLoss

        self.sweep_manager = SweepManager(
            folder_name,
            number_of_epochs,
            h1,
            h2,
            training_set,
            test_set,
            batch_size_train,
            batch_size_test,
            OptimizerClass,
            optimizer_kwargs,
            NewErrorLossClass, StrictImitationLossClass,
            lambda_c_stepsize=lambda_c_stepsize,
            new_error_loss_kwargs=new_error_loss_kwargs,
            strict_imitation_loss_kwargs=strict_imitation_loss_kwargs,
            device=device)

        resource_package = __name__
        javascript_path = '/'.join(('resources', 'widget-build.js'))
        css_path = '/'.join(('resources', 'widget.css'))
        html_template_path = '/'.join(('resources', 'widget.html'))
        widget_javascript = pkg_resources.resource_string(
            resource_package, javascript_path).decode("utf-8")
        widget_css = pkg_resources.resource_string(
            resource_package, css_path).decode("utf-8")
        app_html_template_string = pkg_resources.resource_string(
            resource_package, html_template_path).decode("utf-8")

        api_service_environment = build_environment_params(self.flask_service.env)
        api_service_environment["port"] = self.flask_service.port

        app_html_template = Template(app_html_template_string)
        html_string = app_html_template.render(
            widget_css=widget_css,
            widget_javascript=widget_javascript,
            api_service_environment=json.dumps(api_service_environment),
            data=json.dumps(None))

        self.html_widget = HTML(html_string)
        display(self.html_widget)

        @FlaskHelper.app.route("/api/v1/start_sweep", methods=["POST"])
        def start_sweep():
            self.sweep_manager.start_sweep()
            return {
                "running": self.sweep_manager.sweep_thread.is_alive(),
                "percent_complete": self.sweep_manager.get_sweep_status()
            }

        @FlaskHelper.app.route("/api/v1/sweep_status", methods=["GET"])
        def get_sweep_status():
            return {
                "running": self.sweep_manager.sweep_thread.is_alive(),
                "percent_complete": self.sweep_manager.get_sweep_status()
            }

        @FlaskHelper.app.route("/api/v1/sweep_summary", methods=["GET"])
        def get_data():
            return Response(
                json.dumps(self.sweep_manager.get_sweep_summary()),
                mimetype="application/json")

        @FlaskHelper.app.route("/api/v1/evaluation_data/<int:evaluation_id>")
        def get_evaluation(evaluation_id):
            return self.sweep_manager.get_evaluation(evaluation_id)
