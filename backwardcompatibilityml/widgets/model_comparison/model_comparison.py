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
from backwardcompatibilityml.comparison_management import ComparisonManager
from backwardcompatibilityml.metrics import model_accuracy
from rai_core_flask.flask_helper import FlaskHelper
from rai_core_flask.environments import (
    AzureNBEnvironment,
    DatabricksEnvironment,
    LocalIPythonEnvironment)
from backwardcompatibilityml.helpers import http
from backwardcompatibilityml.helpers import comparison


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


def render_widget_html(api_service_environment, data):
    """
    Renders the HTML for the compatibility analysis widget.

    Args:
        api_service_environment: A dictionary of the environment
            type, the base URL, and the port for the Flask service.
    Returns:
        The widget HTML rendered as a string.
    """

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

    app_html_template = Template(app_html_template_string)
    return app_html_template.render(
        widget_css=widget_css,
        widget_javascript=widget_javascript,
        api_service_environment=json.dumps(api_service_environment),
        data=json.dumps(data))


def default_get_instance_metadata(instance_id):
    return str(instance_id)


def init_app_routes(app, comparison_manager):
    """
    Defines the API for the Flask app.

    Args:
        app: The Flask app to use for the API.
        comparison_manager: The ComparisonManager that will be controlled by the API.
    """

    @app.route("/api/v1/instance_data/<int:instance_id>")
    @http.no_cache
    def get_instance_data(instance_id):
        return comparison_manager.get_instance_image(instance_id)


class ModelComparison(object):
    """
    Model Comparison widget
    ...
    """

    def __init__(self, h1, h2, dataset,
                 performance_metric=model_accuracy,
                 port=None,
                 get_instance_image_by_id=None,
                 get_instance_metadata=None,
                 device="cpu",
                 use_ml_flow=False,
                 ml_flow_run_name="model_comparison"):

        if get_instance_metadata is None:
            get_instance_metadata = default_get_instance_metadata

        self.comparison_manager = ComparisonManager(
            h1,
            h2,
            dataset,
            performance_metric=performance_metric,
            get_instance_image_by_id=get_instance_image_by_id,
            get_instance_metadata=get_instance_metadata,
            device=device,
            use_ml_flow=use_ml_flow,
            ml_flow_run_name=ml_flow_run_name)

        self.flask_service = FlaskHelper(ip="0.0.0.0", port=port)
        app_has_routes = False
        for route in FlaskHelper.app.url_map.iter_rules():
            if route.endpoint == 'instance_data':
                app_has_routes = True
                break
        if app_has_routes:
            FlaskHelper.app.logger.info("Routes already defined. Skipping route initialization.")
        else:
            FlaskHelper.app.logger.info("Initializing routes")
            init_app_routes(FlaskHelper.app, self.comparison_manager)
        api_service_environment = build_environment_params(self.flask_service.env)
        api_service_environment["port"] = self.flask_service.port
        comparison_data = comparison.compare_models(
            h1, h2, dataset,
            performance_metric=performance_metric,
            get_instance_metadata=get_instance_metadata,
            device=device)

        html_string = render_widget_html(api_service_environment, comparison_data)
        self.html_widget = HTML(html_string)
        display(self.html_widget)
