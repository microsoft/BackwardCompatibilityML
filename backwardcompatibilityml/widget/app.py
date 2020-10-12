# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

from flask import Flask
from .compatibility_analysis import init_dev_server

app = Flask(__name__)
init_dev_server(app)