# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

from flask import Flask
from flask_cors import CORS
from .compatibility_analysis import init_dev_server

app = Flask(__name__)
CORS(app)
init_dev_server(app)
