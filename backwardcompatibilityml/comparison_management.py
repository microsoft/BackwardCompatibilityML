# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import threading
import io
import numpy as np
import mlflow
from flask import send_file
from PIL import Image
from queue import Queue
from backwardcompatibilityml.helpers import training
from backwardcompatibilityml.metrics import model_accuracy


class ComparisonManager(object):
    """
    The ComparisonManager class is used to manage an experiment that performs
    model comparison.
    """

    def __init__(self, folder_name, h1, h2, dataset,
                 performance_metric=model_accuracy,
                 get_instance_image_by_id=None,
                 get_instance_metadata=None,
                 device="cpu",
                 use_ml_flow=False,
                 ml_flow_run_name="model_comparison"):
        self.folder_name = folder_name
        self.h1 = h1
        self.h2 = h2
        self.dataset = dataset
        self.performance_metric = performance_metric
        self.get_instance_image_by_id = get_instance_image_by_id
        self.get_instance_metadata = get_instance_metadata
        self.device = device
        self.use_ml_flow = use_ml_flow
        self.ml_flow_run_name = ml_flow_run_name
        self.last_sweep_status = 0.0
        self.percent_complete_queue = Queue()
        self.sweep_thread = None

    def get_instance_image(self, instance_id):
        get_instance_image_by_id = self.get_instance_image_by_id
        if get_instance_image_by_id is not None:
            return get_instance_image_by_id(instance_id)

        # Generate a blank white PNG image as the default
        data = np.uint8(np.zeros((30, 30)) + 255)
        image = Image.fromarray(data)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return send_file(img_bytes, mimetype="image/png")
