# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
import numpy as np
from flask import send_file
from PIL import Image


class ComparisonManager(object):
    """
    The ComparisonManager class is used to field any REST requests by the ModelComparison
    widget UI components from within the Jupyter notebook.

    Args:
        training_set: The list of training samples as (batch_ids, input, target).
        dataset: The list of dataset samples as (batch_ids, input, target).
        get_instance_image_by_id: A function that returns an image representation of the data corresponding to the instance id, in PNG format. It should be a function of the form:
                get_instance_image_by_id(instance_id)
                    instance_id: An integer instance id

            And should return a PNG image.
    """

    def __init__(self, dataset,
                 get_instance_image_by_id=None):
        self.dataset = dataset
        self.get_instance_image_by_id = get_instance_image_by_id

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
