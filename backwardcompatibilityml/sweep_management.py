# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import threading
from queue import Queue
from backwardcompatibilityml.helpers import training


class SweepManager(object):
    """
    The SweepManager class is used to manage an experiment that performs
    training / updating a model h2, with respect to a reference model h1
    in a way that preserves compatibility between the models. The experiment
    performs a sweep of the parameter space of the regularization parameter
    lambda_c, by performing compatibility trainings for small increments
    in the value of lambda_c for some settable step size.

    The sweep manager can run the sweep experiment either synchronously,
    or within a separate thread. In the latter case, it provides some
    helper functions that allow you to check on the percentage of the
    sweep that is complete.

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
        OptimizerClass: The class to instantiate an optimizer from for training.
        optimizer_kwargs: A dictionary of the keyword arguments to be used to
            instantiate the optimizer.
        NewErrorLossClass: The class of the New Error style loss function to
            be instantiated and used to perform compatibility constrained
            training of our model h2.
        StrictImitationLossClass: The class of the Strict Imitation style loss
            function to be instantiated and used to perform compatibility
            constrained training of our model h2.
        performance_metric: Optional performance metric to be used when evaluating the model.
            If not specified then accuracy is used.
        lambda_c_stepsize: The increments of lambda_c to use as we sweep the parameter
            space between 0.0 and 1.0.
        device: A string with values either "cpu" or "cuda" to indicate the
            device that Pytorch is performing training on. By default this
            value is "cpu". But in case your models reside on the GPU, make sure
            to set this to "cuda". This makes sure that the input and target
            tensors are transferred to the GPU during training.
    """

    def __init__(self, folder_name, number_of_epochs, h1, h2, training_set, test_set,
                 batch_size_train, batch_size_test,
                 OptimizerClass, optimizer_kwargs,
                 NewErrorLossClass, StrictImitationLossClass, lambda_c_stepsize=0.25,
                 new_error_loss_kwargs=None,
                 strict_imitation_loss_kwargs=None,
                 device="cpu"):
        self.folder_name = folder_name
        self.number_of_epochs = number_of_epochs
        self.h1 = h1
        self.h2 = h2
        self.training_set = training_set
        self.test_set = test_set
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.OptimizerClass = OptimizerClass
        self.optimizer_kwargs = optimizer_kwargs
        self.NewErrorLossClass = NewErrorLossClass
        self.StrictImitationLossClass = StrictImitationLossClass
        self.lambda_c_stepsize = lambda_c_stepsize
        self.new_error_loss_kwargs = new_error_loss_kwargs
        self.strict_imitation_loss_kwargs = strict_imitation_loss_kwargs
        self.device = device
        self.last_sweep_status = 0.0
        self.percent_complete_queue = Queue()
        self.sweep_thread = threading.Thread(
            target=training.compatibility_sweep,
            args=(self.folder_name, self.number_of_epochs, self.h1, self.h2,
                  self.training_set, self.test_set,
                  self.batch_size_train, self.batch_size_test,
                  self.OptimizerClass, self.optimizer_kwargs,
                  self.NewErrorLossClass, self.StrictImitationLossClass,),
            kwargs={
                "lambda_c_stepsize": self.lambda_c_stepsize,
                "percent_complete_queue": self.percent_complete_queue,
                "new_error_loss_kwargs": self.new_error_loss_kwargs,
                "strict_imitation_loss_kwargs": self.strict_imitation_loss_kwargs,
                "device": self.device
            })

    def start_sweep(self):
        self.sweep_thread.start()

    def start_sweep_synchronous(self):
        training.compatibility_sweep(
            self.folder_name, self.number_of_epochs, self.h1, self.h2, self.training_set, self.test_set,
            self.batch_size_train, self.batch_size_test,
            self.OptimizerClass, self.optimizer_kwargs,
            self.NewErrorLossClass, self.StrictImitationLossClass,
            lambda_c_stepsize=self.lambda_c_stepsize, percent_complete_queue=self.percent_complete_queue,
            new_error_loss_kwargs=self.new_error_loss_kwargs,
            strict_imitation_loss_kwargs=self.strict_imitation_loss_kwargs,
            device=self.device)

    def get_sweep_status(self):
        if not self.percent_complete_queue.empty():
            while not self.percent_complete_queue.empty():
                self.last_sweep_status = self.percent_complete_queue.get()

        return self.last_sweep_status

    def get_sweep_summary(self):
        sweep_summary = {
            "h1_performance": None,
            "data": []
        }

        if os.path.exists(f"{self.folder_name}/sweep_summary.json"):
            with open(f"{self.folder_name}/sweep_summary.json", "r") as sweep_summary_file:
                loaded_sweep_summary = json.loads(sweep_summary_file.read())
                sweep_summary.update(loaded_sweep_summary)

        return sweep_summary

    def get_evaluation(self, evaluation_id):
        with open(f"{self.folder_name}/{evaluation_id}-evaluation-data.json", "r") as evaluation_data_file:
            evaluation_data = json.loads(evaluation_data_file.read())

        return evaluation_data
