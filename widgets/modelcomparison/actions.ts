// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import { makeGetCall, makePostCall } from "../common/api.ts";


function toggleTraining() {
  return {
    type: "TOGGLE_TRAINING"
  }
}

function toggleTesting() {
  return {
    type: "TOGGLE_TESTING"
  }
}

function toggleNewError() {
  return {
    type: "TOGGLE_NEW_ERROR"
  }
}

function toggleStrictImitation() {
  return {
    type: "TOGGLE_STRICT_IMITATION"
  }
}

function selectDataPoint(dataPoint) {
  return {
    type: "SELECT_DATA_POINT",
    dataPoint: dataPoint
  }
}

function setSelectedClass(selectedClass) {
  return {
    type: "SET_SELECTED_CLASS",
    selectedClass: selectedClass
  }
}

function setSelectedRegion(selectedRegion) {
  return {
    type: "SET_SELECTED_REGION",
    selectedRegion: selectedRegion
  }
}

function requestTrainingAndTestingData() {
  return {
    type: "REQUEST_TRAINING_AND_TESTING_DATA"
  }
}

function requestTrainingAndTestingDataSucceeded(data) {
  return {
    type: "REQUEST_TRAINING_AND_TESTING_DATA_SUCCEEDED",
    data: data
  }
}

function requestTrainingAndTestingDataFailed(error) {
  return {
    type: "REQUEST_TRAINING_AND_TESTING_DATA_FAILED",
    error: error
  }
}

function getTrainingAndTestingData() {
    return function(dispatch) {
        dispatch(requestTrainingAndTestingData);
        makeGetCall("api/v1/sweep_summary")
          .then(response => {dispatch(requestTrainingAndTestingDataSucceeded(response.data))})
          .catch(error => dispatch(requestTrainingAndTestingDataFailed(error)));
    }
}

function requestModelEvaluationData() {
  return {
    type: "REQUEST_MODEL_EVALUATION_DATA"
  }
}

function requestModelEvaluationDataSucceeded(evaluationData) {
  return {
    type: "REQUEST_MODEL_EVALUATION_DATA_SUCCEEDED",
    evaluationData: evaluationData
  }
}

function requestModelEvaluationDataFailed(error) {
  return {
    type: "REQUEST_MODELEVALUATION_DATA_FAILED",
    error: error
  }
}

function getModelEvaluationData(evaluationId) {
    return function(dispatch) {
        dispatch(requestModelEvaluationData);
        makeGetCall(`api/v1/evaluation_data/${evaluationId}`)
          .then(response => {dispatch(requestModelEvaluationDataSucceeded(response.data))})
          .catch(error => dispatch(requestModelEvaluationDataFailed(error)));
    }
}

function requestSweepStatusSucceeded(sweepStatus) {
  return {
    type: "REQUEST_SWEEP_STATUS_SUCCEEDED",
    sweepStatus: sweepStatus
  }
}

function requestSweepStatusFailed(error) {
  return {
    type: "REQUEST_SWEEP_STATUS_FAILED",
    error: error
  }
}

function getSweepStatus() {
    return function(dispatch) {
        makeGetCall("api/v1/sweep_status")
          .then(response => {dispatch(requestSweepStatusSucceeded(response.data))})
          .catch(error => dispatch(requestSweepStatusFailed(error)));
    }
}

function startSweep() {
    return function(dispatch) {
        makePostCall("api/v1/start_sweep", {});
    }
}

function filterByInstanceIds(filterInstances) {
  return {
    type: "FILTER_BY_INSTANCE_IDS",
    filterInstances: filterInstances
  }
}


export {
  toggleTraining,
  toggleTesting,
  toggleNewError,
  toggleStrictImitation,
  selectDataPoint,
  setSelectedClass,
  setSelectedRegion,
  getTrainingAndTestingData,
  getModelEvaluationData,
  getSweepStatus,
  startSweep,
  filterByInstanceIds
};
