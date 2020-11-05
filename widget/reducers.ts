// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

const rawInitialState = {
   data: null,
   sweepStatus: null,
   selectedDataPoint: null,
   filterInstances: null,
   training: true,
   testing: true,
   newError: true,
   strictImitation: true,
   error: null,
   loading: false
}

const initialState = window["WIDGET_STATE"] || rawInitialState;

function rootReducer(state = initialState, action) {
    switch(action.type) {
        case "TOGGLE_TRAINING":
          return Object.assign({}, state, {training: !state.training});

        case "TOGGLE_TESTING":
          return Object.assign({}, state, {testing: !state.testing});

        case "TOGGLE_NEW_ERROR":
          return Object.assign({}, state, {newError: !state.newError});

        case "TOGGLE_STRICT_IMITATION":
          return Object.assign({}, state, {strictImitation: !state.strictImitation});

        case "SELECT_DATA_POINT":
          return Object.assign({}, state, {selectedDataPoint: action.dataPoint});

        case "REQUEST_TRAINING_AND_TESTING_DATA":
          return Object.assign({}, state, {loading: true});

        case "REQUEST_TRAINING_AND_TESTING_DATA_SUCCEEDED":
          return Object.assign({}, state, {data: action.data, loading: false, error: null});

        case "REQUEST_TRAINING_AND_TESTING_DATA_FAILED":
          console.log("Error: ", action.error);
          return Object.assign({}, state, {error: "Failed to load training and testing data", loading: false});

        case "REQUEST_MODEL_EVALUATION_DATA":
          return Object.assign({}, state, {loading: true});

        case "REQUEST_MODEL_EVALUATION_DATA_SUCCEEDED":
          return Object.assign({}, state, {selectedDataPoint: action.evaluationData, loading: false});

        case "REQUEST_MODEL_EVALUATION_DATA_FAILED":
          console.log("Error: ", action.error);
          return Object.assign({}, state, {error: "Failed to load evaluation data", loading: false});

        case "REQUEST_SWEEP_STATUS_SUCCEEDED":
          return Object.assign({}, state, {sweepStatus: action.sweepStatus});

        case "REQUEST_SWEEP_STATUS_FAILED":
          console.log("Error: ", action.error);
          return Object.assign({}, state, {error: "Failed to get sweep status"});

        case "FILTER_BY_INSTANCE_IDS":
          return Object.assign({}, state, {filterInstances: action.filterInstances});

        default:
            return state
    }
}

export default rootReducer;
