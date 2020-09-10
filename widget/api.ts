// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import axios from 'axios';


var apiServiceEnvironment = window["API_SERVICE_ENVIRONMENT"]
var apiBaseUrl = ""

if (apiServiceEnvironment.environment_type == "local") {
  apiBaseUrl = `${window.location.protocol}//${window.location.hostname}:${apiServiceEnvironment.port}`;
} else if (apiServiceEnvironment.environment_type == "azureml") {
  apiBaseUrl = apiServiceEnvironment.base_url;
} else if (apiServiceEnvironment.environment_type == "databricks") {
  apiBaseUrl = apiServiceEnvironment.base_url;
} else {
  apiBaseUrl = `${window.location.protocol}//${window.location.hostname}:${apiServiceEnvironment.port}`;
}


function makeGetCall(endpoint: string) {
  return axios.get<any>(`${apiBaseUrl}/${endpoint}`);
}

function makePostCall(endpoint: string, payload: any) {
  return axios.post(`${apiBaseUrl}/${endpoint}`, payload);
}

export {
  makeGetCall,
  makePostCall
};
