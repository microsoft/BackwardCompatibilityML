// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React from "react";
import ReactDOM from "react-dom";
import PerformanceCompatibility from "./PerformanceCompatibility";
import Legend from "./Legend";
import IntersectionBetweenModelErrors from "./IntersectionBetweenModelErrors";
import IncompatiblePointDistribution from "./IncompatiblePointDistribution";
import ErrorInstancesTable from "./ErrorInstancesTable";
import DataSelector from "./DataSelector"
import SweepManager from "./SweepManager";
import SelectedModelDetails from "./SelectedModelDetails";
import { bindActionCreators } from "redux";
import { connect } from 'react-redux';
import {
  toggleTraining,
  toggleTesting,
  toggleNewError,
  toggleStrictImitation,
  selectDataPoint,
  getTrainingAndTestingData,
  getModelEvaluationData,
  getSweepStatus,
  startSweep,
  filterByInstanceIds
} from './actions';


function Container({
  data,
  sweepStatus,
  selectedDataPoint,
  filterInstances,
  training,
  testing,
  newError,
  strictImitation,
  error,
  loading,
  toggleTraining,
  toggleTesting,
  toggleNewError,
  toggleStrictImitation,
  selectDataPoint,
  getTrainingAndTestingData,
  getModelEvaluationData,
  getSweepStatus,
  startSweep,
  filterByInstanceIds}) {

    if (loading) {
      return (
        <div>Loading...</div>
      );
    } else if (error != null) {
      return (
        <div>Error loading data</div>
      );
    } else if (error == null && data == null) {
      getTrainingAndTestingData();
      return (
        <div>Loading...</div>
      );
    } else if (error == null && data.data.length == 0 && sweepStatus == null) {
      return (
        <div className="container">
            <div className="row">
              <SweepManager
                sweepStatus={sweepStatus}
                getSweepStatus={getSweepStatus}
                startSweep={startSweep}
                getTrainingAndTestingData={getTrainingAndTestingData}
              />
            </div>
        </div>
      );
    }

    return (
      <div className="container">
          <div className="row">
              <SweepManager
                sweepStatus={sweepStatus}
                getSweepStatus={getSweepStatus}
                startSweep={startSweep}
                getTrainingAndTestingData={getTrainingAndTestingData}
              />
          </div>
          <div className="row">
            <DataSelector
              toggleTraining={toggleTraining}
              toggleTesting={toggleTesting}
              toggleNewError={toggleNewError}
              toggleStrictImitation={toggleStrictImitation}
            />
            <Legend
              testing={testing}
              training={training}
              newError={newError}
              strictImitation={strictImitation}
            />
          </div>
          <div className="row">
            <PerformanceCompatibility
              data={data.data}
              h1Performance={data.h1_performance}
              performanceMetric={data.performance_metric}
              training={training}
              testing={testing}
              newError={newError}
              strictImitation={strictImitation}
              compatibilityScoreType="btc"
              selectDataPoint={selectDataPoint}
              getModelEvaluationData={getModelEvaluationData}
              selectedDataPoint={selectedDataPoint}
            />
            <PerformanceCompatibility
              data={data.data}
              h1Performance={data.h1_performance}
              performanceMetric={data.performance_metric}
              training={training}
              testing={testing}
              newError={newError}
              strictImitation={strictImitation}
              compatibilityScoreType="bec"
              selectDataPoint={selectDataPoint}
              getModelEvaluationData={getModelEvaluationData}
              selectedDataPoint={selectedDataPoint}
            />
          </div>
          {(selectedDataPoint != null)? 
            <SelectedModelDetails btc={selectedDataPoint.btc} bec={selectedDataPoint.bec} h1Performance={data.h1_performance} h2Performance={selectedDataPoint.h2_performance} performanceMetric={data.performance_metric} lambdaC={selectedDataPoint.lambda_c} />
            : null}
          <div className="row">
            <IntersectionBetweenModelErrors selectedDataPoint={selectedDataPoint} filterByInstanceIds={filterByInstanceIds}/>
            <IncompatiblePointDistribution selectedDataPoint={selectedDataPoint} filterByInstanceIds={filterByInstanceIds} />
          </div>
          <div className="row">
            <ErrorInstancesTable selectedDataPoint={selectedDataPoint} filterInstances={filterInstances} />
          </div>
      </div>
    );
}

function mapStateToProps (state) {
  return {
    data: state.data,
    sweepStatus: state.sweepStatus,
    selectedDataPoint: state.selectedDataPoint,
    filterInstances: state.filterInstances,
    training: state.training,
    testing: state.testing,
    newError: state.newError,
    strictImitation: state.strictImitation,
    error: state.error,
    loading: state.loading
  };
}

function mapDispatchToProps (dispatch) {
  return bindActionCreators({
    toggleTraining: toggleTraining,
    toggleTesting: toggleTesting,
    toggleNewError: toggleNewError,
    toggleStrictImitation: toggleStrictImitation,
    selectDataPoint: selectDataPoint,
    getTrainingAndTestingData: getTrainingAndTestingData,
    getModelEvaluationData: getModelEvaluationData,
    getSweepStatus: getSweepStatus,
    startSweep: startSweep,
    filterByInstanceIds: filterByInstanceIds
  }, dispatch);
 }

const MainContainer = connect(mapStateToProps, mapDispatchToProps)(Container)

export default MainContainer;
