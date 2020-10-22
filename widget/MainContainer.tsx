// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React from "react";
import ReactDOM from "react-dom";
import PerformanceCompatibility from "./PerformanceCompatibility.tsx";
import Legend from "./Legend.tsx";
import IntersectionBetweenModelErrors from "./IntersectionBetweenModelErrors.tsx";
import IncompatiblePointDistribution from "./IncompatiblePointDistribution.tsx";
import RawValues from "./RawValues.tsx";
import ErrorInstancesTable from "./ErrorInstancesTable.tsx";
import DataSelector from "./DataSelector.tsx"
import SweepManager from "./SweepManager.tsx";
import SelectedModelDetails from "./SelectedModelDetails.tsx";
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
  startSweep
} from './actions.ts';


function Container({
  data,
  sweepStatus,
  selectedDataPoint,
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
  startSweep}) {

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
              performanceFunctionLabel={data.performance_metric}
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
              performanceFunctionLabel={data.performance_metric}
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
          {(selectedDataPoint != null)? <SelectedModelDetails btc={selectedDataPoint.btc} bec={selectedDataPoint.bec} performance={selectedDataPoint.h2_performance} lambda_c={selectedDataPoint.lambda_c} />: null}
          <div className="row">
            <IntersectionBetweenModelErrors selectedDataPoint={selectedDataPoint} />
            <IncompatiblePointDistribution selectedDataPoint={selectedDataPoint} />
          </div>
          <div className="row">
            <RawValues data={data.data} />
          </div>
          <div className="row">
            <ErrorInstancesTable data={data.data} />
          </div>
      </div>
    );
}

function mapStateToProps (state) {
  return {
    data: state.data,
    sweepStatus: state.sweepStatus,
    selectedDataPoint: state.selectedDataPoint,
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
    startSweep: startSweep
  }, dispatch);
 }

const MainContainer = connect(mapStateToProps, mapDispatchToProps)(Container)

export default MainContainer;
