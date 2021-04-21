// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React from "react";
import ReactDOM from "react-dom";
import PerformanceCompatibility from "../common/PerformanceCompatibility.tsx";
import Legend from "../common/Legend.tsx";
import IntersectionBetweenModelErrors from "../common/IntersectionBetweenModelErrors.tsx";
import IncompatiblePointDistribution from "../common/IncompatiblePointDistribution.tsx";
import ErrorInstancesTable from "../common/ErrorInstancesTable.tsx";
import DataSelector from "../common/DataSelector.tsx"
import SweepManager from "../common/SweepManager.tsx";
import SelectedModelDetails from "../common/SelectedModelDetails.tsx";
import { Slider } from '@fluentui/react';
import { bindActionCreators } from "redux";
import { connect } from 'react-redux';
import {
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
} from './actions.ts';


function Container({
  data,
  sweepStatus,
  selectedDataPoint,
  selectedClass,
  selectedRegion,
  setSelectedClass,
  setSelectedRegion,
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
      <div className="main-container">
          <div className="row">
              <SweepManager
                sweepStatus={sweepStatus}
                getSweepStatus={getSweepStatus}
                startSweep={startSweep}
                getTrainingAndTestingData={getTrainingAndTestingData}
              />
          </div>
          <div className="two-column-row">
            <DataSelector
              toggleTraining={toggleTraining}
              toggleTesting={toggleTesting}
              toggleNewError={toggleNewError}
              toggleStrictImitation={toggleStrictImitation}
            />
            <Slider
              label="λ value"
              ranged
              min={0}
              max={1}
              defaultValue={1}
              defaultLowerValue={0}
              step={0.1}
            />
          </div>
          <div className="row">
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
            <IntersectionBetweenModelErrors selectedDataPoint={selectedDataPoint} setSelectedRegion={setSelectedRegion} selectedRegion={selectedRegion} filterByInstanceIds={filterByInstanceIds}/>
            <IncompatiblePointDistribution selectedDataPoint={selectedDataPoint} setSelectedClass={setSelectedClass} selectedClass={selectedClass} filterByInstanceIds={filterByInstanceIds} />
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
    selectedClass: state.selectedClass,
    selectedRegion: state.selectedRegion,
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
    setSelectedClass: setSelectedClass,
    setSelectedRegion: setSelectedRegion,
    getTrainingAndTestingData: getTrainingAndTestingData,
    getModelEvaluationData: getModelEvaluationData,
    getSweepStatus: getSweepStatus,
    startSweep: startSweep,
    filterByInstanceIds: filterByInstanceIds
  }, dispatch);
 }

const MainContainer = connect(mapStateToProps, mapDispatchToProps)(Container)

export default MainContainer;
