// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type SweepManagerState = {
  sweepStatus: any
}

type SweepManagerProps = {
  sweepStatus: any,
  getSweepStatus: () => void,
  startSweep: () => void,
  getTrainingAndTestingData: () => void
}

class SweepManager extends Component<SweepManagerProps, SweepManagerState> {
  constructor(props) {
    super(props);

    this.state = {
      sweepStatus: this.props.sweepStatus
    };

    this.pollSweepStatus = this.pollSweepStatus.bind(this);
    this.startSweep = this.startSweep.bind(this);
  }

  timeoutVar: any = null

  componentWillReceiveProps(nextProps) {
    this.setState({
      sweepStatus: nextProps.sweepStatus
    });
  }

  componentWillUnmount() {
    if (this.timeoutVar != null) {
      clearTimeout(this.timeoutVar);
    }
  }

  pollSweepStatus() {
    this.props.getSweepStatus();
    if (this.timeoutVar != null) {
      clearTimeout(this.timeoutVar);
      this.timeoutVar = null;
    }
    this.timeoutVar = setTimeout(this.pollSweepStatus, 5000);
  }

  startSweep(evt) {
    this.props.startSweep();
    this.pollSweepStatus();
  }

  render() {

    if (this.state.sweepStatus == null || !this.state.sweepStatus.running) {
      if (this.timeoutVar != null && this.state.sweepStatus.percent_complete == 1.0) {
        clearTimeout(this.timeoutVar);
        this.timeoutVar = null;
        this.props.getTrainingAndTestingData();
      }

      return (
        <div className="table">
          <a href="#" onClick={this.startSweep}>Start Sweep</a>
        </div>
      );
    }

    return (
      <div className="table">
        Sweep in progress
        <div>
         {Math.floor(this.state.sweepStatus.percent_complete * 100)} % complete
        </div>
      </div>
    );
  }
}
export default SweepManager;
