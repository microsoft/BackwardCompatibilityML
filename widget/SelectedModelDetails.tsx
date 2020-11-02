// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type SelectedModelDetailsProps = {
  btc: number,
  bec: number,
  h1Performance: number,
  h2Performance: number,
  performanceMetric: string,
  lambdaC: number
}

class SelectedModelDetails extends Component<SelectedModelDetailsProps, null> {
  constructor(props) {
    super(props);
  }

  render() {
    let deltaPerf = this.props.h1Performance != 0 ? (this.props.h2Performance - this.props.h1Performance) / this.props.h1Performance * 100 : 0;

    return (
      <table className="model-details">
        <tr className="model-details-row">
          <td><span className="model-details-info-type">BTC</span>&nbsp;<span className="model-details-info-data">{this.props.btc.toFixed(3)}</span></td>
          <td><span className="model-details-info-type">h1 {this.props.performanceMetric}</span>&nbsp;<span className="model-details-info-data">{this.props.h1Performance.toFixed(3)}</span></td>
          <td><span className="model-details-info-type">Δ{this.props.performanceMetric}</span>&nbsp;<span className="model-details-info-data">{(deltaPerf >= 0 ? "+" : "-") + deltaPerf.toFixed(3)}%</span></td>
        </tr>
        <tr className="model-details-row">
          <td><span className="model-details-info-type">BEC</span>&nbsp;<span className="model-details-info-data">{this.props.bec.toFixed(3)}</span></td>
          <td><span className="model-details-info-type">h2 {this.props.performanceMetric}</span>&nbsp;<span className="model-details-info-data">{this.props.h2Performance.toFixed(3)}</span></td>
          <td><span className="model-details-info-type">λ<sub>c</sub></span>&nbsp;<span className="model-details-info-data">{this.props.lambdaC.toFixed(2)}</span></td>
        </tr>
      </table>
    )
  }
}
export default SelectedModelDetails;
