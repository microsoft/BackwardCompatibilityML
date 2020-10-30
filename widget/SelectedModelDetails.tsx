// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type SelectedModelDetailsProps = {
  btc: number,
  bec: number,
  h1_performance: number,
  h2_performance: number,
  lambda_c: number
}

class SelectedModelDetails extends Component<SelectedModelDetailsProps, null> {
  constructor(props) {
    super(props);
  }

  render() {
    let h2ModelName = "Updated Model";
    let deltaPerf = this.props.h1_performance != 0 ? (this.props.h2_performance - this.props.h1_performance) / this.props.h1_performance * 100 : 0;

    return (
      <table className="model-details">
        <tr className="model-details-row">
          <td>&nbsp;</td>
          <td><span className="model-details-info-data">{h2ModelName} (h2)</span></td>
          <td>&nbsp;</td>
        </tr>
        <tr className="model-details-row">
          <td><span className="model-details-info-data">{this.props.btc.toFixed(3)}</span><br/><span className="model-details-info-type">(BTC)</span></td>
          <td><span className="model-details-info-data">{this.props.h1_performance.toFixed(3)}</span><br/><span className="model-details-info-type">(h1 performance)</span></td>
          <td><span className="model-details-info-data">{(deltaPerf >= 0 ? "+" : "-") + deltaPerf.toFixed(3)}%</span><br/><span className="model-details-info-type">(Δperformance)</span></td>
        </tr>
        <tr className="model-details-row">
          <td><span className="model-details-info-data">{this.props.bec.toFixed(3)}</span><br/><span className="model-details-info-type">(BEC)</span></td>
          <td><span className="model-details-info-data">{this.props.h2_performance.toFixed(3)}</span><br/><span className="model-details-info-type">(h2 performance)</span></td>
          <td><span className="model-details-info-data">{this.props.lambda_c.toFixed(2)}</span><br/><span className="model-details-info-type">(lambda_c)</span></td>
        </tr>
      </table>
    )
  }
}
export default SelectedModelDetails;
