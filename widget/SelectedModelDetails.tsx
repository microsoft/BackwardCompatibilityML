// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type SelectedModelDetailsProps = {
  btc: number,
  bec: number,
  performance: number,
  lambda_c: number
}

class SelectedModelDetails extends Component<SelectedModelDetailsProps, null> {
  constructor(props) {
    super(props);
  }

  render() {
    var h2ModelName = "MyModel";

    return (
      <div className="model-details">
        <div>Selected Model Details</div>
        <div>Model h2: {h2ModelName}</div>
        <div>BTC score: {this.props.btc.toFixed(3)}</div>
        <div>BEC score: {this.props.bec.toFixed(3)}</div>
        <div>Performance: {this.props.performance.toFixed(3)}</div>
        <div>lambda_c: {this.props.lambda_c.toFixed(2)}</div>
      </div>
    );
  }
}
export default SelectedModelDetails;
