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
        <div className="model-details-row">
          <div className="model-details-info"><div className="model-details-info-data">{h2ModelName}</div><div className="model-details-info-type">(h2)</div></div>
          <div className="model-details-info"><div className="model-details-info-data">{this.props.lambda_c.toFixed(2)}</div><div className="model-details-info-type">(lambda_c)</div></div>
          <div className="model-details-info"><div className="model-details-info-data">{this.props.performance.toFixed(3)}</div><div className="model-details-info-type">(Performance)</div></div>
          <div className="model-details-info"><div className="model-details-info-data">{this.props.btc.toFixed(3)}</div><div className="model-details-info-type">(BTC)</div></div>
          <div className="model-details-info"><div className="model-details-info-data">{this.props.bec.toFixed(3)}</div><div className="model-details-info-type">(BEC)</div></div>
        </div>
      </div>
    );
  }
}
export default SelectedModelDetails;
