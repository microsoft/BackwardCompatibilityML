// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type RawValuesState = {
  data: any
}

type RawValuesProps = {
  data: any
}

class RawValues extends Component<RawValuesProps, RawValuesState> {
  constructor(props) {
    super(props);

    this.state = {
      data: this.props.data
    };
  }

  componentWillReceiveProps(nextProps) {
    this.setState({
      data: nextProps.data
    });
  }

  render() {
    return (
      <div className="table">
        Raw Values Table goes here
      </div>
    );
  }
}
export default RawValues;
