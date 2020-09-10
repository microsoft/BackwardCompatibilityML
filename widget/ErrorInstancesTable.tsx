// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type ErrorInstancesTableState = {
  data: any
}

type ErrorInstancesTableProps = {
  data: any
}

class ErrorInstancesTable extends Component<ErrorInstancesTableProps, ErrorInstancesTableState> {
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
        Error Instances Table goes here
      </div>
    );
  }
}
export default ErrorInstancesTable;
