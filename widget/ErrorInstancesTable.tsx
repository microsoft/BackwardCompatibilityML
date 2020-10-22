// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import {
  DetailsList,
  DetailsListLayoutMode,
  Selection,
  SelectionMode,
  IColumn
} from "office-ui-fabric-react/lib/DetailsList";
import { Fabric } from "office-ui-fabric-react/lib/Fabric";


type ErrorInstancesTableState = {
  selecedDataPoint: any
}

type ErrorInstancesTableProps = {
  selectedDataPoint: any
}

class ErrorInstancesTable extends Component<ErrorInstancesTableProps, ErrorInstancesTableState> {
  constructor(props) {
    super(props);
  }

  componentWillReceiveProps(nextProps) {
    this.setState({
      selecedDataPoint: nextProps.selecedDataPoint
    });
  }

  render() {

    if (this.props.selectedDataPoint == null) {
      return (
        <React.Fragment />
      );
    }
    var columns = [
      { key: 'instanceId', name: 'Instance ID', fieldName: 'instance_id', minWidth: 100, maxWidth: 100, isResizable: false },
      { key: 'h1Prediction', name: 'h1 Prediction', fieldName: 'h1_prediction', minWidth: 100, maxWidth: 100, isResizable: false },
      { key: 'h2Prediction', name: 'h2 Prediction', fieldName: 'h2_prediction', minWidth: 100, maxWidth: 100, isResizable: false },
      { key: 'groundTruth', name: 'Ground Truth', fieldName: 'ground_truth', minWidth: 100, maxWidth: 100, isResizable: false },
    ];

    return (
      <Fabric>
        <DetailsList
          selectionMode={SelectionMode.none}
          items={this.props.selectedDataPoint.error_instances}
          columns={columns}
        />
      </Fabric>
    );
  }
}
export default ErrorInstancesTable;
