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
import { DefaultButton } from 'office-ui-fabric-react';
import { Fabric } from "office-ui-fabric-react/lib/Fabric";
import { apiBaseUrl } from "./api.ts";


type ErrorInstancesTableState = {
  selecedDataPoint: any,
  page: number
}

type ErrorInstancesTableProps = {
  selectedDataPoint: any,
  pageSize?: number,
  filterInstances: number[]
}

class ErrorInstancesTable extends Component<ErrorInstancesTableProps, ErrorInstancesTableState> {

  public static defaultProps = {
      pageSize: 5
  };

  constructor(props) {
    super(props);
    this.state = {
      selecedDataPoint: null,
      page: 0
    }
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
      {
        key: 'instanceImage',
        name: '',
        minWidth: 50,
        maxWidth: 50,
        isResizable: false,
        onRender: (instance) => {
          return (<img src={`${apiBaseUrl}/api/v1/instance_data/${instance.instance_id}`} />);
        }
      },
      {
        key: 'instanceData',
        name: 'Instance',
        fieldName: 'metadata',
        minWidth: 100,
        maxWidth: 100,
        isResizable: false
      },
      { key: 'h1Prediction', name: 'h1 Prediction', fieldName: 'h1_prediction', minWidth: 100, maxWidth: 100, isResizable: false },
      { key: 'h2Prediction', name: 'h2 Prediction', fieldName: 'h2_prediction', minWidth: 100, maxWidth: 100, isResizable: false },
      { key: 'groundTruth', name: 'Ground Truth', fieldName: 'ground_truth', minWidth: 100, maxWidth: 100, isResizable: false },
    ];

    var errorInstances = this.props.selectedDataPoint.error_instances;
    if (this.props.filterInstances != null) {
      errorInstances = this.props.selectedDataPoint.error_instances.filter(errorInstance => {
        return (this.props.filterInstances.indexOf(errorInstance.instance_id) != -1)
      });
    }

    var items = [];
    for(var i=(this.state.page * this.props.pageSize);
        i < Math.min((this.state.page * this.props.pageSize) + this.props.pageSize,
                     errorInstances.length);
        i++) {
      items.push(errorInstances[i]);
    }

    return (
      <Fabric>
        <DetailsList
          selectionMode={SelectionMode.none}
          items={items}
          columns={columns}
        />
        <DefaultButton
          text="Previous"
          onClick={() => {
            this.setState({
              page: Math.max(0, this.state.page - 1)
            })
          }}
        />
        <DefaultButton
          text="Next"
          onClick={() => {
            this.setState({
              page: Math.min(this.state.page + 1, Math.ceil(this.props.selectedDataPoint.error_instances.length / this.props.pageSize))
            })
          }}
        />
      </Fabric>
    );
  }
}
export default ErrorInstancesTable;
