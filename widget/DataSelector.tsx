// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type DataSelectorState = {
  training: boolean,
  testing: boolean,
  newError: boolean,
  strictImitation: boolean
}

type DataSelectorProps = {
  toggleTraining: () => void,
  toggleTesting: () => void,
  toggleNewError: () => void,
  toggleStrictImitation: () => void
}

class DataSelector extends Component<DataSelectorProps, DataSelectorState> {
  constructor(props) {
    super(props);

    this.state = {
      training: true,
      testing: true,
      newError: true,
      strictImitation: true
    };

    this.selectTraining = this.selectTraining.bind(this);
    this.selectTesting = this.selectTesting.bind(this);
    this.selectNewError = this.selectNewError.bind(this);
    this.selectStrictImitation = this.selectStrictImitation.bind(this);
  }

  selectTraining(evt) {
    this.setState({
      training: !this.state.training
    });
    this.props.toggleTraining();
  }

  selectTesting(evt) {
    this.setState({
      testing: !this.state.testing
    });
    this.props.toggleTesting();
  }

  selectNewError(evt) {
    this.setState({
      newError: !this.state.newError
    });
    this.props.toggleNewError();
  }

  selectStrictImitation(evt) {
    this.setState({
      strictImitation: !this.state.strictImitation
    });
    this.props.toggleStrictImitation();
  }

  render() {
    return (
      <div className="data-selector">
        <div className="control-group">
          <div className="control-subgroup">
            <input className="control" type="checkbox" name="training" value="training" checked={this.state.training} onClick={this.selectTraining} />
            <div className="control">Training</div>
          </div>
          <div className="control-subgroup">
            <input className="control" type="checkbox" name="testing" value="testing" checked={this.state.testing} onClick={this.selectTesting} />
            <div className="control">Testing</div>
          </div>
        </div>
        <div className="control-group">
          <div className="control-subgroup">
            <input className="control" type="checkbox" name="training" value="training" checked={this.state.newError} onClick={this.selectNewError} />
            <div className="control">New-Error</div>
          </div>
          <div className="control-subgroup">
            <input className="control" type="checkbox" name="testing" value="testing" checked={this.state.strictImitation} onClick={this.selectStrictImitation} />
            <div className="control">Strict-Imitation</div>
          </div>
        </div>
      </div>
    );
  }
}
export default DataSelector;
