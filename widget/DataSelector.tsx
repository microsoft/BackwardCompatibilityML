// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type DataSelectorState = {
  training: boolean,
  testing: boolean,
  newError: boolean,
  strictImitation: boolean,
  showDatasetDropdown: boolean,
  showDissonanceDropdown: boolean,
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
      strictImitation: true,
      showDatasetDropdown: false,
      showDissonanceDropdown: false
    };

    this.selectTraining = this.selectTraining.bind(this);
    this.selectTesting = this.selectTesting.bind(this);
    this.selectNewError = this.selectNewError.bind(this);
    this.selectStrictImitation = this.selectStrictImitation.bind(this);
    this.selectTrainingAndTesting = this.selectTrainingAndTesting.bind(this);
    this.toggleDatasetDropdown = this.toggleDatasetDropdown.bind(this);
    this.selectNewErrorAndStrictImitation = this.selectNewErrorAndStrictImitation.bind(this);
    this.toggleDissonanceDropdown = this.toggleDissonanceDropdown.bind(this);
  }

  toggleDatasetDropdown(evt) {
    this.setState({
      showDatasetDropdown: !this.state.showDatasetDropdown
    });
  }

  toggleDissonanceDropdown(evt) {
    this.setState({
      showDissonanceDropdown: !this.state.showDissonanceDropdown
    });
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

  selectTrainingAndTesting(evt) {
    if (!this.state.training && !this.state.testing) {
      this.setState({
        training: true,
        testing: true
      });
      this.props.toggleTraining();
      this.props.toggleTesting();
    } else if (!this.state.training && this.state.testing) {
      this.setState({
        training: true
      });
      this.props.toggleTraining();
    } else if (this.state.training && !this.state.testing) {
      this.setState({
        testing: true
      });
      this.props.toggleTesting();
    } else if (this.state.training && this.state.testing) {
      this.setState({
        training: false,
        testing: false
      });
      this.props.toggleTraining();
      this.props.toggleTesting();
    }
  }

  selectNewErrorAndStrictImitation(evt) {
    if (!this.state.newError && !this.state.strictImitation) {
      this.setState({
        newError: true,
        strictImitation: true
      });
      this.props.toggleNewError();
      this.props.toggleStrictImitation();
    } else if (!this.state.newError && this.state.strictImitation) {
      this.setState({
        newError: true
      });
      this.props.toggleNewError();
    } else if (this.state.newError && !this.state.strictImitation) {
      this.setState({
        strictImitation: true
      });
      this.props.toggleStrictImitation();
    } else if (this.state.newError && this.state.strictImitation) {
      this.setState({
        newError: false,
        strictImitation: false
      });
      this.props.toggleNewError();
      this.props.toggleStrictImitation();
    }
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
    let getDatasetDropdownClasses = () => {
      if (this.state.showDatasetDropdown) {
        return "dropdown-check-list visible";
      } else {
        return "dropdown-check-list";
      }
    }

    let getDissonanceDropdownClasses = () => {
      if (this.state.showDissonanceDropdown) {
        return "dropdown-check-list visible";
      } else {
        return "dropdown-check-list";
      }
    }
    return (
      <div className="data-selector">
        <div className="control-group">
          <div className={getDatasetDropdownClasses()}>
            <span className="anchor" onClick={this.toggleDatasetDropdown}>Dataset</span>
            <ul className="items">
              <li><input type="checkbox" checked={this.state.training && this.state.testing} onClick={this.selectTrainingAndTesting} />(Select All)</li>
              <li><input type="checkbox" checked={this.state.training} onClick={this.selectTraining} />Training</li>
              <li><input type="checkbox" checked={this.state.testing} onClick={this.selectTesting} />Testing</li>
            </ul>
          </div>
        </div>
        <div className="control-group">
          <div className={getDissonanceDropdownClasses()}>
            <span className="anchor" onClick={this.toggleDissonanceDropdown}>Dissonance</span>
            <ul className="items">
              <li><input type="checkbox" checked={this.state.newError && this.state.strictImitation} onClick={this.selectNewErrorAndStrictImitation} />(Select All)</li>
              <li><input type="checkbox" checked={this.state.newError} onClick={this.selectNewError} />New Error</li>
              <li><input type="checkbox" checked={this.state.strictImitation} onClick={this.selectStrictImitation} />Strict Imitation</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }
}
export default DataSelector;
