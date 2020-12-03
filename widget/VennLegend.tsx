// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";
import { bisect } from "./optimization.tsx";
import { InfoTooltip } from "./InfoTooltip.tsx"
import { getRegionFill } from "./IntersectionBetweenModelErrors.tsx";
import { DirectionalHint } from 'office-ui-fabric-react/lib/Tooltip';

type VennLegendProps = {
  selectedRegion: any
  setSelectedRegion: Function
}

class VennLegend extends Component<VennLegendProps> {
  render() {
    const progressInfo = "Indicates errors made by the previous model that the newly trained model does not make.​";
    const regressInfo = "Indicates errors made by the newly trained model that the previous model did not make.​";
    const intersectionInfo = "Indicates errors made by both the previous and newly trained models.​";
    const selectedRegion = this.props.selectedRegion;

    function getBlockClass(regionName: string) : string {
      if (selectedRegion == regionName) {
        return "venn-legend-row-block-selected";
      } else {
        return "venn-legend-row-block";
      }
    }

    return (
      <div className="venn-legend-row">
        <div id="progress-container" className={getBlockClass("progress")} onClick={() => this.props.setSelectedRegion("progress")}>
          <div id="progress" className="venn-legend-color-box" style={{ background: getRegionFill("progress", selectedRegion) }} />
          Progress
          <InfoTooltip direction={DirectionalHint.topCenter} message={progressInfo} />
        </div>
        <div id="commonerror-container" className={getBlockClass("intersection")} onClick={() => this.props.setSelectedRegion("intersection")}>
          <div id="commonerror" className="venn-legend-color-box" style={{ background: getRegionFill("intersection", selectedRegion) }} />
          Common
          <InfoTooltip direction={DirectionalHint.topCenter} message={intersectionInfo} />
        </div>
        <div id="regress-container" className={getBlockClass("regress")} onClick={() => this.props.setSelectedRegion("regress")}>
          <div id="regress" className="venn-legend-color-box" style={{ background: getRegionFill("regress", selectedRegion) }} />
          Regress
          <InfoTooltip direction={DirectionalHint.topCenter} message={regressInfo} />
        </div>
      </div>
    )
  }
}

export default VennLegend;