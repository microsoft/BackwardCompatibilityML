// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";
import { bisect } from "./optimization.tsx";
import { InfoTooltip } from "./InfoTooltip.tsx"
import { DirectionalHint } from 'office-ui-fabric-react/lib/Tooltip';

function calculateCircleRadiiAndDistance(a, b, ab, datasetSize) {
  let aProportion = a / datasetSize;
  let bProportion = b / datasetSize;
  let abProportion = ab / datasetSize;

  // We use the following reasoning:
  // aProportion = Area(circleA) / totalArea
  // bProportion = Area(circleB) / totalArea
  // We want the largest circle to have a radius of 50.
  // So we select the larget of the two circular regions and set
  // its area to be Pi * 50 * 50.
  // Thus if aProportion is the larger proportion, we have
  // that Ra must be 50 and aProportion = (Pi * 50 * 50) / totalArea
  // Thus totalArea = (Pi * 50 * 50) / aProportion.
  // Similarly if bProportion is the larger proportion:
  // We have that totalArea = (Pi * 50 * 50) / bProportion.
  let Ra;
  let Rb;
  let Aab;
  let totalArea = 1;
  if (a >= b) {
    totalArea = (50 * 50 * 3.14) / aProportion;
    Ra = 50;
    Rb = Math.sqrt(bProportion * totalArea / 3.14);
    Aab = abProportion * totalArea;
  } else {
    totalArea = (50 * 50 * 3.14) / bProportion;
    Rb = 50;
    Ra = Math.sqrt(aProportion * totalArea / 3.14);
    Aab = abProportion * totalArea;
  }

  // This function calcuates the area of overlap
  // of two circles of radii r1 and r2 whose
  // centers are separated by a distance d.
  function areaOverlap(r1, r2, dist) {
    var r = Math.min(r1, r2);
    var R = Math.max(r1, r2);
    if (dist == 0) {
      return (3.14 * Math.pow(r, 2));
    } else if (dist >= (R + r)) {
      return 0;
    }

    var sectorAreas = Math.pow(r, 2) * Math.acos((Math.pow(dist, 2) +
      Math.pow(r, 2) - Math.pow(R, 2)) / (2 * dist * r)) +
      Math.pow(R, 2) * Math.acos((Math.pow(dist, 2) + Math.pow(R, 2) - Math.pow(r, 2)) / (2 * dist * R));

    var triangleAreas = 1/2 * Math.sqrt((-dist + r + R) * (dist + r - R) * (dist - r + R) * (dist + r + R));
    var overlapArea = sectorAreas - triangleAreas;

    return overlapArea;
  }

  // In order to decide the distance d that the circles
  // Need to be from each other, we use the bisection method
  // to search for the point at which the difference between
  // The area of overlap between the circles and the actual
  // are of the intersection is minimal.
  function aIntersection(dist) {
    return (areaOverlap(Ra, Rb, dist) - Aab);
  }

  var r = Math.min(Ra, Rb);
  var R = Math.max(Ra, Rb);

  let d;
  if ((ab > 0) && (ab < a) && (ab < b)) {
    // We perform the bisection search between the two extreme values
    d = bisect(aIntersection, (r + R - 0.00001), (R - r + 0.00001));
  } else if (ab > 0) {
    d = R - r;
  } else {
    d = R + r + 10;
  }

  return [Ra, Rb, d];
}


type IntersectionBetweenModelErrorsState = {
  selectedDataPoint: any,
  regionSelected: any,
  progressFill: string,
  regressFill: string,
  intersectionFill: string
}

type IntersectionBetweenModelErrorsProps = {
  selectedDataPoint: any,
  filterByInstanceIds: any
}

class IntersectionBetweenModelErrors extends Component<IntersectionBetweenModelErrorsProps, IntersectionBetweenModelErrorsState> {
  constructor(props) {
    super(props);

    this.state = {
      selectedDataPoint: this.props.selectedDataPoint,
      regionSelected: null,
      progressFill: this.getRegionFill("progress", null),
      regressFill: this.getRegionFill("regress", null),
      intersectionFill: this.getRegionFill("intersection", null)
    };

    this.node = React.createRef<HTMLDivElement>();
    this.createVennDiagramPlot = this.createVennDiagramPlot.bind(this);
    this.setSelectedRegion = this.setSelectedRegion.bind(this);
  }

  node: React.RefObject<HTMLDivElement>
  progress: any
  regress: any
  intersection: any

  componentDidMount() {
    this.createVennDiagramPlot();
  }

  componentWillReceiveProps(nextProps) {
    this.setState({
      selectedDataPoint: nextProps.selectedDataPoint,
    });
  }

  componentDidUpdate() {
    this.createVennDiagramPlot();
  }

  getRegionFill(regionName, regionSelected) {
    if (regionName == "intersection" && regionSelected != "intersection") {
      return "rgba(241, 241, 127, 0.8)";
    } else if (regionName == "intersection" && regionSelected == "intersection") {
      return "rgba(141, 141, 27, 0.8)";
    } else if(regionName == "progress" && regionSelected != "progress") {
      return "rgba(175, 227, 141, 0.8)";
    } else if (regionName == "progress" && regionSelected == "progress") {
      return "rgba(75, 127, 41, 0.8)";
    } else if (regionName == "regress" && regionSelected != "regress") {
      return "rgba(206, 160, 205, 0.8)";
    } else if (regionName == "regress" && regionSelected == "regress") {
      return "rgba(106, 60, 105, 0.8)";
    }
  }

  setSelectedRegion(regionName: string) {
    if (regionName == "intersection") {
      this.props.filterByInstanceIds(this.intersection);
    } else if (regionName == "progress") {
      this.props.filterByInstanceIds(this.progress);
    } else if (regionName == "regress") {
      this.props.filterByInstanceIds(this.regress);
    } else {
      console.log("invalid regionName " + (regionName ?? "null"));
    }

    const progressFill = this.getRegionFill("progress", regionName);
    const regressFill = this.getRegionFill("regress", regionName);
    const intersectionFill = this.getRegionFill("intersection", regionName);
    
    this.setState({
      regionSelected: regionName,
      progressFill: progressFill,
      intersectionFill: intersectionFill,
      regressFill: regressFill
    });
  }

  createVennDiagramPlot() {
    var _this = this;
    var body = d3.select(this.node.current);

    var margin = { top: 5, right: 15, bottom: 50, left: 55 }
    var h = 220 - margin.top - margin.bottom
    var w = 320 - margin.left - margin.right

    var tooltip = d3.select("#venntooltip");

    // SVG
    d3.select("#venndiagram").remove();
    var svg = body.append('svg')
        .attr('id', "venndiagram")
        .attr('height',h + margin.top + margin.bottom)
        .attr('width',w + margin.left + margin.right)
      .append('g')
        .attr('transform',`translate(55,${margin.top})`)

    svg.append('text')
      .attr('id','xAxisLabel')
      .attr('y', -20)
      .attr('x', 200)
      .attr('dy','.71em')
      .style('text-anchor','end');

    svg.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", 240)
      .attr("height", h)
      .attr("fill", "rgba(255, 255, 255, 0.8)")
      .attr("stroke", "black")
      .attr("stroke-width", 0.5);

    if (this.state.selectedDataPoint != null) {
      var errorPartition = this.state.selectedDataPoint.models_error_overlap;

      var a = errorPartition[0].length;
      var b = errorPartition[1].length;
      var ab = errorPartition[2].length;

      // Error instance ids of regress instances
      var regress = errorPartition[1].filter(instanceId => (errorPartition[2].indexOf(instanceId) == -1));
      this.regress = regress;
      // Error instance ids of progress instances
      var progress = errorPartition[0].filter(instanceId => (errorPartition[2].indexOf(instanceId) == -1));
      this.progress = progress;
      var regressSize = regress.length;
      var regressProportion = regressSize / this.state.selectedDataPoint.dataset_size;
      var progressSize = progress.length;
      var progressProportion = progressSize / this.state.selectedDataPoint.dataset_size;
      var intersection = errorPartition[2];
      this.intersection = intersection;
      var intersectionSize = intersection.length;
      var intersectionProportion = intersectionSize / this.state.selectedDataPoint.dataset_size;

      var totalErrors = a + b - ab
      var aProportion = 0.0
      var bProportion = 0.0
      var abProportion = 0.0

      var data = [
        {"name": "intersectionRaRb", "area": ab},
        {"name": "Ra", "area": a},
        {"name": "Rb", "area": b}
      ];

      if (totalErrors > 0) {
        aProportion = a / this.state.selectedDataPoint.dataset_size;
        bProportion = b / this.state.selectedDataPoint.dataset_size;
        abProportion = ab / this.state.selectedDataPoint.dataset_size;

        let Ra;
        let Rb;
        let Aab;

        var r = Math.min(Ra, Rb);
        var R = Math.max(Ra, Rb);

        // We perform the bisection search between the two extreme values
        //let d = bisect(aIntersection, (r + R - 0.00001), (R - r + 0.00001));
        let d;
        [Ra, Rb, d] = calculateCircleRadiiAndDistance(a, b, ab, this.state.selectedDataPoint.dataset_size);

        var areas = svg.append("g").attr("id", "areas");

        if ((Math.abs(Ra - Rb) < d) && (d < (Ra + Rb))) {
          // This is the case when the circles have some partial overlap.
          // That is when there is non-trivial overlap between the errors of h1 and h2.

          var circleRad = 50;
          var xCenter = w/2 - d/2;
          var yCenter = h/2
          var xCenter2 = xCenter + d;

          // Draw the path that demarcates the boundary of the Intersection region
          var path = areas.append("path");
          var intersectionPath = d3.path();
          intersectionPath.arc(xCenter, yCenter, Ra, -Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)), Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)));
          intersectionPath.arc(xCenter2, yCenter, Rb, Math.PI - Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)), Math.PI + Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)));
          intersectionPath.closePath();

          // Draw the path that demarcates the boundary of the Regress region
          var rPath = areas.append("path");
          var progressPath = d3.path();
          progressPath.arc(xCenter2, yCenter, Rb,
                          Math.PI + Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)),
                          Math.PI - Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)), true);
          progressPath.arc(xCenter, yCenter, Ra,
                           Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)),
                          -Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)), false);
          progressPath.closePath();

          // Draw the path that demarcates the boundary of the Progress region
          var pPath = areas.append("path");
          var regressPath = d3.path();
          regressPath.arc(xCenter2, yCenter, Rb,
                          Math.PI - Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)),
                          Math.PI + Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)), true);
          regressPath.arc(xCenter, yCenter, Ra,
                          -Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)),
                          Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)), false);
          regressPath.closePath();

          // Draw and style the Intersection region
          path.attr("d", intersectionPath)
            .attr("stroke", "black")
            .attr("stroke-width", "1px")
            .attr("fill", _this.getRegionFill("intersection", _this.state.regionSelected))
              .on("mouseover", function() {
                tooltip.text(`${intersectionSize} (${(intersectionProportion * 100).toFixed(3)}%)`)
                  .style("opacity", 0.8);

                d3.select(this).attr("stroke-width", "3px");
              })
              .on("mousemove", function() {
                var vennDiagramPlot = document.getElementById("venndiagramplot");
                var coords = d3.mouse(vennDiagramPlot);
                tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2}px`)
                  .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
              })
              .on("mouseout", function() {
                tooltip.style("opacity", 0);
                d3.select(this).attr("stroke-width", "1px");
              })
              .on("click", () => _this.setSelectedRegion("intersection"));

          // Draw and style the Regress region
          rPath.attr("d", regressPath)
            .attr("stroke", "black")
            .attr("stroke-width", "1px")
            .attr("fill", _this.getRegionFill("regress", _this.state.regionSelected))
            .on("mouseover", function() {
              tooltip.text(`${regressSize} (${(regressProportion * 100).toFixed(3)}%)`)
                .style("opacity", 0.8);

              d3.select(this).attr("stroke-width", "2px");
            })
            .on("mousemove", function() {
              var vennDiagramPlot = document.getElementById("venndiagramplot");
              var coords = d3.mouse(vennDiagramPlot);
              tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2}px`)
                .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
            })
            .on("mouseout", function() {
              tooltip.style("opacity", 0);
              d3.select(this).attr("stroke-width", "1px");
            })
            .on("click", () => _this.setSelectedRegion("regress"));

          // Draw and style the Progress region
          pPath.attr("d", progressPath)
            .attr("stroke", "black")
            .attr("stroke-width", "1px")
            .attr("fill", _this.getRegionFill("progress", _this.state.regionSelected))
            .on("mouseover", function() {
              tooltip.text(`${progressSize} (${(progressProportion * 100).toFixed(3)}%)`)
                .style("opacity", 0.8);

              d3.select(this).attr("stroke-width", "2px");
            })
            .on("mousemove", function() {
              var vennDiagramPlot = document.getElementById("venndiagramplot");
              var coords = d3.mouse(vennDiagramPlot);
              tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2}px`)
                .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
            })
            .on("mouseout", function() {
              tooltip.style("opacity", 0);
              d3.select(this).attr("stroke-width", "1px");
            })
            .on("click", () => _this.setSelectedRegion("progress"));

          areas.selectAll("g")
            .data(data);
        } else if (d <= Math.abs(Ra - Rb) && (Ra == Rb)) {
          // This is the case when one circle is completely contained within the other.
          // And when h1 and h2 errors are identical.

          var xCenter = w/2 - d/2;
          var yCenter = h/2
          var xCenter2 = xCenter + d;

          areas.append("circle")
              .attr("r", Ra)
              .attr('transform',
                  "translate(" +
                  xCenter + "," +
                  yCenter + ")")
              .attr("fill", _this.getRegionFill("intersection", _this.state.regionSelected))
              .attr("stroke", "black")
              .attr("stroke-width", "1px")
              .on("mouseover", function() {
                tooltip.text(`${intersectionSize} (${(intersectionProportion * 100).toFixed(0)}%)`)
                  .style("opacity", 0.8);
                d3.select(this).attr("stroke-width", "2px");
              })
              .on("mousemove", function() {
                var vennDiagramPlot = document.getElementById("venndiagramplot");
                var coords = d3.mouse(vennDiagramPlot);
                tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2}px`)
                  .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
              })
              .on("mouseout", function() {
                tooltip.style("opacity", 0);
                d3.select(this).attr("stroke-width", "1px");
              })
              .on("click", () => _this.setSelectedRegion("intersection"));
        } else if (d <= Math.abs(Ra - Rb) && (Ra > Rb)) {
          // This is the case when one circle is completely contained within the other.
          // And h1 errors fully contain h2 errors
          var xCenter = w/2 - d/2;
          var yCenter = h/2
          var xCenter2 = xCenter + d;
          areas.append("circle")
              .attr("r", Rb)
              .attr('transform',
                  "translate(" +
                  xCenter2 + "," +
                  yCenter + ")")
              .attr("fill", _this.getRegionFill("intersection", _this.state.regionSelected))
              .attr("stroke", "black")
              .attr("stroke-width", "1px")
              .on("mouseover", function() {
                tooltip.text(`${intersectionSize} (${(intersectionProportion * 100).toFixed(0)}%)`)
                  .style("opacity", 0.8);
                d3.select(this).attr("stroke-width", "2px");
              })
              .on("mousemove", function() {
                var vennDiagramPlot = document.getElementById("venndiagramplot");
                var coords = d3.mouse(vennDiagramPlot);
                tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2}px`)
                  .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
              })
              .on("mouseout", function() {
                tooltip.style("opacity", 0);
                d3.select(this).attr("stroke-width", "1px");
              })
              .on("click", () => _this.setSelectedRegion("intersection"));

          // Draw the path that demarcates the boundary of the Progress region
          var pPath = areas.append("path");
          var progressPath = d3.path();
          progressPath.arc(xCenter2, yCenter, Rb, 0, 2 * Math.PI, true);
          progressPath.arc(xCenter, yCenter, Ra, 0, 2 * Math.PI, false);
          progressPath.closePath();

          // Draw and style the Regress region
          pPath.attr("d", progressPath)
            .attr("stroke", "black")
            .attr("stroke-width", "1px")
            .attr("fill", _this.getRegionFill("progress", _this.state.regionSelected))
            .on("mouseover", function() {
              tooltip.text(`${progressSize} (${(progressProportion * 100).toFixed(3)}%)`)
                .style("opacity", 0.8);

              d3.select(this).attr("stroke-width", "2px");
            })
            .on("mousemove", function() {
              var vennDiagramPlot = document.getElementById("venndiagramplot");
              var coords = d3.mouse(vennDiagramPlot);
              tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2}px`)
                .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
            })
            .on("mouseout", function() {
              tooltip.style("opacity", 0);
              d3.select(this).attr("stroke-width", "1px");
            })
            .on("click", () => _this.setSelectedRegion("progress"));
        } else if (d <= Math.abs(Ra - Rb) && (Ra < Rb)) {
          // This is the case when one circle is completely contained within the other.
          // And h2 errors filly contain h1 errors
          var xCenter = w/2 - d/2;
          var yCenter = h/2
          var xCenter2 = xCenter + d;
          areas.append("circle")
              .attr("r", Ra)
              .attr('transform',
                  "translate(" +
                  xCenter2 + "," +
                  yCenter + ")")
              .attr("fill", _this.getRegionFill("intersection", _this.state.regionSelected))
              .attr("stroke", "black")
              .attr("stroke-width", "1px")
              .on("mouseover", function() {
                tooltip.text(`${intersectionSize} (${(intersectionProportion * 100).toFixed(0)}%)`)
                  .style("opacity", 0.8);
                d3.select(this).attr("stroke-width", "2px");
              })
              .on("mousemove", function() {
                var vennDiagramPlot = document.getElementById("venndiagramplot");
                var coords = d3.mouse(vennDiagramPlot);
                tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2}px`)
                  .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
              })
              .on("mouseout", function() {
                tooltip.style("opacity", 0);
                d3.select(this).attr("stroke-width", "1px");
              })
              .on("click", () => _this.setSelectedRegion("intersection"));

          // Draw the path that demarcates the boundary of the Regress region
          var rPath = areas.append("path");
          var regressPath = d3.path();
          regressPath.arc(xCenter2, yCenter, Ra, 0, 2 * Math.PI, true);
          regressPath.arc(xCenter, yCenter, Rb, 0, 2 * Math.PI, false);
          regressPath.closePath();

          // Draw and style the Regress region
          rPath.attr("d", regressPath)
            .attr("stroke", "black")
            .attr("stroke-width", "1px")
            .attr("fill", _this.getRegionFill("regress", _this.state.regionSelected))
            .on("mouseover", function() {
              tooltip.text(`${regressSize} (${(regressProportion * 100).toFixed(3)}%)`)
                .style("opacity", 0.8);

              d3.select(this).attr("stroke-width", "2px");
            })
            .on("mousemove", function() {
              var vennDiagramPlot = document.getElementById("venndiagramplot");
              var coords = d3.mouse(vennDiagramPlot);
              tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2}px`)
                .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
            })
            .on("mouseout", function() {
              tooltip.style("opacity", 0);
              d3.select(this).attr("stroke-width", "1px");
            })
            .on("click", () => _this.setSelectedRegion("regress"));
        } else if (d >= Math.abs(Ra + Rb)) {
          // This is the case when the circles are completely disjoint.
          // That is when there is no overlap between the errors of h1 and h2.

          var xCenter = w/2 - d/2;
          var yCenter = h/2
          var xCenter2 = xCenter + d;
          
          areas.append("circle")
              .attr("r", Ra)
              .attr('transform',
                  "translate(" +
                  xCenter + "," +
                  yCenter + ")")
              .attr("fill", _this.getRegionFill("progress", _this.state.regionSelected))
              .attr("stroke", "black")
              .attr("stroke-width", "1px")
              .on("mouseover", function() {
                tooltip.text(`${progressSize} (${(progressProportion * 100).toFixed(0)}%)`)
                  .style("opacity", 0.8);
                d3.select(this).attr("stroke-width", "2px");
              })
              .on("mousemove", function() {
                var vennDiagramPlot = document.getElementById("venndiagramplot");
                var coords = d3.mouse(vennDiagramPlot);
                tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2}px`)
                  .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
              })
              .on("mouseout", function() {
                tooltip.style("opacity", 0);
                d3.select(this).attr("stroke-width", "1px");
              })
              .on("click", () => _this.setSelectedRegion("progress"));

          areas.append("circle")
              .attr("r", Rb)
              .attr('transform',
                  "translate(" +
                  xCenter2 + "," +
                  yCenter + ")")
              .attr("fill", _this.getRegionFill("regress", _this.state.regionSelected))
              .attr("stroke", "black")
              .attr("stroke-width", "1px")
              .on("mouseover", function() {
                tooltip.text(`${regressSize} (${(regressProportion * 100).toFixed(0)}%)`)
                  .style("opacity", 0.8);
                d3.select(this).attr("stroke-width", "2px");
              })
              .on("mousemove", function() {
                var vennDiagramPlot = document.getElementById("venndiagramplot");
                var coords = d3.mouse(vennDiagramPlot);
                tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2}px`)
                  .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
              })
              .on("mouseout", function() {
                tooltip.style("opacity", 0);
                d3.select(this).attr("stroke-width", "1px");
              })
              .on("click", () => _this.setSelectedRegion("regress"));
        }
      }

    }

  }

  render() {
    const diagramInfo = "Displays the newly trained model’s error counts and percentages in relation to those of the previous model.";
    const progressInfo = "Indicates errors made by the previous model that the newly trained model does not make.​";
    const regressInfo = "Indicates errors made by the newly trained model that the previous model did not make.​";
    const intersectionInfo = "Indicates errors made by both the previous and newly trained models.​";
    const _this = this;
    function getBlockClass(regionName: string) : string {
      if (_this.state.regionSelected == regionName) {
        return "venn-legend-row-block-selected";
      } else {
        return "venn-legend-row-block";
      }
    }

    return (
      <div className="plot plot-venn" ref={this.node} id="venndiagramplot">
        <div className="plot-title-row">
          Intersection Between Model Errors
          <InfoTooltip direction={DirectionalHint.topCenter} message={diagramInfo}/>
        </div>
        { this.state.selectedDataPoint != null ? 
        <div className="venn-legend-row">
          <div id="progress-container" className={getBlockClass("progress")} onClick={() => _this.setSelectedRegion("progress")}>
            <div id="progress" className="venn-legend-color-box" style={{background: _this.state.progressFill}}/>
            Progress
            <InfoTooltip direction={DirectionalHint.topCenter} message={progressInfo}/>
          </div>
          <div id="regress-container" className={getBlockClass("regress")} onClick={() => _this.setSelectedRegion("regress")}>
            <div id="regress" className="venn-legend-color-box" style={{background: _this.state.regressFill}}/>
            Regress
            <InfoTooltip direction={DirectionalHint.topCenter} message={regressInfo}/>
          </div>
          <div id="commonerror-container" className={getBlockClass("intersection")} onClick={() => _this.setSelectedRegion("intersection")}>
            <div id="commonerror" className="venn-legend-color-box" style={{background: _this.state.intersectionFill}}/>
            Common
            <InfoTooltip direction={DirectionalHint.topCenter} message={intersectionInfo}/>
          </div>
        </div> : null }
        <div className="tooltip" id="venntooltip" />
      </div>
    );
  }
}
export default IntersectionBetweenModelErrors;
