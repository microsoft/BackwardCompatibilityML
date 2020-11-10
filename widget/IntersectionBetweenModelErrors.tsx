// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";
import { bisect } from "./optimization.tsx";


type IntersectionBetweenModelErrorsState = {
  selectedDataPoint: any,
  regionSelected: any
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
      regionSelected: null
    };

    this.node = React.createRef<HTMLDivElement>();
    this.createVennDiagramPlot = this.createVennDiagramPlot.bind(this);
  }

  node: React.RefObject<HTMLDivElement>

  componentDidMount() {
    this.createVennDiagramPlot();
  }

  componentWillReceiveProps(nextProps) {
    this.setState({
      selectedDataPoint: nextProps.selectedDataPoint
    });
  }

  componentDidUpdate() {
    this.createVennDiagramPlot();
  }

  createVennDiagramPlot() {
    var _this = this;
    var body = d3.select(this.node.current);

    var margin = { top: 15, right: 15, bottom: 50, left: 55 }
    var h = 250 - margin.top - margin.bottom
    var w = 320 - margin.left - margin.right

    var tooltip = d3.select("#venntooltip");

    // SVG
    d3.select("#venndiagram").remove();
    var svg = body.append('svg')
        .attr('id', "venndiagram")
        .attr('height',h + margin.top + margin.bottom)
        .attr('width',w + margin.left + margin.right)
      .append('g')
        .attr('transform',`translate(55,${margin.top + 15})`)

    svg.append('text')
      .attr('id','xAxisLabel')
      .attr('y', -20)
      .attr('x', 200)
      .attr('dy','.71em')
      .style('text-anchor','end')
      .text("Intersection Between Model Errors")
      .attr("font-family", "sans-serif")
      .attr("font-size", "10px")
      .attr("fill", "black");

    svg.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", 240)
      .attr("height", h)
      .attr("fill", "rgba(255, 255, 255, 0.8)")
      .attr("stroke", "black")
      .attr("stroke-width", 0.5);

    if (this.state.selectedDataPoint != null) {
      //var selectedDataPoint = this.state.selectedDataPoint;
      var errorPartition = this.state.selectedDataPoint.models_error_overlap;
      var a = errorPartition[0].length;
      var b = errorPartition[1].length;
      var ab = errorPartition[2].length;

      // Error instance ids of regress instances
      var regress = errorPartition[0].filter(instanceId => (errorPartition[2].indexOf(instanceId) == -1));
      // Error instance ids of progress instances
      var progress = errorPartition[1].filter(instanceId => (errorPartition[2].indexOf(instanceId) == -1));
      var regressSize = regress.length;
      var regressProportion = regressSize / this.state.selectedDataPoint.dataset_size;
      var progressSize = progress.length;
      var progressProportion = progressSize / this.state.selectedDataPoint.dataset_size;

      var totalErrors = a + b - ab
      var aProportion = 0.0
      var bProportion = 0.0
      var abProportion = 0.0

      var data = [
        {"name": "intersectionRaRb", "area": ab},
        {"name": "Ra", "area": a},
        {"name": "Rb", "area": b}
      ];

      // Colors
      var green = "rgba(175, 227, 141, 0.8)";
      var red = "rgba(206, 160, 205, 0.8)";
      var yellow = "rgba(241, 241, 127, 0.8)";

      // Draw the legend of the Vnn diagram
      var legendEntries = [
        {"label": "Progress", "color": green},
        {"label": "Regress", "color": red},
        {"label": "Common", "color": yellow},
      ];
      var vennLegend = svg.append("g").attr("id", "vennlegend");
      vennLegend.append("rect")
        .attr("id", "progress")
        .attr("width", "10px")
        .attr("height", "10px")
        .attr("x", "10px")
        .attr("y", "5px");

      vennLegend.append("rect")
        .attr("id", "regress")
        .attr("width", "10px")
        .attr("height", "10px")
        .attr("x", "90px")
        .attr("y", "5px");

      vennLegend.append("rect")
        .attr("id", "commonerror")
        .attr("width", "10px")
        .attr("height", "10px")
        .attr("x", "170px")
        .attr("y", "5px");

      vennLegend.append("text")
        .attr("x", "25px")
        .attr("y", "12px")
        .attr("font-size", "10px")
        .attr("text-anchor", "left")
        .style("alignment-baseline", "middle");

      vennLegend.append("text")
        .attr("x", "105px")
        .attr("y", "12px")
        .attr("font-size", "10px")
        .attr("text-anchor", "left")
        .style("alignment-baseline", "middle");

      vennLegend.append("text")
        .attr("x", "185px")
        .attr("y", "12px")
        .attr("font-size", "10px")
        .attr("text-anchor", "left")
        .style("alignment-baseline", "middle");

      vennLegend.selectAll("rect")
        .data(legendEntries)
        .attr("fill", function(d) {
          return d["color"];
        });

      vennLegend.selectAll("text")
        .data(legendEntries)
        .text(function(d) {
          return d["label"];
        });

      if (totalErrors > 0) {
        aProportion = a / this.state.selectedDataPoint.dataset_size;
        bProportion = b / this.state.selectedDataPoint.dataset_size;
        abProportion = ab / this.state.selectedDataPoint.dataset_size;

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

        // We perform the bisection search between the two extreme values
        let d = bisect(aIntersection, (r + R - 0.00001), (R - r + 0.00001));

        var circleRad = 50;
        var xCenter = w/2 - d/2;
        var yCenter = h/2
        var xCenter2 = xCenter + d;

        var areas = svg.append("g").attr("id", "areas");

        // Draw the path that demarcates the boundary of the Intersection region
        var path = areas.append("path");
        var intersectionPath = d3.path();
        intersectionPath.arc(xCenter, yCenter, Ra, -Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)), Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)));
        intersectionPath.arc(xCenter2, yCenter, Rb, Math.PI - Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)), Math.PI + Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)));
        intersectionPath.closePath();

        // Draw the path that demarcates the boundary of the Regress region
        var rPath = areas.append("path");
        var regressPath = d3.path();
        regressPath.arc(xCenter2, yCenter, Rb,
                        Math.PI + Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)),
                        Math.PI - Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)), true);
        regressPath.arc(xCenter, yCenter, Ra,
                         Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)),
                        -Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)), false);
        regressPath.closePath();

        // Draw the path that demarcates the boundary of the Progress region
        var pPath = areas.append("path");
        var progressPath = d3.path();
        progressPath.arc(xCenter2, yCenter, Rb,
                        Math.PI - Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)),
                        Math.PI + Math.acos((Math.pow(d, 2) + Math.pow(Rb, 2) - Math.pow(Ra, 2))/(2 * d *Rb)), true);
        progressPath.arc(xCenter, yCenter, Ra,
                        -Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)),
                        Math.acos((Math.pow(d, 2) + Math.pow(Ra, 2) - Math.pow(Rb, 2))/(2 * d *Ra)), false);
        progressPath.closePath();

        function getRegionStrokeWidth(regionName) {
          if ((_this.state.regionSelected == regionName) && (regionName == "intersection")) {
            return "3px";
          } else if (_this.state.regionSelected == regionName) {
            return "2px";
          } else {
            return "1px";
          }
        }

        function getRegionFill(regionName) {
          if ((regionName == "intersection") && !(_this.state.regionSelected == "intersection")) {
            return "rgba(241, 241, 127, 0.8)";
          } else if ((regionName == "intersection") && (_this.state.regionSelected == "intersection")) {
            return "rgba(141, 141, 27, 0.8)";
          } else if((regionName == "progress") && !(_this.state.regionSelected == "progress")) {
            return "rgba(206, 160, 205, 0.8)";
          } else if ((regionName == "progress") && (_this.state.regionSelected == "progress")) {
            return "rgba(106, 60, 105, 0.8)";
          } else if ((regionName == "regress") && !(_this.state.regionSelected == "regress")) {
            return "rgba(175, 227, 141, 0.8)";
          } else if ((regionName == "regress") && (_this.state.regionSelected == "regress")) {
            return "rgba(75, 127, 41, 0.8)";
          }
        }

        // Draw and style the Intersection region
        path.attr("d", intersectionPath)
          .attr("stroke", "black")
          .attr("stroke-width", getRegionStrokeWidth("intersection"))
          .attr("fill", getRegionFill("intersection"))
            .on("mouseover", function() {
              tooltip.text(`${ab} (${(abProportion * 100).toFixed(3)}%)`)
                .style("opacity", 0.8);

              //bringToTop("intersectionRaRb");

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
              d3.select(this).attr("stroke-width", getRegionStrokeWidth("intersection"));
            })
            .on("click", function() {
              _this.props.filterByInstanceIds(errorPartition[2]);
              _this.setState({
                regionSelected: "intersection"
              });
            });

        // Draw and style the Regress region
        rPath.attr("d", regressPath)
          .attr("stroke", "black")
          .attr("stroke-width", getRegionStrokeWidth("regress"))
          .attr("fill", getRegionFill("regress"))
          .on("mouseover", function() {
            tooltip.text(`${regressSize} (${(regressProportion * 100).toFixed(3)}%)`)
              .style("opacity", 0.8);

            //bringToTop("Ra");

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
            d3.select(this).attr("stroke-width", getRegionStrokeWidth("regress"));
          })
          .on("click", function() {
            _this.props.filterByInstanceIds(regress);
            _this.setState({
              regionSelected: "regress"
            });
          });

        // Draw and style the Progress region
        pPath.attr("d", progressPath)
          .attr("stroke", "black")
          .attr("stroke-width", getRegionStrokeWidth("progress"))
          .attr("fill", getRegionFill("progress"))
          .on("mouseover", function() {
            tooltip.text(`${progressSize} (${(progressProportion * 100).toFixed(3)}%)`)
              .style("opacity", 0.8);

            //bringToTop("Rb");

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
            d3.select(this).attr("stroke-width", getRegionStrokeWidth("progress"));
          })
          .on("click", function() {
            _this.props.filterByInstanceIds(progress);
            _this.setState({
              regionSelected: "progress"
            });
          });

        areas.selectAll("g")
          .data(data);
      }

    }

  }

  render() {
    return (
      <div className="plot plot-venn" ref={this.node} id="venndiagramplot">
        <div className="tooltip" id="venntooltip" />
      </div>
    );
  }
}
export default IntersectionBetweenModelErrors;
