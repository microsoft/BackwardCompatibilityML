// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type IntersectionBetweenModelErrorsState = {
  selectedDataPoint: any
}

type IntersectionBetweenModelErrorsProps = {
  selectedDataPoint: any
}

class IntersectionBetweenModelErrors extends Component<IntersectionBetweenModelErrorsProps, IntersectionBetweenModelErrorsState> {
  constructor(props) {
    super(props);

    this.state = {
      selectedDataPoint: this.props.selectedDataPoint
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
    var w = 250 - margin.left - margin.right

    // SVG
    d3.select("#venndiagram").remove();
    var svg = body.append('svg')
        .attr('id', "venndiagram")
        .attr('height',h + margin.top + margin.bottom)
        .attr('width',w + margin.left + margin.right)
      .append('g')
        .attr('transform',`translate(0,${margin.top + 15})`)

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
      .attr("width", w + margin.left + margin.right)
      .attr("height", h)
      .attr("fill", "rgba(255, 255, 255, 0.8)")
      .attr("stroke", "black")
      .attr("stroke-width", 0.5);

    if (this.state.selectedDataPoint != null) {
      //var selectedDataPoint = this.state.selectedDataPoint;
      var errorPartition = this.state.selectedDataPoint.models_error_overlap;
      var a = errorPartition[0];
      var b = errorPartition[1];
      var ab = errorPartition[2];

      let Ra;
      let Rb;
      let Aab;
      let x = 1;
      if (a >= b) {
        x = (50 * 50 * 3.14) / a;
        Ra = 50;
        Rb = Math.sqrt(b * x / 3.14);
        Aab = ab * x;
      } else {
        x = (50 * 50 * 3.14) / b;
        Rb = 50;
        Ra = Math.sqrt(a * x / 3.14);
        Aab = ab * x;
      }

      let d = (Ra + Rb) / 1.5;
      if (ab == a || ab == b) {
        d = Math.abs(Ra - Rb) / 1.5;
      } else if (ab == 0) {
        d = Ra + Rb + 10;
      }

      var circleRad = 50;
      var xCenter = w/4 + margin.left;
      var yCenter = h/2
      svg.append("circle")
          .attr("r", Ra)
          .attr('transform',
              "translate(" +
              xCenter + "," +
              yCenter + ")")
          .attr("fill", "rgba(170, 170, 255, 0.8)");

      var xCenter2 = xCenter + d;
      svg.append("circle")
          .attr("r", Rb)
          .attr('transform',
              "translate(" +
              xCenter2 + "," +
              yCenter + ")")
          .attr("fill", "rgba(206, 160, 205, 0.8)");
    }

  }

  render() {
    return (
      <div className="plot" ref={this.node} />
    );
  }
}
export default IntersectionBetweenModelErrors;
