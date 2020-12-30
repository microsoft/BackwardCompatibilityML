// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type LegendProps = {
  testing: boolean,
  training: boolean,
  newError: boolean,
  strictImitation: boolean
}

class Legend extends Component<LegendProps, null> {
  constructor(props) {
    super(props);

    this.node = React.createRef<HTMLDivElement>();
    this.createLegend = this.createLegend.bind(this);
  }

  node: React.RefObject<HTMLDivElement>

  componentDidMount() {
    this.createLegend();
  }

  componentWillReceiveProps(nextProps) {
    this.setState({
      testing: nextProps.testing,
      training: nextProps.training,
      newError: nextProps.newError,
      strictImitation: nextProps.strictImitation
    });
  }

  componentDidUpdate() {
    this.createLegend();
  }

  createLegend() {
    var _this = this;
    var body = d3.select(this.node.current);

    var margin = { top: 15, right: 5, bottom: 50, left: 10 }
    var h = 50 - margin.top - margin.bottom
    var w = 550 - margin.left - margin.right

    var formatPercent = d3.format('.3f');
    var colorMap = {
      "training": {
        "new-error": "rgba(170, 170, 255, 0.8)",
        "strict-imitation": "rgba(113, 113, 255, 0.8)"
      },
      "testing": {
        "new-error": "rgba(206, 160, 205, 0.8)",
        "strict-imitation": "rgba(226, 75, 158, 0.8)"
      }
    };

    var legendEntries = [];
    if (this.props.training && this.props.newError) {
      legendEntries.push(["training", "new-error", "New Error - Training"]);
    }

    if (this.props.training && this.props.strictImitation) {
      legendEntries.push(["training", "strict-imitation", "Strict Imitation - Training"]);
    }

    if (this.props.testing && this.props.newError) {
      legendEntries.push(["testing", "new-error", "New Error - Testing"]);
    }

    if (this.props.testing && this.props.strictImitation) {
      legendEntries.push(["testing", "strict-imitation", "Strict Imitation - Testing"]);
    }

    d3.select(`#legend`).remove();
    var svg = body.append('svg')
        .attr('id', "legend")
        .attr('height',h + margin.top + margin.bottom)
        .attr('width',w + margin.left + margin.right)
      .append('g')
        .attr('transform','translate(' + margin.left + ',' + margin.top + ')')

    var size = 10
    svg.selectAll("mydots")
      .data(legendEntries)
      .enter()
      .append("rect")
        .attr("x", function(d,i) {
          return i*130
        })
        .attr("y", 10) // 100 is where the first dot appears. 25 is the distance between dots
        .attr("width", size)
        .attr("height", size)
        .style("fill", function(d) {
          return colorMap[d[0]][d[1]];
        });

    // Add one dot in the legend for each name.
    svg.selectAll("mylabels")
      .data(legendEntries)
      .enter()
      .append("text")
        .attr("x", function(d,i) {
          return (i * 130) + (size * 1.25)})
        .attr("y", 17) // 100 is where the first dot appears. 25 is the distance between dots
        .attr("font-size", "10px")
        .text(function(d){ return d[2]})
        .attr("text-anchor", "left")
        .style("alignment-baseline", "middle");
  }

  render() {
    return (
      <div className="legend" ref={this.node} />
    );
  }
}
export default Legend;
