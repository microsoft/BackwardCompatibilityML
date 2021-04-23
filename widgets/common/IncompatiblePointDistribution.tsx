// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";
import { InfoTooltip } from "./InfoTooltip.tsx"
import { DirectionalHint } from '@fluentui/react';


type IncompatiblePointDistributionState = {
  selectedDataPoint: any,
  page: number
}

type IncompatiblePointDistributionProps = {
  selectedDataPoint: any,
  selectedClass?: number,
  setSelectedClass: any,
  pageSize?: number,
  filterByInstanceIds: any
}

class IncompatiblePointDistribution extends Component<IncompatiblePointDistributionProps, IncompatiblePointDistributionState> {

  public static defaultProps = {
      pageSize: 5
  };

  constructor(props) {
    super(props);

    this.state = {
      selectedDataPoint: this.props.selectedDataPoint,
      page: 0
    };

    this.node = React.createRef<HTMLDivElement>();
    this.createDistributionBarChart = this.createDistributionBarChart.bind(this);
  }

  node: React.RefObject<HTMLDivElement>

  componentDidMount() {
    this.createDistributionBarChart();
  }

  componentWillReceiveProps(nextProps) {
    this.setState({
      selectedDataPoint: nextProps.selectedDataPoint,
    });
  }

  componentDidUpdate() {
    this.createDistributionBarChart();
  }

  createDistributionBarChart() {
    var _this = this;
    var body = d3.select(this.node.current);

    var margin = { top: 5, right: 15, bottom: 50, left: 55 }
    var h = 250 - margin.top - margin.bottom
    var w = 392;

    // SVG
    d3.select("#incompatiblepointdistribution").remove();
    var svg = body.append('svg')
        .attr('id', "incompatiblepointdistribution")
        .attr('height',h + margin.top + margin.bottom)
        .attr('width',w + margin.left + margin.right)
        .attr('float', 'left');

    if (this.props.selectedDataPoint != null) {
      // Sort the data into the dataRows based on the ordering of the sorted classes
      var totalIncompatible = 0;
      var startI = this.state.page * this.props.pageSize;
      var endI = Math.min(startI + this.props.pageSize, this.props.selectedDataPoint.sorted_classes.length);
      for (var i = startI; i < endI; i++) {
        var instanceClass = this.props.selectedDataPoint.sorted_classes[i];
        var dataRow = this.props.selectedDataPoint.h2_incompatible_instance_ids_by_class.filter(
          dataDict => (dataDict["class"] == instanceClass)).pop();
        if (dataRow) {
          totalIncompatible += dataRow["incompatibleInstanceIds"]?.length ?? 0;
        }
      }

      // We add the following so that we do not get a divide by zero
      // error later on if there are no incompatible points.
      if (totalIncompatible == 0) {
        totalIncompatible = 1;
      }

      var dataRows = [];
      for (var i=startI; i < endI; i++) {
        var instanceClass = this.props.selectedDataPoint.sorted_classes[i];
        var dataRow = this.props.selectedDataPoint.h2_incompatible_instance_ids_by_class.filter(
          dataDict => (dataDict["class"] == instanceClass)).pop();
        if (dataRow) {
          dataRows.push(dataRow);
        } else {
          dataRows.push({class: instanceClass, incompatibleInstanceIds: []})
        }
      }

      var xScale = d3.scaleBand().range([0, w]).padding(0.4),
          yScale = d3.scaleLinear().range([h, 0]);

      var g = svg.append("g")
                 .attr("transform", "translate(" + 55 + "," + margin.top + ")");

        xScale.domain(dataRows.map(function(d) { return d.class; }));
        yScale.domain([0, 100]);

        g.append("g")
         .attr("transform", "translate(0," + h + ")")
         .call(d3.axisBottom(xScale))
         .append("text")
         .attr("y", 30)
         .attr("x", (w + margin.left)/2)
         .attr("text-anchor", "end")
         .attr("fill", "black")
         .text("Classes");

        g.append("g")
         .call(d3.axisLeft(yScale).tickFormat(function(d){
             return d;
         })
         .ticks(10))
         .append("text")
         .attr("transform", "rotate(-90)")
         .attr("y", 6)
         .attr("dy", "-5.1em")
         .attr("text-anchor", "end")
         .attr("fill", "black")
         .text("Percentage of Incompatible Points");

        g.selectAll(".bar")
         .data(dataRows)
         .enter().append("rect")
         .attr("class", "bar")
         .attr("x", function(d) { return xScale(d.class); })
         .attr("y", function(d) { return yScale(d.incompatibleInstanceIds.length/totalIncompatible * 100); })
         .attr("width", xScale.bandwidth())
         .attr("height", function(d) { return h - yScale(d.incompatibleInstanceIds.length/totalIncompatible * 100); })
         .classed("highlighted-bar", function(d) { return d.class == _this.props.selectedClass })
         .on("click", function(d) {
           _this.props.filterByInstanceIds(d.incompatibleInstanceIds);
           _this.props.setSelectedClass(d.class);
         });
      }
  }

  render() {
    let numClasses = this.props.selectedDataPoint?.sorted_classes?.length ?? 0;
    let numPages = Math.ceil(numClasses/this.props.pageSize) - 1;
    const message = "Displays distribution of errors not made by the previous model as they occur across classes of the newly trained model.​";
    return (
      <div className="plot plot-distribution">
        <div className="plot-title-row">
          Distribution of Incompatible Points
          <InfoTooltip message={message} direction={DirectionalHint.topCenter} />
        </div>
        <div ref={this.node}/>
        <div className="page-button-row">
          <button onClick={() => {
            this.setState({
              page: Math.max(0, this.state.page-1)
            })
          }}>&lt;</button>
          <span>{this.state.page+1} of {numPages+1}</span>
          <button onClick={() => {
            this.setState({
              page: Math.min(numPages, this.state.page+1)
            })
          }}>&gt;</button>
        </div>
      </div>
    );
  }
}
export default IncompatiblePointDistribution;
