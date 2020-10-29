// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type IncompatiblePointDistributionState = {
  selectedDataPoint: any,
  page: number
}

type IncompatiblePointDistributionProps = {
  selectedDataPoint: any,
  pageSize?: number
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

    var margin = { top: 15, right: 15, bottom: 50, left: 55 }
    var h = 250 - margin.top - margin.bottom
    var w = 250;

    // SVG
    d3.select("#incompatiblepointdistribution").remove();
    var svg = body.append('svg')
        .attr('id', "incompatiblepointdistribution")
        .attr('height',h + margin.top + margin.bottom)
        .attr('width',w + margin.left + margin.right)
        .attr('float', 'left');

    svg.append("text")
       .attr("x", margin.left + 20)
       .attr("y", 15)
       .attr("font-size", "10px")
       .text("Distribution of Incompatible Points")

    if (this.props.selectedDataPoint != null) {
      // Sort the data into the dataRows based on the ordering of the sorted classes
      var totalErrors = 0;
      var start_i = this.state.page * this.props.pageSize;
      var end_i = Math.min(start_i + this.props.pageSize, this.props.selectedDataPoint.sorted_classes.length);
      for (var i = start_i; i < end_i; i++) {
        var instanceClass = this.props.selectedDataPoint.sorted_classes[i];
        var dataRow = this.props.selectedDataPoint.h2_error_instance_ids_by_class.filter(
          dataDict => (dataDict["class"] == instanceClass)).pop();
        totalErrors += dataRow["errorInstanceIds"].length;
      }

      // We add the following so that we do not get a divide by zero
      // error later on if there are no errors.
      if (totalErrors == 0) {
        totalErrors = 1;
      }

      var dataRows = [];
      for (var i=start_i; i < end_i; i++) {
        var instanceClass = this.props.selectedDataPoint.sorted_classes[i];
        var dataRow = this.props.selectedDataPoint.h2_error_instance_ids_by_class.filter(
          dataDict => (dataDict["class"] == instanceClass)).pop();
        dataRows.push(dataRow);
      }

      var xScale = d3.scaleBand().range([0, w]).padding(0.4),
          yScale = d3.scaleLinear().range([h, 0]);

      var g = svg.append("g")
                 .attr("transform", "translate(" + 55 + "," + 30 + ")");

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
         .attr("y", function(d) { return yScale(d.errorInstanceIds.length/totalErrors * 100); })
         .attr("width", xScale.bandwidth())
         .attr("height", function(d) { return h - yScale(d.errorInstanceIds.length/totalErrors * 100); });
      }
  }

  render() {
    let numClasses = this.props.selectedDataPoint?.sorted_classes?.length ?? 0;
    let numPages = Math.ceil(numClasses/this.props.pageSize) - 1;
    return (
      <div className="plot plot-distribution">
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
