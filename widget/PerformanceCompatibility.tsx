// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";


type PerformanceCompatibilityState = {
  data: any,
  testing: boolean,
  training: boolean,
  newError: boolean,
  strictImitation: boolean
}

type PerformanceCompatibilityProps = {
  data: any,
  testing: boolean,
  training: boolean,
  newError: boolean,
  strictImitation: boolean,
  compatibilityScoreType: string,
  selectDataPoint: (d: any) => void,
  getModelEvaluationData: (evaluationId: number) => void
}

class PerformanceCompatibility extends Component<PerformanceCompatibilityProps, PerformanceCompatibilityState> {
  constructor(props) {
    super(props);

    this.state = {
      data: this.props.data,
      testing: this.props.testing,
      training: this.props.training,
      newError: this.props.newError,
      strictImitation: this.props.strictImitation
    };

    this.node = React.createRef<HTMLDivElement>();
    this.createPVCPlot = this.createPVCPlot.bind(this);
  }

  node: React.RefObject<HTMLDivElement>

  componentDidMount() {
    this.createPVCPlot();
  }

  componentWillReceiveProps(nextProps) {
    this.setState({
      data: nextProps.data,
      testing: nextProps.testing,
      training: nextProps.training,
      newError: nextProps.newError,
      strictImitation: nextProps.strictImitation
    });
  }

  componentDidUpdate() {
    this.createPVCPlot();
  }

  createPVCPlot() {
    var _this = this;
    var body = d3.select(this.node.current);
    var data = this.state.data;

    var margin = { top: 15, right: 15, bottom: 50, left: 55 }
    var h = 250 - margin.top - margin.bottom
    var w = 250 - margin.left - margin.right

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

    var allDataPoints = [];
    if (_this.state.training && _this.state.newError) {
      allDataPoints = allDataPoints.concat(data.filter(d => (d["training"] && d["new-error"])));
    }

    if (_this.state.training && _this.state.strictImitation) {
      allDataPoints = allDataPoints.concat(data.filter(d => (d["training"] && d["strict-imitation"])));
    }

    if (_this.state.testing && _this.state.newError) {
      allDataPoints = allDataPoints.concat(data.filter(d => (d["testing"] && d["new-error"])));
    }

    if (_this.state.testing && _this.state.strictImitation) {
      allDataPoints = allDataPoints.concat(data.filter(d => (d["testing"] && d["strict-imitation"])));
    }

    // var xScale = d3.scaleLinear()
    //   .domain([
    //     d3.min([0,d3.min(data,function (d) { return d[_this.props.compatibilityScoreType] })]),
    //     d3.max([0,d3.max(data,function (d) { return d[_this.props.compatibilityScoreType] })])
    //     ])
    //   .range([0,w])
    // var yScale = d3.scaleLinear()
    //   .domain([
    //     d3.min([0,d3.min(data,function (d) { return d['performance'] })]),
    //     d3.max([0,d3.max(data,function (d) { return d['performance'] })])
    //     ])
    //   .range([h,0])

    var xScale = d3.scaleLinear()
      .domain([
        d3.min(allDataPoints,function (d) { return d[_this.props.compatibilityScoreType] }),
        d3.max(allDataPoints,function (d) { return d[_this.props.compatibilityScoreType] })
        ])
      .range([0,w])
    var yScale = d3.scaleLinear()
      .domain([
        d3.min(allDataPoints,function (d) { return d['performance'] }),
        d3.max(allDataPoints,function (d) { return d['performance'] })
        ])
      .range([h,0])

    // d3.select(`#lambdactooltip-${_this.props.compatibilityScoreType}`).remove();
    // var tooltip = body.append("div")
    //   .attr("class", "lambdactooltip")
    //   .attr("id", `lambdactooltip-${_this.props.compatibilityScoreType}`)
    //   .style("position", "absolute");
    var tooltip = d3.select(`#lambdactooltip-${_this.props.compatibilityScoreType}`);

    // SVG
    d3.select(`#${this.props.compatibilityScoreType}`).remove();
    var svg = body.append('svg')
        .attr('id', this.props.compatibilityScoreType)
        .attr('height',h + margin.top + margin.bottom)
        .attr('width',w + margin.left + margin.right)
      .append('g')
        .attr('transform','translate(' + margin.left + ',' + margin.top + ')')
    // X-axis
    var xAxis = d3.axisBottom()
      .scale(xScale)
      .tickFormat(formatPercent)
      .ticks(5);

    // Y-axis
    var yAxis = d3.axisLeft()
      .scale(yScale)
      .tickFormat(formatPercent)
      .ticks(5);

    // X-axis
    svg.append('g')
        .attr('class','axis')
        .attr('id','xAxis')
        .attr('transform', 'translate(0,' + h + ')')
        .call(xAxis)
      .append('text')
        .attr('id','xAxisLabel')
        .attr('y', 25)
        .attr('x',w/2)
        .attr('dy','.71em')
        .style('text-anchor','end')
        .text(this.props.compatibilityScoreType.toUpperCase())
        .attr("font-family", "sans-serif")
        .attr("font-size", "20px")
        .attr("fill", "black");

    // Y-axis
    svg.append('g')
        .attr('class','axis')
        .attr('id','yAxis')
        .call(yAxis)
      .append('text')
        .attr('id', 'yAxisLabel')
        .attr('transform','rotate(-90)')
        .attr('x',-h/2)
        .attr('y',-50)
        .attr('dy','.71em')
        .style('text-anchor','end')
        .text('Performance')
        .attr("font-family", "sans-serif")
        .attr("font-size", "20px")
        .attr("fill", "black");

    function drawCircles() {
      // var allDataPoints = [];
      // if (_this.state.training && _this.state.newError) {
      //   allDataPoints = allDataPoints.concat(data.filter(d => (d["training"] && d["new-error"])));
      // }

      // if (_this.state.training && _this.state.strictImitation) {
      //   allDataPoints = allDataPoints.concat(data.filter(d => (d["training"] && d["strict-imitation"])));
      // }

      // if (_this.state.testing && _this.state.newError) {
      //   allDataPoints = allDataPoints.concat(data.filter(d => (d["testing"] && d["new-error"])));
      // }

      // if (_this.state.testing && _this.state.strictImitation) {
      //   allDataPoints = allDataPoints.concat(data.filter(d => (d["testing"] && d["strict-imitation"])));
      // }

      var circles = svg.selectAll('circle')
          .data(allDataPoints)
          .enter()
        .append('circle')
          .attr('cx',function (d) { return xScale(d[_this.props.compatibilityScoreType]) })
          .attr('cy',function (d) { return yScale(d['performance']) })
          .attr('r','4')
          .attr('stroke','black')
          .attr('stroke-width',1)
          .attr('fill',function (d,i) {
            if (d["training"] && d["new-error"]) {
              return colorMap["training"]["new-error"];
            } else if (d["training"] && d["strict-imitation"]) {
              return colorMap["training"]["strict-imitation"];
            } else if (d["testing"] && d["new-error"]) {
              return colorMap["testing"]["new-error"];
            } else if (d["testing"] && d["strict-imitation"]) {
              return colorMap["testing"]["strict-imitation"];
            }
          })
          .on('mouseover', function (d) {
            d3.select(this)
              .transition()
              .duration(500)
              .attr('r',8)
              .attr('stroke-width',3);

            tooltip.text(d["lambda_c"].toFixed(2))
              .style("opacity", 0.8);
          })
          .on('mouseout', function () {
            d3.select(this)
              .transition()
              .duration(500)
              .attr('r',4)
              .attr('stroke-width',1);

            tooltip.style("opacity", 0);
            tooltip.text("");
          })
          .on("mousemove", function() {
            var scatterPlot = document.getElementById(`scatterplot-${_this.props.compatibilityScoreType}`);
            var coords = d3.mouse(scatterPlot);
            tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2}px`)
              .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
          })
          .on('click', (d, i) => {
            _this.props.getModelEvaluationData(d["datapoint_index"]);
          });
    }

    drawCircles();
  }

  render() {
    return (
      <div className="plot" ref={this.node} id={`scatterplot-${this.props.compatibilityScoreType}`}>
        <div className="tooltip" id={`lambdactooltip-${this.props.compatibilityScoreType}`} />
      </div>
    );
  }
}
export default PerformanceCompatibility;
