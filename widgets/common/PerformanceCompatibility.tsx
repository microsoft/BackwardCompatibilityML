// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";
import { InfoTooltip } from "./InfoTooltip.tsx"
import { DirectionalHint } from '@fluentui/react';


type PerformanceCompatibilityState = {
  data: any,
  h1Performance: any,
  testing: boolean,
  training: boolean,
  newError: boolean,
  strictImitation: boolean,
  selectedDataPoint: any
}

type PerformanceCompatibilityProps = {
  data: any,
  h1Performance: any,
  testing: boolean,
  training: boolean,
  newError: boolean,
  strictImitation: boolean,
  selectedDataPoint: any,
  lambdaLowerBound: number,
  lambdaUpperBound: number,
  compatibilityScoreType: string,
  performanceMetric: string,
  selectDataPoint: (d: any) => void,
  getModelEvaluationData: (evaluationId: number) => void
}


class PerformanceCompatibility extends Component<PerformanceCompatibilityProps, PerformanceCompatibilityState> {
  constructor(props) {
    super(props);

    this.state = {
      data: props.data,
      h1Performance: props.h1Performance,
      testing: props.testing,
      training: props.training,
      newError: props.newError,
      strictImitation: props.strictImitation,
      selectedDataPoint: props.selectedDataPoint
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

    var margin = { top: 15, right: 15, bottom: 20, left: 55 }
    var h = 350 - margin.top - margin.bottom
    var w = 425 - margin.left - margin.right

    var formatPercentX = d3.format('.3f');
    var formatPercentY = d3.format('.2f');
    var colorMap = {
      "training": "rgba(118, 197, 255, 0.15)",
      "testing": "rgba(245, 81, 179, 0.15)"
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

    allDataPoints = allDataPoints.filter(d => d.lambda_c >= this.props.lambdaLowerBound && d.lambda_c <= this.props.lambdaUpperBound);

    var btcValues = allDataPoints.map(d => d["btc"]);
    var becValues = allDataPoints.map(d => d["bec"]);
    var allValues = [].concat(btcValues).concat(becValues);
    var allPerformance = allDataPoints.map(d => d["performance"]);
    var allPerformanceWithH1 = allPerformance.concat(this.props.h1Performance);

    var xScale = d3.scaleLinear()
      .domain([
        d3.min(allValues),
        d3.max(allValues)
        ])
      .range([0,w])
    var yScale = d3.scaleLinear()
      .domain([
        (d3.min(allPerformanceWithH1) - 0.005),
        (d3.max(allPerformanceWithH1) + 0.005)
        ])
      .range([h,0])

    var tooltip = d3.select(`#lambdactooltip-${_this.props.compatibilityScoreType}`);

    // SVG
    d3.select(`#${this.props.compatibilityScoreType}`).remove();
    var svg = body.insert('svg', '.plot-title-row')
        .attr('id', this.props.compatibilityScoreType)
        .attr('height',h + margin.top + margin.bottom)
        .attr('width',w + margin.left + margin.right)
      .append('g')
        .attr('transform','translate(' + margin.left + ',' + margin.top + ')')
    // X-axis
    var xAxis = d3.axisBottom()
      .scale(xScale)
      .tickFormat(formatPercentX)
      .ticks(5);

    // Y-axis
    var yAxis = d3.axisLeft()
      .scale(yScale)
      .tickFormat(formatPercentY)
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
        .attr('x',-h/2+2.5*this.props.performanceMetric.length)
        .attr('y',-50)
        .attr('dy','.71em')
        .style('text-anchor','end')
        .text(this.props.performanceMetric)
        .attr("font-family", "sans-serif")
        .attr("font-size", "20px")
        .attr("fill", "black");

    svg.append("line")
      .attr("x1", xScale(d3.min(allValues) - 0.005))
      .attr("y1", yScale(this.props.h1Performance))
      .attr("x2", xScale(d3.max(allValues)))
      .attr("y2", yScale(this.props.h1Performance))
      .attr("stroke", "#6f27db")
      .attr("stroke-width", "1px")
      .attr("stroke-dasharray", "5,5");

    function drawNewError() {
      const newErrorData = allDataPoints.filter(d =>  d["new-error"]);
      const radius = 8;
      var circles = svg.selectAll('circle')
          .data(newErrorData)
          .enter()
        .append('circle')
          .attr('cx',function (d) { return xScale(d[_this.props.compatibilityScoreType]) })
          .attr('cy',function (d) { return yScale(d['performance']) })
          .attr('r', function(d) {
            var datapointIndex = (_this.props.selectedDataPoint != null)? _this.props.selectedDataPoint.datapoint_index: null;
            if (d.datapoint_index == datapointIndex) {
              return 1.5 * radius;
            } else {
              return radius;
            }
          })
          .attr('stroke','black')
          .attr('stroke-width', function(d) {
            var datapointIndex = (_this.props.selectedDataPoint != null)? _this.props.selectedDataPoint.datapoint_index: null;
            if (d.datapoint_index == datapointIndex) {
              return 3;
            } else {
              return 1;
            }
          })
          .attr('fill',function (d,i) {
            if (d["training"]) {
              return colorMap["training"];
            } else if (d["testing"]) {
              return colorMap["testing"];
            }
          })
          .on('mouseover', function (d) {
            d3.select(this)
              .transition()
              .duration(500)
              .attr('r',1.5*radius)
              .attr('stroke-width',3);

            tooltip.text(`lambda ${d["lambda_c"].toFixed(2)}`)
              .style("opacity", 0.8);
          })
          .on('mouseout', function (d) {
            var datapointIndex = (_this.props.selectedDataPoint != null)? _this.props.selectedDataPoint.datapoint_index: null;
            if (d.datapoint_index != datapointIndex) {
              d3.select(this)
                .transition()
                .duration(500)
                .attr('r',4)
                .attr('stroke-width',1);
             }

            tooltip.style("opacity", 0);
            tooltip.text("");
          })
          .on("mousemove", function() {
            var scatterPlot = document.getElementById(`scatterplot-${_this.props.compatibilityScoreType}`);
            var coords = d3.mouse(scatterPlot);
            tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2 - 40}px`)
              .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
          })
          .on('click', (d, i) => {
            _this.props.getModelEvaluationData(d["datapoint_index"]);
          });
    }

    // Calculates coordinates for the vertices of an equilateral triangle
    // with centroid coordinates cx,cy and scaled by size
    function trianglePoints(cx: number, cy: number, size: number) {
      const p1 = (xScale(cx) - Math.sqrt(3)*size) + ',' + (yScale(cy) + size);
      const p2 = (xScale(cx) + Math.sqrt(3)*size) + ',' + (yScale(cy) + size);
      const p3 = xScale(cx) + ',' + (yScale(cy) - 2*size);
      // x,y points delimited by spaces
      return p1 + ' ' + p2 + ' ' + p3 + ' ' + p1;
    }

    function drawStrictImitation() {
      const strictImitationData = allDataPoints.filter(d =>  d["strict-imitation"]);
      const getPoints = (d: any, size: number) => trianglePoints(d[_this.props.compatibilityScoreType], d['performance'], size);
      svg.selectAll('polyline')
          .data(strictImitationData)
          .enter()
        .append('polyline')
          .attr('points', d => getPoints(d, 4))
          .attr('stroke','black')
          .attr('stroke-width', function(d) {
            var datapointIndex = (_this.props.selectedDataPoint != null)? _this.props.selectedDataPoint.datapoint_index: null;
            if (d.datapoint_index == datapointIndex) {
              return 3;
            } else {
              return 1;
            }
          })
          .attr('fill',function (d,i) {
            if (d["training"]) {
              return colorMap["training"];
            } else if (d["testing"]) {
              return colorMap["testing"];
            }
          })
          .on('mouseover', function (d) {
            d3.select(this)
              .transition()
              .duration(500)
              .attr('points', d => getPoints(d, 8))
              .attr('stroke-width',3);

            tooltip.text(`lambda ${d["lambda_c"].toFixed(2)}`)
              .style("opacity", 0.8);
          })
          .on('mouseout', function (d) {
            var datapointIndex = (_this.props.selectedDataPoint != null)? _this.props.selectedDataPoint.datapoint_index: null;
            if (d.datapoint_index != datapointIndex) {
              d3.select(this)
                .transition()
                .duration(500)
                .attr('points', d => getPoints(d, 4))
                .attr('stroke-width',1);
             }

            tooltip.style("opacity", 0);
            tooltip.text("");
          })
          .on("mousemove", function() {
            var scatterPlot = document.getElementById(`scatterplot-${_this.props.compatibilityScoreType}`);
            var coords = d3.mouse(scatterPlot);
            tooltip.style("left", `${coords[0] - (margin.left + margin.right)/2 - 40}px`)
              .style("top", `${coords[1] - (margin.top + margin.bottom)/2}px`);
          })
          .on('click', (d, i) => {
            _this.props.getModelEvaluationData(d["datapoint_index"]);
          });
    }

    drawNewError();
    drawStrictImitation();
  }


  render() {
    let classes = `plot plot-${this.props.compatibilityScoreType}`;
    let title = this.props.compatibilityScoreType.toUpperCase();
    let message = "UNDEFINED";

    if (title == "BTC") {
      message = "Backward Trust Compatibility (BTC) describes the percentage of trust preserved after an update."
    } else if (title == "BEC") {
      message = "Backward Error Compatibility (BEC) captures the probability that a mistake made by the newly trained model is not new.​"
    }

    return (
      <React.Fragment>
        <div className={classes} ref={this.node} id={`scatterplot-${this.props.compatibilityScoreType}`}>
          <div className="tooltip" id={`lambdactooltip-${this.props.compatibilityScoreType}`} />
          <div className="plot-title-row">
            {title}
            <InfoTooltip message={message} direction={DirectionalHint.bottomCenter} />
          </div>
        </div>
      </React.Fragment>
    );
  }

}
export default PerformanceCompatibility;
