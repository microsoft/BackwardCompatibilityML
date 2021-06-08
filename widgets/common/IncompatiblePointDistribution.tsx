// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as d3 from "d3";
import { InfoTooltip } from "./InfoTooltip.tsx"
import { DirectionalHint } from '@fluentui/react';
import { PrimaryButton, DefaultButton } from '@fluentui/react/lib/Button';


type IncompatiblePointDistributionState = {
  selectedDataPoint: any,
  page: number
}

type IncompatiblePointDistributionProps = {
  selectedDataPoint: any,
  selectedClass?: number,
  setSelectedClass: any,
  setSelectedChart: any,
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

    var margin = { top: 20, right: 15, bottom: 20, left: 55 }
    var h = 191 - margin.top - margin.bottom
    var w = 363;

    // SVG
    d3.select("#incompatiblepointdistribution").remove();
    var svg = body.append('svg')
        .attr('id', "incompatiblepointdistribution")
        .attr('height',h + margin.top + margin.bottom + 16)
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
        <BarChartSelector selectedChart={"incompatible-points"} setSelectedChart={this.props.setSelectedChart} />
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


type BarChartSelectorProps = {
  selectedChart: string
  setSelectedChart: Function
}


class BarChartSelector extends Component<BarChartSelectorProps, null> {

  render () {

    let getButtons = () => {
      if (this.props.selectedChart == "model-accuracy") {
        return (
          <React.Fragment>
            <PrimaryButton
              text="Model Accuracy"
              onClick={() => this.props.setSelectedChart("model-accuracy")}
              styles={{root: {marginLeft: '2px', marginRight: '2px'} }}
            />
            <DefaultButton
              text="Incompatible Points"
              onClick={() => this.props.setSelectedChart("incompatible-points")}
              styles={{root: {marginLeft: '2px', marginRight: '2px'} }}
            />
          </React.Fragment>
        );
      } else {
        return (
          <React.Fragment>
            <DefaultButton
              text="Model Accuracy"
              onClick={() => this.props.setSelectedChart("model-accuracy")}
              styles={{root: {marginLeft: '2px', marginRight: '2px'} }}
            />
            <PrimaryButton
              text="Incompatible Points"
              onClick={() => this.props.setSelectedChart("incompatible-points")}
              styles={{root: {marginLeft: '2px', marginRight: '2px'}}}
            />
          </React.Fragment>
        );
      }
    }

    return (
      <div className="chart-selector-row">
        {getButtons()}
      </div>
    )
  }
};


type ModelAccuracyByClassState = {
  selectedDataPoint: any,
  page: number
}

type ModelAccuracyByClassProps = {
  selectedDataPoint: any,
  pageSize?: number,
  filterByInstanceIds: any,
  setSelectedModelAccuracyClass: any,
  selectedModelAccuracyClass: any,
  setSelectedChart: any
}

class ModelAccuracyByClass extends Component<ModelAccuracyByClassProps, ModelAccuracyByClassState> {

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
    var h = 177 - margin.top - margin.bottom
    var w = 363;

    if (this.props.selectedDataPoint != null) {
      // SVG
      d3.select("#modelaccuracybyclass").remove();
      var svg = body.append('svg')
          .attr('id', "modelaccuracybyclass")
          .attr('height',h + margin.top + margin.bottom)
          .attr('width',w + margin.left + margin.right)
          .attr('float', 'left');

      // Sort the data into the dataRows based on the ordering of the sorted classes
      var startI = this.state.page * this.props.pageSize;
      var endI = Math.min(startI + this.props.pageSize, this.props.selectedDataPoint.sorted_classes.length);

      var h1DataRows = [];
      var h2DataRows = [];
      for (var i=startI; i < endI; i++) {
        var instanceClass = this.props.selectedDataPoint.sorted_classes[i];
        var h1dataRow = this.props.selectedDataPoint.h1_accuracy_by_class.filter(
          dataDict => (dataDict["class"] == instanceClass)).pop();
        var h2dataRow = this.props.selectedDataPoint.h2_accuracy_by_class.filter(
          dataDict => (dataDict["class"] == instanceClass)).pop();

        if (h1dataRow) {
          h1DataRows.push(h1dataRow);
        } else {
          h1DataRows.push({class: instanceClass, "accuracy": 0.0, incompatibleInstanceIds: []})
        }

        if (h2dataRow) {
          h2DataRows.push(h2dataRow);
        } else {
          h2DataRows.push({class: instanceClass, "accuracy": 0.0, incompatibleInstanceIds: []})
        }
      }

      var xScale = d3.scaleBand().range([0, w]).padding(0.4),
          yScale = d3.scaleLinear().range([h, 0]);

      var g = svg.append("g")
                 .attr("transform", "translate(" + 55 + "," + margin.top + ")");

        xScale.domain(h1DataRows.map(function(d) { return d.class; }));
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
         .attr("x", -50)
         .attr("dy", "-5.1em")
         .attr("text-anchor", "end")
         .attr("fill", "black")
         .text("Model Accuracy");

        g.selectAll(".barh1")
         .data(h1DataRows)
         .enter().append("rect")
         .attr("class", "barh1")
         .attr("x", function(d) { return xScale(d.class); })
         .attr("y", function(d) { return yScale(d.accuracy * 100); })
         .attr("width", xScale.bandwidth()/2)
         .attr("height", function(d) { return h - yScale(d.accuracy * 100); })
         .classed("highlighted-bar", function(d) { return (_this.props.selectedModelAccuracyClass != null) && (d.class == _this.props.selectedModelAccuracyClass.classLabel && _this.props.selectedModelAccuracyClass.modelName == "h1") })
         .on("click", function(d) {
           _this.props.filterByInstanceIds(d.error_instance_ids);
           _this.props.setSelectedModelAccuracyClass("h1", d.class);
         });

        g.selectAll(".barh2")
         .data(h2DataRows)
         .enter().append("rect")
         .attr("class", "barh2")
         .attr("x", function(d) { return xScale(d.class) + xScale.bandwidth()/2; })
         .attr("y", function(d) { return yScale(d.accuracy * 100); })
         .attr("width", xScale.bandwidth()/2)
         .attr("height", function(d) { return h - yScale(d.accuracy * 100); })
         .classed("highlighted-bar", function(d) { return (_this.props.selectedModelAccuracyClass != null) && (d.class == _this.props.selectedModelAccuracyClass.classLabel && _this.props.selectedModelAccuracyClass.modelName == "h2") })
         .on("click", function(d) {
           _this.props.filterByInstanceIds(d.error_instance_ids);
           _this.props.setSelectedModelAccuracyClass("h2", d.class);
         });
    } else {
      // SVG
      d3.select("#modelaccuracybyclass").remove();
      var svg = body.append('svg')
          .attr('id', "modelaccuracybyclass")
          .attr('height', 216)
          .attr('width',w + margin.left + margin.right)
          .attr('float', 'left');
    }
  }

  render() {
    let numClasses = this.props.selectedDataPoint?.sorted_classes?.length ?? 0;
    let numPages = Math.ceil(numClasses/this.props.pageSize) - 1;
    const message = "Displays the distribution of model accuracies per class.​";

    let getClassAccuracyLegend = () => {
      if (this.props.selectedDataPoint != null) {
        return (
          <div className="class-accuracy-legend-row">
            <div className="class-accuracy-legend-row-block">
              <div className="class-accuracy-h1" />
              h1
            </div>
            <div className="class-accuracy-legend-row-block">
              <div className="class-accuracy-h2" />
              h2
            </div>
          </div>
        );
      } else {
        return (<React.Fragment />);
      }
    }

    return (
      <div className="plot plot-distribution">
        <BarChartSelector selectedChart={"model-accuracy"} setSelectedChart={this.props.setSelectedChart} />
        <div className="plot-title-row">
          Model Accuracy by Class
          <InfoTooltip message={message} direction={DirectionalHint.topCenter} />
        </div>
        {getClassAccuracyLegend()}
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


type ClassStatisticsState = {
  selectedPlot: string
}

type ClassStatisticsProps = {
  selectedDataPoint: any,
  selectedClass?: number,
  selectedModelAccuracyClass: any,
  setSelectedClass: any,
  setSelectedModelAccuracyClass: any,
  pageSize?: number,
  filterByInstanceIds: any
}


class ClassStatisticsPanel extends Component<ClassStatisticsProps, ClassStatisticsState> {

  constructor(props) {
    super(props);

    this.state = {
      selectedPlot: "model-accuracy"
    };

    this.selectModelAccuracyPlot = this.selectModelAccuracyPlot.bind(this);
    this.selectIncompatiblePointsPlot = this.selectIncompatiblePointsPlot.bind(this);
    this.setSelectedPlot = this.setSelectedPlot.bind(this);
  }

  selectModelAccuracyPlot() {
    this.setState({
      selectedPlot: "model-accuracy"
    });
  }

  selectIncompatiblePointsPlot() {
    this.setState({
      selectedPlot: "incompatible-points"
    });
  }

  setSelectedPlot(plotName: string) {
    this.setState({
      selectedPlot: plotName
    });
  }

  render() {
    var renderPlot = () => {
      if (this.state.selectedPlot == "model-accuracy") {
        return (
          <ModelAccuracyByClass
            selectedDataPoint={this.props.selectedDataPoint}
            setSelectedModelAccuracyClass={this.props.setSelectedModelAccuracyClass}
            selectedModelAccuracyClass={this.props.selectedModelAccuracyClass}
            filterByInstanceIds={this.props.filterByInstanceIds}
            setSelectedChart={this.setSelectedPlot}
          />
        );
      } else if (this.state.selectedPlot == "incompatible-points") {
        return (
          <IncompatiblePointDistribution
            selectedDataPoint={this.props.selectedDataPoint}
            setSelectedClass={this.props.setSelectedClass}
            selectedClass={this.props.selectedClass}
            filterByInstanceIds={this.props.filterByInstanceIds}
            setSelectedChart={this.setSelectedPlot}
          />
        );
      }
    }

    
    return (
      <div className="plot">
        {renderPlot()}
      </div>
    );
  }
}

export default ClassStatisticsPanel;
