// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React from 'react'
import { Slider } from '@fluentui/react';

interface LambdaSliderProps {
  setLambdaLowerBound: Function,
  setLambdaUpperBound: Function,
  lambdaLowerBound: number,
  lambdaUpperBound: number
}

const LambdaSlider: React.FunctionComponent<LambdaSliderProps> = ({setLambdaLowerBound, setLambdaUpperBound, lambdaLowerBound, lambdaUpperBound}) => {
  const onChange = (_: unknown, range: [number, number]) => {
    if (range[0] !== lambdaLowerBound) {
      setLambdaLowerBound(range[0]);
    }
    if (range[1] !== lambdaUpperBound) {
      setLambdaUpperBound(range[1]);
    }
  };
  return <Slider
    label="λ value"
    ranged
    min={0}
    max={1}
    defaultValue={1}
    defaultLowerValue={0}
    step={0.1}
    onChange={onChange}
    styles={{root: {marginLeft: "25px"}}}
  />
}

export default LambdaSlider