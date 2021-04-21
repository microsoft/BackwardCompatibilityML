// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React from 'react'
import { Slider } from '@fluentui/react';

interface LambdaSliderProps {
}

const LambdaSlider: React.FunctionComponent<LambdaSliderProps> = () => {
  console.log("Rendering LambdaSlider");
  return <Slider
    label="λ value"
    ranged
    min={0}
    max={1}
    defaultValue={1}
    defaultLowerValue={0}
    step={0.1}
  />
}

export default LambdaSlider