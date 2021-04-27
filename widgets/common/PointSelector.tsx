// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React from 'react'

type LegendProps = {
  testing: boolean,
  training: boolean,
  newError: boolean,
  strictImitation: boolean
}

const PointSelector: React.FunctionComponent<LegendProps> = ({testing, training, newError, strictImitation}) => {
  const legendEntries: Array<JSX.Element> = [];

  if (training) {
    if (newError) {
      legendEntries.push(
        <div className="point-legend-entry">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="8" cy="8" r="7" fill="#76C5FF" stroke="black"/>
          </svg>
          New Error - Training
        </div>
      );
    }
    if (strictImitation) {
      legendEntries.push(
        <div className="point-legend-entry">
          <svg width="17" height="15" viewBox="0 0 17 15" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M8.5 1L15.8612 13.75H1.13878L8.5 1Z" fill="#76C5FF" stroke="black"/>
          </svg>
          Strict Imitation - Training
        </div>
      );
    }
  }

  if (testing) {
    if (newError) {
      legendEntries.push(
        <div className="point-legend-entry">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="8" cy="8" r="7" fill="#F551B3" stroke="black"/>
          </svg>
          New Error - Testing
        </div>
      );
    }
    if (strictImitation) {
      legendEntries.push(
        <div className="point-legend-entry">
          <svg width="17" height="15" viewBox="0 0 17 15" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M8.5 1L15.8612 13.75H1.13878L8.5 1Z" fill="#F551B3" stroke="black"/>
          </svg>
          Strict Imitation - Testing
        </div>
      );
    }
  }

  return (
    <div className="point-selector">
      <div className="point-legend-entry-container">
        {legendEntries}
      </div>
    </div>
  )
}

export default PointSelector;