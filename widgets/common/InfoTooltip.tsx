// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { FunctionComponent } from "react";
import ReactDOM from "react-dom";
import { IconButton } from '@fluentui/react/lib/Button';
import {
  TooltipHost,
  TooltipDelay,
  DirectionalHint,
} from '@fluentui/react';
import { useId } from '@fluentui/react-hooks';

type InfoTooltipProps = {
  message: string
  direction: DirectionalHint
}

export const InfoTooltip: FunctionComponent<InfoTooltipProps> = ({ message, direction }) => {
  // Use useId() to ensure that the ID is unique on the page.
  // (It's also okay to use a plain string and manually ensure uniqueness.)
  const tooltipId = useId('tooltip');

  return (
    <TooltipHost
      delay={TooltipDelay.zero}
      id={tooltipId}
      directionalHint={direction}
      content={message}
      styles={{root: {width: "24px", height: "18px"}}}
    >
      <IconButton iconProps={{ iconName: 'Info' }}
                  aria-describedby={tooltipId}
                  styles={{icon: {fontSize: '18px', color: "black"}, root: {width: "24px", height: "18px"}}}/>
    </TooltipHost>
  );
};