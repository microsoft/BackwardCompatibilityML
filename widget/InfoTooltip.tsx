// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import React, { FunctionComponent } from "react";
import ReactDOM from "react-dom";
import { IconButton } from 'office-ui-fabric-react/lib/Button';
import {
  TooltipHost,
  TooltipDelay,
  DirectionalHint,
  ITooltipProps,
  ITooltipHostStyles,
} from 'office-ui-fabric-react/lib/Tooltip';
import { useId } from '@uifabric/react-hooks';

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
    >
      <IconButton iconProps={{ iconName: 'Info' }}
                  aria-describedby={tooltipId}
                  styles={{icon: {fontSize: '18px', color: "black"}}}/>
    </TooltipHost>
  );
};