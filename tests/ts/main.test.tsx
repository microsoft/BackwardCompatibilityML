import React from 'react'
/*
import {rest} from 'msw'
import { setupServer } from 'msw/node'
*/
import { render, fireEvent, waitFor, screen, getByTestId, cleanup } from '@testing-library/react'
import '@testing-library/jest-dom'
import { TestScheduler } from 'jest'
import { act } from 'react-dom/test-utils'
import { unmountComponentAtNode } from 'react-dom'

import RawValues from '@App/RawValues'
import DataSelector from '@App/DataSelector'

afterEach(cleanup);

test('RawValues renders', () => {
    render(<RawValues data={{}}/>);
    expect(screen.getByText("Raw Values Table goes here")).toBeInTheDocument();
});

test('DataSelector expected behavior', () => {
    let trainingToggleCount = 0;
    let testingToggleCount = 0;
    let newErrorToggleCount = 0;
    let strictImitationToggleCount = 0;

    render(<DataSelector 
        toggleTraining={() => trainingToggleCount++}
        toggleTesting={() => testingToggleCount++}
        toggleNewError={() => newErrorToggleCount++}
        toggleStrictImitation={() => strictImitationToggleCount++}
        />);

    let testCheckboxes = (label) => {
        expect(screen.getByLabelText(label)).toBeChecked();
        fireEvent.click(screen.getByLabelText(label));
        expect(screen.getByLabelText(label)).not.toBeChecked();
    };

    testCheckboxes("training");
    testCheckboxes("testing");
    testCheckboxes("new error");
    testCheckboxes("strict imitation");

    expect(trainingToggleCount).toBe(1);
    expect(testingToggleCount).toBe(1);
    expect(newErrorToggleCount).toBe(1);
    expect(strictImitationToggleCount).toBe(1);
});

