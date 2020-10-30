import React from 'react'
import { render, fireEvent, waitFor, screen, getByTestId, cleanup } from '@testing-library/react'
import '@testing-library/jest-dom'
import { TestScheduler } from 'jest'
import { act } from 'react-dom/test-utils'
import { unmountComponentAtNode } from 'react-dom'

import DataSelector from '@App/DataSelector'

afterEach(cleanup);

function testCheckbox(label: string): void {
    expect(screen.getByLabelText(label)).toBeChecked();
    fireEvent.click(screen.getByLabelText(label));
    expect(screen.getByLabelText(label)).not.toBeChecked();
}

test('DataSelector single checkboxes', () => {
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

    testCheckbox("training");
    testCheckbox("testing");
    testCheckbox("new error");
    testCheckbox("strict imitation");

    expect(trainingToggleCount).toBe(1);
    expect(testingToggleCount).toBe(1);
    expect(newErrorToggleCount).toBe(1);
    expect(strictImitationToggleCount).toBe(1);
});

test('DataSelector select all checkboxes', () => {
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

    testCheckbox("select all datasets");
    testCheckbox("select all dissonance");
    expect(screen.getByLabelText("training")).not.toBeChecked();
    expect(screen.getByLabelText("testing")).not.toBeChecked();
    expect(screen.getByLabelText("new error")).not.toBeChecked();
    expect(screen.getByLabelText("strict imitation")).not.toBeChecked();
});