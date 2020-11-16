import React from 'react'
import { render, fireEvent, waitFor, screen, getByTestId, cleanup } from '@testing-library/react'
import '@testing-library/jest-dom'
import { TestScheduler } from 'jest'
import { act } from 'react-dom/test-utils'
import { unmountComponentAtNode } from 'react-dom'

import DataSelector from '@App/DataSelector'
import ErrorInstancesTable from '@App/ErrorInstancesTable'
import IntersectionBetweenModelErrors from '@App/IntersectionBetweenModelErrors'

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

test('ErrorInstancesTable null props', () => {
    render(<ErrorInstancesTable
        selectedDataPoint={null}
        pageSize={null}
        filterInstances={null}
        />);

    let nextButton = screen.getByLabelText("next error page");
    let previousButton = screen.getByLabelText("previous error page");
    let pageNum = screen.getByLabelText("error page number");

    expect(pageNum).toBe(1);
    fireEvent.click(nextButton);
    fireEvent.click(previousButton);

    //TODO: Expect not to crash. How do we detect a crash?
});

test('ErrorInstancesTable pagination', () => {
    render(<ErrorInstancesTable
        selectedDataPoint={{error_instances: Array(100)}}
        pageSize={10}
        filterInstances={null}
        />);

    let nextButton = screen.getByLabelText("next error page");
    let previousButton = screen.getByLabelText("previous error page");
    let pageNum = screen.getByLabelText("error page number");

    expect(pageNum).toBe(1);
    fireEvent.click(nextButton);
    expect(pageNum).toBe(2);
    fireEvent.click(previousButton);
    expect(pageNum).toBe(1);

    for (let i=0; i<9; i++) {
        fireEvent.click(nextButton);
    }

    expect(pageNum).toBe(10);
    fireEvent.click(nextButton);
    expect(pageNum).toBe(10);
    fireEvent.click(previousButton);
    expect(pageNum).toBe(9);
    fireEvent.click(nextButton);
    expect(pageNum).toBe(10);

    for (let i=0; i<9; i++) {
        fireEvent.click(previousButton);
    }

    expect(pageNum).toBe(1);
    fireEvent.click(previousButton);
    expect(pageNum).toBe(1);
    fireEvent.click(nextButton);
    expect(pageNum).toBe(2);
});

test('Venn diagram null props', () => {
    render(<IntersectionBetweenModelErrors selectedDataPoint={null} filterByInstanceIds={null}/>);
    // TODO: Expect that the render worked
});

test('Venn diagram click progress', () => {
    render(<IntersectionBetweenModelErrors selectedDataPoint={null} filterByInstanceIds={null}/>);
});

test('Venn diagram click regress', () => {
    render(<IntersectionBetweenModelErrors selectedDataPoint={null} filterByInstanceIds={null}/>);
});

test('Venn diagram disjoint errors', () => {
    render(<IntersectionBetweenModelErrors selectedDataPoint={null} filterByInstanceIds={null}/>);
});

test('Venn diagram identical errors', () => {
    render(<IntersectionBetweenModelErrors selectedDataPoint={null} filterByInstanceIds={null}/>);
});