import React from 'react'
/*
import {rest} from 'msw'
import { setupServer } from 'msw/node'
*/
import { render, fireEvent, waitFor, screen } from '@testing-library/react'
import '@testing-library/jest-dom/extend-expect'
import RawValues from './widget/RawValues'
import { TestScheduler } from 'jest'
import { act } from 'react-dom/test-utils'
import { unmountComponentAtNode } from 'react-dom'

let container = null;
beforeEach(() => {
    // setup a DOM element as a render target
    container = document.createElement("div");
    document.body.appendChild(container);
})

afterEach(() => {
    // cleanup on exiting
    unmountComponentAtNode(container);
    container.remove();
    container = null;
});

test('RawValues renders', () => {
    act(() => {
        render(<RawValues />, container);
    });
    expect(container.textContent).toBe("Raw Values Table goes here");
});

