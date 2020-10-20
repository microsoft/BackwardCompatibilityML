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

test('RawValues renders', () => {
    act(() => {
        render(<RawValues />);
    });
    expect(screen.getByText("Raw Values Table goes here")).toBeInTheDocument();
});

