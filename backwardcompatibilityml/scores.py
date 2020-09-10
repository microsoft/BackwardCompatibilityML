# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def trust_compatibility_score(h1_output_labels, h2_output_labels, expected_labels):
    """
    The fraction of instances labeled correctly by both h1 and h2
    out of the total number of instances labeled correctly by h1.

    Args:
        h1_output_labels: A list of the labels outputted by the model h1.
        h2_output_labels: A list of the labels output by the model h2.
        expected_labels: A list of the corresponding ground truth target labels.

    Returns:
        If h1 has any errors, then we return the trust compatibility score of h2 with respect to h1.
        If h1 has no errors then we return 0.
    """
    h1_correct_count = 0
    h1h2_correct_count = 0
    for i in range(len(expected_labels)):
        h1_label = h1_output_labels[i]
        h2_label = h2_output_labels[i]
        expected_label = expected_labels[i]
        if h1_label == expected_label:
            h1_correct_count = h1_correct_count + 1

        if h1_label == expected_label and h2_label == expected_label:
            h1h2_correct_count = h1h2_correct_count + 1

    if h1_correct_count > 0:
        return (h1h2_correct_count / h1_correct_count)

    return 0


def error_compatibility_score(h1_output_labels, h2_output_labels, expected_labels):
    """
    The fraction of instances labeled incorrectly by h1 and h2
    out of the total number of instances labeled incorrectly by h1.

    Args:
        h1_output_labels: A list of the labels outputted by the model h1.
        h2_output_labels: A list of the labels output by the model h2.
        expected_labels: A list of the corresponding ground truth target labels.

    Returns:
        If h1 has any errors, then we return the error compatibility score of h2 with respect to h1.
        If h1 has no errors then we return 0.
    """
    h1_error_count = 0
    h1h2_error_count = 0
    for i in range(len(expected_labels)):
        h1_label = h1_output_labels[i]
        h2_label = h2_output_labels[i]
        expected_label = expected_labels[i]
        if h1_label != expected_label:
            h1_error_count = h1_error_count + 1

        if h1_label != expected_label and h2_label != expected_label:
            h1h2_error_count = h1h2_error_count + 1

    if h1_error_count > 0:
        return (h1h2_error_count / h1_error_count)

    return 0
