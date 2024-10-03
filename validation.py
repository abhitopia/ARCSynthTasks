# validation.py
def inputs_changed(task):
    """
    Validation function to check if any input has changed after transformation.
    Returns True if inputs have changed, False otherwise.
    """
    for (original_input, transformed_input) in task.train + task.test:
        if original_input != transformed_input:
            return True  # Inputs have changed
    return False  # Inputs have not changed

def output_size_within_limits(task, max_size=(30, 30)):
    """
    Validation function to check if the size of outputs is within specified limits.
    Returns True if all outputs are within limits, False otherwise.
    """
    max_rows, max_cols = max_size
    for (_, transformed_input) in task.train + task.test:
        num_rows = len(transformed_input)
        num_cols = len(transformed_input[0]) if num_rows > 0 else 0
        if num_rows > max_rows or num_cols > max_cols:
            return False  # Output size exceeds limits
    return True  # All outputs are within limits``