import logging
from arckit import Task

class Validator:
    def __init__(self):
        self.validation_funcs = [
            self.inputs_changed,
            self.output_size_within_limits,
            self.not_original_task
            ]

    def __call__(self, task: Task) -> bool:
        # Apply all validation functions
        for validate in self.validation_funcs:
            if not validate(task):
                logging.info(f"Task {task.id} failed validation: {validate.__name__}")
                return False  # Task fails validation
        return True
    
    @staticmethod
    def inputs_changed(task):
        """
        Validation function to check if any input has changed after transformation.
        Returns True if inputs have changed, False otherwise.
        """
        input_changed_array = []
        for (original_input, transformed_input) in task.train + task.test:
            if original_input.shape != transformed_input.shape or (original_input != transformed_input).any():
                input_changed_array.append(True)
            else:
                input_changed_array.append(False)

        return all(input_changed_array)

    @staticmethod
    def output_size_within_limits(task: Task, max_size=(30, 30)):
        """
        Validation function to check if the size of outputs is within specified limits.
        Returns True if all outputs are within limits, False otherwise.
        """
        max_rows, max_cols = max_size
        for (_, transformed_input) in task.train + task.test:
            num_rows, num_cols = transformed_input.shape
            if num_rows > max_rows or num_cols > max_cols:
                return False  # Output size exceeds limits
        return True  # All outputs are within limits`
    

    def not_original_task(self, task: Task) -> bool:
        """
        Prevents the original task from being submitted.
        """
        splited = task.id.split('_')
        input_id = splited[0][1:] if splited[0][0] == 'I' or splited[0][0] == 'O' else splited[0]
        verifiers = splited[1:]
        return input_id not in verifiers



# ADD VALIDATOR ALL OUTPUT CELLS ARE BLACK all across