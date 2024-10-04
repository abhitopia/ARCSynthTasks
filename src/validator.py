from arckit import Task

class Validator:
    def __init__(self):
        self.validation_funcs = [
            self.inputs_changed,
            self.output_size_within_limits]

    def __call__(self, task: Task) -> bool:
        # Apply all validation functions
        for validate in self.validation_funcs:
            if not validate(task):
                print(f"Task {task.id} failed validation: {validate.__name__}")
                return False  # Task fails validation
        return True
    
    @staticmethod
    def inputs_changed(task):
        """
        Validation function to check if any input has changed after transformation.
        Returns True if inputs have changed, False otherwise.
        """
        for (original_input, transformed_input) in task.train + task.test:
            
            if original_input.shape != transformed_input.shape:
                return True
            
            if (original_input != transformed_input).any():
                return True  # Inputs have changed
        return False  # Inputs have not changed

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
