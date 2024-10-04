import logging
from arckit import Task
from .utils import compose_verifiers


class TaskSynthesizer:
    def __init__(self, verifiers, task_inputs) -> None:
        self.verifiers = verifiers
        self.task_inputs = task_inputs
        
    def generate_synthetic_task(self, input_id, verifier1_id, verifier2_id=None):
        compoud_verifier = self.verifiers[verifier1_id]
        if verifier2_id is not None:
            compoud_verifier = compose_verifiers([compoud_verifier, self.verifiers[verifier2_id]])

        input_task = self.task_inputs[input_id]

        train_data = []

        task_id = f"{input_id}_{verifier1_id}" if verifier2_id is None else f"{input_id}_{verifier1_id}_{verifier2_id}"

        for original_input in input_task.train:
            try:
                transformed_input = compoud_verifier(original_input)
                assert transformed_input is not None, "Verifier returned None"
                train_data.append({'input': original_input, 'output': transformed_input})
            except Exception as e:
                logging.warning(f"Error applying verifier on training data: {e}")
                return None  # Return None to indicate failure

        test_data = []
        for original_input in input_task.test:
            try:
                transformed_input = compoud_verifier(original_input)
                assert transformed_input is not None, "Verifier returned None"
                test_data.append({'input': original_input, 'output': transformed_input})
            except Exception as e:
                logging.warning(f"Error applying verifier on test data: {e}")
                return None  # Return None to indicate failure
       
        return Task(id=task_id, train=train_data, test=test_data)
    
