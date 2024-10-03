from dataclasses import dataclass, field
import inspect
from itertools import product
import json
from pathlib import Path
from typing import List, Optional, Tuple
from .dsl import Grid, fill
from . import verifiers


def get_verifiers():
    # Retrieve all functions defined in verifiers.py
    functions_dict = {
        name.split('_')[-1]: func
        for name, func in inspect.getmembers(verifiers, inspect.isfunction)
        if inspect.getmodule(func) == verifiers
    }
    return functions_dict

def format_task(task: dict) -> dict:
    def format_example(example: dict) -> dict:
        def format_grid(grid: List[List[int]]) -> Grid:
            return tuple(tuple(row) for row in grid)
        return {
            'input': format_grid(example['input']),
            'output': format_grid(example['output'])
        }

    return {
        'train': [format_example(example) for example in task['train']],
        'test': [format_example(example) for example in task['test']]
    }

def fix_bugs(dataset: dict) -> None:
    """
    fixes bugs in the original ARC training dataset
    """
    dataset['a8d7556c']['train'][2]['output'] = fill(dataset['a8d7556c']['train'][2]['output'], 2, {(8, 12), (9, 12)})
    dataset['6cf79266']['train'][2]['output'] = fill(dataset['6cf79266']['train'][2]['output'], 1, {(6, 17), (7, 17), (8, 15), (8, 16), (8, 17)})
    dataset['469497ad']['train'][1]['output'] = fill(dataset['469497ad']['train'][1]['output'], 7, {(5, 12), (5, 13), (5, 14)})
    dataset['9edfc990']['train'][1]['output'] = fill(dataset['9edfc990']['train'][1]['output'], 1, {(6, 13)})
    dataset['e5062a87']['train'][1]['output'] = fill(dataset['e5062a87']['train'][1]['output'], 2, {(1, 3), (1, 4), (1, 5), (1, 6)})
    dataset['e5062a87']['train'][0]['output'] = fill(dataset['e5062a87']['train'][0]['output'], 2, {(5, 2), (6, 3), (3, 6), (4, 7)})

@dataclass
class InputTask:
    task_id: str
    train: List[Grid] = field(default_factory=list)
    test: List[Grid] = field(default_factory=list)

def load_input_tasks(include_outputs=False):
    dataset = dict()
    for task_json in Path('/Users/abhishekaggarwal/synced_repos/ARCSynthTasks/data/arc_original/').rglob('*.json'):
        task_id = task_json.stem
        task = json.load(task_json.open())
        task = format_task(task)
        dataset[task_id] = task

    fix_bugs(dataset)

    input_tasks = []

    for key, task in dataset.items():
        # Create InputTask from inputs
        input_task_inputs = InputTask(f"{key}_I")
        for example in task['train']:
            input_task_inputs.train.append(tuple(map(tuple, example['input'])))
        for example in task['test']:
            input_task_inputs.test.append(tuple(map(tuple, example['input'])))
        input_tasks.append(input_task_inputs)

        if not include_outputs:
            continue
        # Create InputTask from outputs
        input_task_outputs = InputTask(f"{key}_O")
        for example in task['train']:
            input_task_outputs.train.append(tuple(map(tuple, example['output'])))
        for example in task['test']:
            input_task_outputs.test.append(tuple(map(tuple, example['output'])))
        input_tasks.append(input_task_outputs)

    return input_tasks


@dataclass
class Verifier:
    name: str
    verifier: callable
    composed_of: List[str] = field(default_factory=list)


def compose_verifiers(verifiers_list):
    def composed(input_grid):
        result = input_grid
        for verifier in verifiers_list:
            result = verifier(result)
        return result
    return composed


def create_compound_verifier(verifiers, verifier_combination):
    verifier_name = '_'.join(verifier_combination)
    compound_verifier_fn =  compose_verifiers([verifiers[verifier_key] for verifier_key in verifier_combination])
    compound_verifier = Verifier(
        name=verifier_name,
        verifier=compound_verifier_fn,
        composed_of=list(verifier_combination)
    )
    return compound_verifier

def get_compound_verfiers(verifiers, max_length):
    verifier_keys = list(verifiers.keys())
    all_combinations = []
    for r in range(1, max_length + 1):
        combinations_r = product(verifier_keys, repeat=r)
        all_combinations.extend(combinations_r)

    compound_verifiers = []
    for combination in all_combinations:
        compound_verifiers.append(create_compound_verifier(verifiers, combination))
    return compound_verifiers

@dataclass
class SyntheticTask:
    task_id: str  # This should be {input_task.task_id}_{verifier_id(s)}
    train: List[Tuple[Grid, Grid]] 
    test: List[Tuple[Grid, Grid]]

def create_synthetic_task(verifier, input_task, new_task_id) -> Optional[SyntheticTask]:    
    train_data = []
    for original_input in input_task.train:
        try:
            transformed_input = verifier(original_input)
            if transformed_input is None:
                raise ValueError("Verifier returned None")
            train_data.append((original_input, transformed_input))
        except Exception as e:
            print(f"Error applying verifier on training data: {e}")
            return None  # Return None to indicate failure

    test_data = []
    for original_input in input_task.test:
        try:
            transformed_input = verifier(original_input)
            if transformed_input is None:
                raise ValueError("Verifier returned None")

            test_data.append((original_input, transformed_input))
        except Exception as e:
            print(f"Error applying verifier on test data: {e}")
            return None  # Return None to indicate failure

    return SyntheticTask(task_id=new_task_id,
                        train=train_data,
                        test=test_data)


class Validator:
    def __init__(self):
        self.validation_funcs = [
            self.inputs_changed,
            self.output_size_within_limits
        ]

    def __call__(self, task: SyntheticTask) -> bool:
        # Apply all validation functions
        for validate in self.validation_funcs:
            if not validate(task):
                print(f"Task {task.task_id} failed validation: {validate.__name__}")
                return False  # Task fails validation
        return True
    
    @staticmethod
    def inputs_changed(task):
        """
        Validation function to check if any input has changed after transformation.
        Returns True if inputs have changed, False otherwise.
        """
        for (original_input, transformed_input) in task.train + task.test:
            if original_input != transformed_input:
                return True  # Inputs have changed
        return False  # Inputs have not changed

    @staticmethod
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
        return True  # All outputs are within limits`


def save_task(task: SyntheticTask, file_path: Path):
    with open(file_path, 'w') as f:
        # Convert Grids to lists for JSON serialization
        task_dict = {
            'task_id': task.task_id,
            'train': [
                (
                    [list(row) for row in orig_inp],
                    [list(row) for row in trans_inp]
                )
                for (orig_inp, trans_inp) in task.train
            ],
            'test': [
                (
                    [list(row) for row in orig_inp],
                    [list(row) for row in trans_inp]
                )
                for (orig_inp, trans_inp) in task.test
            ]
        }
        json.dump(task_dict, f, indent=4)

# def process_input_task_verifier(input_task: InputTask, verifier: Verifier, validator: Validator):
#     new_task_id = f"{input_task.task_id}_{verifier.name}"

#     # Construct the directory path
#     dir_path = Path(directory)/verifier.name
#     dir_path.mkdir(parents=True, exist_ok=True)

#     file_path = dir_path/f'{new_task_id}.json'

#     if file_path.exists():
#         print(f"Task {new_task_id} already exists.")
#         return

#     synthetic_task = create_synthetic_task(verifier, input_task, new_task_id)
#     if synthetic_task is None:
#         print(f"Verifier {verifier_name} failed on some inputs.")
#         return

#     if not validate_task(synthetic_task, validation_funcs):
#         return
    
#     save_task(synthetic_task, file_path)


## Validation functions
