from dataclasses import dataclass, field
import inspect
import logging
import json
from pathlib import Path
from typing import List, Union
from .dsl import Grid, fill
from . import verifiers
from arckit import Task

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
class TaskInput:
    train: List[Grid] = field(default_factory=list)
    test: List[Grid] = field(default_factory=list)

def load_task_inputs(include_outputs=False):
    dataset = dict()
    for task_json in Path('/Users/abhishekaggarwal/synced_repos/ARCSynthTasks/data/arc_original/').rglob('*.json'):
        task_id = task_json.stem
        task = json.load(task_json.open())
        task = format_task(task)
        dataset[task_id] = task

    fix_bugs(dataset)

    task_inputs = {}

    for key, task in dataset.items():
        # Create InputTask from inputs
        input_id = f"I{key}"
        input_task_inputs = TaskInput()
        for example in task['train']:
            input_task_inputs.train.append(tuple(map(tuple, example['input'])))
        for example in task['test']:
            input_task_inputs.test.append(tuple(map(tuple, example['input'])))
        task_inputs[input_id] = input_task_inputs

        if not include_outputs:
            continue

        input_id = f"O{key}"
        # Create InputTask from outputs
        input_task_outputs = TaskInput()
        for example in task['train']:
            input_task_outputs.train.append(tuple(map(tuple, example['output'])))
        for example in task['test']:
            input_task_outputs.test.append(tuple(map(tuple, example['output'])))
        task_inputs[input_id] = input_task_outputs

    return task_inputs


def compose_verifiers(verifiers_list):
    def composed(input_grid):
        result = input_grid
        for verifier in verifiers_list:
            result = verifier(result)
        return result
    return composed



def save_task(task: Task, path: Union[Path, str]):
    path = Path(path)
    if path.exists():
        logging.info(f"Task {task.id} already exists at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(task.to_dict(), path.open('w'), indent=2)
