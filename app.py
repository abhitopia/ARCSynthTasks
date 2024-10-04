from pathlib import Path
import queue
import time
import streamlit as st
import json
from arckit.vis import draw_task
from src.utils import get_verifiers, load_task_inputs, save_task
from src.sampler import TaskSampler
from src.synthesizer import TaskSynthesizer
from src.validator import Validator
import threading
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


# CONFIG
INCLUDE_OUTPUTS = False
SCORE_FILE = 'scores.json'
KEPT_TASK_DIR = Path('data/synthetic_tasks')
NUM_VERIFIERS = 1
BUFFER_SIZE = 5  # Number of tasks to keep in the buffer

@st.cache_resource
def get_task_buffer():
    """
    Initializes and returns a thread-safe queue to act as the task buffer.
    """
    return queue.Queue(maxsize=BUFFER_SIZE)


@st.cache_resource
def initialize_resources():
    VERIFIERS = get_verifiers()
    TASK_INPUTS = load_task_inputs(include_outputs=INCLUDE_OUTPUTS)
    sampler = TaskSampler(
        input_tasks=list(TASK_INPUTS.keys()),
        verifiers=list(VERIFIERS.keys()), 
        save_path=SCORE_FILE
    )
    synthesizer = TaskSynthesizer(VERIFIERS, TASK_INPUTS)
    validator = Validator()
    return sampler, synthesizer, validator


# Initialize SAMPLER, SYNTHESIZER, VALIDATOR
SAMPLER, SYNTHESIZER, VALIDATOR = initialize_resources()

# Initialize session state variables
if 'task_history' not in st.session_state:
    st.session_state.task_history = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = -1
if 'current_task' not in st.session_state:
    st.session_state.current_task = None
if 'kept_tasks' not in st.session_state:
    st.session_state.kept_tasks = []
if 'action' not in st.session_state:
    st.session_state.action = None  # To track user actions
if 'selected_kept_task_index' not in st.session_state:
    st.session_state.selected_kept_task_index = None  # Index of the selected kept task
if 'last_unactioned_task_index' not in st.session_state:
    st.session_state.last_unactioned_task_index = -1  # To track the last unactioned task


# Initialize buffer-related variables using the cached task buffer
task_buffer = get_task_buffer()

# Flag to ensure the buffer thread starts only once
if 'buffer_thread_running' not in st.session_state:
    st.session_state.buffer_thread_running = False

def fill_task_buffer():
    """
    Background thread function to continuously fill the task buffer.
    """
    while True:
        if not task_buffer.full():
            try:
                task, components = generate_synthetic_task_until_valid(
                    synthesizer=SYNTHESIZER,
                    sampler=SAMPLER,
                    validator=VALIDATOR,
                    num_verifiers=NUM_VERIFIERS
                )
                task_dict = {
                    'sampled_task': components,
                    'synthetic_task': task,
                    'status': None
                }
                task_buffer.put(task_dict)
                logging.info(f"Task added to buffer. Buffer size: {task_buffer.qsize()}")
            except Exception as e:
                logging.info(f"Error generating task in buffer: {e}")
        else:
            # Buffer is full; wait before trying again
            time.sleep(1)


def generate_synthetic_task_until_valid(synthesizer: TaskSynthesizer, sampler: TaskSampler, validator: Validator, num_verifiers: int, max_tries=20):
    for _ in range(max_tries):
        invalid_task = False
        task_components = sampler.sample_task(num_verifiers)
        try:
            task = synthesizer.generate_synthetic_task(*task_components)
            if task is None or not validator(task):
                invalid_task = True
            draw_task(task, include_test='all')
        except Exception as e:
            logging.warning(f"Error creating task (Expected): {e}, skipping task")
            invalid_task = True

        if invalid_task:
            sampler.update_scores(task_components, feedback='invalid')
        else:
            break
        
    return task, task_components


def visualize_task(synthetic_task):
    st.write("### Task Visualization")
    svg = draw_task(synthetic_task, include_test='all')
    
    # Assuming svg.as_svg() returns the SVG string
    svg_content = svg.as_svg()
    
    # Optionally, remove the XML declaration if present
    if svg_content.strip().startswith('<?xml'):
        svg_content = '\n'.join(svg_content.strip().split('\n')[1:])
    
    # Embed the SVG using st.markdown
    st.markdown(f'<div>{svg_content}</div>', unsafe_allow_html=True)

def save_tasks_and_scores():
    # Save kept tasks
    for task in st.session_state.kept_tasks:
        task_obj = task['synthetic_task']
        save_task(task_obj, KEPT_TASK_DIR / f"{task_obj.id}.json")
    st.sidebar.success('Tasks and scores saved.')

def clear_kept_tasks():
    st.session_state.kept_tasks = []
    st.session_state.selected_kept_task_index = None
    st.session_state.last_unactioned_task_index = st.session_state.current_index
    st.sidebar.success('Kept tasks cleared.')
    st.rerun()  # Updated function call

# Button click handlers
def on_keep_clicked():
    task = st.session_state.current_task
    if task['status'] != 'kept':
        # Update scores
        SAMPLER.update_scores(task['sampled_task'], feedback='keep')
        # Save the kept task
        st.session_state.kept_tasks.append(task)
        # Update task status
        task['status'] = 'kept'
    # Update last unactioned task index
    st.session_state.last_unactioned_task_index = st.session_state.current_index
    # Proceed to next task
    st.session_state.action = 'next_task'
    st.rerun()  # Updated function call

def on_discard_clicked():
    task = st.session_state.current_task
    # If the task was previously kept, remove it from kept_tasks
    if task['status'] == 'kept':
        st.session_state.kept_tasks.remove(task)
    if task['status'] != 'discarded':
        # Update scores
        SAMPLER.update_scores(task['sampled_task'], feedback='discard')
        # Update task status
        task['status'] = 'discarded'
    # Update last unactioned task index
    st.session_state.last_unactioned_task_index = st.session_state.current_index
    # Proceed to next task
    st.session_state.action = 'next_task'
    st.rerun()  # Updated function call

def on_return_to_current_task_clicked():
    # Resume the last unactioned task
    if st.session_state.last_unactioned_task_index != -1:
        st.session_state.selected_kept_task_index = None
        st.session_state.current_index = st.session_state.last_unactioned_task_index
        st.session_state.current_task = st.session_state.task_history[st.session_state.current_index]
    else:
        # If no unactioned task, sample a new one
        sample_next_task()
    st.rerun()  # Updated function call

def main():
    st.title('ARC Task Synthesizer')

    # Start the background buffer thread if not already running
    if not st.session_state.buffer_thread_running:
        buffer_thread = threading.Thread(target=fill_task_buffer, daemon=True)
        buffer_thread.start()
        st.session_state.buffer_thread_running = True

    # Handle actions based on button clicks BEFORE rendering the sidebar
    if st.session_state.action == 'next_task' or st.session_state.current_index == -1 or st.session_state.current_task is None:
        sample_next_task()
        st.session_state.action = None  # Reset action

    # Sidebar: Display kept tasks and options
    st.sidebar.write("## Kept Tasks")
    if st.session_state.kept_tasks:
        st.sidebar.write("### Select a kept task:")
        # List kept tasks as individual buttons
        for idx, task in enumerate(st.session_state.kept_tasks):
            if st.sidebar.button(f"Kept Task {idx +1}", key=f"kept_task_{idx}"):
                st.session_state.selected_kept_task_index = idx
                st.rerun()  # Update the app to show the selected kept task
    else:
        st.sidebar.write("No tasks kept yet.")

    # Buttons in the sidebar
    st.sidebar.write("---")
    if st.sidebar.button('Clear Kept Tasks'):
        clear_kept_tasks()

    if st.sidebar.button('Save'):
        save_tasks_and_scores()

    # Display the current or selected task
    if st.session_state.selected_kept_task_index is not None:
        # Display the selected kept task
        selected_task = st.session_state.kept_tasks[st.session_state.selected_kept_task_index]
        display_task(selected_task, is_kept_task=True)
    else:
        # Display the current task
        display_task(st.session_state.current_task)

def sample_next_task():
    # Sample a new task
    try:
        task = task_buffer.get_nowait()
        logging.info(f"Task retrieved from buffer. Buffer size: {task_buffer.qsize()}")
    except queue.Empty:
        logging.info("Buffer empty. Generating task synchronously.")
        task, components = generate_synthetic_task_until_valid(
            synthesizer=SYNTHESIZER,
            sampler=SAMPLER,
            validator=VALIDATOR,
            num_verifiers=NUM_VERIFIERS
        )
        task = {
            'sampled_task': components,
            'synthetic_task': task,
            'status': None
        }
    # Add to history
    st.session_state.task_history.append(task)
    st.session_state.current_index = len(st.session_state.task_history) - 1
    # Set as current task
    st.session_state.current_task = task
    # Update last unactioned task index
    st.session_state.last_unactioned_task_index = st.session_state.current_index

def display_task(task, is_kept_task=False):
    with st.container():
        # Display the visualization
        visualize_task(task['synthetic_task'])

        # Generate a unique key for buttons based on the task
        key_base = f"task_{id(task)}_"

        # Display status or feedback buttons
        col1, col2 = st.columns(2)

        if is_kept_task:
            # Viewing a kept task
            with col1:
                if st.button('Return to Current Task', key=key_base + 'return_button'):
                    on_return_to_current_task_clicked()
            with col2:
                if st.button('Discard', key=key_base + 'discard_button'):
                    # Remove from kept tasks
                    st.session_state.kept_tasks.remove(task)
                    task['status'] = 'discarded'
                    # Update last unactioned task index
                    st.session_state.last_unactioned_task_index = st.session_state.current_index
                    # Proceed to next task
                    st.session_state.action = 'next_task'
                    st.rerun()  # Updated function call
        else:
            # Viewing the current task
            if task['status'] == 'kept':
                with col1:
                    st.button('Kept', key=key_base + 'kept_button', disabled=True)
                with col2:
                    if st.button('Discard', key=key_base + 'discard_button'):
                        on_discard_clicked()
            else:
                with col1:
                    if st.button('Keep', key=key_base + 'keep_button'):
                        on_keep_clicked()
                with col2:
                    if st.button('Discard', key=key_base + 'discard_button'):
                        on_discard_clicked()

if __name__ == '__main__':
    main()
