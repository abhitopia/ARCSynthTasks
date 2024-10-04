import logging
from pathlib import Path
import tempfile
import numpy as np
import json
import os

class TaskSampler:
    def __init__(self, input_tasks, verifiers, save_path='scores.json'):
        """
        Initializes the TaskGenerator with unnormalized scores.

        Parameters:
        - input_tasks: List of input task identifiers.
        - verifiers: List of verifier identifiers.
        - save_path: Path to save/load the scores.
        """
        self.input_tasks = input_tasks
        self.verifiers = verifiers
        self.save_path = save_path
        self.backup_path = Path(self.save_path).with_suffix('.backup.json')


        # Initialize scores
        self.scores_input_task = {task: 1.0 for task in self.input_tasks}
        self.scores_verifier_given_input = {
            task: {verifier: 1.0 for verifier in self.verifiers}
            for task in self.input_tasks
        }

        # Prevent two verifiers from being the same
        self.scores_verifier2_given_verifier1 = {
            verifier1: {verifier2: 1.0 if verifier1 != verifier2 else 0.0 for verifier2 in self.verifiers}
            for verifier1 in self.verifiers
        }

        # Load saved scores if available
        if os.path.exists(self.save_path):
            self.load_scores()
            # Merge current identifiers with loaded scores
            self._merge_scores()

    def _merge_scores(self):
        """
        Merges loaded scores with current identifiers.
        Assigns average scores to new identifiers per Y.
        """
        # Handle input tasks
        existing_tasks = set(self.scores_input_task.keys())
        current_tasks = set(self.input_tasks)

        # Calculate average score for input tasks
        if existing_tasks:
            avg_score_input_task = np.mean(list(self.scores_input_task.values()))
        else:
            avg_score_input_task = 1.0

        for task in current_tasks:
            if task not in self.scores_input_task:
                # Assign average score to new tasks
                self.scores_input_task[task] = avg_score_input_task

        # Handle verifiers
        current_verifiers = set(self.verifiers)

        # Update scores_verifier_given_input
        for task in current_tasks:
            if task not in self.scores_verifier_given_input:
                # No previous data for this task
                avg_score_verifier = 1.0
                self.scores_verifier_given_input[task] = {
                    verifier: avg_score_verifier for verifier in self.verifiers
                }
            else:
                existing_verifier_scores = self.scores_verifier_given_input[task]
                # Compute average score for existing verifiers under this task
                avg_score_verifier = np.mean(list(existing_verifier_scores.values()))
                for verifier in current_verifiers:
                    if verifier not in existing_verifier_scores:
                        self.scores_verifier_given_input[task][verifier] = avg_score_verifier


        # Update scores_verifier2_given_verifier1
        for verifier1 in current_verifiers:
            if verifier1 not in self.scores_verifier2_given_verifier1:
                # No previous data for this verifier1
                self.scores_verifier2_given_verifier1[verifier1] = {
                    verifier2: (1.0 if verifier2 != verifier1 else 0.0) for verifier2 in self.verifiers
                }
            else:
                existing_verifier2_scores = self.scores_verifier2_given_verifier1[verifier1]
                # Compute average score for existing verifier2 under this verifier1 (excluding self)
                non_self_scores = [
                    score for verifier2, score in existing_verifier2_scores.items() if verifier2 != verifier1
                ]
                avg_score_verifier2 = np.mean(non_self_scores) if non_self_scores else 1.0
                for verifier2 in current_verifiers:
                    if verifier2 not in existing_verifier2_scores:
                        # Ensure that score is zero when verifier2 == verifier1
                        self.scores_verifier2_given_verifier1[verifier1][verifier2] = (
                            avg_score_verifier2 if verifier2 != verifier1 else 0.0
                        )

    def sample_task(self, num_verifiers=1):
        """
        Samples a task based on the current scores.

        Parameters:
        - num_verifiers: Number of verifiers to include (1 or 2).

        Returns:
        - A tuple containing the sampled input task and verifiers.
        """
        # Sample InpTask
        input_task = self._weighted_random_choice(self.scores_input_task)

        # Sample Verifier1 given InpTask
        verifier_scores = self.scores_verifier_given_input[input_task]
        verifier1 = self._weighted_random_choice(verifier_scores)

        if num_verifiers == 1:
            return (input_task, verifier1)
        elif num_verifiers == 2:
            # Sample Verifier2 given Verifier1
            verifier2_scores = self.scores_verifier2_given_verifier1[verifier1]
            verifier2 = self._weighted_random_choice(verifier2_scores)
            return (input_task, verifier1, verifier2)
        else:
            raise ValueError("num_verifiers must be 1 or 2")

    def _weighted_random_choice(self, scores_dict):
        """
        Chooses a random key from a dictionary based on its values as weights.

        Parameters:
        - scores_dict: Dictionary of {key: score}

        Returns:
        - A randomly selected key.
        """
        items = list(scores_dict.items())
        keys, scores = zip(*items)
        scores = np.array(scores, dtype=np.float64)
        # Handle case where all scores are zero
        if np.all(scores == 0):
            scores = np.ones_like(scores)
        # Normalize scores to sum to 1
        probabilities = scores / scores.sum()
        choice = np.random.choice(keys, p=probabilities)
        return str(choice)

    def update_scores(self, task_components, feedback: str, save=True):
        """
        Updates the scores based on feedback.

        Parameters:
        - task_components: Tuple containing the input task and verifiers.
        - feedback: 'keep' to increase scores, 'discard' or 'invalid' to decrease.
        """
        input_task = task_components[0]
        verifier1 = task_components[1]

        if len(task_components) == 2:
            # One verifier task
            # Update scores_input_task
            self._update_score(
                self.scores_input_task, input_task, feedback
            )
            # Update scores_verifier_given_input
            self._update_score(
                self.scores_verifier_given_input[input_task], verifier1, feedback
            )
        elif len(task_components) == 3:
            # Two verifier task
            verifier2 = task_components[2]
            # Update scores_input_task
            self._update_score(
                self.scores_input_task, input_task, feedback
            )
            # Update scores_verifier_given_input
            self._update_score(
                self.scores_verifier_given_input[input_task], verifier1, feedback
            )
            # Update scores_verifier2_given_verifier1
            self._update_score(
                self.scores_verifier2_given_verifier1[verifier1], verifier2, feedback
            )
        else:
            raise ValueError("Invalid number of task components")
        
        if save:
            self.save_scores()

    def _update_score(self, scores_dict, key, feedback):
        """
        Helper method to update a score in a dictionary.

        Parameters:
        - scores_dict: The scores dictionary.
        - key: The key to update.
        - feedback: 'keep' to increase, 'discard' or 'invalid' to decrease.
        """
        if key not in scores_dict:
            # Assign average score if key is new
            avg_score = np.mean(list(scores_dict.values())) if scores_dict else 1.0
            scores_dict[key] = avg_score

        if feedback == 'keep':
            # Increase score at the key
            scores_dict[key] *= 1.1  # Increase by 10%
        elif feedback in ['discard', 'invalid']:
            # Decrease score at the key
            scores_dict[key] *= 0.9  # Decrease by 10%
        else:
            raise ValueError("Feedback must be 'keep', 'discard', or 'invalid'")
        
    def save_scores(self):
        """
        Saves the scores to a JSON file atomically.
        Always creates a backup before saving.
        """
        scores_path = Path(self.save_path)
        backup_path = self.backup_path

        # Prepare the data to be saved
        saved_data = {
            'scores_input_task': self.scores_input_task,
            'scores_verifier_given_input': self.scores_verifier_given_input,
            'scores_verifier2_given_verifier1': self.scores_verifier2_given_verifier1
        }

        try:
            # Create a temporary file in the same directory as scores.json
            with tempfile.NamedTemporaryFile('w', delete=False, dir=scores_path.parent, encoding='utf-8') as tmp_file:
                json.dump(saved_data, tmp_file, indent=4)
                temp_path = Path(tmp_file.name)

            # Create a backup of the current scores.json if it exists
            if scores_path.exists():
                scores_path.replace(backup_path)
                logging.info(f"Backup created at {backup_path}")

            # Atomically replace scores.json with the temporary file
            temp_path.replace(scores_path)
            logging.info(f"Scores successfully saved to {self.save_path}")
        except Exception as e:
            logging.error(f"Failed to save scores to {self.save_path}: {e}")
            # Clean up the temporary file if it exists
            if 'temp_path' in locals() and temp_path.exists():
                try:
                    temp_path.unlink()
                    logging.info(f"Temporary file {temp_path} deleted due to an error.")
                except Exception as cleanup_error:
                    logging.error(f"Failed to delete temporary file {temp_path}: {cleanup_error}")


    def load_scores(self):
        """
        Loads the scores from the main JSON file.
        If loading fails due to corruption, attempts to load from the backup.
        If both fail, terminates the program to allow for manual intervention.
        """
        scores_path = Path(self.save_path)
        backup_path = self.backup_path

        # Attempt to load the main scores.json
        try:
            logging.info(f"Attempting to load scores from {self.save_path}")
            with scores_path.open('r', encoding='utf-8') as f:
                saved_data = json.load(f)

            self.scores_input_task = saved_data.get('scores_input_task', {})
            self.scores_verifier_given_input = saved_data.get('scores_verifier_given_input', {})
            self.scores_verifier2_given_verifier1 = saved_data.get('scores_verifier2_given_verifier1', {})
            logging.info(f"Scores successfully loaded from {self.save_path}")
            return
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding failed for {self.save_path}: {e}")
        except FileNotFoundError:
            logging.warning(f"{self.save_path} does not exist. A new file will be created upon saving.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {self.save_path}: {e}")

        # If loading main scores.json failed, attempt to load from backup
        if backup_path.exists():
            try:
                logging.info(f"Attempting to load scores from backup {backup_path}")
                with backup_path.open('r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                self.scores_input_task = saved_data.get('scores_input_task', {})
                self.scores_verifier_given_input = saved_data.get('scores_verifier_given_input', {})
                self.scores_verifier2_given_verifier1 = saved_data.get('scores_verifier2_given_verifier1', {})
                logging.info(f"Scores successfully loaded from backup {backup_path}")
                return
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding failed for backup {backup_path}: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while loading backup {backup_path}: {e}")

        # If both main and backup loading failed, terminate the program
        logging.critical("Both main and backup score files are corrupted or missing. Terminating the program for manual intervention.")
        raise SystemExit("Critical Error: Failed to load scores from both main and backup files. Please restore the files manually.")
    

    def print_scores(self):
        """
        Prints the current scores for debugging.
        """
        print("Scores for InpTask:")
        for task, score in self.scores_input_task.items():
            print(f"  {task}: {score:.4f}")

        print("\nScores for Verifier | InpTask:")
        for task, verifier_scores in self.scores_verifier_given_input.items():
            print(f"  Given {task}:")
            for verifier, score in verifier_scores.items():
                print(f"    {verifier}: {score:.4f}")

        print("\nScores for Verifier2 | Verifier1:")
        for verifier1, verifier2_scores in self.scores_verifier2_given_verifier1.items():
            print(f"  Given {verifier1}:")
            for verifier2, score in verifier2_scores.items():
                print(f"    {verifier2}: {score:.4f}")
