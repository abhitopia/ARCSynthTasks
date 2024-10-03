#%%
from src.utils import get_compound_verfiers, get_verifiers, load_input_tasks
from src.utils import process_input_task_verifier
verifiers = get_verifiers()
# %%
len(verifiers)
# %%
verifiers.keys()
# %%
input_tasks = load_input_tasks(include_outputs=False)
# %%
len(input_tasks)
# %%
max_length = 2
compound_verifiers = get_compound_verfiers(verifiers, max_length)
# %%

verifier_name = next(iter(compound_verifiers.keys()))
verifier = compound_verifiers[verifier_name]
input_task = input_tasks[0]

#%%s
process_input_task_verifier(input_task,
                            verifier_name, 
                            verifier,
                            [],
                            'test')

# %%
