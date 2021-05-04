Models notebooks
==============================
Here you can play with our implementation of the models on a smaller dataset.
Before running the notebooks you should prepare the required data. Possible way is:
- use the notebook *notebooks/data/get_subset.ipynb* to generate the subset of the original dataset, 
- prepare the datasets by running the script:
 ```bash
python run.py --dataset 'jobs_published-part' --steps '["prepare"]' --validation_fraction_users 1 --validation_fraction_items 1
 ```
