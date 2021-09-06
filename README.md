Jobs recommendations
==============================

This repository contains the implementation of:

- several recommender system models appropriate
for large-scale jobs recommendations,
- a hyperparameter tuning,
- evaluation metrics.

Currently implemented models:

- ALS/WRMF: proposed in [Collaborative Filtering for Implicit Feedback Datasets](https://www.researchgate.net/publication/220765111_Collaborative_Filtering_for_Implicit_Feedback_Datasets);
  implementation based on the [Implicit](https://implicit.readthedocs.io/en/latest/als.html) implementation
- Prod2vec: proposed in [E-commerce in Your Inbox: Product Recommendations at Scale](https://www.researchgate.net/publication/304350592_E-commerce_in_Your_Inbox_Product_Recommendations_at_Scale);
  implementation based on [Gensim](https://github.com/RaRe-Technologies/gensim) Word2vec implementation
- RP3Beta proposed in [Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications](https://www.researchgate.net/publication/312430075_Updatable_Accurate_Diverse_and_Scalable_Recommendations_for_Interactive_Applications)
- SLIM: proposed in [SLIM: Sparse Linear Methods for Top-N Recommender Systems](https://www.researchgate.net/publication/220765374_SLIM_Sparse_Linear_Methods_for_Top-N_Recommender_Systems)
- LightFM: proposed in [Metadata Embeddings for User and Item Cold-start Recommendations](https://www.researchgate.net/publication/280589936_Metadata_Embeddings_for_User_and_Item_Cold-start_Recommendations);
  implementation based on the original [LightFM](https://github.com/lyst/lightfm) implementation
  
## Environment configuration

If you use conda, set-up conda environment with a kernel (working with anaconda3):

 ```bash
 make ckernel
 ```

If you use virtualenv, set-up virtual environment with a kernel:

 ```bash
 make vkernel
 ```

Then activate the environment:

 ```bash
source activate jobs-research
 ```

## Steps to reproduce the results

### Getting data

The input data file *interactions.csv* should be stored in the directory *data/raw/your-dataset-name*.
For example, *data/raw/jobs_published/interactions.csv*.
The file is expected to contain the following columns: *user, item, event, timestamp*.

To reproduce our results [download](https://www.kaggle.com/olxdatascience/olx-jobs-interactions) the
**olx-jobs dataset** from Kaggle.

### Running

Execute the command:

```bash
python run.py
 ```

The script will:

- split the input data,
- run the hyperparameter optimization for all models,
- train the models,
- generate the recommendations,
- evaluate the models. <br>

#### Details about each step

By default script executes all aforementioned steps, namely:

```bash
--steps '["prepare", "tune", "run", "evaluate"]'
 ```

##### Step *prepare*

This step:

- loads the raw interactions,
- splits the interactions into the *train_and_validation* and *test* sets,
- splits the *train_and_validation* set into *train* and *validation* sets,
- prepares *target_users* sets for whom recommendations are generated,
- saves all the prepared datasets.

Due to the large size of our dataset, we introduced additional parameters which enable us
to decrease the size of the *train* and *validation* sets used in the hyperparameter tuning:

```bash
--validation_target_users_size 30000
--validation_fraction_users 0.2
--validation_fraction_items 0.2
 ```

##### Step *tune*

This step performs Bayesian hyperparameter tuning on the *train* and *validation* sets.
<br>
For each model, the search space and the tuning parameters are defined in the *src/tuning/config.py* file.
The results of all iterations are stored.

##### Step *run*

This step, for each model:

- loads the best hyperparameters (if available),
- trains the model,
- generates and saves recommendations,
- saves efficiency metrics.

##### Step *evaluate*

This step, for each model:

- loads stored recommendations,
- evaluates them based on the implemented metrics,
- displays and stores the evaluation results.

### Notebooks

#### data

Notebooks to analyze the dataset structure and distribution.

#### models

Notebooks to demonstrate the usage of the particular models.

#### evaluation

Notebooks to better understand the results.
They utilize recommendations and metrics generated during the execution of the *run* script.
