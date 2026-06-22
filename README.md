````markdown
# optimaltransport
The following code represents the method for the findings found in the project report "02466 Project in Optimal Transport" By Frederik Kliim, Eren Chaglar and Aksel Mads Madsen. 
The following readme file will explain how to use and replicate our findings.


## Prerequisites 
To be on the safe side an installation of the newest python would be ideal. Then one would need to use UV, the python package manager
https://docs.astral.sh/uv/
And run the command in the terminal from this project folder
``
uv sync
``

If for any reason one would want to use pip install to get the required packages one could run the following command
``
python -m pip install -r requirements-pip.txt
``
Though this is not recommended as this project have only been tested using UV sync. 


## Project structure
usage: __main__.py [-h] [--config CONFIG] [--mode {train,train_classifier,evaluate,knn,visualize,all,sinkhorn,transport_classifier,variance}]
                   [--checkpoint CHECKPOINT] [--classifier-checkpoint CLASSIFIER_CHECKPOINT] [--split {train,test}] [--source-class SOURCE_CLASS]
                   [--target-class TARGET_CLASS] [--max-points MAX_POINTS] [--latent-dims LATENT_DIMS [LATENT_DIMS ...]] [--variance-csv VARIANCE_CSV]

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --mode {train,train_classifier,evaluate,knn,visualize,all,sinkhorn,transport_classifier,variance}
  --checkpoint CHECKPOINT
  --classifier-checkpoint CLASSIFIER_CHECKPOINT
  --split {train,test}
  --source-class SOURCE_CLASS
  --target-class TARGET_CLASS
  --max-points MAX_POINTS
  --latent-dims LATENT_DIMS [LATENT_DIMS ...]
                        Latent dimensions to evaluate, e.g. --latent-dims 1 2 3 5 10 20 30 50
  --variance-csv VARIANCE_CSV
                        CSV file for saving transported/target variance diagnostics.

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

````
