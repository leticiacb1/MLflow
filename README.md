### 💡️ MLflow

MLflow is an API that allows you to integrate MLOps principles into your projects with minimal changes made to existing code, providing a comprehensive framework for managing and organizing ML workflows.

With MLflow tracking, developers can easily log and monitor parameters, metrics, and artifacts generated during ML runs and analyze relevant details of ML projects.

* Download data source [here](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction?resource=download). (Put the file in `data` folder)

#### 📌 Dependencies

Create a `venv` and install dependencies:

```bash
    # Create environment
    $ python3 -m venv venv  

    # Activate environment
    $ source venv/bin/activate

    # Install dependencies
    $ pip install -r requirements.txt
``` 

Configure the secrets in your repository : go to the repository site on `github / settings / Secrets and variables / Actions` and add a **new repository secrets**.

Set all the secrests :

* `AWS_ACCESS_KEY_ID`
  
* `AWS_SECRET_ACCESS_KEY`
  
* `AWS_REGION`
  
* `AWS_LAMBDA_ROLE_ARN`

Also create a `.env` file with the following:

```bash
    # .env content'
    AWS_ACCESS_KEY_ID="XXXXXXXXXXXXXX"
    AWS_SECRET_ACCESS_KEY="aaaaaaaaaaaaaaaaaaaaaaaaaaa"
    AWS_REGION="xx-xxxx-2"
    AWS_LAMBDA_ROLE_ARN="arn:xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
``` 

Configure your AWS credentials:

```bash
    aws configure
```

#### ❓️ How to use the project


```bash
# Root of the project
python src/train.py
```

The Tracking UI provides a user-friendly interface to visualize, search, and compare runs in MLflow. Navigate to the parent directory of mlruns (project root directory) and run:

```bash
# Root of the project
mlflow ui -p 5005
```

Acess [http://localhost:5005](http://localhost:5005) in a browser an see the MLflow graphical interface.
The a `mlruns` folder , created by MLflow, store information about the runs (metrics, parameters, artifacts) of the model.

> Keep local storage of experiments may be a problem, as one developer will not have vision of what was attempted by another developer.
> Its appropriate to keep information about experiments centrally.
> * To keep metrics and parameter data centrally we can use **relational database**, for example.
> * To store artifacts, such as images and files, centrally we can use **S3 bucket**. 
>
> ```bash
>     |_Database_| ↔️  |_MLFOW_Server_| ↔️  |_S3_Bucket_|
>                            ↕️ 
>                    |_ML_Model_Code_|
> ```  
>
> **OBS:** The MLFlow server could also run centrally (like on an EC2 instance). However, we will keep it running locally, but storing data centrally.                   

---
##### 1. Centralization 
---

1. Using [DBeaver](https://dbeaver.io/download/) create a database connection. 
Create a database using this sql script:

```sql
CREATE DATABASE mlflow_USERNAME;
```


2. Create a S3 bucket using [aws cli](https://aws.amazon.com/pt/cli/)

To store your experiment artifacts.

```bash
  # Make sure you have run this before
  # $ aws configure 
  $ aws s3api create-bucket --bucket mlflow-exp-tracking-USERNAME --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2

  # Verify bucket created:
  $ aws s3api head-bucket --bucket mlflow-exp-tracking-USERNAME

  # List all buckets
  $ aws s3 list
```

<br> **Configure MLFlow server**

```bash
$ mlflow server --backend-store-uri postgresql://USERNAME:PASSWORD@HOST:PORT/<DATABASE-NAME> --default-artifact-root s3://mlflow-exp-tracking-USERNAME
```

> MLflow will make requests to the REST API of the MLflow server that is running locally and, the server will store the experiment logs in PostgreSQL and AWS S3.

For use the MLFlow server URL change the file `src/train.py` :

```python

# ---- Code ommited ----

def main():
    experiment_name = "churn-exp"
    run_name = "churn-knn"
    data_file = "data/Churn_Modelling.csv"

    model_type = "knn"  # ['knn', 'logistic']

    # Add this line :
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # ---- Code ommited ----

```

Create a `.env` file inside the `src/` folder:

```bash
    AWS_ACCESS_KEY_ID="*******"
    AWS_SECRET_ACCESS_KEY="*******"
    AWS_REGION="*******"
    AWS_ACCOUNT_ID="*******"
```

In root directory run:

```bash
    $ python src/train.py
```

Go to http://localhost:5000 in your browser and check if the experiment results are available.

```bash
    # Check the objects created in the bucket
    $ aws s3api list-objects-v2 --bucket mlflow-exp-tracking-USERNAME

    # Also check the created tables and their contents in DBeaver
```

---
##### 2. Model Registry 
---

It is common for ML products to have hundreds of iterations and changes (to data, parameters, algorithms) in an attempt to improve performance.

There comes a time when we are simply lost! Tracking module of MLFlow, which allows users to log and organize their experiments, allows data scientists to reproduce the results of an experiment.

Another component of MLFlow is the `Model Registry`, which serves as a centralized model store, provides a set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model.

In `train.py` file is possible to see that the model was logged at the same time it was registered:

```python
    # ---- Code ommited ----
    mlflow.sklearn.log_model(log_reg,
                            "model",
                            signature=signature,
                            registered_model_name="churn-model", # <-- LOOK HERE!
                            input_example=X_train.iloc[:3],
                            )
    # ---- Code ommited ----
```

Access the _Models_ tab in the MLflow UI to see the model registration.

---
##### 3. Docker for MLflow
---

1. Choose a model and a version to deploy in docker image
2. Create a env variable:
   
    ```bash
        $ export MLFLOW_TRACKING_URI=http://localhost:5000
    ```

3. Generate the docker image of the model
   
    ```bash
        $ mlflow models build-docker --name <docker-img-name> --model-uri "models:/<model-name>/<version-number>"
    ```

4. Run the docker image
   
    ```bash
        $ docker run -d -p 8080:8080 <docker-img-name>:latest   
    ```

5. Fot test the model run the file `test/model.py`

```bash
  [EXPECTED OUTPUT]
    Status code: 200
    Response: {"predictions": [0, 0]}
```

<br>
@2024, Insper. 9° Semester,  Computer Engineering.
<br>

_Machine Learning Ops & Interviews Discipline_