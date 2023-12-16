# Dev Insights

Identifying Programming Languages Trends based on GitHub and Stack Overflow Survey Data

## Live Demo
You can visit [here](http://157.230.217.156/) to view the application dashboard.

## GitHub Repository
You can visit [here](https://github.com/melvin1117/dev-insights) to view the GitHub repository.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* Git
* You have installed Docker and Docker Compose.
* You have a basic understanding of containerization and Docker.

## Configuration
1. Edit the `src/assets/config.json` file with the necessary configs and `src/assets/api_endpoints.py` for API endpoints. **Provided with the code, no changes required.**

2. Must have `.env` file at root directory which contains environment related configurations

**`.env` file looks like below. Create `.env` file if not present at the root directory of the project.**

```yml
  APP_NAME=DEV-INSIGHTS
  LOGGING_LEVEL=INFO
  DB_USER=<username>
  DB_PASS=<password>
  DB_HOST=mongo
  DB_SERVER_HOST=157.230.217.156 #IP address of server where the project is hosted
  DB_PORT=27017
  DB_NAME=dev-insights
  ELASTICSEARCH_USER=<username>
  ELASTICSEARCH_PASS=<password>
  MODULE_ETL=ETL
  MODULE_DATA_MINER=DATA_MINER
  MAX_FALLBACK_ATTEMPTS=2
  # GitHub Miner ENVs
  GITHUB_API_KEYS=gh-token1,gh-token2,gh-token3
  MAX_RECORD_PER_SESSION=30
  GAP_BETWEEN_CALL_SEC=60
  FETCH_PAST_NUM_DAYS=600
  NUM_DAYS_CHUNK_SIZE=60
  GET_USER_REPO=True
  #ETL GitHub
  RATING_DELTA=0.02
  #Google Maps
  GMAP_API_KEY=gmap-token
```
In the above `.env` file sample, replace the `GITHUB_API_KEYS` and `GMAP_API_KEY` token value with tokens generated from GITHUB and GOOGLE respectively.

**Note:** `GITHUB_API_KEYS` accepts multiple comma separated tokens, the number of token provided is equivalent to the number of threads/works executing in parallel to collect/mine the data from GitHub.

## Dataset
The project requires two dataset - GitHub and Stack Overflow.
- The GitHub data is fetched using the `data-miner` module ans is saved to the MongoDB database. Below are the steps to run the module to collect the data.
- The Stack Overflow dataset on the other hand is saved as a zipped file and is part of codebase located at `\src\assets\so\raw` directory and named as `survey_results.zip`. The processed data is saved at location `\src\assets\so` which is done automatically by the code `data_processing.ipynb`.


## Running Locally

### Overview
To run the project on your local machine for development and testing purposes, follow these steps:

1. Clone the repository: 
```git clone https://github.com/melvin1117/dev-insights.git```
2. Navigate to the project directory: `cd dev-insights`
3. Add `.env` file at root directory. (As per the instruction above)
4. Build the Docker images:
Run command `docker compose build` (if you have docker compose v2, else run `docker-compose build`)
5. Once the build is completed. Run the below command to up the containers and run the project and its environment.
`docker compose up` or `docker-compose up`
6. Run command `docker compose down` to stop the execution and the containers.

### Module/Algorithm Stage Execution (Collection, Processing and Analyzing of Data)
Each module/stage in the project has to be executed separately. The above steps are only responsible to setting up environment and containers.
Each of the stages needs to be executed separately one after the other as execution of large dataset takes a lot of time. If done in one go, may lead to heavy system usage and crashing.

In order to run each stage/module as discussed in the algorithm, there is a shell script file for each of the stage.

**Note:**

If you are on **Linux** machine, run the from command from the root directory `chmod +x ./<shell script file_name>` in order to change the permission of the file. Once done just run the shell script file using `./<shell script file_name>`. This will make sure to spawn the relevant stage container and starts its execution. For more information on how to run the shell script please refer [here](https://www.cyberciti.biz/faq/run-execute-sh-shell-script/).

If you are on **Windows** machine, open the shell script file, copy the command (except `#!/bin/bash`) and run it in the root directory. (If you face any issue, make sure to make sure to make the command in single line by removing the `\` from the end of each file and make the entire command in one line, and then copy and run the command)


#### Run GitHub Data Miner module to collect and clean GitHub data and save to database
&emsp;*Prerequisite:* None
- Run the `data-miner-gh.sh` shell script file.
- This will start the execution of the program and start collecting the github data using its API, clean it and save it to mongo database. The program fetches the github data in chunks of 60 days for the past ~2 years from current date (configuration in `.env` file). Every incremental run will fetch 60 days chunk data for all the programming languages selected for the project.
- We ran incremental loads using CRON jobs.

#### Run Stage 1, Step 1 of the algorithm: GitHub Repository Column Weight Calculation Classifier
&emsp;*Prerequisite:* GitHub Data in database collected using data-miner module

&emsp;Run the `etl-repo-weight.sh` shell script file.

#### Run Stage 1, Step 2 of the algorithm: GitHub Repository Rating Component
&emsp;*Prerequisite:* Stage 1, Step 1 of the algorithm

&emsp;Run the `etl-repo-rating.sh` shell script file.

#### Run Stage 1, Step 3 of the algorithm: GitHub Rating Normalization

&emsp;*Prerequisite:* Stage 1, Step 2 of the algorithm

&emsp;Run the `etl-repo-normalize.sh` shell script file.

#### Run Stage 2, Step 1 of the algorithm: GitHub User Column Weight Calculation Classifier

&emsp;*Prerequisite:* Stage 1 of the algorithm

&emsp;Run the `etl-user-weight.sh` shell script file.

#### Run Stage 2, Step 2 of the algorithm: GitHub User Rating Component

&emsp;*Prerequisite:* Stage 2, Step 1 of the algorithm

&emsp;Run the `etl-user-rating.sh` shell script file.

#### Run Stage 2, Step 3 of the algorithm: GitHub User Rating Normalization

&emsp;*Prerequisite:* Stage 2, Step 2 of the algorithm

&emsp;Run the `etl-user-normalize.sh` shell script file.

#### Run Stage 2, Step 4 and Stage 3 of the algorithm: Geocoding and Categorizing User proficiency

&emsp;*Prerequisite:* Stage 2, Step 3 of the algorithm

&emsp;Run the `etl-user-proficiency.sh` shell script file.

#### Run StackOveflow Jupyter Dashboard
&emsp;*Prerequisite:*
- Create virtual environment at root directory, run `python -m venv venv`
- Install `requirements.txt` packages using `pip install -r requirements.tx`

&emsp;Run the `data_processing.ipynb` under `src/etl/so` directory in IDE of your choice

#### View Dashboard

&emsp;*Prerequisite:* All the above stages.

&emsp;Once the above stages execution is completed, Open the URL [http://localhost/](http://localhost/) in your browser to view the dashboard.

**By default the dashboard when ran displayed the visualization using the data from the database deployed in cloud.**

**Note:**  
Even to view the dashboard we need to complete all the stages mentioned above. If you want to avoid the data collection, cleaning, rating calculation, normalization (all the above mentioned stages) stages and just want to view the dashboard from already processed data. You can simply run the project using steps in `Overview` section and view the dashboard using the URL [http://localhost/](http://localhost/) on the browser. By default the dashboard when ran shows data from the database deployed in cloud.


## Deployment

To deploy the project on a server or production environment, follow these steps:

1. Set up a Docker environment on your server. (Refer prerequisite for more information)
2. Pull the latest version of your project from the repository:
`git pull origin master`
3. Build and start your containers in detached mode:
`docker compose up --build -d` or `docker-compose up --build -d`

### Restart Container

`docker compose restart <container-name>`

## Monitoring & Logging

* The project has its own logging set up already. Logs are saved to `logs` directory under `src` folder. Each file has its own log file.
* Alternatively, you can view logs of each container in the docker using the command. `docker logs --follow <container-name>`

## Contributing to Dev Insights

To contribute to Dev Insights, follow these steps:

1. Fork this repository.
2. Create a new branch: `git checkout -b branch_name`.
3. Make your changes and commit them: `git commit -m 'commit_message'`.
4. Push to the original branch: `git push origin project_name/path`.
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Maintainers

Below are the people who contributed to the project
1. Shubham Melvin Felix
2. Uddesh
3. Abhishek Gavali
4. Uday
5. Vinuth
