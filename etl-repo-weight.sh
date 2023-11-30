#!/bin/bash
docker run -d --rm --name etl-repo-weight \
-v ./src:/app \
--network=dev-insights_data-insights-network \
--env-file=.env  data-miner \
python /app/run.py ETL.CAL_REPO_WEIGHT
