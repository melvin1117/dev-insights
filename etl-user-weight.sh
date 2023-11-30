#!/bin/bash
docker run -d --rm --name etl-user-weight \
-v ./src:/app \
--network=dev-insights_data-insights-network \
--env-file=.env  data-miner \
python /app/run.py ETL.CAL_USER_WEIGHT
