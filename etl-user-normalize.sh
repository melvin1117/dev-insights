#!/bin/bash
docker run -d --rm --name etl-user-normalize \
-v ./src:/app \
--network=dev-insights_data-insights-network \
--env-file=.env  data-miner \
python /app/run.py ETL.NORM_USER_RATING
