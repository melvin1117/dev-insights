#!/bin/bash
docker run -d --rm --name etl-clear-rating \
-v ./src:/app \
--network=dev-insights_data-insights-network \
--env-file=.env  data-miner \
python /app/run.py ETL.CLR_DB_RATING
