#!/bin/bash
docker run -d --rm --name data-miner-gh \
-v ./src:/app \
--network=dev-insights_data-insights-network \
--env-file=.env  data-miner \
python /app/run.py DATA_MINER.MINE_GH
