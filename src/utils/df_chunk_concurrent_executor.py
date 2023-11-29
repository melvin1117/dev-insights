import numpy as np
import concurrent.futures
from typing import Callable, Any
from queue import Queue, Empty
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger


class DFChunkConcurrentExecutor:
    def __init__(
        self,
        df,
        exec_func: Callable[[Any], Any],
        workers_count = 5,
        chunk_size: int = 100,
        executor_name = 'DF_CHUNK_EXECUTOR'
    ):
        self.df = df
        self.workers_count = workers_count
        self.task_queue = Queue()
        self.exec_func = exec_func
        self.chunk_size = chunk_size
        self.executor_name = executor_name

        # Split DataFrame into chunks
        chunks = np.array_split(self.df, len(self.df) // self.chunk_size + 1)
        for chunk in chunks:
            self.task_queue.put(chunk)

    def worker(self, executor: concurrent.futures.ThreadPoolExecutor) -> None:
        while True:
            try:
                chunk = self.task_queue.get(block=False)
            except Empty:
                break  # No more tasks in the queue

            logger.info(f"{self.executor_name}: Processing chunk: {chunk.index}")
            future = executor.submit(self.exec_func, chunk)
            future.add_done_callback(lambda x: self.task_queue.task_done())

    def start(self) -> None:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.workers_count
        ) as executor:
            worker_threads = [
                executor.submit(self.worker, executor)
                for _ in range(self.workers_count)
            ]

            # Wait for all tasks to be completed
            self.task_queue.join()

            # Wait for all tasks to be completed
            concurrent.futures.wait(
                worker_threads, return_when=concurrent.futures.ALL_COMPLETED
            )

            logger.info(f"{self.executor_name}: DF Chunk concurrent tasks completed!")
