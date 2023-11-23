from typing import List, Callable
from queue import Queue, Empty
import concurrent.futures
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger


class ConcurrentExecutor:
    """
    ConcurrentExecutor class for parallel execution of tasks using ThreadPoolExecutor.
    """

    def __init__(
        self, tasks: List[str], workers_count: int, exec_func: Callable[[str], str]
    ) -> None:
        """
        Initialize ConcurrentExecutor.

        Args:
            tasks (List[str]): List of tasks to be executed.
            workers_count (int): Number of worker threads to be used.
            exec_func (Callable[[str], str]): Function to execute on each task.

        """
        self.tasks = tasks
        self.workers_count = workers_count
        self.task_queue = Queue()
        self.exec_func = exec_func

        for task in self.tasks:
            self.task_queue.put(task)

    def worker(self, executor: concurrent.futures.ThreadPoolExecutor) -> None:
        """
        Worker function for executing tasks in parallel.

        Args:
            executor (concurrent.futures.ThreadPoolExecutor): ThreadPoolExecutor instance.

        """
        while True:
            try:
                task = self.task_queue.get(block=False)
            except Empty:
                break  # No more tasks in the queue
            logger.info(f"{task}: Pushed to executor")
            future = executor.submit(self.exec_func, task)
            future.add_done_callback(lambda x: self.task_queue.task_done())

    def start(self) -> None:
        """
        Start parallel execution of tasks using ThreadPoolExecutor.
        """
        # Create a ThreadPoolExecutor with the desired number of workers
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.workers_count
        ) as executor:
            # Start worker threads
            worker_threads = [
                executor.submit(self.worker, executor)
                for _ in range(self.workers_count)
            ]

            # Wait for all tasks to be completed
            self.task_queue.join()

            # Wait for all tasks to be completed
            # concurrent.futures.wait(worker_threads, return_when=concurrent.futures.ALL_COMPLETED)

            # Stop worker threads
            for thread in worker_threads:
                thread.result()

            logger.info("All concurrent tasks completed!")
