from typing import List, Callable
from queue import Queue, Empty
import concurrent.futures

class ConcurrentExecutor:
    def __init__(self, tasks: List[str], workers_count: int, exec_func):
        self.tasks = tasks
        self.workers_count = workers_count
        self.task_queue = Queue()
        self.exec_func = exec_func
        for task in self.tasks:
            self.task_queue.put(task)

    def worker(self, executor):
        while True:
            try:
                task = self.task_queue.get(block=False)
            except Empty:
                break  # No more tasks in the queue
            print(f'\nPushing {task} to executor...\n')
            future = executor.submit(self.exec_func, task)
            future.add_done_callback(lambda x: self.task_queue.task_done())

    def start(self):
        # Create a ThreadPoolExecutor with the desired number of workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_count) as executor:
            # Start worker threads
            worker_threads = [executor.submit(self.worker, executor) for _ in range(self.workers_count)]

            # Wait for all tasks to be completed
            self.task_queue.join()

            # Stop worker threads
            for thread in worker_threads:
                thread.result()
