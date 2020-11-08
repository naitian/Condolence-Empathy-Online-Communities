"""
Runs multiple labeling jobs in parallel:
    Guarantees that no two jobs will be working on the same file at the same
    time
"""
import argparse
import queue
import threading

from grab_comments import main as label


class Worker:
    def __init__(self, queue, cuda, args):
        self.queue = queue
        self.cuda = cuda
        self.year = args.year
        self.sample = args.sample

    def __call__(self):
        while True:
            if self.queue.empty():
                return
            month = self.queue.get()
            args = argparse.Namespace(
                timeframe="{}-{:02}".format(self.year, month),
                device=self.cuda,
                sample=self.sample,
            )
            label(args)


def main(args):
    num_workers = min(12, args.num_workers)
    q = queue.Queue()
    for i in range(1, 13):
        # TODO: this can be better
        q.put(i)
    threads = []
    for i in range(1, num_workers + 1):
        cuda_device = "cuda:{}".format(i)
        worker = Worker(q, cuda_device, args)
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kicks off parallel jobs to label comments"
    )
    parser.add_argument(
        "num_workers", action="store", help="Number of GPUs to use", type=int,
    )
    parser.add_argument(
        "year", action="store", help="Year to get comments from (YYYY)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use reservoir sampling to only process a random subset of comments",
    )
    main(parser.parse_args())
