import functools
import heapq
import json
import math
import multiprocessing
import pathlib
import pickle
import re
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from typing import Any

import pyarrow.parquet as pq
import tqdm

FINEWEB_PATH = "/mnt/NFS/data/hf/fineweb-edu"
OUTPUT_PATH = "/mnt/NFS/data/top_1_000_000_text.txt"


def get_single_column_iterator(file_path: str, column_name: str) -> Iterator[Any]:
    """
    Creates a memory-efficient iterator over a single column from a Parquet file.

    This function reads the Parquet file in chunks (row groups) and yields
    the values for the selected column one by one.

    Args:
        file_path: The path to the Parquet source file.
        column_name: The name of the column to iterate over.

    Yields:
        An iterator where each item is a single value from the specified column.
    """
    try:
        parquet_file = pq.ParquetFile(file_path)

        # iter_batches can accept a list of columns, so we pass the single column name
        for batch in parquet_file.iter_batches(columns=[column_name]):
            # Extract the single column array from the batch
            column_array = batch.column(column_name)

            # Iterate through the values in the array
            for value in column_array:
                # .as_py() converts the PyArrow scalar type to a native Python object
                yield value.as_py()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_column(file_paths: str, column_name: str) -> Iterator[Any]:
    for file in file_paths:
        yield from get_single_column_iterator(file, column_name)


def _worker_accumulator(func, zero, input_queue, output_queue):
    """
    A long-running worker function for the producer-consumer model.

    The worker initializes its running total with the provided 'zero' value.
    It then pulls batches from the input_queue, processes them, and accumulates
    the results. When it receives the shutdown signal (None), it pushes its
    final accumulated total to the output_queue.
    """
    # Initialize the local total with the provided identity value (zero).
    local_total = zero
    while True:
        batch = input_queue.get()

        # Check for the shutdown signal (poison pill).
        if batch is None:
            break
        # Apply the mapping function, reduce the batch, and add to the running total.
        # This requires that the 'zero' object supports __add__ with the result type.
        local_total += func(batch)

    # Put the final accumulated result onto the output queue.
    output_queue.put(local_total)


def parallel_reduce(func, zero, generator, batch_size=10000, num_processes=None):
    """
    Performs a parallel reduction using a producer-consumer model.

    Args:
        func: A function that maps an element from the generator to an object
              that supports the `__add__` operation.
        generator: A generator that yields the items to be processed.
        zero: The identity value for the sum (e.g., 0 for numbers, Vector(0,0) for vectors).
              This is the value returned for an empty generator.
        batch_size: The number of items from the generator to be processed in each batch.
        num_processes: The number of worker processes to use. Defaults to the number of CPU cores.

    Returns:
        The total sum of the mapped and reduced items.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Create the shared queues for communication.
    input_queue = multiprocessing.Queue(maxsize=num_processes * 2)
    output_queue = multiprocessing.Queue()

    # Create and start the consumer processes.
    processes = []
    for _ in range(num_processes):
        proc = multiprocessing.Process(
            target=_worker_accumulator,
            # Pass the 'zero' value to each worker for initialization.
            args=(func, zero, input_queue, output_queue),
        )
        processes.append(proc)
        proc.start()

    with tqdm.tqdm() as pbar:
        # The main process acts as the producer.
        for batch in iter(lambda: list(islice(generator, batch_size)), []):
            input_queue.put(batch)
            pbar.update(len(batch))
            pbar.refresh()

    # Signal to the workers that the generator is exhausted.
    for _ in range(num_processes):
        input_queue.put(None)

    # Collect the final result from each worker.
    # Each worker will at least return its initial 'zero' value.
    r0, *rs = (output_queue.get() for _ in range(num_processes))

    # Wait for all worker processes to finish.
    for proc in processes:
        proc.join()

    # Perform the final reduction on the partial results, starting with the initial 'zero'.
    # This elegantly handles the case of an empty generator.
    return sum(rs, r0)


class TopK:
    def __init__(self, top=None, k=10000):
        self.top = top if top is not None else []
        self.k = k

    def __add__(self, other):
        assert self.k == other.k
        top = list(islice(heapq.merge(self.top, other.top, reverse=True), self.k))
        return TopK(top, self.k)

    @classmethod
    def from_docs(cls, f, values, k=10000):
        return cls(list(heapq.nlargest(k, (f(v) for v in values))), k)


class Extractor:
    def __init__(self, top=None):
        self.top = top if top is not None else []

    def __add__(self, other):
        top = self.top + other.top
        return Extractor(top)

    @classmethod
    def from_docs(cls, f, values, threshold):
        return cls([v for v in values if f(v) >= threshold])


class BM25Stats:
    def __init__(self, term_freq=None, doc_freq=None, num_docs=0, tot_toks=0):
        self.term_freq = term_freq if term_freq else Counter()
        self.doc_freq = doc_freq if doc_freq else Counter()
        self.num_docs = num_docs
        self.tot_toks = tot_toks

    def __add__(left, right):
        term_freq = left.term_freq + right.term_freq
        doc_freq = left.doc_freq + right.doc_freq
        num_docs = left.num_docs + right.num_docs
        tot_toks = left.tot_toks + right.tot_toks
        return BM25Stats(term_freq, doc_freq, num_docs, tot_toks)

    def __iadd__(left, right):
        left.term_freq += right.term_freq
        left.doc_freq += right.doc_freq
        left.num_docs += right.num_docs
        left.tot_toks += right.tot_toks
        return left

    @classmethod
    def from_doc(cls, tokens, vocab=None):
        rtokens = tokens if vocab is None else [t for t in tokens if t in vocab]
        term_freq = Counter(rtokens)
        doc_freq = Counter(set(rtokens))
        tot_toks = len(tokens)
        num_docs = 1
        return cls(term_freq, doc_freq, tot_toks, num_docs)

    @classmethod
    def from_docs(cls, docs, vocab=None):
        term_freq = Counter()
        doc_freq = Counter()
        tot_toks = 0
        num_docs = 0
        for tokens in docs:
            rtokens = tokens if vocab is None else [t for t in tokens if t in vocab]
            term_freq.update(rtokens)
            doc_freq.update(set(rtokens))
            tot_toks += len(tokens)
            num_docs += 1
        return cls(term_freq, doc_freq, tot_toks, num_docs)

    def idf(self):
        return {w: math.log1p((self.num_docs - wf + 0.5) / (wf + 0.5)) for w, wf in self.doc_freq.items()}

    def etf(self, k=1.2, b=0.75):
        return {w: wf / self.num_docs for w, wf in self.term_freq.items()}


tp = re.compile(r"\w*[a-zA-Z]\w*")


def tokenize(doc):
    return tp.findall(doc.lower())


def score(doc, scorer):
    return sum(scorer.get(word, 0.0) for word in set(tokenize(doc)))


if __name__ == "__main__":
    paths = list(pathlib.Path(FINEWEB_PATH).rglob("*.parquet"))

    with open("fw.pkl", "rb") as f:
        fw = pickle.load(f)  #  noqa: S301
    with open("pharma.pkl", "rb") as f:
        pharma = pickle.load(f)  #  noqa: S301

    fw_idf = fw.idf()
    fw_etf = fw.etf()
    ph_idf = pharma.idf()
    ph_etf = pharma.etf()

    scorer = {}
    for w in ph_idf:
        if (fw.doc_freq.get(w, 0) > 10) & (pharma.doc_freq.get(w, 0) > 1):
            scorer[w] = ph_etf[w] / fw_etf[w] * fw_idf[w]
    # k = 1_000_000
    threshold = 164575.0

    def mapping(paths):
        return Extractor.from_docs(functools.partial(score, scorer=scorer), get_column(paths, "text"), threshold)

    topk = parallel_reduce(mapping, Extractor(), iter(paths), batch_size=1)

    with open(OUTPUT_PATH, "w") as f_out:
        for doc in topk.top:
            f_out.write(json.dumps({"text": doc}))
            f_out.write("\n")
