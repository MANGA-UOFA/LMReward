import pathlib
import torch.multiprocessing as mp
import time
import copy
import transformers
import numpy as np
import utils.logger


CHUNK_SIZE = 1024
LARGE_THRESHOLD = 10000000

logger = utils.logger.Logger()


"""How index is converted:
i-th iter -> shuffled batch id -> sorted continuous sent ids -> raw #row
"""


def process(nprocs, inq, outq, tokenizer):
    batch = []
    while True:
        if len(batch) < nprocs * CHUNK_SIZE and inq.qsize() > 0:
            poses = []
            ss = []
            ts = []
            for pos, s, t in inq.get():
                poses.append(pos)
                ss.append(s)
                ts.append(t)      
            tok_ss = tokenizer(ss, verbose=False)['input_ids']
            tok_ts = tokenizer(ts, verbose=False)['input_ids']
            del ss, ts
            for pos, s, t in zip(poses, tok_ss, tok_ts):
                batch.append((pos, len(s), len(t)))
            
        elif len(batch) > 0:
            outq.put_nowait(batch)
            batch = []
        else:
            time.sleep(1)


class DatasetForSeq2Seq:
    def __init__(self, dir, split, suffixes, max_length, ratio, tokenizer, num_workers) -> None:
        super().__init__()
        assert len(suffixes) == 2
        ctx = mp.get_context('fork')
        self.dir = pathlib.Path(dir)
        self.split = split
        self.src, self.tgt = suffixes
        self.f0 = self.dir / (self.split + '.' + self.src)
        self.f1 = self.dir / (self.split + '.' + self.tgt)
        self.max_length = max_length
        self.ratio = max(1/ratio, ratio)

        pos = []
        lengths = []
        lengths_sum = []
        input_q = [ctx.Queue() for _ in range(num_workers)]
        output_q = ctx.Queue()
        processes = [ctx.Process(target=process, args=(
            num_workers, input_q[i], output_q, tokenizer)) for i in range(num_workers)]
        for p in processes:
            p.start()
        with open(self.dir / (self.split + '.' + self.src), 'rb') as fp0, \
                open(self.dir / (self.split + '.' + self.tgt), 'rb') as fp1:
            idx = 0
            worker_id = 0
            while True:
                batch = []
                for _ in range(CHUNK_SIZE):
                    _pos = [fp0.tell(), fp1.tell()]
                    s = fp0.readline().decode('utf8', errors='ignore')
                    t = fp1.readline().decode('utf8', errors='ignore')
                    if s == '' and t == '':
                        break
                    batch.append((_pos, s, t))
                    idx += 1
                    if idx % LARGE_THRESHOLD == 0:
                        logger.log(
                            f'\t #sent: {idx} | #processed: {output_q.qsize() * num_workers * CHUNK_SIZE}')
                if len(batch) == 0:
                    break
                input_q[worker_id % num_workers].put_nowait(batch)
                worker_id += 1
        total = idx
        idx = 0
        discarded = 0
        while idx < total:
            for _pos, len_s, len_t in output_q.get():
                if 0 < len_s < self.max_length \
                        and 0 < len_t < self.max_length \
                        and 1/self.ratio <= len_s / len_t <= self.ratio:
                    pos.append(_pos)
                    lengths_sum.append(len_s + len_t)
                    lengths.append([len_s, len_t])
                else:
                    discarded += 1
                idx += 1
                if idx % LARGE_THRESHOLD == 0:
                    logger.log(
                        f'\t #received: {idx} | #remaining {total - idx}')
        for p in processes:
            p.terminate()

        self.pos = np.array(pos)
        self.index = np.argsort(np.array(lengths_sum))
        self.lengths = np.array(lengths)

        logger.log(
            f'#discarded {discarded} | #preseved {len(lengths)} | #total {idx}')

    def __getitem__(self, index):
        fp0, fp1, index = index
        index = self.index[index]
        p0, p1 = self.pos[index]
        fp0.seek(p0)
        fp1.seek(p1)
        return {
            'src_sent': fp0.readline().decode('utf8', errors='ignore').strip(),
            'tgt_sent': fp1.readline().decode('utf8', errors='ignore').strip(),
            'len_src_sent': self.lengths[index][0],
            'len_tgt_sent': self.lengths[index][1],
        }

    def __len__(self):
        return len(self.index)


class BatchedDataset:
    def __init__(self, dataset, max_tokens, tokenizer, num_workers):
        ctx = mp.get_context('fork')
        self.dataset = copy.deepcopy(dataset)
        self.tokenizer = copy.deepcopy(tokenizer)
        self.max_tokens = max_tokens
        self.num_workers = num_workers
        self.q = ctx.Queue()
        self.batches = self._bachify()
        
    def shuffle(self):
        np.random.shuffle(self.batches)

    def __getitem__(self, index):
        fp0, fp1, index = index
        start_idx, end_idx = self.batches[index]
        src_data = []
        tgt_data = []
        for i in range(start_idx, end_idx):
            data = self.dataset[(fp0, fp1, i)]
            src_data.append(data['src_sent'])
            tgt_data.append(data['tgt_sent'])
        return {
            'src': self.tokenizer(src_data, padding=True, return_tensors='pt'),
            'tgt': self.tokenizer(tgt_data, padding=True, return_tensors='pt')
        }
    
    def _bachify(self):
        length = len(self.dataset)
        splits = [0]
        for i in range(self.num_workers):
            splits.append(splits[-1] + length // self.num_workers)
        splits[-1] = length
        ctx = mp.get_context('fork')
        processes = [ctx.Process(target=self._bachify_proc, args=(self.max_tokens, splits[i], splits[i+1], self.dataset)) for i in range(self.num_workers)]
        for p in processes:
            p.start()

        full = []

        for _ in range(self.num_workers):
            full.append(np.array(self.q.get()))
            time.sleep(1)
        for p in processes:
            p.terminate()
        return np.concatenate(full, 0)
        
    def _bachify_proc(self, max_tokens, start, end, dataset):
        batches = []
        cur = start
        fp0 = open(dataset.f0, 'rb', buffering=0)
        fp1 = open(dataset.f1, 'rb', buffering=0)
        while cur < end:
            current_len = 0
            start_cur = cur
            while cur < end:
                data = dataset[(fp0, fp1, cur)]
                diff =  data['len_src_sent'] + data['len_tgt_sent']
                if current_len + diff >= max_tokens:
                    batches.append([start_cur, cur])
                    break
                current_len += diff
                cur += 1
        if start_cur < end:
            batches.append([start_cur, end])
        fp0.close()
        fp1.close()
        # return np.array(batches)
        self.q.put(batches)
        while True:
            time.sleep(30)

    def __len__(self):
        return len(self.batches)


class DataCollector:
    def __init__(
        self,
        dataset,
        num_workers,
        rank,
        world_size,
        shuffle,
    ):
        ctx = mp.get_context('fork')
        self.dataset = dataset
        self.q = ctx.Queue(maxsize=num_workers * 32)
        self.finished = []
        self.processes = []
        self.rank = rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.step_size = ctx.Value('i', world_size * num_workers)
        self.shuffle = shuffle
        self.length = self._init_length()
        
    def __iter__(self):
        ctx = mp.get_context('fork')
        self.cleanup()
        if self.shuffle:
            self.dataset.shuffle()
        self.finished =  [ctx.Value('i', 0) for _ in range(self.num_workers)]
        self.processes = [ctx.Process(target=self.fill_queue, args=(i,  i + self.rank * self.num_workers, self.dataset)) for i in range(self.num_workers)]
        for p in self.processes:
            p.start()
        return self

    def fill_queue(self, pid, offset, dataset):
        with open(dataset.dataset.f0, 'rb', buffering=0) as fp0, \
                open(dataset.dataset.f1, 'rb', buffering=0) as fp1:
            for i in range(offset, len(dataset), self.step_size.value):
                res = dataset[(fp0, fp1, i)]
                self.q.put(res)
        with self.finished[pid].get_lock():
            self.finished[pid].value = 1
        while True:
            time.sleep(30)

    @property
    def processes_end(self):
        return all(v.value == 1 for v in self.finished)

    def __next__(self):
        if self.processes_end and self.q.qsize() == 0:
            self.cleanup()
            raise StopIteration
        return self.q.get()

    def __len__(self):
        return self.length

    def cleanup(self):
        self.clear_queue()
        for p in self.processes:
            p.terminate()
        self.finished = []
        self.processes = []

    def clear_queue(self):
        logger.log(f'clear queue of size {self.q.qsize()}')
        while self.q.qsize() > 0:
            self.q.get()

    def __del__(self):
        self.cleanup()
    
    def _init_length(self):
        length = 0
        for worker in range(self.num_workers):
            offset = self.rank * self.num_workers + worker
            for _ in range(offset, len(self.dataset), self.step_size.value):
                length += 1
        return length
