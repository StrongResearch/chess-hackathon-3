from argparse import ArgumentParser
from pathlib import Path
import torch
from threading import Thread
from queue import Queue

import numpy as np
from pathlib import Path
from random import shuffle
from tqdm import tqdm, trange
import deflate
import zstandard
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from numpy.random import default_rng
import sys
import tarfile


# Example usage
# dataset = LeelaDataset(
#    chunk_dir="/data",
#    batch_size=1024,
#    skip_factor=32,
#    num_workers=4,
#    shuffle_buffer_size=2 ** 19,
# )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, pin_memory=True)


# DEFAULT SETTINGS
HISTORY_LENGTH = 7
USE_REPETITION_PLANE = True
CASTLING_PLANES = 4
SIDE_TO_MOVE_PLANES = 1
USE_RULE_50 = True
USE_ZERO_PLANE = True
USE_ONE_PLANE = True


# MINIMAL SETTINGS
HISTORY_LENGTH = 0
USE_REPETITION_PLANE = False
CASTLING_PLANES = 4
SIDE_TO_MOVE_PLANES = 1
USE_RULE_50 = False
USE_ZERO_PLANE = True
USE_ONE_PLANE = True

NUM_BOARDS = (12 + USE_REPETITION_PLANE) * (1 + HISTORY_LENGTH) # 104
NUM_PLANES = NUM_BOARDS + CASTLING_PLANES + SIDE_TO_MOVE_PLANES + USE_RULE_50 + USE_ZERO_PLANE + USE_ONE_PLANE # 112
NUM_ACTIONS = 1858

RECORD_SIZE = 8356
ARRAY_SHAPES_WITHOUT_BATCH = [(NUM_PLANES, 8, 8), (NUM_ACTIONS,), (3,), (3,), (1,)]


def file_generator(file_list, random):
    zstd_context = zstandard.ZstdDecompressor()
    while True:
        if random:
            shuffle(file_list)
        else:
            file_list = sorted(file_list)
        for file in file_list:
            if file.name.endswith(".tar"):
                with tarfile.open(file, 'r') as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            if member.name.endswith(".gz"):
                                yield deflate.gzip_decompress(tar.extractfile(member).read())
                            elif file.name.endswith(".zst"):
                                yield zstd_context.decompress(tar.extractfile(member).read())
                            elif member.name=="LICENSE":
                                pass
                            else:
                                print("Unknown file type!", file.name)
            # yield gzip.open(file, 'rb').read()
            elif file.name.endswith(".gz"):
                yield deflate.gzip_decompress(file.read_bytes())
            elif file.name.endswith(".zst"):
                yield zstd_context.decompress(file.read_bytes())
            else:
                raise RuntimeError("Unknown file type!")


def extract_rule50_zero_one(raw):
    # Tested equivalent but there were a lot of zeros so I'm unsure
    # rule50 count plane.
    rule50_plane = (
        raw[:, 8277 : 8277 + 1].reshape(-1, 1, 1, 1).astype(np.float32) / 99.0
    )
    rule50_plane = np.tile(rule50_plane, [1, 1, 8, 8])
    # zero plane and one plane
    zero_plane = np.zeros_like(rule50_plane)
    one_plane = np.ones_like(rule50_plane)
    return rule50_plane, zero_plane, one_plane


def extract_byte_planes(raw):
    # Checked and confirmed equivalent to the existing extract_byte_planes
    # 5 bytes in input are expanded and tiled
    planes = raw[:, 8272 : 8272 + CASTLING_PLANES + SIDE_TO_MOVE_PLANES].reshape(-1, CASTLING_PLANES + SIDE_TO_MOVE_PLANES, 1, 1)
    unit_planes = np.tile(planes, [1, 1, 8, 8])
    return unit_planes


def extract_policy_bits(raw):
    # Checked and confirmed equivalent to the existing extract_policy_bits
    # Next 7432 are easy, policy extraction.
    policy = np.ascontiguousarray(raw[:, 8 : 8 + 7432]).view(dtype=np.float32)
    # Next are 104 bit packed chess boards, they have to be expanded.
    bit_planes = raw[:, 7440 : 7440 + NUM_BOARDS * 8].reshape((-1, NUM_BOARDS, 8))
    bit_planes = np.unpackbits(bit_planes, axis=-1).reshape((-1, NUM_BOARDS, 8, 8))
    return policy, bit_planes


def extract_outputs(raw):
    # Checked and confirmed equivalent to the existing extract_outputs
    # Result distribution needs to be calculated from q and d.
    z_q = np.ascontiguousarray(raw[:, 8308 : 8308 + 4]).view(dtype=np.float32)
    z_d = np.ascontiguousarray(raw[:, 8312 : 8312 + 4]).view(dtype=np.float32)
    z_q_w = 0.5 * (1.0 - z_d + z_q)
    z_q_l = 0.5 * (1.0 - z_d - z_q)

    z = np.concatenate((z_q_w, z_d, z_q_l), axis=1)

    # Outcome distribution needs to be calculated from q and d.
    root_q = np.ascontiguousarray(raw[:, 8280 : 8280 + 4]).view(dtype=np.float32)
    root_d = np.ascontiguousarray(raw[:, 8288 : 8288 + 4]).view(dtype=np.float32)

    best_q = np.ascontiguousarray(raw[:, 8284 : 8284 + 4]).view(dtype=np.float32)
    best_d = np.ascontiguousarray(raw[:, 8292 : 8292 + 4]).view(dtype=np.float32)

    result_q = np.ascontiguousarray(raw[:, 8308 : 8308 + 4]).view(dtype=np.float32)
    result_d = np.ascontiguousarray(raw[:, 8312 : 8312 + 4]).view(dtype=np.float32)

    played_q = np.ascontiguousarray(raw[:, 8316 : 8316 + 4]).view(dtype=np.float32)
    played_d = np.ascontiguousarray(raw[:, 8320 : 8320 + 4]).view(dtype=np.float32)

    orig_q = np.ascontiguousarray(raw[:, 8328 : 8328 + 4]).view(dtype=np.float32)
    orig_d = np.ascontiguousarray(raw[:, 8332 : 8332 + 4]).view(dtype=np.float32)

    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)
    
    root_q_w = 0.5 * (1.0 - root_d + root_q)
    root_q_l = 0.5 * (1.0 - root_d - root_q)

    result_q_w = 0.5 * (1.0 - result_d + result_q)
    result_q_l = 0.5 * (1.0 - result_d - result_q)

    played_q_w = 0.5 * (1.0 - best_d + best_q)
    played_q_l = 0.5 * (1.0 - best_d - best_q)

    orig_q_w = 0.5 * (1.0 - orig_d + orig_q)
    orig_q_l = 0.5 * (1.0 - orig_d - orig_q)

    best_q = np.concatenate((best_q_w, best_d, best_q_l), axis=1)
    root_q = np.concatenate((root_q_w, root_d, root_q_l), axis=1)
    result_q = np.concatenate((result_q_w, result_d, result_q_l), axis=1)
    played_q = np.concatenate((played_q_w, played_d, played_q_l), axis=1)
    orig_q = np.concatenate((orig_q_w, orig_d, orig_q_l), axis=1)

    ply_count = np.ascontiguousarray(raw[:, 8304 : 8304 + 4]).view(dtype=np.float32)
    return z, best_q, root_q, result_q, played_q, orig_q, ply_count


def extract_inputs_outputs_if1(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 4 change how we construct some of the unit planes.
    policy, bit_planes = extract_policy_bits(raw)

    # Next 5 are castling + stm, all of which simply copy the byte value to all squares.
    unit_planes = extract_byte_planes(raw).astype(np.float32)

    rule50_plane, zero_plane, one_plane = extract_rule50_zero_one(raw)

    inputs = np.concatenate(
        [
            bit_planes,
            unit_planes,
            #rule50_plane,
            zero_plane,
            one_plane
    ], 1
    ).reshape([-1, NUM_PLANES, 8, 8])

    z, best_q, root_q, result_q, played_q, orig_q, ply_count = extract_outputs(raw)

    #return inputs, policy, z, best_q, root_q, result_q, played_q, orig_q, ply_count
    return inputs, policy, z, orig_q, ply_count


def offset_generator(batch_size, record_size, skip_factor, random):
    # The offset generator is a generator that yields batch_size random offsets
    # from a range up to batch_size * skip_factor
    initial_offset = 0
    rng = default_rng()
    while True:
        if random:
            retained_indices = rng.choice(
                batch_size * skip_factor, size=batch_size, replace=False
            )
        else:
            retained_indices = np.array([i * skip_factor for i in range(batch_size)])
        retained_indices = np.sort(retained_indices)
        next_offset = (
            batch_size * skip_factor - retained_indices[-1]
        )  # Bump us up to the end of the current skip-batch
        skip_offsets = np.diff(retained_indices, prepend=0)
        skip_offsets[0] += initial_offset
        for offset in skip_offsets:
            yield offset * record_size
        initial_offset = next_offset


def data_generator(files, batch_size, skip_factor, validation):
    # This is a singlethreaded generator for debugging
    file_gen = file_generator(files, random=not validation)
    offset_gen = offset_generator(
        batch_size=batch_size,
        record_size=RECORD_SIZE,
        skip_factor=skip_factor,
        random=not validation,
    )
    data = np.zeros((batch_size, RECORD_SIZE), dtype=np.uint8)
    current_file = next(file_gen)
    file_ptr = 0
    data_ptr = 0
    offset = next(offset_gen)
    while True:
        if offset + file_ptr < len(current_file):
            data[data_ptr] = np.frombuffer(
                current_file[offset + file_ptr : offset + file_ptr + RECORD_SIZE],
                dtype=np.uint8,
            )
            data_ptr += 1
            if data_ptr == batch_size:
                yield extract_inputs_outputs_if1(data)
                data.fill(0)
                data_ptr = 0
            file_ptr += offset + RECORD_SIZE
            offset = next(offset_gen)
        else:
            offset -= len(current_file) - file_ptr
            current_file = next(file_gen)
            file_ptr = 0


def data_worker(
    files,
    batch_size,
    skip_factor,
    array_ready_event,
    main_process_access_event,
    shared_array_names,
    validation,
):
    shared_mem = [SharedMemory(name=name, create=False) for name in shared_array_names]
    array_shapes = [[batch_size] + list(shape) for shape in ARRAY_SHAPES_WITHOUT_BATCH]
    shared_arrays = [
        np.ndarray(shape, dtype=np.float32, buffer=mem.buf)
        for shape, mem in zip(array_shapes, shared_mem)
    ]
    file_gen = file_generator(files, random=not validation)
    offset_gen = offset_generator(
        batch_size=batch_size,
        record_size=RECORD_SIZE,
        skip_factor=skip_factor,
        random=not validation,
    )
    data = np.zeros((batch_size, RECORD_SIZE), dtype=np.uint8)
    current_file = next(file_gen)
    file_ptr = 0
    data_ptr = 0
    offset = next(offset_gen)
    while True:
        if offset + file_ptr < len(current_file):
            data[data_ptr] = np.frombuffer(
                current_file[offset + file_ptr : offset + file_ptr + RECORD_SIZE],
                dtype=np.uint8,
            )
            data_ptr += 1
            if data_ptr == batch_size:
                processed_batch = extract_inputs_outputs_if1(data)
                main_process_access_event.wait()
                main_process_access_event.clear()
                for batch_array, shared_array in zip(processed_batch, shared_arrays):
                    shared_array[:] = batch_array
                    array_ready_event.set()
                data.fill(0)
                data_ptr = 0
            file_ptr += offset + RECORD_SIZE
            offset = next(offset_gen)
        else:
            offset -= len(current_file) - file_ptr
            current_file = next(file_gen)
            file_ptr = 0


def multiprocess_generator(
    chunk_dir,
    batch_size,
    num_workers,
    skip_factor,
    shuffle_buffer_size,
    validation=False,
):
    assert shuffle_buffer_size % batch_size == 0  # This simplifies my life later on
    print("Scanning directory for game data chunks...")
    print("!!", chunk_dir)
    files = list(tqdm(chunk_dir.glob("**/*"), desc="Files scanned", unit=" files")) + list(tqdm(chunk_dir.glob("*"), desc="Files scanned", unit=" files"))
    files = [file for file in files if file.suffix in (".gz", ".zst", ".tar")]
    if len(files) == 0:
        raise FileNotFoundError("No valid input files!")
    print(f"{len(files)} matching files.")
    print("Done!")
    #raise Exception
    if validation:
        files = sorted(files)
    else:
        shuffle(files)
    worker_file_lists = [files[i::num_workers] for i in range(num_workers)]
    ctx = get_context("spawn")  # For Windows compatibility
    array_ready_events = []
    main_process_access_events = []
    shared_arrays = []
    shared_mem = []
    array_shapes = [[batch_size] + list(shape) for shape in ARRAY_SHAPES_WITHOUT_BATCH]
    array_sizes = [int(np.prod(shape)) * 4 for shape in array_shapes]
    shuffle_buffer_shapes = [
        [shuffle_buffer_size] + list(shape[1:]) for shape in array_shapes
    ]
    shuffle_buffers = [
        np.zeros(shape=shape, dtype=np.float32) for shape in shuffle_buffer_shapes
    ]

    for i in trange(num_workers, desc="Initializing worker processes"):
        array_ready_event = ctx.Event()
        main_process_access_event = ctx.Event()
        main_process_access_event.set()
        array_ready_events.append(array_ready_event)
        main_process_access_events.append(main_process_access_event)
        process_shared_mem = [
            SharedMemory(size=size, create=True) for size in array_sizes
        ]
        process_shared_arrays = [
            np.ndarray(
                array_shapes[i], dtype=np.float32, buffer=process_shared_mem[i].buf
            )
            for i in range(len(array_shapes))
        ]
        shared_mem.append(process_shared_mem)
        shared_arrays.append(process_shared_arrays)
        shared_mem_names = [mem.name for mem in process_shared_mem]
        process = ctx.Process(
            target=data_worker,
            kwargs={
                "files": worker_file_lists[i],
                "skip_factor": skip_factor,
                "batch_size": batch_size,
                "array_ready_event": array_ready_event,
                "main_process_access_event": main_process_access_event,
                "shared_array_names": shared_mem_names,
                "validation": validation,
            },
            daemon=True,
        )
        process.start()

    for i in trange(shuffle_buffer_size // batch_size, desc="Filling shuffle buffer"):
        proc = i % num_workers
        array_ready_events[proc].wait()
        for array, shuffle_buffer in zip(shared_arrays[proc], shuffle_buffers):
            shuffle_buffer[i * batch_size : (i + 1) * batch_size] = array
        array_ready_events[proc].clear()
        main_process_access_events[proc].set()

    rng = default_rng()
    while True:
        for array_ready_event, main_process_access_event, shared_arrs in zip(
            array_ready_events, main_process_access_events, shared_arrays
        ):
            if not array_ready_event.is_set():
                continue
            random_indices = rng.choice(
                shuffle_buffer_size, size=batch_size, replace=False
            )
            # I tried using np.take() to fill pre-allocated arrays, but it wasn't any faster
            batch = tuple(
                [shuffle_buffer[random_indices] for shuffle_buffer in shuffle_buffers]
            )
            yield batch
            for arr, shuffle_buffer in zip(shared_arrs, shuffle_buffers):
                shuffle_buffer[random_indices] = arr
            array_ready_event.clear()
            main_process_access_event.set()


def make_callable(chunk_dir, batch_size, num_workers, skip_factor, shuffle_buffer_size):
    # Because tf.data needs to be able to reinitialize
    def return_gen():
        return multiprocess_generator(
            chunk_dir=chunk_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_factor=skip_factor,
            shuffle_buffer_size=shuffle_buffer_size,
        )

    return return_gen


def queued_generator(queue, **kwargs):
    generator = multiprocess_generator(**kwargs)
    for batch in generator:
        batch = [torch.from_numpy(tensor).pin_memory() for tensor in batch]
        queue.put(batch)


class LeelaDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, **kwargs
    ):
        self.queue = Queue(maxsize=4)
        kwargs['queue'] = self.queue
        self.thread = Thread(target=queued_generator, kwargs=kwargs, daemon=True)
        self.thread.start()

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:
        #     raise RuntimeError("This dataset does multiprocessing internally, and should only have a single torch worker!")
        return self

    def __next__(self):
        return self.queue.get(block=True)