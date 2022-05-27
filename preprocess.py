#!/usr/bin/env python3

"""
Usage:
./preprocess.py --data_path maestro-v1.0.0/ --data_type original --problem score2perf_maestro_perf_conditional_aug_10x --out_path magenta_data2
./preprocess.py --data_path maestro-v1.0.0/ --data_type original --with_melody --problem score2perf_maestro_mel_perf_conditional_aug_10x --out_path magenta_data2
"""

import argparse
import copy
import h5py
import hashlib
import itertools
import json
import magenta
from magenta.models.score2perf import score2perf, datagen_beam, music_encoders
import multiprocessing
import note_seq
from note_seq.sequences_lib import stretch_note_sequence, transpose_note_sequence, augment_note_sequence
from os import PathLike
from pathlib import Path
import queue
import random
from tensor2tensor.data_generators import generator_utils
import tensorflow as tf

NUM_VELOCITY_BINS = 32
STEPS_PER_SECOND = 100
STEPS_PER_QUARTER = 16 # for future
MIN_PITCH = 21
MAX_PITCH = 108


SPLITS = ['train', 'validation', 'test']
# SPLITS = ['test']

TRANSPOSE_LIST = [-3, -2, -1, 0, 1, 2, 3]
STRETCH_LIST = [0.95, 0.975, 1.0, 1.025, 1.5]


# =============================================================================
# MAESTRO data loader
# =============================================================================


def load_from_tfrecord(prefix_path):
    kv_list = {split: [] for split in SPLITS}

    for split in SPLITS:
        filename = f'{prefix_path}/maestro-v1.0.0_{split}.tfrecord'
        dataset = tf.data.TFRecordDataset(filename)
        for ns_str in iter(dataset):
            ns_str = ns_str.numpy()
            ns = note_seq.NoteSequence.FromString(ns_str)
            kv = (ns.id, ns_str)
            kv_list[split].append(kv)

    return kv_list

def load_from_original(prefix_path):
    if isinstance(prefix_path, str):
        prefix_path = Path(prefix_path)
    elif not isinstance(prefix_path, PathLike):
        raise TypeError("prefix_path is neither str or os.PathLike")

    with open(f'{prefix_path}/maestro-v1.0.0.json', 'r') as f:
        dataset_info = json.load(f)

    kv_list = {split: [] for split in SPLITS}

    for data_info in dataset_info:
        filename = data_info['midi_filename']
        split = data_info['split']
        ns = note_seq.midi_file_to_note_sequence(prefix_path/filename)
        ns.filename = filename
        ns.id = '/id/midi/maestro-v1.0.0/' + hashlib.sha1(filename.encode('utf-8')).hexdigest()
        ns_str = ns.SerializeToString()
        kv = (ns.id, ns_str)
        kv_list[split].append(kv)

    return kv_list


# Generator version
def load_from_tfrecord_gen(prefix_path, split):
    filename = f'{prefix_path}/maestro-v1.0.0_{split}.tfrecord'
    dataset = tf.data.TFRecordDataset(filename)
    for ns_str in iter(dataset):
        ns_str = ns_str.numpy()
        ns = note_seq.NoteSequence.FromString(ns_str)
        yield ns.id, ns_str

def load_from_original_gen(prefix_path, split):
    if isinstance(prefix_path, str):
        prefix_path = Path(prefix_path)
    elif not isinstance(prefix_path, PathLike):
        raise TypeError("prefix_path is neither str or os.PathLike")

    with open(f'{prefix_path}/maestro-v1.0.0.json', 'r') as f:
        dataset_info = json.load(f)

    for data_info in dataset_info:
        if split != data_info['split']:
            continue
        filename = data_info['midi_filename']
        ns = note_seq.midi_file_to_note_sequence(prefix_path/filename)
        ns.filename = filename
        ns.id = '/id/midi/maestro-v1.0.0/' + hashlib.sha1(filename.encode('utf-8')).hexdigest()
        ns_str = ns.SerializeToString()
        yield ns.id, ns_str

"""
datagen_beam.ConditionalExtractExamplesDoFn(self, melody, noisy, encode_performance_fn, encode_score_fns,
               augment_fns, num_replications, *unused_args, **unused_kwargs)
"""


# =============================================================================
# downstream preprocessing code
# =============================================================================


def filter_invalid_notes_2(min_pitch, max_pitch, ns):
    """Filter notes with out-of-range pitch from NoteSequence protos."""
    valid_notes = [note for note in ns.notes
                if min_pitch <= note.pitch <= max_pitch]
    if len(valid_notes) < len(ns.notes):
        del ns.notes[:]
        ns.notes.extend(valid_notes)
    return ns

def quantize_note_sequence(ns, absolute_time_step=True):
    if absolute_time_step:
        return note_seq.quantize_note_sequence_absolute(ns, STEPS_PER_SECOND)
    else:
        return note_seq.quantize_note_sequence(ns, STEPS_PER_QUARTER)


# Encoders
def get_performance_encoder():
    # TODO figure out why the +1 exists
    return note_seq.MelodyOneHotEncoding(
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH+1)

def get_melody_encoder():
    return note_seq.PerformanceOneHotEncoding(
        num_velocity_bins=NUM_VELOCITY_BINS,
        max_shift_steps=STEPS_PER_SECOND,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH)

def encode_performance_sequence(ns, absolute_time_step=True, num_reserved_ids=2):
    encoder = get_performance_encoder()
    quantized_seq = quantize_note_sequence(ns, absolute_time_step)
    performance = note_seq.Performance(quantized_seq, num_velocity_bins=NUM_VELOCITY_BINS)

    event_ids = [encoding.encode_event(event) + num_reserved_ids for event in performance]
    # TODO compress events and add EOS token?
    return event_ids


def get_input_from_augmented_seq(ns_aug, performance_encoder, melody=False, melody_encoder=None, noisy=False):
    sample = {}

    perf_tokens = performance_encoder.encode_note_sequence(ns_aug)
    if len(perf_tokens) >= 2048:
        max_offset = len(perf_tokens) - 2048
        offset = random.randrange(max_offset + 1)
        perf_tokens = perf_tokens[offset:offset + 2048]
    sample['inputs'] = perf_tokens
    sample['targets'] = perf_tokens
    if not melody:
        return sample

    cropped_ns = performance_encoder.decode_to_note_sequence(perf_tokens)
    if melody:
        sample['performance'] = perf_tokens
        melody_inst = note_seq.infer_melody_for_sequence(cropped_ns)
        melody_seq = copy.deepcopy(cropped_ns)
        melody_notes = []
        for note in melody_seq.notes:
            if note.instrument == melody_inst:
                melody_notes.append(note)
        del melody_seq.notes[:]
        melody_seq.notes.extend(melody_notes)
        melody_tokens = melody_encoder.encode_note_sequence(melody_seq)
        sample['melody'] = melody_tokens
        del sample['inputs']

        if noisy:
            # ns_noisy = augment_note_sequence(ns_aug, 0.95, 1.05, -6, 6) # FIXME incorrect
            transpose_amount = 0
            try:
                all_pitches = [x.pitch for x in cropped_ns.notes]
                min_val = min(all_pitches)
                max_val = max(all_pitches)
                transpose_range = list(range(-(min_val - 21), 108 - max_val + 1))
                transpose_range.remove(0)  # make sure you transpose
                transpose_amount = random.choice(transpose_range)
            except:
                # TODO check we should just skip in this case
                pass
            ns_noisy, _ = transpose_note_sequence(decoded_ns, transpose_amount, MIN_PITCH, MAX_PITCH, in_place=False)
            sample['performance'] = performance_encoder.encode_note_sequence(ns_noisy)
    return sample


def get_inputs_from_seq(key, ns, augment_data, performance_encoder, melody=False, melody_encoder=None, noisy=False):
    assert performance_encoder
    if melody:
        assert melody_encoder

    if not melody and noisy:
        raise ValueError('noisy=True is only available for melody=True')

    # For reproducible results
    m = hashlib.md5(key.encode('utf-8'))
    random.seed(int(m.hexdigest(), 16))

    samples = []

    ns = filter_invalid_notes_2(MIN_PITCH, MAX_PITCH, ns)
    ns = note_seq.apply_sustain_control_changes(ns)
    del ns.control_changes[:]

    if augment_data:
        for _ in range(10):
            for transpose, stretch in itertools.product(TRANSPOSE_LIST, STRETCH_LIST):
                ns_trans, deleted_note_count = transpose_note_sequence(ns, transpose, MIN_PITCH, MAX_PITCH, in_place=False)
                # TODO check deleted_note_count
                if deleted_note_count:
                    continue
                ns_aug = stretch_note_sequence(ns_trans, stretch)
                yield get_input_from_augmented_seq(ns_aug, performance_encoder, melody, melody_encoder, noisy)
    else:
        yield get_input_from_augmented_seq(ns, performance_encoder, melody, melody_encoder, noisy)


def process_one(kv, augment_data):
    noisy = FLAGS.noisy and augment_data # Only apply noise injection to training dataset
    key, ns_str = kv
    print(key)
    ns = note_seq.NoteSequence.FromString(ns_str)
    samples = get_inputs_from_seq(key, ns, augment_data,
        performance_encoder, FLAGS.with_melody, melody_encoder, noisy)
    yield from samples

def process_part(kv_list, augment_data):
    for kv in kv_list:
        yield from process_one(kv, augment_data)

def write_tfrecord(data_iterator, out_name):
    with tf.io.TFRecordWriter(out_name) as writer:
        for sample in data_iterator:
            writer.write(generator_utils.to_example(sample).SerializeToString())
            writer.flush()

def write_hdf5(data_iterator, out_name):
    with h5py.File(out_name, 'w') as f:
        sample_count = 0
        for sample in data_iterator:
            for k, v in sample.items():
                f.create_dataset(f'{sample_count}/{k}', data=v)
            sample_count += 1
        f.create_dataset('sample_count', data=sample_count)

def worker_fn(queue_in, queue_out):
    for args in iter(queue_in.get, None):
        for result in process_one(*args):
            queue_out.put(result)
    queue_out.put(None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--data_type', required=True, choices=['tfrecord', 'original'])
    parser.add_argument('--problem', required=True)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--out_format', required=True, choices=['hdf5', 'tfrecord'])
    parser.add_argument('--with_melody', action='store_true')
    parser.add_argument('--noisy', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()

    Path(FLAGS.out_path).mkdir(parents=True, exist_ok=True)

    performance_encoder = music_encoders.MidiPerformanceEncoder(
        steps_per_second=STEPS_PER_SECOND,
        num_velocity_bins=NUM_VELOCITY_BINS,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH,
        add_eos=False)

    melody_encoder = music_encoders.TextMelodyEncoderAbsolute(
        steps_per_second=STEPS_PER_SECOND,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH)

    if FLAGS.data_type == 'tfrecord':
        loader_fn = load_from_tfrecord_gen
    elif FLAGS.data_type == 'original':
        loader_fn = load_from_original_gen
    else:
        assert False


    num_workers = multiprocessing.cpu_count()
    for split in SPLITS:
        split_for_output = split
        augment_data = False
        if split == 'train':
            augment_data = True
        if split == 'validation' and FLAGS.out_format == 'tfrecord':
            split_for_output = 'dev'

        results = []
        workers = []
        input_queue = multiprocessing.Queue(num_workers)
        result_queue = multiprocessing.Queue()
        for _ in range(num_workers):
            worker = multiprocessing.Process(target=worker_fn, args=(input_queue, result_queue))
            workers.append(worker)
            worker.start()

        print('loading split', split)
        for i, x in enumerate(loader_fn(FLAGS.data_path, split)):
            input_queue.put((x, augment_data))
            print(split, i)

        for i in range(num_workers):
            input_queue.put(None)

        done_count = 0
        while done_count < num_workers:
            result = result_queue.get()
            if result is None:
                done_count += 1
            else:
                results.append(result)

        for worker in workers:
            worker.join()

        random.shuffle(results)
        if FLAGS.out_format == 'tfrecord':
            out_path = f'{FLAGS.out_path}/{FLAGS.problem}-{split_for_output}.tfrecord'
            write_tfrecord(results, out_path)
        elif FLAGS.out_format == 'hdf5':
            out_path = f'{FLAGS.out_path}/{FLAGS.problem}-{split_for_output}.h5'
            write_hdf5(results, out_path)
        else:
            raise ValueError(f'Unknown output format {out_format}')

        print('processed', split)
        print(len(results), results[0].keys())
        input_queue.close()
        input_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()
