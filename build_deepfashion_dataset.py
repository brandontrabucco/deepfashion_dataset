"""Author: Brandon Trabucco
Builds the deep fashion dataset into tensorflow sequence examples.
Usage: python build_deepfashion_dataset.py --output_dir="./" --all_dir="./" --embeddings_dir="./embeddings/"
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from collections import Counter
from collections import namedtuple
from datetime import datetime
import json
import os.path
import random
import sys
import threading


import nltk.tokenize
import numpy as np
import tensorflow as tf
import glove
import glove.configuration


tf.flags.DEFINE_string("all_dir", "./",
                       "Training image directory.")
tf.flags.DEFINE_string("embeddings_dir", "./embeddings/",
                       "Directory containing GloVe word embedidngs txt.")
tf.flags.DEFINE_string("output_dir", "./", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS


class ImageMetadata(object):
    """Helper class for organizing images by annotations.
    """
    
    def __init__(self, partition, filename, category, attributes):
        self.partition = partition
        self.filename = filename
        self.category = category
        self.attributes = attributes


class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(image, decoder, vocab):
    """Builds a SequenceExample proto for an image-caption pair.

    Args:
        image: An ImageMetadata object.
        decoder: An ImageDecoder object.
        vocab: A Vocabulary object.

    Returns:
        A SequenceExample proto.
    """
    with tf.gfile.FastGFile(image.filename, "rb") as f:
        encoded_image = f.read()

    try:
        decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image.filename)
        return

    context = tf.train.Features(feature={
        "image/filename": _bytes_feature(bytes(image.filename, "utf-8")),
        "image/data": _bytes_feature(encoded_image),
        "image/category": _int64_feature(vocab.word_to_id(image.category.strip().lower())),
    })
    attribute_ids = [vocab.word_to_id(a.strip().lower()) for a in image.attributes]
    feature_lists = tf.train.FeatureLists(feature_list={
        "image/attributes": _int64_feature_list(attribute_ids)
    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _process_image_files(thread_index, ranges, name, images, decoder, vocab,
                         num_shards):
    """Processes and saves a subset of images as TFRecord files in one thread.

    Args:
        thread_index: Integer thread identifier within [0, len(ranges)].
        ranges: A list of pairs of integers specifying the ranges of the dataset to
            process in parallel.
        name: Unique identifier specifying the dataset.
        images: List of ImageMetadata.
        decoder: An ImageDecoder object.
        vocab: A Vocabulary object.
        num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]

            sequence_example = _to_sequence_example(image, decoder, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                    (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d images to %s" %
            (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d images to %d shards." %
        (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
    """Processes a complete data set and saves it as a TFRecord.

    Args:
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
    """

    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(images)

    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d images in data set '%s'." %
        (datetime.now(), len(images), name))


def _load_and_process_metadata(all_dir):
    """Loads image metadata from txt files and process annotations.

    Args:
        all_dir: Directory containing the deep fashion dataset.

    Returns:
        A list of ImageMetadata.
    """
    
    # Load the partitions.
    with tf.gfile.FastGFile(os.path.join(all_dir, "Eval/list_eval_partition.txt"), "r") as f:
        partition_lines = f.readlines()
        
    # Read the header information.
    num_images = int(partition_lines.pop(0).strip())
    header_part = partition_lines.pop(0).split()
    assert len(partition_lines) == num_images, "Wrong number of images."
    
    # Load the dataset partitions into memory.
    images_hashmap = {}
    for element in partition_lines:
        columns = element.split()
        filename = columns[0]
        partition = columns[1]
        images_hashmap[filename] = ImageMetadata(
            partition=partition, filename=filename, category=None, attributes=[])
        
    # Load the category names into memory.
    with tf.gfile.FastGFile(os.path.join(all_dir, "Anno/list_category_cloth.txt"), "r") as f:
        category_name_lines = f.readlines()
        
    # Process the names of categories.
    num_cats = int(category_name_lines.pop(0).strip())
    header_cat = category_name_lines.pop(0).split()
    assert len(category_name_lines) == num_cats, "Wrong number of categories."
    category_names = list(map((lambda s: s.split()[0].lower()), category_name_lines))
       
    # Load the category labels into memory.
    with tf.gfile.FastGFile(os.path.join(all_dir, "Anno/list_category_img.txt"), "r") as f:
        category_lines = f.readlines()
        
    # Check if we have loaded properly.
    num_images = int(category_lines.pop(0).strip())
    header_cat = category_lines.pop(0).split()
    assert len(category_lines) == num_images, "Wrong number of images."
    
    # Assign the category to existing metadata
    for element in category_lines:
        columns = element.split()
        filename = columns[0]
        category = category_names[int(columns[1])]
        images_hashmap[filename].category = category
        
    # Load the attributes names into memory.
    with tf.gfile.FastGFile(os.path.join(all_dir, "Anno/list_attr_cloth.txt"), "r") as f:
        attribute_name_lines = f.readlines()
        
    # Process attribute names.
    num_attrs = int(attribute_name_lines.pop(0).strip())
    header_attr = attribute_name_lines.pop(0).split()
    assert len(attribute_name_lines) == num_attrs, "Wrong number of attributes."
    attribute_names = list(map((lambda s: s.split()[0].lower()), attribute_name_lines))
        
    # Load the attribute labels into memory.
    with tf.gfile.FastGFile(os.path.join(all_dir, "Anno/list_attr_img.txt"), "r") as f:
        attribute_lines = f.readlines()
        
    # Check images are all loaded.
    num_images = int(attribute_lines.pop(0).strip())
    header_attr = attribute_lines.pop(0).split()
    assert len(attribute_lines) == num_images, "Wrong number of images."
    
    # Assign attribute labels to existing image metadata.
    for element in attribute_lines:
        columns = element.split()
        filename = columns[0]
        attributes = [attribute_names[int(x)] for x in columns[1:]]
        images_hashmap[filename].attributes = attributes
        
    # Separate the partitions based on name.
    train_image_metadata = [value for value in images_hashmap.values() if value.partition == "train"]
    val_image_metadata = [value for value in images_hashmap.values() if value.partition == "val"]
    test_image_metadata = [value for value in images_hashmap.values() if value.partition == "test"]
        
    print("Finished processing %d images from %s" %
        (num_images, all_dir))

    return train_image_metadata, val_image_metadata, test_image_metadata


def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
    assert _is_valid_num_shards(FLAGS.test_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # Load image metadata from caption files.
    train_dataset, val_dataset, test_dataset = _load_and_process_metadata(FLAGS.all_dir)

    # Create vocabulary from the glove embeddings.
    config = glove.configuration.Configuration(
        embedding=300,
        filedir=FLAGS.embeddings_dir,
        length=70000,
        start_word="<S>",
        end_word="</S>",
        unk_word="<UNK>")
    vocab = glove.load(config)[0]

    _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
    _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
    _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)


if __name__ == "__main__":
    tf.app.run()