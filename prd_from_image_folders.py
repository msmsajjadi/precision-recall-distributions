# coding=utf-8
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import cv2
import hashlib

import numpy as np
import inception
import prd_score as prd


parser = argparse.ArgumentParser(
    description='Assessing Generative Models via Precision and Recall',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('real_dir', type=str,
                    help='directory containing reference images')
parser.add_argument('fake_dir', type=str, nargs='+',
                    help='directory containing images to be evaluated')
parser.add_argument('--num_clusters', type=int, default=20,
                    help='number of cluster centers to fit')
parser.add_argument('--num_angles', type=int, default=1001,
                    help='number of angles for which to compute PRD, must be '
                         'in [3, 1e6]')
parser.add_argument('--num_runs', type=int, default=10,
                    help='number of independent runs over which to average the '
                         'PRD data')
parser.add_argument('--plot_path', type=str, default=None,
                    help='path for final plot file (can be .png or .pdf)')
parser.add_argument('--cache_dir', type=str,
                    default='/tmp/prd_cache/',
                    help='cache directory')
parser.add_argument('--inception_path', type=str,
                    default='/tmp/prd_cache/inception.pb',
                    help='path to pre-trained Inception.pb file')

args = parser.parse_args()


def generate_inception_embedding(imgs, inception_path, layer_name='pool_3:0'):
    return inception.embed_images_in_inception(imgs, inception_path, layer_name)


def load_or_generate_inception_embedding(directory, cache_dir, inception_path):
    hash = hashlib.md5(directory).hexdigest()
    path = os.path.join(cache_dir, hash + '.npy')
    if os.path.exists(path):
        embeddings = np.load(path)
        return embeddings
    imgs = load_images_from_dir(directory)
    embeddings = generate_inception_embedding(imgs, inception_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(path, 'wb') as f:
        np.save(f, embeddings)
    return embeddings


def load_images_from_dir(directory, types=('png', 'jpg', 'bmp', 'gif')):
    paths = [os.path.join(directory, fn) for fn in os.listdir(directory)
             if os.path.splitext(fn)[-1][1:] in types]
    # images are in [0, 255]
    imgs = [cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            for path in paths]
    return np.array(imgs)


if __name__ == '__main__':
    real_embeddings = load_or_generate_inception_embedding(
        args.real_dir, args.cache_dir, args.inception_path)
    prd_data = []
    for directory in args.fake_dir:
        fake_embeddings = load_or_generate_inception_embedding(
            directory, args.cache_dir, args.inception_path)
        prd_data.append(prd.compute_prd_from_embedding(
            eval_data=fake_embeddings,
            ref_data=real_embeddings,
            num_clusters=args.num_clusters,
            num_angles=args.num_angles,
            num_runs=args.num_runs))
    prd.plot(prd_data, labels=args.fake_dir, out_path=args.plot_path)
