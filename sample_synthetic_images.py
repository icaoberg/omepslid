#!/usr/bin/env python2
"""
This script samples synthetic images according to various criteria from an OMERO
server running the syntheticsampler webapp.
"""


from __future__ import print_function

import argparse
import json
import os
import random
import sys
import urllib
import urllib2


OMERO_ROOT = "http://omero.compbio.cs.cmu.edu:8080/"
IDS_ENDPOINT = OMERO_ROOT + "syntheticsampler/image_ids/"
IMAGES_ENDPOINT = OMERO_ROOT + "syntheticsampler/images/"


def get_args():
  """
  Make an argparser and get the arguments from the command line. Exits on bad
  CLI args.
  """
  parser = argparse.ArgumentParser(
    description="Download synthetic images from an OMERO server",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--dimensionality", default="2D", help="2D or 3D")
  parser.add_argument("--cell-line", default="HeLa", help="type of cell line")
  parser.add_argument("--downsampling-factor",
    help="amount to downsample the images by using bicubic interpolation")
  parser.add_argument("--number-of-images", type=int, default=1,
    help="number of images to download")
  parser.add_argument("--channel0-model-class",
    help="model class for channel 0 (e.g. framework)")
  parser.add_argument("--channel0-model-type",
    help="model type for channel 0 (e.g. cylindrical, diffeomorphic)")
  parser.add_argument("--channel1-model-class",
    help="model class for channel 1 (e.g. framework)")
  parser.add_argument("--channel1-model-type",
    help="model type for channel 1 (e.g. ratio, diffeomorphic)")
  parser.add_argument("--channel2-model-class",
    help="model class for channel 2 (e.g. vesicles, microtubule)")
  parser.add_argument("--channel2-model-type",
    help="model type for channel 2 (e.g. gmm, network)")
  parser.add_argument("--output-dir", default=".",
    help="directory to save the images in")
  return parser.parse_args()

def check_downsampling_factor(downsampling_factor_arg):
  """
  Check that the provided downsampling factor is either None or a number greater
  than 1.
  """
  if (downsampling_factor_arg is not None):
    try:
      downsampling_factor = float(downsampling_factor_arg)
    except:
      msg = "Downsampling factor must be a floating point number (got {})."
      error_exit(msg.format(str(downsampling_factor_arg)))
    if downsampling_factor <= 1:
      msg = "Downsampling factor must be greater than 1 (got {:f})."
      error_exit(msg.format(downsampling_factor))

def print_inline(s):
  """
  Return the cursor to the beginning of the line after printing.
  """
  print(s, end='\r')
  sys.stdout.flush()

def get_matching_image_ids(
    dimensionality=None,
    cell_line=None,
    channel0_model_class=None,
    channel0_model_type=None,
    channel1_model_class=None,
    channel1_model_type=None,
    channel2_model_class=None,
    channel2_model_type=None
  ):
  """
  Ask the OMERO server for all ids of images that match the given criteria
  (passed as query parameters).
  """
  query_options = {
    'dimensionality': dimensionality,
    'cell_line': cell_line,
    'channel0_model_class': channel0_model_class,
    'channel0_model_type': channel0_model_type,
    'channel1_model_class': channel1_model_class,
    'channel1_model_type': channel1_model_type,
    'channel2_model_class': channel2_model_class,
    'channel2_model_type': channel2_model_type
  }
  none_keys = set()
  for key in query_options:
    if query_options[key] is None:
      none_keys.add(key)
  for none_key in none_keys:
    del query_options[none_key]
  query_string = urllib.urlencode(query_options)
  try:
    response = json.load(urllib2.urlopen(IDS_ENDPOINT + "?" + query_string))
    return response['data']
  except Exception as e:
    err_msg = "Failed to retrieve the ids: {}"
    error_exit(err_msg.format(e))

def error_exit(s):
  """
  Print the message and exit with non-zero status.
  """
  print(s)
  sys.exit(1)

def download_image(image_id, downsampling_factor, output_path,
    percentage_template):
  """
  Download the image with the given id from the OMERO server, requesting that it
  be downsampled by the given amount (if any), and save to the provided output
  path.
  """
  image_url = IMAGES_ENDPOINT + str(image_id)
  if downsampling_factor:
    image_url += "?downsampling_factor=" + downsampling_factor
  response = urllib2.urlopen(image_url)
  total_size = int(response.info().getheader('Content-Length').strip())
  so_far = 0
  chunk_read_size = 16 * 1024
  chunk = response.read(chunk_read_size)
  with open(output_path, 'wb') as f:
    while chunk:
      f.write(chunk)
      so_far += len(chunk)
      percentage = round((float(so_far) / total_size) * 100, 2)
      print_inline(percentage_template.format(percentage))
      chunk = response.read(chunk_read_size)

def get_synthetic_images(
    dimensionality='2D',
    cell_line='HeLa',
    downsampling_factor=None,
    number_of_images=1,
    channel0_model_class=None,
    channel0_model_type=None,
    channel1_model_class=None,
    channel1_model_type=None,
    channel2_model_class=None,
    channel2_model_type=None,
    output_dir='out'
  ):
  """
  Sample from an image collection in omero.compbio.cs.cmu.edu and saves the
  image(s) to disk.

  For a full list of model classess and types visit www.cellorganizer.org

  :param dimensionality: image dimensionality
  :type dimensionality: string
  :param cell_line: image cell lines. Default is HeLa.
  :type cell_line: string
  :param downsampling_factor: amount to downsample the images (bicubic
                              interpolation)
  :type downsampling_factor: float|None
  :param number_of_images: number of synthesized images
  :type number_of_images: int
  :param channel0_model_class: model class
  :type channel0_model_class: string
  :param channel0_model_type: model type
  :type channel0_model_type: string
  :param channel1_model_class: model class
  :type channel1_model_class: string
  :param channel1_model_type: model type
  :type channel1_model_type: string
  :param channel2_model_class: model class
  :type channel2_model_class: string
  :param channel2_model_type: model type
  :type channel2_model_type: string
  :param output_dir: directory for saving the images
  :type output_dir: string
  """
  print_inline("Getting matching ids...")
  all_matching_image_ids = get_matching_image_ids(
    dimensionality=dimensionality,
    cell_line=cell_line,
    channel0_model_class=channel0_model_class,
    channel0_model_type=channel0_model_type,
    channel1_model_class=channel1_model_class,
    channel1_model_type=channel1_model_type,
    channel2_model_class=channel2_model_class,
    channel2_model_type=channel2_model_type
  )
  if len(all_matching_image_ids) < number_of_images:
    err = "{} ids requested; only {} ids matching the criteria were found."
    error_exit(err.format(number_of_images, len(all_matching_image_ids)))
  image_ids = random.sample(all_matching_image_ids, number_of_images)
  s = "Downloading {{}}... {{{{:3.2f}}}}% [{{:{}}}/{}]      "
  status_template = s.format(len(str(number_of_images)), number_of_images)
  for (i, image_id) in enumerate(image_ids):
    output_path = os.path.join(output_dir, "{}.tif".format(i + 1))
    percentage_template = status_template.format(output_path, i + 1)
    download_image(
      image_id, downsampling_factor, output_path, percentage_template
    )
  pad = " " * len(status_template)
  print("Downloaded {} images.{}".format(number_of_images, pad))

def main():
  """
  Parse the command line options and call get_synthetic_images with the passed
  options.
  """
  args = get_args()
  check_downsampling_factor(args.downsampling_factor)
  get_synthetic_images(
    dimensionality=args.dimensionality,
    cell_line=args.cell_line,
    downsampling_factor=args.downsampling_factor,
    number_of_images=args.number_of_images,
    channel0_model_class=args.channel0_model_class,
    channel0_model_type=args.channel0_model_type,
    channel1_model_class=args.channel1_model_class,
    channel1_model_type=args.channel1_model_type,
    channel2_model_class=args.channel2_model_class,
    channel2_model_type=args.channel2_model_type,
    output_dir=args.output_dir
  )

if __name__ == "__main__":
  main()
