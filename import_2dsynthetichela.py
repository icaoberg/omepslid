#!/usr/bin/env python2
"""
This script takes a path to a folder with the 2D Hela Synthetic image files,
merges them appropriately into a single file, and uploads this to the specified
OMERO server.

Each cell and nucleus image file corresponds to several images with each
different channel type (e.g. endosome, lysosome, ...).
Cell and nucleus files are named like:
  2Dhela<cell_number>cell.tif
  (e.g. 2Dhela1cell.tif)
and:
  2Dhela<cell_number>nucleus.tif
  (e.g. 2Dhela1nucleus.tif)
respectively. The other channel image files are named like:
  2Dhela<cell_number><channel_type><image_number>.tif
  (e.g. 2Dhela1lysosome1.tif)
so one image would, for example, have channels corresponding to files:
  2Dhela1cell.tif
  2Dhela1nucleus.tif
  2Dhela1endosome1.tif
and another image would correspond to:
  2Dhela1cell.tif
  2Dhela1nucleus.tif
  2Dhela1endosome2.tif
and another corresponding to:
  2Dhela1cell.tif
  2Dhela1nucleus.tif
  2Dhela1lysosome1.tif
"""


from __future__ import print_function

import argparse
import getpass
import glob
import os
import re
import string
import sys

import numpy
try:
  import omero
except ImportError as e:
  print("ImportError: {}".format(e))
  print("Don't forget to set your PYTHONPATH to include the OMERO.py library!")
  sys.exit(1)
import omero.gateway
import omero.rtypes
from PIL import Image


DEFAULT_HOST = 'localhost'
DEFAULT_PORT = '4064'
DEFAULT_USERNAME = 'root'
DEFAULT_PASSWORD = 'omero'

DATASETS = [
  {
    'name' : 'Lysosome',
    'infix' : 'lysosome'
  }, {
    'name': 'Mitochondrion',
    'infix' : 'mitochondrion'
  }, {
    'name' : 'Nucleolus',
    'infix': 'nucleolus'
  } , {
    'name' : 'Endosome',
    'infix' : 'endosome'
  }
]

CHANNELS = [
  'Cell membrane',
  'DNA',
  'protein'
]

TAG_ANNOTATIONS = dict()


def get_args():
  """
  Parse the command line args using the argparse library.
  """
  parser = argparse.ArgumentParser(
    description="2D Synthetic HeLa collection uploader"
  )
  parser.add_argument("-t", "--test", action="store_true",
    help="upload a single image as a test")
  parser.add_argument("path", help="path to the 2D synthetic HeLa collection")
  return parser.parse_args()  # exits on malformed arguments

def error_exit(msg):
  """
  Print the specified message and exit with non-zero status.
  """
  print(msg)
  sys.exit(1)

def get_cell_i_from_other_tif_name(tif_name):
  """
  Given a tif name like
    /users/foo/2Dhela5lysosome7.tif
  get the 5, which is the index of the cell this tif corresponds to.
  """
  without_extension = tif_name[:-4]
  without_i = without_extension.rstrip(string.digits)
  without_name = without_i.rstrip(string.letters)
  return int(without_name[without_name.rindex("2Dhela") + 6:])

def check_structure(data_path):
  """
  Make sure the directory exists and that each cell file has a nucleus file (and
  vice versa). Also check that all the extra channel images have a cell file.
  Note that we don't check number of images or continuity of the indices.
  """
  if not os.path.isdir(data_path):
    error_exit("'%s' is not a valid directory.".format(data_path))
  all_tifs = set(glob.glob(os.path.join(data_path, "2Dhela*.tif")))
  cell_tifs = set(glob.glob(os.path.join(data_path, "2Dhela*cell.tif")))
  nucleus_tifs = set(glob.glob(os.path.join(data_path, "2Dhela*nucleus.tif")))
  other_tifs = all_tifs.difference(cell_tifs.union(nucleus_tifs))
  # Check each cell.tif has a nucleus.tif and vice versa
  for cell_tif in cell_tifs:
    expected_nucleus_tif = cell_tif[:-8] + "nucleus.tif"
    if expected_nucleus_tif not in nucleus_tifs:
      err_template = "Found cell tif '{}' without matching nucleus tif."
      error_exit(err_template.format(cell_tif))
  for nucleus_tif in nucleus_tifs:
    expected_cell_tif = nucleus_tif[:-11] + "cell.tif"
    if expected_cell_tif not in cell_tifs:
      err_template = "Found nucleus tif '{}' without matching cell tif."
      error_exit(err_template.format(nucleus_tif))
  # Check each other channel tif has a cell.tif
  for other_tif in other_tifs:
    cell_i = get_cell_i_from_other_tif_name(other_tif)
    cell_i_i = other_tif.rindex("2Dhela") + 6
    expected_cell_tif = other_tif[:cell_i_i] + str(cell_i) + "cell.tif"
    if expected_cell_tif not in cell_tifs:
      err_template = "Found other tif '{}' without matching cell tif."
      error_exit(err_template.format(other_tif))

def prompt_for(item, default, secret=False):
  """
  Ask the user to provide item, defaulting to default. Don't echo back input if
  secret.
  """
  prompt = "{:s} [{:s}]: ".format(item, default)
  if secret:
    s = getpass.getpass(prompt)
  else:
    s = raw_input(prompt)
  if s == "":
    return default
  else:
    return s

def get_omero_connection():
  """
  Prompt the user for OMERO connection information, defaulting to the parameters
  defined in the constants at the top.
  """
  host = prompt_for("OMERO host", DEFAULT_HOST)
  port = prompt_for("OMERO port", DEFAULT_PORT)
  username = prompt_for("OMERO username", DEFAULT_USERNAME)
  password = prompt_for("OMERO password", DEFAULT_PASSWORD, secret=True)
  conn = omero.gateway.BlitzGateway(username, password, host=host, port=port)
  conn.connect()
  return conn

def print_inline(s):
  """
  Print the provided string on the same line, flushing output so it's displayed.
  """
  print('\r' + s, end='')
  sys.stdout.flush()

def organize_by_cell_i(tifs):
  """
  Given a list of tifs, return a dictionary mapping a cell index to the tifs
  that correspond to that cell index.
  """
  cell_i_to_tifs = dict()
  for tif in tifs:
    cell_i = get_cell_i_from_other_tif_name(tif)
    if cell_i in cell_i_to_tifs:
      cell_i_to_tifs[cell_i].append(tif)
    else:
      cell_i_to_tifs[cell_i] = [tif]
  return cell_i_to_tifs

def get_i_from_other_tif(tif):
  """
  Given a tif name like
    /users/foo/2Dhela5lysosome7.tif
  get the 7.
  """
  without_extension = tif[:-4]
  return int(without_extension[len(without_extension.rstrip(string.digits)):])

def merge_image_channel_files(tif_name, infix):
  """
  Given the name of the tif being uploaded (not the cell or nucleus, but one of
  the others with two indices), get the cell and nucleus image and merge them
  with the tif being uploaded into a single plane generator which is returned.

  The plane generator should return an x-y plane of values for each z, channel,
  and time combination (in that order). In this case, we only have one z and
  time, but multiple channels, so we return a generator that returns a plane for
  each channel in order.
  """
  prefix = tif_name[:tif_name.rindex(infix)]
  def plane_generator():
    for channel in CHANNELS:
      if channel == "Cell membrane":
        channel_path = prefix + "cell.tif"
      elif channel == "DNA":
        channel_path = prefix + "nucleus.tif"
      elif channel == "protein":
        channel_path = tif_name
      yield numpy.array(Image.open(channel_path))
  return plane_generator

def get_omero_name_from_tif_name(tif_name):
  """
  Given a tif name, like '/foo/blah/2Dhela1endosome1.tif', return a string like
  '01endosome01'.
  """
  unique_part = tif_name[tif_name.rindex("2Dhela") + 6:-4]
  [cell_i, image_i] = map(int, re.findall(r"\d+", unique_part))
  protein = unique_part.strip(string.digits)
  return "{:02d}{}{:02d}".format(cell_i, protein, image_i)

def upload_image(plane_generator, name, conn, c_count):
  """
  Upload the image (stored in the plane_generator as the OMERO API specifies)
  using the OMERO connection conn.
  """
  z_count = 1
  channel_count = c_count
  time_count = 1
  return conn.createImageFromNumpySeq(
      plane_generator(), name, z_count, channel_count, time_count
  )

def add_channel_labels(image, protein_name):
  """
  Add the name for each channel (e.g. DNA, Cell membrane, ...)
  """
  for (i, channel) in enumerate(image.getChannels()):
    channel_name = CHANNELS[i]
    if channel_name == 'protein':
      channel_name = protein_name
    logical_channel = channel.getLogicalChannel()
    logical_channel.setName(channel_name)
    logical_channel.save()

def add_key_value_pairs(conn, image):
  """
  Add the key value pairs to the image. This consists of the models used in
  generating the differnet channels, as well as Murphy Lab as the owner, and
  the data url.
  """
  map_annotation = omero.gateway.MapAnnotationWrapper(conn)
  map_annotation.setNs(omero.constants.metadata.NSCLIENTMAPANNOTATION)
  key_value_pairs = [
      ['owner', 'Murphy Lab'],
      ['data-url', 'http://murphylab.web.cmu.edu/data/'],
      ['channel0_class', 'framework'],
      ['channel0_type', 'cylindrical'],
      ['channel1_class', 'framework'],
      ['channel1_type', 'ratio'],
      ['channel2_class', 'vesicles'],
      ['channel2_type', 'GMM']
  ]
  map_annotation.setValue(key_value_pairs)
  map_annotation.save()
  image.linkAnnotation(map_annotation)

def add_pixel_size(image, conn):
  """
  Add the physical pixel size to the image metadata (0.49 x 0.49).
  """
  image = conn.getObject("Image", image.getId())  # reloads the pixels
  pixels = image.getPrimaryPixels()._obj
  size = omero.model.LengthI(0.49, omero.model.enums.UnitsLength.MICROMETER)
  pixels.setPhysicalSizeX(size)
  pixels.setPhysicalSizeY(size)
  conn.getUpdateService().saveObject(pixels)

def get_tag_annotation(name, conn):
  """
  Look for the tag with the given name. If multiple exist, ask the user which
  one to choose and remember it for later.
  """
  if name in TAG_ANNOTATIONS:
    return TAG_ANNOTATIONS[name]
  else:
    attributes = {"textValue": name}
    matches = list(conn.getObjects("TagAnnotation", attributes=attributes))
    if len(matches) == 0:
      tag_annotation = omero.gateway.TagAnnotationWrapper(conn)
      tag_annotation.setValue(name)
      tag_annotation.save()
    elif len(matches) == 1:
      tag_annotation = matches[0]
    else:
      print()
      print("  Found multiple tags with name '{}':".format(name))
      tag_ids = map(lambda t: int(t.getId()), matches)
      print("    {}".format(tag_ids))
      prompt = "  Enter preferred tag ID:"
      tag_annotation = None
      while tag_annotation is None:
        tag_id_input = prompt_for(prompt, str(tag_ids[0]))
        if tag_id_input.isdigit() and int(tag_id_input) in tag_ids:
          tag_annotation = matches[tag_ids.index(int(tag_id_input))]
        else:
          print("  IDs: {}".format(tag_ids))
    TAG_ANNOTATIONS[name] = tag_annotation
    return tag_annotation

def add_tags(image, conn):
  """
  Add the 2D, HeLa, and synthetic tags to the image.
  """
  for tag_name in ['2D', 'HeLa', 'synthetic']:
    tag_annotation = get_tag_annotation(tag_name, conn)
    image.linkAnnotation(tag_annotation)

def add_metadata(image, protein_name, conn):
  """
  Add channel labels and the marker used for the protein.
  """
  add_channel_labels(image, protein_name)
  add_key_value_pairs(conn, image)
  add_pixel_size(image, conn)
  add_tags(image, conn)

def upload_dataset(conn, name, infix, data_path, is_test=False):
  """
  Go through the channels for each image in this dataset and merge them into a
  plane generator before uploading and adding proper metadata. Return the list
  of image ids uploaded so that they can be linked to the dataset later on.

  If is_test is true, only upload one image.
  """
  print('Uploading {} dataset...'.format(name))
  all_tifs = glob.glob(os.path.join(data_path, "2Dhela*.tif"))
  dataset_tifs = filter(lambda t: infix in os.path.basename(t), all_tifs)
  if is_test:
    dataset_tifs = dataset_tifs[0:1]
  count = len(dataset_tifs)
  i = 0
  cell_i_to_tifs = organize_by_cell_i(dataset_tifs)
  im_ids = []
  for cell_i in sorted(cell_i_to_tifs.keys()):
    tifs = sorted(cell_i_to_tifs[cell_i], key=get_i_from_other_tif)
    for tif in tifs:
      i += 1
      plane_generator = merge_image_channel_files(tif, infix)
      omero_name = get_omero_name_from_tif_name(tif)
      status = "[{:4d}/{:d}] {:12.12} |".format(i, count, omero_name)
      print_inline("  {} Uploading image...     ".format(status))
      im = upload_image(plane_generator, omero_name, conn, len(CHANNELS))
      im_ids.append(im.getId())
      print_inline("  {} Adding metadata...     ".format(status))
      add_metadata(im, name, conn)
  pad = ' ' * (len(status) + 8)  # cover up last in-line printing
  print("\rUploaded {:d} images.{}".format(count, pad))
  return im_ids

def get_hela_project(conn):
  """
  Look for a project titled "HeLa" and return it if it exists. Otherwise, make a
  new project called "HeLa" and return that.
  """
  hela_project = None
  user_id = conn.getUser().getId()
  for project in conn.listProjects(user_id):
    if project.getName() == "HeLa":
      hela_project = project
  if hela_project is None:
    hela_project = omero.model.ProjectI()
    hela_project.setName(omero.rtypes.rstring("HeLa"))
    hela_project = conn.getUpdateService().saveAndReturnObject(hela_project)
  return hela_project

def organize_uploaded_images(conn, dataset_to_ids):
  """
  Given a dictionary mapping dataset name to a list of ids belonging to that
  dataset, create a dataset for each of these and link its images to it, looking
  for a HeLA project and creating one if it doesn't exist.
  """
  print_inline('Organizing images...')
  update_service = conn.getUpdateService()
  project = get_hela_project(conn)
  for (dataset_name, im_ids) in dataset_to_ids.items():
    dataset = omero.model.DatasetI()
    omero_dataset_name = "2D Synthetic HeLa - " + dataset_name
    dataset.setName(omero.rtypes.rstring(omero_dataset_name))
    dataset = update_service.saveAndReturnObject(dataset)
    link = omero.model.ProjectDatasetLinkI()
    link.setParent(omero.model.ProjectI(project.getId(), False))
    link.setChild(omero.model.DatasetI(dataset.getId(), False))
    update_service.saveObject(link)
    for im_id in im_ids:
      link = omero.model.DatasetImageLinkI()
      link.setParent(omero.model.DatasetI(dataset.getId(), False))
      link.setChild(omero.model.ImageI(im_id, False))
      update_service.saveObject(link)
  print('\rOrganized images.   ')

def upload_2d_synthetic_hela_collection(data_path, is_test=False):
  """
  Given the path to the 2d synthetic hela images, they are in a
  one-image-per-channel format. For each dataset, merge corresponding channels
  of the same image into one and upload them to an OMERO instance. Look for a
  HeLa project and create it if it doesn't exist. Otherwise, upload the datasets
  to the existing HeLa project.

  If is_test is true, only upload one image.
  """
  check_structure(data_path)
  conn = get_omero_connection()
  dataset_to_ids = dict()
  for dataset in DATASETS:
    name, infix = dataset['name'], dataset['infix']
    im_ids = upload_dataset(conn, name, infix, data_path, is_test)
    dataset_to_ids[name] = im_ids
    if is_test:
      break
  organize_uploaded_images(conn, dataset_to_ids)
  conn._closeSession()

def main():
  """
  Get the argument specifying the directory and run the script to upload images
  from that directory.
  """
  args = get_args()
  upload_2d_synthetic_hela_collection(args.path, args.test)

if __name__ == '__main__':
  main()
