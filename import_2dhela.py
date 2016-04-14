#!/usr/bin/env python2
"""
This script merges the images for each channel of the 2D HeLa dataset into a
single OME tiff which gets uploaded to a specified OMERO database. It checks for
a directory structure as follows relative to the provided path:

  LAM/crop/cell1.tif
           cell2.tif
           ...
           cellx.tif
     /orgcell/""
     /orgdna/""
     /orgprot/""
  Mit/crop/cell1.tif
           cell2.tif
           ...
           celly.tif
     /orgcell/""
     /orgdna/""
     /orgprot/""
  Nuc/crop/cell1.tif
           cell2.tif
           ...
           cellz.tif
     /orgcell/""
     /orgdna/""
     /orgprot/""
  TfR/crop/cell1.tif
           cell2.tif
           ...
           cellw.tif
     /orgcell/""
     /orgdna/""
     /orgprot/""

https://ome.irp.nia.nih.gov/iicbu2008/hela/index.html
"""


from __future__ import print_function

import argparse
import getpass
import os
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

CROP_DIR = 'crop'

DATASETS = [
  {
    'name' : 'Lysosomes',
    'dir' : 'LAM',
    'marker' : 'LAMP2'
  }, {
    'name': 'Mitochondria',
    'dir' : 'Mit',
    'marker' : 'Mitochondrial outer membrane protein - clone H6/C12'
  }, {
    'name' : 'Nucleolus',
    'dir': 'Nuc',
    'marker' : 'Nucleolin'
  } , {
    'name' : 'Endosomes',
    'dir' : 'TfR',
    'marker' : 'Transferrin receptor - clone 236-15 375'
  }
]

CHANNELS = {
  0 : {
    'dir': 'orgdna',
    'name': 'DNA',
  },
  1 : {
    'dir': 'orgcell',
    'name': 'Cell membrane'
  },
  2 : {
    'dir': 'orgprot',
    'name': 'protein'
  }
}

DIR4_DELTAS = [(0,-1),(-1,0),(0,1),(1,0)]
DIR8_DELTAS = [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]
ADD_DELTA = lambda (a,b): lambda (c,d): (a+c,b+d)

def get_args():
  """
  Parse the command line args using the argparse library.
  """
  parser = argparse.ArgumentParser(description="2D HeLa collection uploader")
  parser.add_argument("-t", "--test", action="store_true",
    help="upload a single image as a test")
  parser.add_argument("path", help="path to the 2D HeLa collection")
  return parser.parse_args()  # exits on malformed arguments

def error_exit(msg):
  """
  Print the specified message and exit with non-zero status.
  """
  print(msg)
  sys.exit(1)

def check_structure(data_path):
  """
  Make sure directories exist for each dataset and images exist for all
  channels.
  """
  if not os.path.isdir(data_path):
    error_exit("'%s' is not a valid directory.".format(data_path))
  for dataset in DATASETS:
    dataset_path = os.path.join(data_path, dataset['dir'])
    if not os.path.isdir(dataset_path):
      err_template = "Expected '{}' directory for the '{}' dataset."
      error_exit(err_template.format(dataset_path, dataset['name']))

    channels = CHANNELS.values()
    for channel in channels:
      channel_path = os.path.join(dataset_path, channel['dir'])
      if not os.path.isdir(channel_path):
        err_template = "Expected '{}' directory for the '{}' channel of '{}'."
        s = err_template.format(channel_path, channel['name'], dataset['name'])
        error_exit(s)
    crop_path = os.path.join(dataset_path, CROP_DIR)
    if not os.path.isdir(crop_path):
      err_template = "Expected '{}' crop directory for dataset '{}'."
      error_exit(err_template.format(CROP_DIR), dataset['name'])

    reference_contents = set(os.listdir(crop_path))
    for channel in channels:
      channel_path = os.path.join(dataset_path, channel['dir'])
      contents = set(os.listdir(channel_path))
      for item in reference_contents:
        if item not in contents:
          err_template = "Missing item '{}' in '{}' channel of '{}' dataset."
          s = err_template.format(item, channel['name'], dataset['name'])
          error_exit(s)
      for item in contents:
        if item not in reference_contents:
          err_template = "Unexpected item '{}' in '{}' channel of '{}' dataset."
          s = err_template.format(item, channel['name'], dataset['name'])
          error_exit(s)

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

def merge_image_channel_files(dataset_path, image_name):
  """
  Given a path containing directories for each channel, grab the channel image
  for image_name in each of the channels, and merge them into a single plane
  generator which is returned.

  The plane generator should return an x-y plane of values for each z, channel,
  and time combination (in that order). In this case, we only have one z and
  time, but multiple channels, so we return a generator that returns a plane for
  each channel in order.
  """
  def plane_generator():
    for i in xrange(len(CHANNELS)):
      channel_path = os.path.join(dataset_path, CHANNELS[i]['dir'], image_name)
      yield numpy.array(Image.open(channel_path))
  return plane_generator

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
    channel_name = CHANNELS[i]['name']
    if channel_name == 'protein':
      channel_name = protein_name
    logical_channel = channel.getLogicalChannel()
    logical_channel.setName(channel_name)
    logical_channel.save()

def add_key_value_pairs(conn, marker, image):
  """
  Add the key value pairs to the image. This consists of the marker used, as
  well as Murphy Lab as the owner, and the data url.
  """
  map_annotation = omero.gateway.MapAnnotationWrapper(conn)
  map_annotation.setNs(omero.constants.metadata.NSCLIENTMAPANNOTATION)
  key_value_pairs = [
      ['marker', marker],
      ['owner', 'Murphy Lab'],
      ['data-url', 'http://murphylab.web.cmu.edu/data/']
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

def add_tags(image, conn):
  """
  Add the 2D and HeLa tags to the image.
  """
  for tag_name in ['2D', 'HeLa']:
    tag_annotation = omero.gateway.TagAnnotationWrapper(conn)
    tag_annotation.setValue(tag_name)
    tag_annotation.save()
    image.linkAnnotation(tag_annotation)

def add_metadata(image, marker, protein_name, conn):
  """
  Add channel labels and the marker used for the protein.
  """
  add_channel_labels(image, protein_name)
  add_key_value_pairs(conn, marker, image)
  add_pixel_size(image, conn)
  add_tags(image, conn)

def get_next_p(arr, dir_i, is_in_bounds, seen, current_p, is_bridge):
  """
  Look in all 8 directions for a valid possible next choice. If it's a bridge,
  look in all 4 directions, and if that part isn't a bridge part, then look in
  all 8. Valid means that it is 1, hasn't been seen before*, and is in bounds.
  Returns None if no valid next is found.

  *note that we may end up choosing a bridge we've seen before to be the next
  point, but this is handled elsewhere in get_crop_outline_points
  """
  deltas = DIR8_DELTAS[dir_i:] + DIR8_DELTAS[:dir_i]  # start search from dir_i
  if is_bridge:
    deltas = filter(lambda d: d in DIR4_DELTAS, deltas)
  neighbors = map(ADD_DELTA(current_p), deltas)
  for (i,j) in neighbors:
    if is_in_bounds((i,j)) and ((i,j) not in seen) and arr[i][j]:
      for (k,l) in map(ADD_DELTA((i,j)), DIR4_DELTAS):
        if (not is_in_bounds((k,l))) or (not arr[k][l]):
          if is_bridge and not is_bridge_part(arr, is_in_bounds, (i,j)):
            # In this case, current_p is the last component of the bridge, so it
            # doesn't have to connect to the next piece only in 4 directions.
            return get_next_p(arr, dir_i, is_in_bounds, seen, current_p, False)
          else:
            return (i,j)
  return None

def is_bridge_part(arr, is_in_bounds, p):
  """
  Check if the given point is part of a bridge. This is defined as having only
  two connections to another 1-valued cell in 4 directions, and if these two
  connections are not opposites, not having a 1-valued cell in the corner
  between them.
  """
  # Check for 2 4-dir 1-value neighbors
  count = 0
  for (i,j) in map(ADD_DELTA(p), DIR4_DELTAS):
    if is_in_bounds((i,j)) and arr[i][j]:
      count += 1
  if count != 2:
    return False
  # Check for no 1-value corner
  wrapped_deltas = DIR8_DELTAS + [DIR8_DELTAS[0]]
  for corner_i in xrange(4):
    start_i = corner_i * 2
    to_check = map(ADD_DELTA(p), wrapped_deltas[start_i:start_i+3])
    all_1 = True
    for (i,j) in to_check:
      if not is_in_bounds((i,j)) or not arr[i][j]:
        all_1 = False
    if all_1:
      return False
  return True

def is_bridge_start(arr, is_in_bounds, p):
  """
  To be the start of a bridge, the point must be connected to only two other
  1-valued cells in 4 directions, and they must be opposite one another.
  """
  neighbor_1_is = []
  for d_i in xrange(len(DIR4_DELTAS)):
    (i,j) = ADD_DELTA(p)(DIR4_DELTAS[d_i])
    if is_in_bounds((i,j)) and arr[i][j]:
      neighbor_1_is.append(d_i)
  return neighbor_1_is in [[0,2], [1,3]]

def get_crop_outline_points(arr):
  """
  Given a 2D boolean array representing the crop image, return the points that
  make up the outline of the shape represented by the 1's. Note that there
  should only be one such shape in the image and it shouldn't have any holes.
  """
  h, w = len(arr), len(arr[0])
  points = []
  # Get the left-most of the top-most pixels
  first = None
  for i in xrange(h):
    if first is None:
      for j in xrange(w):
        if arr[i][j]:
          first = (i,j)
          break
  points.append(first)
  # Walk along the edge of the image, clockwise, adding in points. Points are
  # added if they have at least one side either on the edge of the image or next
  # to a pixel not included in the image. This is because we assume the boolean
  # values form a single continuous shape with no holes (i.e. no 0-value pixel
  # has 1's on all 4 sides of it).
  #
  # A special edge case of note is a 'bridge' where two portions of the boolean
  # shape are connected by one or more connected pixels, where removing any
  # pixel from this group would disconnect the two regions. When a bridge is
  # detected, we follow it until it finishes, get the pixel boundary for the
  # connected region, and then return along the bridge the reverse order we
  # originally traversed it. We maintain a bridge stack for when multiple
  # bridges are crossed before returning to the original region.
  is_in_bounds = lambda (i,j): ((i > 0) and (j > 0)) and ((i < h) and (j <= w))
  seen = set(points)
  dir_i = 0   # start search direction index
  bridge_stack = []
  current_old_bridge = None
  current_new_bridge = None
  current_p = first
  while True:
    if current_new_bridge:
      # We are traversing a new bridge we haven't seen before
      next_p = get_next_p(arr, dir_i, is_in_bounds, seen, current_p, True)
      if is_bridge_part(arr, is_in_bounds, next_p):
        current_new_bridge.append(next_p)
      else:
        bridge_stack.append(current_new_bridge)
        current_new_bridge = None
    elif current_old_bridge:
      # We are going back across a bridge we already crossed
      next_p = current_old_bridge.pop()
      if current_old_bridge == []:
        current_old_bridge = None
    else:
      # Regular clockwise traversal
      next_p = get_next_p(arr, dir_i, is_in_bounds, seen, current_p, False)
      if next_p is None:
        neighbors = map(ADD_DELTA(current_p), DIR8_DELTAS)
        if bridge_stack and bridge_stack[-1][-1] in neighbors:
          current_old_bridge = bridge_stack.pop()
          continue
        elif first in neighbors:
          return points
        else:
          err_msg_template = "Couldn't find next pixel from {}: {}"
          error_exit("\n" + err_msg_template.format(current_p, points))
      elif is_bridge_start(arr, is_in_bounds, next_p):
        current_new_bridge = [next_p]
    reverse_delta = (current_p[0] - next_p[0], current_p[1] - next_p[1])
    dir_i = DIR8_DELTAS.index(reverse_delta)   # start search from opposite dir
    current_p = next_p
    points.append(current_p)
    seen.add(current_p)

def create_polygon_from_points(raw_points):
  """
  Given an array of points corresponding to the outline of a polygon, return an
  OMERO polygon representing said polygon.
  """
  # Need to flip across y=x line
  raw_rows, raw_cols = len(raw_points), len(raw_points[0])
  points = []
  for (x,y) in raw_points:
    points.append((y,x))
  # OMERO polygons expect a string formatted like:
  #   "points[x1,y1 x2,y2] points1[x1,y1 x2,y2] points2[x1,y1 x2,y2] mask[x1,y1 x2,y2]"
  coords_s = " ".join(map(lambda p: "{},{}".format(*p), points))
  mask_s = ",".join(["0" for p in points])
  points_s_template = "points[{0}] points1[{0}] points2[{0}] mask[{1}]"
  points_s = points_s_template.format(coords_s, mask_s)
  polygon = omero.model.PolygonI()
  polygon.theZ = omero.rtypes.rint(0)
  polygon.theT = omero.rtypes.rint(0)
  polygon.points = omero.rtypes.rstring(points_s)
  return polygon

def add_crop_regions(conn, image, dataset_path, image_name):
  """
  Given the path to the dataset and the image name, parse the crop file for the
  image and create a region of interest polygon for the crop. There should be
  one contiguous shape of 1's marking the region.
  """
  # Turn crop image into OMERO polygon
  crop_filename = os.path.join(dataset_path, CROP_DIR, image_name)
  crop_points = get_crop_outline_points(numpy.array(Image.open(crop_filename)))
  crop_polygon = create_polygon_from_points(crop_points)
  # Save polygon in an ROI for the image
  roi = omero.model.RoiI()
  roi.setImage(omero.model.ImageI(image.getId(), False))
  roi.addShape(crop_polygon)
  conn.getUpdateService().saveObject(roi)

def upload_dataset(conn, name, path, marker, is_test=False):
  """
  Go through the channels for each image in this dataset and merge them into a
  plane generator before uploading and adding proper metadata. Return the list
  of image ids uploaded so that they can be linked to the dataset later on.

  If is_test is true, only upload one image.
  """
  print('Uploading {} dataset...'.format(name))
  unsorted_image_names = os.listdir(os.path.join(path, CHANNELS[0]['dir']))
  get_number = lambda s: int(s.split('.')[0][4:])  # cell23.tif -> 23
  image_names = list(sorted(unsorted_image_names, key=get_number))
  if is_test:
    image_names = image_names[0:1]
  count = len(image_names)
  im_ids = []
  for (i, image_name) in enumerate(image_names):
    plane_generator = merge_image_channel_files(path, image_name)
    status = "[{:4d}/{:d}] {:12.12} |".format(i + 1, count, image_name)
    print_inline("  {} Uploading image...     ".format(status))
    omero_image_name = "cell{:02d}".format(get_number(image_name))
    im = upload_image(plane_generator, omero_image_name, conn, len(CHANNELS))
    im_ids.append(im.getId())
    print_inline("  {} Adding metadata...     ".format(status))
    add_metadata(im, marker, name, conn)
    print_inline("  {} Creating crop region...".format(status))
    add_crop_regions(conn, im, path, image_name)
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
    omero_dataset_name = "2D HeLa - " + dataset_name
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

def upload_2d_hela_collection(data_path, is_test=False):
  """
  Given the path to the 2d hela images, they are in a one-image-per-channel
  format. For each dataset, merge corresponding channels of the same image
  into one and upload them to an OMERO instance. Look for a HeLa project and
  create it if it doesn't exist. Otherwise, upload the datasets to the existing
  HeLa project.

  If is_test is true, only upload one image.
  """
  check_structure(data_path)
  conn = get_omero_connection()
  dataset_to_ids = dict()
  for dataset in DATASETS:
    dataset_path = os.path.join(data_path, dataset['dir'])
    name, marker = dataset['name'], dataset['marker']
    im_ids = upload_dataset(conn, name, dataset_path, marker, is_test)
    dataset_to_ids[name] = im_ids
    if is_test:
      break
  organize_uploaded_images(conn, dataset_to_ids)
  conn._closeSession()

def main():
  """
  Get the argument specifying the directory and run the script from there.
  """
  args = get_args()
  upload_2d_hela_collection(args.path, args.test)

if __name__ == '__main__':
  main()
