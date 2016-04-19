#!/usr/bin/env python2
"""
This script merges the images for each channel of the 3D HeLa dataset into a
single OME tiff which gets uploaded to a specified OMERO database. Channel files
are expected to be organized in a single directory like:
  LAM_cell1_ch0_t1.tif
  LAM_cell1_ch1_t1.tif
  LAM_cell1_ch2_t1.tif
  LAM_cell1_mask_t1.tif
for each image. Each channel file contains several z planes as well.
"""


from __future__ import print_function

import argparse
import getpass
import glob
import os
import sys

import bioformats
import bioformats.log4j
import javabridge
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

CROP_FILE_INFIX = 'mask'

DATASETS = [
  {
    'name' : 'Lysosomes',
    'file_prefix' : 'LAM',
    'marker' : 'LAMP2'
  }, {
    'name': 'Mitochondria',
    'file_prefix' : 'Mit',
    'marker' : 'Mitochondrial outer membrane protein - clone H6/C12'
  }, {
    'name' : 'Nucleolus',
    'file_prefix': 'Nuc',
    'marker' : 'Nucleolin'
  } , {
    'name' : 'Endosomes',
    'file_prefix' : 'TfR',
    'marker' : 'Transferrin receptor - clone 236-15 375'
  }
]

CHANNELS = {
  0 : {
    'name': 'DNA',
    'file_infix': 'ch0'
  },
  1 : {
    'name': 'Cell membrane',
    'file_infix': 'ch1'

  },
  2 : {
    'name': 'protein',
    'file_infix': 'ch2'
  }
}

DIR4_DELTAS = [(0,-1),(-1,0),(0,1),(1,0)]
DIR8_DELTAS = [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]
ADD_DELTA = lambda (a,b): lambda (c,d): (a+c,b+d)

TAG_ANNOTATIONS = dict()

def get_args():
  """
  Parse the command line args using the argparse library.
  """
  parser = argparse.ArgumentParser(description="3D HeLa collection uploader")
  parser.add_argument("-t", "--test", action="store_true",
    help="upload a single image as a test")
  parser.add_argument("path", help="path to the 3D HeLa collection")
  return parser.parse_args()  # exits on malformed arguments

def error_exit(msg):
  """
  Print the specified message and exit with non-zero status.
  """
  print(msg)
  sys.exit(1)

def get_dataset_image_count(data_path, dataset):
  """
  Given the path to the folder containing all of the data, return the count of
  cell images corresponding to the provided dataset. (Note: we assume the
  numbering doesn't skip and starts from 1.)
  """
  get_i = lambda s: int(s.split('_')[1][4:])  #..._cell21_... -> 21
  file_regex = os.path.join(data_path, dataset['file_prefix']) + '*'
  filenames = set(glob.glob(file_regex))
  return max(map(get_i, filenames))

def check_structure(data_path):
  """
  Make sure each dataset has a file for each channel of each cell.
  Files are of the format
    <dataset prefix>_cell<i>_<channel type>_t1.tif
  where <dataset prefix> is one of LAM, Mit, Nuc, TfR, and <channel type> is one
  of ch0, ch1, ch2, or mask.
  """
  if not os.path.isdir(data_path):
    error_exit("'%s' is not a valid directory.".format(data_path))
  for dataset in DATASETS:
    prefix = dataset['file_prefix']
    max_i = get_dataset_image_count(data_path, dataset)
    for i in xrange(1, max_i + 1):
      channel_infixes = map(lambda c: c['file_infix'], CHANNELS.values())
      for file_infix in channel_infixes + [CROP_FILE_INFIX]:
        cell_name = "cell{}".format(i)
        filename = "_".join([prefix, cell_name, file_infix, 't1.tif'])
        expected_path = os.path.join(data_path, filename)
        if not os.path.isfile(expected_path):
          error_exit("Expected file '{}' is missing.")

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

def merge_image_channel_files(data_path, dataset, i):
  """
  Given the path to the folder containing all of the image files, look up the
  channel files corresponding to cell i of the dataset and merge its channel
  files into a plane generator.

  The plane generator should return an x-y plane of values for each z, channel,
  and time combination (in that order). In this case, we have a variable number
  of z planes, 3 channels, and 1 time.
  """
  cell_name = 'cell{}'.format(i)
  prefix = dataset['file_prefix']
  first_channel_file = '_'.join([prefix, cell_name, 'ch0', 't1.tif'])
  first_channel_path = os.path.join(data_path, first_channel_file)
  with bioformats.ImageReader(first_channel_path) as reader:
    z_count = reader.rdr.getImageCount()
  def plane_generator():
    for z_i in xrange(z_count):
      for i in xrange(len(CHANNELS)):
        channel_infix = CHANNELS[i]['file_infix']
        channel_file = '_'.join([prefix, cell_name, channel_infix, 't1.tif'])
        channel_path = os.path.join(data_path, channel_file)
        with bioformats.ImageReader(channel_path) as reader:
          yield reader.read(index=z_i)
  return plane_generator, z_count

def upload_image(plane_generator, name, conn, c_count, z_count):
  """
  Upload the image (stored in the plane_generator as the OMERO API specifies)
  using the OMERO connection conn.
  """
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
  xy_size = omero.model.LengthI(0.049, omero.model.enums.UnitsLength.MICROMETER)
  z_size = omero.model.LengthI(0.2, omero.model.enums.UnitsLength.MICROMETER)
  pixels.setPhysicalSizeX(xy_size)
  pixels.setPhysicalSizeY(xy_size)
  pixels.setPhysicalSizeZ(z_size)
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
  Add the 3D and HeLa tags to the image.
  """
  for tag_name in ['3D', 'HeLa']:
    tag_annotation = get_tag_annotation(tag_name, conn)
    image.linkAnnotation(tag_annotation)

def add_metadata(image, marker, protein_name, conn):
  """
  Add various metadata to the image such as tags, key-value pairs, and channel
  labels.
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
  Given a 2D boolean array (255 or 0) representing the crop image, return the
  points that make up the outline of the shape represented by the 255's. Note
  that there should only be one such shape in the image and it shouldn't have
  any holes.
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

def create_polygons_from_points(raw_points, z_count):
  """
  Given an array of points corresponding to the outline of a polygon, return an
  OMERO polygon representing said polygon for each z index.
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
  polygons = []
  for z in xrange(z_count):
    polygon = omero.model.PolygonI()
    polygon.theZ = omero.rtypes.rint(z)
    polygon.theT = omero.rtypes.rint(0)
    polygon.points = omero.rtypes.rstring(points_s)
    polygons.append(polygon)
  return polygons

def add_crop_regions(conn, image, data_path, dataset, i, z_count):
  """
  Given the path to the dataset and the image name, parse the crop file for the
  image and create a region of interest polygon for the crop. There should be
  one contiguous shape of 1's marking the region.
  """
  # Turn crop image into OMERO polygon
  cell_name = 'cell{}'.format(i)
  prefix = dataset['file_prefix']
  crop_filename = '_'.join([prefix, cell_name, CROP_FILE_INFIX, 't1.tif'])
  crop_path = os.path.join(data_path, crop_filename)
  crop_points = get_crop_outline_points(numpy.array(Image.open(crop_path)))
  crop_polygons = create_polygons_from_points(crop_points, z_count)
  # Save polygon in an ROI for the image
  roi = omero.model.RoiI()
  roi.setImage(omero.model.ImageI(image.getId(), False))
  for crop_polygon in crop_polygons:
    roi.addShape(crop_polygon)
  conn.getUpdateService().saveObject(roi)

def upload_dataset(conn, data_path, dataset, is_test=False):
  """
  Go through the channels for each image in this dataset and merge them into a
  plane generator before uploading and adding proper metadata. Return the list
  of image ids uploaded so that they can be linked to the dataset later on.

  If is_test is true, only upload one image.
  """
  name, marker = dataset['name'], dataset['marker']
  print('Uploading {} dataset...'.format(name))
  max_i = get_dataset_image_count(data_path, dataset)
  if is_test:
    max_i = 1
  im_ids = []
  for i in xrange(1, max_i + 1):
    plane_generator, z_count = merge_image_channel_files(data_path, dataset, i)
    image_name = "cell{:02d}".format(i)
    status = "[{:4d}/{:d}] {:12.12} |".format(i, max_i, image_name)
    print_inline("  {} Uploading image...     ".format(status))
    im = upload_image(plane_generator, image_name, conn, len(CHANNELS), z_count)
    im_ids.append(im.getId())
    print_inline("  {} Adding metadata...     ".format(status))
    add_metadata(im, marker, name, conn)
    print_inline("  {} Creating crop region...".format(status))
    add_crop_regions(conn, im, data_path, dataset, i, z_count)
  pad = ' ' * (len(status) + 8)  # cover up last in-line printing
  print("\rUploaded {:d} images.{}".format(max_i, pad))
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
    omero_dataset_name = "3D HeLa - " + dataset_name
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

def upload_3d_hela_collection(data_path, is_test=False):
  """
  The HeLa images are all in the same directory with the dataset, channel,
  and cell indicated by the filename. We combine each cell's channel files into
  a single image and upload it to OMERO, organizing the datasets together under
  the HeLa project, which is created it if it doesn't exist.

  If is_test is true, only upload one image.
  """
  check_structure(data_path)
  conn = get_omero_connection()
  dataset_to_ids = dict()
  javabridge.start_vm(class_path=bioformats.JARS)
  bioformats.log4j.basic_config()
  for dataset in DATASETS:
    im_ids = upload_dataset(conn, data_path, dataset, is_test)
    dataset_to_ids[dataset['name']] = im_ids
    if is_test:
      break
  organize_uploaded_images(conn, dataset_to_ids)
  javabridge.kill_vm()
  conn._closeSession()

def main():
  """
  Get the argument specifying the directory and run the script from there.
  """
  args = get_args()
  upload_3d_hela_collection(args.path, args.test)

if __name__ == '__main__':
  main()
