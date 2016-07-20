import fractions
import math
import os
import re
import shutil
import StringIO
import tempfile

import django.http
import lxml.etree
import numpy
import omeroweb.webclient.decorators
import omero.constants
import PIL
import tifffile


class TiffOperationError(RuntimeError): pass
"""
This error is raised when something goes wrong during the tiff downsampling
process and an innacurate file is created.
"""


SYNTHETIC_DATASETS = dict()
"""
These are the IDs of datasets we will consider looking through for images,
mapped to the properties that get queried against (dimensionality and
cell_line). For example, with 2D Synthetic HeLa datasets with ids 402 and 403,
and a 3D Synthetic HeLa dataset with id 412, it should look like:

  SYNTHETIC_DATASETS = {
    402: {
      "dimensionality": "2D",
      "cell_line": "HeLa"
    }, 403: {
      "dimensionality": "2D",
      "cell_line": "HeLa"
    }, 412: {
      "dimensionality": "3D",
      "cell_line": "HeLa"
    }
  }
"""


def get_matching_datasets(conn, dimensionality, cell_line):
  """
  Return the loaded datasets that have the queried dimensionality and cell_line.
  If any of these are none, don't check that property.
  """
  matching_datasets = []
  for (dataset_id, attributes) in SYNTHETIC_DATASETS.items():
    if ((dimensionality is None) or \
        (attributes['dimensionality'].lower() == dimensionality.lower())) and\
        ((cell_line is None) or \
        (attributes['cell_line'].lower() == cell_line.lower())):
      dataset = conn.getObject("Dataset", dataset_id)
      matching_datasets.append(dataset)
  return matching_datasets

def is_matching_image(image, channel0_model_class, channel0_model_type,
    channel1_model_class, channel1_model_type, channel2_model_class,
    channel2_model_type):
  """
  Checks if the image's properties match those queried for. If any of the
  queried properties are None, don't check for them.
  """
  key_value_ns = omero.constants.metadata.NSCLIENTMAPANNOTATION
  key_values_annotation = image.getAnnotation(key_value_ns)
  key_value_tuples = key_values_annotation.getValue()
  key_value_dict = dict(key_value_tuples)
  if (channel0_model_class is not None):
    image_channel0_class = key_value_dict.get('channel0_class', '')
    if (image_channel0_class.lower() != channel0_model_class.lower()):
      return False
  if (channel0_model_type is not None):
    image_channel0_type = key_value_dict.get('channel0_type', '')
    if (image_channel0_type.lower() != channel0_model_type.lower()):
      return False
  if (channel1_model_class is not None):
    image_channel1_class = key_value_dict.get('channel1_class', '')
    if (image_channel1_class.lower() != channel1_model_class.lower()):
      return False
  if (channel1_model_type is not None):
    image_channel1_type = key_value_dict.get('channel1_type', '')
    if (image_channel1_type.lower() != channel1_model_type.lower()):
      return False
  if (channel2_model_class is not None):
    image_channel2_class = key_value_dict.get('channel2_class', '')
    if (image_channel2_class.lower() != channel2_model_class.lower()):
      return False
  if (channel2_model_type is not None):
    image_channel2_type = key_value_dict.get('channel2_type', '')
    if (image_channel2_type.lower() != channel2_model_type.lower()):
      return False
  return True

def get_matching_image_ids(
    conn,
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
  Look for existing synthetic images that match the given criteria and return
  their ids in a list.
  """
  datasets = get_matching_datasets(conn, dimensionality, cell_line)
  matching_image_ids = []
  for dataset in datasets:
    for image in dataset.listChildren():
      should_keep = is_matching_image(
        image, channel0_model_class, channel0_model_type, channel1_model_class,
        channel1_model_type, channel2_model_class, channel2_model_type
      )
      if should_keep:
        matching_image_ids.append(image.getId())
  return matching_image_ids

@omeroweb.webclient.decorators.login_required()
def image_ids(request, conn=None, **kwargs):
  """
  Endpoint for getting image ids that match a criteria that is specified via
  query parameters.
  Returns a json object with the ids as a list in a 'data' field.
  """
  dimensionality = request.GET.get('dimensionality')
  cell_line = request.GET.get('cell_line')
  channel0_model_class = request.GET.get('channel0_model_class')
  channel0_model_type = request.GET.get('channel0_model_type')
  channel1_model_class = request.GET.get('channel1_model_class')
  channel1_model_type = request.GET.get('channel1_model_type')
  channel2_model_class = request.GET.get('channel2_model_class')
  channel2_model_type = request.GET.get('channel2_model_type')
  image_ids = get_matching_image_ids(
    conn,
    dimensionality=dimensionality,
    cell_line=cell_line,
    channel0_model_class=channel0_model_class,
    channel0_model_type=channel0_model_type,
    channel1_model_class=channel1_model_class,
    channel1_model_type=channel1_model_type,
    channel2_model_class=channel2_model_class,
    channel2_model_type=channel2_model_type
  )
  return django.http.JsonResponse({'data': image_ids})

def downsample_scale(dimension, downsampling_factor):
  """
  Given a desired downsampling factor and the original dimension, return the
  nearest integer dimension (rounding up) that the downsampling_factor would
  result in.
  """
  return int(math.ceil(float(dimension) / downsampling_factor))

def save_downsampled_pages_to_dir(tiff_path, save_dir, factor):
  """
  Given the path to a multipage tiff, downsample each page by the provided
  factor, saving each page to a separate file in the save directory. Return a
  list of the saved filenames in page order.
  """
  method = PIL.Image.BICUBIC
  array_tiff = tifffile.imread(tiff_path)
  if (len(array_tiff.shape) > 3):  # 3D; flatten to be array of pages
    shape = array_tiff.shape
    tiff_pages = numpy.reshape(array_tiff, (-1, shape[-2], shape[-1]))
  else:
    tiff_pages = array_tiff
  original_height, original_width = tiff_pages[0].shape
  new_height = downsample_scale(original_height, factor)
  new_width = downsample_scale(original_width, factor)
  paths = []
  for i in range(len(tiff_pages)):
    pil_page = PIL.Image.fromarray(tiff_pages[i])
    downsampled_page = pil_page.resize((new_width, new_height), method)
    path = os.path.join(save_dir, "{}.tif".format(i))
    downsampled_page.save(path, format="tiff")
    paths.append(path)
  return paths

def limit_fraction(raw_fraction):
  """
  The struct field saved in tiffs can only support numbers up to 4294967295, and
  the fractions.Fraction class has a method to keep the denominator below a
  threshold, so see what the highest denominator would be to keep the numerator
  within this threshold and limit the denominator to that size.
  """
  max_num = 4294967295
  max_num_denom =\
    (raw_fraction.denominator * float(max_num)) / raw_fraction.numerator
  max_denom = int(math.floor(min(max_num, max_num_denom)))
  return raw_fraction.limit_denominator(50)

def get_resolution_tag_values(original_tiff_path, downsampling_factor):
  """
  The resolution tags for a tiff expect a tuple fraction. We get the fraction
  used in the original tiff, then the fraction corresponding to the downsampling
  of each dimension, then multiply the original fraction by this one.
  """
  with tifffile.TiffFile(original_tiff_path) as original_tiff:
    p0 = original_tiff.pages[0]
    original_x_res = p0.tags['x_resolution'].value
    original_y_res = p0.tags['y_resolution'].value
    original_h, original_w = original_tiff.asarray().shape[-2:]
  new_w = downsample_scale(original_w, downsampling_factor)
  new_h = downsample_scale(original_h, downsampling_factor)
  w_fraction = fractions.Fraction(new_w, original_w)
  h_fraction = fractions.Fraction(new_h, original_h)
  raw_new_x_res_fraction = fractions.Fraction(*original_x_res) * w_fraction
  raw_new_y_res_fraction = fractions.Fraction(*original_y_res) * h_fraction
  bounded_x_frac = limit_fraction(raw_new_x_res_fraction)
  bounded_y_frac = limit_fraction(raw_new_y_res_fraction)
  new_x_res = (bounded_x_frac.numerator, bounded_x_frac.denominator)
  new_y_res = (bounded_y_frac.numerator, bounded_y_frac.denominator)
  return new_x_res, new_y_res

def make_ome_double_annotation(name, description, value):
  """
  Given the name to be put in the ID field, a description of what the annotation
  is for, and the value for the tag, return the corresponding XML element. This
  conforms to the OME XML specification when added under the
  StructuredAnnotations element.

  An example is:
  <DoubleAnnotation ID="Annotation:RequestedDownsamplingFactor">
    <Description>
      This element contains the requested downsampling factor the client passed
      to the OMERO server.
    </Description>
    <Value>
      1.1
    </Value>
  </DoubleAnnotation>
  """
  annotation = lxml.etree.Element("DoubleAnnotation")
  annotation.set("ID", "Annotation:" + name)
  description_elem = lxml.etree.Element("Description")
  description_elem.text = description
  annotation.append(description_elem)
  value_elem = lxml.etree.Element("Value")
  value_elem.text = str(value)
  annotation.append(value_elem)
  return annotation

def add_annotation_refs(image_elem, names):
  """
  Given the image xml element and a list of the names used in ID's of the
  various added annotations, create an annotation reference to each of these
  added annotations and append it to the pixels element.
  """
  annotation_ref = image_elem.find("{*}AnnotationRef")
  annotation_ns = re.match("\{.*\}", annotation_ref.tag).group(0)
  for name in names:
    new_ref = lxml.etree.Element(annotation_ns + "AnnotationRef")
    new_ref.set("ID", "Annotation:" + name)
    image_elem.append(new_ref)

def update_xml_metadata(ome_xml, physical_size_x, physical_size_y, size_x,
    size_y, requested_factor, x_factor, y_factor):
  """
  Update each of the given variables in their respective positions in the
  OME-TIFF metadata XML object. Then, add annotations containing information
  about the downsampling performed on the image.
  """
  pixels_elem = ome_xml.find(".//{*}Pixels")
  original_phys_x = pixels_elem.get("PhysicalSizeX")
  original_phys_y = pixels_elem.get("PhysicalSizeY")
  pixels_elem.set("PhysicalSizeX", str(physical_size_x))
  pixels_elem.set("PhysicalSizeY", str(physical_size_y))
  pixels_elem.set("SizeX", str(size_x))
  pixels_elem.set("SizeY", str(size_y))
  structured_annotations = ome_xml.find("{*}StructuredAnnotations")
  structured_annotations.append(make_ome_double_annotation(
    "RequestedDownsamplingFactor",
    "This element contains the requested downsampling factor the client " +\
      "passed to the OMERO server.",
    requested_factor
  ))
  structured_annotations.append(make_ome_double_annotation(
    "XDownsamplingFactor",
    "This is the amount by which the x dimension of the image was " +\
      "downsampled by the OMERO server.",
    x_factor
  ))
  structured_annotations.append(make_ome_double_annotation(
    "YDownsamplingFactor",
    "This is the amount by which the y dimension of the image was " +\
      "downsampled by the OMERO server.",
    y_factor
  ))
  structured_annotations.append(make_ome_double_annotation(
    "OriginalPhysicalSizeX",
    "This is the PhysicalSizeX attribute of the Pixels object before the " +\
      "downsampling was performed on the image.",
    original_phys_x
  ))
  structured_annotations.append(make_ome_double_annotation(
    "OriginalPhysicalSizeY",
    "This is the PhysicalSizeY attribute of the Pixels object before the " +\
      "downsampling was performed on the image.",
    original_phys_y
  ))
  image_elem = ome_xml.find("{*}Image")
  add_annotation_refs(image_elem, ["RequestedDownsamplingFactor",
    "XDownsamplingFactor", "YDownsamplingFactor", "OriginalPhysicalSizeX",
    "OriginalPhysicalSizeY"])

def get_new_xml_string(original_tiff_path, x_res_fraction, y_res_fraction,
    downsampling_factor):
  """
  Grab the original OME XML info stored in the original tiff, then return an
  updated OME XML structure as a string.
  """
  with tifffile.TiffFile(original_tiff_path) as original_tiff:
    ome_xml_string = original_tiff.pages[0].tags['image_description'].value
    original_y, original_x = original_tiff.asarray().shape[-2:]
  ome_xml = lxml.etree.fromstring(ome_xml_string)
  cm_to_um = 10000  # units are cm, want to get in um (micro-meters)
  physical_size_x = (x_res_fraction[1] * float(cm_to_um)) / x_res_fraction[0]
  physical_size_y = (y_res_fraction[1] * float(cm_to_um)) / y_res_fraction[0]
  size_x = downsample_scale(original_x, downsampling_factor)
  size_y = downsample_scale(original_y, downsampling_factor)
  x_downsampling_factor = float(original_x) / size_x
  y_downsampling_factor = float(original_y) / size_y
  update_xml_metadata(ome_xml, physical_size_x, physical_size_y, size_x, size_y,
    downsampling_factor, x_downsampling_factor, y_downsampling_factor)
  return lxml.etree.tostring(ome_xml, encoding="UTF-8", xml_declaration=True)

def get_downsampled_tiff(image, factor):
  """
  Given an OMERO image object, export it to a tiff, then downsample that tiff,
  update its metadata, and return it.

  Note that the code ends up being a little tedious because we have to alternate
  a few modules: PIL can resize a single tif page, but not save a multipage tif
  or reliably read one in; tifffile can save and read a multipage tif, but not
  resize it; StringIO can't do either, but can be returned in an HTTP request.
  
  We first save the original exported tiff from OMERO to a file, then read that
  file into an array using tifffile, turn each page of that array into a PIL
  image, downsample and save each page into a temp directory, read back those
  pages in tifffile, save that to a file, read that file into a StringIO object,
  and finally return that StringIO object.
  """
  original_tiff = image.exportOmeTiff()
  original_tiff_path = None
  downsampled_dir = None
  downsampled_tiff_path = None
  retry = False
  try:
    raw_original_tiff_file_handler, original_tiff_path =\
      tempfile.mkstemp(suffix='.tif')
    with os.fdopen(raw_original_tiff_file_handler, 'w') as original_tiff_file:
      original_tiff_file.write(original_tiff)
    downsampled_dir = tempfile.mkdtemp()
    downsampled_page_files = save_downsampled_pages_to_dir(
      original_tiff_path, downsampled_dir, factor
    )
    downsampled_tiff = tifffile.imread(downsampled_page_files)
    if downsampled_tiff.shape[0] == 1:
      # For some reason, potentially due to either tifffile's lazy handling of
      # files or PIL's not-so-great tiff reliability, there is a ~10% chance
      # this process will create a tiff file with a single page only. If we
      # ended up in this undesirable case, we humbly raise this exception which
      # will take us straight to cleaning up the files we made and starting
      # over.
      raise TiffOperationError
    x_resolution_tag_value, y_resolution_tag_value =\
      get_resolution_tag_values(original_tiff_path, factor)
    new_xml_string = get_new_xml_string(
      original_tiff_path, x_resolution_tag_value, y_resolution_tag_value, factor
    )
    raw_downsampled_tiff_file_handler, downsampled_tiff_path =\
      tempfile.mkstemp(suffix='.tif')
    with os.fdopen(raw_downsampled_tiff_file_handler, 'w') as downsampled_file:
      tifffile.imsave(downsampled_file, downsampled_tiff,
        description=new_xml_string,
        extratags=(
          # These tags were determined by seeing what OME exported originally
          # and comparing against what was missing from the downsampled version
          ("x_resolution", "2I", 1, x_resolution_tag_value),
          ("y_resolution", "2I", 1, y_resolution_tag_value),
          ("resolution_unit", "H", 1, 3),
        )
      )
    with open(downsampled_tiff_path, 'r') as downsampled_file:
      downsampled_tiff_io = StringIO.StringIO(downsampled_file.read())
  except TiffOperationError:
    retry = True
  finally:
    if original_tiff_path:
      os.unlink(original_tiff_path)
    if downsampled_dir:
      shutil.rmtree(downsampled_dir)
    if downsampled_tiff_path:
      os.unlink(downsampled_tiff_path)
  if retry:
    return get_downsampled_tiff(image, factor)
  return downsampled_tiff_io

@omeroweb.webclient.decorators.login_required()
def images(request, raw_image_id, conn=None, **kwargs):
  """
  Endpoint for downloading an image by id, and optionally downsampling it
  beforehand via the "downsampling_factor" query parameter.
  Returns an OME-TIFF.
  """
  image_id = int(raw_image_id)
  image = conn.getObject("Image", image_id)
  if image is None:
    return django.http.HttpResponse("Image not found", status=404)
  downsampling_factor_arg = request.GET.get('downsampling_factor')
  if downsampling_factor_arg is None:
    tiff_data = image.exportOmeTiff()
    tiff_len = len(tiff_data)
  else:
    try:
      downsampling_factor = float(downsampling_factor_arg)
    except:
      msg = "Downsampling factor must be a floating point number."
      return django.http.HttpResponse(msg, status=400)
    if downsampling_factor <= 1:
      msg = "Downsampling factor must be greater than 1."
      return django.http.HttpResponse(msg, status=400)
    tiff_data = get_downsampled_tiff(image, downsampling_factor)
    tiff_len = tiff_data.len
  response = django.http.HttpResponse(tiff_data, content_type='image/tiff')
  response['Content-Length'] = tiff_len
  return response
