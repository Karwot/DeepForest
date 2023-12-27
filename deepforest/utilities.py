"""Utilities model"""
import json
import os
import urllib
import warnings
import functools

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import xmltodict
import yaml
from tqdm import tqdm

from deepforest import _ROOT
import geopandas as gpd
from shapely.geometry import Point
import geopandas as gpd
from shapely.geometry import Point


def read_config(config_path):
    """Read config yaml file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))

    return config


class DownloadProgressBar(tqdm):
    """Download progress bar class."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update class attributes
        Args:
            b:
            bsize:
            tsize:

        Returns:

        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def use_bird_release(
        save_dir=os.path.join(_ROOT, "data/"), prebuilt_model="bird", check_release=True):
    """
    Check the existence of, or download the latest model release from github
    Args:
        save_dir: Directory to save filepath, default to "data" in deepforest repo
        prebuilt_model: Currently only accepts "NEON", but could be expanded to include other prebuilt models. The local model will be called prebuilt_model.h5 on disk.
        check_release (logical): whether to check github for a model recent release. In cases where you are hitting the github API rate limit, set to False and any local model will be downloaded. If no model has been downloaded an error will raise.
    Returns: release_tag, output_path (str): path to downloaded model

    """

    # Naming based on pre-built model
    output_path = os.path.join(save_dir, prebuilt_model + ".pt")

    if check_release:
        # Find latest github tag release from the DeepLidar repo
        _json = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    'https://api.github.com/repos/Weecology/BirdDetector/releases/latest',
                    headers={'Accept': 'application/vnd.github.v3+json'},
                )).read())
        asset = _json['assets'][0]
        url = asset['browser_download_url']

        # Check the release tagged locally
        try:
            release_txt = pd.read_csv(save_dir + "current_bird_release.csv")
        except BaseException:
            release_txt = pd.DataFrame({"current_bird_release": [None]})

        # Download the current release it doesn't exist
        if not release_txt.current_bird_release[0] == _json["html_url"]:

            print("Downloading model from BirdDetector release {}, see {} for details".
                  format(_json["tag_name"], _json["html_url"]))

            with DownloadProgressBar(unit='B',
                                     unit_scale=True,
                                     miniters=1,
                                     desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url,
                                           filename=output_path,
                                           reporthook=t.update_to)

            print("Model was downloaded and saved to {}".format(output_path))

            # record the release tag locally
            release_txt = pd.DataFrame({"current_bird_release": [_json["html_url"]]})
            release_txt.to_csv(save_dir + "current_bird_release.csv")
        else:
            print("Model from BirdDetector Repo release {} was already downloaded. "
                  "Loading model from file.".format(_json["html_url"]))

        return _json["html_url"], output_path
    else:
        try:
            release_txt = pd.read_csv(save_dir + "current_release.csv")
        except BaseException:
            raise ValueError("Check release argument is {}, but no release has been "
                             "previously downloaded".format(check_release))

        return release_txt.current_release[0], output_path


def use_release(
        save_dir=os.path.join(_ROOT, "data/"), prebuilt_model="NEON", check_release=True):
    """
    Check the existence of, or download the latest model release from github
    Args:
        save_dir: Directory to save filepath, default to "data" in deepforest repo
        prebuilt_model: Currently only accepts "NEON", but could be expanded to include other prebuilt models. The local model will be called prebuilt_model.h5 on disk.
        check_release (logical): whether to check github for a model recent release. In cases where you are hitting the github API rate limit, set to False and any local model will be downloaded. If no model has been downloaded an error will raise.
        
    Returns: release_tag, output_path (str): path to downloaded model

    """
    # Naming based on pre-built model
    output_path = os.path.join(save_dir, prebuilt_model + ".pt")

    if check_release:
        # Find latest github tag release from the DeepLidar repo
        _json = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    'https://api.github.com/repos/Weecology/DeepForest/releases/latest',
                    headers={'Accept': 'application/vnd.github.v3+json'},
                )).read())
        asset = _json['assets'][0]
        url = asset['browser_download_url']

        # Check the release tagged locally
        try:
            release_txt = pd.read_csv(save_dir + "current_release.csv")
        except BaseException:
            release_txt = pd.DataFrame({"current_release": [None]})

        # Download the current release it doesn't exist
        if not release_txt.current_release[0] == _json["html_url"]:

            print("Downloading model from DeepForest release {}, see {} "
                  "for details".format(_json["tag_name"], _json["html_url"]))

            with DownloadProgressBar(unit='B',
                                     unit_scale=True,
                                     miniters=1,
                                     desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url,
                                           filename=output_path,
                                           reporthook=t.update_to)

            print("Model was downloaded and saved to {}".format(output_path))

            # record the release tag locally
            release_txt = pd.DataFrame({"current_release": [_json["html_url"]]})
            release_txt.to_csv(save_dir + "current_release.csv")
        else:
            print("Model from DeepForest release {} was already downloaded. "
                  "Loading model from file.".format(_json["html_url"]))

        return _json["html_url"], output_path
    else:
        try:
            release_txt = pd.read_csv(save_dir + "current_release.csv")
        except BaseException:
            raise ValueError("Check release argument is {}, but no release "
                             "has been previously downloaded".format(check_release))

        return release_txt.current_release[0], output_path


def xml_to_annotations(xml_path):
    """
    Load annotations from xml format (e.g. RectLabel editor) and convert
    them into retinanet annotations format.
    Args:
        xml_path (str): Path to the annotations xml, formatted by RectLabel
    Returns:
        Annotations (pandas dataframe): in the
            format -> path-to-image.png,x1,y1,x2,y2,class_name
    """
    # parse
    with open(xml_path) as fd:
        doc = xmltodict.parse(fd.read())

    # grab xml objects
    try:
        tile_xml = doc["annotation"]["object"]
    except Exception as e:
        raise Exception("error {} for path {} with doc annotation{}".format(
            e, xml_path, doc["annotation"]))

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    label = []

    if isinstance(tile_xml, list):
        # Construct frame if multiple trees
        for tree in tile_xml:
            xmin.append(tree["bndbox"]["xmin"])
            xmax.append(tree["bndbox"]["xmax"])
            ymin.append(tree["bndbox"]["ymin"])
            ymax.append(tree["bndbox"]["ymax"])
            label.append(tree['name'])
    else:
        xmin.append(tile_xml["bndbox"]["xmin"])
        xmax.append(tile_xml["bndbox"]["xmax"])
        ymin.append(tile_xml["bndbox"]["ymin"])
        ymax.append(tile_xml["bndbox"]["ymax"])
        label.append(tile_xml['name'])

    rgb_name = os.path.basename(doc["annotation"]["filename"])

    # set dtypes, check for floats and round
    xmin = [round_with_floats(x) for x in xmin]
    xmax = [round_with_floats(x) for x in xmax]
    ymin = [round_with_floats(x) for x in ymin]
    ymax = [round_with_floats(x) for x in ymax]

    annotations = pd.DataFrame({
        "image_path": rgb_name,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "label": label
    })

    gdf = pandas_to_geopandas(annotations)

    return gdf

def convert_point_to_bbox(gdf, buffer_size):
    """
    Convert an input point type annotation to a bounding box by buffering the point with a fixed size.
    
    Args:
        gdf (GeoDataFrame): The input point type annotation.
        buffer_size (float): The size of the buffer to be applied to the point.
        
    Returns:
        gdf (GeoDataFrame): The output bounding box type annotation.
    """
    # define in image coordinates and buffer to create a box
    gdf["geometry"] = [
        shapely.geometry.Point(x, y)
        for x, y in zip(gdf.geometry.x.astype(float), gdf.geometry.y.astype(float))
    ]
    gdf["geometry"] = [
        shapely.geometry.box(left, bottom, right, top)
        for left, bottom, right, top in gdf.geometry.buffer(buffer_size).bounds.values
    ]

    return gdf

def shapefile_to_annotations(shapefile,
                             rgb,
                             buffer_size=0.5,
                             savedir="."):
    """
    Convert a shapefile of annotations into annotations csv file for DeepForest training and evaluation
    
    Args:
        shapefile: Path to a shapefile on disk. If a label column is present, it will be used, else all labels are assumed to be "Tree"
        rgb: Path to the RGB image on disk
        savedir: Directory to save csv files
    Returns:
        results: a pandas dataframe
    """
    # Read shapefile
    if isinstance(shapefile, str):
        gdf = gpd.read_file(shapefile)
    else:
        gdf = shapefile

    # Determine geometry type and report to user
    if gdf.geometry.type.unique().shape[0] > 1:
        raise ValueError(
            "Multiple geometry types found in shapefile. Please ensure all geometries are of the same type.")
    else:
        geometry_type = gdf.geometry.type.unique()[0]
        print("Geometry type of shapefile is {}".format(geometry_type))

    # raster bounds
    with rasterio.open(rgb) as src:
        left, bottom, right, top = src.bounds
        resolution = src.res[0]
        raster_crs = src.crs

    # Check matching the crs
    if not gdf.crs.to_string() == raster_crs.to_string():
        raise ValueError("The shapefile crs {} does not match the image crs {}".format(
            gdf.crs, src.crs))

    if src.crs is not None:
        print("CRS of shapefile is {}".format(src.crs))
        gdf = geo_to_image_coordinates(gdf,src.bounds, src.res[0])

    # check for label column
    if "label" not in gdf.columns:
        raise ValueError(
            "No label column found in shapefile. Please add a column named 'label' to your shapefile.") 
    else:
        gdf["label"] = gdf["label"]
  
    # add filename
    gdf["image_path"] = os.path.basename(rgb)

    return gdf

def determine_geometry_type(df):
    """Determine the geometry type of a geodataframe
    Args:
        df: a pandas dataframe
    Returns:
        geometry_type: a string of the geometry type
    """
     
    columns = df.columns
    if "Polygon" in columns:
        raise ValueError("Polygon column is capitalized, please change to lowercase")
    
    if "xmin" in columns and "ymin" in columns and "xmax" in columns and "ymax" in columns:
        geometry_type = "box"
    elif "polygon" in columns:
        geometry_type = "polygon"
    elif "x" in columns and "y" in columns:
        geometry_type = 'point'
    else:
        raise ValueError("Could not determine geometry type from columns {}".format(columns))

    # Report number of annotations, unique images and geometry type
    print("Found {} annotations in {} unique images with {} geometry type".format(
        df.shape[0], df.image_path.unique().shape[0], geometry_type))
    
    return geometry_type

def pandas_to_geopandas(df):
    """Parse annotations csv file and return a geodataframe
    Args:
        path: path to annotations csv file, 
    
    Geometry types:
        box: xmin, ymin, xmax, ymax
        polygon: polygon
        point: x, y

    Returns:
        df: a geopandas dataframe with columns: name, label and geometry
    """
    # read file
    if isinstance(df, str):
        df = pd.read_csv(df)

    # Detect geometry type
    geom_type = determine_geometry_type(df)

    # Check for uppercase names and set to lowercase
    df.columns = [x.lower() for x in df.columns]

    # convert to geodataframe
    if geom_type == "box":
        df['geometry'] = df.apply(
                lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    elif geom_type == "polygon":
        df['geometry'] = gpd.GeoSeries.from_wkt(df["polygon"])

    elif geom_type == "point":
        df["geometry"] = [shapely.geometry.Point(x, y)
        for x, y in zip(df.x.astype(float), df.y.astype(float))]
    else:
        raise ValueError("Geometry type {} not supported".format(geom_type))
    
    # convert to geodataframe
    df = gpd.GeoDataFrame(df, geometry='geometry')
    
    # remove any of the csv columns
    df = df.drop(columns=["polygon", "x", "y","xmin","ymin","xmax","ymax"], errors="ignore")
                          
    return df

def geo_to_image_coordinates(gdf, image_bounds, image_resolution):
    """
    Convert from projected coordinates to image coordinates
    Args:
        gdf: a pandas type dataframe with columns: name, xmin, ymin, xmax, ymax. Name is the relative path to the root_dir arg.
        image_bounds: bounds of the image
        image_resolution: resolution of the image
    Returns:
        gdf: a geopandas dataframe with the transformed to image origin
        """
    
    # unpack image bounds
    left, bottom, right, top = image_bounds
    gdf.geometry = gdf.geometry.translate(xoff=-left, yoff=-top)
    gdf.geometry = gdf.geometry.scale(xfact=1/image_resolution, yfact=1/image_resolution, origin=(0,0))

    return gdf

def round_with_floats(x):
    """Check if string x is float or int, return int, rounded if needed."""

    try:
        result = int(x)
    except BaseException:
        warnings.warn(
            "Annotations file contained non-integer coordinates. "
            "These coordinates were rounded to nearest int. "
            "All coordinates must correspond to pixels in the image coordinate system. "
            "If you are attempting to use projected data, "
            "first convert it into image coordinates see FAQ for suggestions.")
        result = int(np.round(float(x)))

    return result


def check_file(df):
    """Check a file format for correct column names and structure"""

    if not all(x in df.columns
               for x in ["image_path", "xmin", "xmax", "ymin", "ymax", "label"]):
        raise IOError("Input file has incorrect column names, "
                      "the following columns must exist "
                      "'image_path','xmin','ymin','xmax','ymax','label'.")

    return df


def check_image(image):
    """Check an image is three channel, channel last format
        Args:
           image: numpy array
        Returns: None, throws error on assert
    """
    if not image.shape[2] == 3:
        raise ValueError("image is expected have three channels, channel last format, "
                         "found image with shape {}".format(image.shape))


def image_to_geo_coordinates(gdf, root_dir, projected=True, flip_y_axis=False):
    """
    Convert from image coordinates to geographic coordinates
    Note that this assumes df is just a single plot being passed to this function
    Args:
        gdf: a geodataframe, see pandas_to_geopandas
        root_dir: directory of images to lookup image_path column
        projected: If True, convert from image to geographic coordinates, if False, keep in image coordinate system
        flip_y_axis: If True, reflect predictions over y axis to align with raster data in QGIS, which uses a negative y origin compared to numpy. See https://gis.stackexchange.com/questions/306684/why-does-qgis-use-negative-y-spacing-in-the-default-raster-geotransform
    Returns:
        df: a geospatial dataframe with the boxes optionally transformed to the target crs
    """
    # Raise a warning and confirm if a user sets projected to True when flip_y_axis is True.
    if flip_y_axis and projected:
        warnings.warn(
            "flip_y_axis is {}, and projected is {}. In most cases, projected should be False when inverting y axis. Setting projected=False"
            .format(flip_y_axis, projected), UserWarning)
        projected = False

    plot_names = gdf.image_path.unique()
    if len(plot_names) > 1:
        raise ValueError("This function projects a single plots worth of data. "
                         "Multiple plot names found {}".format(plot_names))
    else:
        plot_name = plot_names[0]

    rgb_path = "{}/{}".format(root_dir, plot_name)
    with rasterio.open(rgb_path) as dataset:
        bounds = dataset.bounds
        left, bottom, right, top = bounds
        pixelSizeX, pixelSizeY = dataset.res
        crs = dataset.crs
        transform = dataset.transform

        gdf.geometry = gdf.geometry.translate(xoff=left, yoff=top)

        if flip_y_axis:
            # Numpy uses top left 0,0 origin, flip along y axis.
            # See https://gis.stackexchange.com/questions/306684/why-does-qgis-use-negative-y-spacing-in-the-default-raster-geotransform
            gdf.geometry = gdf.geometry.scale(xfact=pixelSizeX, yfact=pixelSizeX, origin=(0,0))
    
    return gdf


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))

    return tuple(zip(*batch))

## Deprecated functions ##

# Should these be removed sooner?

def boxes_to_shapefile(df, root_dir, projected=True, flip_y_axis=False):
    """
    Convert from image coordinates to geographic coordinates
    Note that this assumes df is just a single plot being passed to this function
    Args:
       df: a pandas type dataframe with columns: name, xmin, ymin, xmax, ymax. Name is the relative path to the root_dir arg.
       root_dir: directory of images to lookup image_path column
       projected: If True, convert from image to geographic coordinates, if False, keep in image coordinate system
       flip_y_axis: If True, reflect predictions over y axis to align with raster data in QGIS, which uses a negative y origin compared to numpy. See https://gis.stackexchange.com/questions/306684/why-does-qgis-use-negative-y-spacing-in-the-default-raster-geotransform
    Returns:
       df: a geospatial dataframe with the boxes optionally transformed to the target crs
    """
    warnings.warn(
        "This method is deprecated and will be removed in version "
        "DeepForest 2.0.0, please use translate_image_to_geo_coordinates which works for points, boxes and polygons.",
        DeprecationWarning)
    
    # Raise a warning and confirm if a user sets projected to True when flip_y_axis is True.
    if flip_y_axis and projected:
        warnings.warn(
            "flip_y_axis is {}, and projected is {}. In most cases, projected should be False when inverting y axis. Setting projected=False"
            .format(flip_y_axis, projected), UserWarning)
        projected = False

    plot_names = df.image_path.unique()
    if len(plot_names) > 1:
        raise ValueError("This function projects a single plots worth of data. "
                         "Multiple plot names found {}".format(plot_names))
    else:
        plot_name = plot_names[0]

    rgb_path = "{}/{}".format(root_dir, plot_name)
    with rasterio.open(rgb_path) as dataset:
        bounds = dataset.bounds
        pixelSizeX, pixelSizeY = dataset.res
        crs = dataset.crs
        transform = dataset.transform

    if projected:
        # Convert image pixel locations to geographic coordinates
        xmin_coords, ymin_coords = rasterio.transform.xy(transform=transform,
                                                         rows=df.ymin,
                                                         cols=df.xmin,
                                                         offset='center')

        xmax_coords, ymax_coords = rasterio.transform.xy(transform=transform,
                                                         rows=df.ymax,
                                                         cols=df.xmax,
                                                         offset='center')

        # One box polygon for each tree bounding box
        # Careful of single row edge case where
        # xmin_coords comes out not as a list, but as a float
        if type(xmin_coords) == float:
            xmin_coords = [xmin_coords]
            ymin_coords = [ymin_coords]
            xmax_coords = [xmax_coords]
            ymax_coords = [ymax_coords]

        box_coords = zip(xmin_coords, ymin_coords, xmax_coords, ymax_coords)
        box_geoms = [
            shapely.geometry.box(xmin, ymin, xmax, ymax)
            for xmin, ymin, xmax, ymax in box_coords
        ]

        geodf = gpd.GeoDataFrame(df, geometry=box_geoms)
        geodf.crs = crs

        return geodf

    else:
        if flip_y_axis:
            # See https://gis.stackexchange.com/questions/306684/why-does-qgis-use-negative-y-spacing-in-the-default-raster-geotransform
            # Numpy uses top left 0,0 origin, flip along y axis.
            df['geometry'] = df.apply(
                lambda x: shapely.geometry.box(x.xmin, -x.ymin, x.xmax, -x.ymax), axis=1)
        else:
            df['geometry'] = df.apply(
                lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
        df = gpd.GeoDataFrame(df, geometry="geometry")

        return df
    
def annotations_to_shapefile(df, transform, crs):
    """
    Convert output from predict_image and  predict_tile to a geopandas data.frame

    Args:
        df: prediction data.frame with columns  ['xmin','ymin','xmax','ymax','label','score']
        transform: A rasterio affine transform object
        crs: A rasterio crs object
    Returns:
        results: a geopandas dataframe where every entry is the bounding box for a detected tree.
    """
    warnings.warn(
        "This method is deprecated and will be "
        "removed in version DeepForest 2.0.0, "
        "please use boxes_to_shapefile which unifies project_boxes and "
        "annotations_to_shapefile functionalities", DeprecationWarning)

    # Convert image pixel locations to geographic coordinates
    xmin_coords, ymin_coords = rasterio.transform.xy(transform=transform,
                                                     rows=df.ymin,
                                                     cols=df.xmin,
                                                     offset='center')

    xmax_coords, ymax_coords = rasterio.transform.xy(transform=transform,
                                                     rows=df.ymax,
                                                     cols=df.xmax,
                                                     offset='center')

    # If there is only one tree, the above comes out as a float, not a list
    if type(xmin_coords) == float:
        xmin_coords = [xmin_coords]
        ymin_coords = [ymin_coords]
        xmax_coords = [xmax_coords]
        ymax_coords = [ymax_coords]

    # One box polygon for each tree bounding box
    box_coords = zip(xmin_coords, ymin_coords, xmax_coords, ymax_coords)
    box_geoms = [
        shapely.geometry.box(xmin, ymin, xmax, ymax)
        for xmin, ymin, xmax, ymax in box_coords
    ]

    geodf = gpd.GeoDataFrame(df, geometry=box_geoms)
    geodf.crs = crs

    return geodf


def project_boxes(df, root_dir, transform=True):
    """
    Convert from image coordinates to geographic coordinates
    Note that this assumes df is just a single plot being passed to this function
    df: a pandas type dataframe with columns: name, xmin, ymin, xmax, ymax.
    Name is the relative path to the root_dir arg.
    root_dir: directory of images to lookup image_path column
    transform: If true, convert from image to geographic coordinates
    """
    warnings.warn(
        "This method is deprecated and will be removed in version "
        "DeepForest 2.0.0, please use boxes_to_shapefile which "
        "unifies project_boxes and annotations_to_shapefile functionalities",
        DeprecationWarning)
    plot_names = df.image_path.unique()
    if len(plot_names) > 1:
        raise ValueError("This function projects a single plots worth of data. "
                         "Multiple plot names found {}".format(plot_names))
    else:
        plot_name = plot_names[0]

    rgb_path = "{}/{}".format(root_dir, plot_name)
    with rasterio.open(rgb_path) as dataset:
        bounds = dataset.bounds
        pixelSizeX, pixelSizeY = dataset.res
        crs = dataset.crs

    if transform:
        # subtract origin. Recall that numpy origin is top left! Not bottom
        # left.
        df["xmin"] = (df["xmin"].astype(float) * pixelSizeX) + bounds.left
        df["xmax"] = (df["xmax"].astype(float) * pixelSizeX) + bounds.left
        df["ymin"] = bounds.top - (df["ymin"].astype(float) * pixelSizeY)
        df["ymax"] = bounds.top - (df["ymax"].astype(float) * pixelSizeY)

    # combine column to a shapely Box() object, save shapefile
    df['geometry'] = df.apply(
        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    df = gpd.GeoDataFrame(df, geometry='geometry')

    df.crs = crs

    return df
