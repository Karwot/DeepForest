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
from PIL import Image

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

    return annotations

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
                             rgb=None):
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
        gdf = shapefile.copy(deep=True)

    if rgb is None:
        if "image_path" not in gdf.columns:
            raise ValueError("No image_path column found in shapefile, please specify rgb path")
        else:
            rgb = gdf.image_path.unique()[0]
            print("Found image_path column in shapefile, using {}".format(rgb))

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
    if gdf.crs is not None:
        if raster_crs is None:
            raise ValueError("Shapefile has a crs, but raster does not. Please add a crs to the raster.")
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

def determine_geometry_type(df, verbose=True):
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
    if verbose:
        print("Found {} annotations in {} unique images with {} geometry type".format(
            df.shape[0], df.image_path.unique().shape[0], geometry_type))
    
    return geometry_type

def read_file(input, root_dir=None):
    """Read a file and return a geopandas dataframe
    Args:
        input: a path to a file or a pandas dataframe
        root_dir: Optional directory to prepend to the image_path column
    Returns:
        df: a geopandas dataframe with the properly formatted geometry column
    """
    # read file
    if isinstance(input, str):
        if input.endswith(".csv"):
            df = pd.read_csv(input)
        elif input.endswith(".shp"):
            df = shapefile_to_annotations(input)
        elif input.endswith(".xml"):
            df = xml_to_annotations(input)
        else:
            raise ValueError("File type {} not supported. DeepForest currently supports .csv, .shp or .xml files. See https://deepforest.readthedocs.io/en/latest/annotation.html ".format(df))
    else:
        if type(input) == pd.DataFrame:
            df = input.copy(deep=True)
        elif type(input) == gpd.GeoDataFrame:
            return shapefile_to_annotations(input)
        else:
            raise ValueError("Input must be a path to a file, geopandas or a pandas dataframe")
    
    if root_dir:
        df["image_path"] = df["image_path"].apply(lambda x: os.path.join(root_dir, x))

    if type(df) == pd.DataFrame:
        # If the geometry column is present, convert to geodataframe directly
        if "geometry" in df.columns:
            df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
            df.crs = None
        else:
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


def crop_raster(bounds, rgb_path=None, savedir=None, filename=None, driver="GTiff"):
    """
    Crop a raster to a bounding box, save as projected or unprojected crop
    Args:
        bounds: a tuple of (left, bottom, right, top) bounds
        rgb_path: path to the rgb image
        savedir: directory to save the crop
        filename: filename to save the crop "{}.tif".format(filename)"
        driver: rasterio driver to use, default to GTiff, can be 'GTiff' for projected data or 'PNG' unprojected data
    Returns:
        filename: path to the saved crop, if savedir specified
        img: a numpy array of the crop, if savedir not specified
    """
    left, bottom, right, top = bounds 
    src = rasterio.open(rgb_path)
    if src.crs is None:
        # Read unprojected data using PIL and crop numpy array
        img = np.array(Image.open(rgb_path))
        img = img[bottom:top, left:right, :]
        img = np.rollaxis(img, 2, 0)
        cropped_transform = None
        if driver == "GTiff":
            warnings.warn("Driver {} not supported for unprojected data, setting to 'PNG',".format(driver), UserWarning)
            driver = "PNG"
    else:
        # Read projected data using rasterio and crop
        img = src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)) 
        cropped_transform = rasterio.windows.transform(rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform), src.transform)
    if img.size == 0:
        raise ValueError("Bounds {} does not create a valid crop for source {}".format(bounds, src.transform))    
    if savedir:
        res = src.res[0]
        height = (top - bottom)/res
        width = (right - left)/res                 
        
        # Write the cropped image to disk with transform
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        if driver == "GTiff":
            filename = "{}/{}.tif".format(savedir, filename)
            with rasterio.open(filename, "w", driver="GTiff",height=height, width=width, count=img.shape[0], dtype=img.dtype, transform=cropped_transform) as dst:
                dst.write(img)
        elif driver == "PNG":
            # PNG driver does not support transform
            filename = "{}/{}.png".format(savedir, filename)
            with rasterio.open(filename, "w", driver="PNG",height=height, width=width, count=img.shape[0], dtype=img.dtype) as dst:
                dst.write(img)
        else:
            raise ValueError("Driver {} not supported".format(driver))
        
    if savedir:
        return filename
    else:
        return img   

def crop_annotations_to_bounds(gdf, bounds):
    """
    Crop a geodataframe of annotations to a bounding box
    Args:
        gdf: a geodataframe of annotations
        bounds: a tuple of (left, bottom, right, top) bounds
    Returns:
        gdf: a geodataframe of annotations cropped to the bounds
    """
    # unpack image bounds
    left, bottom, right, top = bounds
    
    # Crop the annotations
    gdf.geometry = gdf.geometry.translate(xoff=-left, yoff=-bottom)
    
    return gdf

def geo_to_image_coordinates(gdf, image_bounds, image_resolution):
    """
    Convert from projected coordinates to image coordinates
    Args:
        gdf: a pandas type dataframe with columns: name, xmin, ymin, xmax, ymax. Name is the relative path to the root_dir arg.
        image_bounds: bounds of the image
        image_resolution: resolution of the image
    Returns:
        gdf: a geopandas dataframe with the transformed to image origin. CRS is removed
        """
    
    # unpack image bounds
    left, bottom, right, top = image_bounds
    gdf.geometry = gdf.geometry.translate(xoff=-left, yoff=-bottom)
    gdf.geometry = gdf.geometry.scale(xfact=1/image_resolution, yfact=1/image_resolution, origin=(0,0))
    gdf.crs = None

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