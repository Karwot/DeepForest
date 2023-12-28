# test_utilities
import numpy as np
import os
import pytest
import pandas as pd
import rasterio as rio
from shapely import geometry
import geopandas as gpd

from deepforest import get_data
from deepforest import utilities
from deepforest import main

#import general model fixture
from .conftest import download_release

@pytest.fixture()
def config():
    config = utilities.read_config("deepforest_config.yml")
    return config

def test_xml_to_annotations():
    annotations = utilities.xml_to_annotations(
        xml_path=get_data("OSBS_029.xml"))
    print(annotations.shape)
    assert annotations.shape[0] == 61

def test_use_release(download_release):
    # Download latest model from github release
    release_tag, state_dict = utilities.use_release(check_release=False)

def test_use_bird_release(download_release):
    # Download latest model from github release
    release_tag, state_dict = utilities.use_bird_release()
    assert os.path.exists(get_data("bird.pt"))    
    
def test_float_warning(config):
    """Users should get a rounding warning when adding annotations with floats"""
    float_annotations = "tests/data/float_annotations.txt"
    with pytest.warns(UserWarning):
        annotations = utilities.xml_to_annotations(float_annotations)

def test_project_boxes():
    csv_file = get_data("OSBS_029.csv")
    df = pd.read_csv(csv_file)
    gdf = utilities.project_boxes(df, root_dir=os.path.dirname(csv_file))
    
    assert df.shape[0] == gdf.shape[0]

def test_shapefile_to_annotations_convert_to_boxes(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree","Tree"]
    df = pd.DataFrame({"geometry":sample_geometry,"label":labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf.to_file("{}/annotations.shp".format(tmpdir))
    image_path = get_data("OSBS_029.tif")
    shp = utilities.shapefile_to_annotations(shapefile="{}/annotations.shp".format(tmpdir), rgb=image_path, savedir=tmpdir)
    assert shp.shape[0] == 2
    
def test_shapefile_to_annotations(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree","Tree"]
    df = pd.DataFrame({"geometry":sample_geometry,"label":labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf["geometry"] = [geometry.box(left, bottom, right, top) for left, bottom, right, top in gdf.geometry.buffer(0.5).bounds.values]
    
    gdf.to_file("{}/annotations.shp".format(tmpdir))
    image_path = get_data("OSBS_029.tif")
    shp = utilities.shapefile_to_annotations(shapefile="{}/annotations.shp".format(tmpdir), rgb=image_path, savedir=tmpdir)
    assert shp.shape[0] == 2

def test_shapefile_to_annotations_incorrect_crs(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree","Tree"]
    df = pd.DataFrame({"geometry":sample_geometry,"label":labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32618")
    gdf["geometry"] = [geometry.box(left, bottom, right, top) for left, bottom, right, top in gdf.geometry.buffer(0.5).bounds.values]
    
    gdf.to_file("{}/annotations.shp".format(tmpdir))
    image_path = get_data("OSBS_029.tif")
    with pytest.raises(ValueError):
        shp = utilities.shapefile_to_annotations(shapefile="{}/annotations.shp".format(tmpdir), rgb=image_path, savedir=tmpdir)
def test_boxes_to_shapefile_projected(m):
    img = get_data("OSBS_029.tif")
    r = rio.open(img)
    df = m.predict_image(path=img)
    gdf = utilities.boxes_to_shapefile(df, root_dir=os.path.dirname(img), projected=True)
    
    #Confirm that each boxes within image bounds
    geom = geometry.box(*r.bounds)
    assert all(gdf.geometry.apply(lambda x: geom.intersects(geom)).values)
    
    #Edge case, only one row in predictions
    gdf = utilities.boxes_to_shapefile(df.iloc[:1,], root_dir=os.path.dirname(img), projected=True)
    assert gdf.shape[0] == 1

def test_boxes_to_shapefile_projected_from_predict_tile(m):
    img = get_data("OSBS_029.tif")
    r = rio.open(img)
    df = m.predict_tile(raster_path=img)
    gdf = utilities.boxes_to_shapefile(df, root_dir=os.path.dirname(img), projected=True)
    
    #Confirm that each boxes within image bounds
    geom = geometry.box(*r.bounds)
    assert all(gdf.geometry.apply(lambda x: geom.intersects(geom)).values)
    
    #Edge case, only one row in predictions
    gdf = utilities.boxes_to_shapefile(df.iloc[:1,], root_dir=os.path.dirname(img), projected=True)
    assert gdf.shape[0] == 1
    
# Test unprojected data, including warning if flip_y_axis is set to True, but projected is False
@pytest.mark.parametrize("flip_y_axis", [True, False])
@pytest.mark.parametrize("projected", [True, False])
def test_boxes_to_shapefile_unprojected(m, flip_y_axis, projected):
    img = get_data("OSBS_029.png")
    r = rio.open(img)
    df = m.predict_image(path=img)

    with pytest.warns(UserWarning):
        gdf = utilities.boxes_to_shapefile(
            df,
            root_dir=os.path.dirname(img),
            projected=projected,
            flip_y_axis=flip_y_axis)
    
    # Confirm that each boxes within image bounds
    geom = geometry.box(*r.bounds)
    assert all(gdf.geometry.apply(lambda x: geom.intersects(geom)).values)

def test_shapefile_to_annotations_boxes_projected(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree","Tree"]
    df = pd.DataFrame({"geometry":sample_geometry,"label":labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf["geometry"] = [geometry.box(left, bottom, right, top) for left, bottom, right, top in gdf.geometry.buffer(0.5).bounds.values]
    
    gdf.to_file("{}/test_shapefile_to_annotations_boxes_projected.shp".format(tmpdir))
    image_path = get_data("OSBS_029.tif")
    shp = utilities.shapefile_to_annotations(shapefile="{}/test_shapefile_to_annotations_boxes_projected.shp".format(tmpdir), rgb=image_path, savedir=tmpdir)
    assert shp.shape[0] == 2

def test_shapefile_to_annotations_polygons_projected(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf["geometry"] = [geometry.Polygon([(left, bottom), (left, top), (right, top), (right, bottom)]) for left, bottom, right, top in gdf.geometry.buffer(0.5).bounds.values]
    gdf.to_file("{}/test_shapefile_to_annotations_polygons_projected.shp".format(tmpdir))
    image_path = get_data("OSBS_029.tif")
    shp = utilities.shapefile_to_annotations(shapefile="{}/test_shapefile_to_annotations_polygons_projected.shp".format(tmpdir), rgb=image_path, savedir=tmpdir)
    assert shp.shape[0] == 2

def test_shapefile_to_annotations_points_projected(tmpdir):
    sample_geometry = [geometry.Point(404211.9 + 10,3285102 + 20),geometry.Point(404211.9 + 20,3285102 + 20)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32617")
    gdf.to_file("{}/test_shapefile_to_annotations_points_projected.shp".format(tmpdir))
    image_path = get_data("OSBS_029.tif")
    shp = utilities.shapefile_to_annotations(shapefile="{}/test_shapefile_to_annotations_points_projected.shp".format(tmpdir), rgb=image_path, savedir=tmpdir)
    assert shp.shape[0] == 2
    assert shp.geometry.iloc[0].type == "Point"

def test_shapefile_to_annotations_boxes_unprojected(tmpdir):
    # Create a sample GeoDataFrame with box geometries
    sample_geometry = [geometry.box(0, 0, 1, 1), geometry.box(2, 2, 3, 3)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf.to_file("{}/test_shapefile_to_annotations_boxes_unprojected.shp".format(tmpdir))
    image_path = get_data("OSBS_029.png")
    annotations = utilities.shapefile_to_annotations(shapefile="{}/test_shapefile_to_annotations_boxes_unprojected.shp".format(tmpdir), rgb=image_path, savedir=tmpdir)

    # Assert the expected number of annotations and geometry type
    assert annotations.shape[0] == 2
    assert annotations.geometry.iloc[0].type == "Polygon"

def test_shapefile_to_annotations_points_unprojected(tmpdir):
    # Create a sample GeoDataFrame with point geometries
    sample_geometry = [geometry.Point(0.5, 0.5), geometry.Point(2.5, 2.5)]
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf.to_file("{}/test_shapefile_to_annotations_points_unprojected.shp".format(tmpdir))
    image_path = get_data("OSBS_029.png")

    annotations = utilities.shapefile_to_annotations(shapefile="{}/test_shapefile_to_annotations_points_unprojected.shp".format(tmpdir), rgb=image_path, savedir=tmpdir)

    # Assert the expected number of annotations
    assert annotations.shape[0] == 2
    assert annotations.geometry.iloc[0].type == "Point"

def test_shapefile_to_annotations_polygons_unprojected(tmpdir):
    # Create a sample GeoDataFrame with polygon geometries with 6 points
    sample_geometry = [geometry.Polygon([(0, 0), (0, 2), (1, 1), (1, 0), (0, 0)]), geometry.Polygon([(2, 2), (2, 4), (3, 3), (3, 2), (2, 2)])]
    
    labels = ["Tree", "Tree"]
    df = pd.DataFrame({"geometry": sample_geometry, "label": labels})
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf.to_file("{}/test_shapefile_to_annotations_polygons_unprojected.shp".format(tmpdir))
    image_path = get_data("OSBS_029.png")

    # Call the function under test
    annotations = utilities.shapefile_to_annotations(shapefile="{}/test_shapefile_to_annotations_polygons_unprojected.shp".format(tmpdir), rgb=image_path, savedir=tmpdir)

    # Assert the expected number of annotations
    assert annotations.shape[0] == 2
    assert annotations.geometry.iloc[0].type == "Polygon"

def test_crop_raster_valid_crop(tmpdir):
    rgb_path = get_data("2018_SJER_3_252000_4107000_image_477.tif")
    raster_bounds = rio.open(rgb_path).bounds
    
    # Define the bounds for cropping
    bounds = (raster_bounds[0] + 10, raster_bounds[1] + 10, raster_bounds[0] + 30, raster_bounds[1] + 30)

    # Call the function under test
    result = utilities.crop_raster(bounds, rgb_path=rgb_path, savedir=tmpdir, filename="crop")

    # Assert the output filename
    expected_filename = str(tmpdir.join("crop.tif"))
    assert result == expected_filename

    # Assert the saved crop
    with rio.open(result) as src:
        # Round to nearest integer to avoid floating point errors
        assert np.round(src.bounds[2] - src.bounds[0]) == 20
        assert np.round(src.bounds[3] - src.bounds[1]) == 20
        assert src.count == 3
        assert src.dtypes == ("uint8", "uint8", "uint8")

def test_crop_raster_invalid_crop(tmpdir):
    rgb_path = get_data("2018_SJER_3_252000_4107000_image_477.tif")
    raster_bounds = rio.open(rgb_path).bounds
    
    # Define the bounds for cropping
    bounds = (raster_bounds[0] - 100, raster_bounds[1] - 100, raster_bounds[0] - 30, raster_bounds[1] - 30)

    # Call the function under test
    with pytest.raises(ValueError):
        result = utilities.crop_raster(bounds, rgb_path=rgb_path, savedir=tmpdir, filename="crop")


def test_crop_raster_no_savedir(tmpdir):
    rgb_path = get_data("2018_SJER_3_252000_4107000_image_477.tif")
    raster_bounds = rio.open(rgb_path).bounds
    
    # Define the bounds for cropping
    bounds = (int(raster_bounds[0] + 10), int(raster_bounds[1] + 10), int(raster_bounds[0] + 20), int(raster_bounds[1] + 20))

    # Call the function under test
    result = utilities.crop_raster(bounds, rgb_path=rgb_path)

    # Assert out is a output numpy array
    assert isinstance(result, np.ndarray)

def test_crop_raster_png_unprojected(tmpdir):
    # Define the bounds for cropping
    bounds = (0, 0, 100, 100)

    # Set the paths
    rgb_path = get_data("OSBS_029.png")
    savedir = str(tmpdir)
    filename = "crop"

    # Call the function under test
    result = utilities.crop_raster(bounds, rgb_path=rgb_path, savedir=savedir, filename=filename, driver="PNG")

    # Assert the output filename
    expected_filename = os.path.join(savedir, "crop.png")
    assert result == expected_filename

    # Assert the saved crop
    with rio.open(result) as src:
        # Assert the driver is PNG
        assert src.driver == "PNG"

        # Assert the crs is not present
        assert src.crs is None