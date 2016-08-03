import os
import json
import numpy as np

basedir = os.path.abspath(os.path.dirname(__file__))


def get_zipcode_shapes():
    f = open(os.path.join(basedir, 'data/zipcodes.json'))
    json_data = json.load(f)
    f.close()

    data = {}
    for i, feature in enumerate(json_data['features']):
        zipcode = int(feature['properties']['ZCTA5CE10'])
        coords = feature['geometry']['coordinates']
        data[zipcode] = {}
        data[zipcode]['coords'] = coords
        data[zipcode]['type'] = feature['geometry']['type']
    return data


def get_boro_shapes():
    f = open(os.path.join(basedir, 'data/boroughs_noclip.json'))
    json_data = json.load(f)
    f.close()

    data = {}
    for i, feature in enumerate(json_data['features']):
        boro = feature['properties']['BoroName']
        coords = feature['geometry']['coordinates']
        data[boro] = {}
        data[boro]['coords'] = coords
        data[boro]['type'] = feature['geometry']['type']
    return data


def arrange_polygons(shape_dict, ks):
    all_xs = []
    all_ys = []
    for k in ks:
        if shape_dict[k]['type'] == 'MultiPolygon':
            xs = np.array([])
            ys = np.array([])
            for polygon in shape_dict[k]['coords']:
                xs = np.append(xs, np.concatenate(
                    [np.array(polygon[0])[:, 0], [np.nan]]))
                ys = np.append(ys, np.concatenate(
                    [np.array(polygon[0])[:, 1], [np.nan]]))
        elif shape_dict[k]['type'] == 'Polygon':
            # this does not take into account patches with holes,
            # we're assuming no holes
            xs = np.array(shape_dict[k]['coords'][0])[:, 0]
            ys = np.array(shape_dict[k]['coords'][0])[:, 1]

        all_xs.append(xs)
        all_ys.append(ys)
    return all_xs, all_ys


def to_web_mercator(x_long, y_lat):
    semimajorAxis = 6378137.0  # WGS84 spheroid semimajor axis
    east = x_long * 0.017453292519943295
    north = y_lat * 0.017453292519943295

    northing = 3189068.5 * np.log((1.0 + np.sin(north)) / (1.0 - np.sin(north)))
    easting = semimajorAxis * east

    return easting, northing
