# TmLibrary - TissueMAPS library for distibuted image analysis routines.
# Copyright (C) 2016-2019 University of Zurich.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''A `Handle` describes a key-value pair which is either passed as
an argument to a Jterator module function or is returned by the function. The
approach can be considered a form of metaprogramming, where the object extends
the code of the actual module function via its properties and methods.
This is used to assert the correct type of arguments and return values and
enables storing data generated by modules to make it accessible outside the
scope of the module or retrieving data from the store when required by modules.
The object's attributes are specified as a mapping in a
`handles` YAML module descriptor file.
'''
import re
import sys
import json
import numpy as np
import pandas as pd
import mahotas as mh
import cv2
import skimage
import logging
import collections
import skimage.draw
import shapely.geometry
from geoalchemy2.shape import to_shape
from abc import ABCMeta
from abc import abstractproperty
from abc import abstractmethod

from tmlib.utils import same_docstring_as
from tmlib.utils import assert_type
from tmlib.image import SegmentationImage
import jtlib.utils

logger = logging.getLogger(__name__)


class Handle(object):

    '''Abstract base class for a handle.'''

    __metaclass__ = ABCMeta

    @assert_type(name='basestring', help='basestring')
    def __init__(self, name, help):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must either match a parameter of the module
            function in case the item represents an input argument or the key
            of a key-value pair of the function's return value
        help: str
            help message
        '''
        self.name = name
        self.help = help

    @property
    def type(self):
        '''str: handle type'''
        return self.__class__.__name__


class InputHandle(Handle):

    '''Abstract base class for a handle whose value is used as an argument for
    a module function.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, name, value, help):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must match a parameter of the module
            function
        value:
            the actual argument of the module function parameter
        help: str
            help message
        '''
        super(InputHandle, self).__init__(name, help)
        self.value = value

    def to_dict(self):
        '''Returns attributes "name", "type", "help" and "value" as
        key-value pairs.

        Return
        ------
        dict
        '''
        attrs = dict()
        attrs['name'] = self.name
        attrs['type'] = self.type
        attrs['help'] = self.help
        attrs['value'] = self.value
        return attrs


class OutputHandle(Handle):

    '''Abstract base class for a handle whose value is returned by a module
    function.
    '''

    __metaclass__ = ABCMeta

    @same_docstring_as(Handle.__init__)
    def __init__(self, name, help):
        super(OutputHandle, self).__init__(name, help)

    @abstractproperty
    def value(self):
        '''value returned by module function'''
        pass

    def to_dict(self):
        '''Returns attributes "name", "type" and "help" as key-value pairs.

        Return
        ------
        dict
        '''
        attrs = dict()
        attrs['name'] = self.name
        attrs['type'] = self.type
        attrs['help'] = self.help
        return attrs


class PipeHandle(Handle):

    '''Abstract base class for a handle whose value can be piped between
    modules, i.e. returned by one module function and potentially passed as
    argument to another.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, name, key, help):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must either match a parameter of the module
            function in case the item represents an input argument or the key
            of a key-value pair of the function's return value
        key: str
            unique and hashable identifier; it serves as
            lookup identifier to retrieve the actual value of the item
        help: str
            help message

        '''
        super(PipeHandle, self).__init__(name, help)
        self.key = str(key)

    @abstractproperty
    def value(self):
        '''Data that's returned by module function and possibly passed
        to other module functions.
        '''
        pass

    def to_dict(self):
        '''Returns attributes "name", "type", "help" and "key" as key-value
        pairs.

        Return
        ------
        dict
        '''
        attrs = dict()
        attrs['name'] = self.name
        attrs['type'] = self.type
        attrs['help'] = self.help
        attrs['key'] = self.key
        return attrs


class Image(PipeHandle):

    '''Base class for an image handle.'''

    @same_docstring_as(PipeHandle.__init__)
    def __init__(self, name, key, help):
        super(Image, self).__init__(name, key, help)

    @property
    def value(self):
        '''numpy.ndarray: pixels/voxels array'''
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                'Returned value for "%s" must have type numpy.ndarray.'
                % self.name
            )
        self._value = value

    def iter_planes(self):
        '''Iterates over 2D pixel planes of the image.

        Returns
        -------
        Generator[Tuple[Tuple[int], numpy.ndarray]]
             pixels at each time point and z-level
        '''
        array = self.value
        if array.ndim == 2:
            array = array[..., np.newaxis, np.newaxis]
        elif array.ndim == 3:
            array = array[..., np.newaxis]
        for t in xrange(array.shape[-1]):
            for z in xrange(array[..., t].shape[-1]):
                yield ((t, z), array[:, :, z, t])

    def iter_volumes(self):
        '''Iterates over 3D voxel volumes of the image.

        Returns
        -------
        Generator[Tuple[int, numpy.ndarray]]
             voxels at each time point
        '''
        array = self.value
        if array.ndim == 2:
            array = array[..., np.newaxis, np.newaxis]
        elif array.ndim == 3:
            array = array[..., np.newaxis]
        for t in xrange(array.shape[-1]):
            yield (t, array[:, :, :, t])


class IntensityImage(Image):

    '''Class for an intensity image handle, where image pixel values encode
    intensity.
    '''

    def __init__(self, name, key, help=''):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must either match a parameter of the module
            function in case the item represents an input argument or the key
            of a key-value pair of the function's return value
        key: str
            unique and hashable identifier; it serves as
            lookup identifier to retrieve the actual value of the item
        help: str, optional
            help message (default: ``""``)
        '''
        super(IntensityImage, self).__init__(name, key, help)

    @property
    def value(self):
        '''
        Returns
        -------
        numpy.ndarray[Union[numpy.uint8, numpy.uint16]]: pixels/voxels array
        '''
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                'Returned value for "%s" must have type numpy.ndarray.'
                % self.name
            )
        if not(value.dtype == np.uint8 or value.dtype == np.uint16):
            raise TypeError(
                'Returned value for "%s" must have data type '
                'uint8 or uint16' % self.name
            )
        self._value = value

    def __str__(self):
        return '<IntensityImage(name=%r, key=%r)>' % (self.name, self.key)


class MaskImage(Image):

    '''Class for an image handle, where pixels of the image with zero values
    represent background and pixels with values greater than zero represent
    foreground.
    '''

    @same_docstring_as(Image.__init__)
    def __init__(self, name, key, help=''):
        super(MaskImage, self).__init__(name, key, help)

    @property
    def value(self):
        '''numpy.ndarray: pixels/voxels array'''
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                'Returned value for "%s" must have type numpy.ndarray.'
                % self.name
            )
        if not(value.dtype == np.int32 or value.dtype == np.bool):
            raise TypeError(
                'Returned value for "%s" must have data type int32 or bool.'
                % self.name
            )
        self._value = value


class LabelImage(MaskImage):

    '''Class for an image handle, where connected pixel components of the image
    are labeled such that each component has a unique positive integer
    label and background is zero.
    '''

    @same_docstring_as(Image.__init__)
    def __init__(self, name, key, help=''):
        super(LabelImage, self).__init__(name, key, help)

    @property
    def value(self):
        '''numpy.ndarray[numpy.int32]: pixels/voxels array'''
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                'Returned value for "%s" must have type numpy.ndarray.'
                % self.name
            )
        if not value.dtype == np.int32:
            raise TypeError(
                'Returned value for "%s" must have data type int32.'
                % self.name
            )
        self._value = value

    def __str__(self):
        return '<LabelImage(name=%r, key=%r)>' % (self.name, self.key)


class BinaryImage(MaskImage):

    '''Class for an image handle, where pixels of the image are
    either background (``False``) or foreground (``True``).
    '''

    @same_docstring_as(Image.__init__)
    def __init__(self, name, key, help=''):
        super(BinaryImage, self).__init__(name, key, help)

    @property
    def value(self):
        '''numpy.ndarray[numpy.bool]: pixels/voxels array'''
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                'Value of key "%s" must have type numpy.ndarray.' % self.name
            )
        if value.dtype != np.bool:
            raise TypeError(
                'Value of key "%s" must have data type bool.' % self.name
            )
        self._value = value

    def __str__(self):
        return '<BinaryImage(name=%r, key=%r)>' % (self.name, self.key)


class SegmentedObjects(LabelImage):

    '''Class for a segmented objects handle, which represents a special type of
    label image handle, where pixel values encode segmented objects that should
    ultimately be visualized by `TissueMAPS` and for which features can be
    extracted.
    '''

    def __init__(self, name, key, help=''):
        '''
        Parameters
        ----------
        name: str
            name of the item
        key: str
            name that should be assigned to the objects
        '''
        super(SegmentedObjects, self).__init__(name, key, help)
        self._features = collections.defaultdict(list)
        self.save = False
        self.represent_as_polygons = True

    @property
    def labels(self):
        '''List[int]: unique object identifier labels'''
        return np.unique(self.value[self.value > 0]).astype(int).tolist()

    def iter_points(self, y_offset, x_offset):
        '''Iterates over point representations of segmented objects.
        The coordinates of the centroid points are relative to the global map,
        i.e. an offset is added to the image site specific coordinates.

        Parameters
        ----------
        y_offset: int
            global vertical offset that needs to be subtracted from
            *y*-coordinates (*y*-axis is inverted)
        x_offset: int
            global horizontal offset that needs to be added to x-coordinates

        Returns
        -------
        Generator[Tuple[Union[int, shapely.geometry.point.Point]]]
            time point, z-plane, label and point geometry
        '''
        logger.debug('calculate centroids for objects of type "%s"', self.key)
        points = dict()
        for (t, z), plane in self.iter_planes():
            centroids = mh.center_of_mass(plane, labels=plane)
            centroids[:, 1] += x_offset
            centroids[:, 0] += y_offset
            centroids[:, 0] *= -1
            for label in self.labels:
                y = int(centroids[label, 0])
                x = int(centroids[label, 1])
                point = shapely.geometry.Point(x, y)
                yield (t, z, label, point)

    def iter_polygons(self, y_offset, x_offset):
        '''Iterates over polygon representations of segmented objects.
        The coordinates of the polygon contours are relative to the global map,
        i.e. an offset is added to the image site specific coordinates.

        Parameters
        ----------
        y_offset: int
            global vertical offset that needs to be subtracted from
            *y*-coordinates (*y*-axis is inverted)
        x_offset: int
            global horizontal offset that needs to be added to *x*-coordinates

        Returns
        -------
        Generator[Tuple[Union[int, shapely.geometry.polygon.Polygon]]]
            time point, z-plane, label and polygon
        '''
        logger.debug('calculate polygons for objects type "%s"', self.key)
        for (t, z), plane in self.iter_planes():
            img = SegmentationImage(plane)
            for label, geometry in img.extract_polygons(y_offset, x_offset):
                yield (t, z, label, geometry)

    def add_polygons(self, polygons, y_offset, x_offset, dimensions):
        '''Creates a label image representation of segmented objects based
        on global map coordinates of object contours.

        Parameters
        ----------
        polygons: List[List[Tuple[Union[int, shapely.geometry.polygon.Polygon]]]]
            label and polygon geometry for segmented objects at each z-plane
            and time point
        y_offset: int
            global vertical offset that needs to be subtracted from
            y-coordinates
        x_offset: int
            global horizontal offset that needs to be subtracted from
            x-coordinates
        dimensions: Tuple[int]
            *y*, *x* dimensions of image pixel planes

        Returns
        -------
        numpy.ndarray[numpy.int32]
            label image
        '''
        zstacks = list()
        for poly in polygons:
            zplanes = list()
            for p in poly:
                image = SegmentationImage.create_from_polygons(
                    p, y_offset, x_offset, dimensions
                )
                zplanes.append(image.array)
            array = np.stack(zplanes, axis=-1)
            zstacks.append(array)
        self.value = np.stack(zstacks, axis=-1)
        return self.value

    @property
    def is_border(self):
        '''Dict[Tuple[int], pandas.Series[bool]]: ``True`` if object lies
        at the border of the image and ``False`` otherwise
        '''
        mapping = dict()
        for (t, z), plane in self.iter_planes():
            for label, is_border in self._find_border_objects(plane).iteritems():
                mapping[(t, z, label)] = bool(is_border)
        return mapping

    @staticmethod
    def _find_border_objects(img):
        '''Finds the objects at the border of a labeled image.

        Parameters
        ----------
        img: numpy.ndarray[int32]
            labeled pixels array

        Returns
        -------
        Dict[int: bool]
            ``True`` if an object lies at the border of the `img` and
            ``False`` otherwise
        '''
        edges = [np.unique(img[0, :]),   # first row
                 np.unique(img[-1, :]),  # last row
                 np.unique(img[:, 0]),   # first col
                 np.unique(img[:, -1])]  # last col

        # Count only unique ids and remove 0 since it signals 'empty space'
        border_ids = list(reduce(set.union, map(set, edges)).difference({0}))
        object_ids = np.unique(img[img != 0])
        return {o: True if o in border_ids else False for o in object_ids}

    @property
    def save(self):
        '''bool: whether objects should be saved'''
        return self._save

    @save.setter
    def save(self, value):
        if not isinstance(value, bool):
            raise TypeError('Attribute "save" must have type bool.')
        self._save = value

    @property
    def represent_as_polygons(self):
        '''bool: whether objects should be represented as polygons'''
        return self._represent_as_polygons

    @represent_as_polygons.setter
    def represent_as_polygons(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                'Attribute "represent_as_polygons" must have type bool.'
            )
        self._represent_as_polygons = value

    @property
    def measurements(self):
        '''List[pandas.DataFrame]: features extracted for
        segmented objects at each time point
        '''
        if self._features:
            values = list()
            for t in sorted(self._features.keys()):
                values.append(self._features[t])
            return [pd.concat(v, axis=1) for v in values]
        else:
            return [pd.DataFrame()]

    @measurements.setter
    def measurements(self, value):
        if not isinstance(value, list):
            raise TypeError(
                'Argument "measurements" must have type list.'
            )
        self._features = collections.defaultdict(list)
        for i, v in enumerate(value):
            if not isinstance(v, pd.DataFrame):
                raise TypeError(
                    'Items of argument "measurements" must have type '
                    'pandas.DataFrame.'
                )
            self._features[i] = [v]

    def add_measurement(self, measurement):
        '''Adds an additional measurement.

        Parameters
        ----------
        measurement: tmlib.workflow.jterator.handles.Measurement
            measured features for each segmented object
        '''
        if not isinstance(measurement, Measurement):
            raise TypeError(
                'Argument "measurement" must have type '
                'tmlib.workflow.jterator.handles.Measurement.'
            )
        for t, val in enumerate(measurement.value):
            if len(val.index) < len(self.labels):
                logger.warn(
                    'missing values for object type "%s" at time point %d',
                    self.key, t
                )
                for label in self.labels:
                    if label not in val.index:
                        logger.warn(
                            'add NaN values for missing object #%d', label
                        )
                        val.loc[label, :] = np.NaN
                val.sort_index(inplace=True)
            elif len(val.index) > len(self.labels):
                if len(np.unique(val.index)) < len(val.index):
                    logger.warn(
                        'duplicate values for "%s" at time point %d',
                        measurement.name, t
                    )
                    logger.info('remove duplicates and keep first')
                    val = val[~val.index.duplicated(keep='first')]
                else:
                    logger.warn(
                        'too many values for object type "%s" at time point %d',
                        self.key, t
                    )
                    for i in val.index:
                        if i not in self.labels:
                            logger.warn('remove values for object #%d', i)
                            val.drop(i, inplace=True)
            if np.any(val.index.values != np.array(self.labels)):
                raise ValueError(
                    'Labels of objects for "%s" at time point %d do not match!'
                    % (measurement.name, t)
                )
            if len(np.unique(val.columns)) != len(val.columns):
                raise ValueError(
                    'Column names of "%s" at time point %d must be unique.'
                    % (measurement.name, t)
                )
            self._features[t].append(val)

    def __str__(self):
        return '<SegmentedObjects(name=%r, key=%r)>' % (self.name, self.key)


class Scalar(InputHandle):

    '''Abstract base class for a handle for a scalar input argument.'''

    __metaclass__ = ABCMeta

    @assert_type(value=['int', 'float', 'basestring', 'bool', 'types.NoneType'])
    def __init__(self, name, value, help='', options=None):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must match a parameter of the module
            function
        value: str or int or float or bool
            value of the item, i.e. the actual argument of the function
            parameter
        help: str, optional
            help message (default: ``""``)
        options: List[str or int or float or bool]
            possible values for `value`
        '''
        if options is None:
            options = []
        if options:
            if value is not None:
                if value not in options:
                    raise ValueError(
                        'Argument "value" can be either "%s"'
                        % '" or "'.join(options)
                    )
        super(Scalar, self).__init__(name, value, help)
        self.options = options


class Boolean(Scalar):

    '''Handle for a boolean input argument.'''

    @assert_type(value='bool')
    def __init__(self, name, value, help='', options=None):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must match a parameter of the module
            function
        value: bool
            value of the item, i.e. the actual argument of the function
            parameter
        help: str, optional
            help message (default: ``""``)
        options: List[bool]
            possible values for `value`
        '''
        if options is None:
            options = [True, False]
        if not all([isinstance(o, bool) for o in options]):
            raise TypeError('Options for "Boolean" can only be boolean.')
        super(Boolean, self).__init__(name, value, help, options)

    def __str__(self):
        return '<Boolean(name=%r, value=%r)>' % (self.name, self.value)


class Numeric(Scalar):

    '''Handle for a numeric input argument.'''

    @assert_type(value=['int', 'float', 'types.NoneType'])
    def __init__(self, name, value, help='', options=None):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must match a parameter of the module
            function
        value: int or float
            value of the item, i.e. the actual argument of the function
            parameter
        help: str, optional
            help message (default: ``""``)
        options: List[int or float]
            possible values for `value`
        '''
        if options is None:
            options = []
        super(Numeric, self).__init__(name, value, help, options)

    def __str__(self):
        return '<Numeric(name=%r, value=%r)>' % (self.name, self.value)


class Character(Scalar):

    '''Handle for a character input argument.'''

    @assert_type(value=['basestring', 'types.NoneType'])
    def __init__(self, name, value, help='', options=None):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must match a parameter of the module
            function
        value: basestring
            value of the item, i.e. the actual argument of the function
            parameter
        help: str, optional
            help message (default: ``""``)
        options: List[basestring]
            possible values for `value`
        '''
        if options is None:
            options = []
        super(Character, self).__init__(name, value, help, options)

    def __str__(self):
        return '<Character(name=%r, value=%r)>' % (self.name, self.value)


class Sequence(InputHandle):

    '''Class for a sequence input argument handle.'''

    @assert_type(value='list')
    def __init__(self, name, value, help=''):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must match a parameter of the module
            function
        value: List[str or int or float]
            value of the item, i.e. the actual argument of the function
            parameter
        help: str, optional
            help message (default: ``""``)
        '''
        for v in value:
            if all([not isinstance(v, t) for t in {int, float, basestring}]):
                raise TypeError(
                    'Elements of argument "value" must have type '
                        'int, float, or basestring.')
        super(Sequence, self).__init__(name, value, help)

    def __str__(self):
        return '<Sequence(name=%r)>' % self.name


class Set(InputHandle):

    '''Unordered set of values. Discards all repeated values.'''

    @assert_type(value=['list', 'set'])
    def __init__(self, name, value, help=''):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must match a parameter of the module
            function
        value: Set[str or int or float]
            value of the item, i.e. the actual argument of the function
            parameter
        help: str, optional
            help message (default: ``""``)
        '''
        for v in value:
            if all([not isinstance(v, t) for t in {int, float, basestring}]):
                raise TypeError(
                    'Elements of argument "value" must have type '
                        'int, float, or basestring.')
        super(Set, self).__init__(name, set(value), help)

    def __str__(self):
        return '<Set(name=%r)>' % self.name


class Plot(InputHandle):

    '''Handle for a plot that indicates whether the module should
    generate a figure or rather run in headless mode.
    '''

    @assert_type(value='bool')
    def __init__(self, name, value=False, help='', options=None):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must match a parameter of the module
            function
        value: bool, optional
            whether plotting should be activated (default: ``False``)
        help: str, optional
            help message (default: ``""``)
        options: List[bool]
            possible values for `value`
        '''
        if options is None:
            options = [True, False]
        if not all([isinstance(o, bool) for o in options]):
            raise TypeError('Options for "Plot" can only be boolean')
        super(Plot, self).__init__(name, value, help)

    def __str__(self):
        return (
            '<Plot(name=%r, active=%r)>' % (self.name, self.value)
        )


class Measurement(OutputHandle):

    '''Handle for a measurement whose value is a list of two-dimensional labeled
    arrays with *n* rows and *p* columns, where *n* is the number of segmented
    objects and *p* the number of features that were measured for the
    referenced segmented objects.
    '''

    _NAME_PATTERN = re.compile(r'^[A-Za-z0-9_-]+$')

    @assert_type(
        objects='basestring',
        objects_ref='basestring', channel_ref=['basestring', 'types.NoneType']
    )
    def __init__(self, name, objects, objects_ref, channel_ref=None, help=''):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must match a parameter of the module
            function
        objects: str
            object type to which features should are assigned
        objects_ref: str
            reference to object type for which features were extracted
            (may be the same as `objects`)
        channel_ref: str, optional
            reference to channel from which features were extracted
            (default: ``None``)
        help: str, optional
            help message (default: ``""``)
        '''
        super(Measurement, self).__init__(name, help)
        self.objects = objects
        self.objects_ref = objects_ref
        self.channel_ref = channel_ref

    @property
    def value(self):
        '''List[pandas.DataFrame]: features extracted for
        segmented objects at each time point
        '''
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value, list):
            raise TypeError(
                'Value of key "%s" must have type list.'
                % self.name
            )
        if not all([isinstance(v, pd.DataFrame) for v in value]):
            raise TypeError(
                'Elements of returned value of "%s" must have type '
                'pandas.DataFrame.' % self.name
            )
        for v in value:
            for name in v.columns:
                if not self._NAME_PATTERN.search(name):
                    raise ValueError(
                        'Feature name "%s" must only contain '
                        'alphanumerical characters or underscores or hyphens'
                        % name
                    )
        self._value = value

    def to_dict(self):
        '''Returns attributes "name", "type", "help", "objects", "objects_ref"
        and "channel_ref" as key-value pairs.

        Return
        ------
        dict
        '''
        attrs = dict()
        attrs['name'] = self.name
        attrs['objects'] = self.objects
        attrs['objects_ref'] = self.objects_ref
        attrs['channel_ref'] = self.channel_ref
        attrs['type'] = self.type
        attrs['help'] = self.help
        return attrs

    def __str__(self):
        return '<Measurement(name=%r, objects=%r)>' % (self.name, self.objects)


class Figure(OutputHandle):

    '''Handle for a figure whose value is a JSON string representing
    a figure created by a module, see
    `Plotly JSON schema <http://help.plot.ly/json-chart-schema/>`_.
    '''

    def __init__(self, name, help=''):
        '''
        Parameters
        ----------
        name: str
            name of the item, which must match a parameter of the module
            function
        key: str
            name that should be given to the objects
        help: str, optional
            help message (default: ``""``)
        '''
        super(Figure, self).__init__(name, help)

    @property
    def value(self):
        '''str: JSON representation of a figure'''
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value, basestring):
            raise TypeError(
                'Value of key "%s" must have type basestring.' % self.name
            )
        if value:
            try:
                json.loads(value)
            except ValueError:
                raise ValueError(
                    'Figure "%s" is not valid JSON.' % self.name
                )
        else:
            # minimal valid JSON
            value = json.dumps(dict())
        self._value = str(value)

    def __str__(self):
        return '<Figure(name=%r)>' % self.name


def create_handle(type, **kwargs):
    '''Factory function to create an instance of an implementation of the
    :class:`tmlib.workflow.jterator.handles.Handle` abstract base class.

    Parameters
    ----------
    type: str
        type of the handle item; must match a name of one of the
        implemented classes in :mod:`tmlib.workflow.jterator.handles`
    **kwargs: dict
        keyword arguments that are passed to the constructor of the class

    Returns
    -------
    tmlib.jterator.handles.Handle

    Raises
    ------
    AttributeError
        when `type` is not a valid class name
    TypeError
        when an unexpected keyword is passed to the constructor of the class
    '''
    current_module = sys.modules[__name__]
    try:
        class_object = getattr(current_module, type)
    except AttributeError:
        raise AttributeError('"%s" is not a valid handles type.' % type)
    return class_object(**kwargs)
