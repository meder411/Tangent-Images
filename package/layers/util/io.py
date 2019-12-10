from skimage import io
import matplotlib.pyplot as plt
import plyfile
import numpy as np


def write_ply(output_path,
              pts,
              normals=None,
              rgb=None,
              faces=None,
              face_rgb=None,
              text=False):
    '''
    Points should be 3 x N. Optionally, faces, normals, and RGB should be 3 x N as well. NOTE 10/2/19: faces seems to need to be a N x 3...
    '''
    names = 'x,y,z'
    formats = 'f4,f4,f4'
    if normals is not None:
        pts = np.vstack((pts, normals))
        names += ',nx,ny,nz'
        formats += ',f4,f4,f4'
    if rgb is not None:
        pts = np.vstack((pts, rgb))
        names += ',red,green,blue'
        formats += ',u1,u1,u1'
    pts = np.core.records.fromarrays(pts, names=names, formats=formats)
    el = [plyfile.PlyElement.describe(pts, 'vertex')]
    if faces is not None:
        faces = faces.astype(np.int32)
        faces = faces.copy().ravel().view([("vertex_indices", "u4", 3)])
        el.append(plyfile.PlyElement.describe(faces, 'face'))
    if face_rgb is not None:
        el.append(plyfile.PlyElement.describe(face_rgb, 'face'))

    plyfile.PlyData(el, text=text).write(output_path)


def writeImage(path, img):
    '''img is numpy format'''
    assert img.ndim == 3, 'Image must be 3-dimensional'
    assert img.shape[-1] == 1 or img.shape[
        -1] == 3, 'Image must have 1 channel or 3 channels'
    assert img.dtype == np.uint8, 'Image dtype must be np.uint8'

    io.imsave(path, img)


def writeHeatmap(path, data, max_val=None, cmap_type='plasma'):
    '''Writes a heatmap visualization of the data. Data is a NxM np.float32 array'''
    assert data.ndim == 2, 'Data must be 2-dimensional'
    assert data.dtype == np.float32, 'Data dtype must be np.float32'

    cmap = plt.get_cmap(cmap_type)
    if max_val is None:
        data = data / data.max()
    else:
        data = data / max_val

    io.imsave(path, cmap(data))