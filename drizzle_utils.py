
import numpy as np
import astropy.units as u
from astropy.modeling import models
from gwcs import coordinate_frames as cf
from gwcs import wcs as gwcs_wcs
from reproject import reproject_interp
from astropy import coordinates as coord

def combine_image_reproject(
    normalized_psf_tensor,
    centroids_d_tensor,
    wts_d_tensor,
    oversample=2,
    pixel_scale_arcsec=0.031,
    output_size=None,
):
    """
    Reproject-and-combine cutouts onto a common oversampled grid using gwcs + reproject_interp.

    Parameters
    ----------
    normalized_psf_tensor : ndarray
        PSF cutouts with shape (N,H,W) or (N,H,W,1). Assumed in detector pixels.
    centroids_d_tensor : ndarray
        (N,2) array of sub/pixel centroids (x,y) in the *cutout pixel frame* for each image.
        Convention here assumes pixel index origin at 0 (i.e., array[0,0] == (x=0,y=0)).
    wts_d_tensor : ndarray
        (N,) weights for each cutout. Will be used for a weighted average.
    oversample : int
        Output oversampling factor (e.g., 3 -> pixel scale is native/3).
    pixel_scale_arcsec : float
        Native detector pixel scale in arcsec/pixel. Default 0.031" (NIRCam SW).
        If you’re on a different detector, change this (e.g., 0.063 for NIRCam LW).
    output_size : int or tuple[int,int] or None
        Shape of the output grid (Hout, Wout). If None, uses
        ceil(max(H,W)*oversample) and makes it square and odd-sized to keep a central pixel.

    Returns
    -------
    reprojected_combined : ndarray
        Weighted-average, reprojected image on the common oversampled grid.
    footprint_stack : list[ndarray]
        List of per-image footprints returned by reproject_interp.
    output_wcs : gwcs.wcs.WCS
        The WCS of the combined oversampled grid.
    """
    # ---- normalize/shape housekeeping ----
    arr = np.asarray(normalized_psf_tensor)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]  # (N,H,W)
    if arr.ndim != 3:
        raise ValueError('normalized_psf_tensor must be (N,H,W) or (N,H,W,1).')
    N, H, W = arr.shape

    centroids = np.asarray(centroids_d_tensor, dtype=float)
    if centroids.shape != (N, 2):
        raise ValueError('centroids_d_tensor must have shape (N,2).')

    wts = np.asarray(wts_d_tensor, dtype=float).reshape(-1)
    if wts.shape != (N,):
        raise ValueError('wts_d_tensor must have shape (N,).')

    # ---- build the *output* WCS (oversampled grid centered on its middle pixel) ----
    # frames
    detector_frame = cf.Frame2D(
        name='detector', axes_names=('x','y'), unit=(u.pix,u.pix), axes_order=(0,1)
    )
    sky_frame = cf.CelestialFrame(
        reference_frame=coord.ICRS(),   # << was None; must be a real frame
        name='icrs',
        unit=(u.deg, u.deg),
        axes_names=('lon', 'lat'),
        axes_order=(0, 1),
    )

    # output grid size
    if output_size is None:
        base = int(np.ceil(max(H, W)*oversample))
        # prefer odd to put a clean central pixel
        base = base if base%2==1 else base+1
        outH = outW = base
    elif isinstance(output_size, int):
        outH = outW = output_size
    else:
        outH, outW = output_size

    # choose the output CRPIX at the geometric center (0-index image coords -> +0.5 to land on pixel center)
    # out_cx = (outW-1)/2
    # out_cy = (outH-1)/2
    out_cx = 0
    out_cy = 0

    # native detector deg/pix and oversampled output deg/pix
    det_deg_per_pix = (pixel_scale_arcsec/3600.0)         # deg/pix
    out_deg_per_pix = det_deg_per_pix/float(oversample)   # deg/pix (finer)

    # shifts to move CRPIX to origin (this mirrors your example: Shift(offset=-CRPIX))
    out_pixel_center = models.Shift(offset=-out_cx, name='crpix1_out') & \
                       models.Shift(offset=-out_cy, name='crpix2_out')

    out_pc = models.AffineTransformation2D(
        matrix=[[1.,0.],[0.,1.]], translation=[0.,0.], name='pc_rotation_out'
    )

    out_pixel_scale = models.Scale(factor=out_deg_per_pix, name='cdelt1_out') & \
                      models.Scale(factor=out_deg_per_pix, name='cdelt2_out')  # deg/pix

    out_tan = models.Pix2Sky_Gnomonic()
    out_rot = models.RotateNative2Celestial(lon=0., lat=0., lon_pole=180.)

    out_forward = out_pixel_center | out_pc | out_pixel_scale | out_tan | out_rot
    output_wcs = gwcs_wcs.WCS(
        forward_transform=out_forward,
        input_frame=detector_frame,
        output_frame=sky_frame
    )

    # ---- per-cutout *input* WCS objects (native detector scale, CRPIX at the centroid) ----
    # We put the cutout’s centroid at the tangent point (0,0) on the sky by setting CRPIX=(cx,cy) and using the
    # same Gnomonic + rotation chain. No absolute RA/Dec needed for PSF stacking.
    in_wcs_list = []
    for i in range(N):
        cx_i, cy_i = centroids[i]  # in cutout pixels
        in_pixel_center = models.Shift(offset=-cx_i, name=f'crpix1_in_{i}') & \
                          models.Shift(offset=-cy_i, name=f'crpix2_in_{i}')
        in_pc = models.AffineTransformation2D(
            matrix=[[1.,0.],[0.,1.]], translation=[0.,0.], name=f'pc_rotation_in_{i}'
        )
        in_pixel_scale = models.Scale(factor=det_deg_per_pix, name=f'cdelt1_in_{i}') & \
                         models.Scale(factor=det_deg_per_pix, name=f'cdelt2_in_{i}')
        in_tan = models.Pix2Sky_Gnomonic()
        in_rot = models.RotateNative2Celestial(lon=0., lat=0., lon_pole=180.)
        in_forward = in_pixel_center | in_pc | in_pixel_scale | in_tan | in_rot
        in_wcs_list.append(
            gwcs_wcs.WCS(
                forward_transform=in_forward,
                input_frame=detector_frame,
                output_frame=sky_frame
            )
        )

    # ---- reproject each cutout to the output grid ----
    # Note: reproject_interp uses bilinear by default, which matches your example.
    footprint_stack = []
    repro_stack = []
    shape_out = (outH, outW)
    for i in range(N):
        img = arr[i]
        # ensure 2D
        if img.ndim != 2:
            raise ValueError('Each cutout must be 2D.')
        re_i, fp_i = reproject_interp(
            input_data=(img, in_wcs_list[i]),
            output_projection=output_wcs,
            shape_out=shape_out,
        )
        repro_stack.append(re_i)
        footprint_stack.append(fp_i)

    repro_stack = np.asarray(repro_stack)           # (N, outH, outW)
    footprint_stack = np.asarray(footprint_stack)   # (N, outH, outW)

    # ---- weighted average with valid-footprint masking ----
    # drizzle-like combine: sum(w * I) / sum(w), ignoring NaNs and regions outside footprint
    w = wts[:, None, None] * (footprint_stack > 0)
    w = np.where(np.isfinite(repro_stack), w, 0.0)  # drop NaNs from contributing
    num = np.nansum(w * repro_stack, axis=0)
    den = np.sum(w, axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        reprojected_combined = num/den
    # If you *really* want to mimic your example's final "/4", do it explicitly:
    # reprojected_combined = (num/den)/4.0

    return reprojected_combined, list(footprint_stack), output_wcs

import numpy as np
import astropy.units as u
from astropy.wcs import WCS

def _make_tan_wcs(nx, ny, crpix_x, crpix_y, cdelt_deg):
    """
    构造一个以 (0,0) 为天球切点的本地 TAN WCS：
    - 负的 CD1_1（RA 向左增加）；
    - 正的 CD2_2（Dec 向上增加）；
    - CRPIX 使用 FITS 1-基坐标（因此 +1）。
    """
    w = WCS(naxis=2)
    w.wcs.crval = [0.0, 0.0]                         # 切点经纬
    w.wcs.crpix = [crpix_x + 1.0, crpix_y + 1.0]     # 1-based
    w.wcs.cd = np.array([[-cdelt_deg, 0.0],
                         [ 0.0,       cdelt_deg]], float)
    w.wcs.ctype = ['RA---TAN','DEC--TAN']
    # 可选但推荐：告知 array_shape（drizzle 会用到）
    w.array_shape = (ny, nx)
    return w

# drizzle

import numpy as np
from astropy.wcs import WCS


# 构造一个以 (0,0) 为切点的本地 TAN WCS；FITS 1-基 CRPIX；CD1_1<0（RA 向左）.
def _make_tan_wcs(nx, ny, crpix_x0, crpix_y0, cdelt_deg):
    w = WCS(naxis=2)
    w.wcs.crval = [0.0, 0.0]
    w.wcs.crpix = [crpix_x0 + 1.0, crpix_y0 + 1.0]  # 1-based
    w.wcs.cd = np.array([[-cdelt_deg, 0.0],
                         [ 0.0,       cdelt_deg]], float)
    w.wcs.ctype = ['RA---TAN','DEC--TAN']
    w.array_shape = (ny, nx)
    return w

def combine_image_drizzle(
    normalized_psf_tensor,
    centroids_d_tensor,
    wts_d_tensor,
    oversample=2,
    pixel_scale_arcsec=0.031,
    output_size=None,
    pixfrac=1.0,
    kernel='square',
):
    """
    用 STScI drizzle 合成（flux-conserving）并对齐到共同 TAN WCS 网格。
    - normalized_psf_tensor: (N,H,W)[,1]，2D cutouts。
    - centroids_d_tensor: (N,2) 质心 (x,y)，NumPy 0-基像素坐标。
    - wts_d_tensor: (N,) 每幅权重（将作为 weight_map 的尺度因子）。
    - oversample: 输出过采样倍数；输出像素尺度 = 原生/oversample。
    - pixel_scale_arcsec: 原生像素尺度（arcsec/pix）。
    - pixfrac/kernel: drizzle 核参数。
    返回：outsci, outwht, outwcs
    """
    from drizzle.resample import Drizzle
    from drizzle.utils import calc_pixmap  # 根据输入/输出 WCS 计算像素映射（pixmap）.  [oai_citation:1‡Space Telescope Drizzle](https://spacetelescope-drizzle.readthedocs.io/en/latest/drizzle/api.html)

    arr = np.asarray(normalized_psf_tensor)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError('normalized_psf_tensor 必须是 (N,H,W) 或 (N,H,W,1).')
    N, H, W = arr.shape

    cents = np.asarray(centroids_d_tensor, float)
    if cents.shape != (N, 2):
        raise ValueError('centroids_d_tensor 形状必须是 (N,2).')
    wts = np.asarray(wts_d_tensor, float).reshape(-1)
    if wts.shape != (N,):
        raise ValueError('wts_d_tensor 形状必须是 (N,).')

    # 输出网格大小
    if output_size is None:
        base = int(np.ceil(max(H, W)*oversample))
        base = base if base%2==1 else base+1   # 用奇数边长便于置中
        outH = outW = base
    elif isinstance(output_size, int):
        outH = outW = output_size
    else:
        outH, outW = output_size

    out_cx = 0#(outW-1)/2.0
    out_cy = 0#(outH-1)/2.0

    det_deg_per_pix = pixel_scale_arcsec/3600.0
    out_deg_per_pix = det_deg_per_pix/float(oversample)

    # 输出 TAN WCS
    outwcs = _make_tan_wcs(outW, outH, out_cx, out_cy, out_deg_per_pix)

    # 初始化 Drizzle 累加器（可传入 out_shape；pixfrac/kernel 可在 add_image 里或这里控制）
    d = Drizzle(kernel=kernel, out_shape=(outH, outW))

    for i in range(N):
        img = arr[i]
        if img.ndim != 2:
            raise ValueError('每个 cutout 必须是 2D.')
        cx_i, cy_i = cents[i]

        # 输入 TAN WCS：CRPIX=各自质心（0-基转 1-基在构造函数里做）
        inwcs = _make_tan_wcs(W, H, cx_i, cy_i, det_deg_per_pix)

        # 计算像素映射：把输入像素映到输出坐标系的 (x,y)（形状 (H,W,2)）
        pixmap = calc_pixmap(inwcs, outwcs)

        # 像素权重图：常数权重×坏像素掩膜（如有的话可相乘）
        wmap = np.ones_like(img, dtype=np.float32) * float(wts[i])

        # 真正的 drizzle：按照 pixmap 做面积再分配；默认单位 cps，exptime=1 保持数值
        d.add_image(
            data=img.astype(np.float32),
            exptime=1.0,
            pixmap=pixmap,
            weight_map=wmap,
            pixfrac=pixfrac,
            in_units='cps'
        )

    # 取输出（science/weight）—— Drizzle 的主输出在属性里
    outsci = d.out_img.copy()
    outwht = d.out_wht.copy()
    return outsci, outwht, outwcs