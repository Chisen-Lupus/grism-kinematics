# Grism calibration functions 

import os
import numpy as np
from astropy.io import ascii, fits
from scipy import interpolate

def get_grism_cal_dir():
    grism_cal_dir = os.getenv('GRISM_CAL_DIR')
    # If not set, use default
    if grism_cal_dir is None:
        grism_cal_dir = '/data/grism_cal'
    if not os.path.isdir(grism_cal_dir):
        raise FileNotFoundError(f'Grism calibration directory not found: {grism_cal_dir}')
    return grism_cal_dir


#%% Copied from Fengwu's code

'''This is the Grism Dispersion function, will be detailed later'''
def fit_disp_order32(data, 
                     a01, a02, a03, a04, a05, a06, 
                     b01, b02, b03, b04, b05, b06, 
                     c01, c02, c03, #c04, c05, c06,
                     d01, #d02, d03, d04, d05, d06,
                    ):
    ## data is an numpy array of the shape (3, N)
    ##     - data[0]:  x_pixel      --> fit with second-degree polynomial
    ##     - data[1]:  y_pixel      --> fit with second-degree polynomial
    ##     - data[2]:  wavelength   --> fit with third-degree polynomial
    xpix, ypix, dx = data[0] - 1024, data[1] - 1024, data[2] - 3.95
    ## return dx = dx(x_pixel, y_pixel, lambda)
    return ((a01 + (a02 * xpix + a03 * ypix) + (a04 * xpix**2 + a05 * xpix * ypix + a06 * ypix**2)
            ) + 
            (b01 + (b02 * xpix + b03 * ypix) + (b04 * xpix**2 + b05 * xpix * ypix + b06 * ypix**2)
            ) * dx +
            (c01 + (c02 * xpix + c03 * ypix) #+ (c04 * xpix**2 + c05 * xpix * ypix + c06 * ypix**2)
            ) * dx**2 + 
            (d01 #+ (d02 * xpix + d03 * ypix) + (d04 * xpix**2 + d05 * xpix * ypix + d06 * ypix**2)
            ) * dx**3
           ) 

func_fit_wave = fit_disp_order32

'''This is the Spectral Tracing function, will be detailed later'''
def fit_disp_order23(data, 
                     a01, a02, a03, a04, a05, a06, a07, a08, a09, a10,
                     b01, b02, b03, b04, b05, b06, b07, b08, b09, b10,
                     c01, c02, c03, c04, c05, c06, c07, c08, c09, c10,
                    ):
    ## data is an numpy array of the shape (3, N)
    ##     - data[0]:  x_pixel  --> fit with second-degree polynomial
    ##     - data[1]:  y_pixel  --> fit with second-degree polynomial
    ##     - data[2]:  dx     --> fit with second-degree polynomial
    xpix, ypix, dx = data[0] - 1024, data[1] - 1024, data[2]
    ## return dy = dy(x_pixel, y_pixel, d_x)
    return ((a01 + (a02 * xpix + a03 * ypix) + (a04 * xpix**2 + a05 * xpix * ypix + a06 * ypix**2)
             + (a07 * xpix**3 + a08 * xpix**2 * ypix + a09 * xpix * ypix**2 + a10 * ypix**3)
            ) + 
            (b01 + (b02 * xpix + b03 * ypix) + (b04 * xpix**2 + b05 * xpix * ypix + b06 * ypix**2)
             + (b07 * xpix**3 + b08 * xpix**2 * ypix + b09 * xpix * ypix**2 + b10 * ypix**3)
            ) * dx +
            (c01 + (c02 * xpix + c03 * ypix) + (c04 * xpix**2 + c05 * xpix * ypix + c06 * ypix**2)
             + (c07 * xpix**3 + c08 * xpix**2 * ypix + c09 * xpix * ypix**2 + c10 * ypix**3)
            ) * dx**2
           ) 


def grism_conf_preparation(x0 = 1024, y0 = 1024, pupil = 'R', 
                           fit_opt_fit = np.zeros(30), w_opt = np.zeros(16)):
    '''
    Prepare grism configuration, dxs, dys, wavelengths based on input (x0, y0) pixel postion
        and filter/pupil/module information.
    -----------------------------------------------
        Parameters
        ----------  
        x0, y0 : float
            Reference position (i.e., in direct image)
        
        pupil: 'R' or 'C'
            pupil of grism ('R' or 'C')
        
        fit_opt_fit: numpy.ndarray, shape: (30,)
            polynomial parameters in the perpendicular direction, used by function `fit_disp_order23` 
        
        w_opt: numpy.ndarray, shape: (16,)
            polynomial parameters in the dispersed direction, used by function `fit_disp_order32` 
        
        Returns
        -------
        
        dxs, dys : `~numpy.ndarray`
            offset of spectral pixel from the direct imaging position
        
        wavs: `~numpy.ndarray`
            Array of wavelengths corresponding to dxs and dys
    '''
    # Load the Grism Configuration file
    # GConf = grismconf.Config(os.environ['MIRAGE_DATA'] + "/nircam/GRISM_NIRCAM/V3/" +
    #                          "NIRCAM_%s_mod%s_%s.conf" % (filter, module, pupil))
    wave_space = np.arange(2.39, 5.15, 0.01)
    disp_space = func_fit_wave(np.vstack((x0 * np.ones_like(wave_space), y0 * np.ones_like(wave_space), wave_space)), *w_opt)
    inverse_wave_disp = interpolate.UnivariateSpline(disp_space[np.argsort(disp_space)], wave_space[np.argsort(disp_space)], s = 0, k = 1)

    if pupil == 'R':
        dxs = np.arange(int(np.min(disp_space)), int(np.max(disp_space))) - x0%1
        # Compute wavelength of each of the pixels
        wavs = inverse_wave_disp(dxs)
        # Compute the dys values for the same pixels
        dys = fit_disp_order23(np.vstack((x0* np.ones_like(dxs), y0 * np.ones_like(dxs), dxs)), *fit_opt_fit)
        # dxs = np.arange(-1800, 1800, 1) - x0%1
        ## Compute the t values corresponding to the exact offsets (with old grism config)
        # ts = GConf.INVDISPX(order = '+1', x0 = x0, y0 = y0, dx = dxs)
        # dys = GConf.DISPY('+1', x0, y0, ts)
        # wavs = GConf.DISPL('+1', x0, y0, ts)
        # tmp_aper = np.max([0.2, 1.5 * tmp_re_maj * np.cos(tmp_pa), 1.5 * tmp_re_min * np.sin(tmp_pa)])
    elif pupil == 'C':
        # Compute the dys values for the same pixels
        dys = np.arange(int(np.min(disp_space)), int(np.max(disp_space))) - y0%1
        # Compute wavelength of each of the pixels
        wavs = inverse_wave_disp(dys)
        dxs = fit_disp_order23(np.vstack((x0* np.ones_like(dys), y0 * np.ones_like(dys), dys)), *fit_opt_fit)
        # dys = np.arange(-1800, 1800, 1) - y0%1
        ## Compute the t values corresponding to the exact offsets (with old grism config)
        # ts = GConf.INVDISPY(order = '+1', x0 = x0, y0 = y0, dy = dys)
        # dxs = GConf.DISPX('+1', x0, y0, ts)
        # wavs = GConf.DISPL('+1', x0, y0, ts)
    return (dxs, dys, wavs)

#%% Modified from Xiaojing's or Fengwu's code

def load_nircam_wfss_model(pupil, module, filter):

    tmp_pupil = pupil
    tmp_module = module
    tmp_filter = filter

    ### Interested wavelength Range
    if tmp_filter == 'F444W': WRANGE = np.array([3.8, 5.1])
    elif tmp_filter == 'F322W2': WRANGE = np.array([2.4, 4.1])
    elif tmp_filter == 'F356W':  WRANGE = np.array([3.05, 4.05])
    # elif tmp_filter == 'F277W':  WRANGE = np.array([2.4, 3.1])

    ### Spectral tracing parameters:
    if tmp_filter in ['F277W', 'F335M', 'F322W2', 'F356W', 'F360M']: disp_filter = 'F322W2'
    elif tmp_filter in ['F410M', 'F444W', 'F480M']: disp_filter = 'F444W' 

    # dir_wavecal = '/home/u24/fengwusun/Comissioning/FS_grism_config_v3_202406/'
    tb_order23_fit_AR = ascii.read(os.path.join(get_grism_cal_dir(), 'dy_dx_tracing_model/DISP_%s_mod%s_grism%s.dat' % (disp_filter, 'A', 'R')))
    fit_opt_fit_AR, fit_err_fit_AR = tb_order23_fit_AR['col0'].data, tb_order23_fit_AR['col1'].data
    tb_order23_fit_BR = ascii.read(os.path.join(get_grism_cal_dir(), 'dy_dx_tracing_model/DISP_%s_mod%s_grism%s.dat' % (disp_filter, 'B', 'R')))
    fit_opt_fit_BR, fit_err_fit_BR = tb_order23_fit_BR['col0'].data, tb_order23_fit_BR['col1'].data
    tb_order23_fit_AC = ascii.read(os.path.join(get_grism_cal_dir(), 'dy_dx_tracing_model/DISP_%s_mod%s_grism%s.dat' % (disp_filter, 'A', 'C')))
    fit_opt_fit_AC, fit_err_fit_AC = tb_order23_fit_AC['col0'].data, tb_order23_fit_AC['col1'].data
    tb_order23_fit_BC = ascii.read(os.path.join(get_grism_cal_dir(), 'dy_dx_tracing_model/DISP_%s_mod%s_grism%s.dat' % (disp_filter, 'B', 'C')))
    fit_opt_fit_BC, fit_err_fit_BC = tb_order23_fit_BC['col0'].data, tb_order23_fit_BC['col1'].data

    ### grism dispersion parameters:
    tb_fit_displ_AR = ascii.read(os.path.join(get_grism_cal_dir(), 'dx_wave_dispersion_model/DISPL_mod%s_grism%s.dat' % ('A', "R")))
    w_opt_AR, w_err_AR = tb_fit_displ_AR['col0'].data, tb_fit_displ_AR['col1'].data
    tb_fit_displ_BR = ascii.read(os.path.join(get_grism_cal_dir(), 'dx_wave_dispersion_model/DISPL_mod%s_grism%s.dat' % ('B', "R")))
    w_opt_BR, w_err_BR = tb_fit_displ_BR['col0'].data, tb_fit_displ_BR['col1'].data
    tb_fit_displ_AC = ascii.read(os.path.join(get_grism_cal_dir(), 'dx_wave_dispersion_model/DISPL_mod%s_grism%s.dat' % ('A', "C")))
    w_opt_AC, w_err_AC = tb_fit_displ_AC['col0'].data, tb_fit_displ_AC['col1'].data
    tb_fit_displ_BC = ascii.read(os.path.join(get_grism_cal_dir(), 'dx_wave_dispersion_model/DISPL_mod%s_grism%s.dat' % ('B', "C")))
    w_opt_BC, w_err_BC = tb_fit_displ_BC['col0'].data, tb_fit_displ_BC['col1'].data

    ### list of module/pupil and corresponding tracing/dispersion function:
    list_mod_pupil   = np.array(['AR', 'BR', 'AC', 'BC'])
    list_fit_opt_fit = np.array([fit_opt_fit_AR, fit_opt_fit_BR, fit_opt_fit_AC, fit_opt_fit_BC])
    list_w_opt       = np.array([w_opt_AR, w_opt_BR, w_opt_AC, w_opt_BC])

    ### Sensitivity curve:
    dir_fluxcal = os.path.join(get_grism_cal_dir(), 'sensitivity_model/')
    # dir_fluxcal = '/home/u24/fengwusun/Comissioning/cycle1_cal_program/fluxcal/product/'
    tb_sens_AR = ascii.read(dir_fluxcal + '%s_mod%s_grism%s_sensitivity.dat' % (tmp_filter, 'A', 'R'))
    tb_sens_BR = ascii.read(dir_fluxcal + '%s_mod%s_grism%s_sensitivity.dat'% (tmp_filter, 'B', 'R'))
    tb_sens_AC = ascii.read(dir_fluxcal + '%s_mod%s_grism%s_sensitivity.dat' % (tmp_filter, 'A', 'C'))
    tb_sens_BC = ascii.read(dir_fluxcal + '%s_mod%s_grism%s_sensitivity.dat'% (tmp_filter, 'B', 'C'))
    f_sens_AR = interpolate.UnivariateSpline(tb_sens_AR['wavelength'], tb_sens_AR['DN/s/Jy'], ext = 'zeros', k = 1, s = 1e2)
    f_sens_BR = interpolate.UnivariateSpline(tb_sens_BR['wavelength'], tb_sens_BR['DN/s/Jy'], ext = 'zeros', k = 1, s = 1e2)
    f_sens_AC = interpolate.UnivariateSpline(tb_sens_AC['wavelength'], tb_sens_AC['DN/s/Jy'], ext = 'zeros', k = 1, s = 1e2)
    f_sens_BC = interpolate.UnivariateSpline(tb_sens_BC['wavelength'], tb_sens_BC['DN/s/Jy'], ext = 'zeros', k = 1, s = 1e2)
    list_f_sens       = (f_sens_AR, f_sens_BR, f_sens_AC, f_sens_BC)

    
    this_spatial_model = list_fit_opt_fit[list_mod_pupil == tmp_module + tmp_pupil][0]
    this_disp_model = list_w_opt[list_mod_pupil == tmp_module + tmp_pupil][0]
    # f_sens = list_f_sens[list_mod_pupil == tmp_module + tmp_pupil][0]

    return WRANGE, this_spatial_model, this_disp_model


def find_pixel_location(WRANGE, this_spatial_model, this_disp_model, this_pupil,
                         pixelx, pixely, line_wavelengths):

    dxs, dys, wavs = grism_conf_preparation(
        x0=pixelx, y0=pixely,
        pupil=this_pupil,
        fit_opt_fit=this_spatial_model,
        w_opt=this_disp_model,
        # module=this_module,
        # filter=this_filter
    )
    order_ = np.argsort(wavs)
    func_dx_wav = interpolate.UnivariateSpline(wavs[order_], dxs[order_], k = 1, s = 0)
    func_dy_wav = interpolate.UnivariateSpline(wavs[order_], dys[order_], k = 1, s = 0)
    # position on the Grism image
    x_on_G_img = dxs + pixelx
    y_on_G_img = dys + pixely
    flg_in_frame = (x_on_G_img > 0) & (x_on_G_img < 2048) & (y_on_G_img > 0) & (y_on_G_img < 2048) & (wavs > WRANGE[0]) & (wavs < WRANGE[1])
    x_on_G_img = x_on_G_img[flg_in_frame]
    y_on_G_img = y_on_G_img[flg_in_frame]

    # position of the targeted wavelength
    xline_on_G_img = func_dx_wav(line_wavelengths)
    yline_on_G_img = func_dy_wav(line_wavelengths)
    xline_on_G_img += pixelx
    yline_on_G_img += pixely

    return xline_on_G_img, yline_on_G_img


