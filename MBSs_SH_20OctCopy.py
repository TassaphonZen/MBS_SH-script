#version 2025.10.20 15:11

import csv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.special import erf
from scipy.optimize import curve_fit, least_squares
from scipy.ndimage import gaussian_filter, gaussian_filter1d, convolve, generic_filter
from scipy import ndimage
from datetime import datetime
from natsort import natsorted
from tqdm.notebook import tqdm

k_B = 8.617333262145 * 10 ** -5  # eV/K

class Spectrum:
    def __init__(self, filename, dpcor = False, spin = False, test = False, f = 'c'):
        self.filename = self._get_filename(filename, f)
        self.header = self.filename.split(".txt")[0]

        metadata, data = self._read_txt_file()
        self.info = self._read_info_file()
        self.i0 = float(self.info.get('I0 (HRFM)', '1').split()[0])

        self.metadata = self._convert_metadata(metadata)
        if not test:
            start = self.metadata['Start K.E.']
            stop  = self.metadata['End K.E.'] - self.metadata['Step Size']
            steps = self.metadata['No. Steps']
        else:
            center = self.metadata['Center K.E.']
            step   = self.metadata['Step Size']
            n      = self.metadata['TotSteps']
            start  = center - (n - 1) / 2 * step
            stop   = center + (n - 1) / 2 * step
            steps  = n
        self.energy_scale = np.linspace(start, stop, steps)
        self.energy_scale_spin = np.linspace(self.metadata['FirstEnergy'],self.metadata['Center K.E.']*2 - self.metadata['FirstEnergy'],self.metadata['TotSteps']) + self.metadata['SpinOffs']
        self.lens_scale = np.linspace(self.metadata['ScaleMin'], self.metadata['ScaleMax'], self.metadata['NoS'])
        self.deflector_scale = np.linspace(self.metadata.get('MapStartX'), self.metadata.get('MapEndX'), self.metadata.get('MapNoXSteps')) if self.metadata.get('MapStartX') is not None else 'Not deflector mapping mode'
        self.time = self._calculate_time()

        # self.data = self._parse_data(data) if dpcor is False else self._dp_pcnt_dither(self._parse_data(data))
        self.data = self._dp_pcnt_dither(self._parse_data(data, spin)) if dpcor is True and self.metadata['AcqMode'] == 'Dither' else self._parse_data(data, spin)
        self.data_spin = self._read_spin_txt_file()
        
        self.normdata = self.data / self._cor_actscans() / self.metadata['Frames Per Step'] / self.i0
        
        if self.normdata.ndim >= 2:
            self.edc = np.sum(self.normdata, axis=1)
            self.mdc = np.sum(self.normdata, axis=0)
        else:
            self.edc = np.array(self.normdata)
            self.mdc = np.array(self.normdata)

        self.edc_spin =self.data_spin / self._cor_actscans() / self.metadata['Frames Per Step'] / self.i0
        
    def _get_filename(self, filename, f):
        if 'MBS' in str(filename):
            return filename + ".txt"
        elif '..\\SMOpt' in str(filename):
            return filename + ".txt"
        else:
            if f == 'p':
                filepattern = "../*" + f'{filename:0>{5}}' + "*.txt"
            else:
                filepattern = "*" + f'{filename:0>{5}}' + "*.txt"
        return natsorted(glob.glob(filepattern))[0]

    def _read_txt_file(self):
        with open(self.header + ".txt") as txt_file:
            reader = csv.reader(txt_file, delimiter = '\t')
            metadata = {}
            for row in reader:
                if 'DATA:' in row:
                    break
                metadata[row[0]] = row[1]
            data = np.loadtxt(txt_file, dtype='int')  ## need to change to float if the file is not correctly converted
        return metadata, data

    def _read_spin_txt_file(self):
        filename = self.header + "S.txt"
        
        if not os.path.exists(filename):
            return 0 
        with open(filename) as txt_file:
            reader = csv.reader(txt_file, delimiter = '\t')
            for row in reader:
                if 'DATA:' in row:
                    break
            data = np.loadtxt(txt_file, dtype='int')
        return data

    def _read_info_file(self):
        info = {}
        try:
            with open(self.header + ".info") as info_file:
                reader = csv.reader(info_file)
                for line in reader:
                    key, value = line[0].strip().split(":", maxsplit=1)
                    info[key.strip()] = self._convert_value(value.strip())
        except FileNotFoundError:
            return {"Info file": None}
        return info

    def _convert_value(self, value):
        try:
            if 'E' in value:
                return float(value.split('E')[0]) * 10**float(value.split('E')[1])
            return float(value)
        except ValueError:
            return value

    def _convert_metadata(self, str_dict):
        exclude_keys = ['Gen. Name', 'NameString']
        def convert_value(key, value):
            if key in exclude_keys:
                return value
            if value.isdigit():
                return int(value)
            try:
                return float(value)
            except ValueError:
                return value
        return {key: convert_value(key, value) for key, value in str_dict.items()}

    def _cor_actscans(self):
        if self.metadata['ActScans'] == 0:
            self.metadata['ActScans'] = 1
        return self.metadata['ActScans'] #Correct the scan number when using external mapping mode
        
    def _parse_data(self, data, spin):
        if spin == False:
            if data.shape[1] != self.lens_scale.size:
                data = data[:,1:]
                # print(f'{self.filename} contains kinetic energy in front of 2D data, and the first column is dropped for correction.')
                if data.shape[1] != self.lens_scale.size:
                    print(f'In {self.filename}, something wrong with the data dimension!')
        return data
        
    def _calculate_time(self):
        for fmt in ('%d/%m/%Y %H:%M', '%m/%d/%Y %I:%M %p'):
            try:
                Tstart = datetime.strptime(self.metadata['STim'], fmt)
                Tend = datetime.strptime(self.metadata['TIMESTAMP:'], fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError("Unknown time format in metadata")
    
        return str(Tend - Tstart)

    def _dp_pcnt_dither(self, data, dithsteps = 36):
        width = 5
        n = dithsteps  # dither steps
        vn = 1.0
        z = (width-1)*n + 4*width
        vz = -vn*n/z
        kernel = np.array([[vz]*width]*2 + [[vz, vz, vn, vz, vz]]*n + [[vz]*width]*2)
    
        test = convolve(data, kernel)
        threshold = 0.01 * test.max()
    
        def det(buffer):
            if buffer[18] > threshold and buffer.argmax() == 18:
                return buffer[18]
            else:
                return 0
    
        testmax = generic_filter(test, det, size=(n, 1), mode='constant', cval=0)
    
        dp = np.argwhere(testmax != 0)
        dp_vals = testmax[testmax != 0]
        dp_t = sorted(zip(dp_vals, dp), key=lambda x: x[0], reverse=True)
    
        fixed2 = np.array(data, dtype=float)
    
        remove_threshold = 0.25*np.percentile(data, 50)
        n_remove = (dp_vals/n > remove_threshold).sum()
        #print('to remove', n_remove, 'treshold', remove_threshold)
        #n_remove = 60
    
        for dp_i, (val, (i, j)) in enumerate(dp_t):
            i = i+1
    
            #print(i, j, ":", val, val/n)
            val = val/n
            start, end = i-n//2, i+n//2
            start = max(start, 0)
    
            #fixedmin = np.percentile(side, 50)
            #print(fixed2.dtype, val.dtype, fixedmin.dtype)
            fixed2[start:end, j] = fixed2[start:end, j] - val
    
            if dp_i > n_remove:
                break
    
        fixed2[fixed2 < 0] = 0
    
        for dp_i, (val, (i, j)) in enumerate(dp_t):
            i = i+1
            val = val/n
            start, end = i-n//2, i+n//2
            start = max(start, 0)
    
            fixed2[start:end, j] = fixed2[start:end, j-2:j+2].mean(axis=1)
            if dp_i > n_remove:
                break
    
        return fixed2

    def __repr__(self):
        return f'{self.header}'

    def l_to_i(self,l):
        return (np.abs(self.lens_scale - l)).argmin()

    def k_to_i(self,k):
        return (np.abs(self.k_scale[0] - k)).argmin()
    
    def e_to_i(self,e):
        return (np.abs(self.energy_scale - e)).argmin()

    def find_fl(self, find_E1 = None, find_E2 = None):
        if find_E1 is None or find_E2 is None:
            find_E1 = self.energy_scale[-1] - self.energy_scale[0]
            find_E2 = self.energy_scale[-1] - self.energy_scale[0]
            deltaE = np.diff(self.energy_scale)[0]
            top = 1
            bottom = round(find_E2 / deltaE)
        else:
            deltaE = np.diff(self.energy_scale)[0]
            top = round(find_E1 / deltaE)
            bottom = round(find_E2 / deltaE)
        grad = np.gradient(self.edc[-bottom:-top]/np.max(self.edc[-bottom:-top]), deltaE)
        grad = gaussian_filter1d(grad, sigma = 0.1 / deltaE)
        fl = (self.energy_scale[-bottom:])[np.argmin(grad)]
        return fl

    def fermi_fit_plot(self, T = 40, de = 1.0):
        fl = self.find_fl()
        kinetic_energy = self.energy_scale
        intensity_raw = self.edc
        
        window = (fl - de / 2, fl + de / 2)
        mask = (kinetic_energy >= window[0]) & (kinetic_energy <= window[1])
        E_fit = kinetic_energy[mask]
        I_fit = intensity_raw[mask]
        
        # === Fermi edge * linear DOS ===
        def fermi_edge_linearDOS(E, EF, A, FWHM, alpha, constant):
            sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
            gauss_arg = (E - EF) / (np.sqrt(2) * sigma)
        
            # Linear DOS centered at EF: DOS(EF) = 1
            dos = 1 + alpha * (E - EF)
        
            fd = 0.5 * (1 - erf(gauss_arg + (k_B * T) / (np.sqrt(2) * sigma)))
            return A * dos * fd + constant
    
        # === Fit the Fermi Edge ===
        # === Fit initial guess and bounds ===
        # Parameters: [EF, A, FWHM, alpha, constant]
        p0 = [np.median(E_fit), max(I_fit), 0.05, 0.0, 0.0]
        bounds = ([E_fit[0], 0, 0.001, -10, 0.0], [E_fit[-1], np.inf, 0.5, 10, np.inf])
        
        # Fit
        popt, pcov = curve_fit(fermi_edge_linearDOS, E_fit, I_fit, p0=p0, bounds=bounds)
    
        plt.plot(E_fit, I_fit, 'ko', markersize=4, label='Raw data')
        plt.plot(E_fit, fermi_edge_linearDOS(E_fit, *popt), 'r-', linewidth=1, label='Fitted Fermi edge* DOS')
        plt.xlabel('Kinetic Energy (eV)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
        plt.show()
        labels = ['EF (eV)', 'Amplitude', 'FWHM (eV)', 'DOS slope (1/eV)', 'Constant']
        print(f"Fermi-Dirac Fit with Linear DOS (T = {T} K):")
        for name, val in zip(labels, popt):
            print(f"{name}: {val:.4f}")
    
    def arpes_to_k(self, lens, Ek):
        return np.sqrt(Ek) * 0.5124 * np.sin(np.radians(lens))
        
    def shirley_BG_cor(self,iteration=10):
        I_loop = self.edc[::-1] - self.edc[-1]
        for i in range(0,iteration):
            sum_array = []
            sum_element = 0
            for j, I_i in enumerate((I_loop)):
                sum_element = sum_element + I_i
                sum_array += [sum_element]
            
            BG = I_loop[-1]  * np.array(sum_array) / sum_element  ## See the effect of iteration, use np.average(I_loop[-50:-1]). Simplily using I_loop[-1] is good.
            I_loop = I_loop - BG
        self.edc_s_BG_cor = I_loop[::-1]
        plt.plot(self.energy_scale, self.edc - self.edc[-1], label='original data')
        plt.plot(self.energy_scale, I_loop[::-1], label='processed data')
        plt.plot(self.energy_scale, self.edc - self.edc[-1] - I_loop[::-1], label='Shirley background')
        plt.legend()
        plt.xlabel('Kinetic energy (eV)')
        plt.ylabel('Intensity (a.u.)')
    
    def edc_plot(self, fl = 0, scaling = 1, zero = False, **kwargs):
        default_kwargs = {
            'label': self.filename
        }
        default_kwargs.update(kwargs)
        
        if fl == 0:
            x = self.energy_scale
            plt.xlabel('Kinetic energy (eV)')
        else:
            x = self.energy_scale - float(fl)
            plt.xlabel('Binding energy (eV)')
        if zero == 'false':
            plt.plot(x, self.edc * scaling, **default_kwargs)
        else:
            plt.plot(x, (self.edc - np.min(self.edc)) * scaling, **default_kwargs)
        plt.ylabel('Intensity (a.u.)')
        
    def rawdata_plot(self, fl=0, sigma=0, v_min=5, v_max=99.5, k_space=True, k_origin=0, slit_rescale=1/1.1, **kwargs):
        X = self.lens_scale * slit_rescale
        Y = self.energy_scale
        X, Y = np.meshgrid(X, Y)
        Z = self.normdata
        
        if k_space:
            X = self.arpes_to_k(X, Y) - k_origin
    
        default_kwargs = {
            'cmap': 'turbo',
            'vmin': np.percentile(Z, v_min),
            'vmax': np.percentile(Z, v_max)
        }
        if k_space:
            default_kwargs['shading'] = 'gouraud'
        default_kwargs.update(kwargs)
    
        if fl == 0:
            plt.ylabel('Kinetic Energy (eV)')
        else:
            Y = Y - fl
            plt.ylabel('Binding Energy (eV)')
    
        plt.pcolormesh(X, Y, Z, **default_kwargs)
        plt.xlabel(r"$k_\parallel$ ($1/\AA$)" if k_space else 'Angle (degree)')
        
    def sigma_cor_plot(self, fl=0, sigma=0, v_min=5, v_max=99.5, k_space=True, k_origin=0, slit_rescale=1/1.1, **kwargs):
        X = self.lens_scale * slit_rescale
        Y = self.energy_scale
        X, Y = np.meshgrid(X, Y)
        MDCs = np.sum(self.normdata, axis=0)
        Z = self.normdata * gaussian_filter(MDCs, sigma=sigma) / (MDCs + 1)
        
        if k_space:
            X = self.arpes_to_k(X, Y) - k_origin

        default_kwargs = {
            'cmap': 'turbo',
            'vmin': np.percentile(Z, v_min),
            'vmax': np.percentile(Z, v_max)
        }
        if k_space:
            default_kwargs['shading'] = 'gouraud'
        default_kwargs.update(kwargs)
    
        if fl == 0:
            plt.ylabel('Kinetic Energy (eV)')
        else:
            Y = Y - fl
            plt.ylabel('Binding Energy (eV)')
    
        plt.pcolormesh(X, Y, Z, **default_kwargs)
        plt.xlabel(r"$k_\parallel$ ($1/\AA$)" if k_space else 'Angle (degree)')

    def mdc_cor_plot(self, fl=0, v_min=5, v_max=99.5, k_space=False, theta_origin=0, slit_rescale=1/1.1, **kwargs):
        X = self.lens_scale * slit_rescale - theta_origin
        Y = self.energy_scale
        X, Y = np.meshgrid(X, Y)
        MDCs = np.sum(self.normdata, axis=0)
        Z = self.normdata / (MDCs + 1)
    
        if k_space:
            X = self.arpes_to_k(X, Y)

        default_kwargs = {
            'cmap': 'turbo',
            'vmin': np.percentile(Z, v_min),
            'vmax': np.percentile(Z, v_max)
        }
        if k_space:
            default_kwargs['shading'] = 'gouraud'
            default_kwargs['rasterized'] = True
        default_kwargs.update(kwargs)

        if fl == 0:
            plt.ylabel('Kinetic Energy (eV)')
        else:
            Y = Y - fl
            plt.ylabel('Binding Energy (eV)')
    
        plt.pcolormesh(X, Y, Z, **default_kwargs)
        plt.xlabel(r"$k_\parallel$ ($1/\AA$)" if k_space else 'Angle (degree)')

    def seg_mdc_cor_plot(self, E1, E2, fl=0, v_min=5, v_max=99.5, k_space=True, k_origin=0, slit_rescale=1/1.1, **kwargs):
        X = self.lens_scale * slit_rescale
        Y = self.energy_scale
        X, Y = np.meshgrid(X, Y)
        MDCs = np.sum(self.normdata[self.e_to_i(E1):self.e_to_i(E2)], axis=0)
        Z = self.normdata / (MDCs + 1)
        
        if k_space:
            X = self.arpes_to_k(X, Y) - k_origin

        default_kwargs = {
            'cmap': 'turbo',
            'vmin': np.percentile(Z, v_min),
            'vmax': np.percentile(Z, v_max)
        }
        if k_space:
            default_kwargs['shading'] = 'gouraud'
            default_kwargs['rasterized'] = True
        default_kwargs.update(kwargs)
    
        if fl == 0:
            plt.ylabel('Kinetic Energy (eV)')
        else:
            Y = Y - fl
            plt.ylabel('Binding Energy (eV)')
    
        plt.pcolormesh(X, Y, Z, **default_kwargs)
        plt.xlabel(r"$k_\parallel$ ($1/\AA$)" if k_space else 'Angle (degree)')

    def BG_cor_plot(self, BGdata, fl=0, v_min=5, v_max=99.5, k_space=True, k_origin=0, slit_rescale=1/1.1, **kwargs):
        X = self.lens_scale * slit_rescale
        Y = self.energy_scale
        X, Y = np.meshgrid(X, Y)
        Z = self.normdata / (BGdata + 1)
        self.BG_cor_data = Z

        if k_space:
            X = self.arpes_to_k(X, Y) - k_origin
            self.k_scale = X
        
        default_kwargs = {
            'cmap': 'turbo',
            'vmin': np.percentile(Z, v_min),
            'vmax': np.percentile(Z, v_max)
        }
        if k_space:
            default_kwargs['shading'] = 'gouraud'
            default_kwargs['rasterized'] = True
        default_kwargs.update(kwargs)
    
        if fl == 0:
            plt.ylabel('Kinetic Energy (eV)')
        else:
            Y = Y - fl
            plt.ylabel('Binding Energy (eV)')
    
        plt.pcolormesh(X, Y, Z, **default_kwargs)
        plt.xlabel(r"$k_\parallel$ ($1/\AA$)" if k_space else 'Angle (degree)')
        
    def BG_cor_sym_plot_k(self, fl = 0, scaling = 1, v_min=5, v_max=99.5,**kwargs):
        # Double left array. Have to execute BG_cor_plot with k_space = True to get self.k_scale, which is used to put high symmetry point to k=0.
        axis = self.k_to_i(0)
        
        left_array = self.BG_cor_data[:,:axis]
        energy_d = np.shape(left_array)[0]
        
        left_array_d2 = np.shape(left_array)[1]
        
        sym_array = np.zeros((energy_d,left_array_d2*2))
        
        for i,j in enumerate(left_array):
            sym_array[i,:left_array_d2] += j
            sym_array[i,left_array_d2:] += j[::-1]
            
        
        left_k = self.k_scale[:,:axis]
        
        sym_k_scale = np.empty((energy_d,left_array_d2*2))
        for i,j in enumerate(left_k):
            sym_k_scale[i,np.shape(left_k)[1]:] = -j[::-1]
            sym_k_scale[i,:np.shape(left_k)[1]] = j
        
        sym_energy_scale = np.empty(np.shape(sym_k_scale))
        for i, element in enumerate(self.energy_scale):
            sym_energy_scale[i,:] = element
        self.k_scale_sym = sym_k_scale
        self.BG_cor_data_sym = sym_array
        default_kwargs = {
                'cmap': 'turbo',
                'vmin':np.percentile(sym_array, v_min),
                'vmax':np.percentile(sym_array, v_max),
                'rasterized': True,
                'shading': 'gouraud'
            }
        default_kwargs.update(kwargs)
        if fl == 0:
            plt.ylabel('Kinetic Energy (eV)')
        else:
            sym_energy_scale = sym_energy_scale - fl
            plt.ylabel('Binding Energy (eV)')
        plt.pcolormesh(self.k_scale_sym, sym_energy_scale, self.BG_cor_data_sym * scaling, **default_kwargs)
        plt.xlabel(r"$k_\parallel$ ($1/\AA$)")

    def edc_spin_plot(self, fl = 0, scaling = 1, zero = False, **kwargs):
        
        default_kwargs = {
            'label': self.filename
        }
        default_kwargs.update(kwargs)
        
        if fl == 0:
            x = self.energy_scale_spin
            plt.xlabel('Kinetic energy (eV)')
        else:
            x = float(fl) - self.energy_scale_spin
            plt.gca().invert_xaxis()
            plt.xlabel('Binding energy (eV)')
        if zero == 'false':
            plt.plot(x, self.edc_spin * scaling, **default_kwargs)
        else:
            plt.plot(x, (self.edc_spin - np.min(self.edc_spin)) * scaling, **default_kwargs)
        plt.ylabel('Intensity (a.u.)')

    def IV_curve_plot(self, SE = 0, step_size=0.2, **kwargs):
        
        default_kwargs = {
            'label': self.filename
        }
        default_kwargs.update(kwargs)
        steps = np.size(self.data)
        SEs = np.linspace(-(steps-1)/2*step_size,(steps-1)/2*step_size,steps)
        plt.plot(SEs+SE,self.normdata, **default_kwargs)
        plt.xlabel('Scattering Energy (eV)')
        plt.ylabel('Intensity (a.u.)')

class MapSpectrum:
    def __init__(self, filename, dpcor = False, spin = False, f = 'c'): 
        start_time = time.time()

        try:
            if f == 'p':
                filepattern_mbs = "../"+ str(filename) + "_*.txt"
                filepattern_sm = "../"+ str(filename) + "_*.txt"
                filepattern = "../*MBS-" + f'{filename:0>{5}}' + "*.txt"
                
            else:
                filepattern_mbs = str(filename) + "_*.txt"
                filepattern_sm = str(filename) + "_*.txt"
                filepattern = "*MBS-" + f'{filename:0>{5}}' + "*.txt"
            if 'MBS' in str(filename):
                self.filenames = natsorted(glob.glob(filepattern_mbs, recursive=True))
            elif 'SMOpt' in str(filename):
                self.filenames = natsorted(glob.glob(filepattern_sm, recursive=True))
            else:
                self.filenames = natsorted(glob.glob(filepattern))
        except TypeError:
            self.filenames = []

            for index in filename:
                if f == 'p':
                    self.filenames += glob.glob("../*" + f'{index:0>{5}}' + "*.txt")
                else:
                    self.filenames += glob.glob("*" + f'{index:0>{5}}' + "*.txt")   

        
        def get_mapping_file_headers(filenames):
            headers = [file_path.split(".txt")[0] for file_path in filenames]
            return headers
        self.headers = get_mapping_file_headers(self.filenames)
        
        self.specs, self.infos, self.i0s = [], [], []
        for header in tqdm(self.headers, desc="Loading spectra"):
            spec = Spectrum(header, dpcor, spin)
            self.specs.append(spec)   #[Spectrum(header1), Spectrum(header2),....]
            self.infos.append(spec.info)
            self.i0s.append(spec.i0) #[Spectrum(header1).i0, Spectrum(header2).i0,....]
            
        self.lens_scale = self.specs[0].lens_scale
        self.deflector_scale = self.specs[0].deflector_scale

        print(f"Elapsed time: {(time.time()-start_time):.2f} seconds")

    def __repr__(self):
        return f'{self.headers[0]}'
        
    def element_1Dscan(self, n, Ek = 0, ra = 1, d='Z',s = 'o',
                       scaling = 1, shift_pos = 0, shift_int = 0,
                       v_min=5, v_max=99.5, **kwargs):
        map_sum = []
        if Ek == 0:
            for i in range(len(self.specs)):
                map_sum += [np.sum(self.specs[i].edc)]    
        else:
            Ei = self.specs[0].e_to_i(e = Ek-ra)
            Ef = self.specs[0].e_to_i(e = Ek+ra)+1
            for i in range(len(self.specs)):
                map_sum += [np.sum(self.specs[i].edc[Ei:Ef])]
                
        if d == 'Z':
            z1 = float(self.specs[0].info['Z'].split()[0])
            z2 = float(self.specs[-1].info['Z'].split()[0])
            pos = np.linspace(z1, z2, n) + shift_pos             
        else:
            x1 = float(self.specs[0].info['X'].split()[0])
            x2 = float(self.specs[-1].info['X'].split()[0])
            pos = np.linspace(x1, x2, n) + shift_pos
        default_kwargs = {
                'c': 'black',
                'label': self.filenames
        }
        default_kwargs.update(kwargs)
        plt.plot(pos,np.array(map_sum)*scaling + shift_int,s, **kwargs)
        plt.xlabel(f'{d} pos (mm)')
        plt.ylabel('Intensity (a.u.)')
         
    def element_mapping(self, nx, nz, Ek=0, ra=1, v_min=5, v_max=99.5, **kwargs):
        if Ek == 0:
            map_sum = np.array([np.sum(spec.edc) for spec in self.specs])
        else:
            Ei = self.specs[0].e_to_i(e=Ek-ra)
            Ef = self.specs[0].e_to_i(e=Ek+ra)+1
            map_sum = np.array([np.sum(spec.edc[Ei:Ef]) for spec in self.specs])
    
        map_sum = map_sum.reshape((nx, nz))
    
        try:
            x1 = float(self.specs[0].info['X'].split()[0])
            x2 = float(self.specs[-1].info['X'].split()[0])
            z1 = float(self.specs[0].info['Z'].split()[0])
            z2 = float(self.specs[-1].info['Z'].split()[0])
            X = np.linspace(x1, x2, nx)
            Z = np.linspace(z1, z2, nz)
            Z, X = np.meshgrid(Z, X)
        except TypeError:
            map_sum = ndimage.rotate(map_sum, 90)
            X = np.arange(0, nx, 1)
            Z = np.arange(nz, 0, -1)
            X, Z = np.meshgrid(X, Z)
    
        default_kwargs = {
            'cmap': 'turbo',
            'vmin': np.percentile(map_sum, v_min),
            'vmax': np.percentile(map_sum, v_max)
        }
        default_kwargs.update(kwargs)
        plt.pcolormesh(X, Z, map_sum, **default_kwargs)
        plt.xlabel('x (mm)' if 'X' in self.specs[0].info else 'x steps')
        plt.ylabel('z (mm)' if 'Z' in self.specs[0].info else 'z steps')
        plt.title(self.specs[0].metadata.get('RegName', ''))
        plt.gca().set_aspect('equal')
        plt.gca().invert_xaxis()

    def deflector_mapping_plot(self, Ek, ra=0.2, fl=0, k_space=True,
                               k_origin=0, slit_rescale=1/1.1, v_min=5, v_max=99.5, **kwargs):
        data_array = np.array([spec.data for spec in self.specs])
        
        Ei = self.specs[0].e_to_i(Ek - ra)
        Ef = self.specs[0].e_to_i(Ek + ra) + 1
        
        X = self.lens_scale * slit_rescale
        Y = self.deflector_scale
        X, Y = np.meshgrid(X, Y)
    
        if k_space:
            kx = 0.5124 * np.sqrt(Ek) * np.sin(np.radians(X)) - k_origin
            ky = 0.5124 * np.sqrt(Ek) * np.sin(np.radians(Y))
            X_plot, Y_plot = kx, ky
            xlabel = r"$k_\parallel^\mathrm{Lens}$ / $1/\mathrm{\AA}$"
            ylabel = r"$k_\parallel^\mathrm{Deflection}$ / $1/\mathrm{\AA}$"
        else:
            X_plot, Y_plot = X, Y
            xlabel = "AngleY (degree)"
            ylabel = "AngleX (degree)"
    
        Z = np.sum(data_array[:, Ei:Ef, :], axis=1)
    
        default_kwargs = {
            'cmap': 'turbo',
            'vmin': np.percentile(Z, v_min),
            'vmax': np.percentile(Z, v_max)
        }
        default_kwargs.update(kwargs)
        plt.pcolormesh(X_plot, Y_plot, Z, **default_kwargs)
        plt.gca().set_aspect('equal')
    
        if fl == 0:
            plt.title(r'$E_\mathrm{k}$' + f'={Ek:.2f} eV ± {ra:.2f} eV')
        else:
            plt.title(r'$E_\mathrm{b}$' + f'={fl-Ek:.2f} eV ± {ra:.2f} eV')
    
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def deflector_mapping_sigma_cor_plot(self, Ek, ra=0.2, fl=0,
                                         sigma = 5, k_space=True, k_origin=0, slit_rescale=1/1.1,
                                         v_min=5, v_max=99.5, **kwargs):
        data_array = []
        for spec in self.specs:
            MDCs = np.average(spec.data, axis=0)
            data_array.append(spec.data * gaussian_filter(MDCs, sigma) / (MDCs + 1))
        data_array = np.array(data_array)
        
        Ei = self.specs[0].e_to_i(Ek - ra)
        Ef = self.specs[0].e_to_i(Ek + ra) + 1
        
        X = self.lens_scale * slit_rescale
        Y = self.deflector_scale
        X, Y = np.meshgrid(X, Y)
    
        if k_space:
            kx = 0.5124 * np.sqrt(Ek) * np.sin(np.radians(X)) - k_origin
            ky = 0.5124 * np.sqrt(Ek) * np.sin(np.radians(Y))
            X_plot, Y_plot = kx, ky
            xlabel = r"$k_\parallel^\mathrm{Lens}$ / $1/\mathrm{\AA}$"
            ylabel = r"$k_\parallel^\mathrm{Deflection}$ / $1/\mathrm{\AA}$"
        else:
            X_plot, Y_plot = X, Y
            xlabel = "AngleY (degree)"
            ylabel = "AngleX (degree)"
    
        Z = np.sum(data_array[:, Ei:Ef, :], axis=1)
    
        default_kwargs = {
            'cmap': 'turbo',
            'vmin': np.percentile(Z, v_min),
            'vmax': np.percentile(Z, v_max)
        }
        default_kwargs.update(kwargs)
        plt.pcolormesh(X_plot, Y_plot, Z, **default_kwargs)
        plt.gca().set_aspect('equal')
    
        if fl == 0:
            plt.title(r'$E_\mathrm{k}$' + f'={Ek:.2f} eV ± {ra:.2f} eV')
        else:
            plt.title(r'$E_\mathrm{b}$' + f'={fl-Ek:.2f} eV ± {ra:.2f} eV')
    
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def deflector_mapping_mdc_cor_plot(self, Ek, ra=0.2, fl=0,
                                       s_mapl=5, s_mapd=5,k_space=True, k_origin=0, slit_rescale=1/1.1,
                                       v_min=5, v_max=99.5, **kwargs):
        data_array = []
        for spec in self.specs:
            MDCs = np.average(spec.data, axis=0)
            data_array.append(spec.data / (MDCs + 1))
        data_array = np.array(data_array)
    
        Ei = self.specs[0].e_to_i(Ek - ra)
        Ef = self.specs[0].e_to_i(Ek + ra) + 1
    
        X = self.lens_scale * slit_rescale
        Y = self.deflector_scale
        X, Y = np.meshgrid(X, Y)

        if k_space:
            X_plot = 0.5124 * np.sqrt(Ek) * np.sin(np.radians(X)) - k_origin
            Y_plot = 0.5124 * np.sqrt(Ek) * np.sin(np.radians(Y))
            xlabel = r"$k_\parallel^\mathrm{Lens}$ / $1/\mathrm{\AA}$"
            ylabel = r"$k_\parallel^\mathrm{Deflection}$ / $1/\mathrm{\AA}$"
        else:
            X_plot, Y_plot = X - k_origin, Y
            xlabel = "AngleY (degree)"
            ylabel = "AngleX (degree)"
        self.center_d = np.abs(X_plot[0]).argmin()
        Z = np.sum(data_array[:, Ei:Ef, :], axis=1)
    
        lens_profile = np.sum(Z, axis=0)
        Z = Z * gaussian_filter(lens_profile, s_mapl) / (lens_profile + 1)
        deflector_profile = np.sum(Z, axis=1)
        Z = (Z.T * gaussian_filter(deflector_profile, s_mapd) / (deflector_profile + 1)).T
    
        default_kwargs = {
            'cmap': 'turbo',
            'vmin': np.percentile(Z, v_min),
            'vmax': np.percentile(Z, v_max)
        }
        default_kwargs.update(kwargs)
    
        plt.pcolormesh(X_plot, Y_plot, Z, **default_kwargs)
        plt.gca().set_aspect('equal')
    
        if fl == 0:
            plt.title(r'$E_\mathrm{k}$' + f'={Ek:.2f} eV ± {ra:.2f} eV')
        else:
            plt.title(r'$E_\mathrm{b}$' + f'={fl-Ek:.2f} eV ± {ra:.2f} eV')
    
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
    def deflector_arpes_mdc_cor_plot(self, fl=0, k_space=True, v_min=5, v_max=99.5, **kwargs):
        data_array = []
        for spec in self.specs:
            MDCs = np.average(spec.data, axis=0)
            data_array.append(spec.data / (MDCs + 1))
        data_array = np.array(data_array)
    
        X = self.specs[0].deflector_scale
        Y = self.specs[0].energy_scale
        X, Y = np.meshgrid(X, Y)
    
        if k_space:
            X_plot = 0.5124 * np.sqrt(Y) * np.sin(np.radians(X))
            Y_plot = Y
            xlabel = r"$k_\parallel^\mathrm{Deflection}$ / $1/\mathrm{\AA}$"
        else:
            X_plot, Y_plot = X, Y
            xlabel = "AngleY (degree)"
    
        Z = data_array[:, :, self.center_d].T
    
        default_kwargs = {
            'cmap': 'turbo',
            'vmin': np.percentile(Z, v_min),
            'vmax': np.percentile(Z, v_max)
        }
        if k_space:
            default_kwargs['shading'] = 'gouraud'
            default_kwargs['rasterized'] = True
        default_kwargs.update(kwargs)
    
        if fl == 0:
            plt.ylabel('Kinetic Energy (eV)')
        else:
            Y_plot = Y_plot - fl
            plt.ylabel('Binding Energy (eV)')
        plt.xlabel(xlabel)
        plt.pcolormesh(X_plot, Y_plot, Z, **default_kwargs)

    def energy_mapping_edc_plot(self, start_E, end_E, WF=4.5, lw = 0.5):
        scans = np.size(self.specs)
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(self.specs)))
        for i in range(len(self.specs)):
            self.specs[i].edc_plot(fl = start_E + (end_E - start_E)/(len(self.specs) - 1)*i - WF, label = str(i*(end_E - start_E)/(scans - 1) + start_E) + ' (eV)', c=colors[i], lw = lw)

    def energy_mapping_edc_cor_plot(self, start_E, end_E, find_E1=2, find_E2=2.8, lw = 0.5):
        scans = np.size(self.specs)
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, scans))
        for i in range(scans):
            self.specs[i].edc_plot(self.specs[i].find_fl(find_E1, find_E2), label = str(i*(end_E - start_E)/(scans - 1) + start_E) + ' (eV)', c=colors[i], lw = lw)

    def energy_mapping_plot(self, start_E, end_E, ra=0.1,WF=4.5,
                            E_cut=0,lower_bound=0,higher_bound=-1, 
                            k_space=True, k_origin=0, slit_rescale=1/1.1, V_0=5, photon_angle=30, 
                            v_min=5, v_max=99.5, **kwargs): #kz map using fixed WF to determine EF=hv-WF
        scans = len(self.specs) #number of loaded Spectrum
        data_array = [spec.data for spec in self.specs] #[[E x theta array 1],[E x theta array 2],...] where each spec.data is a @D matrix of intensity at each (Ek, theta)
        data_array = np.array(data_array) #change to numpy array
    
        hv = np.linspace(start_E, end_E, scans) #list of hv
    
        Ei = self.specs[0].e_to_i(start_E - WF - ra - E_cut) #Index for lowest E_window (selected from the self.specs[0], so have to assume that the measured energy range is shifted equally to hv measured)
        Ef = self.specs[0].e_to_i(start_E - WF + ra - E_cut) + 1 #+1 for final index
    
        Z_array = [np.sum(Z[Ei:Ef, :], axis=0) for Z in data_array] #[[int11, int21,..],[int12,int22,...],...] (each int is sum value from Ei to Ef at each theta)
        Z_array = np.array(Z_array) # change to numpy array
        Z_array = Z_array[:, lower_bound:higher_bound] # for all hv, select subset of theta index (defaults = take all)
        lens_profile = np.sum(Z_array, axis=0) #sum along same B.E. [[int11+int12+..],[int21+int22+..],...] each index=each theta
        Z_array = Z_array * gaussian_filter(lens_profile, 5) / (lens_profile + 1) # fix vertical lines
        energies_profile = np.sum(Z_array, axis=1) # sum along theta (after fixing the burnt theta pixels)
        Z_array = (Z_array.T * gaussian_filter(energies_profile, 5)/ (energies_profile + 1)).T # fix intensity difference across energies

        X = self.lens_scale * slit_rescale #[theta1*scaling, theta2*scaling,..]_hv1, [theta1*scaling, theta2*scaling,..]_hv2
        X, hv = np.meshgrid(X, hv) #[[theta1,hv1],[theta2,hv1],,...]
                                   #[[theta1,hv2],[theta2,hv2],,...]
        X = X[:, lower_bound:higher_bound] #restricted X for specific theta range [theta1,theta2,...],[theta1,theta2,...],...
        hv = hv[:, lower_bound:higher_bound] #restricted hv for specific theta range [hv1,hv1,hv1,...],[hv2,hv2,...],...
    
        if k_space:
            kx = np.array([[0.5124 * np.sqrt(hv[i][j] - E_cut - WF) * np.sin(np.deg2rad(X[i][j]))
                            for j in range(X.shape[1])] for i in range(X.shape[0])]) #X.shape[0]= number of row in X= each hv , X.shape[1]= number of column in X= each theta 
            kz = np.array([[0.5124 * np.sqrt((hv[i][j] - E_cut - WF) * np.cos(np.deg2rad(X[i][j]))**2 + V_0)
                            + np.sin(np.deg2rad(photon_angle)) * hv[i][j] * 5.067e-4
                            for j in range(X.shape[1])] for i in range(X.shape[0])])
            X_plot, Y_plot = kx - k_origin, kz
            xlabel = r"$k_\parallel$ ($1/\AA$)"
            ylabel = r"$k_\perp$ ($1/\AA$)"
            default_kwargs = {'shading': 'gouraud'}
        else:
            X_plot, Y_plot = X - k_origin, hv
            xlabel = "Angle (degree)"
            ylabel = "Photon Energy (eV)"
            default_kwargs = {}
        
        default_kwargs.update({
            'cmap': 'viridis',
            'vmin': np.percentile(Z_array, v_min),
            'vmax': np.percentile(Z_array, v_max)
        })
        default_kwargs.update(kwargs)
    
        plt.pcolormesh(X_plot, Y_plot, Z_array, **default_kwargs)
        plt.gca().set_aspect('equal' if k_space else 'auto')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def plot_energy_line(self, energy, E_cut=0, k_origin = 0, slit_rescale=1/1.1,
                         WF=4.5, V_0=5,photon_angle = 30,
                         lower_bound = 0, higher_bound = -1,
                         c='r', lw=1, alpha=0.5):
        X = self.lens_scale * slit_rescale
        kx_1 = [(0.5124*np.sqrt(energy-E_cut-WF)*np.sin(np.deg2rad(X[j]))) for j in range(lower_bound,len(X)+higher_bound)]
        kz_1 = [(0.5124*np.sqrt((energy-E_cut-WF)*np.cos(np.deg2rad(X[j]))**2+V_0) + np.sin(np.deg2rad(photon_angle))*energy*5.067*10**-4) for j in range(lower_bound,len(X)+higher_bound)]
        kx_1 = np.array(kx_1)
        plt.plot(kx_1 - k_origin,kz_1,label=r'{} eV'.format(str(energy)),c=c, lw=lw, alpha=alpha)

    def energy_mapping_cor_plot(self, start_E, end_E, ra=0.1, E_cut=0,
                            find_E1=2, find_E2=2.8,
                            s_spec=15, s_mapv=5, s_maph=5,
                            lower_bound=0, higher_bound=-1,
                            k_space=True, k_origin=0, slit_rescale=1/1.1, V_0=5, photon_angle=30,
                            v_min=5, v_max=99.5, **kwargs):
        scans = len(self.specs)
        data_array = []
        fls, Eis, Efs = [], [], []
    
        for i in range(scans):
            MDCs = np.average(self.specs[i].normdata, axis=0)
            data_array.append(self.specs[i].normdata * gaussian_filter(MDCs, s_spec) / (MDCs + 1))
            fl = self.specs[i].find_fl(find_E1, find_E2)
            fls.append(fl)
            Eis.append(self.specs[i].e_to_i(fl - E_cut - ra))
            Efs.append(self.specs[i].e_to_i(fl - E_cut + ra) + 1)
        data_array = np.array(data_array)
        hv = np.linspace(start_E, end_E, scans)
        Z_array = np.array([np.sum(data_array[i][Eis[i]:Efs[i], :], axis=0) for i in range(scans)])
        lens_profile = np.sum(Z_array, axis=0)
        Z_array = Z_array * gaussian_filter(lens_profile, s_mapv) / (lens_profile + 1)
        energies_profile = np.sum(Z_array, axis=1)
        Z_array = (Z_array.T * gaussian_filter(energies_profile, s_maph) / (energies_profile + 1)).T
    
        X = self.lens_scale * slit_rescale
        X, hv = np.meshgrid(X, hv)
    
        if k_space:
            kx = np.array([[0.5124 * np.sqrt(fls[i] - E_cut) * np.sin(np.deg2rad(X[i][j]))
                            for j in range(len(X[i]))] for i in range(len(X))])
            kz = np.array([[0.5124 * np.sqrt((fls[i] - E_cut) * np.cos(np.deg2rad(X[i][j]))**2 + V_0)
                            + np.sin(np.deg2rad(photon_angle)) * hv[i][j] * 5.067e-4
                            for j in range(len(X[i]))] for i in range(len(X))])
            X_plot, Y_plot = kx - k_origin, kz
            xlabel = r"$k_\parallel$ ($1/\AA$)"
            ylabel = r"$k_\perp$ ($1/\AA$)"
            default_kwargs = {'shading': 'gouraud'}
            plt.gca().set_aspect('equal')
        else:
            X_plot, Y_plot =X - k_origin, hv
            xlabel = "Angle (degree)"
            ylabel = "Photon Energy (eV)"
            default_kwargs = {}
            plt.gca().set_aspect('auto')
        self.center_e = np.abs(X_plot[0]).argmin()
        default_kwargs.update({
            'cmap': 'viridis',
            'vmin': np.percentile(Z_array, v_min),
            'vmax': np.percentile(Z_array, v_max)
        })
        default_kwargs.update(kwargs)
    
        plt.pcolormesh(X_plot, Y_plot, Z_array, **default_kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def kz_arpes_plot_k(self, start_E, end_E, k_origin = 0, find_E1=2, find_E2=2.8, E_cut=0,photon_angle = 30, V_0=5, s_spec = 15, s_mapv = 5, s_maph = 5,lower_bound=0, higher_bound = -1, v_min=5, v_max=99.5, **kwargs):
        scans = len(self.specs)
        data_array = []
        fls = []
        for i in range(scans):
            MDCs = np.average(self.specs[i].normdata, axis = 0)
            data_array += [self.specs[i].normdata * gaussian_filter(MDCs, s_spec) / (MDCs + 1)]
            fls += [self.specs[i].find_fl(find_E1, find_E2)]
        data_array =  np.array(data_array)
        hv = np.linspace(start_E,end_E,scans)
        
        X = self.lens_scale
        X, hv = np.meshgrid(X, hv)
        kx = np.array([[0.5124 * np.sqrt(fls[i] - E_cut) * np.sin(np.deg2rad(X[i][j]))
                        for j in range(len(X[i]))] for i in range(len(X))])
        # kx0_n = np.argmin(np.abs(kx[0] - k_origin))
        kz = np.array([[0.5124 * np.sqrt((fls[i] - E_cut) * np.cos(np.deg2rad(X[i][j]))**2 + V_0)
                        + np.sin(np.deg2rad(photon_angle)) * hv[i][j] * 5.067e-4
                        for j in range(len(X[i]))] for i in range(len(X))])
        
        Z_array = []
        # n = 10
        # Z_array += [np.sum(Z[:,kx0_n - n:kx0_n + n], axis=1) for Z in data_array]
        
        Z_array += [Z[:,self.center_e] for Z in data_array] # Z_array += [Z[:,kx0_n] for Z in data_array]
        Z_array = np.array(Z_array)
        
        lens_profile = np.sum(Z_array, axis=0)
        Z_array = Z_array * gaussian_filter(lens_profile, s_mapv) / (lens_profile + 1) # fix vertical lines
        energies_profile = np.sum(Z_array, axis=1)
        Z_array = (Z_array.T * gaussian_filter(energies_profile, s_maph)/ (energies_profile + 1)).T # fix intensity difference across energies
        
        y = self.specs[0].energy_scale - fls[0]
        kz = np.array(kz)
        x = kz[:,self.center_e] # x = kz[:,kx0_n]
        x, y = np.meshgrid(x, y)
        
        default_kwargs = {
                'cmap': 'turbo',
                'vmin':np.percentile(Z_array, v_min),
                'vmax':np.percentile(Z_array, v_max)
            }
        default_kwargs.update(kwargs)
        plt.pcolormesh(x,y,Z_array.T,**default_kwargs)
        plt.xlabel(r"k$_{z}$ ($1/\AA$)")
        plt.ylabel('Binding Energy (eV)')

    def SM_map_plot(self):
        X_0 = np.linspace(self.specs[0].metadata["MapStartX"],
                          self.specs[0].metadata["MapEndX"],
                          self.specs[0].metadata["MapNoXSteps"])
        Y_0 = np.linspace(self.specs[0].metadata["MapStartY"],
                          self.specs[0].metadata["MapEndY"],
                          self.specs[0].metadata["MapNoYSteps"])
        X,Y = np.meshgrid(X_0,Y_0)
        map_data = np.zeros(np.size(X))
        for i in range(len(self.specs)):
            map_data[i] = self.specs[i].data
        map_data = map_data.reshape(np.size(X_0),np.size(Y_0))
        plt.pcolormesh(X,Y, map_data, cmap='inferno')
        plt.xlabel('SM X direction')
        plt.ylabel('SM Y direction')
        plt.gca().set_aspect('equal')
        
class ResPES():
    def __init__(self,ResPES_list, f = 'c'):
        start_time = time.time()
        
        self.filenames = []
        for file in ResPES_list:
            if f == 'p':
                if 'MBS' in str(file):
                    self.filenames += ["../" + file + ".txt"]
                else:
                    self.filenames += [natsorted(glob.glob("../*MBS-" + f'{file:0>{5}}' + "*.txt"))[-1]]
            else:
                if 'MBS' in str(file):
                    self.filenames += [file + ".txt"]
                else:
                    self.filenames += [natsorted(glob.glob("*MBS-" + f'{file:0>{5}}' + "*.txt"))[-1]]
        def get_mapping_file_headers(filenames):
            headers = [file_path.split(".txt")[0] for file_path in filenames]
            return headers
        self.headers = get_mapping_file_headers(self.filenames)
        self.specs = [Spectrum(header) for header in tqdm(self.headers, desc="Loading spectra")]
        self.infos = [self.specs[i].info for i in range(len(self.headers))]
        self.i0s = [self.specs[i].i0 for i in range(len(self.headers))]
        
        print(f"Elapsed time: {(time.time()-start_time):.2f} seconds")
   
    def BE_plot(self, energy_list, phi):
        fls = []
        for i in range(len(energy_list)):
            fls += [energy_list[i] - phi]
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(self.specs)))
        for i in range(len(energy_list)):
            self.specs[i].edc_plot(fl = fls[i], c = colors[i], label = f'{energy_list[i]} eV')
    
    def BE_map_plot(self, energy_list, phi, v_min=5, v_max=99.5, **kwargs):
        BEs = []
        for i in range(len(energy_list)):
            BEs += [energy_list[i] - phi - self.specs[i].energy_scale]
        edcs = []
        for i in range(len(energy_list)):
            edcs += [self.specs[i].edc]
        sum_edcs = np.sum(edcs, axis=1)
        default_kwargs = {
                'cmap': 'viridis',
                'vmin':np.percentile(edcs, v_min),
                'vmax':np.percentile(edcs, v_max),
            }
        default_kwargs.update(kwargs)
#         plt.pcolormesh(BEs, energy_list, edcs, **default_kwargs)
#         plt.gca().invert_xaxis()
        fig, main_ax = plt.subplots()
        main_plot = main_ax.pcolormesh(BEs, energy_list, edcs, **default_kwargs)
        main_ax.invert_xaxis()
        plt.xlabel('Binding Energy (eV)')
        plt.ylabel('Photon Energy (eV)')
        left_ax = fig.add_axes([main_ax.get_position().x0 - 0.27, main_ax.get_position().y0, 0.15, main_ax.get_position().height], sharey=main_ax)
        left_ax.plot(sum_edcs, energy_list, '-ok', markersize=2.5, linewidth=1, label='XAS')
        left_ax.invert_xaxis()
