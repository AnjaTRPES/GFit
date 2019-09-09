import sys
import math
from PyQt5.QtWidgets import QApplication, QDialog,QMainWindow,QMessageBox,QFileDialog
from PyQt5.QtCore import pyqtSlot,QRect
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
from ui_gfit_main2 import Ui_Gfit

import pyqtgraph as pg
pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
import numpy as np
from scipy.optimize import leastsq
from matplotlib import cm
from scipy import interpolate
from scipy.optimize import curve_fit
import copy as copy

class Gaussian(object):
    def __init__(self, pos, fwhm,intensity=1.):
        self.pos=pos
        self.fwhm=fwhm
        self.intensity=intensity

    def get_gaussian(self,x):
        sig=self.fwhm/(2*np.sqrt(2*np.log(2)))
        y=self.intensity*np.exp(-(x-self.pos)**2/(2*sig**2))
        return y
    def setPos(self,pos):
        self.pos=pos
    def setFwhm(self,fwhm):
        self.fwhm=fwhm
    def getPos(self):
        return self.pos
    def getFwhm(self):
        return self.fwhm
    def return_values(self):
        return self.pos,self.fwhm,self.intensity

class Gaussian_energy(object):
    def __init__(self, pos, fwhm,intensity=1.):
        self.pos=np.sqrt(pos)
        self.fwhm=fwhm
        self.intensity=intensity

    def get_gaussian(self,x):
        x=np.sqrt(x)
        sig=self.fwhm/(2*np.sqrt(2*np.log(2)))
        sig=np.sqrt(sig)/15
        y=self.intensity*np.exp(-(x-self.pos)**2/(2*sig**2))
        return y
    def setPos(self,pos):
        self.pos=pos
    def setFwhm(self,fwhm):
        self.fwhm=fwhm
    def getPos(self):
        return self.pos
    def getFwhm(self):
        return self.fwhm
    def return_values(self):
        return self.pos,self.fwhm,self.intensity

class fits(object):
    def __init__(self):
        pass
    def get_gaussian(self,x,pos,fwhm,intensity):
        sig=fwhm/(2*np.sqrt(2*np.log(2)))
        y=intensity*np.exp(-(x-pos)**2/(2*sig**2))
        return y

    def get_gaussian_energy(self,x,pos,fwhm,intensity):
        sig=fwhm/(2*np.sqrt(2*np.log(2)))
        pos=np.sqrt(pos)
        sig=np.sqrt(sig)/15
        y=intensity*np.exp(-(np.sqrt(x)-pos)**2/(2*sig**2))
        return y
    
    def return_fitted(self,gaussians,x,which_fit_function):
        y_fit=np.zeros(x.shape)
        i=0
        for g in range(int(len(gaussians)/3)):
            if which_fit_function==0:
                y_fit+=self.get_gaussian(x,gaussians[i],
                                         gaussians[i+1],gaussians[i+2])
            elif which_fit_function==1:
                y_fit+=self.get_gaussian_energy(x,gaussians[i],
                                         gaussians[i+1],gaussians[i+2])
            i+=3
        return y_fit
    
    def fit_function(self,gaussians,x,y_exp,which_fit_function,sigmas=None):
        #print('jo')
        #gaussians is a list of gaussian values [pos,fwhm, intensity
        y_fit=np.zeros(x.shape)
        i=0
        for g in range(int(len(gaussians)/3)):
            if which_fit_function==0:
                y_fit+=self.get_gaussian(x,gaussians[i],
                                     gaussians[i+1],gaussians[i+2])
            elif which_fit_function==1:
                y_fit+=self.get_gaussian_energy(x,gaussians[i],
                                     gaussians[i+1],gaussians[i+2])
            i+=3
        if type(sigmas)!=type(None):
            return_root=np.sqrt(np.power(np.divide(y_fit-y_exp,sigmas),2))
        else:
            return_root=np.sqrt(np.power(y_fit-y_exp,2))
        return return_root
    
    def get_deltas(self,plsq,cov,x,y_exp,which_fit_function):
        #get residual
        #plsq,cov,x,self.summed,self.comboBox_type_gaussian.currentIndex()
        y_fit=np.zeros(x.shape)
        i=0
        gaussians=plsq
        for g in range(int(len(gaussians)/3)):
            if which_fit_function==0:
                y_fit+=self.get_gaussian(x,gaussians[i],
                                     gaussians[i+1],gaussians[i+2])
            elif which_fit_function==1:
                y_fit+=self.get_gaussian_energy(x,gaussians[i],
                                     gaussians[i+1],gaussians[i+2])
            i+=3
        Yres=y_fit
        res=np.sqrt(np.power(Yres,2).sum()/Yres.size)
        plsqs=[]
        Deltas=[]
        for n in range(len(plsq)):
            plsqs.append(plsq[n])
            if type(cov)==type(None):
                Delta='Inf'
            elif cov.any()==None:
                Delta='Inf'
            else:
                Delta=(np.sqrt(cov.diagonal()*res))[n]
            Deltas.append(Delta)
        return Deltas
        

class Gfit(QMainWindow,Ui_Gfit):
    def __init__(self,parent=None):
        super(Gfit,self).__init__(parent)
        self.setupUi(self)
        #define all variables and stuff
        self.viewer_sum_fitted_legend=[]
        self.z=[]
        self.x=[]
        self.y=''
        self.dir='home'
        self.switch_TRPES_axis=False
        self.display_log_t=False
        self.colors=[(255,0,0),(76,153,0),(0,255,255),(0,0,204),
                     (153,0,153),(255,0,127),(255,255,51),(255,128,0),
                     (255,0,0),(76,153,0),(0,255,255),(0,0,204),
                     (153,0,153),(255,0,127),(255,255,51),(255,128,0)]
        self.gaussians=[]
        self.gaussians_isolines=[]
        self.fit_x=[]
        self.roi=[]
        self.initialize_gaussians()
        self.initialize_possible_fits()
        self.switch_TRPES_axis=False
        self.calibrated=False
        self.y_cal=''
        self.sigmas=None
        self.original_z=None

    def initialize_possible_fits(self):
        self.comboBox_type_gaussian.blockSignals(True)
        self.comboBox_type_gaussian.addItem('Gaussian')
        self.comboBox_type_gaussian.addItem('Gaussian_energy')
        self.comboBox_type_gaussian.addItem('Limited range Gaussian Fit (+-5)')
        self.comboBox_type_gaussian.blockSignals(False)
        self.label_range.hide()
        self.doubleSpinBox_range.hide()

    @pyqtSlot()
    def on_pushButton_load_csv_clicked(self):
        '''
        loading a .csv file as saved in Andreys format
        '''
        print('loading csv file')
        filename = QFileDialog.getOpenFileName(self, 'Open File',self.dir)[0]
        self.dir=filename[:(-len(filename.split('/')[-1]))]
        self.filename=filename
        try:
            eV,t,z=self.load_csv_file(filename)
            self.z=z
            self.original_z=z
            self.x=t
            self.y=eV
            self.checkBox_subtract_background.setCheckState(False)
            self.sigmas=None
            self.plot_trpes()
            self.update_limits()
            self.add_roi()
            self.plot_extracted_roi()          
        except:
            print('stopped loading')
    
    def load_csv_file(self,filename):
        data=np.loadtxt(filename,delimiter=',')
        if data.shape[0] > 2:
            t=data[1:,0]
            eV=data[0,1:]
            z=data[1:,1:]/data[1:,1:].max()
            if t[0]>t[-1]:
                z=np.flip(z,axis=0) 
                t=np.flip(t,axis=0)
        else:
            eV=data[0,:]
            z=np.array([data[1,:],data[1,:]])
            t=np.array([1,2])
        return eV,t,z
            
    @pyqtSlot()
    def on_pushButton_load_sigmas_clicked(self): 
        #load sigma file
        print('clicked')
        if type(self.original_z)!=type(None):
            filename = QFileDialog.getOpenFileName(self, 'Open File',self.dir)[0]
            self.dir=filename[:(-len(filename.split('/')[-1]))]
            self.filename=filename
            eV,t,z=self.load_csv_file(filename)
            if z.shape[0]==self.original_z.shape[0] and z.shape[1]==self.original_z.shape[1]:
                self.sigmas=z
                print('loaded sigmas')
                self.plot_extracted_roi()
            else:
                self.sigmas=None
                print('not the same dimensions')

    @pyqtSlot('int')
    def on_tabWidget_currentChanged(self,value):
        if value==0:
            self.calibrated=False
            if type(self.y)!=type('f'):
                self.plot_trpes()
                self.update_limits()
                self.add_roi()
                self.plot_extracted_roi()
        else:
            self.calibrated=True
            if type(self.y_cal)!=type('f'):
                self.plot_calibrated_trpes()
                self.update_limits()
                self.add_roi()
                self.plot_extracted_roi()

    def update_limits(self):
        '''
        upadate the limits and values of the background, x and y values
        selection stuff
        '''
        for function in [self.doubleSpinBox_bg_subtract_from,self.doubleSpinBox_bg_subtract_to,
                         self.doubleSpinBox_x_lower,self.doubleSpinBox_y_lower,self.doubleSpinBox_x_upper,
                         self.doubleSpinBox_y_upper]:
            function.blockSignals(True)
        self.doubleSpinBox_bg_subtract_from.setRange(self.x.min(),self.x.max())
        self.doubleSpinBox_bg_subtract_from.setValue(self.x.min())
        self.doubleSpinBox_bg_subtract_to.setRange(self.x.min(),self.x.max())
        self.doubleSpinBox_bg_subtract_to.setValue(self.x.max())
        if self.switch_TRPES_axis==False:
            self.doubleSpinBox_x_lower.setRange(self.x.min(),self.x.max())
            self.doubleSpinBox_x_upper.setRange(self.x.min(),self.x.max())
            if self.calibrated==False:
                self.doubleSpinBox_y_lower.setRange(self.y.min(),self.y.max())
                self.doubleSpinBox_y_upper.setRange(self.y.min(),self.y.max())
                self.doubleSpinBox_y_lower.setValue(self.y.min())
                self.doubleSpinBox_y_upper.setValue(self.y.max())
            else:
                self.doubleSpinBox_y_lower.setRange(self.y_cal.min(),self.y_cal.max())
                self.doubleSpinBox_y_upper.setRange(self.y_cal.min(),self.y_cal.max())
                self.doubleSpinBox_y_lower.setValue(self.y_cal.min())
                self.doubleSpinBox_y_upper.setValue(self.y_cal.max())
            self.doubleSpinBox_x_lower.setValue(self.x.min())
            self.doubleSpinBox_x_upper.setValue(self.x.max())
        else:
            print('switched the axis')
            if self.calibtrated==False:
                self.doubleSpinBox_x_lower.setRange(self.y.min(),self.y.max())
                self.doubleSpinBox_x_upper.setRange(self.y.min(),self.y.max())
                self.doubleSpinBox_x_lower.setValue(self.y.min())
                self.doubleSpinBox_x_upper.setValue(self.y.max())
            else:
                self.doubleSpinBox_x_lower.setRange(self.y_cal.min(),self.y_cal.max())
                self.doubleSpinBox_x_upper.setRange(self.y_cal.min(),self.y_cal.max())
                self.doubleSpinBox_x_lower.setValue(self.y_cal.min())
                self.doubleSpinBox_x_upper.setValue(self.y_cal.max())
            self.doubleSpinBox_y_lower.setRange(self.x.min(),self.x.max())
            self.doubleSpinBox_y_upper.setRange(self.x.min(),self.x.max())
            
            self.doubleSpinBox_y_lower.setValue(self.x.min())
            self.doubleSpinBox_y_upper.setValue(self.x.max())
            
        for function in [self.doubleSpinBox_bg_subtract_from,self.doubleSpinBox_bg_subtract_to,
                         self.doubleSpinBox_x_lower,self.doubleSpinBox_y_lower,self.doubleSpinBox_x_upper,
                         self.doubleSpinBox_y_upper]:
            function.blockSignals(False)

    def add_roi(self):
        '''
        add the roi rectangle
        '''
        if self.roi==[]:
            x_min=self.x.min()
            x_max=self.x.max()
            if self.calibrated==False:
                y_min=self.y.min()
                y_max=self.y.max()
            else:
                y_min=self.y_cal.min()
                y_max=self.y_cal.max()

            print('added roi')
        else:
            print('update roi')
            x_min=self.doubleSpinBox_x_lower.value()
            x_max=self.doubleSpinBox_x_upper.value()
            y_min=self.doubleSpinBox_y_lower.value()
            y_max=self.doubleSpinBox_y_upper.value()
        if self.switch_TRPES_axis==False:
            self.roi = pg.ROI([x_min, y_min],[x_max-x_min, y_max-y_min])
        else:
            self.roi = pg.ROI([x_min, y_min],[x_max-x_min, y_max-y_min])         
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.p1.addItem(self.roi)
        self.roi.sigRegionChangeFinished.connect(self.selection_changed_box)
        self.roi.setZValue(10)  # make sure ROI is drawn above image

    def on_actionSwitch_TRPES_axis_toggled(self,value):
        print('clicked',value)
        self.switch_TRPES_axis=value
        if self.z!=[]:
            self.plot_trpes()
            #self.checkBox_subtract_background.blockSignals(True)
            self.checkBox_subtract_background.setChecked(False)
            #self.checkBox_subract_background.blockSignals(False)
            self.update_limits()
            self.add_roi()
            self.plot_extracted_roi()
            
        

    def plot_trpes(self):
        """
        function to plot the TRPES on an appropriate viewer, need to implement it once I know how to properly display such stuff
        viewer: the plotwidget
        z: np 2D array of dimensions [M,N]
        y: 1D array of dimensions N
        x: 1D array of dimensions M
        """
        '''
        self.D2Trpes = pg.ImageView()
        self.D2Trpes.setImage(z)
        viewer.addItem(self.D2Trpes)
        '''
        print('self.roi_beginning_trpes',self.roi)
        viewer=self.viewer_global_orig_TRPES
        z=self.z
        x=self.x
        y=self.y
        try:
            if self.display_log_t==True:
                f = interpolate.interp2d(x, y, z.transpose(), kind='linear')
                xi1=-np.flip(np.logspace(np.log10(0.1),np.log10(-x.min()),250,endpoint=True),axis=0)
                xi2=np.logspace(np.log10(0.1),np.log10(x.max()),250,endpoint=True)
                xi=np.concatenate([xi1,xi2],axis=0)
                yi=np.linspace(y.min(),y.max()+np.abs(y[0]-y[1]),500)
                print('shapes',xi.shape,yi.shape)
                z2=f(xi,yi).transpose()
                if self.switch_TRPES_axis==True:
                    y3=xi
                    x3=yi
                    z2=z2.transpose()
                else:
                    y3=yi
                    x3=xi
                #alright, clear the viewer and add that stuff
                viewer.clear()
                self.p1 = viewer.addPlot()
                self.filterMenu = QtGui.QMenu("Logharithmic Scale")
                self.change_to_logharithmic = QtGui.QAction("Change x-axis to logarithmic", self.filterMenu,checkable=True,checked=self.display_log_t)
                self.change_to_logharithmic.triggered.connect(lambda: self.change_viewer_scale_to_log(self.change_to_logharithmic.isChecked()))
                self.filterMenu.addAction(self.change_to_logharithmic)
                self.p1.ctrlMenu=[self.filterMenu]
                #add the colormap
                colormap = cm.get_cmap("nipy_spectral")  # cm.get_cmap("CMRmap")
                #color=np.array([[0,0,0,255],[255,128,0,255],[255,255,0,255]],dtype=np.ubyte)
                colormap._init()
                color=np.array((colormap._lut * 255).view(np.ndarray)[:-4,:],dtype=np.ubyte)
                pos=np.array(np.arange(0.,1.,1./color.shape[0]))
                map=pg.ColorMap(pos,color)
                lut=map.getLookupTable(0.,1.,256)
                # Item for displaying image data
                self.img = pg.ImageItem()
                self.p1.addItem(self.img)
                self.img.setImage(z2)
                self.img.setLookupTable(lut)
                self.img.setLevels([0,z2.max()])
                if self.switch_TRPES_axis==False:
                    self.p1.setLabel('left', "energy", units='eV')
                    self.p1.setLabel('bottom', "log(time)", units='ps')
                else:
                    self.p1.setLabel('left', "log(time)", units='ps')
                    self.p1.setLabel('bottom', "energy", units='eV')
                    
                     # Contrast/color control
                hist = pg.HistogramLUTItem()
                hist.setImageItem(self.img)
                viewer.addItem(hist)
        
                # Draggable line for setting isocurve level
                isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
                hist.vb.addItem(isoLine)
                hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
                isoLine.setValue(0.)
                isoLine.setZValue(z2.max()) # bring iso line above contrast controls
                hist.gradient.setColorMap(map)
        
                self.img.setLookupTable(lut)
                self.img.setLevels([0,z2.max()])
        
                if self.switch_TRPES_axis==False:
                    size_y=np.abs(y3[-1]-y3[0])
                    y_factor=size_y/y3.shape[0]
                    self.img.translate(0., y3.min())
                    self.img.scale(1., y_factor) 
                    
                    
                    xaxis=self.p1.getAxis('bottom')
        
                    values=[]
                    strings=[]
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((-10**(i+1)))
                                strings.append('-10^'+str(i+1))
                            else:
                                values.append(-j*(10**i))
                                strings.append('')
                    values.append(0)
                    strings.append('0')
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((10**(i+1)))
                                strings.append('10^'+str(i+1))
                            else:
                                values.append(j*(10**i))
                                strings.append('')
                    ticks=[list(zip(values,strings))]
                    #xaxis.setTicks(ticks)    
        
                    #second try:
                    f2=interpolate.interp1d(x3,range(len(x3)),  fill_value='extrapolate') 
                    y4=f2(np.array(values))
                    y4=list(y4)
                    ticks=[list(zip(y4,strings))]
                    xaxis.setTicks(ticks) 
                else:
                    size=np.abs(x3[-1]-x3[0])
                    x_factor=size/x3.shape[0]
                    self.img.translate(x3.min(),0.)
                    self.img.scale(x_factor,1) 
                    
                    
                    xaxis=self.p1.getAxis('left')
        
                    values=[]
                    strings=[]
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((-10**(i+1)))
                                strings.append('-10^'+str(i+1))
                            else:
                                values.append(-j*(10**i))
                                strings.append('')
                    values.append(0)
                    strings.append('0')
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((10**(i+1)))
                                strings.append('10^'+str(i+1))
                            else:
                                values.append(j*(10**i))
                                strings.append('')
                    ticks=[list(zip(values,strings))]
                    #xaxis.setTicks(ticks)    
        
                    #second try:
                    f2=interpolate.interp1d(y3,range(len(y3)),  fill_value='extrapolate') 
                    y4=f2(np.array(values))
                    y4=list(y4)
                    ticks=[list(zip(y4,strings))]
                    xaxis.setTicks(ticks)
    
            else:
                self.plot_TRPES_normally(viewer, z,x,y)
        except AttributeError:
            self.plot_TRPES_normally(viewer, z,x,y)
        print('self.roi_ending_trpes',self.roi)

            
    def plot_TRPES_normally(self, viewer, z,x,y):
        #test whether x is equally spaced
        x2=x[:-1]-x[1:]
        y2=y[:-1]-y[1:]
              
        if np.any(np.abs(x2-x2[0])>0.0001)==True or np.any(np.abs(x2-x2[0])>0.001)==True:
            print('interpolating')
            f = interpolate.interp2d(x, y, z.transpose(), kind='linear')
            xi=np.arange(x.min(),x.max(),np.abs(x2).min())
            yi=np.arange(y.min(),y.max()+np.abs(y2[0]),np.abs(y2).min())
            z=f(xi,yi).transpose()
        else:
            xi=x
            yi=y
        if self.switch_TRPES_axis==True:
            y3=xi
            x3=yi
            z=z.transpose()
        else:
            y3=yi
            x3=xi
    
        # A plot area (ViewBox + axes) for displaying the image
        viewer.clear()
        self.p1 = viewer.addPlot()
        self.filterMenu = QtGui.QMenu("Logharithmic Scale")
        self.change_to_logharithmic = QtGui.QAction("Change x-axis to logarithmic", self.filterMenu,checkable=True,checked=self.display_log_t)
        self.change_to_logharithmic.triggered.connect(lambda: self.change_viewer_scale_to_log(self.change_to_logharithmic.isChecked()))
        self.filterMenu.addAction(self.change_to_logharithmic)
        self.p1.ctrlMenu=[self.filterMenu]
        #pos=np.array([0.,0.5,1.0])
        colormap = cm.get_cmap("nipy_spectral")  # cm.get_cmap("CMRmap")
        #color=np.array([[0,0,0,255],[255,128,0,255],[255,255,0,255]],dtype=np.ubyte)
        colormap._init()
        color=np.array((colormap._lut * 255).view(np.ndarray)[:-4,:],dtype=np.ubyte)
        pos=np.array(np.arange(0.,1.,1./color.shape[0]))
        map=pg.ColorMap(pos,color)
        lut=map.getLookupTable(0.,1.,256)
        # Item for displaying image data
        self.img = pg.ImageItem()
        self.p1.addItem(self.img)
        self.img.setImage(z)
        self.img.setLookupTable(lut)
        self.img.setLevels([0,z.max()])
        if self.switch_TRPES_axis==False:
            self.p1.setLabel('left', "energy", units='eV')
            self.p1.setLabel('bottom', "time", units='ps')
        else:
            self.p1.setLabel('left', "time", units='ps')
            self.p1.setLabel('bottom', "energy", units='eV')
        

        # Contrast/color control
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img)
        viewer.addItem(hist)

        # Draggable line for setting isocurve level
        isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
        hist.vb.addItem(isoLine)
        hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
        isoLine.setValue(0.)
        isoLine.setZValue(z.max()) # bring iso line above contrast controls
        hist.gradient.setColorMap(map)

        self.img.setLookupTable(lut)
        self.img.setLevels([0,z.max()])

        # set position and scale of image
        size=np.abs(x3[-1]-x3[0])
        x_factor=size/x3.shape[0]
 
        size_y=np.abs(y3[-1]-y3[0])
        y_factor=size_y/y3.shape[0]
        self.img.translate(x3.min(), y3.min())
        self.img.scale(x_factor, y_factor) 
        self.p1.setXRange(x3[0],x3[-1])
        self.p1.setYRange(y3[0],y3[-1])   

    def find_nearest_index(self,array,value):
        return (np.abs(array - value)).argmin()
    
    @pyqtSlot('bool')
    def on_checkBox_subtract_background_toggled(self):
        self.subtract_bg()

    @pyqtSlot('double')
    def on_doubleSpinBox_bg_subtract_from_valueChanged(self,value):
        self.subtract_bg()

    @pyqtSlot('double')
    def on_doubleSpinBox_bg_subtract_to_valueChanged(self,value):
        self.subtract_bg()

    def subtract_bg(self):
        #print('substracting background')
        if self.checkBox_subtract_background.isChecked()==True:
            #get the limits of the time (x side)
            x_min_ind=self.find_nearest_index(self.x, self.doubleSpinBox_bg_subtract_from.value())
            x_max_ind=self.find_nearest_index(self.x, self.doubleSpinBox_bg_subtract_to.value())
            if x_min_ind > x_max_ind:
                to_subtract=np.mean(self.original_z[x_max_ind:x_min_ind+1,:],axis=0)
            elif x_min_ind<x_max_ind:
                #print(self.original_z[x_m_ind:x_min_ind,:].shape)
                to_subtract=np.mean(self.original_z[x_min_ind:x_max_ind+1,:],axis=0)
            elif x_min_ind==x_max_ind:
                to_subtract=self.original_z[x_max_ind,:]
            print(self.original_z.shape,to_subtract.shape)
            self.z=self.original_z-to_subtract[None,:]
        else:
            self.z=copy.deepcopy(self.original_z)
        self.plot_trpes()
        if self.calibrated==True:
            self.plot_calibrated_trpes()
        #self.update_limits()
        self.add_roi()
        self.plot_extracted_roi()
        

    @pyqtSlot('bool')
    def on_checkBox_sum_y_toggled(self):
        self.change_integration_direction()

    def change_integration_direction(self):
        print('change integration direction')
        self.plot_extracted_roi()

    @pyqtSlot('double')
    def on_doubleSpinBox_x_lower_valueChanged(self,value):
        self.change_selection('value')

    @pyqtSlot('double')
    def on_doubleSpinBox_x_upper_valueChanged(self,value):
        self.change_selection('value')

    @pyqtSlot('double')
    def on_doubleSpinBox_y_lower_valueChanged(self,value):
        self.change_selection('value')

    @pyqtSlot('double')
    def on_doubleSpinBox_y_upper_valueChanged(self,value):
        self.change_selection('value')
    def selection_changed_box(self):
        self.change_selection('box')
    def change_selection(self,what):
        #print('change selection to integrate')
        #check whether the selection was moved outside of the image
            
        if what=='box':
            #get current limits of the box            
            bounds=self.roi.parentBounds()
            coords=bounds.getCoords() #transform from QRectF to an array
            coords=list(coords)
            #check whether the selection was moved outside of the image
            if self.calibrated==False:
                y_min=self.y.min()
                y_max=self.y.max()
            else:
                y_min=self.y_cal.min()
                y_max=self.y_cal.max()
            if self.switch_TRPES_axis==False:
                if coords[0]<self.x.min() or coords[1]<y_min or coords[2]>self.x.max() or coords[3]>y_max:
                    #okay, set it back so that it is inside the window
                    if coords[0]<self.x.min():
                        coords[0]=self.x.min()
                    if coords[1]<y_min:
                        coords[1]=y_min
                    if coords[3]>y_max:
                        coords[3]=y_max
                    if coords[2]>self.x.max():
                        coords[2]=self.x.max()
                    
            else:
                if coords[0]<y_min or coords[1]<self.x.min() or coords[2]>y_max or coords[3]>self.x.max():
                    #okay, set it back so that it is inside the window
                    if coords[0]<y_min:
                        coords[0]=y_min
                    if coords[1]<self.x.min():
                        coords[1]=self.x.min()
                    if coords[2]>y_max:
                        coords[2]=y_max
                    if coords[3]>self.x.max():
                        coords[3]=self.x.max()
            #update the rectangle
            self.roi.blockSignals(True)
            self.roi.setPos((coords[0],coords[1]))
            self.roi.setSize((coords[2]-coords[0],coords[3]-coords[1]))
            self.roi.blockSignals(False)
            #update the spinboxes
            for spinBox in [self.doubleSpinBox_x_lower,self.doubleSpinBox_x_upper,
                            self.doubleSpinBox_y_lower,self.doubleSpinBox_y_upper]:
                spinBox.blockSignals(True)
            self.doubleSpinBox_x_lower.setValue(coords[0])
            self.doubleSpinBox_y_lower.setValue(coords[1])
            #self.doubleSpinBox_x_upper.setValue(coords[0]+coords[2])
            #self.doubleSpinBox_y_upper.setValue(coords[1]+coords[3])
            self.doubleSpinBox_x_upper.setValue(coords[2])
            self.doubleSpinBox_y_upper.setValue(coords[3])           
            for spinBox in [self.doubleSpinBox_x_lower,self.doubleSpinBox_x_upper,
                            self.doubleSpinBox_y_lower,self.doubleSpinBox_y_upper]:
                spinBox.blockSignals(False)
        elif what=='value':
            self.roi.blockSignals(True)
            self.roi.setPos((self.doubleSpinBox_x_lower.value(),self.doubleSpinBox_y_lower.value()))
            self.roi.setSize((self.doubleSpinBox_x_upper.value()-self.doubleSpinBox_x_lower.value(),
                              self.doubleSpinBox_y_upper.value()-self.doubleSpinBox_y_lower.value()))
            self.roi.blockSignals(False)

                     
        self.plot_extracted_roi()

    def plot_extracted_roi(self):
        #print('plot_extracted_roi')
        #selected=self.roi.getArrayRegion(self.z,self.img) not using this one, as it uses interpolated data!
        if self.calibrated==False:
            y_data=self.y
        else:
            y_data=self.y_cal
        if self.switch_TRPES_axis==False:
            x_min=self.find_nearest_index(self.x,self.doubleSpinBox_x_lower.value())
            x_max=self.find_nearest_index(self.x,self.doubleSpinBox_x_upper.value())
            y_min=self.find_nearest_index(y_data,self.doubleSpinBox_y_lower.value())
            y_max=self.find_nearest_index(y_data,self.doubleSpinBox_y_upper.value())
        else:
            x_min=self.find_nearest_index(self.x,self.doubleSpinBox_y_lower.value())
            x_max=self.find_nearest_index(self.x,self.doubleSpinBox_y_upper.value())
            y_min=self.find_nearest_index(y_data,self.doubleSpinBox_x_lower.value())
            y_max=self.find_nearest_index(y_data,self.doubleSpinBox_x_upper.value())
        
        self.viewer_sum_original.setLabel('left', "intensity", units='arb.u.')
        self.viewer_sum_original.plotItem.ctrlMenu=[]
        #if self.switch_TRPES_axis==False:
        if (self.checkBox_sum_y.isChecked()==False and self.switch_TRPES_axis==False) or (self.checkBox_sum_y.isChecked()==True and self.switch_TRPES_axis==True):
            selection=self.z[x_min:x_max+1,:]
            #sum over the x side
            self.viewer_sum_original.setLabel('bottom', "energy", units='pixels')
            self.summed=np.mean(selection,axis=0)
            self.viewer_sum_original.clear()
            self.viewer_sum_original.plot(y_data,self.summed/self.summed.max(),pen=pg.mkPen(self.colors[0],width=2))
            self.viewer_sum_original.setXRange(y_data.min(),y_data.max())
            self.summed_sigmas=self.get_sigmas_roi('x',x_min,x_max)  
            item1=self.viewer_sum_original.plot(y_data,self.summed/self.summed.max()+self.summed_sigmas/self.summed.max(),brush=None,pen=(50,50,200,0))
            item2=self.viewer_sum_original.plot(y_data,self.summed/self.summed.max()-self.summed_sigmas/self.summed.max(),brush=None,pen=(50,50,200,0))
            errors=pg.FillBetweenItem(curve1=item1, curve2=item2, brush=(self.colors[0][0],self.colors[0][1],self.colors[0][2],100), pen=None)
            self.viewer_sum_original.addItem(errors)
            for isoline in self.gaussians_isolines:
                isoline.setBounds((y_data.min(),y_data.max()))
        else:
            selection=self.z[:,y_min:y_max+1]
            #sum over the y side
            self.viewer_sum_original.setLabel('bottom', "time", units='ps')
            self.summed=np.mean(selection,axis=1)
            self.viewer_sum_original.clear()
            self.viewer_sum_original.plot(self.x,self.summed/self.summed.max(),
                                pen=pg.mkPen(self.colors[0],width=2))
            self.viewer_sum_original.setXRange(self.x.min(),self.x.max())
            self.summed_sigmas=self.get_sigmas_roi('y',y_min,y_max)/self.summed.max()   
            item1=self.viewer_sum_original.plot(self.x,self.summed/self.summed.max()+self.summed_sigmas/self.summed.max() ,brush=None,pen=(50,50,200,0))
            item2=self.viewer_sum_original.plot(self.x,self.summed/self.summed.max()-self.summed_sigmas/self.summed.max() ,brush=None,pen=(50,50,200,0))
            errors=pg.FillBetweenItem(curve1=item1, curve2=item2, brush=(self.colors[0][0],self.colors[0][1],self.colors[0][2],100), pen=None)
            self.viewer_sum_original.addItem(errors)
            for isoline in self.gaussians_isolines:
                isoline.setBounds((self.x.min(),self.x.max()))
        self.plot_gaussians()
        self.plot_isolines()
    
    def get_sigmas_roi(self,direction,v_min,v_max):
        """
        get the sigmas
        """
        summed_sigmas=None
        if type(self.sigmas)==type(None):
            if direction=='x':
                #get original one
                selection=self.original_z[v_min:v_max+1,:]
                summed_sigmas=np.std(selection,axis=0)
                #background subtracted?
                print('jo')
                if self.checkBox_subtract_background.isChecked()==True:
                #get the limits of the time (x side)
                    x_min_ind=self.find_nearest_index(self.x, self.doubleSpinBox_bg_subtract_from.value())
                    x_max_ind=self.find_nearest_index(self.x, self.doubleSpinBox_bg_subtract_to.value())
                    selection2=self.original_z[x_min_ind:x_max_ind+1,:]
                    bg_sigmas=np.std(selection2,axis=0)
                    summed_sigmas=np.sqrt(np.power(summed_sigmas,2)+np.power(bg_sigmas,2))
            elif direction=='y':
                selection=self.original_z[:,v_min:v_max+1]
                summed_sigmas=np.std(selection,axis=1)
                if self.checkBox_subtract_background.isChecked()==True:
                #get the limits of the time (x side)
                    x_min_ind=self.find_nearest_index(self.x, self.doubleSpinBox_bg_subtract_from.value())
                    x_max_ind=self.find_nearest_index(self.x, self.doubleSpinBox_bg_subtract_to.value())
                    selection2=self.original_z[:,x_min_ind:x_max_ind+1]
                    bg_sigmas=np.std(selection2,axis=1)
                    summed_sigmas=np.sqrt(np.power(summed_sigmas,2)+np.power(bg_sigmas,2))
                
        else:
            if direction=='x':
                selection=self.sigmas[v_min:v_max+1,:]
                #selection_z=self.original_z[v_min:v_max+1,:]
                dividor=v_max-v_min
                if dividor==0:
                    dividor=1
                if self.checkBox_subtract_background.isChecked()==True:
                    #subtract bg
                    x_min_ind=self.find_nearest_index(self.x, self.doubleSpinBox_bg_subtract_from.value())
                    x_max_ind=self.find_nearest_index(self.x, self.doubleSpinBox_bg_subtract_to.value())
                    x_dividor=x_max_ind-x_min_ind
                    if x_dividor==0:
                        x_dividor=1
                    selection2_bg=self.sigmas[x_min_ind:x_max_ind+1,:]
                    summed_sigmas_bg=np.sqrt(np.sum(np.power(selection2_bg,2),axis=0))
                    summed_sigmas_bg=summed_sigmas_bg/x_dividor
                    selection=np.power(selection,2)
                    selection+=np.power(summed_sigmas_bg,2)*np.ones(selection.shape) #error here! Figure that out tomorrow!
                    selection=np.sqrt(selection)
                    #get final
                    #z_sum=np.mean(selection_z,axis=0)
                    summed_sigmas=np.sqrt(np.sum(np.power(selection,2),axis=0))
                    summed_sigmas=summed_sigmas/(dividor)
                else:
                    #z_sum=np.mean(selection_z,axis=0)
                    summed_sigmas=np.sqrt(np.sum(np.power(selection,2),axis=0))

                    #summed_sigmas=np.divide(summed_sigmas/(dividor),z_sum)
                    summed_sigmas=summed_sigmas/(dividor)
        
        return summed_sigmas
            
        
    def initialize_gaussians(self):
        gaussian=Gaussian(self.doubleSpinBox_origin.value(),self.doubleSpinBox_fwhm.value())
        self.gaussians.append(gaussian)
        isoLine=pg.InfiniteLine(angle=90, movable=True, pen=self.colors[1])
        isoLine.sigPositionChangeFinished.connect(self.movedIsolines)
        self.gaussians_isolines.append(isoLine)
        isoLine.setZValue(1000)
        self.comboBox_Gaussians.blockSignals(True)
        self.comboBox_Gaussians.addItem('Gaussian '+str(1))
        self.comboBox_Gaussians.blockSignals(False)

    def movedIsolines(self): 
        #first update the Gaussians themselves
        for i,isoLine in enumerate(self.gaussians_isolines):
            self.gaussians[i].setPos(isoLine.value())
        #update the shown doubleSpinBox_origin
        self.doubleSpinBox_origin.blockSignals(True)
        self.doubleSpinBox_origin.setValue(self.gaussians[self.comboBox_Gaussians.currentIndex()].getPos())
        self.doubleSpinBox_origin.blockSignals(False)
        #update the graph
        self.plot_extracted_roi()

        
    def plot_isolines(self):        
        for i,isoLine in enumerate(self.gaussians_isolines):
            self.viewer_sum_original.addItem(isoLine)
            isoLine.setValue(self.gaussians[i].getPos())
            isoLine.setZValue(1000)

    def plot_gaussians(self):
        #get current limits in the viewer
        if (self.checkBox_sum_y.isChecked()==False and self.switch_TRPES_axis==False) or (self.checkBox_sum_y.isChecked()==True and self.switch_TRPES_axis==True):
            if self.calibrated==False:
                low=self.y.min()
                upper=self.y.max()
            else:
                low=self.y_cal.min()
                upper=self.y_cal.max()
            x=np.arange(low,upper, (upper-low)/200)
        else:
            low=self.x.min()
            upper=self.x.max()
            x=np.arange(low,upper, (upper-low)/200)
        y_sum=np.zeros(x.shape)
        for i,gauss in enumerate(self.gaussians):
            if self.comboBox_type_gaussian.currentIndex()==0:
                y=gauss.get_gaussian(x)
            elif self.comboBox_type_gaussian.currentIndex()==1:
                g_interm=Gaussian_energy(gauss.getPos(),gauss.getFwhm())
                y=g_interm.get_gaussian(x)
            elif self.comboBox_type_gaussian.currentIndex()==2:
                y=gauss.get_gaussian(x)
            if i==(self.comboBox_Gaussians.currentIndex()):
                self.viewer_sum_original.plot(x,y/y.max(),pen=pg.mkPen(self.colors[i+1],width=4))
            else:
                self.viewer_sum_original.plot(x,y/y.max(),pen=pg.mkPen(self.colors[i+1],width=2))
            y_sum+=y
        #self.viewer_sum_original.plot(x,y_sum/y_sum.max(),pen=pg.mkPen(self.colors[i+2],width=2))

    @pyqtSlot('int')
    def on_comboBox_type_gaussian_currentIndexChanged(self,value):
        if value==2:
            self.label_range.show()
            self.doubleSpinBox_range.show()
        else:
            self.label_range.hide()
            self.doubleSpinBox_range.hide()
        self.plot_extracted_roi()

    @pyqtSlot('int')
    def on_spinBox_number_of_gaussians_valueChanged(self,value):
        #print('change number of gaussians')
        if value>len(self.gaussians):
            self.comboBox_Gaussians.addItem('Gaussian '+str(value))
            gaussian=Gaussian(self.doubleSpinBox_origin.value(),self.doubleSpinBox_fwhm.value())
            self.gaussians.append(gaussian)
            isoLine=pg.InfiniteLine(angle=90, movable=True, pen=self.colors[value])
            isoLine.sigPositionChangeFinished.connect(self.movedIsolines)
            self.gaussians_isolines.append(isoLine)
        else:
            self.comboBox_Gaussians.removeItem(len(self.gaussians)-1)
            self.gaussians.pop()
            self.gaussians_isolines.pop()
        self.plot_extracted_roi()
    @pyqtSlot('int')
    def on_comboBox_Gaussians_currentIndexChanged(self,value):
        #print('select different gaussian')   
        self.doubleSpinBox_origin.blockSignals(True)
        self.doubleSpinBox_fwhm.blockSignals(True)
        self.doubleSpinBox_origin.setValue(self.gaussians[self.comboBox_Gaussians.currentIndex()].getPos())
        self.doubleSpinBox_fwhm.setValue(self.gaussians[self.comboBox_Gaussians.currentIndex()].getFwhm())
        self.doubleSpinBox_origin.blockSignals(False)
        self.doubleSpinBox_fwhm.blockSignals(False)
        self.plot_extracted_roi()
        
    @pyqtSlot('double')
    def on_doubleSpinBox_origin_valueChanged(self,value):
        self.gaussians[self.comboBox_Gaussians.currentIndex()].setPos(value)
        self.plot_extracted_roi()
            
    @pyqtSlot('double')
    def on_doubleSpinBox_fwhm_valueChanged(self,value):
        self.gaussians[self.comboBox_Gaussians.currentIndex()].setFwhm(value)
        self.plot_extracted_roi()

    def gauss(self, x,x0,FWHM,a=1):
        sigma = FWHM/2.355
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    def mike_fit(self,gauss_values,x,summed,fit_range):
        """
        fits each gaussian seperatly, assuming no overlap between them. 
        Cuts the data off left/right for that
        """
        #fit range fixed to 3 for now, should change to user input too.
        fit_range=fit_range
        plsq=[]
        summed_fit=np.zeros(x.shape)
        for i in range(int(len(gauss_values)/3)):
            ke_min = gauss_values[i*3] - fit_range
            ke_max = gauss_values[i*3] + fit_range
            ke0 = (np.abs(x - ke_min)).argmin()
            ke1 = (np.abs(x - ke_max)).argmin()
            try:
                popt, pcov = curve_fit(self.gauss,x[ke0:ke1],summed[ke0:ke1],p0=[gauss_values[i*3],
                                               gauss_values[i*3+1],gauss_values[i*3+2]],maxfev=20000)
            except RuntimeError:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)

                msg.setText("Poor starting values!")
                msg.setInformativeText("Fit did not converge")
                msg.setWindowTitle("Warning")
                #msg.setDetailedText("Fit did not converge")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                popt=[0.,1.,0.]
            print('popt',popt)
            summed_fit+=self.gauss(x,popt[0],popt[1],popt[2])
            for p in popt:
                plsq.append(p)
        return plsq,summed_fit
        
    @pyqtSlot()
    def on_pushButton_fit_clicked(self):
        print('fitting!')
        test_fit=fits()
        if (self.checkBox_sum_y.isChecked()==False and self.switch_TRPES_axis==False) or (self.checkBox_sum_y.isChecked()==True and self.switch_TRPES_axis==True):
            if self.calibrated==False:
                x=self.y
            else:
                x=self.y_cal
        else:
            x=self.x
        gauss_values=[]
        for gauss in self.gaussians:
            pos,fwhm,intensity=gauss.return_values()
            gauss_values.append(pos)
            gauss_values.append(fwhm)
            gauss_values.append(intensity)
        if self.comboBox_type_gaussian.currentIndex()!=2:
            summed_sigmas=copy.deepcopy(self.summed_sigmas)
            if type(summed_sigmas)!=type(None):
                for i, value in enumerate(summed_sigmas):
                    if value==0:
                        summed_sigmas[i]=1.
            plsq,cov,info,msg,ier=leastsq(test_fit.fit_function,gauss_values,
                              args=(x,self.summed,self.comboBox_type_gaussian.currentIndex(),summed_sigmas),full_output=True)
            self.summed_fit=test_fit.return_fitted(plsq,x,self.comboBox_type_gaussian.currentIndex())
            sigmas=test_fit.get_deltas(plsq,cov,x,self.summed,self.comboBox_type_gaussian.currentIndex())
            print('cov',cov)
            print('sigmas',sigmas)
        else:
            #fit using mike's strategy
            plsq,self.summed_fit=self.mike_fit(gauss_values,x,self.summed,self.doubleSpinBox_range.value())
            sigmas=['None']*len(plsq)
            
        print('successful fit',plsq)
        
        self.fit_x=x
        print('type self.fit_x',type(self.fit_x))
        if self.checkBox_sum_y.isChecked()==False:
            self.unit_fit='eV'
        else:
            self.unit_fit='ps'
    
        self.fitted_values=plsq
        self.fitted_sigmas=sigmas
        print('jo')
        self.plot_fit()

    def plot_fit(self):
        #plot the fit!
        print('plotting the fit')
        self.viewer_sum_fitted.clear()
        #if self.viewer_sum_fitted_legend==[]:
        #    self.viewer_sum_fitted_legend=self.viewer_sum_fitted.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
        """
        if self.viewer_sum_fitted_legend==[]:
            self.viewer_sum_fitted_legend=self.viewer_sum_fitted.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
        else:
            self.viewer_sum_fitted_legend.scene().removeItem(self.viewer_sum_fitted_legend)
            self.viewer_sum_fitted_legend=self.viewer_sum_fitted.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
        """
        self.viewer_sum_fitted_legend=self.viewer_sum_fitted.addLegend(size=(0.2,0.2),offset=(-0.4,0.4))
        print('?')
        self.viewer_sum_fitted.setLabel('left', "intensity", units='au')
        if (self.checkBox_sum_y.isChecked()==False and self.switch_TRPES_axis==False) or (self.checkBox_sum_y.isChecked()==True and self.switch_TRPES_axis==True):
            self.viewer_sum_fitted.plot(self.y,self.summed, name='original',
                                    pen=pg.mkPen(self.colors[0],width=2))
            item1=self.viewer_sum_fitted.plot(self.y,self.summed+self.summed_sigmas,brush=None,pen=(50,50,200,0))
            item2=self.viewer_sum_fitted.plot(self.y,self.summed-self.summed_sigmas,brush=None,pen=(50,50,200,0))
            errors=pg.FillBetweenItem(curve1=item1, curve2=item2, brush=(self.colors[0][0],self.colors[0][1],self.colors[0][2],100), pen=None)
            self.viewer_sum_fitted.addItem(errors)
            self.viewer_sum_fitted.setLabel('bottom', "energy", units='pixels')
            self.viewer_sum_fitted.plot(self.y,self.summed_fit, name='fitted',
                                    pen=pg.mkPen(self.colors[1],width=2))
            unit=' eV'
            if self.calibrated==False:
                x=copy.deepcopy(self.y)
            else:
                x=copy.deepcopy(self.y_cal)
        else:
            self.viewer_sum_fitted.plot(self.x,self.summed, name='original',
                                    pen=pg.mkPen(self.colors[0],width=2))
            item1=self.viewer_sum_fitted.plot(self.x,self.summed+self.summed_sigmas ,brush=None,pen=(50,50,200,0))
            item2=self.viewer_sum_fitted.plot(self.x,self.summed-self.summed_sigmas ,brush=None,pen=(50,50,200,0))
            errors=pg.FillBetweenItem(curve1=item1, curve2=item2, brush=(self.colors[0][0],self.colors[0][1],self.colors[0][2],100), pen=None)
            self.viewer_sum_fitted.addItem(errors)
            self.viewer_sum_fitted.setLabel('bottom', "time", units='ps')
            self.viewer_sum_fitted.plot(self.x,self.summed_fit, name='fitted',
                                    pen=pg.mkPen(self.colors[1],width=2))
            unit=' ps'
            x=copy.deepcopy(self.x)
        i=0
        g=2
        for n in range(int(len(self.fitted_values)/3)): 
            label='pos='+'{:.5}'.format(self.fitted_values[i])+'\xb1'+'{:.5}'.format(self.fitted_sigmas[i])+unit+'<br>'
            label+='fwhm='+'{:.5}'.format(self.fitted_values[i+1])
            label+='\xb1'+'{:.5}'.format(self.fitted_sigmas[i+1])+unit+'<br>'
            label+=' int='+'{:.2f}'.format(self.fitted_values[i+2])+'\xb1'+'{:.2f}'.format(self.fitted_sigmas[i+2])
            if self.comboBox_type_gaussian.currentIndex()==0:
                gauss=Gaussian(self.fitted_values[i],self.fitted_values[i+1],self.fitted_values[i+2])
                self.viewer_sum_fitted.plot(x,gauss.get_gaussian(x), name=label,
                                        pen=pg.mkPen(self.colors[g],width=2))
                gauss_min=Gaussian(self.fitted_values[i]-self.fitted_sigmas[i],
                                   self.fitted_values[i+1]-self.fitted_sigmas[i+1],
                                   self.fitted_values[i+2]-self.fitted_sigmas[i+2])
                gauss_max=Gaussian(self.fitted_values[i]+self.fitted_sigmas[i],
                                   self.fitted_values[i+1]+self.fitted_sigmas[i+1],
                                   self.fitted_values[i+2]+self.fitted_sigmas[i+2])
                item1=self.viewer_sum_fitted.plot(x,gauss_max.get_gaussian(x) ,brush=None,pen=(50,50,200,0))
                item2=self.viewer_sum_fitted.plot(x,gauss_min.get_gaussian(x) ,brush=None,pen=(50,50,200,0))
                errors=pg.FillBetweenItem(curve1=item1, curve2=item2, brush=(self.colors[g][0],self.colors[g][1],self.colors[g][2],100), pen=None)
                self.viewer_sum_fitted.addItem(errors)
            elif self.comboBox_type_gaussian.currentIndex()==1:               
                gauss=Gaussian_energy(self.fitted_values[i],self.fitted_values[i+1],self.fitted_values[i+2])
                self.viewer_sum_fitted.plot(x,gauss.get_gaussian(x), name=label,
                                        pen=pg.mkPen(self.colors[g],width=2))
            if self.comboBox_type_gaussian.currentIndex()==2:
                gauss=Gaussian(self.fitted_values[i],self.fitted_values[i+1],self.fitted_values[i+2])
                self.viewer_sum_fitted.plot(x,gauss.get_gaussian(x), name=label,
                                        pen=pg.mkPen(self.colors[g],width=2)) 
            g+=1
            i+=3
        print("plotted it",self.fitted_values)
        print("uncertainties",self.fitted_sigmas)
            
        

    @pyqtSlot('int')
    def on_spinBox_steps_valueChanged(self,value):
        print('number of fitting steps changed')
        

    @pyqtSlot()
    def on_pushButton_save_fit_clicked(self):
        print('save fit')
        #save the original, the fitted total, and the individual gaussians
        #put the data in the header
        #make header:
        if self.fit_x!=[]:
            try:
                filename=QFileDialog.getSaveFileName(self, 'Save File',self.dir)[0]
                self.dir=filename[:(-len(filename.split('/')[-1]))]
                data=np.zeros((len(self.fit_x),3+int(len(self.fitted_values)/3)))
                print('2',type(self.fit_x),data.shape,self.fit_x.shape)
                data[:,0]=self.fit_x
                data[:,1]=self.summed
                data[:,2]=self.summed_fit
                header='Fit with FitGaussian\ncolumn0: x values in'+self.unit_fit
                header+='\ncolumn1: original values\ncolumn2:fitted values'
                i=0
                g=3
                for n in range(int(len(self.fitted_values)/3)):          
                    gauss=Gaussian(self.fitted_values[i],self.fitted_values[i+1],self.fitted_values[i+2])
                    label='\ncolumn'+str(g)+':pos='+str(self.fitted_values[i])+self.unit_fit+'fwhm='+str(self.fitted_values[i+1])+self.unit_fit+'\int='+str(self.fitted_values[i+2])
                    gauss_data=gauss.get_gaussian(self.fit_x)
                    data[:,n+3]=gauss_data
                    header+=label
                    i+=3
                    g+=1
                    
                np.savetxt(filename,data,header=header)
            except:
                print('select valid file')
        
    @pyqtSlot()
    def on_pushButton_saveCalibration_file_clicked(self):
        print('save the fitted value in a calibration file, andrey style!')
        if self.fit_x!=[]:
            if int(len(self.fitted_values)/3)==2:
                data='%r1,r2,dKE\n'
                if self.fitted_values[0]<self.fitted_values[3]:
                    data+=str(self.fitted_values[0])+'\n'
                    data+=str(self.fitted_values[3])+'\n'
                else:
                    data+=str(self.fitted_values[3])+'\n'
                    data+=str(self.fitted_values[0])+'\n'
                data+=str(1.3064)+'  % spin-orbit splitting in Xenon = 10537.01/8065.6\n'
                data+='%% calibration is made from xenon taken on'+self.filename+'\n'
                filename=QFileDialog.getSaveFileName(self, 'Save File',self.dir)[0]
                self.dir=filename[:(-len(filename.split('/')[-1]))]
                f=open(filename,'w+')
                f.write(data)
                f.close()
                
    @pyqtSlot()
    def on_pushButton_plot_calibrated_clicked(self):
        print('calibrating the one shown in original with a known calibration file')
        try:
            if type(self.z)!=type([]):
                filename = QFileDialog.getOpenFileName(self, 'Open File',self.dir)[0]
                self.dir=filename[:(-len(filename.split('/')[-1]))]
                #get the three constants out of the file
                f=open(filename,'r')
                lines=f.readlines()
                pixel1=float(lines[1])
                pixel2=float(lines[2])
                splitting=float(lines[3].split('%')[0])
                f.close()
                print(pixel1,pixel2,splitting)
                K=splitting/(pixel2**2-pixel1**2)
                print('h')
                #okay, now let's calibrate the whole thing!
                self.y_cal=K*np.square(self.y)
                print("K=",K)
                print('g')
                self.plot_calibrated_trpes()
                self.update_limits()
                self.add_roi()
                self.plot_extracted_roi()
        except:
            print('Format of the calibration file is wrong!')
            

    def plot_calibrated_trpes(self):
        """
        function to plot the TRPES on an appropriate viewer, need to implement it once I know how to properly display such stuff
        viewer: the plotwidget
        z: np 2D array of dimensions [M,N]
        y: 1D array of dimensions N
        x: 1D array of dimensions M
        """
        '''
        self.D2Trpes = pg.ImageView()
        self.D2Trpes.setImage(z)
        viewer.addItem(self.D2Trpes)
        '''
        print('plotting calibrated trpes')
        viewer=self.viewer_global_orig_TRPES_calibrated
        z=self.z
        x=self.x
        y=self.y_cal
        try:
            if self.display_log_t==True:
                f = interpolate.interp2d(x, y, z.transpose(), kind='linear')
                xi1=-np.flip(np.logspace(np.log10(0.1),np.log10(-x.min()),250,endpoint=True),axis=0)
                xi2=np.logspace(np.log10(0.1),np.log10(x.max()),250,endpoint=True)
                xi=np.concatenate([xi1,xi2],axis=0)
                yi=np.linspace(y.min(),y.max()+np.abs(y[0]-y[1]),500)
                print('shapes',xi.shape,yi.shape)
                z2=f(xi,yi).transpose()
                if self.switch_TRPES_axis==True:
                    y3=xi
                    x3=yi
                    z2=z2.transpose()
                else:
                    y3=yi
                    x3=xi
                #alright, clear the viewer and add that stuff
                viewer.clear()
                self.p1 = viewer.addPlot()
                self.filterMenu = QtGui.QMenu("Logharithmic Scale")
                self.change_to_logharithmic = QtGui.QAction("Change x-axis to logarithmic", self.filterMenu,checkable=True,checked=self.display_log_t)
                self.change_to_logharithmic.triggered.connect(lambda: self.change_viewer_scale_to_log(self.change_to_logharithmic.isChecked()))
                self.filterMenu.addAction(self.change_to_logharithmic)
                self.p1.ctrlMenu=[self.filterMenu]
                #add the colormap
                colormap = cm.get_cmap("nipy_spectral")  # cm.get_cmap("CMRmap")
                #color=np.array([[0,0,0,255],[255,128,0,255],[255,255,0,255]],dtype=np.ubyte)
                colormap._init()
                color=np.array((colormap._lut * 255).view(np.ndarray)[:-4,:],dtype=np.ubyte)
                pos=np.array(np.arange(0.,1.,1./color.shape[0]))
                map=pg.ColorMap(pos,color)
                lut=map.getLookupTable(0.,1.,256)
                # Item for displaying image data
                self.img = pg.ImageItem()
                self.p1.addItem(self.img)
                self.img.setImage(z2)
                self.img.setLookupTable(lut)
                self.img.setLevels([0,z2.max()])
                if self.switch_TRPES_axis==False:
                    self.p1.setLabel('left', "energy", units='eV')
                    self.p1.setLabel('bottom', "log(time)", units='ps')
                else:
                    self.p1.setLabel('left', "log(time)", units='ps')
                    self.p1.setLabel('bottom', "energy", units='eV')
                    
                     # Contrast/color control
                hist = pg.HistogramLUTItem()
                hist.setImageItem(self.img)
                viewer.addItem(hist)
        
                # Draggable line for setting isocurve level
                isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
                hist.vb.addItem(isoLine)
                hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
                isoLine.setValue(0.)
                isoLine.setZValue(z2.max()) # bring iso line above contrast controls
                hist.gradient.setColorMap(map)
        
                self.img.setLookupTable(lut)
                self.img.setLevels([0,z2.max()])
        
                if self.switch_TRPES_axis==False:
                    size_y=np.abs(y3[-1]-y3[0])
                    y_factor=size_y/y3.shape[0]
                    self.img.translate(0., y3.min())
                    self.img.scale(1., y_factor) 
                    
                    
                    xaxis=self.p1.getAxis('bottom')
        
                    values=[]
                    strings=[]
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((-10**(i+1)))
                                strings.append('-10^'+str(i+1))
                            else:
                                values.append(-j*(10**i))
                                strings.append('')
                    values.append(0)
                    strings.append('0')
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((10**(i+1)))
                                strings.append('10^'+str(i+1))
                            else:
                                values.append(j*(10**i))
                                strings.append('')
                    ticks=[list(zip(values,strings))]
                    #xaxis.setTicks(ticks)    
        
                    #second try:
                    f2=interpolate.interp1d(x3,range(len(x3)),  fill_value='extrapolate') 
                    y4=f2(np.array(values))
                    y4=list(y4)
                    ticks=[list(zip(y4,strings))]
                    xaxis.setTicks(ticks) 
                else:
                    size=np.abs(x3[-1]-x3[0])
                    x_factor=size/x3.shape[0]
                    self.img.translate(x3.min(),0.)
                    self.img.scale(x_factor,1) 
                    
                    
                    xaxis=self.p1.getAxis('left')
        
                    values=[]
                    strings=[]
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((-10**(i+1)))
                                strings.append('-10^'+str(i+1))
                            else:
                                values.append(-j*(10**i))
                                strings.append('')
                    values.append(0)
                    strings.append('0')
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((10**(i+1)))
                                strings.append('10^'+str(i+1))
                            else:
                                values.append(j*(10**i))
                                strings.append('')
                    ticks=[list(zip(values,strings))]
                    #xaxis.setTicks(ticks)    
        
                    #second try:
                    f2=interpolate.interp1d(y3,range(len(y3)),  fill_value='extrapolate') 
                    y4=f2(np.array(values))
                    y4=list(y4)
                    ticks=[list(zip(y4,strings))]
                    xaxis.setTicks(ticks)
    
            else:
                self.plot_TRPES_normally(viewer, z,x,y)
        except AttributeError:
            self.plot_TRPES_normally(viewer, z,x,y)

    @pyqtSlot()
    def on_pushButton_save_trpes_clicked(self):
        try:
            filename=QFileDialog.getSaveFileName(self, 'Save File',self.dir)[0]
            self.dir=filename[:(-len(filename.split('/')[-1]))]
            to_save=np.zeros((self.z.shape[0]+1,self.z.shape[1]+1))
            to_save[1:,1:]=self.z
            if self.calibrated==False:
                to_save[0,1:]=self.y
            else:
                to_save[0,1:]=self.y_cal
            to_save[1:,0]=self.x
            filename+='.csv'
            np.savetxt(filename,to_save,delimiter=',')
        except:
            print('problem saving (function not extensively tested')

            
        









if __name__ == '__main__':
    from PyQt5 import QtWidgets
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance() 
    main = Gfit()
    main.show()
    sys.exit(app.exec_())
