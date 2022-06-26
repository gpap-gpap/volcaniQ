import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CleanReadCSV:
    def __init__(self, data_path: str = None):

        # load data from data path or user-specified directory
        if data_path is None:
            self.data_path = '../data/attenuation_and_moduli_3D_grid_centre_uturuncu.csv'
        else:
            self.data_path = data_path
        temp_df = pd.read_csv(self.data_path)
        self._data = pd.DataFrame([]) 

        # hard coded auxiliary functions and renaming of columns---->>>>
        
        # column had a comma and was interpreted as two columns, wrong column removed
        temp_columns = temp_df.keys().drop(labels = "bsl)").values 
        # naming columns with single symbols for ease of .sym calling
        temp_columns[0:3] = ["x","y","z"]
        temp_columns[6:9] = ["K", "μ", "ρ"]

        # setting column names in dataframe
        self._data[temp_columns] = temp_df.values[:,:9]

        # rounding to 2 dp and values of elastic constants in GPa and density in g/cm3
        self._data["K"]=self._data["K"].apply(lambda x: round(x/10**9,2))
        self._data["μ"]=self._data["μ"].apply(lambda x: round(x/10**9,2))
        self._data["ρ"]=self._data["ρ"].apply(lambda x: round(x/10**3,2))

        #setting limits of data for plotting
        self.min_x, self.max_x = self._data["x"].min(), self._data["x"].max()
        self.min_y, self.max_y = self._data["y"].min(), self._data["y"].max()
        
        # <<<---- end hard coded auxiliary functions and renaming of columns

    @property
    def data(self):
        return self._data

    def window_by_xy(self, x_range: int or list = None, y_range: int or list = None) -> pd.DataFrame:
        df =  self.data
        min_x, max_x = self.min_x, self.max_x
        min_y, max_y = self.min_y, self.max_y
        if type(x_range) is int and min_x <= x_range <= max_x:
            x_cond = (df["x"]==x_range)
        elif type(x_range) is list:
            min_x = max([min(x_range), min_x])
            max_x = min([max(x_range), max_x])
            x_cond  = df["x"].between(min_x, max_x)
        if type(y_range) is int and min_y <= y_range <= max_y:
            y_cond = (df["y"]==y_range)
        elif type(y_range) is list:
            min_y = max([min(y_range), min_y])
            max_y = min([max(y_range), max_y])
            y_cond = df["y"].between(min_y, max_y)
        return df[x_cond][y_cond]
    
    def existing_q(self):
        df =  self.data
        return df[df["Qp"].notnull()]
    
    def hexplot(self, plot:str=None, direction:str = None, value:float = 0., grid = 25, func= (lambda x: x), **kwargs):
        df =  self.data
        min_x, max_x = self.min_x, self.max_x
        min_y, max_y = self.min_y, self.max_y
        if direction == "x":
            plotdir = "y"
            assert min_x <= value <= max_x, "x out of bounds"
        elif direction == "y":
            plotdir = "x"
            assert min_y <= value <= max_y, "y out of bounds"
        else:
            print("generic and boring error")
        windowed = df[(df[direction]==value)]
        l = windowed[plotdir]
        z = -windowed.z
        C = func(windowed[plot])
        plt.xlabel(plotdir+' (km)')
        plt.ylabel('Depth (km)')
        hx = plt.hexbin(l, z, C, gridsize=grid, cmap='seismic', **kwargs)
        cb = plt.colorbar(hx)
        cb.set_label(plot)
        plt.show()
        plt.figure(figsize=(5*len(l), 5*len(z)))

class RockPhysicsModelling:
    def __init__(self, dry_modulus: float = None, shear_modulus: float = None, mineral_modulus: float = None, porosity: float = None, density : float = None):
        self._dry_modulus = dry_modulus
        self._shear_modulus = shear_modulus
        self._mineral_modulus = mineral_modulus
        self._porosity = porosity
        self._density = density

    @property
    def dry_modulus(self):
        return self._dry_modulus

    @property
    def shear_modulus(self):
        return self._shear_modulus

    @property
    def mineral_modulus(self):
        return self._mineral_modulus
    
    @property
    def porosity(self):
        return self._porosity

    @property
    def density(self):
        return self._density
    
    def gassmann_model(self, fluid_modulus:float = None, fluid_density: float = None):
        pass

    def squirt_flow_model(self):
        pass
