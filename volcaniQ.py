from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Fluids(object):
    """
    Fluids class for accessing fluid properties from a csv file

    _extended_summary_

    Args:
        object (_type_): _description_
    """
    def __init__(self, fluid_data_file: str = None):
        if fluid_data_file is None:
            fluid_data_file = 'fluid_data.csv'
        self.dataset = pd.read_csv(fluid_data_file)
        self._current_fluid = None
        self._temp_df = None

    @property
    def temp_df(self):
        return self._temp_df
    
    @temp_df.setter
    def temp_df(self, df: pd.DataFrame):
        self._temp_df = df

    @property
    def current_fluid(self):
        return self._current_fluid
    
    @current_fluid.setter
    def current_fluid(self, fluid_name: str):
        self._current_fluid = fluid_name
        
    @property
    def modulus(self):
        return self.temp_df["Modulus(GPa)"]
    
    @property
    def density(self):
        return self.temp_df["Density(g/cm3)"]
    
    @property
    def viscosity(self):
        return self.temp_df["Viscosity(Pa.ms)"]
    
    @property
    def depth(self):
        return self.temp_df["Depth(Km)"]
    
    def __call__(self, fluid_name: str) -> Fluids:
        assert fluid_name in self.dataset["Fluid"].values, f"Fluid not in database, use one of {self.dataset['Fluid'].values}"
        self.current_fluid = fluid_name
        self.temp_df = self.dataset[(self.dataset['Fluid']==self.current_fluid)]
        return self

class CleanReadCSV(object):
    def __init__(self, data_path: str = None):

        # load data from data path or user-specified directory
        if data_path is None:
            self.data_path = 'attenuation_and_moduli_3D_grid_centre_uturuncu.csv'
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
        self._data["K"] = self._data["K"].apply(lambda x: round(x/10**9,2))
        self._data["μ"] = self._data["μ"].apply(lambda x: round(x/10**9,2))
        self._data["ρ"] = self._data["ρ"].apply(lambda x: round(x/10**3,2))

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
        fig, ax = plt.subplots(1,1)
        plt.figure(figsize=(len(z)/50,len(l)/50))
        ax.set_aspect('auto', 'box')
        ax.set_xlabel(plotdir +' (km)')
        ax.set_ylabel('Depth (km)')
        hx = ax.hexbin(l, z, C, gridsize=grid, cmap='seismic', **kwargs)
        cb = fig.colorbar(hx, ax = ax)
        cb.set_label(plot)
        plt.show()
        plt.close()
        
class EffectiveFluid(object):
    def __init__(self, fluid_1:str, fluid_2:str):
        self.fluids = Fluids()
        try:
            self.fluids(fluid_1)
        except:
            raise ValueError(f"Fluid {fluid_1} not in database")
        try:
            self.fluids(fluid_2)
        except:
            raise ValueError(f"Fluid {fluid_2} not in database")
        
        self.mod1 = self.fluids(fluid_1).modulus
        self.fluid2 = self.fluids(fluid_2).modulus
        self.saturation = None
        self.patch_parameter = None
        self.reference_frequency = None
        
    def _effective_fluid_modulus(self, fluid1_modulus: float, fluid2_modulus: float, saturation: float=1., patch_parameter:float=1.):
        s, q = saturation, patch_parameter
        k1, k2 = fluid1_modulus, fluid2_modulus
        keff = (s*k1 + q * (1-s) * k2) / (s + q * (1-s))
        return keff
    
    def _effective_fluid_tau(self, fluid1_viscosity: float, fluid2_viscosity: float, saturation: float=1., patch_parameter:float=1., reference_frequency: float=1.):
        s, q = saturation, patch_parameter
        def BrooksCorey(s_wetting, pore_lambda):
            s = s_wetting
            return s**((2+3*pore_lambda)/pore_lambda), (1-s)**2 * (1-s**((2+pore_lambda)/pore_lambda ))
        eta1, eta2 = fluid1_viscosity, fluid2_viscosity
        k1, k2 = BrooksCorey(s, q)
        m1, m2 = k1/eta1, k2/eta2
        eta_eff = reference_frequency*eta1(s + q * (1-s))/(s * m1 + q * (1-s) * m2)
        return eta_eff
    
    def _effective_fluid_density(self, fluid1_density: float, fluid2_density: float, saturation: float=1.):
        s = saturation
        rho1, rho2 = fluid1_density, fluid2_density
        rho_eff = (s*rho1 +  (1-s) * rho2)
        return rho_eff
    

    

class RockPhysicsModel:
    def __init__(self, dry_modulus: float = None, shear_modulus: float = None, mineral_modulus: float = None, porosity: float = None):
        self._dry_modulus = dry_modulus
        self._shear_modulus = shear_modulus
        self._mineral_modulus = mineral_modulus
        self.L0 =  self.dry_modulus- 2 / 3 * self.shear_modulus
        self._porosity = porosity

    @classmethod
    def from_data(cls, minQp:float, minQs:float, meanK:float, meanmu:float, porosity:float, Kf:float):
        return cls(dry_modulus = minQp**2 * meanK, shear_modulus = minQs**2 * meanmu, mineral_modulus = meanK, porosity = porosity, fluid_modulus = Kf)

    @property
    def dry_modulus(self):
        return self._dry_modulus
    
    @dry_modulus.setter
    def dry_modulus(self, value):
        self._dry_modulus = value
        self.L0 =  self.dry_modulus- 2 / 3 * self.shear_modulus

    @property
    def shear_modulus(self):
        return self._shear_modulus
    
    @shear_modulus.setter
    def shear_modulus(self, value):
        self._shear_modulus = value

    @property
    def mineral_modulus(self):
        return self._mineral_modulus

    @mineral_modulus.setter
    def mineral_modulus(self, value):
        self._mineral_modulus = value
    
    @property
    def porosity(self):
        return self._porosity


    def gassmann_model(self, fluid_modulus:float = None):
        cij = np.zeros((6, 6))
        Km, Kd, mu, phi = (
                self.mineral_modulus,
                self.dry_modulus,
                self.shear_modulus,
                self.porosity,
            )
        Kf = fluid_modulus

        # Gassmann's model
        Pmod = Kd + (1 - Kd / Km) ** 2 / (phi / Kf - Kd / Km ** 2 + (1 - phi) / Km) + 4 / 3 * mu

        # Might as well define cij's in case this ever needs anisotropic modelling
        cij[0, 0] = cij[1, 1] = cij[2, 2] = Pmod
        cij[3, 3] = cij[4, 4] = cij[5, 5] = mu
        cij[0, 1] = cij[0, 2] = cij[2, 0] = cij[1, 0] = cij[1, 2] = cij[2, 1] = Pmod - 2 * mu
        
        return cij

    def squirt_flow_model(self, fluid_modulus:float = None, epsilon:float = None, omegac:float = 0.):
        cij = np.zeros((6, 6), dtype=np.cfloat)
        Kf = fluid_modulus
        l, m = self.L0, self.shear_modulus
        lamb = (16 * (15 * l * (-Kf + l) + 4 * (-3 * Kf + 5 * l) * m + 4 * m ** 2)) / (45.0 * (l + m) * (3 * Kf + 4 * m))
        mu = (16 * m) / (45.0 * (l + m))
        low_freq = self.gassmann_model(fluid_modulus = fluid_modulus)
        cij[0, 0] = cij[1, 1] = cij[2, 2] = lamb + 2 * mu
        cij[3, 3] = cij[4, 4] = cij[5, 5] = mu
        cij[0, 1] = cij[0, 2] = cij[2, 0] = cij[1, 0] = cij[1, 2] = cij[2, 1] = lamb
        return lambda omega:  low_freq + epsilon* (l + 2 * m)* (1j * (10 ** (omega-omegac))/ (1 + 1j * (10 ** (omega-omegac))* cij))

    def max_attenuation(self, fluid_modulus:float = None, epsilon:float =None):
        low_freq = self.gassmann_model(fluid_modulus = fluid_modulus)
        max_att = self.squirt_flow_model(fluid_modulus = fluid_modulus, epsilon=epsilon, tau = 0)
        qij = (max_att)/(2*low_freq + max_att)
        maxQp = qij[0, 0]
        maxQs = qij[3, 3]
        return maxQp, maxQs
    # def __call__(self, *args: Any, **kwds: Any) -> Any:
        

    def plot(self, fluid_modulus:float = None, epsilon:float =None, omegac:float = None):
        omega_axis = omegac + np.arange(-2, 2, .1)
        cij0 = self.gassmann_model(fluid_modulus = fluid_modulus)
        cij = self.squirt_flow_model(fluid_modulus = fluid_modulus, epsilon=epsilon, omegac = omegac)
        f_cij = np.array([cij(omega) for omega in omega_axis])
        fig, ax = plt.subplots(1,1)
        ax.axhline(cij0[0, 0], linestyle='--', color='k')
        ax.plot(omega_axis, np.real(f_cij[:,0,0]))
        plt.show()
        plt.close()

if __name__ == "__main__":
    rock = RockPhysicsModel(dry_modulus=30, shear_modulus=10, mineral_modulus=50, porosity=0.2, density=2.65)
    rock.plot(fluid_modulus=10, epsilon=0.1, omegac=0.)