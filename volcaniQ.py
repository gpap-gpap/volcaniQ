from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


class FluidsReadCSV(object):
    """
    Fluids class for accessing fluid properties from a csv file

    _extended_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, fluid_data_file: str = None):
        if fluid_data_file is None:
            fluid_data_file = "fluid_data.csv"
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

    def __call__(self, fluid_name: str) -> FluidsReadCSV:
        assert (
            fluid_name in self.dataset["Fluid"].values
        ), f"Fluid not in database, use one of {self.dataset['Fluid'].values}"
        self.current_fluid = fluid_name
        self.temp_df = self.dataset[(self.dataset["Fluid"] == self.current_fluid)]
        return self


class CleanReadCSV(object):
    def __init__(self, data_path: str = None):
        # load data from data path or user-specified directory
        if data_path is None:
            self.data_path = "attenuation_and_moduli_3D_grid_centre_uturuncu.csv"
        else:
            self.data_path = data_path
        temp_df = pd.read_csv(self.data_path)
        self._data = pd.DataFrame([])

        # hard coded auxiliary functions and renaming of columns---->>>>

        # column had a comma and was interpreted as two columns, wrong column removed
        temp_columns = temp_df.keys().drop(labels="bsl)").values
        # naming columns with single symbols for ease of .sym calling
        temp_columns[0:3] = ["x", "y", "z"]
        temp_columns[6:9] = ["K", "μ", "ρ"]

        # setting column names in dataframe
        self._data[temp_columns] = temp_df.values[:, :9]

        # rounding to 2 dp and values of elastic constants in GPa and density in g/cm3
        self._data["K"] = self._data["K"].apply(lambda x: round(x / 10**9, 2))
        self._data["μ"] = self._data["μ"].apply(lambda x: round(x / 10**9, 2))
        self._data["ρ"] = self._data["ρ"].apply(lambda x: round(x / 10**3, 2))

        # setting limits of data for plotting
        self.min_x, self.max_x = self._data["x"].min(), self._data["x"].max()
        self.min_y, self.max_y = self._data["y"].min(), self._data["y"].max()

        # <<<---- end hard coded auxiliary functions and renaming of columns

    @property
    def data(self):
        return self._data

    def window_by_xy(
        self, x_range: int or list = None, y_range: int or list = None
    ) -> pd.DataFrame:
        df = self.data
        min_x, max_x = self.min_x, self.max_x
        min_y, max_y = self.min_y, self.max_y
        if type(x_range) is int and min_x <= x_range <= max_x:
            x_cond = df["x"] == x_range
        elif type(x_range) is list:
            min_x = max([min(x_range), min_x])
            max_x = min([max(x_range), max_x])
            x_cond = df["x"].between(min_x, max_x)
        if type(y_range) is int and min_y <= y_range <= max_y:
            y_cond = df["y"] == y_range
        elif type(y_range) is list:
            min_y = max([min(y_range), min_y])
            max_y = min([max(y_range), max_y])
            y_cond = df["y"].between(min_y, max_y)
        return df[x_cond][y_cond]

    def existing_q(self):
        df = self.data
        return df[df["Qp"].notnull()]

    def hexplot(
        self,
        plot: str = None,
        direction: str = None,
        value: float = 0.0,
        grid=25,
        func=(lambda x: x),
        **kwargs,
    ):
        df = self.data
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
        windowed = df[(df[direction] == value)]
        l = windowed[plotdir]
        z = -windowed.z
        C = func(windowed[plot])
        fig, ax = plt.subplots(1, 1)
        plt.figure(figsize=(len(z) / 50, len(l) / 50))
        ax.set_aspect("auto", "box")
        ax.set_xlabel(plotdir + " (km)")
        ax.set_ylabel("Depth (km)")
        hx = ax.hexbin(l, z, C, gridsize=grid, cmap="seismic", **kwargs)
        cb = fig.colorbar(hx, ax=ax)
        cb.set_label(plot)
        plt.show()
        plt.close()


class Fluid(object):
    def __init__(self, density: float, viscosity: float, modulus: float) -> None:
        self._density = density
        self._viscosity = viscosity
        self._modulus = modulus

    @property
    def density(self):
        return self._density

    @property
    def viscosity(self):
        return self._viscosity

    @property
    def modulus(self):
        return self._modulus

    def __call__(self) -> Tuple[float, float, float]:
        return self.density, self.viscosity, self.modulus


class EffectiveFluid(object):
    def __init__(self, fluid_1: Fluid, fluid_2: Fluid) -> None:
        self.fluid1 = fluid_1
        self.fluid2 = fluid_2
        self._saturation = None
        self._patch_parameter = None
        self._reference_frequency = None

    @property
    def saturation(self):
        return self._saturation

    @saturation.setter
    def saturation(self, value):
        if 0 <= value <= 1:
            self._saturation = value
        else:
            raise ValueError("Saturation must be between 0 and 1")

    @property
    def patch_parameter(self):
        return self._patch_parameter

    @patch_parameter.setter
    def patch_parameter(self, value):
        if (
            min(self.fluid1.modulus, self.fluid2.modulus)
            / max(self.fluid1.modulus, self.fluid2.modulus)
            <= value
            <= 1
        ):
            self._patch_parameter = value
        else:
            raise ValueError(
                f"Patch parameter must be between {min(self.fluid1.modulus, self.fluid2.modulus)/max(self.fluid1.modulus, self.fluid2.modulus)} and 1"
            )

    @property
    def reference_frequency(self):
        return self._reference_frequency

    @reference_frequency.setter
    def reference_frequency(self, value):
        self._reference_frequency = value

    @property
    def modulus(self):
        s, q = self.saturation, self.patch_parameter
        k1, k2 = self.fluid1.modulus, self.fluid2.modulus
        keff = (s + q * (1 - s)) / (s / k1 + q * (1 - s) / k2)
        return keff

    @property
    def omega_c(self):
        s, q = self.saturation, self.patch_parameter

        def BrooksCorey(s_wetting, pore_lambda: float = 1.0):
            s = s_wetting
            return s ** ((2 + 3 * pore_lambda) / pore_lambda), (1 - s) ** 2 * (
                1 - s ** ((2 + pore_lambda) / pore_lambda)
            )

        eta1, eta2 = self.fluid1.viscosity, self.fluid2.viscosity
        k1, k2 = BrooksCorey(s)
        m1, m2 = k1 / eta1, k2 / eta2
        eta_eff = (
            self.reference_frequency
            * eta1
            * (s * m1 + q * (1 - s) * m2)
            / (s + q * (1 - s))
        )
        return eta_eff

    @property
    def density(self):
        s = self.saturation
        rho1, rho2 = self.fluid1.density, self.fluid2.density
        rho_eff = s * rho1 + (1 - s) * rho2
        return rho_eff

    def __call__(
        self,
        saturation: float = 1.0,
        patch_parameter: float = 1.0,
        reference_frequency: float = 1.0,
    ) -> EffectiveFluid:
        self.saturation = saturation
        self.patch_parameter = patch_parameter
        self.reference_frequency = reference_frequency
        return self


class RockPhysicsModelCalibrator(object):
    def __init__(self):
        self._instance = None

    def __call__(
        self,
        Vp: np.float64,
        Vs: np.float64,
        Rho: np.float64,
        Qp: np.float64,
        phi: np.float64,
        Kf: np.float64,
        Km: np.float64 = 36.5,  # assume quartz,
    ) -> RockPhysicsModel:
        mu = Vs**2 * Rho
        lam = Vp**2 * Rho - 2 * mu
        QP = 1 / Qp
        epsilon = (45 / 8 * QP * (lam + mu) * (3 * Kf + 4 * mu)) / (
            15 * (lam**2)
            + 20 * lam * mu
            + 12 * (mu**2)
            - 3 * Kf * (5 * lam + 2 * mu)
        )
        Kd = (
            Km
            * (
                -(
                    Km
                    * phi
                    * (3 * lam + 2 * mu)
                    * (
                        15 * (-1 + QP) * (lam**2)
                        + 20 * (-1 + 2 * QP) * lam * mu
                        + 4 * (-3 + 5 * QP) * (mu**2)
                    )
                )
                + 3
                * (Kf**2)
                * (
                    3 * Km * (5 * lam + 2 * mu)
                    + -(
                        (-1 + phi)
                        * (3 * lam + 2 * mu)
                        * (5 * (-1 + QP) * lam + 2 * (-1 + 5 * QP) * mu)
                    )
                )
                + Kf
                * (
                    (-1 + phi)
                    * (3 * lam + 2 * mu)
                    * (
                        15 * (-1 + QP) * (lam**2)
                        + 20 * (-1 + 2 * QP) * lam * mu
                        + 4 * (-3 + 5 * QP) * (mu**2)
                    )
                    + 3
                    * Km
                    * (
                        15 * (-1 + phi * (-1 + QP)) * (lam**2)
                        + 4 * (-5 + -4 * phi + 10 * phi * QP) * lam * mu
                        + 4 * (-3 + -phi + 5 * phi * QP) * (mu**2)
                    )
                )
            )
        ) / (
            45
            * (Kf + -lam)
            * lam
            * (Km * (Kf + Kf * phi + -(Km * phi)) + Kf * (-1 + QP) * lam)
            + 6
            * (
                3 * Kf * Km * (Kf + Kf * phi + -(Km * phi))
                + 2
                * (
                    5 * (Km**2) * phi
                    + -5 * Kf * Km * (1 + phi)
                    + 2 * (Kf**2) * (-2 + 5 * QP)
                )
                * lam
                + 5 * Kf * (3 + -5 * QP) * (lam**2)
            )
            * mu
            + 4
            * (
                9 * (Km**2) * phi
                + 3 * (Kf**2) * (-1 + 5 * QP)
                + -(Kf * (9 * Km * (1 + phi) + (-19 + 35 * QP) * lam))
            )
            * (mu**2)
            + 8 * Kf * (3 + -5 * QP) * (mu**3)
        )
        self._instance = RockPhysicsModel(
            lam=lam,
            shear_modulus=mu,
            porosity=phi,
            crack_density=epsilon,
            dry_modulus=Kd,
            mineral_modulus=Km,
        )
        # self._instance.omega_squirt = 0.0
        return self._instance


class RockPhysicsModel:
    def __init__(
        self,
        lam: np.float64,
        shear_modulus: np.float64,
        porosity: np.float64,
        crack_density: np.float64,
        dry_modulus: np.float64,
        mineral_modulus: np.float64,
    ) -> None:
        self._lam = lam
        self._shear_modulus = shear_modulus
        self._porosity = porosity
        self._crack_density = crack_density
        self._dry_modulus = dry_modulus
        self._mineral_modulus = mineral_modulus
        self._fluid_modulus = None
        self._omega_squirt = None

    @property
    def lam(self):
        return self._lam

    @property
    def shear_modulus(self):
        return self._shear_modulus

    @property
    def porosity(self):
        return self._porosity

    @property
    def crack_density(self):
        return self._crack_density

    @property
    def dry_modulus(self):
        return self._dry_modulus

    @property
    def mineral_modulus(self):
        return self._mineral_modulus

    @property
    def fluid_modulus(self):
        return self._fluid_modulus

    @fluid_modulus.setter
    def fluid_modulus(self, value):
        self._fluid_modulus = value

    @property
    def omega_squirt(self):
        return self._omega_squirt

    @omega_squirt.setter
    def omega_squirt(self, value):
        self._omega_squirt = value

    @property
    def low_frequency_model(self) -> np.ndarray:
        if self._fluid_modulus is None:
            raise ValueError("Fluid modulus not set")
        Kf = self.fluid_modulus
        mu = self.shear_modulus
        phi = self.porosity
        cij = np.zeros((6, 6), dtype=np.float64)
        Km, Kd, mu, phi = (
            self.mineral_modulus,
            self.dry_modulus,
            self.shear_modulus,
            self.porosity,
        )

        # Gassmann's model
        Pmod = (
            Kd
            + (1 - Kd / Km) ** 2 / (phi / Kf - Kd / Km**2 + (1 - phi) / Km)
            + 4 / 3 * mu
        )

        # Might as well define cij's in case this ever needs anisotropic modelling

        cij[0, 0] = cij[1, 1] = cij[2, 2] = Pmod
        cij[3, 3] = cij[4, 4] = cij[5, 5] = mu
        cij[0, 1] = cij[0, 2] = cij[2, 0] = cij[1, 0] = cij[1, 2] = cij[2, 1] = (
            Pmod - 2 * mu
        )

        return cij

    def squirt_flow_model(self, omega) -> np.ndarray:
        if self.crack_density is None:
            raise ValueError("Crack density not set")
        if self.fluid_modulus is None:
            raise ValueError("Fluid modulus not set")
        if self.omega_squirt is None:
            raise ValueError("Squirt frequency not set")
        Kf = self.fluid_modulus
        epsilon = self.crack_density
        omegac = self.omega_squirt
        cij = np.zeros((6, 6), dtype=complex)
        l, m = self.lam, self.shear_modulus
        lamb = (16 * (15 * l * (-Kf + l) + 4 * (-3 * Kf + 5 * l) * m + 4 * m**2)) / (
            45.0 * (l + m) * (3 * Kf + 4 * m)
        )
        mu = (16 * m) / (45.0 * (l + m))
        low_freq = self.low_frequency_model
        cij[0, 0] = cij[1, 1] = cij[2, 2] = lamb + 2 * mu
        cij[3, 3] = cij[4, 4] = cij[5, 5] = mu
        cij[0, 1] = cij[0, 2] = cij[2, 0] = cij[1, 0] = cij[1, 2] = cij[2, 1] = lamb

        return low_freq + epsilon * cij * (l + 2 * m) * (
            1j * (10 ** (omega - omegac)) / (1 + 1j * (10 ** (omega - omegac)))
        )

    # def __call__(self, *args: Any, **kwds: Any) -> Any:

    # def plot(self) -> None:
    #     omegac = self.omega_squirt
    #     omega_axis = np.arange(-2, 2, 0.1) - omegac
    #     cij0 = self.gassmann_model
    #     cij = self.squirt_flow_model()
    #     # f_cij = np.array([cij(omega) for omega in omega_axis])
    #     f_cij = np.array([cij(omega) for omega in omega_axis])
    #     fig, ax = plt.subplots(1, 1)
    #     ax.axhline(cij0[0, 0], linestyle="--", color="k")
    #     ax.plot(omega_axis, np.real(f_cij[:, 0, 0]))
    #     plt.show()
    #     plt.close()

    def __call__(
        self,
        fluid_modulus: np.float64 | None = None,
        omegac: np.float64 | None = None,
    ) -> RockPhysicsModel:
        if fluid_modulus is not None:
            self.fluid_modulus = fluid_modulus
        if omegac is not None:
            self.omega_squirt = omegac
        return self


# if __name__ == "__main__":
# rock = RockPhysicsModel(
#     dry_modulus=30, shear_modulus=10, mineral_modulus=50, porosity=0.2
# )
# rock(fluid_modulus=10, epsilon=0.1, omegac=0.0).plot
