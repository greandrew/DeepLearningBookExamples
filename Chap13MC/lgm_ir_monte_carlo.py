""" lgm_ir_utility monte carlo

This module contains the lgm_ir_monte_carlo class that provides the Monte Carlo model

"""
import torch as torch
import QuantLib as ql
from lgm_ir_utility import LGMIRUtility
from search_utils import index_lt

class LGMIRMonteCarlo:
    NUMERAIRE_TNR = "0D"

    """ LGM Interest Rate Monte Carlo Class

    The class implements a Monte Carlo model for a single currency. The class generates tensors of simulated discount
    bond prices at each time step. The model includes deterministic tenor basis

    The class also provides the ability to pass adjoints back through the calculation using PyTorch.
    """

    def __init__(self,
                 lgm_ir_utility: LGMIRUtility,
                 simulation_dates: list,
                 yield_curves: dict,
                 bond_dates: list,
                 mc_params: dict,
                 device: torch.device,
                 float_type: torch.dtype = torch.float32,
                 requires_grad: bool = False):
        """ LGM Interest Rate Utility Class __init__

            Args:
                lgm_ir_utility (LGMIRUtility): Model parameter integrals
                simulation_dates (list): list of ql.Date defining the simulation schedule including t=0
                yield_curves (dict): dictionary with tenors as keys and QuantLib yield curve objects
                bond_dates (dict): dictionary with tenors as keys and required dates as lists of ql.Date
                mc_params (dict): Monte Carlo parameters as a dictionary
                device (torch.device): The PyTorch device to use
                requires_grad (bool): Use gradients 
                
            Raises:
               AssertionError
        """
        # check simulation dates
        assert(len(simulation_dates)> 1), "Must be at least two simulation dates"
        # check the numeraire yield curve has been supplied
        assert(self.NUMERAIRE_TNR in yield_curves.keys())


        # store parameters
        self.mc_params = mc_params
        self.mc_params["no_time_steps"] = len(simulation_dates) - 1
        self.device = device
        self.required_grad = requires_grad
        self.float_type = float_type
        self.simulation_dates = simulation_dates
        self.lgm_ir_utility = lgm_ir_utility
        self.yield_curves = yield_curves

        # For the discount curve we need any date specified by the user and all the simulation dates
        # For the projection curves we need the dates at the start of each required forward provided by the user,
        # all the associated projection end dates and all the simulation dates
        self.input_bond_dates = bond_dates
        self._setup_disc_dates()

        # If gradients are required then build a dict for each simulation date to store them
        self.get_disc_bonds_count = 0
        self.get_fwd_rates_count = 0
        self.get_set_fwd_rates_count = 0
        if requires_grad:
            self.disc_bond_tensors = list()
            self.fwd_rate_tensors = list()
            self.set_fwd_rate_tensors = list()

        # Call remaining set up functions
        self._setup_rand()
        self._setup_bond_prices()
        self._setup_psi_integral_values()

    def _setup_disc_dates(self):
        """ Build the set of discount bond dates """
        date_set = set()
        self.disc_bond_dates = list()
        for idate in self.input_bond_dates:
            date_set.add(idate)
        self.disc_bond_dates = sorted(list(date_set.union(self.simulation_dates)))

    def _setup_rand(self):
        """ Generate and scale the random numbers """

        # generate the random numbers
        torch.manual_seed(self.mc_params["seed"])
        rand = torch.randn([self.mc_params["no_paths"], len(self.simulation_dates)], device=self.device,
                           requires_grad=self.required_grad, dtype=self.float_type)

        # scale the random numbers
        # extract the integral of phi squared at each simulation date
        self.phi2integral = self.lgm_ir_utility.get_phi2_integral(self.simulation_dates)

        # loop over calculating the square root of the differences in the integral
        time_step_vol = list()
        time_step_vol.append(torch.zeros([1], dtype=self.float_type))
        for isimdate in range(1, len(self.simulation_dates)):
            # We fill a tensor with the intermediate values and then sum - this is to avoid in place operations
            tvar = self.phi2integral[isimdate] - self.phi2integral[isimdate - 1]
            tvol = torch.sqrt(tvar).view(1)
            time_step_vol.append(tvol)

        ttime_step_vol = torch.cat(time_step_vol).view(len(self.simulation_dates))
        dtime_step_vol = ttime_step_vol.to(self.device)

        # Now scale the normal samples by the variance
        scaled_rand = torch.mul(rand, dtime_step_vol)

        # sum the integral values
        self.rand = torch.cumsum(scaled_rand, dim=1)

    def _setup_bond_prices(self):
        """ Extract the input bond prices from the yield curves """

        # All yield curves
        self.bond_prices = dict()
        for tnr in self.yield_curves.keys():
            bond_prices = list()
            yc = self.yield_curves[tnr]
            for ibond_date in self.disc_bond_dates:
                bond_prices.append(yc.discount(ibond_date))
            self.bond_prices[tnr] = torch.tensor(bond_prices, device=self.device, requires_grad=self.required_grad,
                                                 dtype=self.float_type)

    def _setup_psi_integral_values(self):
        """ Extract the required Psi values from the LGMIRUtility object """
        self.psi_integral = self.lgm_ir_utility.get_psi_integral(self.disc_bond_dates)

    def _get_exponential_factor_ex(self,
                                  sim_date: ql.Date,
                                  mat_dates: list,
                                  scaled_rand: torch.tensor,
                                  phi2_integral_scale: torch.tensor)->torch.tensor:
        """ Get the exponential factors for discount bond on the given simulation and maturity dates

            Args:
                sim_date (ql.Date): Simulation date
                mat_dates  (list): Maturity dates
                scaled_rand (torch.tensor): Scaled random numbers
                phi2_integral_scale (torch.tensor): phi squared integral x 0.5

            Returns:
                  torch tensor containing the exponential factor for each maturity date

        """
        # Get the index of the time step in the psi integral dates
        itime_step_psi = self.disc_bond_dates.index(sim_date)

        # Form a tensor with the Psi integral difference multiplier for the random numbers  (\Psi(T) - \Psi(t))
        # and a tensor with the drift term
        psi_integral_drift = []
        psi_integral_diffusion = []

        ipsi_h_date = self.disc_bond_dates.index(self.simulation_dates[-1])

        for ibond_idx, ibond_date in enumerate(mat_dates):
            if ibond_date >= sim_date:
                ipsi_date = self.disc_bond_dates.index(ibond_date)

                # Calculate (\Psi(T) - \Psi(t))
                psi_int_diff = torch.add(self.psi_integral[ipsi_date], - self.psi_integral[itime_step_psi])
                psi_integral_diffusion.append(psi_int_diff.view(1))

                # Calculate (\Psi(T) + \Psi(t) - 2\Psi(H))
                psi_integral_h2 = torch.mul(self.psi_integral[ipsi_h_date], 2.0)
                psi_int_horizon = torch.add(torch.add(self.psi_integral[ipsi_date], self.psi_integral[itime_step_psi]),
                                            -psi_integral_h2)

                # Calculate (\Psi(T) + \Psi(t) - 2\Psi(H)) * (\Psi(T) - \Psi(t))
                psi_drift = torch.mul(psi_int_horizon, psi_int_diff)

                # Calculate -\frac{1}{2}(\Psi(T) + \Psi(t) - 2\Psi(H)) * (\Psi(T) - \Psi(t))\int_0^t \phi^2(s) ds
                psi_scaled_drift = -torch.mul(phi2_integral_scale, psi_drift)
                psi_integral_drift.append(psi_scaled_drift.view(1))

        tpsi_int_diff = torch.cat(psi_integral_diffusion)
        dpsi_int_diff = tpsi_int_diff.to(self.device)
        tpsi_int_drift = torch.cat(psi_integral_drift)
        dpsi_int_drift = tpsi_int_drift.to(self.device)

        # Calculate the diffusion integral
        dpsi_scaled_rand = torch.mul(scaled_rand, dpsi_int_diff.view(1, -1))

        # Calculate the sum of drift and diffusion
        dpsi_exponent = torch.add(dpsi_scaled_rand, dpsi_int_drift)

        # Exponentiate
        return torch.exp(dpsi_exponent)

    def _get_exponential_factor(self,
                                sim_date: ql.Date,
                                mat_dates: list) -> torch.tensor:
        """ Get the exponential factors for discount bond on the given simulation and maturity dates

            Args:
                sim_date (ql.Date): Simulation date
                mat_dates  (list): Maturity dates

            Returns:
                torch tensor containing the exponential factor for each maturity date

            Raises:
                AssertionError
        """
        assert (sim_date in self.simulation_dates), "Simulation date not found in simulation date list."

        # Get the time step index
        itime_step = self.simulation_dates.index(sim_date)

        # Calculate \frac{1}{2}\int_0^t \phi^(s) ds
        phi2_integral_scale = torch.mul(0.5, self.phi2integral[itime_step])

        return self._get_exponential_factor_ex(sim_date, mat_dates, self.rand[:, itime_step].view(-1, 1),
                                              phi2_integral_scale)

    def _get_zcb_ex(self,
                    sim_date: ql.Date,
                    bond_prices: torch.tensor,
                    bond_dates: list,
                    scaled_rand: torch.tensor,
                    phi2_integral_scale: torch.tensor) -> torch.tensor:
        """ Get the future bond prices for the specified yield curve tenor and time step

            Args:
                sim_date (ql.Date): Simulation date
                bond_prices (torch.tensor): Zero-coupon bond prices for numerator in ratio
                bond_dates (list): Bond dates (ql.Date)
                scaled_rand (torch.tensor): Scaled random numbers
                phi2_integral_scale (torch.tensor): phi squared integral x 0.5

            Returns:
                 torch tensor containing the bond prices for all paths
        """
        # Get the index of the simulation date in the bond dates
        isim_date = self.disc_bond_dates.index(sim_date)

        # Get the indices of all the input bond dates in the cached bond dates
        bond_indices = list()
        for ibond_date in bond_dates:
            bond_indices.append(self.disc_bond_dates.index(ibond_date))

        # Get the exponential factor for the bond dates
        dpsi_exp = self._get_exponential_factor_ex(sim_date, bond_dates, scaled_rand, phi2_integral_scale)

        bond_ratio = []
        for ibond_idx, ibond_date in enumerate(bond_dates):
            if ibond_date >= sim_date:

                # Calculate P(0, T) / P(0, t)
                p_ratio = torch.div(bond_prices[bond_indices[ibond_idx]], bond_prices[isim_date])
                bond_ratio.append(p_ratio.view(1))

        tbond_ratio = torch.cat(bond_ratio)
        dbond_ratio = tbond_ratio.to(self.device)

        # Scale by bond ratio
        dsim_bond_prices = torch.mul(dbond_ratio, dpsi_exp)

        return dsim_bond_prices

    def _get_zcb(self,
                 sim_date: ql.Date,
                 bond_prices: torch.tensor,
                 bond_dates: list) -> torch.tensor:
        """ Get the future bond prices for the specified yield curve tenor and time step

            Args:
                ssim_date (ql.Date): Simulation date
                bond_prices (torch.tensor): Zero-coupon bond prices for numerator in ratio
                bond_dates (list): Bond dates (ql.Date)

            Returns:
                 torch tensor containing the bond prices for all paths

            Raises:
               AssertionError
        """
        assert (sim_date in self.simulation_dates), "Simulation date " + str(sim_date.dayOfMonth()) + "/" + str(sim_date.month())+ "/" + str(sim_date.year()) + " not found in simulation date list."

        # Get the time step index
        itime_step = self.simulation_dates.index(sim_date)

        # Calculate \frac{1}{2}\int_0^t \phi^(s) ds
        phi2_integral_scale = torch.mul(0.5, self.phi2integral[itime_step])

        return self._get_zcb_ex(sim_date, bond_prices, bond_dates, self.rand[:, itime_step].view(-1, 1),
                                phi2_integral_scale)

    def get_disc_bonds(self,
                       sim_date: ql.Date,
                       bond_dates: list) -> (torch.tensor, int):
        """ Get the future bond prices for the specified yield curve tenor and time step

            Args:
                sim_date (ql.Date): Simulation date
                bond_dates (list): List of bond dates (must be in the list supplied during __init__)

            Returns:
                 torch tensor containing the bond prices for all paths, index

            Raises:
               AssertionError
        """

        dsim_bond_prices = self._get_zcb(sim_date, self.bond_prices["0D"],
                                         bond_dates)
        if self.required_grad:
            self.disc_bond_tensors.append(dsim_bond_prices)

        self.get_disc_bonds_count = self.get_disc_bonds_count + 1

        return dsim_bond_prices, (self.get_disc_bonds_count)

    def get_numeraire_t0(self) -> torch.tensor:
        """ Get the numeraire bond prices t=0

            Returns:
                torch tensor containing the numeraire for t=0

        """
        # The numeraire is the last bond price on the last simulation date
        isim_date = self.disc_bond_dates.index(self.simulation_dates[-1])
        return self.bond_prices["0D"][isim_date]

    def get_fwd_rates(self,
                      tnr: str,
                      sim_date: ql.Date,
                      fwd_start_dates: list,
                      fwd_end_dates: list,
                      dcfs: torch.tensor) -> (torch.tensor, int):
        """ Get the future forward rates for the specified tenor and start dates

            Args:
                tnr (str): Tenor ("1D", "1M", "3M", "6M", "12M")
                sim_date (ql.Date): Simulation date
                fwd_start_dates (list): List of ql.Date for the start dates of forward rates
                fwd_end_dates (list): List of ql.Date for the start dates of forward rates
                dcfs (torch.tensor): tensor containing day-count fractions

            Returns:
                torch tensor containing the forward rates for all paths, index

            Raises:
               AssertionError
        """
        assert (sim_date in self.simulation_dates), "Simulation date not found in simulation date list."
        assert (len(fwd_start_dates) == len(fwd_end_dates)), "Forward start and end dates not same length"
        assert (len(fwd_start_dates) == dcfs.shape[0]), "Forward start dates and dcfs not same length"
        for ifwd_start_date in fwd_start_dates:
            assert(ifwd_start_date >= sim_date), "Forward start dates must be on or after the simulation date"

        # get start bond prices
        dsim_start_bond_prices = self._get_zcb(sim_date, self.bond_prices[tnr], fwd_start_dates)

        # get end bond prices
        dsim_end_bond_prices = self._get_zcb(sim_date, self.bond_prices[tnr], fwd_end_dates)

        # Calculate forward rates
        sim_bond_ratio = torch.div(dsim_start_bond_prices, dsim_end_bond_prices)
        unadj_fwds = torch.add(sim_bond_ratio, -1.0)
        adj_fwds = torch.div(unadj_fwds, dcfs)

        self.get_fwd_rates_count = self.get_fwd_rates_count + 1

        if self.required_grad:
            self.fwd_rate_tensors.append(adj_fwds)

        return adj_fwds, self.get_fwd_rates_count-1

    def get_set_fwd_rates(self,
                          tnr: str,
                          fwd_start_dates: list,
                          fwd_end_dates: list,
                          dcfs: torch.tensor) -> (torch.tensor, int):
        """ Get the set forward rates for the specified start and end date

            Args:
                tnr (str): Tenor ("1D", "1M", "3M", "6M", "12M")
                fwd_start_dates (list): List of ql.Date for the start dates of forward rates
                fwd_end_dates (list): List of ql.Date for the start dates of forward rates
                dcfs (torch.tensor): tensor containing day-count fractions

            Returns:
                torch tensor containing the forward rates for all paths, index

            Raises:
                AssertionError
        """
        assert (len(fwd_start_dates) == len(fwd_end_dates)), "Forward start and end dates not same length"
        assert (len(fwd_start_dates) == dcfs.shape[0]), "Forward start dates and dcfs not same length"

        fwds_list = list()
        for ifwd_start, ifwd_end, idcf in zip(fwd_start_dates, fwd_end_dates, dcfs):
            # Find the timestep index prior to the fixing date
            ifwd_start_lidx = index_lt(self.simulation_dates, ifwd_start)

            # calculate time weight
            dc = ql.Actual365Fixed()

            alpha = dc.yearFraction(self.simulation_dates[ifwd_start_lidx], ifwd_start) \
                    / dc.yearFraction(self.simulation_dates[ifwd_start_lidx],
                                      self.simulation_dates[ifwd_start_lidx + 1])

            # Calculate the weighted stochastic integral
            stoch_inta = torch.mul(self.rand[:, ifwd_start_lidx].view(-1, 1), 1.0-alpha)
            stoch_intb = torch.mul(self.rand[:, ifwd_start_lidx+1].view(-1, 1), alpha)
            stoch_int = torch.add(stoch_inta, stoch_intb)

            # Calculate the weighted phi2_integral
            phi2_integrala = torch.mul(self.phi2integral[ifwd_start_lidx], 1.0 - alpha)
            phi2_integralb = torch.mul(self.phi2integral[ifwd_start_lidx+1], alpha)
            phi2_integral = torch.add(phi2_integrala, phi2_integralb)
            phi2_integral_scale = torch.mul(0.5, phi2_integral)

            # get the bond at the end of the forward
            fwd_end_dates = list()
            fwd_end_dates.append(ifwd_end)
            dsim_end_bond_prices = self._get_zcb_ex(ifwd_start, self.bond_prices[tnr], fwd_end_dates, stoch_int,
                                                    phi2_integral_scale)

            # Calculate forward rates
            sim_bond_ratio = torch.pow(dsim_end_bond_prices, -1.0)
            tones = torch.ones_like(sim_bond_ratio, device=self.device, dtype=self.float_type)
            unadj_fwds = torch.add(sim_bond_ratio, -tones)
            adj_fwds = torch.div(unadj_fwds, idcf)

            fwds_list.append(adj_fwds)

        dset_fwds = torch.cat(fwds_list)

        self.get_set_fwd_rates_count = self.get_set_fwd_rates_count + 1

        if self.required_grad:
            self.set_fwd_rate_tensors.append(dset_fwds)

        return dset_fwds, self.get_set_fwd_rates_count-1

    def get_no_paths(self)-> int:
        return self.mc_params["no_paths"]
