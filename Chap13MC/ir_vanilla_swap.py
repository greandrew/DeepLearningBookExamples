""" ir_vanilla_swap module

This module contains a representation of a vanilla fixed-float interest rate swap

"""

import torch as torch
import QuantLib as ql
from lgm_ir_monte_carlo import LGMIRMonteCarlo

class IRVanillaSwap:
    """ IRVanillaSwap trade class

        Represents a vanilla fixed-float interest rate swap

    """
    def __init__(self,
                 start_date: ql.Date,
                 tenor: str,
                 fixed_freq: str,
                 float_freq: str,
                 calendar: ql.Calendar,
                 short_conv,
                 long_conv,
                 date_gen_rule,
                 end_of_month: bool,
                 fixed_dc: ql.DayCounter,
                 float_dc: ql.DayCounter,
                 notional: float,
                 fixed_rate: float,
                 pay_fixed: bool):
        """ IRVanillaSwap __init__

            Args:
                start_date (ql.Date): swap start date
                tenor (str): swap length (tenor)
                fixed_freq (str):     Fixed leg frequency (tenor)
                float_freq (str):     Float leg frequency (tenor)
                calendar   (ql.Calendar):   Calendar
                short_conv (ql.BusinessDayConvention):  short-end day count convention
                long_conv (ql.BusinessDayConvention):  long-end day count convention
                date_gen_rule (ql.DateGeneration.Rule):  Date generation rule (forward / backward)
                end_of_month (bool): End-of-month schedule flag
                fixed_dc (ql.DayCounter):   Fixed-leg day count convention
                float_dc (ql.DayCounter):   Float-leg day count convention
                notional (float):  Swap notional
                pay_fixed (bool):  Pay the fixed leg?
        """
        self.tenor = ql.Period(tenor)
        end_date = calendar.advance(start_date, self.tenor)
        self.fixed_freq = fixed_freq
        self.float_freq = float_freq
        self.fixed_dc = fixed_dc
        self.float_dc = float_dc
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.pay_fixed = pay_fixed

        fixed_schedule = ql.Schedule(start_date,
                                     end_date,
                                     ql.Period(self.fixed_freq),
                                     calendar,
                                     short_conv,
                                     long_conv,
                                     date_gen_rule,
                                     end_of_month)
        self.fixed_schedule = [x for x in fixed_schedule]


        # We know that on every date bar the first date in the fixed schedule there is a cash flow
        # The size of each cash flow is Fixed DCF * Notional and so we build a dict with these weights
        self.fixed_dcf = list()
        for idate in range(1, len(self.fixed_schedule)):
            dcf = self.fixed_dc.yearFraction(self.fixed_schedule[idate - 1], self.fixed_schedule[idate])
            self.fixed_dcf.append(dcf)

        float_schedule = ql.Schedule(start_date,
                                     end_date,
                                     ql.Period(self.float_freq),
                                     calendar,
                                     short_conv,
                                     long_conv,
                                     date_gen_rule,
                                     end_of_month)
        self.float_schedule = [x for x in float_schedule]

        # The size of each cash flow is Fixed DCF * Notional and so we build a dict with these weights
        self.float_dcf = list()
        for idate in range(1, len(self.float_schedule)):
            dcf = self.float_dc.yearFraction(self.float_schedule[idate - 1], self.float_schedule[idate])
            self.float_dcf.append(dcf)

    def get_bond_dates(self,
                       tenor: str):
        """ Return a list of dates on which bonds of the specified tenor are required

            Args:
                tenor (str): Curve tenor

            Returns:
                list of ql.Dates
        """
        date_set = set()
        if tenor == "0D":
            # Discount curve
            adj_float_schedule = self.float_schedule[1:]
            date_set = date_set.union(adj_float_schedule)
            date_set = date_set.union(self.fixed_schedule[1:])
        elif tenor == self.float_freq:
            date_set = date_set.union(self.float_schedule)

        return sorted(list(date_set))

    def _value_fixed_leg_mc(self,
                            monte_carlo: LGMIRMonteCarlo,
                            sim_date: ql.Date,
                            device: torch.device,
                            float_type: torch.dtype) -> torch.tensor:
        """ Value the (residual) fixed leg of the swap on the given simulation date

            Args:
                monte_carlo (LGMIRMonteCarlo):  Monte Carlo model
                sim_date (ql.Date):             Simulation time step date
                device (torch.device):          PyTorch device holding the tensors
                float_type (torch.dtype):       PyTorch floating type
            Returns:
                Tensor of fixed leg values
        """
        # Now build a tensor that can be multiplied with the input bond prices
        fixed_dcfs = list()
        discount_dates = list()
        # First date in schedule does not have a cash flow
        for idate in range(1, len(self.fixed_schedule)):
            if self.fixed_schedule[idate] >= sim_date:
                discount_dates.append(self.fixed_schedule[idate])
                fixed_dcfs.append(self.fixed_dcf[idate-1] )

        tfixed_multiplier = torch.tensor(fixed_dcfs, device=device, dtype=float_type)

        if len(discount_dates) > 0:
            discount_bonds, _ = monte_carlo.get_disc_bonds(sim_date, discount_dates)
            return torch.mul(torch.matmul(discount_bonds, tfixed_multiplier.view(len(fixed_dcfs), 1)), self.fixed_rate)
        else:
            return torch.zeros([monte_carlo.get_no_paths(), 1], device=device, dtype=float_type)

    def _value_float_leg(self,
                         monte_carlo: LGMIRMonteCarlo,
                         sim_date: ql.Date,
                         device: torch.device,
                         float_type: torch.dtype) -> torch.tensor:
        """ Value the (residual) float leg of the swap on the given simulation date

            Args:
                monte_carlo (LGMIRMonteCarlo):  Monte Carlo model
                sim_date (ql.Date):             Simulation time step date
                device (torch.device):          PyTorch device holding the tensors
                float_type (torch.dtype):       PyTorch floating type
            Returns:
                Tensor of float leg values
        """
        # Now build a tensor that can be multiplied with the input bond prices
        float_dcfs = list()
        discount_dates = list()
        fwd_dates = list()

        for idate in range(0, len(self.float_schedule) - 1):
            if self.float_schedule[idate] >= sim_date:
                fwd_dates.append(self.float_schedule[idate])
                discount_dates.append(self.float_schedule[idate+1])
                float_dcfs.append(self.float_dcf[idate])

        tfloat_dcfs = torch.tensor(float_dcfs, device=device, dtype=float_type)

        if len(fwd_dates) > 0:

            discount_bonds, _ = monte_carlo.get_disc_bonds(sim_date, discount_dates)
            fwd_rates, _ = monte_carlo.get_fwd_rates(self.float_freq, sim_date, fwd_dates, discount_dates, tfloat_dcfs)

            tfloat_cash_flows = torch.mul(fwd_rates, tfloat_dcfs)
            tdisc_future_flows = torch.mul(tfloat_cash_flows, discount_bonds)
            tfloat_value = torch.sum(tdisc_future_flows, dim=1)
        else:
            tfloat_value = torch.zeros([monte_carlo.get_no_paths()], device=device, dtype=float_type)

        set_float_dcfs = list()
        set_discount_dates = list()
        set_fwd_dates = list()
        for idate in range(0, len(self.float_schedule) - 1):
            if self.float_schedule[idate] < sim_date:
                if self.float_schedule[idate+1] >= sim_date:
                    set_fwd_dates.append(self.float_schedule[idate])
                    set_discount_dates.append(self.float_schedule[idate+1])
                    set_float_dcfs.append(self.float_dcf[idate])

        tset_float_dcfs = torch.tensor(set_float_dcfs, device=device, dtype=float_type)

        if len(set_fwd_dates) > 0:
            set_discount_bonds, _ = monte_carlo.get_disc_bonds(sim_date, set_discount_dates)
            set_fwd_rates, _ = monte_carlo.get_set_fwd_rates(self.float_freq, set_fwd_dates, set_discount_dates,
                                                          tset_float_dcfs)

            tset_float_cash_flows = torch.mul(set_fwd_rates, tset_float_dcfs)
            tset_disc_flows = torch.mul(tset_float_cash_flows, set_discount_bonds)
            tset_value = torch.sum(tset_disc_flows, dim=1)


            tvalue_sum = torch.add(tfloat_value, tset_value)
            return tvalue_sum.view(len(tfloat_value), 1)
        else:
            return tfloat_value.view(len(tfloat_value), 1)

    def value(self,
              monte_carlo: LGMIRMonteCarlo,
              sim_date: ql.Date,
              device: torch.device,
              float_type: torch.dtype) -> torch.tensor:
        """ Value the swap on the given simulation date

            Args:
                monte_carlo (LGMIRMonteCarlo):  Monte Carlo model
                sim_date (ql.Date):             Simulation time step date
                device (torch.device):          PyTorch device holding the tensors
                float_type (torch.dtype):       PyTorch floating type

            Returns:
                Tensor of swap values
        """
        tfloat_leg = self._value_float_leg(monte_carlo, sim_date, device, float_type)
        tfixed_leg = self._value_fixed_leg_mc(monte_carlo, sim_date, device, float_type)

        if self.pay_fixed == True:
            tvalue = torch.add(tfloat_leg, -tfixed_leg)
        else:
            tvalue = torch.add(-tfloat_leg, tfixed_leg)

        tscaled_value = torch.mul(tvalue, self.notional)
        return tscaled_value

    def par_rate(self,
                 monte_carlo: LGMIRMonteCarlo,
                 sim_date: ql.Date,
                 device: torch.device,
                 float_type: torch.dtype) -> torch.tensor:
        """ Get par swap rates given simulation date

            Args:
                monte_carlo (LGMIRMonteCarlo):  Monte Carlo model
                sim_date (ql.Date):             Simulation time step date
                device (torch.device):          PyTorch device holding the tensors
                float_type (torch.dtype):       PyTorch floating type

            Returns:
                Tensor of swap values
        """
        tfloat_leg = self._value_float_leg(monte_carlo, sim_date, device, float_type)
        tfixed_annuity = torch.div(self._value_fixed_leg_mc(monte_carlo, sim_date, device, float_type), self.fixed_rate)

        return torch.div(tfloat_leg, tfixed_annuity)

    def get_disc_bonds(self,
                       monte_carlo: LGMIRMonteCarlo,
                       sim_date: ql.Date,
                       device: torch.device,
                       float_type: torch.dtype) -> (torch.tensor, bool):
        """ Get discount bonds used by the swap valuation

            Args:
                monte_carlo (LGMIRMonteCarlo):  Monte Carlo model
                sim_date (ql.Date):             Simulation time step date
                device (torch.device):          PyTorch device holding the tensors
                float_type (torch.dtype):       PyTorch floating type
            Returns:
                Tensor of discount bond values, flag indicating if present
        """
        # Now build a tensor that can be multiplied with the input bond prices
        discount_dates = set()
        # First date in schedule does not have a cash flow
        for idate in range(1, len(self.fixed_schedule)):
            if self.fixed_schedule[idate] >= sim_date:
                discount_dates.add(self.fixed_schedule[idate])

        for idate in range(0, len(self.float_schedule) - 1):
            if self.float_schedule[idate] >= sim_date:
                discount_dates.add(self.float_schedule[idate+1])

        if len(discount_dates) > 0:
            discount_bonds, _ = monte_carlo.get_disc_bonds(sim_date, list(sorted(discount_dates)))
            present = True
        else:
            discount_bonds = torch.zeros([monte_carlo.get_no_paths(), 1], device=device, dtype=float_type)
            present = False

        return discount_bonds, present

    def get_proj_bonds(self,
                       monte_carlo: LGMIRMonteCarlo,
                       sim_date: ql.Date,
                       device: torch.device,
                       float_type: torch.dtype) -> (torch.tensor, bool):
        """ Get projection bonds used by the swap valuation

            Args:
                monte_carlo (LGMIRMonteCarlo):  Monte Carlo model
                sim_date (ql.Date):             Simulation time step date
                device (torch.device):          PyTorch device holding the tensors
                float_type (torch.dtype):       PyTorch floating type
            Returns:
                Tensor of projection bond values, flag indicating if present
        """
        set_discount_dates = list()
        for idate in range(0, len(self.float_schedule) - 1):
            if self.float_schedule[idate] < sim_date:
                if self.float_schedule[idate + 1] >= sim_date:
                    set_discount_dates.append(self.float_schedule[idate + 1])

        if len(set_discount_dates) > 0:
            set_discount_bonds, _ = monte_carlo.get_disc_bonds(sim_date, set_discount_dates)
            present = True
        else:
            set_discount_bonds = torch.zeros([monte_carlo.get_no_paths(), 1], device=device, dtype=float_type)
            present = False

        return set_discount_bonds, present