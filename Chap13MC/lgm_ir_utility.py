""" lgm_ir_utility module

This module contains the lgm_ir_utility class that provides the integrated model parameters

"""

from search_utils import index_lt
import torch as torch
import QuantLib as ql

class LGMIRUtility:
    """ LGM Interest Rate Utility Class

    The class takes the piecewise constant parameters of the two factor Linear Gaussian Model (LGM):
    - :math:`\phi`
    - :math:`\psi`
    and generates daily numeric intergal values.

    The class also provides the ability to pass adjoints back through the calculation using PyTorch.
    """

    def __init__(self,
                 dates: list,
                 phi,
                 psi,
                 start_date: ql.Date,
                 end_date: ql.Date,
                 float_type: torch.dtype = torch.float32):
        """ LGM Interest Rate Utility Class __init__

            Args:
               dates (list):  List of ql dates defining the time partition
               phi (torch 1D tensor): Array of phi
               psi (torch 1D tensor): Array of psi
               start_date (date): Integral start date
               end_date (date): Integral end date

            Raises:
               AssertionError
        """
        assert(len(dates) == len(phi) + 1), "dates array should be one element longer than phi"
        assert(len(phi) == len(psi)), "phi and psi should be same length"
        self.dates = dates
        self.phi = torch.tensor(phi, dtype=float_type, requires_grad=True)
        self.psi = torch.tensor(psi, dtype=float_type, requires_grad=True)
        self.float_type = float_type
        # Generate internal integrals
        self._generate_partition(start_date, end_date)
        self._integrate_psi()
        self._integrate_phi2()

    def _generate_partition(self,
                            start_date: ql.Date,
                            end_date: ql.Date):
        """ Generate a daily date partition between start_date and end_date inclusively

            Args:
                start_date (ql date): Start date of partition
                end_date (ql date): End date of partition
        """

        self.partition_dates = []
        self.partition_dates.append(start_date)
        curr_date = start_date + 1
        while curr_date <= end_date:
            self.partition_dates.append(curr_date)
            curr_date = curr_date + 1

    def _lookup_param_index(self,
                           lookup_date: ql.Date) -> int:
        """ Lookup the date in the dates list

            Args:
                lookup_date (ql date): The date to search for

            Returns:
                index of the date below
        """
        return index_lt(self.dates, lookup_date) - 1

    def _integrate_psi(self):
        """ Integrate :math:`\psi` to give :math:`\Psi`"""

        dc = ql.Actual365Fixed()
        delta_t = dc.yearFraction(self.partition_dates[0], self.partition_dates[0] + 1)

        psi_delta_t = torch.mul(self.psi, delta_t)

        # Loop over the partition integrating
        psi_integral_comp = []
        psi_integral_comp.append(torch.zeros([1], dtype=self.float_type))
        for idate in range(1, len(self.partition_dates)):

            # lookup correct date in dates array
            param_index = self._lookup_param_index(self.partition_dates[idate])

            # We fill a tensor with the intermediate values and then sum - this is to avoid in place operations
            psi_integral_comp.append(psi_delta_t[param_index].clone().view(1))

        tpsi_integral = torch.cat(psi_integral_comp)
        self.Psi = torch.cumsum(tpsi_integral, dim=0)

    def _integrate_phi2(self):
        """ Integrate :math:`\phi^2` """

        dc = ql.Actual365Fixed()
        delta_t = dc.yearFraction(self.partition_dates[0], self.partition_dates[0] + 1)

        # Allocate tensors for the values and to store gradients (before call to backward)

        phisqrd = torch.pow(self.phi, 2)
        phisqrddelta_t = torch.mul(phisqrd, delta_t)

        # Loop over the partition integrating
        phi2_integral_comp = []
        phi2_integral_comp.append(torch.zeros([1], dtype=self.float_type))
        for idate in range(1, len(self.partition_dates)):

            # lookup correct date in dates array
            param_index = self._lookup_param_index(self.partition_dates[idate])

            # We fill a tensor with the intermediate values and then sum - this is to avoid in place operations
            phi2_integral_comp.append(phisqrddelta_t[param_index].clone().view(1))

        tphi2_integral = torch.cat(phi2_integral_comp)
        self.Phi2 = torch.cumsum(tphi2_integral, dim=0)

    def _lookup_partition_date(self,
                               lookup_date: ql.Date) -> int:
        """ Return the index of a date in the internal date partition

            Args:
                lookup_date (ql date): The date to search for

            Returns:
                index of the date

            Raises:
                 AssertionError
        """
        index = lookup_date - self.partition_dates[0]
        assert (index >= 0), "lookup_date falls before start of partition"
        assert (index < len(self.partition_dates)), "lookup_date falls after end of partition"
        return index

    def get_phi2_integral(self,
                 dates: list) -> torch.tensor:
        """ Get Phi2 for a set of dates

            Args:
                dates (list of ql.Date): Array of dates defining the simulation schedule

            Returns:
                 torch tensor containing Phi2 values
        """
        phi2_output = torch.zeros([len(dates)], dtype=self.float_type)
        for idate in range(len(dates)):
            index = self._lookup_partition_date(dates[idate])
            phi2_output[idate] = self.Phi2[index]
        return phi2_output

    def insert_phi2_integral_adjoints(self,
                             dates: list,
                             adjoints):
        """ Insert phi2 adjoints at the specified dates

            Args:
                dates (list of ql.Date): Array of dates defining the simulation schedule
                adjoints : Iterable of Adjoints

            Raises:
                AssertionError
        """

        assert(len(dates) == len(adjoints)), "Adjoints and date array length mismatch"
        self.Phi2Grad = torch.zeros([len(self.partition_dates)], dtype=self.float_type, requires_grad=False)
        for idate in range(len(dates)):
            index = self._lookup_partition_date(dates[idate])
            self.Phi2Grad[index] = adjoints[idate]


    def get_psi_integral(self,
                 dates: list) -> torch.tensor:
        """ Get Phi2 for a set of dates

            Args:
                dates (list of ql.Date): Array of dates defining the simulation schedule

            Returns:
                 torch tensor containing Phi2 values
        """
        psi_output = torch.zeros([len(dates)], dtype=self.float_type)
        for idate in range(len(dates)):
            index = self._lookup_partition_date(dates[idate])
            psi_output[idate] = self.Psi[index]
        return psi_output

    def insert_psi_integral_adjoints(self,
                             dates: list,
                             adjoints):
        """ Insert phi2 adjoints at the specified dates

            Args:
                dates (list of ql.Date): Array of dates defining the simulation schedule
                adjoints : Iterable of Adjoints

            Raises:
                AssertionError
        """

        assert (len(dates) == len(adjoints)), "Adjoints and date array length mismatch"
        self.PsiGrad = torch.zeros([len(self.partition_dates)], dtype=self.float_type, requires_grad=False)
        for idate in range(len(dates)):
            index = self._lookup_partition_date(dates[idate])
            self.PsiGrad[index] = adjoints[idate]

    def backward(self):
        """ Propogate Adjoints back """

        self.Phi2.backward(self.Phi2Grad)
        self.Psi.backward(self.PsiGrad)

    def extract_phi_adjoints(self) -> torch.tensor:
        """ Extract phi adjoint tensor """

        return self.phi.grad

    def extract_psi_adjoints(self) -> torch.tensor:
        """ Extract psi adjoint tensor """

        return self.psi.grad

