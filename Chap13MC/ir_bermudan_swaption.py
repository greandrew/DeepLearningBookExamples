""" ir_bermudan_swaption module

This module contains a representation of a vanilla fixed-float interest rate swap

"""

import torch as torch
import QuantLib as ql
from ir_vanilla_swap import IRVanillaSwap
from lgm_ir_utility import LGMIRUtility
from lgm_ir_monte_carlo import LGMIRMonteCarlo
from copy import deepcopy
import torch as torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import timeit
from TorchGenericFeedForward import NeuralNetVariable

class IRBermudanSwaption:
    """ IRBermudanSwaption trade class

        Represents a Bermudan style option on a fixed-float interest rate swap
        Can be pay or recieve
        Can be long or short the option

    """
    def __init__(self,
                 ir_swap: IRVanillaSwap,
                 ex_schedule: list,
                 longshort: bool):
        """ IRBermudanSwaption __init__

            Args:
                ir_swap (IRVanillaSwap): Underlying swap
                ex_schedule (list): List of qlDates with option exercises
                longshort (bool): Long (True) or short the option
        """
        self.ir_swap = ir_swap
        self.ex_schedule = ex_schedule
        self.longshort = longshort

    from torch.utils import data

    class Dataset(data.Dataset):
        """Characterizes a dataset for PyTorch"""

        def __init__(self, feature_tensor, label_tensor):
            'Initialization'
            self.labels = label_tensor
            self.features = feature_tensor

        def __len__(self):
            'Denotes the total number of samples'
            return len(self.labels)

        def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample

            # Load data and get label
            X = self.features[index, :]
            y = self.labels[index]

            return X, y

    def _get_simulation_dates(self,
                              valdate: ql.Date) -> list:
        """ _get_simulation_dates

            Args:
                valdate (ql.Date): Valuation date

            Returns:
                list of ql.Dates
        """
        sim_dates = self.ex_schedule.copy()
        sim_dates.insert(0, valdate)
        return sim_dates

    def _get_bond_dates(self,
                        valdate: ql.Date) -> list:
        """ _get_bond_dates

            Args:
                valdate (ql.Date): Valuation date

            Returns:
                list of ql.Dates
        """
        disc_bond_date_set = set()

        # Get the bond dates from the swap
        for idate in self.ir_swap.get_bond_dates("0D"):
            disc_bond_date_set.add(idate)
        for idate in self.ir_swap.get_bond_dates(self.ir_swap.float_freq):
            disc_bond_date_set.add(idate)

        # Add the simulation dates
        sim_dates = self._get_simulation_dates(valdate)
        for idate in sim_dates:
            disc_bond_date_set.add(idate)

        return sorted(list(disc_bond_date_set))

    def _create_monte_carlo(self,
                            lgmirutility: LGMIRUtility,
                            yield_curves: dict,
                            mc_params: dict,
                            device: torch.device,
                            dtype,
                            valdate: ql.Date) -> LGMIRMonteCarlo:
        """
            Args:
                lgmirutility (LGMIRUtility): Monte Carlo calibration
                yield_curves (dict): Yield curve objects
                mc_params (dict): Monte Carlo parameters
                device (torch.device): PyTorch device
                dtype: PyTorch floating point type
                valdate (ql.Date): valuation date

            Returns:
                Monte Carlo object
        """
        sim_dates = self._get_simulation_dates(valdate)
        disc_bond_dates = self._get_bond_dates(valdate)
        return LGMIRMonteCarlo(lgmirutility, sim_dates, yield_curves, disc_bond_dates, mc_params, device, dtype)

    def _get_numeraire_ratios(self,
                              lgm_ir_monte_carlo: LGMIRMonteCarlo)-> list:
        """ _get_numeraire_ratios - return a list of numeraire ratios as tensors for each simulation date

            Args:
                lgm_ir_monte_carlo (LGMIRMonteCarlo): Monte Carlo object

            Returns:
                list of tensors containing numeraire ratios
        """
        t0_numeraire = lgm_ir_monte_carlo.get_numeraire_t0()
        horizon_date = list()
        sim_dates = lgm_ir_monte_carlo.simulation_dates
        horizon_date.append(sim_dates[-1])
        numeraire_ratios = list()

        for idate in sim_dates:
            # Get the numeraire
            df, _ = lgm_ir_monte_carlo.get_disc_bonds(idate, horizon_date)
            numeraire_ratio = torch.div(t0_numeraire, df)
            numeraire_ratios.append(numeraire_ratio)

        return numeraire_ratios

    def _get_swap_values_par_rates(self,
                                   lgm_ir_monte_carlo: LGMIRMonteCarlo)->tuple:
        """ _get_swap_values_par_rates - return a tuple containing:
                a list of swap values for each simulation date
                a list of par rates for each simulation date

            Args:
                lgm_ir_monte_carlo (LGMIRMonteCarlo): Monte Carlo object

            Returns:
                tuple of swap rates and par rates
        """
        swap_values = list()
        option_intrinsics = list()
        sim_dates = lgm_ir_monte_carlo.simulation_dates
        device = lgm_ir_monte_carlo.device
        dtype = lgm_ir_monte_carlo.float_type
        par_rates = list()
        for idate in sim_dates:
            # Swap values (unadjusted)
            swap_value = self.ir_swap.value(lgm_ir_monte_carlo, idate, device, dtype)
            swap_values.append(swap_value)

            # Par swap rates
            par_rate = self.ir_swap.par_rate(lgm_ir_monte_carlo, idate, device, dtype)
            par_rates.append(par_rate)

        return swap_values, par_rates

    def _get_bonds(self,
                               lgm_ir_monte_carlo: LGMIRMonteCarlo)->tuple:
        """ _get_bonds - return a list of bond values for each simulation date

            Args:
                lgm_ir_monte_carlo (LGMIRMonteCarlo): Monte Carlo object

            Returns:
                tuple of swap rates and par rates
        """
        bonds = list()
        sim_dates = lgm_ir_monte_carlo.simulation_dates
        device = lgm_ir_monte_carlo.device
        dtype = lgm_ir_monte_carlo.float_type
        for idate in sim_dates:
            disc_bonds, disc_present = self.ir_swap.get_disc_bonds(lgm_ir_monte_carlo, idate, device, dtype)
            proj_bonds, proj_present = self.ir_swap.get_proj_bonds(lgm_ir_monte_carlo, idate, device, dtype)

            if disc_present and proj_present:
                # concatenate
                bonds.append(torch.cat((disc_bonds, proj_bonds), 1))
            elif disc_present:
                bonds.append(disc_bonds)
            else:
                bonds.append(proj_present)

        return bonds

    def _get_option_intrinsic(self,
                              swap_values: list,
                              sim_dates: list) -> list:
        """ _get_option_intrinsic - get the option intrinsic values from the list of swap values

            Args:
                swap_values (list): List of tensors containing swap values on each simulation date
                sim_dates (list): List of ql.Date - simulation dates

            Returns:
                List of tensors containing option intrinsic values on each simulation date

        """
        option_intrinsics = list()
        for idate in range(0, len(sim_dates)):
            swap_value = swap_values[idate]

            # Option intrinsic
            values_zero = torch.zeros_like(swap_value)
            if self.longshort:
                call_value = torch.max(swap_value, values_zero)
            else:
                call_value = torch.max(-swap_value, values_zero)
            option_intrinsics.append(call_value)

        return option_intrinsics

    def _generate_polynomial_regressor(self,
                                       x: torch.tensor,
                                       order: int)-> list:
        """ _generate_polynomial_regressor - generate all the powers of the input up to order

            Args:
                x (torch.tensor): tensor containing the variable to be used
                order (int): Maximum power
            Returns:
                list of tensors, one for each power in range (1, order)
        """
        regressor_list = list()
        for iregressor in range(0, order):
            xpow = torch.pow(x, iregressor + 1)
            regressor_list.append(xpow)
        return regressor_list

    def get_swap_values(self,
                        lgm_ir_monte_carlo: LGMIRMonteCarlo)->list:
        """ get_swap_values - value the residual swap on each simulation date

            Args:
                lgm_ir_monte_carlo (LGMIRMonteCarlo): Monte Carlo model
            Return:
                list of swap values per time step
        """

        swap_values, _ = self._get_swap_values_par_rates(lgm_ir_monte_carlo)
        numeraire_ratios = self._get_numeraire_ratios(lgm_ir_monte_carlo)

        swap_values_per_timestep = list()

        for idate in range(0, len(lgm_ir_monte_carlo.simulation_dates)):
            value = swap_values[idate]
            numeraire = numeraire_ratios[idate]
            discounted_value = torch.mul(value, numeraire)
            value_mean = torch.mean(discounted_value, dim=0)
            value_mean_cpu = value_mean.cpu()
            swap_values_per_timestep.append(value_mean)

        return swap_values_per_timestep

    def get_european_values(self,
                            lgm_ir_monte_carlo: LGMIRMonteCarlo)->list:
        """ get_european_values - value the underlying european options swap on each exercise date

            Args:
                lgm_ir_monte_carlo (LGMIRMonteCarlo): Monte Carlo model
            Return:
                list of european swaption values per exercise date
        """

        swap_values, _ = self._get_swap_values_par_rates(lgm_ir_monte_carlo)
        option_values = self._get_option_intrinsic(swap_values, self.ex_schedule)
        numeraire_ratios = self._get_numeraire_ratios(lgm_ir_monte_carlo)

        european_option_values = list()

        for idx, idate in enumerate(self.ex_schedule):
            if idate in self.ex_schedule:
                value = option_values[idx]
                numeraire = numeraire_ratios[idx]
                discounted_value = torch.mul(value, numeraire)
                value_mean = torch.mean(discounted_value, dim=0)
                value_mean_cpu = value_mean.cpu()
                european_option_values.append(value_mean)

        return european_option_values

    def value_DNN(self,
                 lgmirutility: LGMIRUtility,
                 yield_curves: dict,
                 mc_params: dict,
                 device: torch.device,
                 dtype,
                 valdate: ql.Date,
                 model: torch.nn.Module,
                 dnn_params: dict)-> tuple:
        """ value_DNN - value Bermudan using DNN regression
            Here use independent Neural Networks for each exercise
            Single input - underlying par rate

            Args:
                lgmirutility (LGMIRUtility): Monte Carlo calibration
                yield_curves (dict): Yield curve objects
                mc_params (dict): Monte Carlo parameters
                device (torch.device): PyTorch device
                dtype: PyTorch floating point type
                valdate (ql.Date): valuation date
                model (torch.nn.Module): a Deep Neural Network model
                dnn_params (dict): deep neural network parameters
            Returns:
                tuple containing Bermudan NPV and regression data
        """
        mc = self._create_monte_carlo(lgmirutility, yield_curves, mc_params, device, dtype, valdate)
        numeraire_ratios = self._get_numeraire_ratios(mc)
        swap_values, par_rates = self._get_swap_values_par_rates(mc)
        option_intrinsics = self._get_option_intrinsic(swap_values, self.ex_schedule)

        # scale by numeraire ratios and notional
        scaled_swap_values = list()
        for inum, iswap in zip(numeraire_ratios, swap_values):
            scaled_swap_values.append(torch.mul(iswap, inum))

        scaled_option_intrinsics = list()
        for idx in range(len(option_intrinsics)):
            scaled_option_intrinsics.append(torch.mul(option_intrinsics[idx], numeraire_ratios[idx+1]))


        # Backward induction loop
        sim_dates = mc.simulation_dates
        cont_value = scaled_option_intrinsics[-1]

        regression_data = list()
        for idate in range(len(sim_dates) - 2, 0, -1):
            print("Exercise: {0}".format(idate))
            # store the regression data
            iregression_data = dict()
            iregression_data["cont_value"] = cont_value.cpu().detach().numpy()
            iregression_data["par_rates"] = par_rates[idate].cpu().detach().numpy()

            # Copy the input neural network model (need one copy per time step of backward induction)
            imodel = deepcopy(model).to(device)

            # Get par rates for the relevant date and generate the regression variables
            X = par_rates[idate]
            y = cont_value
            #y = torch.nn.functional.normalize(y)
            # Training generator to vend the data to the network
            data_set = self.Dataset(X, y)
            dataset_size = len(data_set)
            indices = list(range(dataset_size))
            split = int(np.floor(dnn_params["test_split"] * dataset_size))

            train_indices, val_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(val_indices)

            train_loader = torch.utils.data.DataLoader(data_set, batch_size=dnn_params["batch_size"],
                                                       sampler=train_sampler)
            test_loader = torch.utils.data.DataLoader(data_set, batch_size=dnn_params["batch_size"],
                                                      sampler=test_sampler)

            criterion = torch.nn.MSELoss()
            tcriterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(imodel.parameters(), dnn_params["learning_rate"])

            # Train the model
            total_step = len(train_loader)
            loss_data = list()
            starttime = timeit.default_timer()
            for epoch in range(dnn_params["epochs"]):
                for i, (features, labels) in enumerate(train_loader):
                    # Move tensors to the configured device

                    # Forward pass
                    outputs = imodel(features)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    iloss_data = dict()
                    iloss_data["epoch"] = epoch + 1
                    iloss_data["step"] = i + 1
                    iloss_data["loss"] = loss.item()

                    with torch.no_grad():
                        tloss = 0.0
                        for tfeatures, tlabels in test_loader:
                            toutputs = imodel(tfeatures)
                            tloss = tcriterion(toutputs, tlabels)
                        iloss_data["test_loss"] = tloss.item()
                    loss_data.append(iloss_data)

                    print("Epoch: {0}, step: {1}, loss: {2}, test_loss: {3}".format(iloss_data["epoch"],
                                                                                    iloss_data["step"],
                                                                                    iloss_data["loss"],
                                                                                    iloss_data["test_loss"]))
                    elapsed_time = timeit.default_timer()
                    iloss_data["Time"] = elapsed_time - starttime

            # Regress
            cont_value_hat = imodel(X)
            iregression_data["loss_data"] = loss_data
            iregression_data["inferred_value"] = cont_value_hat.cpu().detach().numpy()
            regression_data.append(iregression_data)

            # determine exercise
            cont_value = torch.max(cont_value_hat, scaled_swap_values[idate])

        return torch.mean(cont_value).cpu().detach().item(), regression_data

    def value_DNN_Reuse(self,
                        lgmirutility: LGMIRUtility,
                        yield_curves: dict,
                        mc_params: dict,
                        device: torch.device,
                        dtype,
                        valdate: ql.Date,
                        model: torch.nn.Module,
                        dnn_params: dict)-> tuple:
        """ value_DNN - value Bermudan using DNN regression
            Here reuse the same Neural Networks for each exercise
            Single input - underlying par rate

            Args:
                lgmirutility (LGMIRUtility): Monte Carlo calibration
                yield_curves (dict): Yield curve objects
                mc_params (dict): Monte Carlo parameters
                device (torch.device): PyTorch device
                dtype: PyTorch floating point type
                valdate (ql.Date): valuation date
                model (torch.nn.Module): a Deep Neural Network model
                dnn_params (dict): deep neural network parameters
            Returns:
                tuple containing Bermudan NPV and regression data
        """
        mc = self._create_monte_carlo(lgmirutility, yield_curves, mc_params, device, dtype, valdate)
        numeraire_ratios = self._get_numeraire_ratios(mc)
        swap_values, par_rates = self._get_swap_values_par_rates(mc)
        option_intrinsics = self._get_option_intrinsic(swap_values, self.ex_schedule)

        # scale by numeraire ratios
        scaled_swap_values = list()
        for inum, iswap in zip(numeraire_ratios, swap_values):
            scaled_swap_values.append(torch.mul(iswap, inum))

        scaled_option_intrinsics = list()
        for idx in range(len(option_intrinsics)):
            scaled_option_intrinsics.append(torch.mul(option_intrinsics[idx], numeraire_ratios[idx+1]))

        # Backward induction loop
        sim_dates = mc.simulation_dates
        cont_value = scaled_option_intrinsics[-1]
        #imodel = model.to(device)

        regression_data = list()
        for idate in range(len(sim_dates) - 2, 0, -1):
            print("Exercise: {0}".format(idate))
            # store the regression data
            iregression_data = dict()
            iregression_data["cont_value"] = cont_value.cpu().detach().numpy()
            iregression_data["par_rates"] = par_rates[idate].cpu().detach().numpy()

            # Copy the input neural network model (need one copy per time step of backward induction)
            imodel = deepcopy(model).to(device)

            # Get par rates for the relevant date and generate the regression variables
            X = par_rates[idate]
            y = cont_value
            # Training generator to vend the data to the network
            data_set = self.Dataset(X, y)

            dataset_size = len(data_set)
            indices = list(range(dataset_size))
            split = int(np.floor(dnn_params["test_split"] * dataset_size))

            train_indices, val_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(val_indices)

            train_loader = torch.utils.data.DataLoader(data_set, batch_size=dnn_params["batch_size"],
                                                       sampler=train_sampler)
            test_loader = torch.utils.data.DataLoader(data_set, batch_size=dnn_params["batch_size"],
                                                      sampler=test_sampler)

            criterion = torch.nn.MSELoss()
            tcriterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(imodel.parameters(), dnn_params["learning_rate"])

            # Train the model
            total_step = len(train_loader)
            loss_data = list()
            starttime = timeit.default_timer()
            for epoch in range(dnn_params["epochs"]):
                for i, (features, labels) in enumerate(train_loader):
                    # Move tensors to the configured device

                    # Forward pass
                    outputs = imodel(features)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    iloss_data = dict()
                    iloss_data["epoch"] = epoch + 1
                    iloss_data["step"] = i + 1
                    iloss_data["loss"] = loss.item()

                    with torch.no_grad():
                        tloss = 0.0
                        for tfeatures, tlabels in test_loader:
                            toutputs = imodel(tfeatures)
                            tloss = tcriterion(toutputs, tlabels)
                        iloss_data["test_loss"] = tloss.item()
                    loss_data.append(iloss_data)

                    print("Epoch: {0}, step: {1}, loss: {2}, test_loss: {3}".format(iloss_data["epoch"],
                                                                                    iloss_data["step"],
                                                                                    iloss_data["loss"],
                                                                                    iloss_data["test_loss"]))
                    elapsed_time = timeit.default_timer()
                    iloss_data["Time"] = elapsed_time - starttime

            # Regress
            cont_value_hat = imodel(X)
            iregression_data["loss_data"] = loss_data
            iregression_data["inferred_value"] = cont_value_hat.cpu().detach().numpy()
            regression_data.append(iregression_data)

            # determine exercise
            cont_value = torch.max(cont_value_hat, scaled_swap_values[idate])

        return torch.mean(cont_value).cpu().detach().item(), regression_data

    def value_DNN_bondregress(self,
                 lgmirutility: LGMIRUtility,
                 yield_curves: dict,
                 mc_params: dict,
                 device: torch.device,
                 dtype,
                 valdate: ql.Date,
                 hiddenlayers: int,
                 layerwidth: int,
                 dnn_params: dict)-> tuple:
        """ value_DNN - value Bermudan using DNN regression
            Here use independent Neural Networks for each exercise
            Multiple input - discount bond prices

            Args:
                lgmirutility (LGMIRUtility): Monte Carlo calibration
                yield_curves (dict): Yield curve objects
                mc_params (dict): Monte Carlo parameters
                device (torch.device): PyTorch device
                dtype: PyTorch floating point type
                valdate (ql.Date): valuation date
                hiddenlayers (int): Number of hidden layers
                layerwidth (int): width of each hidden layer
                dnn_params (dict): deep neural network parameters
            Returns:
                tuple containing Bermudan NPV and regression data
        """
        mc = self._create_monte_carlo(lgmirutility, yield_curves, mc_params, device, dtype, valdate)
        numeraire_ratios = self._get_numeraire_ratios(mc)
        swap_values, par_rates = self._get_swap_values_par_rates(mc)
        bonds = self._get_bonds(mc)
        option_intrinsics = self._get_option_intrinsic(swap_values, self.ex_schedule)

        # scale by numeraire ratios and notional
        scaled_swap_values = list()
        for inum, iswap in zip(numeraire_ratios, swap_values):
            scaled_swap_values.append(torch.mul(iswap, inum))

        scaled_option_intrinsics = list()
        for idx in range(len(option_intrinsics)):
            scaled_option_intrinsics.append(torch.mul(option_intrinsics[idx], numeraire_ratios[idx+1]))


        # Backward induction loop
        sim_dates = mc.simulation_dates
        cont_value = scaled_option_intrinsics[-1]

        regression_data = list()
        for idate in range(len(sim_dates) - 2, 0, -1):
            print("Exercise: {0}".format(idate))
            # store the regression data
            iregression_data = dict()
            iregression_data["cont_value"] = cont_value.cpu().detach().numpy()
            iregression_data["par_rates"] = par_rates[idate].cpu().detach().numpy()
            iregression_data["bonds"] = bonds[idate].cpu().detach().numpy()

            # Copy the input neural network model (need one copy per time step of backward induction)
            model = NeuralNetVariable(bonds[idate].shape[1], layerwidth, hiddenlayers, 1)
            imodel = deepcopy(model).to(device)

            # Get par rates for the relevant date and generate the regression variables
            X = bonds[idate]
            y = cont_value
            # Training generator to vend the data to the network
            data_set = self.Dataset(X, y)
            dataset_size = len(data_set)
            indices = list(range(dataset_size))
            split = int(np.floor(dnn_params["test_split"] * dataset_size))

            train_indices, val_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(val_indices)

            train_loader = torch.utils.data.DataLoader(data_set, batch_size=dnn_params["batch_size"],
                                                       sampler=train_sampler)
            test_loader = torch.utils.data.DataLoader(data_set, batch_size=dnn_params["batch_size"],
                                                      sampler=test_sampler)

            criterion = torch.nn.MSELoss()
            tcriterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(imodel.parameters(), dnn_params["learning_rate"])

            # Train the model
            total_step = len(train_loader)
            loss_data = list()
            starttime = timeit.default_timer()
            for epoch in range(dnn_params["epochs"]):
                for i, (features, labels) in enumerate(train_loader):
                    # Move tensors to the configured device

                    # Forward pass
                    outputs = imodel(features)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    iloss_data = dict()
                    iloss_data["epoch"] = epoch + 1
                    iloss_data["step"] = i + 1
                    iloss_data["loss"] = loss.item()

                    with torch.no_grad():
                        tloss = 0.0
                        for tfeatures, tlabels in test_loader:
                            toutputs = imodel(tfeatures)
                            tloss = tcriterion(toutputs, tlabels)
                        iloss_data["test_loss"] = tloss.item()
                    loss_data.append(iloss_data)

                    print("Epoch: {0}, step: {1}, loss: {2}, test_loss: {3}".format(iloss_data["epoch"],
                                                                                    iloss_data["step"],
                                                                                    iloss_data["loss"],
                                                                                    iloss_data["test_loss"]))
                    elapsed_time = timeit.default_timer()
                    iloss_data["Time"] = elapsed_time - starttime

            # Regress
            cont_value_hat = imodel(X)
            iregression_data["loss_data"] = loss_data
            iregression_data["inferred_value"] = cont_value_hat.cpu().detach().numpy()
            regression_data.append(iregression_data)

            # determine exercise
            cont_value = torch.max(cont_value_hat, scaled_swap_values[idate])

        return torch.mean(cont_value).cpu().detach().item(), regression_data
    
    def value_DNN_multi(self,
                 lgmirutility: LGMIRUtility,
                 yield_curves: dict,
                 mc_params: dict,
                 device: torch.device,
                 dtype,
                 valdate: ql.Date,
                 model: torch.nn.Module,
                 dnn_params: dict)-> tuple:
        """ value_DNN - value Bermudan using DNN regression
            Here use independent Neural Networks for each exercise
            Multi input - underlying par rate polynomial

            Args:
                lgmirutility (LGMIRUtility): Monte Carlo calibration
                yield_curves (dict): Yield curve objects
                mc_params (dict): Monte Carlo parameters
                device (torch.device): PyTorch device
                dtype: PyTorch floating point type
                valdate (ql.Date): valuation date
                model (torch.nn.Module): a Deep Neural Network model
                dnn_params (dict): deep neural network parameters
            Returns:
                tuple containing Bermudan NPV and regression data
        """
        mc = self._create_monte_carlo(lgmirutility, yield_curves, mc_params, device, dtype, valdate)
        numeraire_ratios = self._get_numeraire_ratios(mc)
        swap_values, par_rates = self._get_swap_values_par_rates(mc)
        option_intrinsics = self._get_option_intrinsic(swap_values, self.ex_schedule)

        # scale by numeraire ratios and notional
        scaled_swap_values = list()
        for inum, iswap in zip(numeraire_ratios, swap_values):
            scaled_swap_values.append(torch.mul(iswap, inum))

        scaled_option_intrinsics = list()
        for idx in range(len(option_intrinsics)):
            scaled_option_intrinsics.append(torch.mul(option_intrinsics[idx], numeraire_ratios[idx+1]))


        # Backward induction loop
        sim_dates = mc.simulation_dates
        cont_value = scaled_option_intrinsics[-1]

        regression_data = list()
        for idate in range(len(sim_dates) - 2, 0, -1):
            print("Exercise: {0}".format(idate))
            # store the regression data
            iregression_data = dict()
            iregression_data["cont_value"] = cont_value.cpu().detach().numpy()
            iregression_data["par_rates"] = par_rates[idate].cpu().detach().numpy()

            # Copy the input neural network model (need one copy per time step of backward induction)
            imodel = deepcopy(model).to(device)

            x_train_list = self._generate_polynomial_regressor(par_rates[idate], dnn_params['order'])

            x_train = torch.cat(x_train_list, dim=1)
            x_train_mean = torch.mean(x_train, dim=0)   
            x_train_std = torch.std(x_train, dim=0)

            scaled_x_train = (x_train - x_train_mean) / x_train_std 
            X = scaled_x_train
            # Get par rates for the relevant date and generate the regression variables
            y = cont_value
            #y = torch.nn.functional.normalize(y)
            # Training generator to vend the data to the network
            data_set = self.Dataset(X, y)
            dataset_size = len(data_set)
            indices = list(range(dataset_size))
            split = int(np.floor(dnn_params["test_split"] * dataset_size))

            train_indices, val_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(val_indices)

            train_loader = torch.utils.data.DataLoader(data_set, batch_size=dnn_params["batch_size"],
                                                       sampler=train_sampler)
            test_loader = torch.utils.data.DataLoader(data_set, batch_size=dnn_params["batch_size"],
                                                      sampler=test_sampler)

            criterion = torch.nn.MSELoss()
            tcriterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(imodel.parameters(), dnn_params["learning_rate"])

            # Train the model
            total_step = len(train_loader)
            loss_data = list()
            starttime = timeit.default_timer()
            for epoch in range(dnn_params["epochs"]):
                for i, (features, labels) in enumerate(train_loader):
                    # Move tensors to the configured device

                    # Forward pass
                    outputs = imodel(features)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    iloss_data = dict()
                    iloss_data["epoch"] = epoch + 1
                    iloss_data["step"] = i + 1
                    iloss_data["loss"] = loss.item()

                    with torch.no_grad():
                        tloss = 0.0
                        for tfeatures, tlabels in test_loader:
                            toutputs = imodel(tfeatures)
                            tloss = tcriterion(toutputs, tlabels)
                        iloss_data["test_loss"] = tloss.item()
                    loss_data.append(iloss_data)

                    print("Epoch: {0}, step: {1}, loss: {2}, test_loss: {3}".format(iloss_data["epoch"],
                                                                                    iloss_data["step"],
                                                                                    iloss_data["loss"],
                                                                                    iloss_data["test_loss"]))
                    elapsed_time = timeit.default_timer()
                    iloss_data["Time"] = elapsed_time - starttime

            # Regress
            cont_value_hat = imodel(X)
            iregression_data["loss_data"] = loss_data
            iregression_data["inferred_value"] = cont_value_hat.cpu().detach().numpy()
            regression_data.append(iregression_data)

            # determine exercise
            cont_value = torch.max(cont_value_hat, scaled_swap_values[idate])

        return torch.mean(cont_value).cpu().detach().item(), regression_data
    
    def value_DNN_multi_Reuse(self,
                          lgmirutility: LGMIRUtility,
                          yield_curves: dict,
                          mc_params: dict,
                          device: torch.device,
                          dtype,
                          valdate: ql.Date,
                          model: torch.nn.Module,
                          dnn_params: dict) -> tuple:
        """ value_DNN - value Bermudan using DNN regression
            Here use independent Neural Networks for each exercise
            Multi input - underlying par rate polynomial

            Args:
                lgmirutility (LGMIRUtility): Monte Carlo calibration
                yield_curves (dict): Yield curve objects
                mc_params (dict): Monte Carlo parameters
                device (torch.device): PyTorch device
                dtype: PyTorch floating point type
                valdate (ql.Date): valuation date
                model (torch.nn.Module): a Deep Neural Network model
                dnn_params (dict): deep neural network parameters
            Returns:
                tuple containing Bermudan NPV and regression data
        """
        mc = self._create_monte_carlo(lgmirutility, yield_curves, mc_params, device, dtype, valdate)
        numeraire_ratios = self._get_numeraire_ratios(mc)
        swap_values, par_rates = self._get_swap_values_par_rates(mc)
        option_intrinsics = self._get_option_intrinsic(swap_values, self.ex_schedule)

        # Scale by numeraire ratios and notional
        scaled_swap_values = [iswap * inum for inum, iswap in zip(numeraire_ratios, swap_values)]
        scaled_option_intrinsics = [opt * numeraire_ratios[idx + 1] for idx, opt in enumerate(option_intrinsics)]

        # Backward induction loop
        sim_dates = mc.simulation_dates
        cont_value = scaled_option_intrinsics[-1]
        imodel = model

        # Copy the input neural network model 
        regression_data = []
        for idate in range(len(sim_dates) - 2, 0, -1):
            print("Exercise: {0}".format(idate))
            # Store the regression data
            iregression_data = {
                "cont_value": cont_value.cpu().detach().numpy(),
                "par_rates": par_rates[idate].cpu().detach().numpy()
            }

            imodel = deepcopy(imodel).to(device)

            x_train_list = self._generate_polynomial_regressor(par_rates[idate], dnn_params['order'])
            x_train = torch.cat(x_train_list, dim=1)
            x_train_mean = torch.mean(x_train, dim=0)
            x_train_std = torch.std(x_train, dim=0)

            scaled_x_train = (x_train - x_train_mean) / x_train_std
            X = scaled_x_train
            y = cont_value

            # Training generator to vend the data to the network
            data_set = self.Dataset(X, y)
            dataset_size = len(data_set)
            indices = list(range(dataset_size))
            split = int(np.floor(dnn_params["test_split"] * dataset_size))

            train_indices, val_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(val_indices)

            train_loader = torch.utils.data.DataLoader(data_set, batch_size=dnn_params["batch_size"], sampler=train_sampler)
            test_loader = torch.utils.data.DataLoader(data_set, batch_size=dnn_params["batch_size"], sampler=test_sampler)

            criterion = torch.nn.MSELoss()
            tcriterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(imodel.parameters(), dnn_params["learning_rate"])

            # Early stopping criteria
            early_stop_patience = dnn_params.get("early_stop_patience", 10)
            best_test_loss = float('inf')
            epochs_no_improve = 0
            early_stop = False

            # Train the model
            total_step = len(train_loader)
            loss_data = []
            starttime = timeit.default_timer()
            for epoch in range(dnn_params["epochs"]):
                if early_stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                for i, (features, labels) in enumerate(train_loader):
                    # Move tensors to the configured device

                    # Forward pass
                    outputs = imodel(features)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    iloss_data = {
                        "epoch": epoch + 1,
                        "step": i + 1,
                        "loss": loss.item()
                    }

                    with torch.no_grad():
                        tloss = sum(tcriterion(imodel(tfeatures), tlabels) for tfeatures, tlabels in test_loader) / len(test_loader)
                        iloss_data["test_loss"] = tloss.item()

                    loss_data.append(iloss_data)

                    print(f"Epoch: {iloss_data['epoch']}, step: {iloss_data['step']}, loss: {iloss_data['loss']}, test_loss: {iloss_data['test_loss']}")
                    elapsed_time = timeit.default_timer()
                    iloss_data["Time"] = elapsed_time - starttime

                    # Early stopping check
                    if tloss.item() < best_test_loss:
                        best_test_loss = tloss.item()
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= early_stop_patience:
                        early_stop = True
                        break

                        # Regress
            cont_value_hat = imodel(X)
            iregression_data["loss_data"] = loss_data
            iregression_data["inferred_value"] = cont_value_hat.cpu().detach().numpy()
            regression_data.append(iregression_data)

            # Determine exercise
            cont_value = torch.max(cont_value_hat, scaled_swap_values[idate])

        return torch.mean(cont_value).cpu().detach().item(), regression_data

    def value_LS(self,
        order: int,
        lgmirutility: LGMIRUtility,
        yield_curves: dict,
        mc_params: dict,
        device: torch.device,
        dtype,
        valdate: ql.Date)-> tuple:

        mc = self._create_monte_carlo(lgmirutility, yield_curves, mc_params, 
                                      device, dtype, valdate)
        numeraire_ratios = self._get_numeraire_ratios(mc)
        swap_values, par_rates = self._get_swap_values_par_rates(mc)
        option_intrinsics = self._get_option_intrinsic(swap_values, 
                                                       self.ex_schedule)
    
        # scale by numeraire ratios
        scaled_swap_values = list()
        for inum, iswap in zip(numeraire_ratios, swap_values):
            scaled_swap_values.append(torch.mul(iswap, inum))
    
        scaled_option_intrinsics = list()
        for idx in range(len(option_intrinsics)):
            scaled_option_intrinsics.append(torch.mul(option_intrinsics[idx], 
                                            numeraire_ratios[idx+1]))
    
        # Backward induction loop
        sim_dates = mc.simulation_dates
        cont_value = scaled_option_intrinsics[-1]
        regression_data = list()
        for idate in range(len(sim_dates) - 2, 0, -1):
    
            # store the regression data
            ireg_data = dict()
            ireg_data["cont_value"] = cont_value.cpu().detach().numpy()
            ireg_data["par_rates"] = par_rates[idate].cpu().detach().numpy()
    
            # Get par rates for the relevant date 
            # and generate the regression variables
            x_train_list = self._generate_polynomial_regressor(par_rates[idate], 
                                                               order)
            x_ones = torch.ones_like(par_rates[idate])
            x_train_list.insert(0, x_ones)
            x_train = torch.cat(x_train_list, dim=1)
            y_train = cont_value
    
            # Regress
            x_train_inv = torch.pinverse(x_train)
            theta = torch.matmul(x_train_inv, y_train)
            cont_value_hat = torch.matmul(x_train, theta)
            ireg_data["inferred_value"] = cont_value_hat.cpu().detach().numpy()
            regression_data.append(ireg_data)
    
            # determine exercise
            cont_value = torch.max(cont_value_hat, scaled_swap_values[idate])
    
        return torch.mean(cont_value).cpu().detach().item(), regression_data
