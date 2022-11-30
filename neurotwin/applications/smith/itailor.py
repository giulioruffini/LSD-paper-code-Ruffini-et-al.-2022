"""
SMITH
itailor
-------------------------------------------------------------------------------
This module is used to estimate parameters to build iNeTs from a binarized
dataset. It takes the binarized pattern that is obtained from the binarization
module, and uses it to estimate the parameter h and J of the Ising model
that best describes the data.
"""
from typing import Optional
import logging
import numpy as np

# Create a logger instance with the name of the current module
from neurotwin.applications.smith.inet import Inet

logger = logging.getLogger(__name__)

class Itailor:
    """
    This class is meant for optimization and personalization of the iNeTs.
    It is capable of creating Archetypes from complete data of different
    subjects, and also it is able to create personalized iNeTs from single
    subject data and an Archetype.
    Input binarized data of all subjects (if creating archetype) has to be a
    dictionary where the key goes from 1 to n, where n are the number of
    subjects. So each of the containers in the dictionary is a binarized
    pattern with the form of ROI x time.
    """

    def __init__(self,
                 inet_inputs: dict,
                 binarized_data: np.ndarray,
                 method_output: Optional[dict] = None) -> None:

        """
        Args:
        inet_inputs : Contains the inputs defining the inet:
            {"iNeT" : object
                An iNet object created with the iNeT
            "class method" : str
                This parameter has three options, either "archetype" to create
                an archetype from data of multiple subjects, or
                "personalized_H" and "personalized_h_j" which estimate the
                parameters for a personalized Ising model given an archetype
                and data for that subject.
            "archetype_h" : np.ndarray
                Here the input is None, if we are creating an archetype, but if
                we are personalizing an iNeT here we provide the Archetype
                parameter h to do so.
            "archetype_j" : np.ndarray
                Here the input is None, if we are creating an archetype, but if
                we are personalizing an iNeT here we provide the Archetype
                parameter J to do so.
            }
        binarized_data : If method = "archetype" this is data from all the
            subjects. If method = "personalized" this is data from the subject
            we are personalizing.
        method_output : Information about the method used, dictionary with
            information about error and precision of the method:
            {"minimum_error" : float
                Value of the minimum error for the solution found in the
                "personalized" method to estimate a personalized inet from an
                archetype.
            "error_evolution" : np.ndarray
                Array with the values of the error across the method
                ("personalized")
            }
        """
        self.method_output: dict = method_output
        self.inet: Inet = inet_inputs["inet"]
        self.binarized_data: np.ndarray = binarized_data
        self.archetype_h: np.ndarray = inet_inputs["archetype_h"]
        self.archetype_j: np.ndarray = inet_inputs["archetype_j"]
        self.lattice_size: int
        self.data_length: int
        [self.lattice_size, self.data_length] = np.shape(self.binarized_data)

    def _compute_error(self,
                       diff_corr: np.matrix,
                       diff_mean: np.matrix) -> float:
        """
        Computes error for an iteration of the Ezaki method (combined error of
        mean spin, and pairwise correlation). Used in Equations 2.9 and 2.10 of
        Ezaki 2017. And Equation 10 of TN0150.

        Args:
            diff_corr: difference between model and data correlations
            diff_mean: difference between model and data mean spins

        Returns:
            error: the error for an estimation of h and J w.r.t data
        """
        _norm1: float = np.linalg.norm(diff_corr, "fro")
        _norm2: float = np.linalg.norm(diff_mean, 2)
        _error_norm: float = np.sqrt(_norm1 ** 2 + _norm2 ** 2)
        error: float
        error = _error_norm / (self.lattice_size * (self.lattice_size + 1))
        return error

    def _estimate_tanh(self,
                       j_ising: np.ndarray,
                       h_ising: np.ndarray) -> np.matrix:
        """
        Computes the hyperbolic tangent term of the analytical expression
        of the mean spin and correlation derived with the pseudolikelihood
        method. Used in Equations 2.11 and 2.12 of Ezaki 2017.

        Args:
            j_ising: Parameter J of the Ising model
            h_ising: Parameter h of the Ising model

        Returns:
            tanh: the hyperbolic tangent term of the mean spin and correlation
            analytical expression (model).
        """
        _hterm: np.ndarray = np.matmul(h_ising, np.ones([1, self.data_length]))
        _jterm: np.matrix = np.matmul(j_ising, self.binarized_data)
        tanh: np.matrix = np.tanh(_hterm + _jterm)
        return tanh

    def _estimate_sigmai_boltz(self, j_ising: np.ndarray,
                               h_ising: np.ndarray) -> np.matrix:
        """
        Computes the estimated value of the model mean spin derived with the
        pseudolikelihood method. Equation 2.11 in Ezaki 2017.

        Args:
            j_ising: Parameter J of the Ising model
            h_ising: Parameter h of the Ising model

        Returns:
            sigmai_boltz: mean spin w.r.t boltzmann-like distribution
        """
        tanh: np.matrix = self._estimate_tanh(j_ising, h_ising)
        sigmai_boltz: np.matrix = np.mean(tanh, axis=1)
        sigmai_boltz = np.reshape(sigmai_boltz, [self.lattice_size, 1])
        return sigmai_boltz

    def _estimate_diff_sigmas(self,
                              j_ising: np.ndarray,
                              h_ising: np.ndarray,
                              data_mean: np.matrix) -> np.matrix:
        """
        Estimates the difference between the mean spin w.r.t boltzmann and the
        empirical mean spin (data). Used in Equation 2.9 of Ezaki 2017.
        Args:
            j_ising: Parameter J of the Ising model
            h_ising: Parameter h of the Ising model
            data_mean: the mean activation computed from the data, this is the
                empirical mean spin

        Returns:
            diff_sigmas: the difference between the two sigmas
        """
        sigmai_boltz: np.matrix
        sigmai_boltz = self._estimate_sigmai_boltz(j_ising, h_ising)
        diff_sigmas: np.matrix = sigmai_boltz - data_mean
        # Reshaped in this way for a product with a matrix
        diff_sigmas = np.reshape(diff_sigmas, [self.lattice_size, 1])
        return diff_sigmas

    def _update_h(self,
                  h_ising: np.ndarray,
                  delta_t: float,
                  diff_sigmas: np.matrix,
                  sparsity: float) -> np.matrix:
        """
        Computes the updated h parameter of the Ising model for each step of
        the "Ezaki" Method. Equation 2.9 of Ezaki 2017.

        Args:
            h_ising: Parameter h of the Ising model
            delta_t: Step size of the Ezaki method
            diff_sigmas: The change of h in this step of the ezaki method,
                given by the pseudolikelihood approximation in Ezaki et al.,
                2017

        Returns:
            new_h: Updated parameter h of the Ising model
        """
        new_h: np.matrix = h_ising - delta_t * (diff_sigmas +sparsity * np.sign(h_ising))
        # Reshape it to be able to do product with matrix afterwards
        new_h = np.reshape(new_h, [self.lattice_size, 1])
        return new_h

    @staticmethod
    def _update_j(j_ising: np.ndarray,
                  delta_t: float,
                  diff_corrs: np.matrix,
                  sparsity: float) -> np.matrix:
        """
        Computes the updated j parameter of the Ising model for
        each step of the "Ezaki" Method. Equation 2.10 in Ezaki 2017.

        Args:
            j_ising: Parameter J of the Ising model
            delta_t: Step size
            diff_corrs: The change of j in this step of the ezaki method, given
                by the pseudolikelihood approximation in Ezaki et al ., 2017

        Returns:
            new_j: Updated J parameter of the Ising model
        """

        new_j: np.matrix = j_ising - delta_t * (diff_corrs + sparsity * np.sign(j_ising))
        return new_j

    def _estimate_model_corr(self,
                             j_ising: np.ndarray,
                             h_ising: np.matrix) -> np.matrix:
        """
        Computes the model correlation given by Boltzmann distribution.
        Equation 2.12 of Ezaki 2017.
        Args:
            j_ising: Parameter J of the Ising model
            h_ising: Parameter h of the Ising model

        Returns:
            corr: Result of the model correlation given by Boltzmann
            distribution
        """
        tanh: np.matrix = self._estimate_tanh(j_ising, h_ising).T
        corr: np.matrix
        corr = np.matmul(self.binarized_data, tanh) / self.data_length
        model_correlation: np.matrix = 0.5 * (corr + corr.T)
        return model_correlation

    def _estimate_diff_corrs(self,
                             data_correlation: np.matrix,
                             j_ising: np.ndarray,
                             h_ising: np.matrix) -> np.matrix:
        """
        Estimates the difference between empirical correlation
        (data_correlation) and correlation given by Boltzmann distribution
        (_estimate_model_corr). This corresponds to the change of j for an
        Ezaki step. Used in Equation 2.10 in Ezaki 2017.
        Args:
            data_correlation: Empirical correlation computed from the data
            j_ising: Parameter J of the Ising model
            h_ising: Parameter h of the Ising model
        Returns:
            diff_corrs: The difference between the two mentioned correlations
        """
        # Compute model correlation
        model_correlation: np.matrix
        model_correlation = self._estimate_model_corr(j_ising, h_ising)
        # Compute difference between model and data correlation
        diff_corrs: np.matrix = model_correlation - data_correlation
        # Eliminate self connections
        diff_corrs = diff_corrs - np.diag(np.diag(diff_corrs))
        return diff_corrs

    # ---------------------------------------------------------------------
    # COMPUTE THE ARCHETYPE

    def _estimate_archetype(self,
                            iteration_max: int,
                            delta_t: float,
                            permissible_error: float,
                            estimate_h: bool,
                            sparsity: float) -> None:
        """
        Estimation of h and J in the archetype method

        Args:
            iteration_max: Maximum number of iterations for the method
            delta_t: Step size for the Ezaki method
            permissible_error: minimum error allowed by the method
            sparsity: L1 constraint multiplier

        Updates:
            self.inet.h_ising: Estimated h parameter of the Ising archetype.
            self.inet.j_ising: Estimated J parameter of the Ising archetype.
            self.inet.beta: set to 1 in the archetype.
            self.method_output["minimum error"] = Error of the solution w.r.t
            to the data.
            self.method_output["error evolution"] = Array with the evolution of
            the error along the method.
        """
        # INITIALIZE VARIABLES AND GET DIMENSIONS OF DATA
        # ---------------------------------------------------------------------
        # Compute mean (h) and correlation (J) of data
        data_mean: np.matrix = np.mean(self.binarized_data, axis=1)
        data_correlation: np.matrix
        data_correlation = ((self.binarized_data * self.binarized_data.T)
                            / self.data_length)
        # Initialize variables
        self.method_output: dict = {}
        h_ising: np.ndarray = np.zeros([self.lattice_size, 1])
        j_ising: np.ndarray = np.zeros([self.lattice_size, self.lattice_size])
        err: np.ndarray = np.zeros([iteration_max])
        iteration: int = 0
        # ---------------------------------------------------------------------
        for iteration in range(iteration_max):
            # Iterative method in Ezaki 2017 to maximize Pseudo-Likelihood
            # UPDATE H PARAMETER - Equation 2.9 in Ezaki 2017.
            # Compute change in h for this Ezaki step.
            diff_sigmas: np.matrix = \
                self._estimate_diff_sigmas(j_ising, h_ising, data_mean)
            # Update the h parameter of the Ising model if this is wanted
            if estimate_h:
                h_ising: np.matrix = self._update_h(h_ising, delta_t, diff_sigmas,sparsity)
            else: # deal with strange matrix side effedts
                h_ising= np.reshape(h_ising, [self.lattice_size, 1])
            # UPDATE J PARAMETER - Equation 2.10 in Ezaki 2017.
            # Compute change in J for this Ezaki step
            diff_corrs: np.matrix = \
                self._estimate_diff_corrs(data_correlation, j_ising, h_ising)
            # Update the J parameter of the Ising model
            j_ising: np.matrix = self._update_j(j_ising, delta_t, diff_corrs,sparsity)
            # -----------------------------------------------------------------
            # Compute error between our objective function (data) and the
            # estimated parameters (model). The error is the epsilon parameter
            # in Equations 2.9 and 2.10 of Ezaki 2017.
            err[iteration] = self._compute_error(diff_corrs, diff_sigmas)
            if iteration % 500 == 0:
                # Show number of steps and error every 500 steps
                print(f'number of steps: {iteration}',
                      f'norm: {err[iteration]}')

            if err[iteration] < permissible_error:
                # One of the stop conditions of the method, get a lower error
                # than the one we tolerate
                print(f'stopped after {iteration} steps')
                break
        # Update iNeT
        self.inet.h_ising = h_ising
        self.inet.j_ising = j_ising
        self.inet.beta = 1
        self.method_output["minimum error"] = err[iteration]
        self.method_output["error evolution"] = err

    # ---------------------------------------------------------------------
    # From here on, we have the functions to estimate an iNeT given an
    # archetype. Refer to TN0150.

    def _mean_si(self, beta: float) -> np.matrix:
        """
        Computes expected value of spin i, for the estimation of h. Equation 8
        of TN0150.
        """
        _hterm: np.matrix = np.matmul((self.archetype_h * beta),
                                      np.ones([1, self.data_length]))
        _jterm: np.matrix = np.matmul((self.archetype_j * beta),
                                      self.binarized_data)
        tanh: np.matrix = np.tanh(_hterm + _jterm)
        mean_si: np.matrix = np.mean(tanh, axis=1)
        return mean_si

    def _estimate_diff_si(self, data_mean: np.matrix,
                          beta: float) -> np.matrix:
        """
        Computes the difference between the model mean value Si (or h) and the
        data mean for each iteration of the the Ising parameters. Used in
        Equation 10 of TN0150.
        """
        model_si: np.matrix = self._mean_si(beta)
        model_si = np.reshape(model_si, [self.lattice_size, 1])
        diff_si: np.matrix = model_si - data_mean
        return diff_si

    def tanh_term(self, beta: float) -> np.matrix:
        """
        Computes the tangential term of Si_Sj. Used in Equation 9 of TN0150.
        """
        _hterm: np.matrix = np.matmul((self.archetype_h * beta),
                                      np.ones([1, self.data_length]))
        _jterm: np.matrix = np.matmul((self.archetype_j * beta),
                                      self.binarized_data)
        tanh: np.matrix = np.tanh(_hterm + _jterm)
        return tanh

    def _model_si_sj(self, beta: float) -> np.matrix:
        """
        Computes the expected value of SiSj. Used in Equation 9 of TN0150.
        """
        _tanh: np.matrix = self.tanh_term(beta).T
        si_sj: np.matrix
        si_sj = np.matmul(self.binarized_data, _tanh) / self.data_length
        model_si_sj: np.matrix = 0.5 * (si_sj + si_sj.T)
        return model_si_sj

    def _estimate_diff_si_sj(self,
                             data_correlation: np.matrix,
                             beta: float) -> np.matrix:
        """
        Computes the difference between the expected value of SiSj and the data
        correlation, for the estimation of J. Used in Equation 10 of TN0150.
        """
        # Compute model SiSj
        model_si_sj: np.matrix = self._model_si_sj(beta)
        # Compute difference between model SiSj and data correlation
        estimate_diff_si_sj: np.matrix = model_si_sj - data_correlation
        # Eliminate diagonal elements
        np.fill_diagonal(estimate_diff_si_sj, 0)
        return estimate_diff_si_sj

    # ---------------------------------------------------------------------
    # ESTIMATE iNeT FROM ARCHETYPE

    def _estimate_h_j(self, betas: np.ndarray) -> None:
        """
        Estimation of the Ising parameters using an archetype. Starting from an
        archetype iNeT, we fit a parameter beta (equal to the inverse
        temperature, 1/T) that makes the resulting scaled iNeT from the
        archetype be the best candidate to fit the data (data correlation and
        data mean).

        Args:
            betas: array with the values for beta search

        Updates:
            self.inet.h_ising: Estimated h parameter of the personalized iNeT.
            self.inet.j_ising: Estimated J parameter of the personalized iNeT.
            self.inet.beta: optimal beta to find the solution.
            self.method_output["minimum error"] = Error of the solution w.r.t
            to the data.
            self.method_output["error evolution"] = Array with the evolution of
            the error along the method.
        """
        # READ DATA AND INITIALIZE VARIABLES
        # Put the data in a convenient way for our operations
        self.binarized_data: np.matrix = np.matrix(self.binarized_data,
                                                   dtype=float)
        # Compute empirical mean and correlation of the data
        data_mean: np.matrix = np.mean(self.binarized_data, axis=1)
        data_correlation: np.matrix = \
            (self.binarized_data * self.binarized_data.T) / self.data_length

        # Initialize variables
        err: np.ndarray = np.zeros(len(betas))
        self.method_output = {}
        # ---------------------------------------------------------------------
        for idx, beta in enumerate(betas):
            # COMPUTE iNeT FROM ARCHETYPE
            # Compute difference between expected value of Si and SiSj by
            # boltzmann distribution and data mean and correlation
            # Mean activation (h or Si)
            estimate_diff_si: np.matrix = \
                self._estimate_diff_si(data_mean, beta)
            # Correlation (J or SiSj)
            estimate_diff_si_sj: np.matrix = \
                self._estimate_diff_si_sj(data_correlation, beta)
            # Compute error of the estimation
            err[idx] = self._compute_error(estimate_diff_si_sj,
                                           estimate_diff_si)
        # ---------------------------------------------------------------------
        # Save the result with the lowest error
        min_error: float = np.min(err)
        # Find the beta that minimized the error
        min_beta: float = betas[np.argmin(err)]
        # Compute the new h and J parameters: Equations 11 and 12 of TN0150.
        h_min: np.matrix = min_beta * self.archetype_h
        j_min: np.matrix = min_beta * self.archetype_j
        # Udate iNeT
        self.inet.h_ising = h_min
        self.inet.j_ising = j_min
        self.inet.beta = min_beta
        self.method_output["minimum error"] = min_error
        self.method_output["error evolution"] = err

    # ---------------------------------------------------------------------
    # itailor described in TN0237 - LSD-induced increase in Ising temperature.
    # From here on, we have the functions to estimate an iNeT given an
    # archetype. Refer to TN0237.

    def _mean_energy_i(self, beta: float) -> float:
        """
        Computes expected value of energy i as described in Equation 17 of
        TN0237.
        """
        # Calculate model Si
        mean_si: np.matrix = self._mean_si(beta)
        # Calculate h term of Hamiltonian
        sum_h: float = \
            np.sum(np.array(self.archetype_h) * np.array(mean_si))
        # Compute model SiSj
        model_si_sj: np.matrix = self._model_si_sj(beta)
        # Calculate J term of Hamiltonian
        _jterm: np.matrix = np.dot(self.archetype_j, model_si_sj)
        # Sum up terms in J term
        sum_j: float = 0.5 * np.trace(_jterm)
        # Hamiltonian: Equation 17 of TN0237
        mean_energy_i: float = - sum_j - sum_h
        return mean_energy_i

    def _estimate_diff_energy_i(self,
                                data_energy: float,
                                beta: float) -> float:
        """
        Computes the difference between the model and the empirical energy for
        each iteration of the the Ising parameters. Used in Equation 18 of
        TN0237.
        """
        # Compute model energy, H
        model_energy_i: float = self._mean_energy_i(beta)
        # Compute difference between model energy and data energy
        diff_energy_i: float = model_energy_i - data_energy
        return diff_energy_i

    def _compute_error_energy(self, diff_mean_energy_i: float) -> float:
        """
        Computes error for an iteration of the Ezaki method (error of energy).
        Used in Equations 18 of TN0237.

        Args:
            diff_mean_energy_i: difference between model and data energy

        Returns:
            error: the error for an estimation of energy, H, w.r.t data

        """
        _error_norm: float = np.sqrt(diff_mean_energy_i ** 2)
        error: float
        error = _error_norm / (self.lattice_size * (self.lattice_size + 1))
        return error

    def _estimate_energy(self, betas: np.ndarray) -> None:
        """
        Estimation of the Ising energy using an archetype. Starting from an
        archetype iNeT, we fit a parameter beta (equal to the inverse
        temperature, 1/T) that makes the resulting scaled iNeT from the
        archetype be the best candidate to fit the energy calculated from the
        data.

        Args:
            betas: array with the values for beta search

        Updates:
            self.inet.h_ising: Estimated h parameter of the personalized iNeT.
            self.inet.j_ising: Estimated J parameter of the personalized iNeT.
            self.inet.beta: optimal beta to find the solution.
            self.method_output["minimum error"] = Error of the solution w.r.t
            to the data.
            self.method_output["error evolution"] = Array with the evolution of
            the error along the method.

        """
        # READ DATA AND INITIALIZE VARIABLES
        # Put the data in a convenient way for our operations
        self.binarized_data: np.matrix = np.matrix(self.binarized_data,
                                                   dtype=float)
        # Compute empirical mean, correlation and energy of the data
        data_mean: np.matrix = np.mean(self.binarized_data, axis=1)
        data_correlation: np.matrix = \
            (self.binarized_data * self.binarized_data.T) / self.data_length

        # Compute the Hamiltonian or energy
        sum_h: float = np.sum(np.array(self.archetype_h) * np.array(data_mean))
        sum_j: float
        sum_j = 0.5 * np.trace(np.dot(self.archetype_j, data_correlation))
        data_energy: float = - sum_j - sum_h

        # Initialize variables
        err: np.ndarray = np.zeros(len(betas))
        self.method_output = {}
        # ---------------------------------------------------------------------
        for idx, beta in enumerate(betas):
            # COMPUTE iNeT FROM ARCHETYPE
            # Compute difference between model and empirical energy
            diff_energy_i: float
            diff_energy_i = self._estimate_diff_energy_i(data_energy, beta)
            # Compute error of the estimation
            err[idx] = self._compute_error_energy(diff_energy_i)
        # ---------------------------------------------------------------------
        # Save the result with the lowest error
        min_error: float = np.min(err)
        # Find the beta that minimized the error
        min_beta: float = betas[np.argmin(err)]
        # Compute the new h and J parameters: Equations 11 and 12 of TN0150.
        # (This is unchanged from previous method)
        h_min: np.matrix = min_beta * self.archetype_h
        j_min: np.matrix = min_beta * self.archetype_j
        # Update iNeT
        self.inet.h_ising = h_min
        self.inet.j_ising = j_min
        self.inet.beta = min_beta
        self.method_output["minimum error"] = min_error
        self.method_output["error evolution"] = err

    def run(self,
            method: str,
            params: dict,
            betas: np.ndarray = None) -> None:
        """
        Runs the itailor to either build an archetype model or to find the
        personalized temperature that best explains the data of each subject
        given an archetype, depending on the specified method.

        Args :
            method:
                method = "archetype": it builds an archetype. It takes
                binarized data from all subjects and estimates the h_ising and
                j_ising parameter for them.
                method = "personalized_H": it personalizes the archetype. It
                updates parameters of Ising model h_ising and j_ising and
                evolution of error and finds the beta that scales the original
                archetype and the minimum error between the scaled Ising model
                and data given that beta. This method minimizes the error
                between the model and empirical energy.
                method = "personalized_h_j": it personalizes the archetype, as
                above. However, this method minimizes the error between the
                mean spin correlation and activity from the model and from the
                data.
            params: Parameters for the archetype method run:
                {"delta_t" : float
                    Size of the step for the archetype estimation method
                "permissible_error" : float
                    minimum change allowed from step to step of the archetype
                    method.
                "iteration_max" : float
                    Maximum number of iterations for the archetype method.
                "estimate_h: bool
                    Set to False if h is not updated (clamped to init value).
                "sparsity" : float
                    sparsity penalty for L1 norm
                }
            betas: array with the values for beta search
        """
        # ---------------------------------------------------------------------
        # handle default values
        params.setdefault('permissible_error',1e-9)
        params.setdefault('delta_t', 0.05)
        params.setdefault('iteration_max', 1000)
        params.setdefault('sparsity', 0)
        params.setdefault('estimate_h', True)

        print("Itailor params:", params)

        # FIND iNeT FROM ARCHETYPE+DATA
        if method == "personalized_H":
            self._estimate_energy(betas)
        elif method == "personalized_h_j":
            # Compute h and J parameters
            self._estimate_h_j(betas)
        # ---------------------------------------------------------------------
        # BUILD AN ARCHETYPE FROM DATA
        elif method == "archetype":
            # Reformat data in convenient way for the operations in this method
            self.binarized_data: np.matrix = \
                np.matrix(self.binarized_data, dtype=float)
            # Compute Archetype
            self._estimate_archetype(params["iteration_max"],
                                     params["delta_t"],
                                     params["permissible_error"],
                                     params["estimate_h"],
                                     params["sparsity"])

        # ---------------------------------------------------------------------
        else:
            err_msg: str = "Method not valid, use archetype or personalized"
            logger.error(err_msg)
            raise ValueError(err_msg)

    def add_sc(self):
        """
        This function utilizes the structural connectome to add personalized
        features to the iNeT
        Returns: modified iNeT
        """
