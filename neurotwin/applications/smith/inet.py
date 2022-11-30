"""
SMITH
inet
-------------------------------------------------------------------------------
This module is used to create iNeT objects from a given set of parameters
of the Ising model, h and J. This objects are able to simulate patterns
based on the data, as well as simulate scaled versions of the system, i.e,
variate the temperature of the system to move it in the phase space and be
able to get information regarding criticality.
"""
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from neurotwin.applications.smith.spinglass import run_ising, entropy


class Inet:
    """
    This class defines an Ising Neurotwin. The methods available lets us
    take the attributes that define an iNeT and simulate data running a
    Metropolis algorithm.

    Contains the statistics dictionary that stores the numerical results of
    the statistical physical variables resulting from the metropolis run.
        {"local_susceptibility" : np.ndarray. Vector containing the
            susceptibility for each node of the lattice.
        "heat_capacity" : float. Contains the heat capacity value for a
            metropolis simulation.
        "susceptibility" : float. Contains the susceptibility value for
            a metropolis simulation.
        "magnetization" : np.ndarray. Contains the mean magnetization
            for each step of the metropolis run.
        "energy" : np.ndarray. Contains the mean energy for each step
            of the metropolis run.
         "simulated_data" : np.ndarray. Contains all the lattice
            patterns simulated in the metropolis run.}
    """

    def __init__(self,
                 j_ising: Union[np.matrix, np.ndarray],
                 h_ising: Union[np.matrix, np.ndarray],
                 beta: float = None) -> None:
        """
        Args:
            h_ising: parameter h of Ising model, vector of dimension N, where N
                is the dimension of the lattice, i.e., number of spins in the
                lattice, for example if EEG is used, this would be number of
                electrodes, or if it's fMRI data, this would be number of ROIs
            j_ising: parameter J of Ising model (pairwise connectivity), array
                of dimension NxN
            beta: Scaling factor for the personalization (w.r.t archetype)
        """
        self.statistics: dict = {}
        self.h_ising = np.array(h_ising)
        self.j_ising = np.array(j_ising)
        self.beta: float = beta
        self.magne_history: dict = {}
        self.ene_history: dict = {}
        self.lattice_history = []


    def run_metropolis(self,
                       n_steps: int,
                       temperature: float,
                       seed: Optional[int] = None) -> None:
        """
        Runs metropolis algorithm to simulate data from the parameters h and J
        following the Boltzmann probability distribution. Computes as well all
        the variables of interest for the system; Magnetization, energy, heat
        capacity, susceptibility and susceptibility per node.
        Local susceptibility has a value for each node. Susceptibility is just
        one value for the whole system, as well as for heat capacity.
        Magnetization and energy are vector containing the magnetization and
        energy for each step of the metropolis algorithm. And Simulated data
        has the pattern (lattice dimension) for each step of the metropolis
        algorithm.
        Args:
            n_steps: Number of iterations in the metropolis algorithm
            temperature: Temperature at which we want to set the system and
                simulate the metropolis algorithm
            seed: In case you want to fix the seed for the random numbers,
                introduce an integer here. If = None, seed won't be fixed.
        """

        ave_M, global_chi, ave_lattice, local_chi, ave_E, Cv, lattice_history = run_ising(
            self.j_ising, self.h_ising, temperature, n_steps)

        self.statistics["local_susceptibility"]: np.ndarray = local_chi
        self.statistics["heat_capacity"]: float = Cv
        self.statistics["susceptibility"]: float = global_chi
        self.statistics["magnetization"]: np.ndarray = ave_M
        self.statistics["energy"]: float = ave_E
        self.statistics["<lattice>"]: np.ndarray =  ave_lattice
        self.statistics["q_EA"]: float = np.mean(ave_lattice**2)
        self.lattice_history =  lattice_history
        lattice_history_bool = np.array((1 + np.array(lattice_history)) / 2, dtype='bool')

        entropies = [entropy(lattice_history_bool[n]) for n in range(len(lattice_history_bool))]
        self.statistics['entropy'] = np.mean(entropies)
        self.statistics['sigma_entropy']= np.std(entropies)


    def simulate_and_plot(self,
                          temperatures: np.ndarray,
                          iterations: int) \
            -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates metropolis algorithm for the selected temperatures and plots
        the variable of interest.
        Args:
            iterations: Iterations for each metropolis algorithm performed
            temperatures: An array with initial, final temperature and number
                of temperature n_points to explore (for example,
                np.linspace(0.0, 3.0, 10) to explore 10 n_points between T=0
                and T=3)
        """
        n_points: int = temperatures.shape[0]
        heats: np.ndarray = np.zeros(n_points)
        susceptibilities: np.ndarray = np.zeros(n_points)
        energies: np.ndarray = np.zeros(n_points)
        magnetizations: np.ndarray = np.zeros(n_points)
        q_EAs: np.ndarray = np.zeros(n_points)
        entropies: np.ndarray = np.zeros(n_points)


        for point in range(n_points):
            temperature: float = temperatures[point]
            self.run_metropolis(iterations, temperature)
            heats[point] = self.statistics["heat_capacity"]
            susceptibilities[point] = self.statistics["susceptibility"]
            energies[point] = self.statistics["energy"]
            magnetizations[point] = self.statistics["magnetization"]
            q_EAs[point] = self.statistics["q_EA"]
            entropies[point] = self.statistics["entropy"]


        fig, axs = plt.subplots(6, figsize=(8, 14), sharex=True)
        fig.suptitle("iNeT temperature profile", fontsize=16)
        axs[0].plot(temperatures, heats)
        axs[0].set(xlabel=r'$T$', ylabel=r"$C_v$")

        axs[1].plot(temperatures, susceptibilities)
        axs[1].set(xlabel=r'$T$', ylabel=r"$\chi$")

        axs[2].plot(temperatures, magnetizations)
        axs[2].set(xlabel=r'$T$', ylabel=r"$M$")

        axs[3].plot(temperatures, energies)
        axs[3].set(xlabel=r'$T$', ylabel="Energy")

        axs[4].plot(temperatures, q_EAs)
        axs[4].set(xlabel=r'$T$', ylabel=r"$q_{EA}$")

        axs[5].plot(temperatures, entropies)
        axs[5].set(xlabel=r'$T$', ylabel="entropy")

        plt.tight_layout()
        plt.show()

        print("Tc:", np.round(temperatures[np.argmax(susceptibilities)], 3))
        print("Max chi:", np.round(np.max(susceptibilities), 2))

        return (heats,
                susceptibilities,
                energies,
                magnetizations, q_EAs, entropies)


    def evaluate_criticality(self,
                             Trange=np.arange(0.5, 2.02, 0.02),
                             Nit=10000000, n_processors=14):

        from neurotwin.applications.smith.spinglass import (
        run_jobs, do_job_plots, plot_complexities,plot_entropies, compute_complexity)

        """Run model for different Temperatures to produce phase plots"""

        tasks = [{'J': self.j_ising, 'h': self.h_ising, 'kT': kT, 'Nit': Nit} for kT in Trange]
        Ts, mags, chis, ave_lattice, local_chis, Es, Cvs, lattice_histories = run_jobs(tasks,
                                                                                       n_processors=n_processors)

        do_job_plots(Ts, mags, chis, ave_lattice, local_chis, Es, Cvs, Nit)

        #e_s, e_std_s, rho0_s, rho0_std_s = compute_complexity(lattice_histories[:, 0:3000000, :])
        #plot_entropies(Ts, e_s, e_std_s)
        #plot_complexities(Ts, rho0_s - e_s, rho0_std_s - e_std_s)

        # Tc = Ts[np.argmax(chis)]
        Tc = Ts[np.argmax(Cvs)]
        plt.show()

        Ts = np.array(Ts)

        ##### Find most susceptible node at T=1
        loss = (Ts - 1) ** 2
        T_loc = np.where(loss == np.min(loss))[0][0]  # stupid singleton dims of np.where
        local_chi_at_T1 = local_chis[:, T_loc]
        # node where max chi is attained at its chi
        node_max = np.argmax(local_chi_at_T1)

        ###

        print("Tc:", np.round(Tc, 3))
        print("Max chi:", np.round(np.max(chis), 2))
        print("Most susceptible node at T=1:", node_max,
              "with local chi=", np.round(local_chi_at_T1[node_max], 2))

        return Tc, np.max(chis), local_chis