import matplotlib.pyplot as plt
import numpy as np
from qutip import tensor, jmat, qeye

from quaccatoo import NV, XY8, RXY8, Analysis

if __name__ == "__main__":
    # define the NV center system
    qsys = NV(
        N=14,
        B0=95,  # external magnetic field in mT
        units_B0='mT',
        theta=0.9,
        units_angles='deg'
    )

    w1 = 20

    XY8_15N = XY8(
        M=2,
        free_duration=np.linspace(.25, .36, 100),  # free evolution time in us
        pi_pulse_duration=1 / 2 / w1,  # pi-pulse duration in us
        system=qsys,  # NV center system
        H1=w1 * qsys.MW_H1,  # control Hamiltonian
        pulse_params={'f_pulse': qsys.MW_freqs[1]},  # MW frequency for the ms=0 --> ms=+1 state transition
        time_steps=100  # Number of time steps in each MW pulse
    )

    # plot the pulses
    XY8_15N.plot_pulses(tau=.1, figsize=(12, 4))

    XY8_15N.run()
    # Analysis(XY8_15N).plot_results(title=r'XY8-2 $B_0=39$ mT $\theta_0=2.6^\circ$')
    # plt.show()