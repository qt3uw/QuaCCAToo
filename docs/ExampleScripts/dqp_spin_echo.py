from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from quaccatoo import QSys, PulsedSim, Analysis, fit_rabi, Rabi, Hahn, square_pulse, NV, PMR

@dataclass
class NVSystemParams():
    bfield: float = 0.09548 # Magnetic field in tesla
    theta: float = 0.9 * np.pi / 180. # Misalignment angle in radian
    nitrogen_isotope: int = 14
    rabi_frequency: float = 60.E6

def get_dqp_params():
    p = NVSystemParams(
        bfield=0.95,
        theta=0.9 * np.pi / 180.,
        nitrogen_isotope=14,
        rabi_frequency=60.E6,
    )
    return p

def get_qsys(p: NVSystemParams):
    qsys = NV(
        N=p.nitrogen_isotope,
        B0=p.bfield,  # external magnetic field in mT
        units_B0='T',
        theta=p.theta,
        units_angles='rad'
    )

    # A = np.empty((3, 3), dtype=object)
    # A[0, 0] = -0.25
    # A[0, 1] = -1.85
    # A[0, 2] = -.49
    # A[1, 0] = A[0, 1]
    # A[1, 1] = 0
    # A[1, 2] = .01
    # A[2, 0] = A[0, 2]
    # A[2, 1] = A[1, 2]
    # A[2, 2] = 1.01
    #
    # HhfC = (
    #         A[0, 0] * tensor(jmat(1, 'x'), qeye(3), jmat(1 / 2, 'x')) + A[0, 1] * tensor(jmat(1, 'x'), qeye(3),
    #                                                                                      jmat(1 / 2, 'y')) + A[
    #             0, 2] * tensor(jmat(1, 'x'), qeye(3), jmat(1 / 2, 'z'))
    #         + A[1, 0] * tensor(jmat(1, 'y'), qeye(3), jmat(1 / 2, 'x')) + A[1, 1] * tensor(jmat(1, 'y'), qeye(3),
    #                                                                                        jmat(1 / 2, 'y')) + A[
    #             1, 2] * tensor(jmat(1, 'y'), qeye(3), jmat(1 / 2, 'z'))
    #         + A[2, 0] * tensor(jmat(1, 'z'), qeye(3), jmat(1 / 2, 'x')) + A[2, 1] * tensor(jmat(1, 'z'), qeye(3),
    #                                                                                        jmat(1 / 2, 'y')) + A[
    #             2, 2] * tensor(jmat(1, 'z'), qeye(3), jmat(1 / 2, 'z'))
    # )
    #
    # # define the nuclear Zeeman Hamiltonian for the 13C nuclear spin
    # def Hzc(B0, theta):
    #     return 10.705e-3 * B0 * tensor(qeye(3), qeye(3), np.cos(theta * np.pi / 180) * jmat(1 / 2, 'z') + np.sin(
    #         theta * np.pi / 180) * jmat(1 / 2, 'x'))
    #
    # qsys.add_spin(HhfC + Hzc(qsys.B0, qsys.theta))

    return qsys

def run_rabi(p:NVSystemParams, qsys=None, pulse_durations=None):
    if qsys is None:
        qsys = get_qsys(p)

    rabi_MHz = p.rabi_frequency / 1.E6

    if pulse_durations is None:
        pulse_durations = np.linspace(0., 10. / rabi_MHz, num=200)

    else:
        pulse_durations = pulse_durations * 1.E6

    rabi_exp_1 = Rabi(
        pulse_duration=pulse_durations,  # pulse duration in us
        system=qsys,  # NV center system
        H1=rabi_MHz * qsys.MW_H1,  # control Hamiltonian
        pulse_params={'f_pulse': qsys.MW_freqs[0]}  # MW frequency for the ms=0 --> ms=-1 state transition
    )
    rabi_exp_1.run()
    # plot results

    rabi_analysis = Analysis(rabi_exp_1)
    rabi_analysis.plot_results()

def run_podmr(p:NVSystemParams, qsys=None, frequencies=None):
    if qsys is None:
        qsys = get_qsys(p)

    if frequencies is None:
        frequencies = np.linspace(qsys.MW_freqs[0] - 10., qsys.MW_freqs[0] + 10, 100)

    rabi_MHz = p.rabi_frequency / 1.E6
    podmr_exp_1 = PMR(
        frequencies=frequencies,  # frequencies to scan in MHz
        pulse_duration=1 / 2 / rabi_MHz,  # pulse duration
        system=qsys,  # NV center system
        H1=rabi_MHz * qsys.MW_H1,  # control Hamiltonian
    )

    # run and plot the experiment
    podmr_exp_1.run()
    Analysis(podmr_exp_1).plot_results()



def run_hanh(p: NVSystemParams, qsys=None, free_durations=None):
    if qsys is None:
        qsys = get_qsys(p)

    rabi_MHz = p.rabi_frequency / 1.E6
    if free_durations is None:
        free_durations = np.linspace(0.01, 10., num=200)

    hahn_exp = Hahn(
        free_duration=free_durations,  # define the array of free durations to simulate
        pi_pulse_duration=0.5 / rabi_MHz,  # define the pi pulse duration
        projection_pulse=True,
        # include the pi/2 pulse after the second free evolution (this line is redundant since it is the default value) !!! The projection pulse is the last pi/2-pulse. It is only present in NV measurements. The first one that is needed to bring the spin to the equator plane is used in NV experiments as well as in EPR/NMR experiments. Therefore, it is better to keep the first pulse always included, while leaving the last pulse optional !!!
        system=qsys,
        H1=rabi_MHz * qsys.MW_H1,
        pulse_shape=square_pulse,
        pulse_params={'f_pulse': qsys.MW_freqs[0]}, #FIXME
        time_steps=100
    )

    hahn_exp.plot_pulses()
    hahn_exp.run()
    Analysis(hahn_exp).plot_results()



if __name__ == "__main__":
    p = NVSystemParams()
    p.nitrogen_isotope = 0
    p.rabi_frequency = 0.5 / (100.E-9)
    qsys = get_qsys(p)

    # qsys.plot_energy()
    # plt.show()


    # run_podmr(p, qsys=qsys)
    run_hanh(p, qsys=qsys)# plot results
    # run_rabi(p, qsys=qsys)
    plt.show()


