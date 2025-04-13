from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
from quaccatoo import QSys, PulsedSim, Analysis, fit_rabi, Rabi, Hahn, square_pulse, NV, PMR, tensor, qeye, jmat, CPMG

@dataclass
class NVSystemParams():
    bfield: float = 0.09548 # Magnetic field in tesla
    theta: float = 0.9 * np.pi / 180. # Misalignment angle in radian
    nitrogen_isotope: int = 14
    rabi_frequency: float = 60.E6
    pulse_timesteps: int = 100
    mw_frequency: float = 196.1E6
    pi_pulse_duration: float = 0.5 / 60.E6
    solver_options: dict = field(default_factory= lambda: {})

def get_dqp_params():

    p = NVSystemParams(
        bfield=0.95,
        theta=0.9 * np.pi / 180.,
        nitrogen_isotope=14,
        rabi_frequency=60.E6,
        pulse_timesteps=100,
        mw_frequency=196.1E6,
        pi_pulse_duration=0.5 / 60.E6,
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

    A = np.empty((3, 3), dtype=object)
    A[0, 0] = -0.25
    A[0, 1] = -1.85
    A[0, 2] = -.49
    A[1, 0] = A[0, 1]
    A[1, 1] = 0
    A[1, 2] = .01
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]
    A[2, 2] = 1.01

    HhfC = get_carbon_hyperfine_hamiltonian(A)
    # define the nuclear Zeeman Hamiltonian for the 13C nuclear spin
    def Hzc(B0, theta):
        return 10.705e-3 * B0 * tensor(qeye(3), qeye(3), np.cos(theta * np.pi / 180) * jmat(1 / 2, 'z') + np.sin(
            theta * np.pi / 180) * jmat(1 / 2, 'x'))

    qsys.add_spin(HhfC + Hzc(qsys.B0, qsys.theta))

    return qsys

def get_carbon_hyperfine_hamiltonian(A):
    """
    Parameters
    ----------
    A: hyperfine tensor

    Returns
    -------

    """
    HhfC = (
            A[0, 0] * tensor(jmat(1, 'x'), qeye(3), jmat(1 / 2, 'x')) + A[0, 1] * tensor(jmat(1, 'x'), qeye(3),
                                                                                         jmat(1 / 2, 'y')) + A[
                0, 2] * tensor(jmat(1, 'x'), qeye(3), jmat(1 / 2, 'z'))
            + A[1, 0] * tensor(jmat(1, 'y'), qeye(3), jmat(1 / 2, 'x')) + A[1, 1] * tensor(jmat(1, 'y'), qeye(3),
                                                                                           jmat(1 / 2, 'y')) + A[
                1, 2] * tensor(jmat(1, 'y'), qeye(3), jmat(1 / 2, 'z'))
            + A[2, 0] * tensor(jmat(1, 'z'), qeye(3), jmat(1 / 2, 'x')) + A[2, 1] * tensor(jmat(1, 'z'), qeye(3),
                                                                                           jmat(1 / 2, 'y')) + A[
                2, 2] * tensor(jmat(1, 'z'), qeye(3), jmat(1 / 2, 'z'))
    )
    return HhfC

def run_rabi(p:NVSystemParams, qsys=None, pulse_durations=None):
    if qsys is None:
        qsys = get_qsys(p)


    if pulse_durations is None:
        pulse_durations = np.linspace(0., 10. / p.rabi_frequency, num=200)

    rabi_exp_1 = Rabi(
        pulse_duration=pulse_durations * 1.E6,  # pulse duration in us
        system=qsys,  # NV center system (sadly in MHz units until we fix Quaccatoo)
        H1=p.rabi_frequency * qsys.MW_H1 / 1.E6,  # control Hamiltonian MHz
        pulse_params={'f_pulse': p.mw_frequency / 1.E6},
        options=p.solver_options
    )
    for key in rabi_exp_1.options.keys():
        print(f'options[{key}={rabi_exp_1.options[key]}]')
    rabi_exp_1.run()
    # plot results

    rabi_analysis = Analysis(rabi_exp_1)
    rabi_analysis.plot_results()

def run_podmr(p:NVSystemParams, frequencies, qsys=None, pulse_duration=None):
    if qsys is None:
        qsys = get_qsys(p)

    if pulse_duration is None:
        pulse_duration = p.pi_pulse_duration


    rabi_MHz = p.rabi_frequency / 1.E6
    podmr_exp_1 = PMR(
        frequencies=frequencies / 1.E6,  # frequencies to scan in MHz
        pulse_duration=pulse_duration * 1.E6,  # pulse duration in us
        system=qsys,  # NV center system
        H1= rabi_MHz * qsys.MW_H1,  # control Hamiltonian
    )

    # run and plot the experiment
    podmr_exp_1.run()
    Analysis(podmr_exp_1).plot_results()



def run_hanh(p: NVSystemParams, free_durations, qsys=None, is_plot=False):
    """

    Parameters
    ----------
    p
    qsys
    free_durations
    is_plot

    Returns
    -------

    """
    if qsys is None:
        qsys = get_qsys(p)

    rabi_MHz = p.rabi_frequency / 1.E6
    if free_durations is None:
        free_durations = np.linspace(0.01E-6, 10.E-6, num=200)

    hahn_exp = Hahn(
        free_duration=free_durations * 1.E6,  # define the array of free durations to simulate (us)
        pi_pulse_duration = p.pi_pulse_duration * 1.E6,  # define the pi pulse duration (us)
        projection_pulse=True,
        # include the pi/2 pulse after the second free evolution (this line is redundant since it is the default value) !!! The projection pulse is the last pi/2-pulse. It is only present in NV measurements. The first one that is needed to bring the spin to the equator plane is used in NV experiments as well as in EPR/NMR experiments. Therefore, it is better to keep the first pulse always included, while leaving the last pulse optional !!!
        system = qsys,
        H1 = rabi_MHz * qsys.MW_H1,
        pulse_shape=square_pulse,
        pulse_params={'f_pulse': p.mw_frequency / 1.E6},
        time_steps=p.pulse_timesteps
    )

    if is_plot:
        hahn_exp.plot_pulses()
    hahn_exp.run()

    if is_plot:
        fft_freqs_MHz, fft_vals = Analysis(hahn_exp).run_FFT()
        fft_vals_db = 10. * np.log10(fft_vals)
        fig, ax = plt.subplots(2, 1)
        Analysis(hahn_exp).plot_results(ax=ax[0])
        ax[1].plot(fft_freqs_MHz, fft_vals)
        ax[1].set_xlabel('Frequency (MHz)')
        ax[1].set_ylabel('Amplitude')

def run_cpmg(p: NVSystemParams, free_durations, M=8, qsys=None, is_plot=False):
    """

    Parameters
    ----------
    p
    qsys
    free_durations
    is_plot

    Returns
    -------

    """
    if qsys is None:
        qsys = get_qsys(p)

    rabi_MHz = p.rabi_frequency / 1.E6
    if free_durations is None:
        free_durations = np.linspace(0.01E-6, 10.E-6, num=200)

    cpmg_exp = CPMG(
        M=M,
        free_duration=free_durations * 1.E6,  # define the array of free durations to simulate (us)
        pi_pulse_duration = p.pi_pulse_duration * 1.E6,  # define the pi pulse duration (us)
        projection_pulse=True,
        # include the pi/2 pulse after the second free evolution (this line is redundant since it is the default value) !!! The projection pulse is the last pi/2-pulse. It is only present in NV measurements. The first one that is needed to bring the spin to the equator plane is used in NV experiments as well as in EPR/NMR experiments. Therefore, it is better to keep the first pulse always included, while leaving the last pulse optional !!!
        system = qsys,
        H1 = rabi_MHz * qsys.MW_H1,
        pulse_shape=square_pulse,
        pulse_params={'f_pulse': p.mw_frequency / 1.E6},
        time_steps=p.pulse_timesteps
    )

    if is_plot:
        cpmg_exp.plot_pulses()
    cpmg_exp.run()

    if is_plot:
        fft_freqs_MHz, fft_vals = Analysis(cpmg_exp).run_FFT()
        fft_vals_db = 10. * np.log10(fft_vals)
        fig, ax = plt.subplots(2, 1)
        Analysis(cpmg_exp).plot_results(ax=ax[0])
        ax[1].plot(fft_freqs_MHz, fft_vals)
        ax[1].set_xlabel('Frequency (MHz)')
        ax[1].set_ylabel('Amplitude')



if __name__ == "__main__":
    p = NVSystemParams()
    p.nitrogen_isotope = 14
    p.rabi_frequency = 0.5 / (19.E-9)
    p.mw_frequency = 196.1E6
    p.pi_pulse_duration = 20.65E-9 #0.5 / p.rabi_frequency
    p.pulse_timesteps = 100
    #p.solver_options = {'nsteps': 10000,}
    qsys = get_qsys(p)

    # qsys.plot_energy()
    # plt.show()

    #frequencies = np.linspace(50.E6, 250.E6, num=200)
    # run_podmr(p, frequencies, qsys=qsys, pulse_duration=3.0 * p.pi_pulse_duration)
    #run_rabi(p, qsys=qsys, pulse_durations=np.linspace(1.E-9, 1.E-7, num=512))
    free_durations = np.linspace(1.E-8, 200.E-6, num=1024)
    #run_hanh(p, free_durations, qsys, is_plot=True)
    run_cpmg(p, free_durations, qsys=qsys, is_plot=True)
    plt.show()


