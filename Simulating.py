import numpy as np
from scipy.signal import butter, filtfilt
from BuildingSystems import BuildingSystems
from ControlTemplate import DynamicRegulator

def simulate(A, B, W, Lambda, Gamma, Prob1, L, K, StyleF, StyleG, x0 = None, w0=None, Horizon = 500, h = 0.05, noise_cutoff_hz = 1.0):

    m = B[0].shape[1]
    l = W[0].shape[1]
    n = A[0].shape[0]

    mode1 = Prob1.shape[0]

    StyleC = np.kron(np.ones(mode1).T, np.hstack([np.eye(n), np.zeros((n, l))])) 

    if x0 is None:
        x = np.ones((n, 1))
    else:
        x = x0
    if w0 is None:
        w = np.ones((l, 1))
    else:
        w = w0

    q = np.zeros(((n+l)*mode1, 1))
    c = np.zeros((m*mode1, 1))
    y = np.zeros((n, 1))
    u = np.zeros((m, 1)) 

    idx1 = np.random.randint(mode1)

    # Store the sequence of states, controls, and disturbances for analysis
    state_sequence = []
    control_sequence = []
    disturbance_sequence = []
    time_sequence = []

    state_estimate_sequence = []

    # Pre-generate process noise and low-pass filter it to <noise_cutoff_hz
    raw_noise = np.random.multivariate_normal(np.zeros(n), np.eye(n), size=Horizon)  # (Horizon, n)
    fs = 1.0 / h
    b, a = butter(4, noise_cutoff_hz, btype='low', fs=fs)
    filtered_noise = filtfilt(b, a, raw_noise, axis=0)  # zero-phase, shape (Horizon, n)

    for k in range(Horizon):

        z_estimated = np.kron(np.ones(mode1).T, np.eye(n+l)) @ q  # Extract the state estimate from q

        time_sequence = np.append(time_sequence, k)  # Store the time step
        state_sequence.append(x.flatten())
        control_sequence.append(u.flatten())
        disturbance_sequence.append(w.flatten())
        state_estimate_sequence.append(z_estimated.flatten())  # Store the state estimate from q

        #idx1 = np.random.choice(mode1, p=Prob1[idx1])
        idx1 = 0
        if k > 100: idx1 = 0
        if k > 200: idx1 = 0

        A_k = A[idx1]
        B_k = B[idx1]
        W_k = W[idx1]
        Lambda_k = Lambda[idx1]
        Gamma_k = Gamma[idx1]

        q = StyleF @ q + StyleG @ c + L @ (StyleC @ q - x) 

        c = K @ q

        u = np.kron(np.ones(mode1).T, np.eye(m)) @ c

        if k > 0 and k < 200: noise = filtered_noise[k].reshape(-1, 1)
        else: noise = np.zeros((n, 1))

        x = A_k @ x + W_k @ w + B_k @ u + noise
        w = Lambda_k @ w + Gamma_k @ u

    return np.array(time_sequence), np.array(state_sequence), np.array(control_sequence), np.array(disturbance_sequence), np.array(state_estimate_sequence)

def main_sim():

    h = 0.05

    A = [np.array([[1, 0],
                    [0, 1]]),np.array([[1, 0],
                                          [0, 1]]), np.array([[1, 0],
                                                              [0, 1]])]
    
    B = [np.array([[10*h, h, h],
                   [0.2*h, -h, h]]), np.array([[0, h, h],
                                               [0, -h, h]]), np.array([[0, h, 0],
                                                                       [0, -h, 0]])]
    
    W = [np.array([[0, 0, 0],
                   [0, 0, 0]]), np.array([[10*h, 0, 0],
                                          [0.2*h, 0, 0]]), np.array([[10*h, 0, 0],
                                                                     [0.2*h, 0, 0]])]

    Lambda = [np.array([[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]), 
                                   np.array([[1, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0]]), np.array([[1, 0, 0],
                                                                    [0, 0, 0],
                                                                    [0, 0, 0]])]

    Gamma = [np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]), np.array([[0, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 1]]),
                                                        np.array([[0, 0, 0],
                                                                  [0, 1, 0],
                                                                  [0, 0, 1]])]


    Prob=np.array([[0.950, 0.035, 0.015], 
                   [0.000, 0.850, 0.015],
                   [0.000, 0.050, 0.950]])
    
    N = np.diag([1, 1, 1e4, 1, 1])
    V = np.eye(2)

    dynReg = DynamicRegulator(A, B, W, Lambda, Gamma, Prob, Q=np.eye(5), R=np.eye(3), V=V, N=N)
    dynReg.solve()

    system = BuildingSystems(A, B, W, Lambda, Gamma, Prob)

    x0 = np.array([[0], [0]])
    w0 = np.array([[0], [0], [0]])

    Time, State, Control, Disturbance, state_estimate_sequence = simulate(A, B, W, Lambda, Gamma, Prob, dynReg.Lfilter, dynReg.Kcontrol, system.F, system.G, x0, w0)

    # Plotting the results
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'legend.frameon': False,
        'lines.linewidth': 1.6,
    })

    n_states   = State.shape[1]
    n_controls = Control.shape[1]
    n_dist     = Disturbance.shape[1]
    colors     = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- State & Estimate ---
    fig, axes = plt.subplots(n_states, 1, figsize=(9, 2.8 * n_states), sharex=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.plot(Time, State[:, i],
                color=colors[i], label=f'$x_{i+1}$ (true)')
        ax.plot(Time, state_estimate_sequence[:, i],
                color=colors[i], linestyle='--', alpha=0.75, label=f'$\\hat{{x}}_{i+1}$ (estimate)')
        ax.set_ylabel(f'$x_{i+1}$')
        ax.legend(loc='upper right')
    axes[-1].set_xlabel('Time step')
    fig.suptitle('State trajectory', fontweight='bold')
    fig.tight_layout()

    # --- Control ---
    fig, axes = plt.subplots(n_controls, 1, figsize=(9, 2.2 * n_controls), sharex=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.plot(Time, Control[:, i], color=colors[i], label=f'$u_{i+1}$')
        ax.set_ylabel(f'$u_{i+1}$')
        ax.legend(loc='upper right')
    axes[-1].set_xlabel('Time step')
    fig.suptitle('Control input', fontweight='bold')
    fig.tight_layout()

    # --- Disturbance ---
    fig, axes = plt.subplots(n_dist, 1, figsize=(9, 2.2 * n_dist), sharex=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.plot(Time, Disturbance[:, i], color=colors[i], label=f'$w_{i+1}$')
        ax.set_ylabel(f'$w_{i+1}$')
        ax.plot(Time, state_estimate_sequence[:, i+n_states],
            color=colors[i], linestyle='--', alpha=0.75, label=f'$\\hat{{x}}_{i+n_states+1}$ (estimate)')
        ax.legend(loc='upper right')
    axes[-1].set_xlabel('Time step')
    fig.suptitle('Disturbance', fontweight='bold')
    fig.tight_layout()

    plt.show()



if __name__ == "__main__":


    main_sim()