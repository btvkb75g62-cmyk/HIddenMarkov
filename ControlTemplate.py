import numpy as np
import scipy.linalg as la
from BuildingSystems import BuildingSystems

class RobustRegulator(object):
    
    def __init__(self, F, EF, G, EG, Q, R, mu=1e8, beta=1.01):

        self.F = F
        self.EF = EF
        self.G = G
        self.EG = EG

        self.Q = np.kron(np.eye(self.mode1), Q)
        self.R = np.kron(np.eye(self.mode1), R)

        self.mu = mu
        self.beta = beta

    def make_Phi(self):

        self.nbar = self.F.shape[0]
        self.mbar = self.G.shape[1]

        self.P = np.eye(self.F.shape[0])

        self.StyleF = np.vstack([self.F, self.EF])
        self.StyleG = np.vstack([self.G, self.EG])
        self.StyleI = np.vstack([np.eye(self.F.shape[0]), np.zeros((self.EF.shape[0], self.F.shape[0]))])

        self.R_inv = np.linalg.solve(self.R, np.eye(self.R.shape[0]))
        self.Q_inv = np.linalg.solve(self.Q, np.eye(self.Q.shape[0]))

        self.Lagrange = self.mu * self.beta

        self.Phi = la.block_diag((self.mu**(-1) - self.Lagrange**(-1))*np.eye(self.nbar), self.Lagrange**(-1)*np.eye(self.nbar))

    def compute_Kernel(self):
        
        self.P_inv = np.linalg.solve(self.P, np.eye(self.P.shape[0]))

        W = la.block_diag(self.P_inv, self.R_inv, self.Q_inv, self.Phi)

        A = np.block([[np.eye(self.nbar), np.zeros((self.nbar, self.mbar))],
                      [np.zeros((self.mbar, self.nbar)), np.eye(self.mbar)],
                      [np.zeros((self.nbar, self.nbar)), np.zeros((self.nbar, self.mbar))],
                      [self.StyleI, -self.StyleG]])

        self.Kernel = np.block([[W, A],
                                [A.T, np.zeros((A.shape[1], A.shape[1]))]])
        
        self.Kernel_inv = np.linalg.solve(self.Kernel, np.eye(self.Kernel.shape[0]))

    def compute_RHS(self):

        RHS = np.vstack([np.zeros((self.nbar, self.nbar)), np.zeros((self.mbar, self.nbar)),-np.eye(self.nbar), self.StyleF, np.zeros((self.nbar, self.nbar)), np.zeros((self.mbar, self.nbar))])

        self.RHS = RHS

    def compute_LHS(self):

        self.LLHS = np.vstack([np.zeros((self.nbar, self.mbar)), np.zeros((self.mbar, self.mbar)), np.zeros((self.nbar, self.mbar)), np.zeros((self.nbar, self.mbar)), np.zeros((self.nbar, self.mbar)), np.zeros((self.nbar, self.mbar)), np.eye(self.mbar)])
        self.RLHS = np.vstack([np.zeros((self.nbar, self.nbar)), np.zeros((self.mbar, self.nbar)), -np.eye(self.nbar),               self.StyleF,                      np.zeros((self.nbar, self.nbar)), np.zeros((self.mbar, self.nbar))])

    def run_Riccati(self, max_iterations=1000, tol=1e-5, verbose=False):

        P = self.P

        for _ in range(max_iterations):

            self.compute_Kernel()

            self.P = self.RLHS.T @ self.Kernel_inv @ self.RHS

            if verbose:
                print(f"Riccati iteration {_+1}, Frobenius norm of difference: {np.linalg.norm(self.P - P, ord='fro'):.6e}")

            if np.linalg.norm(self.P - P, ord='fro') < tol:
                break
                
            P = self.P

        self.K = self.LLHS.T @ self.Kernel_inv @ self.RHS

class DynamicRegulator(BuildingSystems, RobustRegulator):

    def __init__(self, A0, B0, W0, Lambda, Gamma, Prob1, Q, R, V, N):

        BuildingSystems.__init__(self, A0, B0, W0, Lambda, Gamma, Prob1)
        RobustRegulator.__init__(self, self.F, self.EF, self.G, self.EG, Q, R)

        self.V = V
        self.N = N

    def solve(self, verbose=True):

        # Solve Control Problem
        self.make_Phi()
        self.compute_LHS()
        self.compute_RHS()
        self.run_Riccati()

        self.Pcontrol = self.P
        self.Kcontrol = self.K

        # Solve Filter Problem

        self.F = self.F.T
        self.G = self.C.T
        self.EF = self.EF.T
        self.EG = self.EC.T

        self.Q = np.kron(np.eye(self.mode1), self.N)
        self.R = self.V

        self.make_Phi()
        self.compute_LHS()
        self.compute_RHS()
        self.run_Riccati()

        self.Pfilter = self.P
        self.Lfilter = self.K.T


        if verbose:
            print("\nControl Gain K:")
            print(self.Kcontrol)
            print("\nFilter Gain L:")
            print(self.Lfilter)

            # Save to CSV
            np.savetxt("Control_Gain_K.csv", self.Kcontrol, delimiter=",", fmt="%.4f")
            np.savetxt("Control_P.csv", self.Pcontrol, delimiter=",", fmt="%.4f")
            np.savetxt("Filter_P.csv", self.Pfilter, delimiter=",", fmt="%.4f")
            np.savetxt("Filter_Gain_L.csv", self.Lfilter, delimiter=",", fmt="%.4f")


if __name__ == "__main__":

    h = 0.05

    A = [np.array([[1, 0],
                    [0, 1]]),np.array([[1, 0],
                                          [0, 1]])]  
    
    B = [np.array([[10*h, h, h],
                   [0.2*h, -h, h]]), np.array([[10*h, 0, h],
                                               [0.2*h, 0, h]])]  
    
    W = [np.array([[0, 0, 0],
                   [0, 0, 0]]), np.array([[0, h, 0],
                                          [0, -h, 0]])]  

    Lambda = [np.array([[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]), 
                                   np.array([[0, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, 0]])]

    Gamma = [np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]), np.array([[1, 0, 0],
                                              [0, 0, 0],
                                              [0, 0, 1]])]

    Prob1=np.array([[0.95, 0.05], 
                    [0.05, 0.95]])
    
    
    dynReg = DynamicRegulator(A, B, W, Lambda, Gamma, Prob1, Q=np.eye(5), R=np.eye(3), V=np.eye(2), N=np.eye(5))
    dynReg.solve()

    system = BuildingSystems(A, B, W, Lambda, Gamma, Prob1)

    # See Control Stability
    eigvals = np.linalg.eigvals(system.F + system.G @ dynReg.Kcontrol)
    print("\nMax abs eigenvalues of the closed-loop control system:")
    print(np.max(np.abs(eigvals)))

    # See Filter Stability
    eigvals = np.linalg.eigvals(system.F + dynReg.Lfilter @ system.C)
    print("\nMax abs eigenvalues of the closed-loop filter system:")
    print(np.max(np.abs(eigvals)))

    # See robustness metric
    robustness_metric = np.linalg.norm(system.EF + system.EG @ dynReg.Kcontrol, ord=np.inf)
    print("\nRobustness metric (norm of EF + EG @ Kcontrol):")
    print(robustness_metric)

    # See robustness metric
    robustness_metric = np.linalg.norm(system.EF + dynReg.Lfilter @ system.C, ord=np.inf)
    print("\nRobustness metric (norm of EF + Lfilter @ C):")
    print(robustness_metric)