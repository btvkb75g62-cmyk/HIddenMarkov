import numpy as np
import scipy.linalg as la
from BuildingSystems import BuildingSystems

class LowerLevelFiltering(BuildingSystems):

    def __init__(self, A0, AV, B0, BV, W0, WV, C0, M, Lambda, Prob1, Prob2, V, N, mu=1e8, beta=1.02):

        super().__init__(A0, AV, B0, BV, W0, WV, C0, M, Lambda, Prob1, Prob2)

        self.N = np.kron(np.eye(self.mode1*self.mode2), N)
        self.Vn = V
        self.beta = beta
        self.mu = mu

        self.Sigma = np.eye(self.nbar)  

        self.compute_Lagrange()
        self.compute_Phi()
        self.compute_Kernel()
        self.compute_RHS()

    def compute_Lagrange(self):

        MTM = self.StyleM.T @ self.StyleM

        self.Lagrange = self.beta * self.mu * np.linalg.norm(MTM, ord=2)

    def compute_Phi(self):
        self.Phi = (self.mu**(-1))*np.eye(self.nbar*self.V) - (self.Lagrange**(-1)) * np.dot(self.StyleM, self.StyleM.T)

    def compute_Kernel(self):
        
        Q = la.block_diag(self.Sigma, self.N, self.Vn, np.eye(self.nbar)*(self.mu**(-1)), np.eye(self.p)*(self.mu**(-1)), self.Phi, (self.Lagrange**(-1))*np.eye(self.nVert))

        eye_nbar = np.eye(self.nbar)
        eye_p = np.eye(self.p)
        zeros_nbar_nbar = np.zeros((self.nbar, self.nbar))
        zeros_nbar_p = np.zeros((self.nbar, self.p))
        zeros_p_nbar = np.zeros((self.p, self.nbar))
        zeros_nbarV_nbar = np.zeros((self.nbar*self.V, self.nbar))
        zeros_nbarV_p = np.zeros((self.nbar*self.V, self.p))

        zeros_2nbarV_nbar = np.zeros((self.nVert, self.nbar))
        zeros_2nbarV_p = np.zeros((self.nVert, self.p))

        A = np.block([[eye_nbar,          zeros_nbar_nbar,      zeros_nbar_p,    zeros_nbar_nbar,      zeros_nbar_nbar],
                      [zeros_nbar_nbar,   eye_nbar,             zeros_nbar_p,    zeros_nbar_nbar,      zeros_nbar_nbar],
                      [zeros_p_nbar,      zeros_p_nbar,         eye_p,           zeros_p_nbar,         zeros_p_nbar],
                      [eye_nbar,          zeros_nbar_nbar,      zeros_nbar_p,   -eye_nbar,             zeros_nbar_nbar],
                      [zeros_p_nbar,      zeros_p_nbar,         eye_p,           self.StyleC0,         zeros_p_nbar],
                      [zeros_nbarV_nbar,  self.StackedStyleI,   zeros_nbarV_p,   self.StackedStyleF0, -self.StackedStyleI],
                      [zeros_2nbarV_nbar, zeros_2nbarV_nbar,    zeros_2nbarV_p,  self.StackedStyleFV,  zeros_2nbarV_nbar]])



        self.Kernel = np.block([[Q, A],
                                [A.T, np.zeros((A.shape[1], A.shape[1]))]])
        
        self.Kernel_inv = np.linalg.solve(self.Kernel, np.eye(self.Kernel.shape[0]))

    def compute_RHS(self):

        I_StyleG0 =   np.hstack([np.zeros((self.nbar*self.V, self.nbar)), -self.StackedStyleG0, np.zeros((self.nbar*self.V, self.p))])
        I_StyleGV =   np.hstack([np.zeros((self.nVert, self.nbar)),       -self.StackedStyleGV, np.zeros((self.nVert, self.p))])
        I_Stylenbar = np.hstack([np.eye(self.nbar), np.zeros((self.nbar, self.mbar + self.p))])
        I_Stylep =    np.hstack([np.zeros((self.p, self.nbar + self.mbar)), np.eye(self.p)])

        zeros_nbar =  np.zeros((self.nbar, self.nbar + self.mbar + self.p))
        zeros_nVert = np.zeros((self.nVert, self.nbar + self.mbar + self.p))
        zeros_p =    np.zeros((self.p, self.nbar + self.mbar + self.p))

        LRHS = np.vstack([zeros_nbar, zeros_nbar, zeros_p , -I_Stylenbar, I_Stylep, I_StyleG0, I_StyleGV, zeros_nbar, zeros_nbar, zeros_p, zeros_nbar, zeros_nbar])

        zeros_nbar =  np.zeros((self.nbar, self.nbar))
        zeros_nVert = np.zeros((self.nVert, self.nbar))
        zeros_p =    np.zeros((self.p, self.nbar))

        RRHS = np.vstack([zeros_nbar, zeros_nbar, zeros_p ,zeros_nbar, zeros_p, np.zeros((self.nbar*self.V, self.nbar)), zeros_nVert, zeros_nbar, zeros_nbar, zeros_p, zeros_nbar, -np.eye(self.nbar)])

        self.RHS = np.hstack([LRHS, RRHS])

    def compute_LHS(self):

        zeros_nbar =  np.zeros((self.nbar, self.nbar))
        zeros_nVert = np.zeros((self.nVert, self.nbar))
        zeros_p =    np.zeros((self.p, self.nbar))

        self.LHS = np.vstack([zeros_nbar, zeros_nbar, zeros_p ,zeros_nbar, zeros_p, np.zeros((self.nbar*self.V, self.nbar)), zeros_nVert, zeros_nbar, zeros_nbar, zeros_p, zeros_nbar, np.eye(self.nbar)])

    def solve(self):

        self.compute_LHS()
        self.compute_RHS()

        gain_width = self.nbar + self.mbar + self.p

        for iteration in range(100):

            self.compute_Kernel()

            solution = np.dot(self.Kernel_inv, self.RHS)
            
            solution = np.dot(self.LHS.T, solution)

            self.S = solution[:, :gain_width]
            self.Sigma = solution[:, gain_width:gain_width + self.nbar]

            print(f"\nIteration {iteration+1}:")
            print(f"Sigma Norm = {np.linalg.norm(self.Sigma, ord=np.inf)}")
            print(f"S Norm = {np.linalg.norm(self.S, ord=np.inf)}")

        return solution


def main(A0, AV, B0, BV, W0, WV, C0, M, Lambda, Prob1, Prob2):

    V = np.eye(3)  # Placeholder for V
    N = np.eye(4)  # Placeholder for N

    LowerLevelSystem = LowerLevelFiltering(A0, AV, B0, BV, W0, WV, C0, M, Lambda, Prob1, Prob2, V, N, mu=1e8)

    solution = LowerLevelSystem.solve()

    # Export S as a CSV file with the floating points formatted to 2 decimal
    np.savetxt("S_solution.csv", LowerLevelSystem.S, delimiter=",", fmt="%.2f")
    np.savetxt("Sigma_solution.csv", LowerLevelSystem.Sigma, delimiter=",", fmt="%.2f")
    np.savetxt("Kernel.csv", LowerLevelSystem.Kernel, delimiter=",", fmt="%.2f")
    np.savetxt("Kernel_inv.csv", LowerLevelSystem.Kernel_inv, delimiter=",", fmt="%.2f")

if __name__ == "__main__":

    A0 = [np.array([[1.1, 0, 0], 
                   [0, 0, 1.2],
                   [-1, 1, 0]]),np.array([[1.1, 0, 0], 
                                          [0, 0, 1.2],
                                          [-1, 1, 0]])]  
    
    B0 = [np.array([[0, 1],
                   [1, 1],
                   [-1, 0]]), np.array([[0, 1],
                                        [1, 1],
                                        [-2, 0]])]  
    
    W0 = [np.array([[0.2],
                   [-0.5],
                   [0.7]]), np.array([[0.2],
                                      [-0.5],
                                      [0.7]])]  
    
    AV = [0*np.array([[1.1, 0, 0], 
                   [0, 0, 1.2],
                   [-1, 1, 0]]), 0*np.array([[1.1, 0, 0], 
                   [0, 0, 1.2],
                   [-1, 1, 0]])]  
    
    BV = [0*np.array([[0, 1],
                   [1, 1],
                   [-1, 0]]), 0*np.array([[0, 1],
                   [1, 1],
                   [-1, 0]])]  

    WV = [0*np.array([[0.2],
                   [-0.5],
                   [0.7]]), 0*np.array([[0.2],
                   [-0.5],
                   [0.7]])]  
    
    C0 = np.eye(3)

    M = [np.eye(3), np.eye(3)]

    Lambda = [np.array([[0.95]]), 
             np.array([[1.10]])]

    Prob1=np.array([[0.5, 0.5], 
                    [0.5, 0.5]])
    
    Prob2=np.array([[0.8, 0.2], 
                    [0.8, 0.2]])
    
    main(A0, AV, B0, BV, W0, WV, C0, M, Lambda, Prob1, Prob2)