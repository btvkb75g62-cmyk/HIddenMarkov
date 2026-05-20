import numpy as np
from scipy.linalg import block_diag


class BuildingSystems:
    def __init__(self, A, B, W, Lambda, Gamma, Prob1):

        self.A = A
        self.B = B
        self.W = W
        self.Gamma = Gamma
        self.Lambda = Lambda
        self.Prob1 = Prob1

        self.n = self.A[0].shape[0]
        self.l = self.W[0].shape[1]
        self.m = self.B[0].shape[1]
        self.p = self.n 

        self.mode1 = self.Prob1.shape[0]

        self.nbar = (self.l + self.n)*self.mode1
        self.mbar = self.m*self.mode1
        self.pbar = self.p*self.mode1

        self.compute_F()
        self.compute_G()
        self.compute_C()

        self.compute_EF()
        self.compute_EG()
        self.compute_EC()

        self.compute_Fstyle()
        self.compute_Gstyle()
        self.compute_Cstyle()



    def compute_F(self):
        self.F = [np.block([[self.A[i], self.W[i]],
                              [np.zeros((self.l, self.n)), self.Lambda[i]]]) for i in range(self.mode1)]
        
    def compute_G(self):
        self.G = [np.block([[self.B[i]],
                            [self.Gamma[i]]]) for i in range(self.mode1)]

    def compute_C(self):
        #self.C = [np.block([[np.eye(self.n), np.zeros((self.n, self.l))]]) for _ in range(self.mode1)]
        self.C = np.block([[np.eye(self.n), np.zeros((self.n, self.l))]])

    def compute_Fstyle(self):
        diagF = block_diag(*[self.F[i] for i in range(self.mode1)])

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            self.F = np.kron(self.Prob1.T, np.eye(self.n + self.l)) @ diagF

    def compute_Gstyle(self):
        diagG = block_diag(*[self.G[i] for i in range(self.mode1)])
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            self.G = np.kron(self.Prob1.T, np.eye(self.n + self.l)) @ diagG

    def compute_Cstyle(self):
        self.C = np.kron(np.ones(self.mode1).T, self.C)
    
    def compute_EF(self):
        self.EF = block_diag(*[self.F[i] for i in range(self.mode1)])

    def compute_EG(self):
        self.EG = block_diag(*[self.G[i] for i in range(self.mode1)])

    def compute_EC(self):
        #self.EC = block_diag(*[self.C[i] for i in range(self.mode1)])
        self.EC = np.zeros_like(np.kron(np.ones(self.mode1).T, self.C))

def main(A, B, W, Lambda, Gamma, Prob1):
    
    StackedSystem = BuildingSystems(A, B, W, Lambda, Gamma, Prob1)



if __name__ == "__main__":

    A = [0.9*np.array([[1, 0, 0], 
                   [0, 0, 1],
                   [-1, 1, 0]]),np.array([[1, 0, 0], 
                                          [0, 0, 1],
                                          [-1, 1, 0]])]  
    
    B = [np.array([[0, 1],
                   [1, 1],
                   [-1, 0]]), np.array([[0, 1],
                                        [1, 1],
                                        [-2, 0]])]  
    
    W = [np.array([[0],
                   [0],
                   [0]]), np.array([[0.2],
                                    [-0.5],
                                    [0.7]])]  

    Lambda = [np.array([[0]]), 
             np.array([[1]])]

    Gamma = [np.array([[1, 0]]), 
             np.array([[0, 0]])]

    Prob1=np.array([[0.5, 0.5], 
                    [0.5, 0.5]])
    
    
    main(A, B, W, Lambda, Gamma, Prob1)