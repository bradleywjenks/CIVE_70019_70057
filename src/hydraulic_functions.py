"""
CIVE 70019 and 70057 modules
Department of Civil and Environmental Engineering, Imperial College London
Prepared by Bradley Jenks
June 2023

Functions used for hydraulic modelling of water networks
    - 'epanet_solver' hydraulic solver using EPANET via WNTR
    - 'nr_solver' hydraulic solver using newton-raphson method
    - 'nr_schur_solver' hydraulic solver using newton-raphson method with schur complement
    - 'null_space solver using method proposed in Abraham and Stoianov (2016)

"""

### import packages ###
import wntr
import networkx as nx
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import cholesky
from pydantic import BaseModel
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# improve matplotlib image quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')




"""
    misc. functions for hydraulic solver code
"""

### define head loss equations
def friction_loss(net_info, df):
    if net_info['headloss'] == 'H-W':
        K = 10.67 * df['length'] * (df['C'] ** -df['n_exp']) * (df['diameter'] ** -4.8704)
    else:
        K = [] # insert DW formula here...
    
    return K

def local_loss(df):
    K = (8 / (np.pi ** 2 * 9.81)) * (df['diameter'] ** -4) * df['C']
    
    return K





"""
    EPANET solver code
"""

def epanet_solver(inp_file):

    ## load network from wntr
    wn = wntr.network.WaterNetworkModel(inp_file)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    nt = int(wn.options.time.duration / wn.options.time.hydraulic_timestep)

    ## get hydraulic head results
    h_df = results.node['head'].T
    col_names_h = [f'h_{t}' for t in range(1, len(h_df.columns)+1)]
    h_df.columns = col_names_h
    h_df.reset_index(drop=False, inplace=True)
    h_df = h_df.rename(columns={'name': 'node_ID'})
    if nt > 1:
        h_df = h_df.iloc[:, :-1] # delete last time step

    reservoir_names = wn.reservoir_name_list
        

    ## get flow results
    q_df = results.link['flowrate'].T
    col_names_q = [f'q_{t}' for t in range(1, len(q_df.columns)+1)]
    q_df.columns = col_names_q
    q_df.reset_index(drop=False, inplace=True)
    q_df = q_df.rename(columns={'name': 'link_ID'})
    if nt > 1:
        q_df = q_df.iloc[:, :-1] # delete last time step


    return q_df, h_df







"""
    Hydraulic solver code
"""

def hydraulic_solver(wdn, method=None, print_status=False):

    ### Step 1: input network data and setup solver parameters

    # unload and compute network data
    A12 = wdn.A12
    A10 = wdn.A10
    net_info = wdn.net_info
    link_df = wdn.link_df
    node_df = wdn.node_df
    demand_df = wdn.demand_df
    h0_df = wdn.h0_df

    K = np.zeros((net_info['np'], 1))
    for idx, row in link_df.iterrows():
        if row['link_type'] == 'pipe':
            K[idx] = friction_loss(net_info, row)

        elif row['link_type'] == 'valve':
            K[idx] = local_loss(row)
            
    n_exp = link_df['n_exp'].astype(float).to_numpy().reshape(-1, 1)

    ## set stopping criteria
    tol = 1e-5
    kmax = 50 
    tol_A11 = 1e-5 # small values in A11 make convergence unsteady; therefore, we need to define a lower bound -- see Todini (1988), page 7

    # set solution arrays
    q = np.zeros((net_info['np'], net_info['nt']))
    h = np.zeros((net_info['nn'], net_info['nt']))


    # run over all time steps
    for t in range(net_info['nt']):
        
        ### Step 2: set initial values
        hk = 130 * np.ones((net_info['nn'], 1))
        qk = 0.03 * np.ones((net_info['np'], 1))

        # set boundary head and demand conditions
        dk = demand_df.iloc[:, t+1].to_numpy(); dk = dk.reshape(-1, 1)
        h0k = h0_df.iloc[:, t+1].to_numpy(); h0k = h0k.reshape(-1, 1)

        # initalize A11 matrix
        A11_diag = K * (abs(qk) ** (n_exp - 1)) # diagonal elements of matrix A11
        A11_diag[A11_diag < tol_A11] = tol_A11 # replace with small value = tol_A11
        A11 = sp.diags(A11_diag.T, [0]) # matrix A11, allocated as a sparse diagonal matrix

        if method == 'null_space':
            F_diag = n_exp * A11_diag # for null space method
            w = A12_fac(dk)
            x = A12 @ w
            x = x.reshape(-1, 1)

        # begin iterations
        for k in range(kmax):

            try:
                if method == 'nr':

                    ### Step 3: compute h^{k+1} and q^{k+1} for each iteration k
                    N = sp.diags(n_exp.T, [0]) # matrix N  
                    I = sp.eye(net_info['np'], format='csr') # identiy matrix with dimension np x np, allocated as a sparse matrix

                    b = np.concatenate([(N - I) @ A11 @ qk - A10 @ h0k, dk])
                    J = sp.bmat([[N @ A11, A12], [A12.T, sp.csr_matrix((net_info['nn'], net_info['nn']))]], format='csr')

                    # solve linear system
                    x = sp.linalg.spsolve(J, b)
                    qk = x[:net_info['np']]; qk = qk.reshape(-1, 1)
                    hk = x[net_info['np']:net_info['np'] + net_info['nn']];hk = hk.reshape(-1, 1)


                    # update A11 matrix
                    A11_diag = K * (abs(qk) ** (n_exp - 1)) # diagonal elements of matrix A11
                    A11_diag[A11_diag < tol_A11] = tol_A11 # replace with small value = tol_A11
                    A11 = sp.diags(A11_diag.T, [0]) # matrix A11, allocated as a sparse diagonal matrix


                elif method == 'nr_schur':

                    ### Step 3: compute h^{k+1} and q^{k+1}
                    inv_A11_diag = 1 / A11_diag; # diagonal elements of the inverse of A11
                    inv_A11 = sp.diags(inv_A11_diag.T, [0]) # inverse of A11, allocated as a sparse, diagonal matrix

                    inv_N = sp.diags(1/n_exp.T, [0]) # inverse of matrix N
                    
                    DD = inv_N @ inv_A11 # matrix inv_N * inv_A11

                    b = -A12.T @ inv_N @ (qk + inv_A11 @ (A10 @ h0k)) + A12.T @ qk - dk # right-hand side of linear system for finding h^{k+1]
                    A = A12.T @ DD @ A12 # Schur complement

                    # solve linear system for h^{k+1]
                    hk = sp.linalg.spsolve(A, b); hk = hk.reshape(-1, 1)
                    
                    # solve q^{k+1} by substitution
                    I = sp.eye(net_info['np'], format='csr') # identiy matrix with dimension np x np, allocated as a sparse matrix
                    qk = (I - inv_N) @ qk - DD @ ((A12 @ hk) + (A10 @ h0k))

                    # update A11 matrix
                    A11_diag = K * (abs(qk) ** (n_exp - 1)) # diagonal elements of matrix A11
                    A11_diag[A11_diag < tol_A11] = tol_A11 # replace with small value = tol_A11
                    A11 = sp.diags(A11_diag.T, [0]) # matrix A11, allocated as a sparse diagonal matrix
                    

            except:
                print('No solver method was inputted.')



            ### Step 4: convergence check 
            err = A11 @ qk + A12 @ hk + A10 @ h0k
            max_err = np.linalg.norm(err, np.inf)

            # print progress
            if print_status == True:
                print(f"Time step t={t+1}, Iteration k={k}. Maximum energy conservation error is {max_err} m.")

            if max_err < tol:
                # if successful,  break from loop
                break
                
        q[:, t] = qk.T
        h[:, t] = hk.T
        
    # convert results to pandas dataframe
    column_names_q = [f'q_{t+1}' for t in range(net_info['nt'])]
    q_df = pd.DataFrame(q, columns=column_names_q)
    q_df.insert(0, 'link_ID', link_df['link_ID'])

    column_names_h = [f'h_{t+1}' for t in range(net_info['nt'])]
    h_df = pd.DataFrame(h, columns=column_names_h)
    h_df.insert(0, 'node_ID', node_df['node_ID'])

    
    return q_df, h_df





