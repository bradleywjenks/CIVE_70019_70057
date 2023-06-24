"""
CIVE 70019 and 70057 modules
Department of Civil and Environmental Engineering, Imperial College London
Prepared by Bradley Jenks
June 2023

Functions used for modelling and optimization of water networks
    - 'load_data' using WNTR Python package
    - 'plot_network' function
    - 'nr_solver' hydraulic solver using newton-raphson method
    - 'nr_schur_solver' hydraulic solver using newton-raphson method with schur complement

"""

### import packages ###
import wntr
import networkx as nx
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.cm as cm







"""
Load network data via wntr
""" 

# class NullData(BaseModel):
#     Pr: any # cholesky derived
#     L_A12: any# cholesky derived
#     Z: any

class WDN(BaseModel):
    A12: Any
    A10: Any
    net_info: dict
    link_df: pd.DataFrame
    node_df: pd.DataFrame
    demand_df: pd.DataFrame
    h0_df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


def load_network_data(inp_file):
    
    wdn = object()

    ## load network from wntr
    wn = wntr.network.WaterNetworkModel(inp_file)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    ## get network elements and simulation info
    nt = int(wn.options.time.duration / wn.options.time.hydraulic_timestep)
    nt = nt if nt>0 else 1
    net_info = dict(
        np=wn.num_links,
        nn=wn.num_junctions,
        n0=wn.num_reservoirs,
        nt=nt,
        headloss=wn.options.hydraulic.headloss,
        units=wn.options.hydraulic.inpfile_units,
        reservoir_names=wn.reservoir_name_list,
        junction_names=wn.junction_name_list,
        pipe_names=wn.pipe_name_list,
        valve_names=wn.valve_name_list,
        prv_names=wn.prv_name_list
    )

    
    ## extract link data
    if net_info['headloss'] == 'H-W':
        n_exp = 1.852
    elif net_info['headloss'] == 'D-W':
        n_exp = 2

    link_df = pd.DataFrame(
        index=pd.RangeIndex(net_info['np']),
        columns=['link_ID', 'link_type', 'diameter', 'length', 'n_exp', 'C', 'node_out', 'node_in'],
    ) # NB: 'C' denotes roughness or HW coefficient for pipes and local (minor) loss coefficient for valves

    def link_dict(link):
        if isinstance(link, wntr.network.Pipe):  # check if the link is a pipe
            return dict(
                link_ID=link.name,
                link_type='pipe',
                diameter=link.diameter,
                length=link.length,
                n_exp=n_exp,
                C=link.roughness,
                node_out=link.start_node_name,
                node_in=link.end_node_name
            )
        elif isinstance(link, wntr.network.Valve): # check if the link is a valve
            return dict(
                link_ID=link.name,
                link_type='valve',
                diameter=link.diameter,
                length=2*link.diameter,
                n_exp=2,
                C=link.minor_loss,
                node_out=link.start_node_name,
                node_in=link.end_node_name
            )
        
    for idx, link in enumerate(wn.links()):
        link_df.loc[idx] = link_dict(link[1])

    
    ## extract node data
    node_df = pd.DataFrame(
        index=pd.RangeIndex(wn.num_nodes), columns=["node_ID", "elev", "xcoord", "ycoord"]
    )

    def node_dict(node):
        if isinstance(node, wntr.network.elements.Reservoir):
            elev = 0
        else:
            elev = node.elevation
        return dict(
            node_ID=node.name,
            elev=elev,
            xcoord=node.coordinates[0],
            ycoord=node.coordinates[1]
        )

    for idx, node in enumerate(wn.nodes()):
        node_df.loc[idx] = node_dict(node[1])


    ## compute graph data
    A = np.zeros((net_info['np'], net_info['nn']+net_info['n0']), dtype=int)
    for k, row in link_df.iterrows():
        # find start node
        out_name = row['node_out']
        out_idx = node_df[node_df['node_ID']==out_name].index[0]
        # find end node
        in_name = row['node_in']
        in_idx = node_df[node_df['node_ID']==in_name].index[0]
        
        A[k, out_idx] = -1
        A[k, in_idx] = 1
        
    junction_idx = node_df.index[node_df['node_ID'].isin(net_info['junction_names'])].tolist()
    reservoir_idx = node_df.index[node_df['node_ID'].isin(net_info['reservoir_names'])].tolist()

    A12 = A[:, junction_idx]; A12 = sp.csr_matrix(A12) # link-junction incident matrix
    A10 = A[:, reservoir_idx]; A10 = sp.csr_matrix(A10) # link-reservoir indicent matrix


    ## extract demand data
    demand_df = results.node['demand'].T
    col_names = [f'demands_{t}' for t in range(1, len(demand_df.columns)+1)]
    demand_df.columns = col_names
    demand_df.reset_index(drop=False, inplace=True)
    demand_df = demand_df.rename(columns={'name': 'node_ID'})

    if net_info['nt'] > 1:
        demand_df = demand_df.iloc[:, :-1] # delete last time step
        
    demand_df = demand_df[~demand_df['node_ID'].isin(net_info['reservoir_names'])] # delete reservoir nodes


    ## extract boundary data
    h0_df = results.node['head'].T
    col_names = [f'h0_{t}' for t in range(1, len(h0_df.columns)+1)]
    h0_df.columns = col_names
    h0_df.reset_index(drop=False, inplace=True)
    h0_df = h0_df.rename(columns={'name': 'node_ID'})

    if net_info['nt'] > 1:
        h0_df = h0_df.iloc[:, :-1] # delete last time step

    h0_df = h0_df[h0_df['node_ID'].isin(net_info['reservoir_names'])] # only reservoir nodes


    ## load data to WDN object
    wdn = WDN(
            A12=A12,
            A10=A10,
            net_info=net_info,
            link_df=link_df,
            node_df=node_df,
            demand_df=demand_df,
            h0_df=h0_df,
    )

    return wdn









"""
Plot network function
""" 

def plot_network(wdn, plot_type='layout', vals=None, t=None):

    ## unload data
    link_df = wdn.link_df
    node_df = wdn.node_df
    net_info = wdn.net_info
    h0_df = wdn.h0_df
    
    ## draw network
    if plot_type == 'layout':
        uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}

        nx.draw(uG, pos, node_size=30, node_shape='o', node_color='black')
        nx.draw_networkx_nodes(uG, pos, nodelist=net_info['reservoir_names'], node_size=75, node_shape='s', node_color='black') # draw reservoir nodes

    elif plot_type == 'head':

        uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
        
        # create dictionary from dataframe to match node IDs
        vals_df = vals.set_index('node_ID')[f'h_{t}']
        h0_df = h0_df.set_index('node_ID')[f'h0_{t}']

        junction_vals = [vals_df[node] for node in net_info['junction_names']]
        reservoir_vals = [h0_df[node] for node in net_info['reservoir_names']]
        node_vals_all = junction_vals + reservoir_vals

        # color scaling
        min_val = min(node_vals_all)
        max_val = max(node_vals_all)

        # plot hydraulic heads
        cmap = cm.get_cmap('RdYlGn')
        nx.draw(uG, pos, nodelist=net_info['junction_names'], node_size=30, node_shape='o', node_color=junction_vals, cmap=cmap, vmin=min_val, vmax=max_val)
        nx.draw_networkx_nodes(uG, pos, nodelist=net_info['reservoir_names'], node_size=75, node_shape='s', node_color=reservoir_vals, cmap=cmap, vmin=min_val, vmax=max_val) 

        # create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(node_vals_all)
        colorbar = plt.colorbar(sm)
        colorbar.set_label('Hydraulic head [m]', fontsize=12)


    elif plot_type == 'flow':

        edge_df = link_df[['link_ID', 'node_out', 'node_in']]
        edge_df.set_index('link_ID', inplace=True)
        vals_df = vals.set_index('link_ID')[f'q_{t}']
        vals_df = abs(vals_df) * 1000
        edge_df = edge_df.join(vals_df)

        uG = nx.from_pandas_edgelist(edge_df, source='node_out', target='node_in', edge_attr=f'q_{t}')
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}

        # Define colormap
        cmap = cm.get_cmap('Blues')

        edge_values = nx.get_edge_attributes(uG, f'q_{t}')
        edge_values = list(edge_values.values())

        # color scaling
        norm = plt.Normalize(min(edge_values), max(edge_values))
        edge_colors = cmap(norm(edge_values))


        nx.draw(uG, pos, node_size=30, node_shape='o', node_color='black')
        nx.draw_networkx_nodes(uG, pos, nodelist=net_info['reservoir_names'], node_size=75, node_shape='s', node_color='black') 
        nx.draw_networkx_edges(uG, pos, edge_color=edge_colors) 

        # create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(edge_values)
        colorbar = plt.colorbar(sm)
        colorbar.set_label('Flow [L/s]', fontsize=12)       

    
    ## reservoir labels
    reservoir_labels = {node: 'Reservoir' for node in net_info['reservoir_names']}
    labels = nx.draw_networkx_labels(uG, pos, reservoir_labels, font_size=12, verticalalignment='bottom')
    for _, label in labels.items():
        label.set_y(label.get_position()[1] + 80)







"""
    Newton-Raphson hydraulic solver
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
    h_df = h_df[~h_df['node_ID'].isin(reservoir_names)] # delete reservoir nodes

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
    Newton-Raphson hydraulic solver
"""

def nr_solver(wdn):

    ### Step 1: unload network and hydraulic data
    A12 = wdn.A12
    A10 = wdn.A10
    net_info = wdn.net_info
    link_df = wdn.link_df
    node_df = wdn.node_df
    demand_df = wdn.demand_df
    h0_df = wdn.h0_df

    # define head loss equations
    def friction_loss(net_info, df):
        if net_info['headloss'] == 'H-W':
            K = 10.67 * df['length'] * (df['C'] ** -df['n_exp']) * (df['diameter'] ** -4.8704)
        else:
            K = [] # insert DW formula here...
        
        return K

    def local_loss(df):
        K = (8 / (np.pi ** 2 * 9.81)) * (df['diameter'] ** -4) * df['C']
        
        return K

    # compute loss coefficients
    K = np.zeros((net_info['np'], 1))
    for idx, row in link_df.iterrows():
        if row['link_type'] == 'pipe':
            K[idx] = friction_loss(net_info, row)

        elif row['link_type'] == 'valve':
            K[idx] = local_loss(row)
            
    n_exp = link_df['n_exp'].astype(float).to_numpy().reshape(-1, 1)
        
    # set stopping criteria
    tol = 1e-5
    kmax = 50 

    # small values in A11 make convergence unsteady; therefore, we need to define a lower bound -- see Todini (1988), page 7
    tol_A11 = 1e-5

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

        # begin iterations
        for k in range(kmax):

            ### Step 3: compute h^{k+1} and q^{k+1} for each iteration k
            A11_diag = K * (abs(qk) ** (n_exp - 1)) # diagonal elements of matrix A11
            A11_diag[A11_diag < tol_A11] = tol_A11 # replace with small value = tol_A11
            A11 = sp.diags(A11_diag.T, [0]) # matrix A11, allocated as a sparse diagonal matrix

            N = sp.diags(n_exp.T, [0]) # matrix N  
            I = sp.eye(net_info['np'], format='csr') # identiy matrix with dimension np x np, allocated as a sparse matrix

            b = np.concatenate([(N - I) @ A11 @ qk - A10 @ h0k, dk])
            J = sp.bmat([[N @ A11, A12], [A12.T, sp.csr_matrix((net_info['nn'], net_info['nn']))]], format='csr')

            # solve linear system
            x = sp.linalg.spsolve(J, b)
            qk = x[:net_info['np']]; qk = qk.reshape(-1, 1)
            hk = x[net_info['np']:net_info['np'] + net_info['nn']];hk = hk.reshape(-1, 1)

            
            ### Step 4: convergence check 
            err = A11 @ qk + A12 @ hk + A10 @ h0k
            max_err = np.linalg.norm(err, np.inf)

            # print progress
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








"""
    Newton-Raphson hydraulic solver using Schur complement for efficient factorisation
"""

def nr_schur_solver(wdn):

    ### Step 1: unload network and hydraulic data
    A12 = wdn.A12
    A10 = wdn.A10
    net_info = wdn.net_info
    link_df = wdn.link_df
    node_df = wdn.node_df
    demand_df = wdn.demand_df
    h0_df = wdn.h0_df

    # define head loss equations
    def friction_loss(net_info, df):
        if net_info['headloss'] == 'H-W':
            K = 10.67 * df['length'] * (df['C'] ** -df['n_exp']) * (df['diameter'] ** -4.8704)
        else:
            K = [] # insert DW formula here...
        
        return K

    def local_loss(df):
        K = (8 / (np.pi ** 2 * 9.81)) * (df['diameter'] ** -4) * df['C']
        
        return K

    # compute loss coefficients
    K = np.zeros((net_info['np'], 1))
    for idx, row in link_df.iterrows():
        if row['link_type'] == 'pipe':
            K[idx] = friction_loss(net_info, row)

        elif row['link_type'] == 'valve':
            K[idx] = local_loss(row)
            
    n_exp = link_df['n_exp'].astype(float).to_numpy().reshape(-1, 1)
        
    # set stopping criteria
    tol = 1e-5
    kmax = 50 

    # small values in A11 make convergence unsteady; therefore, we need to define a lower bound -- see Todini (1988), page 7
    tol_A11 = 1e-5

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

        # begin iterations
        for k in range(kmax):

            ### Step 3: compute h^{k+1} and q^{k+1}
            A11_diag = K * (abs(qk) ** (n_exp - 1)) # diagonal elements of matrix A11
            A11_diag[A11_diag < tol_A11] = tol_A11 # replace with small value = tol_A11
            A11 = sp.diags(A11_diag.T, [0]) # matrix A11, allocated as a sparse diagonal matrix
            
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
            
            ### Step 4: convergence check 
            err = A11 @ qk + A12 @ hk + A10 @ h0k
            max_err = np.linalg.norm(err, np.inf)

            # print progress
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