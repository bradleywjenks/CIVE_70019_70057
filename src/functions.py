"""
CIVE 70019 and 70057 modules
Department of Civil and Environmental Engineering, Imperial College London
Prepared by Bradley Jenks
June 2023

Functions used for modelling and optimization of water networks
    - 'load_data' using WNTR Python package
    - 'hydraulic_solver_nr' hydraulic solver using newton-raphson method
    - 'plot_network' function

"""

### import packages ###
import wntr
import networkx as nx
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel
from typing import Any







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

def plot_network(wdn):

    # unload data
    link_df = wdn.link_df
    node_df = wdn.node_df
    net_info = wdn.net_info

    uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
    pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
    reservoir_labels = {node: 'Reservoir' for node in net_info['reservoir_names']}
            
    # draw entire network
    nx.draw(uG, pos, node_size=30, node_shape='o', node_color='skyblue')

    # draw reservoir nodes

    nx.draw_networkx_nodes(uG, pos, nodelist=net_info['reservoir_names'], node_size=50, node_shape='s', node_color='black')
    
    # reservoir labels
    labels = nx.draw_networkx_labels(uG, pos, reservoir_labels, font_size=12, verticalalignment='bottom')
    for _, label in labels.items():
        label.set_y(label.get_position()[1] + 80)