"""
CIVE 70019 and 70057 modules
Department of Civil and Environmental Engineering, Imperial College London
Prepared by Bradley Jenks
Summer 2023

Functions used for hydraulic modelling of water networks
    - 'load_data' using WNTR Python package
    - 'plot_network' function

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
    create classes for storing data as objects
"""


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



"""
Load network data via wntr
""" 


def load_network_data(inp_file):
    
    wdn = object()

    ## load network from wntr
    wn = wntr.network.WaterNetworkModel(inp_file)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    ## get network elements and simulation info
    nt = int(wn.options.time.duration / wn.options.time.report_timestep)
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

def plot_network(wdn, plot_type='layout', pcv_nodes=None, sensor_nodes=None, vals=None, t=None):
    fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and axis object

    # Unload data
    link_df = wdn.link_df
    node_df = wdn.node_df
    net_info = wdn.net_info
    h0_df = wdn.h0_df
    
    # Draw network
    if plot_type == 'layout':
        uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}

        nx.draw(uG, pos, node_size=20, node_shape='o', node_color='black', ax=ax)
        nx.draw_networkx_nodes(
            uG, pos, 
            nodelist=net_info['reservoir_names'], 
            node_size=80, 
            node_shape='s',  # Square shape
            node_color='black', 
            ax=ax
        )

        if pcv_nodes is not None:
            nx.draw_networkx_nodes(uG, pos, nodelist=pcv_nodes, node_size=100, node_shape='d', node_color='red', ax=ax)

        if sensor_nodes is not None:
            sensor_names = [net_info['junction_names'][i] for i in sensor_nodes]
            nx.draw_networkx_nodes(uG, pos, sensor_names, node_size=100, node_shape='o', node_color='red', edgecolors='white', ax=ax)

    elif plot_type in ['hydraulic head', 'pressure head', 'flow']:
        uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
        pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}

        # Prepare data and color scaling
        cmap = cm.get_cmap('RdYlBu')

        if plot_type == 'hydraulic head':
            vals_df = vals.set_index('node_ID')[f'h_{t}']
            h0_df = h0_df.set_index('node_ID')[f'h0_{t}']

            junction_vals = [vals_df[node] for node in net_info['junction_names']]
            reservoir_vals = [h0_df[node] for node in net_info['reservoir_names']]
            node_vals_all = junction_vals + reservoir_vals

        elif plot_type == 'pressure head':
            vals_df = vals.set_index('node_ID')[f'h_{t}']
            h0_df = h0_df.set_index('node_ID')[f'h0_{t}']

            junction_vals = [vals_df[node] - node_df.loc[node_df['node_ID'] == node, 'elev'].to_numpy()[0]
                             for node in net_info['junction_names']]
            reservoir_vals = [0 for node in net_info['reservoir_names']]
            node_vals_all = junction_vals + reservoir_vals

        elif plot_type == 'flow':
            edge_df = link_df[['link_ID', 'node_out', 'node_in']].set_index('link_ID')
            vals_df = vals.set_index('link_ID')[f'q_{t}'].abs() * 1000
            edge_df = edge_df.join(vals_df)

            uG = nx.from_pandas_edgelist(edge_df, source='node_out', target='node_in', edge_attr=f'q_{t}')
            edge_values = nx.get_edge_attributes(uG, f'q_{t}').values()
            norm = plt.Normalize(min(edge_values), max(edge_values))
            edge_colors = cmap(norm(list(edge_values)))

            nx.draw(uG, pos, node_size=0, node_color='black', ax=ax)
            nx.draw_networkx_edges(uG, pos, edge_color=edge_colors, width=2, ax=ax)

            # Add colorbar for flow
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(list(edge_values))
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Flow [L/s]', fontsize=14)
            return  # Exit after drawing edges to avoid additional labels

        # Color scaling
        min_val, max_val = min(node_vals_all), max(node_vals_all)

        # Plot nodes
        nx.draw(uG, pos, nodelist=net_info['junction_names'], node_size=20, node_color=junction_vals,
                cmap=cmap, vmin=min_val, vmax=max_val, ax=ax)
        nx.draw_networkx_nodes(
            uG, pos, 
            nodelist=net_info['reservoir_names'], 
            node_size=100, 
            node_shape='s',  # Square shape
            node_color=reservoir_vals, 
            cmap=cmap, 
            vmin=min_val, 
            vmax=max_val, 
            ax=ax
        )

        if pcv_nodes is not None:
            nx.draw_networkx_nodes(uG, pos, nodelist=pcv_nodes, node_size=100, node_shape='d', node_color='black', ax=ax)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_val, max_val))
        sm.set_array(node_vals_all)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Hydraulic head [m]' if plot_type == 'hydraulic head' else 'Pressure head [m]', fontsize=14)
 
    # Add labels (common for all plots)
    reservoir_labels = {node: 'Reservoir' for node in net_info['reservoir_names']}
    _reservoir_labels = nx.draw_networkx_labels(uG, pos, reservoir_labels, font_size=12, ax=ax)
    for _, label in _reservoir_labels.items():
        label.set_y(label.get_position()[1] + 80)
    
    if pcv_nodes is not None:
        pcv_labels = {node: 'PCV' for node in pcv_nodes}
        _pcv_labels = nx.draw_networkx_labels(uG, pos, pcv_labels, font_size=12, ax=ax)
        for _, label in _pcv_labels.items():
            label.set_y(label.get_position()[1] + 80)

    if sensor_nodes is not None:
        sensor_labels = {node: str(idx + 1) for idx, node in enumerate(sensor_names)}
        _sensor_labels = nx.draw_networkx_labels(uG, pos, sensor_labels, font_size=12, ax=ax)
        for _, label in _sensor_labels.items():
            label.set_y(label.get_position()[1] + 120)












