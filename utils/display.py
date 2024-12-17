from spektral.datasets import TUDataset
from pyvis.network import Network
import pandas as pd
import numpy as np
from numpy import arange
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


class DispGraph():
    def __init__(self, dataset, colors, n, name):
        self.dataset = dataset
        self.name    = name
        self.colors  = colors
        self.colors_map = []
        self.n = n

    def color_map(self):
        col_u = self.n
        minima = 0
        maxima = col_u

        norm = cm.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='inferno')

        #for v in self.colors:
        #    self.colors_map.append(cm.colors.to_hex(mapper.to_rgba(v*1000)))
        for v in self.colors:
            if v == 0:
                cc = '#ffff00'
            elif v == 1:
                cc = '#cc0000'
            else:
                cc = '#33cc33'

            self.colors_map.append(cc)

    def display(self):
        node_number = -1
        sources  = []
        targets  = []
        weights = []

        for a in self.dataset: #dataset[0].a:
            node_number += 1  # source
            ind_g = a.indices  # target
            ind_w = a.data     # weights
            for ind in  range(len(ind_g)):
                sources.append(node_number)
                targets.append(ind_g[ind])
                weights.append(ind_w[ind])

        got_net = Network(height='750px', width='100%', bgcolor='#222222', font_color='black')
        # set the physics layout of the network
        got_net.barnes_hut()
        edge_data = zip(sources, targets, weights)

        for e in edge_data:
            src = str(e[0])
            dst = str(e[1])
            w = e[2]

            got_net.add_node(dst, src, title=dst)
            got_net.add_node(src, dst, title=src)
            got_net.add_edge(src, dst, value=w)
            got_net.add_edge(dst, src, value=-w)

        neighbor_map = got_net.get_adj_list()
        print(neighbor_map)
        input()

        self.color_map()

        # add neighbor data to node hover data
        count = -1
        for node in got_net.nodes:
            count +=1
            node_index = int(node['id'])
            node['title'] += ' Neighbors:<br>' + '<br>'.join(str(neighbor_map[node['id']]))
            node['value'] = str(len(neighbor_map[node['id']]))
            node['color'] = str(self.colors_map[node_index])

        for edges in got_net.edges:
            edges['color'] = "blue"

        input()
        got_net.show(self.name+'.html')

