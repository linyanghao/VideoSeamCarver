#!/usr/bin/env python
# -*- coding: utf-8 -*-
import igraph as ig 
import igraph.vendor.texttable


def minimum_cut(G, S_Node, T_Node):
    G = G.G
    S = G.vs.find(S_Node).index
    T = G.vs.find(T_Node).index
    cut = G.st_mincut(S, T, capacity='capacity')
    partition0 = set(map(lambda index:eval(G.vs[index]['name'], {'S':'S', 'T':'T'}), cut[0]))
    partition1 = set(map(lambda index:eval(G.vs[index]['name'], {'S':'S', 'T':'T'}), cut[1]))
    return cut.value, (partition0, partition1)


class DiGraph():
    def __init__(self):
        self.G = ig.Graph(directed=True)
        self.add_node_queue = []
        self.remove_node_queue = []
        self.add_edge_queue = []
        self.remove_edge_queue = []
        self.BUFFER_MODE = True

    def flush(self):
        if self.BUFFER_MODE:
            if self.remove_edge_queue:
                self.G.delete_edges(self.remove_edge_queue)
                self.remove_edge_queue.clear()
            if self.add_node_queue:
                self.G.add_vertices(self.add_node_queue)
                self.add_node_queue.clear()
            if self.remove_node_queue:
                self.G.delete_vertices(self.remove_node_queue)
                self.remove_node_queue.clear()
            if self.add_edge_queue:
                add_edge_queue_with_index = []
                for u, v, capacity in self.add_edge_queue:
                    index_u = self.G.vs.find(u).index
                    index_v = self.G.vs.find(v).index
                    add_edge_queue_with_index.append( (index_u, index_v, capacity) )
                start = len(self.G.es)
                self.G.add_edges(map(lambda tup:tup[:-1], add_edge_queue_with_index))
                end = len(self.G.es)
                self.G.es[start:end]['capacity'] = list(map(lambda tup:tup[-1], self.add_edge_queue))
                self.add_edge_queue.clear()
            
        else:
            pass

    def add_node(self, node):
        node = str(node)
        if self.BUFFER_MODE:
            self.add_node_queue.append(node)
        else:
            self.G.add_vertex(node)

    def add_nodes_from(self, list_of_nodes):
        #list_of_nodes = map(self._generate_index, list_of_nodes)
        list_of_nodes = map(str, list_of_nodes)
        self.G.add_vertices(list_of_nodes)
    
    def remove_node(self, node):
        #index = self.name2index.pop(node)
        #self.index2name.pop(index)
        node = str(node)
        index = self.G.vs.find(node).index
        if self.BUFFER_MODE:
            self.remove_node_queue.append(index)
        else:
            self.G.delete_vertices(index)

    def add_edge(self, u, v, capacity):
        u, v = str(u), str(v)
        
        if capacity == float('inf'):
            capacity = 9999999
        else:
            capacity = int(capacity)
        if self.BUFFER_MODE:
            self.add_edge_queue.append( (u, v, capacity) )
        else:
            index_u, index_v = self.G.vs.find(u).index, self.G.vs.find(v).index
            self.G.add_edge(index_u, index_v, capacity=capacity)
    def remove_edge(self, u, v):
        u, v = str(u), str(v)
        index_u, index_v = self.G.vs.find(u).index, self.G.vs.find(v).index
        
        edgeID = self.G.get_eid(index_u, index_v, directed=True)
        if self.BUFFER_MODE:
            self.remove_edge_queue.append(edgeID)
        else:
            self.G.delete_edges(edgeID)