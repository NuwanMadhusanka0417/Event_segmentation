
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft
import sys
sys.path.append("models/")
# from mlp import MLP

class GraphCNN(nn.Module):
    def __init__(self, input_dim, num_layers, delta, graph_pooling_type, neighbor_pooling_type, device, equation):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            input_dim: dimensionality of input features
            delta: usage of binding or not
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
            equation: which equation to use (10,11)
        '''

        super(GraphCNN, self).__init__()
        # print("Input feature size: ", input_dim)
        # self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = True
        self.delta = delta
        self.equation = equation



    def __preprocess_neighbors_sumavepool(self, edge_index, num_nodes = None):
        # Assuming edge_index is a 2xN tensor where N is the number of edges

        # Convert edge_index to a sparse adjacency matrix
        if(num_nodes==None):
            num_nodes = edge_index.max().item() + 1  # Assuming node indices start at 0
        Adj_block_elem = torch.ones(edge_index.size(1), dtype=torch.float32)  # Elements for adjacency matrix

        # Create sparse adjacency matrix
        Adj_block = torch.sparse.FloatTensor(edge_index, Adj_block_elem, torch.Size([num_nodes, num_nodes]))

        # Optionally add self-loops
        if not self.learn_eps:
            self_loop_edge = torch.arange(0, num_nodes, dtype=torch.long)
            self_loop_edge = torch.stack((self_loop_edge, self_loop_edge))  # Create self-loop edges
            self_loop_elem = torch.ones(num_nodes, dtype=torch.float32)  # Elements for self-loops
            Adj_block_idx = torch.cat([edge_index, self_loop_edge], dim=1)
            Adj_block_elem = torch.cat([Adj_block_elem, self_loop_elem], dim=0)
            Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([num_nodes, num_nodes]))

        return Adj_block.to(self.device)



    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))
            
            else:
            ###sum pooling
                elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep
    
    def permutation_to_matrix(self, perm):
        """Converts a permutation vector to its corresponding permutation matrix."""
        n = len(perm)
        matrix = torch.zeros(n, n, dtype=torch.float32)
        matrix[torch.arange(n), perm] = 1
        return matrix
    
    def bind(self, x, y):
        # Perform FFT on each hypervector in the tensors
        fft_self = fft(x, dim=1)
        fft_other = fft(y, dim=1)

        # Multiply element-wise in the frequency domain
        product = torch.mul(fft_self, fft_other)

        # Perform inverse FFT to get back to the spatial domain
        result = ifft(product, dim=1)

        # Return the real part of the result as the final bound hypervectors
        return torch.real(result)

    def invert_permutation(self, perm):
        """Generate the inverse of a permutation."""
        inverse = [0] * len(perm)
        for i, p in enumerate(perm):
            inverse[p] = i
        return inverse
    
    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None, delta = 1, equation = 10):
        if(equation==10):
            n_rows, n_cols = h.shape
            torch.manual_seed(0)
            rotated_matrix = h.clone()  # Start with the original matrix
            shift = 1# layer + 1
            rotated_matrix = torch.roll(rotated_matrix, shifts=shift, dims=1)
            if self.neighbor_pooling_type == "max":
                ##If max pooling
                pooled = self.maxpool(rotated_matrix, padded_neighbor_list)
                # pooled_no_perm = self.maxpool(h, padded_neighbor_list)
            else:
                #If sum or average pooling
                pooled = torch.spmm(Adj_block, rotated_matrix)
                # pooled_no_perm = torch.spmm(Adj_block, h)
                if self.neighbor_pooling_type == "average":
                    #If average pooling
                    degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                    pooled = pooled/degree
            
            if(delta ==1):
                pooled = self.bind(h,pooled)+h #self.bind(h,pooled) +  h   #pooled + h  #self.bind(h,pooled) + #
            elif(delta ==2):
                pooled = self.bind(h,pooled)+h+pooled #self.bind(h,pooled) +  h   #pooled + h  #self.bind(h,pooled) + #
            else:
                # print("AT zero NO BINDING")
                pooled = pooled+h
            
            
        else:
            n_rows, n_cols = h.shape
            torch.manual_seed(0)

            if self.neighbor_pooling_type == "max":
                ##If max pooling
                # pooled = self.maxpool(rotated_matrix, padded_neighbor_list)
                pooled_no_perm = self.maxpool(h, padded_neighbor_list)
            else:
                #If sum or average pooling
                # pooled = torch.spmm(Adj_block, rotated_matrix)
                pooled_no_perm = torch.spmm(Adj_block, h)
                if self.neighbor_pooling_type == "average":
                    #If average pooling
                    degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                    # pooled = pooled/degree
                    pooled_no_perm = pooled_no_perm/degree
            pooled = pooled_no_perm
            if(delta ==1):
                pooled = self.bind(h,pooled)+h #self.bind(h,pooled) +  h   #pooled + h  #self.bind(h,pooled) + #
            elif(delta ==2):
                pooled = self.bind(h,pooled)+h+pooled #self.bind(h,pooled) +  h   #pooled + h  #self.bind(h,pooled) + #
            else:
                # print("AT zero NO BINDING")
                pooled = pooled+h

            rotated_matrix = pooled.clone()  # Start with the original matrix
            shift = 1  # layer+1
            #### APPLY PERMUTATION
            # inverse_permutation = self.invert_permutation(permutation)

            # for _ in range( shift): #layer + 1
            #     rotated_matrix = rotated_matrix[:, inverse_permutation]  # Apply permutation multiple times

            #### APPLY ROTATION
            # Circularly shift columns
            rotated_matrix = torch.roll(rotated_matrix, shifts=shift, dims=1)
            pooled = rotated_matrix


        # pooled = F.normalize(pooled, p=2, dim=1)
        pooled = torch.sign(pooled)
        # pooled = 2*torch.sigmoid(pooled)-1

        # pooled = torch.tanh(pooled)
        # pooled = torch.relu(pooled)

        # pooled=torch.clamp(pooled, min=-1, max=1) 
        return pooled


    
    def forward(self, x, edge_index):
        # Start with the input node features
        h = x.to(self.device)

        # Preprocess the adjacency block based on edge indices
        Adj_block = self.__preprocess_neighbors_sumavepool(edge_index,x.shape[0])

        # List of hidden representations at each layer (including input)
        hidden_rep = [h]

        # Iterating over each layer to update the node features
        for layer in range(self.num_layers - 1):
            h = self.next_layer_eps(h, layer, Adj_block=Adj_block, delta=self.delta, equation= self.equation)
            hidden_rep.append(h)

        # Stack the hidden representations from all layers along a new dimension
        all_layers_output = torch.stack(hidden_rep, dim=0)

        return torch.sum(torch.sum(all_layers_output, dim=0), dim=0) 

        

    
