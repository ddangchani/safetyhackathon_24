import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import momepy
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import integrate
from tqdm import tqdm
import warnings
from itertools import product
import shapely

warnings.filterwarnings("ignore")

def distR(segu, tu, segv, tv, L):
    # Source : Moller, J., & Rasmussen (2024) 
    ILM = np.linalg.inv(L.calcLaplacian()) # Inverse of Laplacian Matrix
    Sigajaj = ILM[L._from[segu], L._from[segu]]
    Sigajbj = ILM[L._from[segu], L._to[segu]]
    Sigbjbj = ILM[L._to[segu], L._to[segu]]
    Sigaiai = ILM[L._from[segv], L._from[segv]]
    Sigaibi = ILM[L._from[segv], L._to[segv]]
    Sigbibi = ILM[L._to[segv], L._to[segv]]
    Sigajai = ILM[L._from[segu], L._from[segv]]
    Sigajbi = ILM[L._from[segu], L._to[segv]]
    Sigbjai = ILM[L._to[segu], L._from[segv]]
    Sigbjbi = ILM[L._to[segu], L._to[segv]]
    li = L._len[segv]
    lj = L._len[segu]
    dist = ((1-tu)**2 * Sigajaj + tu**2 * Sigbjbj + 2*tu*(1-tu) * Sigajbj
          + (1-tv)**2 * Sigaiai + tv**2 * Sigbibi + 2*tv*(1-tv) * Sigaibi
          - 2*(1-tu)*(1-tv) * Sigajai - 2*(1-tu)*tv * Sigajbi
          - 2*tu*(1-tv) * Sigbjai - 2*tu*tv * Sigbjbi
          + tv*(1-tv)*li + tu*(1-tu)*lj
          - 2*(segu==segv)*min(tu*(1-tv),tv*(1-tu))*li)

    return dist

def pairdistR(X, L): 
    """
    Calculate pairwise resistance distance between points
    Args:
        X (gpd.GeoSeries) : GeoSeries of points
        L (Linnet) : Linnet object
    """
    # Source : Moller, J., & Rasmussen (2024) 
    # X : Geoseries/GeoDataFrame of points
    X = snap_points(L, X)
    n = len(X)
    ILM = np.linalg.inv(L.calcLaplacian()) # Inverse of Laplacian Matrix
    segu = X['edge_name'].values
    tu = X['tp'].values

    # segu outer segv 

    onesu = np.ones(n)
    onesv = np.ones(n)
    Sigajaj = np.outer(np.diagonal(ILM)[L._from[segu]], onesv)
    Sigajbj = np.outer(ILM[L._from[segu], L._to[segu]], onesv)
    Sigbjbj = np.outer(np.diagonal(ILM)[L._to[segu]], onesv)
    Sigaiai = np.outer(onesu, np.diagonal(ILM)[L._from[segu]])
    Sigaibi = np.outer(onesu, ILM[L._from[segu], L._to[segu]])
    Sigbibi = np.outer(onesu, np.diagonal(ILM)[L._to[segu]])
    Sigajai = ILM[np.array(L._from[segu])[:,None], L._from[segu]]
    Sigajbi = ILM[np.array(L._from[segu])[:,None], L._to[segu]]
    Sigbjai = ILM[np.array(L._to[segu])[:,None], L._from[segu]]
    Sigbjbi = ILM[np.array(L._to[segu])[:,None], L._to[segu]]
    li = np.outer(onesu, L._len[segu])
    liv = L._len[segu]
    lj = np.outer(L._len[segu], onesv)
    tum = np.outer(tu, onesv)
    tvm = np.outer(onesu, tu)

    dist = ((1-tum)**2 * Sigajaj + tum**2 * Sigbjbj + 2*tum*(1-tum) * Sigajbj
            + (1-tvm)**2 * Sigaiai + tvm**2 * Sigbibi + 2*tvm*(1-tvm) * Sigaibi
            - 2*(1-tum)*(1-tvm) * Sigajai - 2*(1-tum)*tvm * Sigajbi
            - 2*tum*(1-tvm) * Sigbjai - 2*tum*tvm * Sigbjbi
            + tvm*(1-tvm)*li + tum*(1-tum)*lj
            - 2*np.equal.outer(segu,segu)*np.minimum(np.outer(tu,1-tu),np.outer(1-tu,tu))*np.outer(onesu,liv))

    return dist * (dist > 0)

def snap_points(L, points):
    """
    Snap points to the nearest edge on the network
    Args:
        L (Linnet) : Linnet object
        points (gpd.GeoSeries) : GeoSeries of points
    Returns:
        gpd.GeoDataFrame : snapped points
    """
    
    snapped_df = pd.DataFrame(points.geometry.apply(lambda x: L.snap(x)).tolist(), columns=['snap', 'edge'])
    snapped_df['node_start'] = snapped_df['edge'].apply(lambda x: x.node_start)
    snapped_df['node_end'] = snapped_df['edge'].apply(lambda x: x.node_end)
    snapped_df['edge_name'] = snapped_df['edge'].apply(lambda x: x.name)
    snapped_df['edge_len'] = snapped_df['edge'].apply(lambda x: x.mm_len)
    snapped_df.drop(columns=['edge'], inplace=True)
    snapped_df['d_start'] = snapped_df.apply(lambda x: x.snap.distance(L.nodes.geometry[x.node_start]), axis=1)
    snapped_df['d_end'] = snapped_df.apply(lambda x: x.snap.distance(L.nodes.geometry[x.node_end]), axis=1)
    snapped_df['tp'] = snapped_df['d_start'] / snapped_df['edge_len']

    return gpd.GeoDataFrame(snapped_df, geometry='snap')


class Linnet(nx.MultiGraph):
    """
    Create a networkx MultiGraph object from a GeoDataFrame of LINESTRINGs
    Attributes:
        nodes (GeoDataFrame) : GeoDataFrame of nodes
        edges (GeoDataFrame) : GeoDataFrame of edges
        sw (libpysal.weights.W) : spatial weights object
    """
    def __init__(self, edges):
        super().__init__()
        assert isinstance(edges, (gpd.GeoSeries, gpd.GeoDataFrame)), "Edges must be a GeoSeries or GeoDataFrame object"
        self.graph = momepy.gdf_to_nx(edges)
        nodes, edges, sw = momepy.nx_to_gdf(self.graph, points=True, lines=True, spatial_weights=True)
        self.nodes = nodes
        self.edges = edges
        self._from = self.edges['node_start']
        self._to = self.edges['node_end']
        self._len = self.edges['mm_len']
        self.shortest_path = nx.floyd_warshall_numpy(self.graph, weight='mm_len')
        self.adjacency = nx.adjacency_matrix(self.graph).toarray()

    def calcLaplacian(self):
        L = - self.adjacency * (1 / self.shortest_path)
        L[np.diag_indices_from(L)] = 0
        L[np.diag_indices_from(L)] = -L.sum(axis=1)
        L[0, 0] += 1
        return L
    
    def discretize(self, length):
        """
        Return the equally spaced points along the network
        Args:
            length (float) : maximum distance between two points
            points (gpd.GeoSeries) : GeoSeries of point patterns
        Returns:
            gpd.GeoDataFrame : generated edges
        """
        vs = []
        edgenum = []
        for edge in self.edges.itertuples():
            n = np.ceil(edge.mm_len / length).astype(int) + 1
            interpolates = [edge.geometry.interpolate(i, normalized=True) for i in np.linspace(0, 1, n)]
            # to vertices
            interpolates_lines = [shapely.geometry.LineString([interpolates[i], interpolates[i+1]]) for i in range(n-1)]
            vs.extend(interpolates_lines)
            edgenum.extend([edge.Index] * (n-1))

        # to edges
        edges = gpd.GeoDataFrame(geometry=vs)
        edges['edge_num'] = edgenum

        return edges
    


    # def d_G(self, source, target):
    #     """
    #     Return the shortest path distance(geodesic) between two nodes
    #     Args:
    #         source : source node index
    #         target : target node index
    #     Returns:
    #         list : shortest path
    #     """
    #     # return nx.shortest_path_length(self.graph, source=source, target=target, weight='mm_len')
    #     return self.shortest_path[source][target]

    # def conductance(self, source, target):
    #     """
    #     Return the conductance between two nodes
    #     Args:
    #         source : source node index
    #         target : target node index
    #     Returns:
    #         float : conductance function value
    #     """
    #     # If source and target are neighbors, conductance is 1/d_G
    #     ## Check if source and target are neighbors

    #     if self.adjacency[source, target] == 1:
    #         return 1/self.d_G(source, target)
    #     else:
    #         # Conductance is 0
    #         return 0

    # def _c(self, source):
    #     """
    #     Return the c(u) function of a node
    #     Args:
    #         source : source node index
    #     Returns:
    #         float : c(u) function value
    #     """
    #     # Return the sum of conductance between origin and all other neighbors
    #     c_u = 0
    #     for neighbor in np.where(self.adjacency[source] == 1)[0]:
    #         c_u += self.conductance(source, neighbor)
    #     return c_u

    # def L_uv(self, source, target):
    #     """
    #     Return the L(u,v) function between two nodes
    #     Args:
    #         source : source node index
    #         target : target node index
    #     Returns:
    #         float : L(u,v) function value
    #     """
    #     if source == target == self.origin:
    #         return 1 + self._c(source)
    #     elif source == target:
    #         return self._c(source)
    #     else:
    #         return (-1) * self.conductance(source, target)

    # def _L_inv(self, return_matrix=False):
    #     """
    #     Return the inverse of the L(u,v) matrix
    #     Returns:
    #         np.array : inverse of L(u,v) matrix (L**-1)
    #     """
    #     V = len(self.graph)
    #     L = np.zeros((V, V))
    #     for i in range(V):
    #         for j in range(V):
    #             L[i, j] = self.L_uv(i, j)
            
    #     assert np.all(np.linalg.eigvals(L) > 0), "L(u,v) matrix is not positive definite"

    #     self.L_inv = np.linalg.inv(L)
    #     if return_matrix:
    #         return self.L_inv

    def snap(self, point):
        """
        Snap a point to the nearest edge on the network
        Args:
            point (Point) : point to snap
        Returns:
            (Point, GeoDataFrame) : snapped point, nearest edge (GeoDataFrame)
        """
        nidx = self.edges.sindex.nearest(point)[1][0]
        nedge = self.edges.loc[nidx]
        snap = nedge.geometry.interpolate(nedge.geometry.project(point))
        return snap, nedge
    
    # def d_u(self, snap, nedge):
    #     """
    #     Return the function d(u) at a point u on the network
    #     Args:
    #         point : point on the network (node or a point on an edge)
    #     Returns:
    #         float : d(u) function value
    #     """
    #     # If point is a node, return 0
    #     if snap in self.graph:
    #         return 0
    #     else:
    #         nedge_l, *_, nedge_u = nedge.geometry.coords
    #         snap_xy = (snap.x, snap.y)

    #         numerator =  distance.euclidean(nedge_l, snap_xy)
    #         denominator = nedge.mm_len

    #         return numerator / denominator
        
    # def resistance_metric(self, source, target):
    #     """
    #     Return a resistance metric between each pair of points on the network.
    #     Args:
    #         source : source point
    #         target : target point
    #     Returns:
    #         float : resistance metric
    #     """
    #     # Snap source and target to nearest edges
    #     source_snap, source_edge = self.snap(source)
    #     target_snap, target_edge = self.snap(target)

    #     source_l, source_u = source_edge.node_start, source_edge.node_end
    #     target_l, target_u = target_edge.node_start, target_edge.node_end

    #     # d_u function
    #     _d_u_source = self.d_u(source_snap, source_edge)
    #     _d_u_target = self.d_u(target_snap, target_edge)

    #     # R_mu
    #     R_mu_1 = _d_u_source * _d_u_target * self.L_inv[source_u, target_u]
    #     R_mu_2 = (1 - _d_u_source) * (1 - _d_u_target) * self.L_inv[source_l, target_l]
    #     R_mu_3 = _d_u_source * (1 - _d_u_target) * self.L_inv[source_u, target_l]
    #     R_mu_4 = (1 - _d_u_source) * _d_u_target * self.L_inv[source_l, target_u]
    #     _R_mu = R_mu_1 + R_mu_2 + R_mu_3 + R_mu_4

    #     # R_e
    #     if source_edge.name != target_edge.name:
    #         _R_e = 0
    #     else:
    #         _R_e = (min(_d_u_source, _d_u_target) - _d_u_source * _d_u_target) * source_edge.mm_len

    #     # Return the resistance metric
    #     return _R_mu + _R_e
    
    # def resistance_matrix(self, points, save=True, filename=None):
    #     """
    #     Build a resistance matrix between each pair of given points on the network.
    #     Args:
    #         points (gpd.GeoSeries) : GeoSeries of points
    #     Returns:
    #         np.array : resistance matrix
    #     """
    #     @ray.remote
    #     def resistance_metric_ray(source, target, linnet, bar: tqdm_ray.tqdm):
    #         bar.update.remote(1)
    #         return linnet.resistance_metric(source, target)
        
    #     remote_tqdm = ray.remote(tqdm_ray.tqdm)
    #     bar = remote_tqdm.remote(total=len(points)**2, desc="Calculating Resistance Matrix")

    #     workers = [
    #         resistance_metric_ray.remote(source, target, self, bar) for source, target in product(points, repeat=2)
    #     ]
    #     result = np.array(ray.get(workers)).reshape(len(points), len(points))
    #     bar.close.remote()
    
    #     # Save
    #     if save:
    #         if filename is not None:
    #             np.save(filename, result)
    #         else:
    #             np.save('data/resistance_matrix.npy', result)

    #     return result
    
    # def geodesic_metric(self, source, target):
    #     """
    #     Return the geodesic metric between two points on the network
    #     Args:
    #         source : source point
    #         target : target point
    #     Returns:
    #         float : geodesic metric
    #     """
    #     # Snap source and target to nearest edges
    #     source_snap, source_edge = self.snap(source)
    #     target_snap, target_edge = self.snap(target)

    #     if source_edge.name == target_edge.name: # on the same edge
    #         return source_snap.distance(target_snap)

    #     # Get the shortest path
    #     source_l, source_u = source_edge.node_start, source_edge.node_end
    #     target_l, target_u = target_edge.node_start, target_edge.node_end

    #     # dist to nearest edges
    #     source_d_start = source_snap.distance(self.nodes.geometry[source_l])
    #     source_d_end = source_snap.distance(self.nodes.geometry[source_u])
    #     target_d_start = target_snap.distance(self.nodes.geometry[target_l])
    #     target_d_end = target_snap.distance(self.nodes.geometry[target_u])


    #     # Get the shortest path
    #     res = [
    #         self.shortest_path[source_l][target_l] + source_d_start + target_d_start,
    #         self.shortest_path[source_l][target_u] + source_d_start + target_d_end,
    #         self.shortest_path[source_u][target_l] + source_d_end + target_d_start,
    #         self.shortest_path[source_u][target_u] + source_d_end + target_d_end
    #     ]
    #     return min(res)
    
    # def geodesic_matrix(self, points):
    #     """
    #     Build a Covariance matrix with geodesic metric between each pair of given points on the network.
    #     Args:
    #         points (gpd.GeoSeries) : GeoSeries of points
    #     Returns:
    #         np.array : geodesic matrix
    #     """
    #     @ray.remote
    #     def geodesic_metric_ray(source, target, linnet, bar: tqdm_ray.tqdm):
    #         bar.update.remote(1)
    #         return linnet.geodesic_metric(source, target)
        
    #     remote_tqdm = ray.remote(tqdm_ray.tqdm)
    #     bar = remote_tqdm.remote(total=len(points)**2, desc="Calculating Geodesic Matrix")

    #     workers = [
    #         geodesic_metric_ray.remote(source, target, self, bar) for source, target in product(points, repeat=2)
    #     ]
    #     result = np.array(ray.get(workers)).reshape(len(points), len(points))
    #     bar.close.remote()

    #     # Save
    #     np.save('data/geodesic_matrix.npy', result)

class LPP:
    """
    Create a Point Pattern object on a linear network
    Args:
        linnet (Linnet) : Linnet object
        points (gpd.GeoSeries) : GeoSeries of points
        buffer (int) : buffer in meters (default=1m)
        resolution (int) : resolution of equally spaced points along the network (default=10)
        load_geo (str) : path to precalculated geodesic matrix
        load_res (str) : path to precalculated resistance matrix
    """
    def __init__(self, linnet, points, resolution=10, buffer=1): # buffer in meters
        if not isinstance(linnet, Linnet):
            linnet = Linnet(linnet)
        assert isinstance(points, (gpd.GeoSeries, gpd.GeoDataFrame)), "Points must be a GeoSeries or GeoDataFrame object"
        self.linnet = linnet
        self.nodes = self.linnet.nodes
        self.edges = self.linnet.edges
        self.resolution = resolution
        self.linnet_union = self.edges.unary_union
        self.v = int(self.linnet_union.length / resolution) # number of equally spaced points along the network
        self.interpolates = gpd.GeoSeries([self.linnet_union.interpolate(i, normalized=True) for i in np.linspace(0, 1, self.v)]) # equally spaced points along the network

        # Check if points are within network
        buffered = self.linnet_union.buffer(buffer)
        self.points = points[points.within(buffered)] # geodataframe

        # Geodesic Matrix
        self.shortest_path = self.linnet.shortest_path

    
    # def geodesic_point(self, point):
    #     """
    #     Return the shortest path distance between a point on the network and each point pattern
    #     Args:
    #         point (shapely.geometry.Point) : point on the network
    #     Returns:
    #         np.array : shortest path distance
    #     """
    #     # Snap point to nearest edge
    #     snap_point, nedge_point = self.linnet.snap(point)
    #     d_start, d_end = snap_point.distance(self.nodes.geometry[nedge_point.node_start]), snap_point.distance(self.nodes.geometry[nedge_point.node_end])

    #     # Get the shortest paths
    #     res = []
    #     for i, row in self.snapped_df.iterrows():
    #         if row.edge_name == nedge_point.name:
    #             res.append(row.snap.distance(snap_point))
    #             continue
            
    #         candidates = [
    #             self.shortest_path[nedge_point.node_start][row.node_start] + row.d_start + d_start,
    #             self.shortest_path[nedge_point.node_start][row.node_end] + row.d_end + d_start,
    #             self.shortest_path[nedge_point.node_end][row.node_start] + row.d_start + d_end,
    #             self.shortest_path[nedge_point.node_end][row.node_end] + row.d_end + d_end
    #         ]
    #         res.append(min(candidates))

    #     return np.array(res)
    
    # def resistance_point(self, point):
    #     """
    #     Return the resistance metric between a point on the network and each point pattern
    #     Args:
    #         point (shapely.geometry.Point) : point on the network
    #     Returns:
    #         np.array : resistance metric
    #     """

    #     # Get the resistance metric
    #     resistance = np.array([self.linnet.resistance_metric(point, row.snap) for i, row in self.snapped_df.iterrows()])

    #     return resistance

    # def edge_correction_factor(self, bw, kernel = 'Gaussian', save=True, parallel=True):
    #     """
    #     Calculate the edge correction factor for a point of pattern
    #     Args:
    #         point (shapely.geometry.Point) : single point pattern
    #         kernel (str) : kernel function (default='Gaussian') ['Gaussian', 'Resistance_exp', 'Resistance_mat']
    #     """
    #     if kernel == 'Gaussian':
    #         def integrand(r, point, bw, edges):
    #             # return Gaussian_kernel(r, bw) * self.linnet.circumference(point, r)
    #             return Gaussian_kernel(r, bw) * circumference(point, r, edges)
    #     elif kernel == 'Resistance_exp':
    #         def integrand(r, point, bw, edges):
    #             return powerexponential(r, bw) * circumference(point, r, edges)
    #     elif kernel == 'Resistance_mat':
    #         def integrand(r, point, bw, edges):
    #             return matern(r, bw) * circumference(point, r, edges)

    #     def compute_integral(pointR, bw, edges):
    #         # pointR to point and R
    #         point, R = pointR
    #         return integrate.quad(integrand, 0, R, args=(point, bw, edges))[0]

    #     # Return the edge correction factor
    #     if parallel:
    #         remote_tqdm = ray.remote(tqdm_ray.tqdm)

    #         @ray.remote
    #         def compute_integral_ray(pointR, bw, edges, bar: tqdm_ray.tqdm):
    #             point, R = pointR
    #             res = integrate.quad(integrand, 0, R, args=(point, bw, edges))[0]
    #             bar.update.remote(1)
    #             return res

    #         bar = remote_tqdm.remote(total=len(self.snapped_df))
    #         workers = [
    #             compute_integral_ray.remote(pointR, bw, self.edges, bar) for pointR in self.snapped_df[['snap', 'R']].values
    #         ]
    #         result = np.array(ray.get(workers))
    #         bar.close.remote()
    #     else:
    #         result = np.array(
    #             [compute_integral(pointR, bw, self.edges) for pointR in tqdm(self.snapped_df[['snap', 'R']].values)]
    #         )

    #     if save:
    #         np.save(f'data/edge_correction_{kernel}_bw{bw}.npy', result)

    #     return result
                    
    # def intensity_diggle(self, bw, point, corrections, kernel='Gaussian'):
    #     """
    #     Return the adapted Diggle's corrected estimator at a single point
    #     Args:
    #         bw (float) : bandwidth (in meters)
    #         point (shapely.geometry.Point) : point on the network
    #         kernel (str) : kernel function (default='Gaussian') ['Gaussian', 'Resistance_exp', 'Resistance_mat']
    #         corrections (np.array) : edge correction factor
    #     Returns:
    #         np.array : intensity function estimate
    #     """
    #     # Get the kernel functions
    #     if kernel == 'Gaussian':
    #         kernelsum = self.geodesic_point(point)
    #         kernelsum = Gaussian_kernel(kernelsum, bw)
    #     elif kernel == 'Resistance_exp':
    #         kernelsum = self.resistance_point(point)
    #         kernelsum = powerexponential(kernelsum, bw)
    #     elif kernel == 'Resistance_mat':
    #         kernelsum = self.resistance_point(point)
    #         kernelsum = matern(kernelsum, bw)

    #     # Return the intensity function estimate (corrected, uncorrected)
    #     return (np.sum(kernelsum / corrections), np.mean(kernelsum))
    
    # def calculate_intensity_diggle(self, bw, corrections, kernel='Gaussian', parallel=True):
    #     """
    #     Calculate the intensity function estimate using the Diggle's corrected estimator
    #     Args:
    #         bw (float) : bandwidth (in meters)
    #         kernel (str) : kernel function (default='Gaussian') ['Gaussian', 'Resistance_exp', 'Resistance_mat']
    #     """
    #     @ray.remote
    #     def intensity_diggle_ray(point, bw, corrections, kernel, bar: tqdm_ray.tqdm):
    #         bar.update.remote(1)
    #         return self.intensity_diggle(bw, point, corrections, kernel)
        
    #     if parallel:
    #         remote_tqdm = ray.remote(tqdm_ray.tqdm)
    #         bar = remote_tqdm.remote(total=self.v, desc="Calculating Intensity Function Estimate")
    #         workers = [intensity_diggle_ray.remote(point, bw, corrections, kernel, bar) for point in self.interpolates]
    #         intensity = ray.get(workers)
    #         bar.close.remote()
    #     else:
    #         intensity = [self.intensity_diggle(bw, point, corrections, kernel) for point in tqdm(self.interpolates)]

    #     # intensity > 2D array (v, 2)
    #     intensity = np.array(intensity)

    #     if hasattr(self, 'intensities'):
    #         self.intensities[kernel] = intensity[:, 0]
    #         self.intensities[f'{kernel}_uncorrected'] = intensity[:, 1]
        
    #     else:
    #         self.intensities = gpd.GeoDataFrame(
    #             {f'{kernel}': intensity[:, 0], f'{kernel}_uncorrected': intensity[:, 1]},
    #             geometry=self.interpolates
    #         )

    # def calculate_intensity_DP(self, bw, eps, seed=None, kernel='Gaussian'):
    #     """
    #     Calculate the intensity function estimate using the Diggle's corrected estimator
    #     Args:
    #         bw (float) : bandwidth (in meters)
    #         kernel (str) : kernel function (default='Gaussian') ['Gaussian', 'Resistance_exp', 'Resistance_mat']
    #         eps (float) : privacy parameter
    #     """
    #     # Check if intensities are already calculated
    #     if not hasattr(self, 'intensities') or kernel not in self.intensities:
    #         print("Calculating intensity function estimate...")
    #         self.calculate_intensity_diggle(bw, kernel)

    #     # Check if corresponding covariance matrix is already calculated
    #     if kernel in ['Resistance_exp', 'Resistance_mat']:
    #         assert hasattr(self, 'res'), "Calculate resistance matrix first"
    #     elif kernel == 'Gaussian':
    #         assert hasattr(self, 'geo'), "Calculate geodesic matrix first"

    #     # Get the intensity function estimate
    #     intensities = self.intensities[kernel]

    #     # Generate the GP noise
    #     if kernel in ['Gaussian', 'Gaussian_uncorrected']:
    #         cov = Gaussian_kernel(self.geo, bw)
    #     elif kernel in ['Resistance_exp', 'Resistance_exp_uncorrected']:
    #         cov = powerexponential(self.res, bw)
    #     elif kernel in ['Resistance_mat', 'Resistance_mat_uncorrected']:
    #         cov = matern(self.res, bw)

    #     # Generate the noise
    #     if seed is not None:
    #         np.random.seed(seed)
    #     noise = np.random.multivariate_normal(np.zeros(self.v), cov)

    #     # Sensitivity
    #     n = len(self.points)
    #     if kernel in ['Gaussian', 'Gaussian_uncorrected']:
    #         sens = np.sqrt(2) / (np.sqrt(2 * np.pi * bw ** 2) * n)
        
    #     # c(delta)
    #     c_delta = np.sqrt(2 * np.log(2 / eps))

    #     # Add noise
    #     intensities_DP = intensities + c_delta * sens * noise / eps

    #     return intensities_DP
        
    

    # def simulate(self, bw, kernel, DP=False):
    #     """
    #     Simulate inhomogeneous Poisson process on the network
    #     Args:
    #         N (int) : number of points to simulate (before thinning)
    #         intensities (np.array) : intensity function estimate at each equally spaced point
    #     """
    #     # Get the intensity function estimate
    #     intensities = self.intensities[kernel]

    #     # Max intensity
    #     max_intensity = np.max(intensities)

    #     # Simulate Poisson process
    #     N_points = np.random.poisson(max_intensity * self.linnet_union.length)
    #     points = np.random.uniform(0, 1, int(N_points))
    #     points = np.array([self.linnet_union.interpolate(i, normalized=True) for i in points])

    #     # Thin the Poisson process
    #     ## Calculate the intensity at each point
    #     intensities_new = np.array([
    #         self.intensity_diggle(bw=bw, point=point, kernel='Gaussian') for point in tqdm(points)
    #     ])
    #     ## Calculate the acceptance probability
    #     accept_prob = intensities_new / max_intensity
    #     ## Accept or reject
    #     accept = np.random.uniform(0, 1, N_points) < accept_prob
    #     points = points[accept]

    #     return gpd.GeoSeries(points)


def summary_lpp(lpp):
    """
    Summarize the LPP object
    Args:
        lpp (LPP) : LPP object
    """
    print("Point pattern on linear network")
    print("Projection:", lpp.linnet.nodes.crs)
    print("Number of points:", len(lpp.points))
    print("Total length of network:", lpp.linnet_union.length)
    print("Average intensity:", lpp.points.shape[0] / lpp.linnet_union.length)

def split_line(edge, point):
    edge_start, edge_end = edge.boundary.geoms

    if point == edge_start or point == edge_end:
        return gpd.GeoSeries([edge])

    line1 = shapely.geometry.LineString([edge_start, point])
    line2 = shapely.geometry.LineString([point, edge_end])
    lines = gpd.GeoSeries([line1, line2])

    return lines



def discretize_network(linnet, resolution):
    """
    Discretize the network into equally spaced points
    Args:
        linnet (Linnet) : Linnet object
        resolution (float) : resolution of equally spaced points along the network
    Returns:
        linnet (Linnet) : Linnet object with discretized network
    """
    # interpolates
    v = int(linnet.edges.unary_union.length / resolution)
    interpolates = gpd.GeoSeries([linnet.edges.unary_union.interpolate(i, normalized=True) for i in np.linspace(0, 1, v)])

    # Discretize the network
    edges = linnet.edges.geometry # GeoSeries
    err = []

    for point in tqdm(interpolates):
        n_index = edges.geometry.sindex.nearest(point)[1][0]
        try:
            new = split_line(edges.geometry[n_index], point,)
            edges.drop(n_index, inplace=True)
            edges = pd.concat([edges, new], ignore_index=True)
        except ValueError:
            err.append(n_index)
    

    edges = gpd.GeoDataFrame(geometry=edges)

    if len(err) > 0:
        print(f"Error in splitting {len(err)} edges")

    return Linnet(edges)


# def intensity_naive(points, L, resolution, kernel):
#     # Discretize network
#     linnet = discretize_network(L, resolution)

#     # Calculate the intensity function estimate (no edge correction)
