#include "boost_mst.hpp"

// Returns the lenght of a minimum spanning tree defined by a fully connected graph with edge weights
//
// Args:
//   adj_list : Python list with edge weights arrange in 'for i: for j:' order.
//                ex: [a_10, a_20, a_21, a_30, a_31, ... ]
//   node_N : Number of nodes in the graph equals number of points.
// 
// Returns:
//   Length of the tree (i.e. sum of edge weights)
double compMstLength(pybind11::list adj_list, const int node_N)
{

  /******************
   * BUILD UP GRAPH
   ******************/
  typedef boost::adjacency_list <boost::vecS,        // How to store vertices
                                 boost::vecS,        // How to store edges
                                 boost::undirectedS, // Type of graph
                                 boost::property<boost::vertex_distance_t, double>, // Type to hold 'vertex properties'
                                 boost::property<boost::edge_weight_t, double>      // Type to hold 'edge properties'
                                 > Graph;
  
  typedef Graph::edge_descriptor Edge;

  //Graph g(edges, edges + sizeof(edges) / sizeof(E), weights, num_nodes);  // node_N
  Graph g(node_N);

  // Read the list element-by-element in row-wise order for the lower triangular indices, and store edges and weights
  int i = 1;
  int j = 0;
  
  // Create edges row-wise according to the lower triangular indices
  // Edge weights in 'adj_list' are ordered accordingly
  for (pybind11::handle obj : adj_list)
  {
    double w = pybind11::cast<double>(obj);
    boost::add_edge(i, j, w, g);

    j++;
    // Skip elements on the diagonal (i.e. self-connections)
    if(i == j)
    {
      i++;
      j = 0;
    }
  } 
  
  /***************
   * COMPUTE MST
   ***************/
  std::vector < boost::graph_traits < Graph >::vertex_descriptor > p(num_vertices(g));

  prim_minimum_spanning_tree(g, &p[0]);

  // Obtain the total tree length by summing the length between all parent-child node edges
  double mst_length = 0.0;
  for (std::size_t i = 0; i != p.size(); ++i)
    if (p[i] != i)
    {
      int node_ch = i;
      int node_par = p[i];

      // Extract weight from edge between parent and child
      std::pair<Edge, bool> ed = boost::edge(node_par, node_ch,g);
      double w = get(boost::edge_weight_t(), g, ed.first);
      
      mst_length += w;
    }

  return mst_length;
  
}
