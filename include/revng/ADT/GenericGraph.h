#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/Support/Debug.h"

struct Empty {};

template<typename Node, size_t SmallSize = 16, bool HasEntryNode = true>
class GenericGraph;

template<typename T>
class Parent {
private:
  T *Parent;

public:
  T *getParent() const { return Parent; }
  void setParent(T *Parent) { this->Parent = Parent; }
};

/// Data structure for edge labels
template<typename Node, typename EdgeLabel>
struct Edge : public EdgeLabel {
  Edge(Node *Neighbor) : Neighbor(Neighbor) {}
  Edge(Node *Neighbor, EdgeLabel EL) : EdgeLabel(EL), Neighbor(Neighbor) {}
  Node *Neighbor;
};

//
// Structured binding support for Edge
//
template<size_t I, typename Node, typename EdgeLabel>
auto &get(Edge<Node, EdgeLabel> &E) {
  if constexpr (I == 0)
    return E.Neighbor;
  else
    return static_cast<EdgeLabel &>(E);
}

template<size_t I, typename Node, typename EdgeLabel>
auto &get(const Edge<Node, EdgeLabel> &E) {
  if constexpr (I == 0)
    return E.Neighbor;
  else
    return static_cast<EdgeLabel &>(E);
}

template<size_t I, typename Node, typename EdgeLabel>
auto &&get(Edge<Node, EdgeLabel> &&E) {
  if constexpr (I == 0)
    return std::move(E.Neighbor);
  else
    return std::move(static_cast<EdgeLabel &&>(E));
}

namespace std {
template<typename Node, typename EdgeLabel>
struct std::tuple_size<Edge<Node, EdgeLabel>>
  : std::integral_constant<size_t, 2> {};

template<typename Node, typename EdgeLabel>
struct std::tuple_element<0, Edge<Node, EdgeLabel>> {
  using type = Node *;
};

template<typename Node, typename EdgeLabel>
struct std::tuple_element<1, Edge<Node, EdgeLabel>> {
  using type = EdgeLabel;
};

} // namespace std

/// Basic nodes type, only forward edges, possibly with parent
template<typename Inheritor,
         typename EdgeLabel = Empty,
         bool HasParent = true,
         size_t SmallSize = 2>
class ForwardNode : public std::conditional_t<HasParent,
                                              Parent<GenericGraph<Inheritor>>,
                                              Empty> {
public:
  static constexpr bool is_forward_node = true;
  static constexpr bool has_parent = HasParent;
  using Edge = Edge<Inheritor, EdgeLabel>;

public:
  static Inheritor *&getNeighbor(Edge &E) { return E.Neighbor; }
  static const Inheritor *&getConstNeighbor(const Edge &E) {
    return E.Neighbor;
  }

  static Edge *getEdgePointer(Edge &E) { return &E; }
  static const Edge *getConstEdgePointer(const Edge &E) { return &E; }

private:
  using iterator_filter = decltype(&getNeighbor);
  using const_iterator_filter = decltype(&getConstNeighbor);

public:
  using NeighborContainer = llvm::SmallVector<Edge, SmallSize>;
  using child_iterator = llvm::mapped_iterator<Edge *, iterator_filter>;
  using const_child_iterator = llvm::mapped_iterator<Edge *,
                                                     const_iterator_filter>;
  using edge_iterator = typename NeighborContainer::iterator;
  using const_edge_iterator = typename NeighborContainer::const_iterator;

public:
  // This stuff is needed by the DominatorTree implementation
  void printAsOperand(llvm::raw_ostream &, bool) const { revng_abort(); }

public:
  void addSuccessor(Inheritor *NewSuccessor) {
    Successors.emplace_back(NewSuccessor);
  }

  void addSuccessor(Inheritor *NewSuccessor, EdgeLabel EL) {
    Successors.emplace_back(NewSuccessor, EL);
  }

public:
  child_iterator removeSuccessor(child_iterator It) {
    auto InternalIt = Successors.erase(It.getCurrent());
    return child_iterator(InternalIt, getNeighbor);
  }

  edge_iterator removeSuccessorEdge(edge_iterator It) {
    return Successors.erase(It);
  }

public:
  llvm::iterator_range<const_child_iterator> successors() const {
    return toNeighborRange(Successors);
  }

  llvm::iterator_range<child_iterator> successors() {
    return toNeighborRange(Successors);
  }

  llvm::iterator_range<const_edge_iterator> successor_edges() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  llvm::iterator_range<edge_iterator> successor_edges() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

protected:
  template<typename T>
  static llvm::iterator_range<child_iterator> toNeighborRange(T &Neighbors) {
    auto Range = llvm::make_range(Neighbors.begin(), Neighbors.end());
    if constexpr (std::is_const_v<T>) {
      return llvm::map_range(Range, &getConstNeighbor);
    } else {
      return llvm::map_range(Range, &getNeighbor);
    }
  }

private:
  NeighborContainer Successors;
};

/// Same as ForwardNode, but with backward links too
template<typename Inheritor,
         typename EdgeLabel = Empty,
         bool HasParent = true,
         size_t SmallSize = 2>
class BidirectionalNode
  : public ForwardNode<Inheritor, EdgeLabel, HasParent, SmallSize> {
public:
  static const bool is_bidirectional_node = true;

public:
  using Base = ForwardNode<Inheritor, EdgeLabel, HasParent, SmallSize>;
  using NeighborContainer = typename Base::NeighborContainer;
  using child_iterator = typename Base::child_iterator;
  using const_child_iterator = typename Base::const_child_iterator;
  using edge_iterator = typename Base::edge_iterator;
  using const_edge_iterator = typename Base::const_edge_iterator;

public:
  void addSuccessor(Inheritor *NewSuccessor) {
    Base::addSuccessor({ NewSuccessor });
    NewSuccessor->Predecessors.emplace_back(static_cast<Inheritor *>(this));
  }

  void addSuccessor(Inheritor *NewSuccessor, EdgeLabel EL) {
    Base::addSuccessor(NewSuccessor, EL);
    NewSuccessor->Predecessors.emplace_back(static_cast<Inheritor *>(this), EL);
  }

  void addPredecessor(Inheritor *NewPredecessor) {
    Predecessors.emplace_back(NewPredecessor);
    NewPredecessor->addSuccessor(static_cast<Inheritor *>(this));
  }

  void addPredecessor(Inheritor *NewPredecessor, EdgeLabel EL) {
    Predecessors.emplace_back(NewPredecessor, EL);
    NewPredecessor->addSuccessor(static_cast<Inheritor *>(this), EL);
  }

public:
  child_iterator removePredecessor(child_iterator It) {
    auto InternalIt = Predecessors.erase(It.getCurrent());
    return child_iterator(InternalIt, Base::getNeighbor);
  }

  edge_iterator removePredecessorEdge(edge_iterator It) {
    return Predecessors.erase(It);
  }

public:
  llvm::iterator_range<const_child_iterator> predecessors() const {
    return this->toNeighborRange(Predecessors);
  }

  llvm::iterator_range<child_iterator> predecessors() {
    return this->toNeighborRange(Predecessors);
  }

  llvm::iterator_range<const_edge_iterator> predecessor_edges() const {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  llvm::iterator_range<edge_iterator> predecessor_edges() {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

private:
  NeighborContainer Predecessors;
};

/// Simple data structure to hold the EntryNode of a GenericGraph
template<typename NodeT>
class EntryNode {
private:
  NodeT *EntryNode;

public:
  NodeT *getEntryNode() const { return EntryNode; }
  void setEntryNode(NodeT *EntryNode) { this->EntryNode = EntryNode; }
};

/// Generic graph parametrized in the node type
///
/// This graph owns its nodes (but not the edges).
/// It can optionally have an elected entry point.
template<typename NodeT, size_t SmallSize, bool HasEntryNode>
class GenericGraph
  : public std::conditional_t<HasEntryNode, EntryNode<NodeT>, Empty> {
public:
  static const bool is_generic_graph = true;
  using NodesContainer = llvm::SmallVector<std::unique_ptr<NodeT>, SmallSize>;
  using Node = NodeT;

private:
  using nodes_iterator_impl = typename NodesContainer::iterator;
  using const_nodes_iterator_impl = typename NodesContainer::const_iterator;

public:
  static NodeT *getNode(std::unique_ptr<NodeT> &E) { return E.get(); }
  static const NodeT *getConstNode(const std::unique_ptr<NodeT> &E) {
    return E.get();
  }

  // TODO: these iterators will not work with llvm::filter_iterator,
  //       since the mapped type is not a reference
  using nodes_iterator = llvm::mapped_iterator<nodes_iterator_impl,
                                               decltype(&getNode)>;
  using const_nodes_iterator = llvm::mapped_iterator<const_nodes_iterator_impl,
                                                     decltype(&getConstNode)>;

  llvm::iterator_range<nodes_iterator> nodes() {
    return llvm::map_range(llvm::make_range(Nodes.begin(), Nodes.end()),
                           getNode);
  }

  llvm::iterator_range<const_nodes_iterator> nodes() const {
    return llvm::map_range(llvm::make_range(Nodes.begin(), Nodes.end()),
                           getConstNode);
  }

  size_t size() const { return Nodes.size(); }

public:
  template<class... Args>
  NodeT *addNode(Args &&... A) {
    Nodes.push_back(std::make_unique<NodeT>(std::forward<Args>(A)...));
    if constexpr (NodeT::has_parent)
      Nodes.back()->setParent(this);
    return Nodes.back().get();
  }

  nodes_iterator removeNode(nodes_iterator It) {
    auto InternalIt = Nodes.erase(It.getCurrent());
    return nodes_iterator(InternalIt, getNode);
  }

private:
  NodesContainer Nodes;
};

//
// GraphTraits implementation for GenericGraph
//
namespace llvm {

/// Implement GraphTraits<ForwardNode>
template<typename T>
struct GraphTraits<T *, std::enable_if_t<T::is_forward_node>> {
public:
  using NodeRef = T *;
  using ChildIteratorType = std::conditional_t<std::is_const_v<T>,
                                               typename T::const_child_iterator,
                                               typename T::child_iterator>;

  using EdgeRef = typename T::Edge &;
  template<typename Ty, typename True, typename False>
  using if_const = std::conditional_t<std::is_const_v<Ty>, True, False>;
  using ChildEdgeIteratorType = if_const<T,
                                         typename T::const_edge_iterator,
                                         typename T::edge_iterator>;

public:
  static ChildIteratorType child_begin(NodeRef N) {
    return N->successors().begin();
  }

  static ChildIteratorType child_end(NodeRef N) {
    return N->successors().end();
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->successor_edges().begin();
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef N) {
    return N->successor_edges().end();
  }

  static NodeRef edge_dest(EdgeRef Edge) { return Edge.Neighbor; }

  static NodeRef getEntryNode(NodeRef N) { return N; };
};

/// Implement GraphTraits<GenericGraph>
template<typename T>
struct GraphTraits<T *, std::enable_if_t<T::is_generic_graph>>
  : public GraphTraits<typename T::Node *> {

  using NodeRef = std::conditional_t<std::is_const_v<T>,
                                     const typename T::Node *,
                                     typename T::Node *>;
  using nodes_iterator = std::conditional_t<std::is_const_v<T>,
                                            typename T::const_nodes_iterator,
                                            typename T::nodes_iterator>;

  static NodeRef getEntryNode(T *G) { return G->getEntryNode(); }

  static nodes_iterator nodes_begin(T *G) { return G->nodes().begin(); }

  static nodes_iterator nodes_end(T *G) { return G->nodes().end(); }

  static size_t size(T *G) { return G->size(); }
};

/// Implement GraphTraits<Inverse<BidirectionalNode>>
template<typename T>
struct GraphTraits<llvm::Inverse<T *>,
                   std::enable_if_t<T::is_bidirectional_node>> {
public:
  using NodeRef = T *;
  using ChildIteratorType = std::conditional_t<std::is_const_v<T>,
                                               typename T::const_child_iterator,
                                               typename T::child_iterator>;

public:
  static ChildIteratorType child_begin(NodeRef N) {
    return N->predecessors().begin();
  }

  static ChildIteratorType child_end(NodeRef N) {
    return N->predecessors().end();
  }

  static NodeRef getEntryNode(llvm::Inverse<NodeRef> N) { return N.Graph; };
};

} // namespace llvm
