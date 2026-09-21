//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>
#include <type_traits>
#include <vector>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Progress.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng/Support/Debug.h"

#include "DLAStep.h"
#include "RemoveBackedges.h"

using namespace llvm;

static Logger LogVerbose("dla-collapse-verbose");

namespace dla {

using LTSN = LayoutTypeSystemNode;
using GraphNodeT = LTSN *;
using InstanceNodeT = EdgeFilteredGraph<GraphNodeT, isInstanceEdge>;
using EqualityNodeT = EdgeFilteredGraph<GraphNodeT, isEqualityEdge>;
using InstanceOffset0EdgeT = EdgeFilteredGraph<GraphNodeT, isInstanceOff0>;
using NonPointerNodeT = EdgeFilteredGraph<GraphNodeT, isNotPointerEdge>;

static bool sizeGreater(const LTSN *LHS, const LTSN *RHS) {
  return LHS->Size > RHS->Size;
}

template<typename NodeT>
static bool collapseSCCs(LayoutTypeSystem &TS) {

  std::set<GraphNodeT> VisitedNodes;

  using scc_t = std::vector<GraphNodeT>;
  llvm::SmallVector<scc_t, 0> ToCollapse;

  // We cannot start just from the roots because we cannot exclude that there
  // are loops without entries.
  for (const auto &Node : llvm::nodes(&TS)) {
    revng_assert(Node != nullptr);
    revng_log(LogVerbose, "## Analyzing SCCs from  " << Node);
    if (VisitedNodes.contains(Node)) {
      revng_log(LogVerbose, "## Was already visited");
      continue;
    }

    llvm::scc_iterator<NodeT> I = llvm::scc_begin(NodeT(Node));
    llvm::scc_iterator<NodeT> E = llvm::scc_end(NodeT(Node));
    for (const auto &SCC : llvm::make_range(I, E)) {
      revng_assert(not SCC.empty());
      if (VisitedNodes.contains(SCC[0]))
        continue;
      VisitedNodes.insert(SCC.begin(), SCC.end());

      if (LogVerbose.isEnabled()) {
        revng_log(LogVerbose, "# SCC has " << SCC.size() << " elements:");
        size_t I = 0;
        for (const LTSN *N : SCC)
          revng_log(LogVerbose, "  - " << I << ": " << N);
        revng_log(LogVerbose, "# SCC End");
      }

      // If the SCC is larger than just a single point we track all the nodes
      // that compose it as EqualityVisitedNodes, and we then add the SCC to the
      // set of SCC ToCollapse.
      if (SCC.size() > 1)
        ToCollapse.push_back(SCC);
    }
  }
  revng_log(LogVerbose, "## Collapsing " << ToCollapse.size() << " SCCs");

  for (scc_t &SCC : ToCollapse) {
    revng_log(LogVerbose, "# SCC with " << SCC.size() << " elements");
    revng_assert(not SCC.empty());
    // Reorder the SCC so that nodes with the same size are adjacent.
    // Nodes with larger sizes come first.
    llvm::stable_sort(SCC, sizeGreater);

    // Now, merge all the nodes that have the same size, while noting the
    // leaders.
    llvm::SmallSetVector<LayoutTypeSystemNode *, 8> Leaders;
    {
      auto LeaderIt = SCC.begin();
      auto CurrentSize = (*LeaderIt)->Size;
      auto End = SCC.end();

      while (LeaderIt != End) {

        Leaders.insert(*LeaderIt);

        // Find the first node with a size different from CurrentSize.
        auto EndCurrentSize = std::next(LeaderIt);
        while (EndCurrentSize != End and (*EndCurrentSize)->Size == CurrentSize)
          ++EndCurrentSize;

        // Merge all the nodes with the same size.
        TS.mergeNodes({ &*LeaderIt, &*EndCurrentSize });

        // Go to the next size.
        LeaderIt = EndCurrentSize;
      }
    }

    // Now, if the last Leader has Size 0, we merge it with the first Leader,
    // which represents the largest type.
    if (Leaders.size() > 1 and Leaders.back()->Size == 0) {
      TS.mergeNodes(llvm::SmallVector{ Leaders.front(), Leaders.back() });
      Leaders.pop_back();
    }

    // If a single leader is left, we're done, all the nodes have been merged
    // together.
    if (Leaders.size() < 2)
      continue;

    // If we're left with more than one leader, we just want to remove all edges
    // between them and replace them with offset-at-0 edges, from the larger
    // leader to the smaller.
    auto LeaderIt = Leaders.begin();
    auto LeaderEnd = Leaders.end();
    for (; LeaderIt != LeaderEnd; ++LeaderIt) {

      // For each leader we erase all the non-pointer edges going to other
      // leaders.
      auto *Leader = *LeaderIt;
      using DLAGraph = llvm::GraphTraits<LayoutTypeSystemNode *>;
      auto EdgeIt = DLAGraph::child_edge_begin(Leader);
      auto EdgeEnd = DLAGraph::child_edge_end(Leader);
      auto EdgeNext = DLAGraph::child_edge_end(Leader);

      for (; EdgeIt != EdgeEnd; EdgeIt = EdgeNext) {

        EdgeNext = std::next(EdgeIt);
        auto &Edge = *EdgeIt;

        // Skip pointers
        if (isPointerEdge(Edge))
          continue;

        const auto &[Child, Tag] = Edge;

        // Skip edges that go to non-leaders
        if (not Leaders.contains(Child))
          continue;

        TS.eraseEdge(Leader, EdgeIt);
      }

      // Then, for each leader, we add an instance-at-offset-0 edge to the next
      // leader, which is the one with immediately lower size.
      if (auto NextLeaderIt = std::next(LeaderIt); NextLeaderIt != LeaderEnd)
        TS.addInstanceLink(*LeaderIt, *NextLeaderIt, dla::OffsetExpression{});
    }
  }

  return ToCollapse.size();
}

/// Collects the strongly connected components of \p NodeT that have more than
/// one node.
template<typename NodeT>
static llvm::SmallVector<std::vector<GraphNodeT>, 0>
getSCCs(LayoutTypeSystem &TS) {
  std::set<GraphNodeT> VisitedNodes;
  llvm::SmallVector<std::vector<GraphNodeT>, 0> Result;

  for (const auto &Node : llvm::nodes(&TS)) {
    revng_assert(Node != nullptr);
    if (VisitedNodes.contains(Node))
      continue;

    for (const auto &SCC : llvm::make_range(llvm::scc_begin(NodeT(Node)),
                                            llvm::scc_end(NodeT(Node)))) {
      revng_assert(not SCC.empty());
      if (VisitedNodes.contains(SCC[0]))
        continue;
      VisitedNodes.insert(SCC.begin(), SCC.end());

      if (SCC.size() > 1)
        Result.push_back(SCC);
    }
  }

  return Result;
}

using NonStridedInstanceT = EdgeFilteredGraph<GraphNodeT, isNonStridedInstance>;

/// Sums the offsets of the non strided instance edges that stay inside \p SCC.
static uint64_t getStrideOf(const std::vector<GraphNodeT> &SCC) {
  llvm::SmallPtrSet<GraphNodeT, 8> InSCC(SCC.begin(), SCC.end());

  uint64_t Result = 0;
  for (GraphNodeT Node : SCC)
    for (const auto &Edge : Node->Successors)
      if (isNonStridedInstance(Edge) and InSCC.contains(Edge.first))
        Result += Edge.second->getOffsetExpr().Offset;

  return Result;
}

static bool collapseInstanceSCCIntoArray(LayoutTypeSystem &TS) {
  bool Changed = false;

  for (std::vector<GraphNodeT> &SCC : getSCCs<NonStridedInstanceT>(TS)) {
    // How much the offsets add up to going around the loop is the stride of
    // the array it describes. When they add up to nothing the loop is made of
    // edges at offset 0 only, which says the nodes are the same thing rather
    // than an array, and collapsing that is somebody else's job.
    uint64_t Stride = getStrideOf(SCC);
    if (Stride == 0)
      continue;

    revng_log(LogVerbose,
              "# SCC with " << SCC.size()
                            << " elements is an array with stride " << Stride);

    // Merging keeps the first node, and it wants the largest one.
    llvm::stable_sort(SCC, sizeGreater);
    TS.mergeNodes(SCC);
    GraphNodeT Collapsed = SCC.front();

    // Collapsing turned the edges that stayed inside the SCC into self edges,
    // and merging drops those, so what is left of the loop is nothing. Put the
    // array back in front of everything that reaches the collapsed node, with
    // the node itself as its element.
    llvm::SmallVector<LTSN::NeighborIterator> Incoming;
    auto It = Collapsed->Predecessors.begin();
    auto PredecessorsEnd = Collapsed->Predecessors.end();
    for (; It != PredecessorsEnd; ++It)
      if (isInstanceEdge(*It))
        Incoming.push_back(It);

    for (LTSN::NeighborIterator &EdgeIt : Incoming) {
      LTSN *Array = TS.createArtificialLayoutType();
      TS.moveEdgeTarget(Collapsed, Array, EdgeIt, 0);

      // The length is unknown: the loop says how far apart the elements are,
      // not how many of them there are.
      OffsetExpression OE{};
      OE.Strides.push_back(Stride);
      OE.TripCounts.push_back(std::nullopt);
      TS.addInstanceLink(Array, Collapsed, std::move(OE));
    }

    Changed = true;
  }

  return Changed;
}

/// Erases the non strided instance edges that reach \p Node from inside \p
/// InSCC.
static bool
dropNonStridedEdgesInto(LayoutTypeSystem &TS,
                        GraphNodeT Node,
                        const llvm::SmallPtrSetImpl<GraphNodeT> &InSCC) {
  llvm::SmallVector<GraphNodeT> Predecessors;
  for (const auto &Edge : Node->Predecessors)
    if (isNonStridedInstance(Edge) and InSCC.contains(Edge.first))
      Predecessors.push_back(Edge.first);

  bool Changed = false;
  for (GraphNodeT Predecessor : Predecessors) {
    auto It = Predecessor->Successors.begin();
    while (It != Predecessor->Successors.end()) {
      if (It->first == Node and isNonStridedInstance(*It)) {
        It = TS.eraseEdge(Predecessor, It);
        Changed = true;
      } else {
        ++It;
      }
    }
  }

  return Changed;
}

/// Collects the nodes of \p SCC that a strided edge staying inside it starts
/// from.
static llvm::SmallPtrSet<GraphNodeT, 8>
getStridedEdgeSources(const std::vector<GraphNodeT> &SCC,
                      const llvm::SmallPtrSetImpl<GraphNodeT> &InSCC) {
  llvm::SmallPtrSet<GraphNodeT, 8> Result;

  for (GraphNodeT Node : SCC)
    for (const auto &Edge : Node->Successors)
      if (isStridedInstance(Edge) and InSCC.contains(Edge.first))
        Result.insert(Node);

  return Result;
}

/// Breaks the loops of instance edges that collecting the arrays left behind.
static bool breakRemainingInstanceSCCs(LayoutTypeSystem &TS) {
  bool Changed = false;

  // A loop with no strided edge in it either became an array just above, or is
  // made of edges at offset 0 only, which say the nodes are the same thing and
  // are collapsed further down the line. Either way it is not for this to
  // touch.

  // A strided edge says its target repeats inside its source, which is worth
  // keeping, so break the loop elsewhere: drop what reaches the node such an
  // edge starts from.
  for (const std::vector<GraphNodeT> &SCC : getSCCs<InstanceNodeT>(TS)) {
    llvm::SmallPtrSet<GraphNodeT, 8> InSCC(SCC.begin(), SCC.end());

    for (GraphNodeT Node : getStridedEdgeSources(SCC, InSCC))
      Changed |= dropNonStridedEdgesInto(TS, Node, InSCC);
  }

  // Whatever is still a loop is made of edges that all say something worth
  // keeping, and there is no way to keep them all, so cut it.
  for (const std::vector<GraphNodeT> &SCC : getSCCs<InstanceNodeT>(TS)) {
    llvm::SmallPtrSet<GraphNodeT, 8> InSCC(SCC.begin(), SCC.end());

    if (getStridedEdgeSources(SCC, InSCC).empty())
      continue;

    revng_log(LogVerbose,
              "# Cutting a loop of " << SCC.size()
                                     << " nodes that could not be an array");

    for (GraphNodeT Node : SCC) {
      auto It = Node->Successors.begin();
      while (It != Node->Successors.end()) {
        if (isInstanceEdge(*It) and InSCC.contains(It->first)) {
          It = TS.eraseEdge(Node, It);
          Changed = true;
        } else {
          ++It;
        }
      }
    }
  }

  return Changed;
}

static bool collapseEqualitySCC(LayoutTypeSystem &TS) {
  return collapseSCCs<EqualityNodeT>(TS);
}

static bool collapseInstanceAtOffset0SCC(LayoutTypeSystem &TS) {
  return collapseSCCs<InstanceOffset0EdgeT>(TS);
}

bool CollapseEqualitySCC::runOnTypeSystem(LayoutTypeSystem &TS) {

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency());

  revng_log(LogVerbose, "#### Collapsing Equality SCC: ... ");
  bool Changed = collapseEqualitySCC(TS);
  revng_log(LogVerbose, "#### Collapsing Equality SCC: Done!");

  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());
    revng_assert(TS.verifyNoEquality());
  }

  return Changed;
}

bool CollapseInstanceSCCIntoArray::runOnTypeSystem(LayoutTypeSystem &TS) {

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency());

  revng_log(LogVerbose, "#### Collapsing Instance SCC into array: ... ");
  bool Changed = collapseInstanceSCCIntoArray(TS);
  Changed |= breakRemainingInstanceSCCs(TS);
  revng_log(LogVerbose, "#### Collapsing Instance SCC into array: Done!");

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency());

  return Changed;
}

bool CollapseInstanceAtOffset0SCC::runOnTypeSystem(LayoutTypeSystem &TS) {
  Task T(2, "runOnTypeSystem");

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency());

  T.advance("collapseInstanceAtOffset0SCC");
  revng_log(LogVerbose, "#### Collapsing Instance-at-offset-0 SCC: ... ");
  bool Changed = collapseInstanceAtOffset0SCC(TS);
  revng_log(LogVerbose, "#### Collapsing Instance-at-offset-0 SCC: Done!");

  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());
    revng_assert(TS.verifyInstanceAtOffset0DAG());
  }

  T.advance("removeInstanceBackedgesFromInstanceAtOffset0Loops");
  Changed |= removeInstanceBackedgesFromInstanceAtOffset0Loops(TS);

  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());
    revng_assert(TS.verifyInstanceAtOffset0DAG());
    revng_assert(TS.verifyInstanceDAG());
  }

  return Changed;
}

} // end namespace dla
