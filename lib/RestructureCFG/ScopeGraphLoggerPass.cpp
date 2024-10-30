//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/RestructureCFG/ScopeGraphLoggerPass.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

// Debug logger
Logger<> ScopeGraphLoggerPassLogger("scope-graph-logger");

class ScopeGraphLoggerPassImpl {
  llvm::Function &F;

public:
  ScopeGraphLoggerPassImpl(llvm::Function &F) : F(F) {}

public:
  bool run() {
    revng_log(ScopeGraphLoggerPassLogger,
              "ScopeGraph of function: " << F.getName().str() << "\n");
    for (llvm::BasicBlock &BB : F) {
      revng_log(ScopeGraphLoggerPassLogger,
                "Block " << BB.getName().str() << " successors:\n");
      using ScopeGraph = llvm::GraphTraits<Scope<llvm::BasicBlock *>>;
      for (auto Succ = ScopeGraph::child_begin(&BB);
           Succ != ScopeGraph::child_end(&BB);
           ++Succ) {
        revng_log(ScopeGraphLoggerPassLogger,
                  "  " << (*Succ)->getName().str() << "\n");
      }
    }

    return false; // The function was not modified
  }
};

char ScopeGraphLoggerPass::ID = 0;

static constexpr const char *Flag = "scope-graph-logger";
using Reg = llvm::RegisterPass<ScopeGraphLoggerPass>;
static Reg X(Flag, "Dump edge information on the `ScopeGraph`");

bool ScopeGraphLoggerPass::runOnFunction(llvm::Function &F) {

  // Instantiate and call the `Impl` class
  ScopeGraphLoggerPassImpl SGLPImpl(F);
  return SGLPImpl.run();
}

void ScopeGraphLoggerPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {

  // This is a read only analysis, that does not touch the IR
  AU.setPreservesAll();
}
