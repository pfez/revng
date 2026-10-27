//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/IPO/StripSymbols.h"

#include "revng/FunctionIsolation/StableStripDeadDebugInfo.h"
#include "revng/Support/Assert.h"
#include "revng/Support/SimplePassManager.h"

using namespace llvm;

char StableStripDeadDebugInfo::ID = 0;

using Register = RegisterPass<StableStripDeadDebugInfo>;
static Register X("stable-strip-dead-debug-info",
                  "Strip Dead Debug Info, Keeping llvm.dbg.cu In Order");

static constexpr const char *CompileUnitsName = "llvm.dbg.cu";

static SmallVector<MDNode *, 8> compileUnits(Module &M) {
  SmallVector<MDNode *, 8> Result;

  if (NamedMDNode *CompileUnits = M.getNamedMetadata(CompileUnitsName)) {
    unsigned Count = CompileUnits->getNumOperands();
    for (unsigned Index = 0; Index < Count; ++Index)
      Result.push_back(CompileUnits->getOperand(Index));
  }

  return Result;
}

bool StableStripDeadDebugInfo::runOnModule(Module &M) {
  // Take note of the order the compile units are in before the LLVM pass
  // rewrites the list
  SmallVector<MDNode *, 8> Before = compileUnits(M);

  SimplePassManager PM;
  PM.addPass(StripDeadDebugInfoPass());
  PM.run(M);

  // With less than two survivors there is no order to disagree about
  SmallVector<MDNode *, 8> After = compileUnits(M);
  if (After.size() < 2)
    return true;

  // Keep the survivors in the order they had. Anything the pass introduced,
  // which it is not expected to do, keeps the position it was left in.
  SmallPtrSet<MDNode *, 8> Survivors(After.begin(), After.end());
  SmallVector<MDNode *, 8> Ordered;
  for (MDNode *CompileUnit : Before)
    if (Survivors.contains(CompileUnit))
      Ordered.push_back(CompileUnit);

  SmallPtrSet<MDNode *, 8> Reordered(Ordered.begin(), Ordered.end());
  for (MDNode *CompileUnit : After)
    if (not Reordered.contains(CompileUnit))
      Ordered.push_back(CompileUnit);

  revng_assert(Ordered.size() == After.size());

  NamedMDNode *CompileUnits = M.getNamedMetadata(CompileUnitsName);
  CompileUnits->clearOperands();
  for (MDNode *CompileUnit : Ordered)
    CompileUnits->addOperand(CompileUnit);

  return true;
}
