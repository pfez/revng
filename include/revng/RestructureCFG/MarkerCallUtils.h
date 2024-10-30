#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ModRef.h"

#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

std::string ScopeCloserMarkerCallFunctionName = "scope-closer";
std::string GotoBlockMarkerCallFunctionName = "goto-block";

inline llvm::Function *
getMarkerCallFunction(FunctionTags::Tag &MarkerFunctionTag, llvm::Module *M) {
  llvm::Function *MarkerCallFunction = nullptr;

  // We could early break from this loop but we would loose the ability of
  // asserting that a single marker function is present
  for (llvm::Function &F : MarkerFunctionTag.functions(M)) {
    revng_assert(not MarkerCallFunction);
    MarkerCallFunction = &F;
  }

  revng_assert(MarkerCallFunction);
  return MarkerCallFunction;
}

inline std::string getMarkerCallFunctionName(llvm::Function *F) {
  return F->getName().str();
}

/// This is a helper class used to create the markers in the basic blocks needed
/// to handle virtual edges, such as the scope closer edges or the goto edges.
/// Its role should be similar somewhat to the `llvm::IRBuilder` role.
class MarkerCallBuilder {
private:
  llvm::BasicBlock *BB = nullptr;
  llvm::Function *MarkerCallF = nullptr;

public:
  MarkerCallBuilder(FunctionTags::Tag &MarkerFunctionTag, llvm::Function *F) {
    llvm::LLVMContext &C = getContext(F);
    llvm::Module *M = getModule(F);
    MarkerCallF = getMarkerCallFunction(MarkerFunctionTag, M);

    // Create the `MarkerCallFunction` if does not exists
    if (not MarkerCallF) {
      // Create or get the marker function declaration, that will be called by
      // the inserted markers
      llvm::Type *BlockAddressTy = llvm::Type::getInt8PtrTy(C);
      auto *FT = llvm::FunctionType::get(llvm::Type::getVoidTy(C),
                                         { BlockAddressTy },
                                         false);

      using llvm::Function;
      using llvm::GlobalValue;
      MarkerCallF = llvm::cast<
        llvm::Function>(M->getOrInsertFunction(MarkerFunctionTag.name(), FT)
                          .getCallee());
      MarkerCallF->setLinkage(llvm::GlobalValue::ExternalLinkage);
      MarkerCallF->addFnAttr(llvm::Attribute::OptimizeNone);
      MarkerCallF->addFnAttr(llvm::Attribute::NoInline);
      MarkerCallF->addFnAttr(llvm::Attribute::NoMerge);
      MarkerCallF->addFnAttr(llvm::Attribute::NoUnwind);
      MarkerCallF->addFnAttr(llvm::Attribute::WillReturn);
      using llvm::MemoryEffects;
      MarkerCallF->setMemoryEffects(MemoryEffects::inaccessibleMemOnly());

      // Add the tag to the
      MarkerFunctionTag.addTo(MarkerCallF);
    }
  }

public:
  // Set the insertion point of the `MarkerCallBuilder`
  void setInsertPoint(llvm::BasicBlock *NewBB) { BB = NewBB; }

  // If the argument is `nullptr`, it means that we are inserting a marker for
  void insertMarkerCall(llvm::BasicBlock *BasicBlockTarget = nullptr) {

    // We must have an insertion point
    revng_assert(BB);

    // We assume that when inserting a `goto` edge, the original block had a
    // single regular successor on the CFG
    if (MarkerCallF->getName() == GotoBlockMarkerCallFunctionName) {
      llvm::Instruction *Terminator = BB->getTerminator();
      revng_assert(Terminator->getNumSuccessors() == 1);
    }

    // We always insert the marker as the penultimate instruction in a
    // `BasicBlock`, regardless of the type of the marker we are inserting
    llvm::Instruction *Terminator = BB->getTerminator();
    llvm::IRBuilder<> Builder(Terminator);

    // When creating a `goto_block` marker, we don't have a target block as
    // argument, and we place a `nullptr` as argument of the marker call
    if (BasicBlockTarget) {
      auto *BasicBlockAddressTarget = llvm::BlockAddress::get(BasicBlockTarget);
      revng_assert(BasicBlockAddressTarget);
      Builder.CreateCall(MarkerCallF, BasicBlockAddressTarget);
    } else {
      llvm::LLVMContext &C = getContext(BB);
      llvm::PointerType *BlockAddressTy = llvm::Type::getInt8PtrTy(C);
      auto
        *NullBasicBlockAddress = llvm::ConstantPointerNull::get(BlockAddressTy);
      Builder.CreateCall(MarkerCallF, NullBasicBlockAddress);
    }
  }
};

/// Helper method to retrieve the `BasicBlock` target of the marker
inline llvm::BasicBlock *getMarkerCallTarget(std::string MarkerCallName,
                                             llvm::BasicBlock *BB) {

  // We must be provided with a `BasicBlock` where to search for the marker
  revng_assert(BB);

  // When using the getter, we assume that the marker function declaration is
  // present in the `Module`
  llvm::Module *M = getModule(BB);
  llvm::Function *MarkerCallFunction = M->getFunction(MarkerCallName);
  revng_assert(MarkerCallFunction);

  // We assume that the call to the marker function can be in the last but one,
  // or last but two position in the `BasicBlock`, without assuming a particular
  // order between the marker call functions having a different type
  auto BBIt = BB->rbegin();
  ++BBIt;
  for (size_t Index = 0; Index < 2 && BBIt != BB->rend(); ++BBIt) {
    llvm::Instruction &TentativeInst = *BBIt;
    if (llvm::CallInst *MarkerCall = getCallTo(&TentativeInst,
                                               MarkerCallFunction)) {
      auto *MarkerCallTargetBlockAddress = llvm::cast<
        llvm::BlockAddress>(MarkerCall->getArgOperand(0));
      using llvm::BasicBlock;
      auto *MarkerCallTargetBB = MarkerCallTargetBlockAddress->getBasicBlock();
      return MarkerCallTargetBB;
    }
  }

  return nullptr;
}

inline llvm::BasicBlock *getScopeCloser(llvm::BasicBlock *BB) {
  return getMarkerCallTarget(ScopeCloserMarkerCallFunctionName, BB);
}

inline bool isGotoBlock(llvm::BasicBlock *BB) {

  // We must be provided with a `BasicBlock` where to search for the marker
  revng_assert(BB);

  // When using the getter, we assume that the marker function declaration is
  // present in the `Module`
  llvm::Module *M = getModule(BB);
  llvm::Function
    *MarkerCallFunction = getMarkerCallFunction(FunctionTags::GotoBlockMarker,
                                                M);
  revng_assert(MarkerCallFunction);

  // We assume that the call to the marker function can be in the last but one,
  // or last but two position in the `BasicBlock`, without assuming a particular
  // order between the marker call functions having a different type
  auto BBIt = BB->rbegin();
  ++BBIt;
  for (size_t Index = 0; Index < 2 && BBIt != BB->rend(); ++BBIt) {
    llvm::Instruction &TentativeInst = *BBIt;
    if (llvm::CallInst *MarkerCall = getCallTo(&TentativeInst,
                                               MarkerCallFunction)) {
      return true;
    }
  }
  return false;
}

inline void verifyBasicBlock(FunctionTags::Tag &Tag, llvm::BasicBlock *BB) {
  llvm::Module *M = getModule(BB);
  llvm::Function *MarkerCallFunction = getMarkerCallFunction(Tag, M);
  std::string
    MarkerCallFunctionName = getMarkerCallFunctionName(MarkerCallFunction);
  revng_assert(MarkerCallFunction);

  // We should find at maximum one occurrence of the call to the marker
  // function in either the last but one or last but two position in the
  // `BasicBlock`
  bool MarkerFound = false;
  auto BBIt = BB->rbegin();
  ++BBIt;
  for (size_t Index = 0; Index < 2 && BBIt != BB->rend(); ++BBIt) {
    llvm::Instruction &TentativeInst = *BBIt;
    if (llvm::CallInst *MarkerCall = getCallTo(&TentativeInst,
                                               MarkerCallFunction)) {
      revng_assert(MarkerFound == false, "Duplicate Marker Call");
      MarkerFound = true;
    }
  }

  // In the rest of the `BasicBlock`, we should not find any call to the
  // marker function
  for (; BBIt != BB->rend(); ++BBIt) {
    llvm::Instruction &TentativeInst = *BBIt;
    if (llvm::CallInst *MarkerCall = getCallTo(&TentativeInst,
                                               MarkerCallFunction)) {
      revng_abort("No Marker Call expected");
    }
  }

  // Additionally, the `ScopeGraph` has the requirement of permitting a single
  // successor for a `BasicBlock` which contains the `goto_block` marker
  if (MarkerFound and Tag.name() == GotoBlockMarkerCallFunctionName) {
    llvm::Instruction *Terminator = BB->getTerminator();
    revng_assert(Terminator->getNumSuccessors() == 1);
  }
}
