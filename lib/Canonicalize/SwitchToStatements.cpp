//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// FIXME TODO drop model in non legacy
//  * add a metadata to call sites with their model return type, in the same
//    place that emits the metadata used for getCallSitePrototype
//    * this pass forwards this onto the allocas, and clifter can use that
//  * for pointer size use data layout

#include <compare>
#include <functional>
#include <iterator>
#include <map>
#include <memory_resource>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassInfo.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/SmallMap.h"
#include "revng/Canonicalize/SwitchToStatements.h"
#include "revng/InitModelTypes/InitModelTypes.h"
#include "revng/LocalVariables/LocalVariableBuilder.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/DecompilationHelpers.h"

static Logger Log{ "switch-to-statements" };

using namespace llvm;

//
// Templated using for types of local variables, copy and assign instructions.
// These are only necessary because we still have the legacy version.
// We can drop the template when we drop legacy mode, and just use StoreInst for
// AssignType, LoadInst for CopyType, and AllocaInst for LocalVarType.
//

template<bool IsLegacy>
using AssignType = std::conditional_t<IsLegacy, CallInst, StoreInst>;

template<bool IsLegacy>
using CopyType = std::conditional_t<IsLegacy, CallInst, LoadInst>;

template<bool IsLegacy>
using LocalVarType = std::conditional_t<IsLegacy, CallInst, AllocaInst>;

//
// Templated helpers for getting pointer and value operands for copy and assign
// instructions.
// These are only necessary because we still have the legacy version where a
// copy is not just a LoadInst but a call to an opaque function, and an
// assignment is not just a StoreInst but it's also a call to another opaque
// function.
// We can drop the template when we drop legacy mode, and just remove these
// helpers or at lease greatly simplify them.
//

template<bool IsLegacy>
static Use *getStorePointerOperandUse(Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (CallInst *Assign = getCallToTagged(I, FunctionTags::Assign))
      return &Assign->getArgOperandUse(1);
  } else {
    if (auto *Assign = dyn_cast<StoreInst>(I))
      return &Assign->getOperandUse(1);
  }
  return nullptr;
}

template<bool IsLegacy>
static const Use *getStorePointerOperandUse(const Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (const CallInst *Assign = getCallToTagged(I, FunctionTags::Assign))
      return &Assign->getArgOperandUse(1);
  } else {
    if (const auto *Assign = dyn_cast<StoreInst>(I))
      return &Assign->getOperandUse(1);
  }
  return nullptr;
}

template<bool IsLegacy>
static Value *getStorePointerOperand(Instruction *I) {
  Use *U = getStorePointerOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static const Value *getStorePointerOperand(const Instruction *I) {
  const Use *U = getStorePointerOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static bool isStorePointerOperand(const Use &U, const Instruction *I) {
  return getStorePointerOperandUse<IsLegacy>(I) == &U;
}

template<bool IsLegacy>
static Use *getStoreValueOperandUse(Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (CallInst *Assign = getCallToTagged(I, FunctionTags::Assign))
      return &Assign->getArgOperandUse(0);
  } else {
    if (auto *Assign = dyn_cast<StoreInst>(I))
      return &Assign->getOperandUse(0);
  }
  return nullptr;
}

template<bool IsLegacy>
static const Use *getStoreValueOperandUse(const Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (const CallInst *Assign = getCallToTagged(I, FunctionTags::Assign))
      return &Assign->getArgOperandUse(0);
  } else {
    if (const auto *Assign = dyn_cast<StoreInst>(I))
      return &Assign->getOperandUse(0);
  }
  return nullptr;
}

template<bool IsLegacy>
static Value *getStoreValueOperand(Instruction *I) {
  Use *U = getStoreValueOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static const Value *getStoreValueOperand(const Instruction *I) {
  const Use *U = getStoreValueOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static Use *getLoadPointerOperandUse(Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (CallInst *Assign = getCallToTagged(I, FunctionTags::Copy))
      return &Assign->getArgOperandUse(0);
  } else {
    if (auto *Assign = dyn_cast<LoadInst>(I))
      return &Assign->getOperandUse(0);
  }
  return nullptr;
}

template<bool IsLegacy>
static const Use *getLoadPointerOperandUse(const Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (const CallInst *Assign = getCallToTagged(I, FunctionTags::Copy))
      return &Assign->getArgOperandUse(0);
  } else {
    if (const auto *Assign = dyn_cast<LoadInst>(I))
      return &Assign->getOperandUse(0);
  }
  return nullptr;
}

template<bool IsLegacy>
static Value *getLoadPointerOperand(Instruction *I) {
  Use *U = getLoadPointerOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static const Value *getLoadPointerOperand(const Instruction *I) {
  const Use *U = getLoadPointerOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static Value *getPointerOperand(Instruction *I) {
  if (auto *V = getLoadPointerOperand<IsLegacy>(I))
    return V;
  if (auto *V = getStorePointerOperand<IsLegacy>(I))
    return V;
  return nullptr;
}

template<bool IsLegacy>
static const Value *getPointerOperand(const Instruction *I) {
  if (const auto *V = getLoadPointerOperand<IsLegacy>(I))
    return V;
  if (const auto *V = getStorePointerOperand<IsLegacy>(I))
    return V;
  return nullptr;
}

//
// Helpers for statements and side effects.
//

static bool causesExponentialDataflowPaths(const Instruction *I) {
  // This forces all various kinds of instructions to get their value stored
  // into a local variable.
  // The reason for doing this is that these instructions often contribute to
  // the formation of pathological dataflows, causing exponential path explosion
  // when expanded into C expressions during decompilation.
  // Serializing their value into a dedicated local variable breaks such an
  // exponential path explosion.
  //
  // This is a temporary workaround, that we've already put in place for
  // SelectInst too, and that will be replaced in the future by a more
  // principled approach for splitting pathological dataflows that lead to
  // exponential path esplosion.

  if (isa<SelectInst>(I))
    return true;

  llvm::Value *Op0 = nullptr;
  llvm::Value *Op1 = nullptr;
  llvm::Value *Op2 = nullptr;

  using namespace llvm::PatternMatch;
  if (match(I, m_FShl(m_Value(Op0), m_Value(Op1), m_Value(Op2))))
    return true;
  if (match(I, m_FShr(m_Value(Op0), m_Value(Op1), m_Value(Op2))))
    return true;

  return false;
}

static bool isStatement(const Instruction *I) {
  return causesExponentialDataflowPaths(I) or I->mayHaveSideEffects();
}

static bool mayReadMemory(const llvm::Instruction &I) {
  // We have to hardcode revng_call_stack_arguments and revng_stack_frame
  // because SegregateStackAccesses has to mark them as functions that read
  // inaccessible memory, in order to prevent some LLVM optimizations.
  if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
    if (llvm::Function *Callee = getCalledFunction(Call)) {
      llvm::StringRef Name = Callee->getName();
      if (Name.startswith("revng_call_stack_arguments")
          or Name.startswith("revng_stack_frame")) {
        return false;
      }
    }
  }

  return I.mayReadFromMemory();
}

//
// Legacy mode helpers for traversing ModelGEPs and discovering the local
// variable accessed by an Instruction.
// They can be dropped when legacy mode goes away.
//

static RecursiveCoroutine<std::optional<const Value *>>
getAccessedLocalVariableFromModelGEP(const CallInst *ModelGEPRefCall) {
  revng_assert(isCallToTagged(ModelGEPRefCall, FunctionTags::ModelGEPRef));

  revng_assert(ModelGEPRefCall->arg_size() >= 2);

  // If the ModelGEPRefCall has more than 2 arguments, and some of them are not
  // constants, we cannot figure out all the list of potentially accessed local
  // variables, so we just return nullptr.
  for (const Use &GEPArg : llvm::drop_begin(ModelGEPRefCall->args(), 2)) {
    if (not isa<Constant>(GEPArg.get()))
      rc_return nullptr;
  }

  // If the Base argument of the ModelGEPRefCall isn't a LocalVariable, nor an
  // Argument, nor another ModelGEPRef, we just return nullopt, meaning that
  // this thing doesn't really access any local variable.
  auto *GEPBase = ModelGEPRefCall->getArgOperand(1);
  // If the GEPBase is directly an argument, we're done
  if (isa<Argument>(GEPBase))
    rc_return GEPBase;

  // If the GEPBase is directly a LocalVariable, we're done
  if (isCallToTagged(GEPBase, FunctionTags::AllocatesLocalVariable))
    rc_return GEPBase;

  // If the GEPBase is another ModelGEPRef we recur.
  // Notice that we don't recur on ModelGEP, only on ModelGEPRef, because simple
  // ModelGEP can have arbitrary base pointers, but they never access
  // LocalVariables.
  if (auto *NestedModelGEPRef = getCallToTagged(GEPBase,
                                                FunctionTags::ModelGEPRef))
    rc_return rc_recur getAccessedLocalVariableFromModelGEP(NestedModelGEPRef);

  // Everything else cannot access local variables, so we return nullopt.
  rc_return std::nullopt;
}

// Get the local variable accessed by I.
// If the returned optional is nullopt, it means that I doesn't accessy memory.
// If the returned optional is engaged:
// - if it holds a null pointer, it means that I accesses memory but we weren't
//   able to figure out where
// - if it holds a valid pointer, it must be either an Argument or the accessed
//   local variable
static std::optional<const Value *>
getAccessedLegacyLocal(const Instruction *I) {
  const CopyType<true> *Copy = getCallToTagged(I, FunctionTags::Copy);
  const AssignType<true> *Assign = getCallToTagged(I, FunctionTags::Assign);

  // If it's not a Copy not an Assign then it's not an access to a local
  // variable.
  // TODO: this doesn't take into consideration stuff like memcpy.
  if (not Copy and not Assign)
    return std::nullopt;

  const CallInst *AccessCall = Copy ? Copy : Assign;

  unsigned AccessArgumentNumber = Assign ? 1 : 0;
  const auto *Accessed = AccessCall->getArgOperand(AccessArgumentNumber);

  // If the accessed thing is directly an Argument or a LocalVariable we're
  // done.
  if (isa<Argument>(Accessed)
      or isCallToTagged(Accessed, FunctionTags::AllocatesLocalVariable)) {
    return Accessed;
  }

  // If the accessed thing is not a ModelGEPRef, then it's not an access to a
  // local variable.
  auto *ModelGEPRef = getCallToTagged(Accessed, FunctionTags::ModelGEPRef);
  if (not ModelGEPRef)
    return std::nullopt;

  return getAccessedLocalVariableFromModelGEP(ModelGEPRef);
}

/// A class that represents an LLVM analysis that computes a set of AllocaInst
/// in a Function whose address doesn't leak.
class AllocasWhoseAddressDoesntLeak
  : public llvm::AnalysisInfoMixin<AllocasWhoseAddressDoesntLeak> {

  friend llvm::AnalysisInfoMixin<AllocasWhoseAddressDoesntLeak>;
  static llvm::AnalysisKey Key;

public:
  using Result = SmallVector<AllocaInst *>;

  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    Result AllocasThatDontLeak;
    BasicBlock &EntryBlock = F.getEntryBlock();

    for (Instruction &I : EntryBlock) {
      auto *Alloca = dyn_cast<AllocaInst>(&I);
      if (not Alloca)
        break;

      if (not leaksAddress(Alloca))
        AllocasThatDontLeak.push_back(Alloca);
    }

    return AllocasThatDontLeak;
  }

private:
  bool leaksAddress(AllocaInst *A) {
    SmallVector<User *> Worklist;
    Worklist.push_back(A);

    while (not Worklist.empty()) {
      User *Current = Worklist.back();
      Worklist.pop_back();

      for (Use &U : Current->uses()) {
        if (leaksAddress(U))
          return true;

        if (propagatesPointer(U))
          Worklist.push_back(U.getUser());
      }
    }
    return false;
  }

  bool propagatesPointer(Use &U) const {
    User *TheUser = U.getUser();
    if (isa<BitCastInst>(TheUser))
      return true;

    if (auto *GEP = dyn_cast<GetElementPtrInst>(TheUser))
      return GEP->getPointerOperandIndex() == U.getOperandNo();

    if (auto *Select = dyn_cast<SelectInst>(TheUser))
      return Select->getCondition() != U.get();

    return false;
  }

  bool leaksAddress(Use &U) const {
    User *TheUser = U.getUser();

    if (isa<GetElementPtrInst>(TheUser) or isa<SelectInst>(TheUser)
        or isa<BitCastInst>(TheUser) or isa<ReturnInst>(TheUser)
        or isa<LoadInst>(TheUser) or isa<ICmpInst>(TheUser))
      return false;

    if (auto *Store = dyn_cast<StoreInst>(TheUser))
      return U.getOperandNo() != Store->getPointerOperandIndex();

    if (isa<PtrToIntInst>(TheUser))
      return true;

    if (auto *Call = dyn_cast<CallInst>(TheUser))
      return Call->isArgOperand(&U);

    revng_abort();
  }
};

llvm::AnalysisKey AllocasWhoseAddressDoesntLeak::Key = {};

/// A class that represents an available expression, along with the assignment
/// that writes its value somewhere, making it available.
template<bool IsLegacy>
struct AvailableExpression {
  using AssignType = AssignType<IsLegacy>;

  // The expression that is available
  Instruction *Expression = nullptr;

  // The Assign/Store that has assigned the Expression to some location.
  // It can be used to retrieve the address of the location itself.
  // nullptr means that we don't have a specific address but the Expression
  // itself can be computed at the given program point without breaking
  // semantics.
  AssignType *Assignment = nullptr;

  bool operator==(const AvailableExpression &) const = default;
  std::strong_ordering operator<=>(const AvailableExpression &) const = default;
};

template<bool IsLegacy>
using AvailableSet = std::set<AvailableExpression<IsLegacy>>;

template<bool IsLegacy>
static auto
findAvailableRange(const AvailableSet<IsLegacy> &Availables, Instruction *I) {
  using AvailableExpression = AvailableExpression<IsLegacy>;
  auto Begin = Availables.lower_bound(AvailableExpression{
    .Expression = I, .Assignment = nullptr });
  auto End = Availables.upper_bound(AvailableExpression{
    .Expression = std::next(I), .Assignment = nullptr });
  return llvm::make_range(Begin, End);
}

constexpr size_t SmallSize = 8;
using InstructionVector = SmallVector<Instruction *, SmallSize>;
using InstructionSetVector = SmallSetVector<Instruction *, SmallSize>;

struct ProgramPointData {
  Instruction *TheInstruction = nullptr;
  ProgramPointData(Instruction *I) : TheInstruction(I){};
};

using ProgramPointNode = BidirectionalNode<ProgramPointData>;
using ProgramPointsCFG = GenericGraph<ProgramPointNode>;

template<bool IsLegacy>
struct AvailableExpressionsMonotoneFramework;

template<bool IsLegacy>
struct AvailableExpressionsMonotoneFramework {
public:
  using GraphType = ProgramPointsCFG *;
  using LatticeElement = AvailableSet<IsLegacy>;
  using Label = ProgramPointNode *;

private:
  AliasAnalysis *AA;
  AllocasWhoseAddressDoesntLeak::Result *AllocasThatDontLeak;

public:
  AvailableExpressionsMonotoneFramework(AliasAnalysis *A,
                                        AllocasWhoseAddressDoesntLeak::Result
                                          *Allocas) :
    AA(A), AllocasThatDontLeak(Allocas) {}

public:
  LatticeElement combineValues(const LatticeElement &LHS,
                               const LatticeElement &RHS) const {
    return SetIntersectionLattice<LatticeElement>::combineValues(LHS, RHS);
  }

  bool isLessOrEqual(const LatticeElement &LHS,
                     const LatticeElement &RHS) const {
    return SetIntersectionLattice<LatticeElement>::isLessOrEqual(LHS, RHS);
  }

  LatticeElement applyTransferFunction(ProgramPointNode *L,
                                       const LatticeElement &E) const;

private:
  void applyTransferFunctionImpl(Instruction *I, LatticeElement &E) const;

  bool noAlias(const Instruction *I, const Instruction *J) const;
};

template<bool IsLegacy>
using AEMFP = AvailableExpressionsMonotoneFramework<IsLegacy>;

template<bool IsLegacy>
using LatticeElement = AEMFP<IsLegacy>::LatticeElement;

template<bool IsLegacy>
using AvailableExpressionsMap = MFP::MFIResultMap<AEMFP<IsLegacy>>;

static bool legacyLocalVariablesNoAlias(const Instruction *I,
                                        const Instruction *J) {

  // Copies from local variables never alias anyone else, except other
  // instructions that copy or assign the same local variable
  std::optional<const Value *> MayBeAccessedByI = getAccessedLegacyLocal(I);
  std::optional<const Value *> MayBeAccessedByJ = getAccessedLegacyLocal(J);

  // If either doesn't access a local variable, they are noAlias.
  if (not MayBeAccessedByI.has_value() or not MayBeAccessedByJ.has_value())
    return true;

  const Value *AccessedByI = *MayBeAccessedByI;
  const Value *AccessedByJ = *MayBeAccessedByJ;

  // If either is nullptr, there is at least one among I and J that access many
  // variables, and we just can't say with certainty that they are noAlias
  if (nullptr == AccessedByI or nullptr == AccessedByJ)
    return false;

  // If both are arguments, they may point overlapping memory, and we have no
  // way of knowing. So we return false, because we're not sure they don't
  // alias.
  if (isa<Argument>(AccessedByI) and isa<Argument>(AccessedByJ))
    return false;

  // For all the other cases they are noAlias only if the accessed
  // local variable is different.
  return AccessedByI != AccessedByJ;
}

static bool
accessesAllocaThatDoesntLeak(const Instruction *I,
                             const AllocasWhoseAddressDoesntLeak::Result
                               &AllocasThatDontLeak,
                             AliasAnalysis &AA) {
  const Value *PointerOp = getPointerOperand<false>(I);
  if (not PointerOp)
    return false;

  const Value *ValueOp = getStoreValueOperand<false>(I);
  const Type *ValueType = ValueOp ? ValueOp->getType() :
                                    cast<LoadInst>(I)->getType();
  if (not ValueType->isIntegerTy())
    return false;

  revng_assert(ValueType);

  for (const AllocaInst *A : AllocasThatDontLeak) {
    // FIXME we shouldn't be using MUST ALIAS
    if (AA.isMustAlias(A, PointerOp))
      return true;

    //        FIXME more raffinate than just isMustAlias
    //        if (hasStackTypeMetadata(Alloca)) {
    //          Type = importModelType(*getStackTypeFromMetadata(Alloca,
    //          Model));
    //        } else if (hasVariableTypeMetadata(Alloca)) {
    //          Type = importModelType(*getVariableTypeFromMetadata(Alloca,
    //          Model));
    //        } else {
    //          Type = importLLVMType(Alloca->getAllocatedType());
    //
    //          if (Alloca->isArrayAllocation())
    //            Type = ArrayType::get(Context, Type, getConstantInt(Size));
    //        }
  }

  return false;
}

template<bool IsLegacy>
bool AEMFP<IsLegacy>::noAlias(const Instruction *I,
                              const Instruction *J) const {
  revng_log(Log, "noAlias?");
  LoggerIndent X{ Log };
  revng_log(Log, "I: " << dumpToString(I));
  revng_log(Log, "J: " << dumpToString(J));
  LoggerIndent XX{ Log };

  // If either instruction doesn't access memory, they are noAlias for sure.
  if (not I->mayReadOrWriteMemory()) {
    revng_log(Log, "I->mayReadOrWriteMemory() == false");
    return true;
  }
  if (not J->mayReadOrWriteMemory()) {
    revng_log(Log, "J->mayReadOrWriteMemory() == false");
    return true;
  }

  // Here both instructions access memory.

  if constexpr (IsLegacy) {
    // First, handle LocalVariables specifically.
    // TODO: this is a poor's man alias analysis, which only explicitly handles
    // stuff that is frequent and that we care about. In the future we have
    // plans to replace it with a full fledged AliasAnalysis from LLVM
    if (legacyLocalVariablesNoAlias(I, J)) {
      revng_log(Log, "I and J both access local variables that do not alias");
      return true;
    }
  } else {

    // If I is not a Load/Store, we don't know how it accesses memory, so we
    // can't prove is noalias with J, unless J is a Load/Store accessing an
    // alloca whose address doesn't leak.
    const Value *IPointerOperand = getPointerOperand<false>(I);
    if (not IPointerOperand) {
      revng_log(Log, "I accesses memory but is not a Load/Store");
      if (accessesAllocaThatDoesntLeak(J, *AllocasThatDontLeak, *AA)) {
        revng_log(Log, "accessesAllocaThatDoesntLeak(J)");
        return true;
      }
      revng_log(Log, "not accessesAllocaThatDoesntLeak(J)");
      return false;
    }
    revng_log(Log, "I pointer operand: " << dumpToString(IPointerOperand));

    // Likewise, if J is not a Load/Store, we don't know how it accesses memory,
    // so we can't prove is noalias with I, unless I is a Load/Store accessing
    // an alloca whose address doesn't leak.
    const Value *JPointerOperand = getPointerOperand<false>(J);
    if (not JPointerOperand) {
      revng_log(Log, "J accesses memory but is not a Load/Store");
      if (accessesAllocaThatDoesntLeak(I, *AllocasThatDontLeak, *AA)) {
        revng_log(Log, "accessesAllocaThatDoesntLeak(I)");
        return true;
      }
      revng_log(Log, "not accessesAllocaThatDoesntLeak(I)");
      return false;
    }
    revng_log(Log, "J pointer operand: " << dumpToString(JPointerOperand));

    if (AA->isNoAlias(IPointerOperand, JPointerOperand)) {
      revng_log(Log, "AA->isNoAlias(IPointerOperand, JPointerOperand) == true");
      return true;
    }
    revng_log(Log, "AA->isNoAlias(IPointerOperand, JPointerOperand) == false");
  }

  // In all the other cases we always fall back to false, meaning that we can't
  // say for sure that I and J do not alias.
  revng_log(Log, "I and J aren't provably noAlias");
  return false;
}

template<bool IsLegacy>
void AEMFP<IsLegacy>::applyTransferFunctionImpl(Instruction *I,
                                                LatticeElement &E) const {
  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AssignType = AssignType<IsLegacy>;

  revng_log(Log, "applyTransferFunction on Instruction: " << dumpToString(I));
  LoggerIndent X{ Log };

  if constexpr (IsLegacy) {
    revng_assert(not isa<LoadInst>(I) and not isa<StoreInst>(I));
  } else {
    revng_assert(not isCallToTagged(I, FunctionTags::Copy)
                 and not isCallToTagged(I, FunctionTags::Assign));
  }

  if (isStatement(I)) {
    revng_log(Log, "isStatement");
    LoggerIndent XX{ Log };
    for (const AvailableExpression &A : llvm::make_early_inc_range(E)) {
      const auto &[Available, Assign] = A;
      revng_log(Log, "Available: " << dumpToString(Available));
      revng_log(Log, "Assign: " << dumpToString(Assign));
      LoggerIndent XXX{ Log };
      if (not noAlias(I, Available)) {
        revng_log(Log, "Available: " << dumpToString(Available));
        revng_log(Log, "is not noAlias (MayAlias) with I");
        revng_log(Log, "erase Available");
        E.erase(A);
      } else if (Assign and not noAlias(I, Assign)) {
        revng_log(Log, "Assign: " << dumpToString(Assign));
        revng_log(Log, "erase Available");
        E.erase(A);
      } else {
        revng_log(Log, "is noAlias with I");
      }
    }
  }

  auto *StoredOperand = getStoreValueOperand<IsLegacy>(I);
  auto *AssignedInstruction = dyn_cast_or_null<Instruction>(StoredOperand);
  if (AssignedInstruction) {
    revng_log(Log, "I is Assign");
    revng_log(Log, "insert Available: " << dumpToString(AssignedInstruction));
    revng_log(Log, "       Assign: " << dumpToString(I));

    E.insert(AvailableExpression{
      .Expression = AssignedInstruction,
      .Assignment = cast<AssignType>(I),
    });
  }

  if (mayReadMemory(*I)) {
    revng_log(Log, "mayReadMemory -> insert Available: I");
    E.insert(AvailableExpression{
      .Expression = I,
      .Assignment = nullptr,
    });
  }
}

template<bool IsLegacy>
AEMFP<IsLegacy>::LatticeElement
AEMFP<IsLegacy>::applyTransferFunction(ProgramPointNode *ProgramPoint,
                                       const AEMFP<IsLegacy>::LatticeElement &E)
  const {

  Instruction *I = ProgramPoint->TheInstruction;

  revng_log(Log, "applyTransferFunction on ProgramPoint: " << dumpToString(I));
  LoggerIndent Indent{ Log };

  LatticeElement Result = E;

  revng_log(Log, "initial set");
  if (Log.isEnabled()) {
    LoggerIndent X{ Log };
    for (const auto &[Available, Assign] : Result) {
      revng_log(Log, "Available: " << dumpToString(Available));
      revng_log(Log, "Assign: " << dumpToString(Assign));
    }
  }

  applyTransferFunctionImpl(I, Result);

  revng_log(Log, "final set");
  if (Log.isEnabled()) {
    LoggerIndent X{ Log };
    for (const auto &[Available, Assign] : Result) {
      revng_log(Log, "Available: " << dumpToString(Available));
      revng_log(Log, "Assign: " << dumpToString(Assign));
    }
  }

  return Result;
}

//
// Helpers for identifying program points that are relevant for available
// expressions.
//

template<bool IsLegacy>
static bool isProgramPoint(const Instruction *I) {

  const Instruction *UnexpectedInstruction = nullptr;
  if constexpr (IsLegacy) {
    // Legacy mode just assumes that we don't have Load/Store/Alloca at all.
    if (isa<LoadInst>(I) or isa<StoreInst>(I) or isa<AllocaInst>(I)
        or isa<PHINode>(I))
      UnexpectedInstruction = I;
  } else {
    // Non-legacy mode assumes that most custom opcode don't exist. Some of them
    // have been replaced by Load/Store/Alloca, and others have been dropped
    // because in the clift-based pipeline they will be only materialized in
    // Clift as regular operators, so we don't need them in LLVM anymore and we
    // want to make sure they disappear over time until we can actually drop
    // them.
    if (isCallToTagged(I, FunctionTags::AllocatesLocalVariable)
        or isCallToTagged(I, FunctionTags::LocalVariable)
        or isCallToTagged(I, FunctionTags::Copy)
        or isCallToTagged(I, FunctionTags::Assign)
        or isCallToTagged(I, FunctionTags::AddressOf)
        or isCallToTagged(I, FunctionTags::Marker)
        or isCallToTagged(I, FunctionTags::IsRef)
        or isCallToTagged(I, FunctionTags::StringLiteral)
        or isCallToTagged(I, FunctionTags::ModelCast)
        or isCallToTagged(I, FunctionTags::ModelGEP)
        or isCallToTagged(I, FunctionTags::ModelGEPRef)
        or isCallToTagged(I, FunctionTags::Parentheses)
        or isCallToTagged(I, FunctionTags::LiteralPrintDecorator)
        or isCallToTagged(I, FunctionTags::HexInteger)
        or isCallToTagged(I, FunctionTags::CharInteger)
        or isCallToTagged(I, FunctionTags::BoolInteger)
        or isCallToTagged(I, FunctionTags::NullPtr)
        or isCallToTagged(I, FunctionTags::SegmentRef)
        or isCallToTagged(I, FunctionTags::UnaryMinus)
        or isCallToTagged(I, FunctionTags::BinaryNot)
        or isCallToTagged(I, FunctionTags::BooleanNot)) {
      UnexpectedInstruction = I;
    }
    // We also don't expect PHINodes, since SwitchToStatements is designed to be
    // the last pass in the decompilation pipeline that injects local variables.
    if (isa<PHINode>(I))
      UnexpectedInstruction = I;
  }

  if (nullptr != UnexpectedInstruction) {
    I->dump();
    revng_abort("Unexpected Instruction");
  }

  return I == &I->getParent()->front() or isStatement(I) or mayReadMemory(*I);
}

template<bool IsLegacy>
static InstructionSetVector getProgramPoints(BasicBlock &B) {
  InstructionSetVector Results;
  for (Instruction &I : B)
    if (isProgramPoint<IsLegacy>(&I))
      Results.insert(&I);
  return Results;
}

using InstructionProgramPoint = std::unordered_map<const Instruction *,
                                                   ProgramPointNode *>;

// An extended version of ProgramPointsCFG, that holds a graph of statements
// points, along with a map from each Instruction to its previous statement.
template<bool IsLegacy>
class AvailableExpressionsResult {
public:
  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AvailableSet = AvailableSet<IsLegacy>;
  using AvailableExpressionsMap = AvailableExpressionsMap<IsLegacy>;

public:
  ProgramPointsCFG ProgramPointsGraph;
  AvailableExpressionsMap AvailableExpressions;

private:
  // Map an Instruction to its associated program point in ProgramPointsGraph
  InstructionProgramPoint ProgramPoint;

  // Map an Instruction to its associated previous program point in
  // ProgramPointsGraph.
  InstructionProgramPoint PreviousProgramPointInBlock;

  // Map an Instruction to its associated next program point in
  // ProgramPointsGraph.
  InstructionProgramPoint NextProgramPointInBlock;

public:
  // Factory from llvm::Function
  static AvailableExpressionsResult makeFromFunction(Function &F) {

    SmallMap<BasicBlock *, std::pair<ProgramPointNode *, ProgramPointNode *>, 8>
      BlockToBeginEndNode;

    AvailableExpressionsResult Result;

    ProgramPointsCFG &TheCFG = Result.ProgramPointsGraph;
    InstructionProgramPoint &ProgramPoint = Result.ProgramPoint;
    InstructionProgramPoint
      &PreviousProgramPointInBlock = Result.PreviousProgramPointInBlock;
    InstructionProgramPoint
      &NextProgramPointInBlock = Result.NextProgramPointInBlock;

    const auto MakeCFGNode = [&TheCFG, &ProgramPoint](Instruction *I) {
      ProgramPointNode *NewNode = TheCFG.addNode(I);
      ProgramPoint[I] = NewNode;
      return NewNode;
    };

    for (BasicBlock &BB : F) {
      InstructionSetVector ProgramPoints = getProgramPoints<IsLegacy>(BB);

      // Reserve space for the new ProgramPoints. This is for performance but
      // also for stability of pointers while adding new nodes, which allows to
      // also save pointers to begin and end nodes of each block in a map, to
      // handle addition of inter-block edges. If we don't reserve the pointers
      // returned by addNode aren't stable and the trick for adding inter-block
      // edges doesn't work.
      TheCFG.reserve(TheCFG.size() + ProgramPoints.size());

      ProgramPointNode *FirstNode = MakeCFGNode(ProgramPoints.front());
      ProgramPointNode *LastNode = FirstNode;
      for (Instruction &I :
           llvm::make_range(BB.begin(), ProgramPoints.front()->getIterator()))
        NextProgramPointInBlock[&I] = LastNode;

      auto ProgramPointPairs = llvm::zip_equal(llvm::drop_end(ProgramPoints),
                                               llvm::drop_begin(ProgramPoints));
      for (const auto &[PreviousProgramPoint, NextProgramPoint] :
           ProgramPointPairs) {
        // Create a new node.
        ProgramPointNode *NewNode = MakeCFGNode(NextProgramPoint);
        // We can already add intra-block edges.
        LastNode->addSuccessor(NewNode);

        // Now we have to initialize PreviousProgramPointInBlock for all the
        // instructions that are not program points and that are among the
        // previous program point and the current new one.
        for (Instruction &I :
             llvm::make_range(std::next(PreviousProgramPoint->getIterator()),
                              NextProgramPoint->getIterator()))
          PreviousProgramPointInBlock[&I] = LastNode;

        // Finally we can update the LastNode.
        LastNode = NewNode;
      }
      for (Instruction &I :
           llvm::make_range(std::next(ProgramPoints.back()->getIterator()),
                            BB.end()))
        PreviousProgramPointInBlock[&I] = LastNode;

      BlockToBeginEndNode[&BB] = { FirstNode, LastNode };
    }

    // Now we add the inter-block edges.
    for (BasicBlock &BB : F)
      for (BasicBlock *Successor : llvm::successors(&BB))
        BlockToBeginEndNode.at(&BB)
          .second->addSuccessor(BlockToBeginEndNode.at(Successor).first);

    // And set the entry node, which makes the MFP later more efficient, because
    // it allows the algorithm to take the structure of the graph into account,
    // instead of iterating in sparse order.
    TheCFG.setEntryNode(BlockToBeginEndNode.at(&F.getEntryBlock()).first);

    return Result;
  }

public:
  auto getAvailableAt(Instruction *I, const Instruction *Where) const {

    revng_log(Log, "IsAvailableAt");
    revng_log(Log, "I: " << dumpToString(I));
    revng_log(Log, "Where: " << dumpToString(Where));

    auto ProgramPointIt = ProgramPoint.find(Where);
    if (ProgramPointIt != ProgramPoint.end()) {
      revng_log(Log, "is ProgramPoint");

      ProgramPointNode *UserProgramPoint = ProgramPointIt->second;
      const AvailableSet &Available = AvailableExpressions.at(UserProgramPoint)
                                        .InValue;
      return findAvailableRange(Available, I);
    }

    auto PreviousPointIt = PreviousProgramPointInBlock.find(Where);
    if (PreviousPointIt != PreviousProgramPointInBlock.end()) {
      revng_log(Log, "is NOT ProgramPoint");

      ProgramPointNode *UserProgramPoint = PreviousPointIt->second;
      revng_log(Log,
                "Previous ProgramPoint: "
                  << dumpToString(UserProgramPoint->TheInstruction));
      const AvailableSet &Available = AvailableExpressions.at(UserProgramPoint)
                                        .OutValue;
      return findAvailableRange(Available, I);
    }

    auto NextProgramPointIt = NextProgramPointInBlock.find(Where);
    if (NextProgramPointIt != NextProgramPointInBlock.end()) {
      revng_log(Log, "is before first ProgramPoint in BasicBlock");

      ProgramPointNode *UserProgramPoint = NextProgramPointIt->second;
      revng_log(Log,
                "first ProgramPoint in BasicBlock: "
                  << dumpToString(UserProgramPoint->TheInstruction));
      const AvailableSet &Available = AvailableExpressions.at(UserProgramPoint)
                                        .InValue;
      return findAvailableRange(Available, I);
    }

    revng_abort();
  }

  auto getAvailableAt(Instruction *I, const Use &U) const {
    const auto *UserInstruction = cast<Instruction>(U.getUser());
    return getAvailableAt(I, UserInstruction);
  }

  bool isAvailableAt(Instruction *I, const Instruction *Where) const {
    bool Result = not getAvailableAt(I, Where).empty();
    revng_log(Log, "Result: " << Result);
    return Result;
  }

  bool isAvailableAt(Instruction *I, const Use &U) const {
    const auto *UserInstruction = cast<Instruction>(U.getUser());
    return isAvailableAt(I, UserInstruction);
  }
};

template<bool IsLegacy>
using AEResult = AvailableExpressionsResult<IsLegacy>;

template<bool IsLegacy>
static AEResult<IsLegacy>
getAvailableExpressions(Function &F,
                        AliasAnalysis *AA,
                        AllocasWhoseAddressDoesntLeak::Result
                          *AllocasThatDontLeak) {
  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AvailableSet = AvailableSet<IsLegacy>;
  using AssignType = AssignType<IsLegacy>;

  auto Result = AEResult<IsLegacy>::makeFromFunction(F);

  AvailableSet Bottom;
  for (ProgramPointNode *N : llvm::nodes(&Result.ProgramPointsGraph)) {
    Instruction *I = N->TheInstruction;

    if (mayReadMemory(*I)) {
      Bottom.insert(AvailableExpression{
        .Expression = I,
        .Assignment = nullptr,
      });
    }

    auto *StoredOperand = getStoreValueOperand<IsLegacy>(I);
    auto *AssignedInstruction = dyn_cast_or_null<Instruction>(StoredOperand);
    if (AssignedInstruction) {
      Bottom.insert(AvailableExpression{
        .Expression = AssignedInstruction,
        .Assignment = cast<AssignType>(I),
      });
    }
  }

  AvailableSet Empty{};
  ProgramPointsCFG *Graph = &Result.ProgramPointsGraph;
  ProgramPointNode *Entry = Graph->getEntryNode();

  AEMFP<IsLegacy> AvailableExpressionsMF{ AA, AllocasThatDontLeak };
  // std::exchange here is only needed to make revng check-conventions happy.
  std::exchange(Result.AvailableExpressions,
                MFP::getMaximalFixedPoint<>(AvailableExpressionsMF,
                                            Graph,
                                            Bottom,
                                            Empty,
                                            { Entry }));
  return Result;
}

template<bool IsLegacy>
struct PickedInstructions {
  SetVector<Instruction *> ToSerialize = {};
  MapVector<Use *, AssignType<IsLegacy> *> ToReplaceWithAvailable = {};
  SmallPtrSet<AssignType<IsLegacy> *, 8> AssignToRemove = {};
};

template<bool IsLegacy>
class AvailableExpressionsAnalysis
  : public llvm::AnalysisInfoMixin<AvailableExpressionsAnalysis<IsLegacy>> {

  friend llvm::AnalysisInfoMixin<AvailableExpressionsAnalysis<IsLegacy>>;
  static llvm::AnalysisKey Key;

public:
  using Result = AvailableExpressionsResult<IsLegacy>;
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    AliasAnalysis *AA = IsLegacy ? nullptr : &FAM.getResult<AAManager>(F);
    AllocasWhoseAddressDoesntLeak::Result
      *AllocasThatDontLeak = IsLegacy ?
                               nullptr :
                               &FAM.getResult<AllocasWhoseAddressDoesntLeak>(F);
    return getAvailableExpressions<IsLegacy>(F, AA, AllocasThatDontLeak);
  }
};

template<>
AnalysisKey AvailableExpressionsAnalysis<true>::Key = {};
template<>
AnalysisKey AvailableExpressionsAnalysis<false>::Key = {};

template<bool IsLegacy>
using AEA = AvailableExpressionsAnalysis<IsLegacy>;

template<bool IsLegacy>
class InstructionToSerializePicker
  : public AnalysisInfoMixin<InstructionToSerializePicker<IsLegacy>> {
  friend llvm::AnalysisInfoMixin<InstructionToSerializePicker<IsLegacy>>;
  static llvm::AnalysisKey Key;

public:
  using Result = PickedInstructions<IsLegacy>;
  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AssignType = AssignType<IsLegacy>;

private:
  const AEResult<IsLegacy> *AvailableExpressions = nullptr;
  std::unordered_map<const Instruction *, size_t> ProgramOrdering = {};
  Result Picked;
  AliasAnalysis *AA;
  AllocasWhoseAddressDoesntLeak::Result *AllocasThatDontLeak;

public:
  InstructionToSerializePicker() :
    AvailableExpressions(nullptr), ProgramOrdering() {}

public:
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    if constexpr (not IsLegacy) {
      AA = &FAM.getResult<AAManager>(F);
      AllocasThatDontLeak = &FAM.getResult<AllocasWhoseAddressDoesntLeak>(F);
    }
    AvailableExpressions = &FAM.getResult<AEA<IsLegacy>>(F);

    Picked = {};
    ProgramOrdering = {};

    return pick(F);
  }

private:
  bool isSerializable(const Instruction &I) const {
    const llvm::Type *T = I.getType();
    return not T->isVoidTy() and not T->isAggregateType()
           and not isCallToTagged(&I, FunctionTags::IsRef);
  }

  bool serialize(llvm::Instruction *I) {
    LoggerIndent Indent{ Log };
    if (isSerializable(*I)) {
      revng_log(Log, "serialize(I), I: " << dumpToString(I));
      Picked.ToSerialize.insert(I);
      return false;
    }
    revng_log(Log, "serialize(I), can't serialize I: " << dumpToString(I));
    return true;
  };

  bool isPickedToSerialize(llvm::Instruction *I) const {
    return Picked.ToSerialize.contains(I);
  }

  Result pick(Function &F) {
    revng_log(Log, "pick: " << F.getName().str());
    LoggerIndent Indent{ Log };

    // Visit in RPO for determinism
    const auto RPO = llvm::ReversePostOrderTraversal(&F);

    // First, pick all the statements amenable for serialization
    // Also compute the program order of instructions.
    {
      revng_log(Log, "pick statements: " << F.getName().str());
      LoggerIndent MoreIndent{ Log };
      size_t NextOrder = 0;
      for (BasicBlock *BB : RPO) {
        for (Instruction &I : *BB) {
          if (isStatement(&I) and isSerializable(I) and I.getNumUses() != 0)
            serialize(&I);
          ProgramOrdering[&I] = NextOrder++;
        }
      }
    }

    // Then, start from memory reads, and traverse the dataflow to pick other
    // instructions that need to be serialized.
    for (BasicBlock *BB : RPO)
      for (Instruction &I : *BB)
        if (mayReadMemory(I))
          pickFrom(&I, &I);

    return Picked;
  }

  RecursiveCoroutine<bool>
  shouldSerializeReadBeforeOrAtI(Instruction *I, Instruction *MemoryRead) {
    revng_log(Log, "PickFrom I: " << dumpToString(I));
    revng_log(Log, "MemoryRead: " << dumpToString(MemoryRead));

    LoggerIndent Indent{ Log };

    // If I has already been picked for serialization it means that I shouldn't
    // be serialied for it.
    if (isPickedToSerialize(I)) {
      revng_log(Log, "isPickedToSerialize(I)");
      rc_return false;
    }

    // If I has no uses, we are done, and there's no reason to require the
    // serialization of MemoryRead before I.
    if (not I->getNumUses()) {
      revng_log(Log, "I has no uses");
      rc_return false;
    }

    // If it's a statement we must have already picked it. Just return false.
    if (isStatement(I)) {
      revng_log(Log, "I isStatement");
      revng_assert(not isSerializable(*I) or isPickedToSerialize(I));
      rc_return false;
    }

    // If it exists a use U of I for which MemoryRead is not available, then
    // MemoryRead should be serialized before or at I, unless the whole
    // expression represented by I is available somewhere else.
    revng_log(Log, "Check users");
    LoggerIndent UserIndent{ Log };

    MapVector<Use *, AssignType *> ToReplaceWithAvailable;
    SmallPtrSet<AssignType *, 8> AssignToRemove;

    // For each U Use of I where MemoryRead is not available, check if the
    // whole expression represented by I is available at U. If so add it to
    // the ToReplaceWithAvailable.
    // Otherwise if we find even a single use of I where MemoryRead is not
    // available and such that I itself is not available, we have to require I
    // to be either be available somewhere else or be serialized in a new local
    // variable.
    for (Use &U : I->uses()) {
      auto *User = cast<Instruction>(U.getUser());
      revng_log(Log, "User: " << dumpToString(User));
      LoggerIndent MoreUserIndent{ Log };

      if (AvailableExpressions->isAvailableAt(MemoryRead, U)) {
        revng_log(Log, "MemoryRead isAvailableAt(User)");
        continue;
      }
      revng_log(Log, "not MemoryRead isAvailableAt(User)");

      // Skip over the Assign operand representing variables that are being
      // assigned, because we need to preserve them.
      if (isStorePointerOperand<IsLegacy>(U, User)) {
        revng_log(Log, "I isAssignedOperand(User)");
        continue;
      }
      revng_log(Log, "not I isAssignedOperand(User)");

      auto Available = AvailableExpressions->getAvailableAt(I, U);
      if (Available.empty()) {
        revng_log(Log, "I is not available at User");
        rc_return serialize(I);
      }
      revng_log(Log, "I is available at User");

      if (auto It = Picked.ToReplaceWithAvailable.find(&U);
          It != Picked.ToReplaceWithAvailable.end()) {
        revng_log(Log,
                  "I is available at User, reading from address: "
                    << dumpToString(It->second));
        continue;
      }
      revng_log(Log, "Find where I is available");

      SmallVector<AssignType *> AssignsWhereIIsAvailable;
      llvm::copy(Available
                   | std::views::transform([](const AvailableExpression &A) {
                       return A.Assignment;
                     })
                   | std::views::filter([](const AssignType *A) {
                       return A != nullptr;
                     }),
                 std::back_inserter(AssignsWhereIIsAvailable));
      llvm::sort(AssignsWhereIIsAvailable,
                 [&PO = ProgramOrdering](const AssignType *LHS,
                                         const AssignType *RHS) {
                   return PO.at(LHS) < PO.at(RHS);
                 });

      AssignType *SelectedAssign = nullptr;

      bool UserMayWriteMemory = User->mayWriteToMemory();
      if (not AssignsWhereIIsAvailable.empty()) {
        if (not UserMayWriteMemory) {
          // If the User of I cannot write to memory, we're fine.
          // We can pick the CandidateAssign as SelectedAssign, and then we'll
          // later replace U with a read from SelectedAssign.
          revng_log(Log, "not User->mayWriteToMemory()");
          SelectedAssign = AssignsWhereIIsAvailable.front();
        } else {
          revng_log(Log, "User->mayWriteToMemory()");
          LoggerIndent IndentCandidate{ Log };

          for (AssignType *CandidateAssign : AssignsWhereIIsAvailable) {
            revng_log(Log,
                      "CandidateAssign: " << dumpToString(CandidateAssign));
            LoggerIndent MoreIndentCandidate{ Log };

            if constexpr (IsLegacy) {
              if (legacyLocalVariablesNoAlias(User, CandidateAssign)) {
                revng_log(Log,
                          "legacyLocalVariablesNoAlias(User, CandidateAssign)");
                SelectedAssign = CandidateAssign;
                break;
              }
              revng_log(Log,
                        "no legacyLocalVariablesNoAlias(User, "
                        "CandidateAssign)");
            } else {
              // If CandidateAssign accesses an alloca that doesn't leak, we can
              // always select CandidateAssign as SelectedAssign
              if (accessesAllocaThatDoesntLeak(CandidateAssign,
                                               *AllocasThatDontLeak,
                                               *AA)) {
                revng_log(Log, "accessesAllocaThatDoesntLeak(CandidateAssign)");
                SelectedAssign = CandidateAssign;
                break;
              }

              // Here User may write to memory. But it's not guaranteed that
              // it's a StoreInst. It could be a call, taking a pointer
              // argument, that writes to the pointee.
              Value *AssignUserPointer = getStorePointerOperand<false>(User);

              // If User is not a StoreInst, it's a call that writes memory.
              // We cannot know what is the pointer operand, so we will not be
              // able to reason about aliasing with CandidateAssign. We have to
              // assume the worse, that they may alias, so CandidateAssign
              // cannot be selected, and we try the next candidate.
              // Unless CandidateAssign assigns an alloca that doesn't leak, but
              // that's already been ruled out above.
              if (not AssignUserPointer) {
                revng_log(Log,
                          "User is not an StoreInst, go to next "
                          "CandidateAssign");
                continue;
              }
              revng_log(Log,
                        "AssignUserPointer: "
                          << dumpToString(AssignUserPointer));

              Value
                *CandidatePointer = getPointerOperand<false>(CandidateAssign);
              revng_assert(CandidatePointer);
              revng_log(Log,
                        "CandidatePointer: " << dumpToString(CandidatePointer));

              // FIXME do we really need to use AA here?? probably not
              if (AA->isNoAlias(CandidatePointer, AssignUserPointer)) {
                revng_log(Log,
                          "AA->isNoAlias(CandidatePointer, AssignUserPointer)");
                SelectedAssign = CandidateAssign;
                break;
              }
              revng_log(Log,
                        "no AA->isNoAlias(CandidatePointer, "
                        "AssignUserPointer)");
            }
          }
        }
      }

      if (not SelectedAssign) {
        revng_log(Log, "no SelectedAssign. Serialize I due to User");
        rc_return serialize(I);
      }

      revng_assert(SelectedAssign->mayWriteToMemory());

      revng_log(Log, "SelectedAssign: " << dumpToString(SelectedAssign));

      // If the user is not writing to memory, it cannot clobber SelectedAssign.
      // Just pick mark U to replace with SelectedAssign.
      if (not UserMayWriteMemory) {
        ToReplaceWithAvailable[&U] = SelectedAssign;
        continue;
      }

      revng_log(Log,
                "UserMayWriteMemory and SelectedAssign->mayWriteToMemory()");
      if constexpr (IsLegacy) {

        if (auto *UserAssign = getCallToTagged(User, FunctionTags::Assign);
            UserAssign and legacyLocalVariablesNoAlias(User, SelectedAssign)) {
          AssignToRemove.insert(UserAssign);
        } else {
          ToReplaceWithAvailable[&U] = SelectedAssign;
        }

      } else {

        auto *UserAssign = dyn_cast<StoreInst>(User);
        // If User is not an assignment, it must be a call that clobbers some
        // memory, and we don't know what it clobbers.
        // In any case, it should use SelecteAssign, in U.
        if (not UserAssign) {
          ToReplaceWithAvailable[&U] = SelectedAssign;
          continue;
        }

        // Here we're trying to detect if User is a StoreInst that stores to the
        // same local variable as SelectedAssign.
        // If that happens we can erase User, because it would be redundant,
        // being a self-assignment.
        // In order to figure this out we use AA, looking for pointers that must
        // alias, and that the two StoreInst are storing same-sized integer
        // values.

        Value *AssignUserValue = getStoreValueOperand<false>(User);
        Type *UserValueType = AssignUserValue->getType();
        if (not UserValueType->isIntegerTy()) {
          ToReplaceWithAvailable[&U] = SelectedAssign;
          continue;
        }

        Value *SelectValue = getStoreValueOperand<false>(SelectedAssign);
        Type *SelectValueType = SelectValue->getType();
        if (not SelectValueType->isIntegerTy()) {
          ToReplaceWithAvailable[&U] = SelectedAssign;
          continue;
        }

        unsigned SelectedBitWidth = SelectValueType->getIntegerBitWidth();
        unsigned UserBitWidth = UserValueType->getIntegerBitWidth();
        if (SelectedBitWidth != UserBitWidth) {
          ToReplaceWithAvailable[&U] = SelectedAssign;
          continue;
        }

        Value *SelectPointer = getStorePointerOperand<false>(SelectedAssign);
        Value *AssignUserPointer = getStorePointerOperand<false>(User);

        if (AA->isMustAlias(AssignUserPointer, SelectPointer))
          AssignToRemove.insert(UserAssign);
        else
          ToReplaceWithAvailable[&U] = SelectedAssign;
      }
    }

    // If we reach this point, it means that no user forced us to serialize I.
    // At this point we can commit ToReplaceWithAvailable into
    // Picked.ToReplaceWithAvailable.
    for (const auto &Element : ToReplaceWithAvailable)
      Picked.ToReplaceWithAvailable.insert(Element);

    // And we can also commit the fact that we want to remove the Assign.
    for (const auto &Assign : AssignToRemove)
      Picked.AssignToRemove.insert(Assign);

    // If we reach this point I is has not been picked for serialization yet, it
    // isn't a statement, and MemoryRead is available to all users of I either
    // directly or through some other local variable where the whole I is
    // available.
    // We have to recur in DFS fashion only on those uses for which we're using
    // MemoryRead directly (not through another local variable where I is
    // available).
    SmallSet<Instruction *, 8> UsersThatRequireMemoryReadSerialized;
    revng_log(Log, "recur on users that aren't available");
    for (Use &TheUse : I->uses()) {

      if (auto It = Picked.ToReplaceWithAvailable.find(&TheUse);
          It != Picked.ToReplaceWithAvailable.end()) {
        revng_log(Log,
                  "TheUse is available: " << dumpToString(TheUse.getUser()));
        continue;
      }

      User *TheUser = TheUse.getUser();

      auto *UserInstruction = cast<Instruction>(TheUser);
      if (rc_recur shouldSerializeReadBeforeOrAtI(UserInstruction, MemoryRead))
        UsersThatRequireMemoryReadSerialized.insert(UserInstruction);
    }

    size_t NumUsersRequiringSerialization = UsersThatRequireMemoryReadSerialized
                                              .size();

    // If no users require MemoryRead to be serialized before them, there's
    // nothing to do, and I doesn't require MemoryRead to be serialized either.
    if (NumUsersRequiringSerialization == 0) {
      revng_log(Log, "No User requires MemoryRead to be serialized");
      rc_return false;
    }

    if (I == MemoryRead) {
      revng_log(Log, "I == MemoryRead");
      revng_assert(isSerializable(*I));
      rc_return serialize(I);
    }

    // If some users of I require MemoryRead to be serialized before them,
    // just serialize I.
    revng_log(Log, "Some of I's users require MemoryRead to be serialized");
    rc_return serialize(I);
  }

  void pickFrom(Instruction *I, Instruction *MemoryRead) {
    shouldSerializeReadBeforeOrAtI(I, MemoryRead);
  }
};

template<>
AnalysisKey InstructionToSerializePicker<true>::Key = {};
template<>
AnalysisKey InstructionToSerializePicker<false>::Key = {};

using TypeMap = std::map<const Value *, const model::UpcastableType>;

template<bool IsLegacy>
using LVB = LocalVariableBuilder<IsLegacy>;

template<bool IsLegacy>
class VariableInserter {
public:
  using PickedInstructions = PickedInstructions<IsLegacy>;

private:
  const model::Binary &Model;
  const TypeMap TheTypeMap;
  Function &F;
  LocalVariableBuilder<IsLegacy> LocalVariableBuilder;

public:
  VariableInserter(Function &TheF,
                   const model::Binary &TheModel,
                   TypeMap &&TMap) :
    Model(TheModel),
    TheTypeMap(std::move(TMap)),
    F(TheF),
    LocalVariableBuilder(LVB<IsLegacy>::make(TheModel, TheF)) {}

public:
  bool run(const PickedInstructions &Picked) {
    LocalVariableBuilder.setTargetFunction(&F);

    bool Changed = false;

    for (const auto &[TheUse, TheAssign] : Picked.ToReplaceWithAvailable) {
      auto *Copy = LocalVariableBuilder.createCopyFromAssignedOnUse(TheAssign,
                                                                    *TheUse);
      TheUse->set(Copy);
    }

    for (Instruction *I : Picked.AssignToRemove) {
      Changed = true;
      I->eraseFromParent();
    }

    for (Instruction *I : Picked.ToSerialize)
      Changed |= serializeToLocalVariable(I);

    return Changed;
  }

private:
  bool serializeToLocalVariable(Instruction *I);

  bool shouldReplaceUseWithCopies(const Instruction *I, const Use &U) const;

  model::UpcastableType getModelType(const Instruction *I) const {
    if constexpr (IsLegacy) {
      return TheTypeMap.at(I);
    } else {
      auto *IType = I->getType();
      revng_assert(IType->isIntOrPtrTy());
      uint64_t ByteSize = 0ULL;
      if (IType->isIntegerTy()) {
        unsigned NumBits = IType->getIntegerBitWidth();
        revng_assert(NumBits);
        revng_assert(NumBits == 1 or (NumBits % 8 == 0));
        ByteSize = (NumBits == 1) ? 1 : (NumBits / 8);
      } else {
        ByteSize = I->getModule()->getDataLayout().getPointerSize();
      }
      return model::PrimitiveType::make(model::PrimitiveKind::Generic,
                                        ByteSize);
    }
  }
};

template<bool IsLegacy>
using VI = VariableInserter<IsLegacy>;

template<bool IsLegacy>
bool VI<IsLegacy>::shouldReplaceUseWithCopies(const Instruction *I,
                                              const Use &U) const {
  auto *Call = getCallToIsolatedFunction(I);
  if (not Call)
    return true;

  const auto *ProtoT = getCallSitePrototype(Model, cast<CallInst>(I));
  abi::FunctionType::Layout Layout = abi::FunctionType::Layout::make(*ProtoT);

  // If the Isolated function doesn't return an aggregate, we have to
  // inject copies from local variables.
  if (Layout.returnMethod() != abi::FunctionType::ReturnMethod::ModelAggregate)
    return true;

  unsigned NumUses = I->getNumUses();

  // SPTAR return aggregates also need copies from local variables,
  // because they are emitted as scalar pointer variables in C.
  if (Layout.hasSPTAR()) {
    revng_assert(0 == NumUses);
    return true;
  }

  if constexpr (IsLegacy) {
    // Non-SPTAR return aggregates, in legacy mode, are special in many ways:
    // 1. they basically imply a LocalVariable;
    // 2. their only expected use is supposed to be in custom opcodes that
    // expect references
    // For these reasons it would be wrong to inject a Copy.
    if (isCallToTagged(U.getUser(), FunctionTags::Assign)) {
      // If it's an assignment, and the operand number is 0, the value of I is
      // being written somewhere (in the location referenced by operand 1).
      // Hence we have to inject a copy.
      return U.getOperandNo() == 0;
    }
    revng_assert(1 == U.getOperandNo());
    revng_assert(isCallToTagged(U.getUser(), FunctionTags::AddressOf)
                 or isCallToTagged(U.getUser(), FunctionTags::ModelGEPRef));
  } else {
    // Non-SPTAR return aggregates expect at most a single use, which is an
    // assignment of their value into a local variable.
    revng_assert(NumUses < 2);
    if (NumUses) {
      const Use &OnlyUse = *I->uses().begin();
      revng_assert(isa<StoreInst>(OnlyUse.getUser()));
      unsigned OpNum = OnlyUse.getOperandNo();
      revng_assert(OpNum != StoreInst::getPointerOperandIndex());
    }
  }
  return false;
}

template<bool IsLegacy>
bool VI<IsLegacy>::serializeToLocalVariable(Instruction *I) {
  // We can't serialize instructions with reference semantics into local
  // variables because C doesn't have references.
  revng_assert(not isCallToTagged(I, FunctionTags::IsRef));

  // Compute the model type returned from the call.
  revng_assert(I->getType()->isIntOrPtrTy());
  const model::UpcastableType &VariableType = getModelType(I);

  const llvm::DataLayout &DL = I->getModule()->getDataLayout();
  auto ModelSize = VariableType->size().value();
  auto *IType = I->getType();
  auto IRSize = DL.getTypeStoreSize(IType);
  if (ModelSize < IRSize) {
    revng_assert(IType->isPointerTy());
    using model::Architecture::getPointerSize;
    auto PtrSize = getPointerSize(Model.Architecture());
    revng_assert(ModelSize == PtrSize);
  } else if (ModelSize > IRSize) {
    auto &Prototype = *getCallSitePrototype(Model, cast<CallInst>(I));
    using namespace abi::FunctionType;
    abi::FunctionType::Layout Layout = Layout::make(Prototype);
    revng_assert(Layout.returnMethod() == ReturnMethod::ModelAggregate);
    if (Layout.hasSPTAR())
      revng_assert(0 == I->getNumUses());
  }

  // First, we have to declare the LocalVariable, always at the entry block.
  // Create instruction that allocates a LocalVariable
  LocalVarType<IsLegacy> *LocalVariable = LocalVariableBuilder
                                            .createLocalVariable(*VariableType);
  LocalVariable->setDebugLoc(I->getDebugLoc());

  // Then, we have to replace all the uses of I so that they make a Copy
  // from the LocalVariable, unless it's a call to an IsolatedFunction that
  // already returns a local variable, in which case we don't have to do
  // anything with uses.
  for (Use &U : llvm::make_early_inc_range(I->uses())) {
    revng_assert(isa<Instruction>(U.getUser()));

    llvm::Instruction *ValueToUse = LocalVariable;
    if (shouldReplaceUseWithCopies(I, U)) {
      ValueToUse = LocalVariableBuilder.createCopyOnUse(LocalVariable, U);
    }
    U.set(ValueToUse);
  }

  LocalVariableBuilder.createAssignmentBefore(LocalVariable,
                                              I,
                                              I->getNextNonDebugInstruction());

  return true;
}

template<bool IsLegacy>
class SwitchToStatements
  : public llvm::PassInfoMixin<SwitchToStatements<IsLegacy>> {

public:
  const model::Binary &Model;

public:
  SwitchToStatements(const model::Binary &M) : Model(M) {}

public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {

    TypeMap InstructionTypes = {};
    if constexpr (IsLegacy) {
      auto ModelFunction = llvmToModelFunction(Model, F);
      revng_assert(ModelFunction != nullptr);

      InstructionTypes = initModelTypesConsideringUses(F,
                                                       ModelFunction,
                                                       Model,
                                                       /* PointersOnly */
                                                       false);
    }
    VariableInserter<IsLegacy> VarInserter{ F,
                                            Model,
                                            std::move(InstructionTypes) };

    const auto
      &Picked = FAM.getResult<InstructionToSerializePicker<IsLegacy>>(F);
    bool Changed = VarInserter.run(Picked);

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

template<bool IsLegacy>
class SwitchToStatementsPass : public FunctionPass {
public:
  static char ID;

  SwitchToStatementsPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
  }

  bool runOnFunction(Function &F) override;
};

template<bool IsLegacy>
static bool switchToStatements(const model::Binary *Model, llvm::Function &F) {

  revng_log(Log, "switchToStatements: " << F.getName());

  // MPM.addPass(RequireAnalysisPass<GlobalsAA, Module>());
  //
  ModuleAnalysisManager MAM;
  FunctionAnalysisManager FAM;
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
  // MAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM);});

  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerFunctionAnalyses(FAM);

  if constexpr (not IsLegacy) {
    FAM.registerPass([] {
      // Taken from LLVM, in PassBuilder::buildDefaultAAPipeline()
      AAManager AA;
      AA.registerFunctionAnalysis<BasicAA>();
      AA.registerFunctionAnalysis<ScopedNoAliasAA>();
      AA.registerFunctionAnalysis<TypeBasedAA>();

      // With the current LLVM version, SCEVAA doesn't play nice with the new
      // pass manager. We'll have to wait future versions of LLVM to integrate
      // it.

      // Add support for querying global aliasing information when available.
      // Because the `AAManager` is a function analysis and `GlobalsAA` is a
      // module analysis, all that the `AAManager` can do is query for any
      // *cached* results from `GlobalsAA` through a readonly proxy.
      // TODO
      // AA.registerModuleAnalysis<GlobalsAA>();

      return AA;
    });
    FAM.registerPass([] { return AllocasWhoseAddressDoesntLeak(); });
  }
  FAM.registerPass([] { return AvailableExpressionsAnalysis<IsLegacy>(); });
  FAM.registerPass([] { return InstructionToSerializePicker<IsLegacy>(); });

  FunctionPassManager FPM;
  FPM.addPass(SwitchToStatements<IsLegacy>(*Model));
  llvm::PreservedAnalyses Preserved = FPM.run(F, FAM);

  return Preserved.areAllPreserved() ? false : true;
}

template<>
char SwitchToStatementsPass<false>::ID = 0;
template class SwitchToStatementsPass<false>;

template<>
char SwitchToStatementsPass<true>::ID = 0;
template class SwitchToStatementsPass<true>;

template<bool IsLegacy>
bool SwitchToStatementsPass<IsLegacy>::runOnFunction(llvm::Function &F) {
  auto
    *Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel().get();
  return switchToStatements<IsLegacy>(Model, F);
}

using RegisterLegacy = RegisterPass<SwitchToStatementsPass<true>>;
static RegisterLegacy
  X("legacy-switch-to-statements", "LegacySwitchToStatements", false, false);

using Register = RegisterPass<SwitchToStatementsPass<false>>;
static Register
  Y("switch-to-statements", "SwitchToStatementsPass", false, false);

namespace revng::pypeline::piperuns {

// TODO: inline switchToStatements once we dismiss the old pipeline
void SwitchToStatements::runOnLLVMFunction(const model::Function &Function,
                                           llvm::Function &LLVMFunction) {
  switchToStatements<false>(Model.get(), LLVMFunction);
}

} // namespace revng::pypeline::piperuns
