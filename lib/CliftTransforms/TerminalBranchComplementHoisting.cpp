//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/CliftTransforms/Expressions.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/CliftTransforms/RewriteHelpers.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTTERMINALBRANCHCOMPLEMENTHOISTING
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

// A branch region falls through when control can reach the end of the branch
// operation through it. A block-less region - a missing else or default, or an
// empty `{}` case body - also falls through.
static bool fallsThrough(mlir::Region &R) {
  return not isIndirectlyNoFallthrough(R);
}

// Computes an approximation of the size of a statement region in C.
static unsigned approximateRegionWeight(mlir::Region &R) {
  if (R.empty())
    return 0;

  revng_assert(R.hasOneBlock());

  unsigned Weight = 0;
  R.walk([&Weight](mlir::Operation *Op) {
    // Label declarations can be ignored, as they have no C representation.
    if (mlir::isa<MakeLabelOp>(Op))
      return;

    // Neither expression statement nor yield operations have a direct C
    // representation.
    if (mlir::isa<ExpressionStatementOp, YieldOp>(Op))
      return;

    ++Weight;
  });
  return Weight;
}

// A non-fallthrough region has a definite kind when control leaves it in a
// single known way (continue, break, goto or return), as opposed to a Mixed
// region whose nested branches leave in differing ways.
static bool hasDefiniteKind(mlir::Region &R) {
  return isIndirectlyNoFallthrough(R) != NoFallthroughKind::Mixed;
}

// When no branch region falls through, any of them may be hoisted. Generalising
// the original if-statement heuristic to any number of regions: prefer a region
// that leaves in a single definite way over a Mixed one, then the one with the
// lowest weight (ties resolved towards the later region).
static unsigned selectByHeuristic(llvm::MutableArrayRef<mlir::Region> Regions) {
  unsigned Best = 0;
  bool BestDefinite = hasDefiniteKind(Regions[0]);
  // The weight is only needed to break ties, so it is computed lazily and
  // cached, rather than re-walking the winning region on every iteration.
  std::optional<unsigned> BestWeight;

  for (unsigned I = 1; I < Regions.size(); ++I) {
    bool CandidateDefinite = hasDefiniteKind(Regions[I]);

    // Prefer a region that leaves in a single definite way.
    if (CandidateDefinite != BestDefinite) {
      if (CandidateDefinite) {
        Best = I;
        BestDefinite = true;
        BestWeight.reset();
      }
      continue;
    }

    // Otherwise prefer the lowest weight (or the later region if equal).
    if (not BestWeight)
      BestWeight = approximateRegionWeight(Regions[Best]);
    unsigned CandidateWeight = approximateRegionWeight(Regions[I]);
    if (CandidateWeight <= *BestWeight) {
      Best = I;
      BestWeight = CandidateWeight;
    }
  }

  return Best;
}

// Hoisting the branch region \p Target of \p Branch makes progress (is not a
// no-op) when it has statements to move out, or when the hoist inverts the
// condition of an if statement.
static bool
hoistMakesProgress(BranchOpInterface Branch, unsigned Target, mlir::Region &R) {
  if (mlir::Block *Block = getOnlyBlock(R); Block and not Block->empty())
    return true;

  // An if statement hoisting its then branch inverts the condition, which is
  // progress even when that branch is empty.
  return mlir::isa<IfOp>(Branch.getOperation()) and Target == 0;
}

// Selects the branch region of \p Branch to be inlined into the nesting scope,
// as an index into getBranchRegions(). If none should be hoisted, the result is
// nullopt.
static std::optional<unsigned> selectHoistingTarget(BranchOpInterface Branch) {
  // Preserve the exact if-statement handling of empty branches.
  if (auto If = mlir::dyn_cast<IfOp>(Branch.getOperation())) {
    // No else branch: nothing complementary to hoist.
    if (If.getElse().empty())
      return std::nullopt;

    // Empty then branch (`then {}`): inverting the if drops it.
    if (If.getThen().empty())
      return 0;

    // Empty else block (`else { ^bb: }`): drop it.
    if (If.getElse().front().empty())
      return 1;
  }

  llvm::MutableArrayRef<mlir::Region> Regions = Branch.getBranchRegions();

  // Collect the branch regions that fall through.
  llvm::SmallVector<unsigned> Fallthrough;
  for (unsigned I = 0; I < Regions.size(); ++I) {
    if (fallsThrough(Regions[I]))
      Fallthrough.push_back(I);
  }

  // With two or more fall-through branches, hoisting one would still leave
  // another that reaches the hoisted code: the rewrite would be invalid.
  if (Fallthrough.size() >= 2)
    return std::nullopt;

  // With a single fall-through branch, all others are non-fallthrough, so it
  // can be hoisted (as long as that makes progress).
  if (Fallthrough.size() == 1) {
    unsigned Target = Fallthrough.front();
    if (hoistMakesProgress(Branch, Target, Regions[Target]))
      return Target;
    return std::nullopt;
  }

  // No branch falls through: the whole operation is non-fallthrough, so any
  // branch may be hoisted. Pick one with the heuristic.
  return selectByHeuristic(Regions);
}

// After its body is hoisted out, the emptied branch region can be removed
// entirely for an if (its else) or a switch default. A switch case is instead
// kept as an empty body, so its value still falls through to the hoisted code.
//
// Dropping a case is never valid, which is subtle: a case is hoisted only when
// the switch has a non-fallthrough default - either the case is the sole
// fallthrough branch, so the default does not fall through, or
// selectByHeuristic ran, which requires no branch (the default included) to
// fall through, so the default is non-empty. Removing the case would then route
// its value to that default instead of to the hoisted code. A switch with no
// default hoists nothing anyway: its implicit fallthrough for unmatched values
// is itself a fallthrough branch, so no lone fallthrough case is ever left to
// hoist.
static bool isDroppableBranch(BranchOpInterface Branch, unsigned Target) {
  if (mlir::isa<IfOp>(Branch.getOperation()))
    return true;

  // getBranchRegions() of a switch is [default, cases...]; only the default may
  // be dropped.
  return Target == 0;
}

struct TerminalBranchComplementHoistingPattern
  : mlir::OpInterfaceRewritePattern<BranchOpInterface> {

  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(BranchOpInterface Branch,
                  mlir::PatternRewriter &Rewriter) const override {
    auto OptTarget = selectHoistingTarget(Branch);
    if (not OptTarget)
      return mlir::failure();
    unsigned Target = *OptTarget;

    // An if statement hoisting its then branch is inverted first, so the
    // hoisted branch becomes the else and the readable `if (!c) { ... }` form
    // is kept.
    if (auto If = mlir::dyn_cast<IfOp>(Branch.getOperation());
        If and Target == 0) {
      invertIfStatement(Rewriter, If);
      Target = 1;
    }

    mlir::Region &R = Branch.getBranchRegions()[Target];
    if (mlir::Block *Block = getOnlyBlock(R)) {
      mlir::Operation *Op = Branch.getOperation();
      Rewriter.updateRootInPlace(Op, [&]() {
        inlineBlockBefore(Rewriter,
                          Block,
                          Op->getBlock(),
                          std::next(Op->getIterator()));
      });

      if (isDroppableBranch(Branch, Target))
        Rewriter.eraseBlock(Block);
    }

    return mlir::success();
  }
};

template<typename T>
using PassBase = clift::impl::CliftTerminalBranchComplementHoistingBase<T>;

struct TerminalBranchComplementHoistingPass
  : PassBase<TerminalBranchComplementHoistingPass> {

  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();
    FunctionOp Function = getOperation();

    // Apply terminal branch complement hoisting:
    {
      mlir::RewritePatternSet Patterns(Context);
      Patterns.add<TerminalBranchComplementHoistingPattern>(Context);

      // TODO: Use walkAndApplyPatterns
      if (mlir::applyPatternsAndFoldGreedily(Function, std::move(Patterns))
            .failed())
        signalPassFailure();
    }

    // Terminal branch complement hoisting may need to invert if-statements.
    // That introduces negated conditions, e.g. `!!x`. This rewrite undoes them:
    {
      mlir::RewritePatternSet Patterns(Context);
      populateWithBooleanNegationPatterns(Patterns);

      // TODO: Use walkAndApplyPatterns
      if (mlir::applyPatternsAndFoldGreedily(Function, std::move(Patterns))
            .failed())
        signalPassFailure();
    }
  }
};

} // namespace

PassPtr<FunctionOp> clift::createTerminalBranchComplementHoistingPass() {
  return std::make_unique<TerminalBranchComplementHoistingPass>();
}
