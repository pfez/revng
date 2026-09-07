//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-statements="enable-patterns=degenerate-branch" | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// An if whose then and else are both empty does nothing but evaluate its
// condition, so it reduces to an expression statement yielding that condition.

module attributes {clift.module} {
  // CHECK-LABEL: clift.func @f
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK-NOT: clift.if
    // CHECK: clift.expr {
    // CHECK: clift.yield %arg0 : !int32_t
    clift.if {
      clift.yield %arg0 : !int32_t
    } then {
    } else {
    }
  }
}
