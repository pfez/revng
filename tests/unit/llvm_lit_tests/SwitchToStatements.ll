;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt -enable-new-pm=true -load-pass-plugin=./lib/revng/analyses/librevngCanonicalize.so -passes=switch-to-statements-test %s -S -o - | FileCheck %s

@segment = external global i64

; =================================
; Arguments and locals don't alias.
; =================================

; Function argument doesn't alias the alloca.
; No new local variables should be created.
;
; CHECK-LABEL: define i64 @a
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NOT: alloca

define i64 @a(ptr %arg) {
  %local = alloca i64
  store i64 15, ptr %local
  %fifteen = load i64, ptr %local
  store i64 7, ptr %arg
  ret i64 %fifteen
}

; Function argument doesn't alias the alloca.
; No new local variables should be created.
;
; This is the same as the previous test, but with alloca and argument swapped.
;
; CHECK-LABEL: define i64 @a_swapped
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NOT: alloca

define i64 @a_swapped(ptr %arg) {
  %local = alloca i64
  store i64 15, ptr %arg
  %fifteen = load i64, ptr %arg
  store i64 7, ptr %local
  ret i64 %fifteen
}

; =================================
; Arguments and globals may alias.
; =================================

; Function argument doesn't alias the @segment global.
; A new local variable should be created, and the result of the load should be
; stored in there.
; The the ret instruction should load from there again.
;
; CHECK-LABEL: define i64 @b
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca [8 x i8], align 1, !revng.variable_type
; CHECK-NEXT: [[LOADED_VALUE:%[a-zA-Z0-9_]+]] = load i64, ptr %arg
; CHECK-NEXT: store i64 [[LOADED_VALUE]], ptr [[ALLOCA]]
; CHECK-NEXT: store i64 7, ptr @segment
; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9_]+]] = load i64, ptr [[ALLOCA]]
; CHECK-NEXT: ret i64 [[RESULT]]

define i64 @b(ptr %arg) {
  %loaded_value = load i64, ptr %arg
  store i64 7, ptr @segment
  ret i64 %loaded_value
}

; Function argument doesn't alias the @segment global.
; A new local variable should be created, and the result of the load should be
; stored in there.
; The the ret instruction should load from there again.
;
; This is the same as the previous test, but with segment and argument swapped.
;
; CHECK-LABEL: define i64 @b_swapped
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca [8 x i8], align 1, !revng.variable_type
; CHECK-NEXT: [[LOADED_VALUE:%[a-zA-Z0-9_]+]] = load i64, ptr @segment
; CHECK-NEXT: store i64 [[LOADED_VALUE]], ptr [[ALLOCA]]
; CHECK-NEXT: store i64 7, ptr %arg
; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9_]+]] = load i64, ptr [[ALLOCA]]
; CHECK-NEXT: ret i64 [[RESULT]]

define i64 @b_swapped(ptr %arg) {
  %loaded_value = load i64, ptr @segment
  store i64 7, ptr %arg
  ret i64 %loaded_value
}

; ===============================
; Locals and globals don't alias.
; ===============================

; Segment global doesn't alias the alloca.
; No new local variables should be created.
;
; CHECK-LABEL: define i64 @c
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NOT: alloca

define i64 @c() {
  %local = alloca i64
  store i64 15, ptr %local
  %fifteen = load i64, ptr %local
  store i64 7, ptr @segment
  ret i64 %fifteen
}

; Segment global doesn't alias the alloca.
; No new local variables should be created.
;
; This is the same as the previous test, but with alloca and argument swapped.
;
; CHECK-LABEL: define i64 @c_swapped
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NOT: alloca

define i64 @c_swapped() {
  %local = alloca i64
  store i64 15, ptr @segment
  %fifteen = load i64, ptr @segment
  store i64 7, ptr %local
  ret i64 %fifteen
}

; ============================================================================
; Non-overlapping accesses to the same allocation (local, global, or argument)
; don't alias, so they don't force the emission of a new local variable.
; ============================================================================

; CHECK-LABEL: define i32 @d
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i64
; CHECK-NOT: alloca
define i32 @d() {
  %stack = alloca i64
  %stack_at_0 = getelementptr i8, ptr %stack, i64 0
  %stack_at_4 = getelementptr i8, ptr %stack_at_0, i64 4
  store i32 15, ptr %stack_at_0
  %fifteen = load i32, ptr %stack_at_0
  store i32 7, ptr %stack_at_4
  ret i32 %fifteen
}

; CHECK-LABEL: define i32 @d_global
; CHECK-NOT: alloca
define i32 @d_global() {
  %segment_at_0 = getelementptr i8, ptr @segment, i64 0
  %segment_at_4 = getelementptr i8, ptr %segment_at_0, i64 4
  store i32 15, ptr %segment_at_4
  %fifteen = load i32, ptr %segment_at_4
  store i32 7, ptr %segment_at_0
  ret i32 %fifteen
}

; CHECK-LABEL: define i32 @d_argument
; CHECK-NOT: alloca
define i32 @d_argument(ptr %arg) {
  %arg_at_0 = getelementptr i8, ptr %arg, i64 0
  %arg_at_4 = getelementptr i8, ptr %arg_at_0, i64 4
  store i32 15, ptr %arg_at_0
  %fifteen = load i32, ptr %arg_at_0
  store i32 7, ptr %arg_at_4
  ret i32 %fifteen
}
