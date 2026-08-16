// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/runtime_info.h"

namespace {

TEST(RuntimeInfo, CommonCompilationStateParticipatesInEquality) {
    Shader::RuntimeInfo lhs{};
    Shader::RuntimeInfo rhs{};
    lhs.Initialize(Shader::Stage::Compute);
    rhs.Initialize(Shader::Stage::Compute);
    EXPECT_EQ(lhs, rhs);

    rhs.num_user_data = 1;
    EXPECT_NE(lhs, rhs);
    rhs.num_user_data = 0;

    rhs.num_allocated_vgprs = 4;
    EXPECT_NE(lhs, rhs);
    rhs.num_allocated_vgprs = 0;

    rhs.fp_round_mode32 = static_cast<AmdGpu::FpRoundMode>(1);
    EXPECT_NE(lhs, rhs);
}

TEST(RuntimeInfo, StageParticipatesInEquality) {
    Shader::RuntimeInfo vertex{};
    Shader::RuntimeInfo compute{};
    vertex.Initialize(Shader::Stage::Vertex);
    compute.Initialize(Shader::Stage::Compute);

    EXPECT_NE(vertex, compute);
}

TEST(RuntimeInfo, ComputeSharedMemoryParticipatesInEquality) {
    Shader::RuntimeInfo lhs{};
    Shader::RuntimeInfo rhs{};
    lhs.Initialize(Shader::Stage::Compute);
    rhs.Initialize(Shader::Stage::Compute);
    rhs.cs_info.shared_memory_size = 1024;

    EXPECT_NE(lhs, rhs);
}

TEST(RuntimeInfo, GeometryStateUsesValueEquality) {
    Shader::RuntimeInfo lhs{};
    Shader::RuntimeInfo rhs{};
    lhs.Initialize(Shader::Stage::Geometry);
    rhs.Initialize(Shader::Stage::Geometry);

    // Zero-valued, identically initialized geometry state must compare equal.
    EXPECT_EQ(lhs, rhs);

    lhs.gs_info.num_invocations = 1;
    rhs.gs_info.num_invocations = 2;
    EXPECT_NE(lhs, rhs);

    rhs.gs_info.num_invocations = 1;
    rhs.gs_info.in_vertex_data_size = 16;
    EXPECT_NE(lhs, rhs);
}

} // namespace
