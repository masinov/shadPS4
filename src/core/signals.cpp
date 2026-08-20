// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <csignal>
#include <exception>
#include <intrin.h>
#include "common/arch.h"
#include "common/assert.h"
#include "common/decoder.h"
#include "common/path_util.h"
#include "common/signal_context.h"
#include "core/libraries/kernel/threads/exception.h"

#include "core/signals.h"
#include "core/veh_stack.h"

#ifdef _WIN32
#include <windows.h>
// dbghelp.h requires the windows.h types above and must not be reordered before it.
// clang-format off
#include <dbghelp.h>
// clang-format on
#else
#include <csignal>
#include <pthread.h>
#ifdef ARCH_X86_64
#include <Zydis/Formatter.h>
#endif
#endif

#ifndef _WIN32
namespace Libraries::Kernel {
void SigactionHandler(int native_signum, siginfo_t* inf, ucontext_t* raw_context);
extern std::array<OrbisKernelExceptionHandler, 32> Handlers;
} // namespace Libraries::Kernel
#endif

namespace Core {

#if defined(_WIN32)

static long SignalHandlerImpl(EXCEPTION_POINTERS* pExp) noexcept {
    const auto* signals = Signals::Instance();
    DWORD code = 0;

    if (pExp != nullptr && pExp->ExceptionRecord != nullptr) {
        code = pExp->ExceptionRecord->ExceptionCode;
    }

    bool handled = false;
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        handled = signals->DispatchAccessViolation(
            pExp, reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]));
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
    case EXCEPTION_PRIV_INSTRUCTION:
        handled = signals->DispatchIllegalInstruction(pExp);
        break;
    case DBG_PRINTEXCEPTION_C:
    case DBG_PRINTEXCEPTION_WIDE_C:
        // Used by OutputDebugString functions.
        return EXCEPTION_CONTINUE_EXECUTION;
    default:
        break;
    }

    if (handled) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    // Vectored handlers run before frame-based SEH and language-runtime handlers. Exceptions that
    // do not belong to the emulator must continue through normal Windows dispatch.
    return EXCEPTION_CONTINUE_SEARCH;
}

static LONG WINAPI SignalHandler(EXCEPTION_POINTERS* pExp) noexcept {
#ifdef _WIN64
    return static_cast<LONG>(RunOnVehStack(SignalHandlerImpl, pExp));
#else
    return static_cast<LONG>(SignalHandlerImpl(pExp));
#endif
}

#else

static std::string DisassembleInstruction(void* code_address) {
    char buffer[256] = "<unable to decode>";

#ifdef ARCH_X86_64
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    const auto status =
        Common::Decoder::Instance()->decodeInstruction(instruction, operands, code_address);
    if (ZYAN_SUCCESS(status)) {
        ZydisFormatter formatter;
        ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_INTEL);
        ZydisFormatterFormatInstruction(&formatter, &instruction, operands,
                                        instruction.operand_count_visible, buffer, sizeof(buffer),
                                        reinterpret_cast<u64>(code_address), ZYAN_NULL);
    }
#endif

    return buffer;
}

void SignalHandler(int sig, siginfo_t* info, void* raw_context) {
    const auto* signals = Signals::Instance();

    auto* code_address = Common::GetRip(raw_context);

    switch (sig) {
    case SIGSEGV:
    case SIGBUS: {
        const bool is_write = Common::IsWriteError(raw_context);
        if (!signals->DispatchAccessViolation(raw_context, info->si_addr)) {
            // If the guest has installed a custom signal handler, and the access violation didn't
            // come from HLE memory tracking, pass the signal on
            if (Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)]) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
            UNREACHABLE_MSG("Unhandled access violation at code address {}: {} address {}",
                            fmt::ptr(code_address), is_write ? "Write to" : "Read from",
                            fmt::ptr(info->si_addr));
        }
        break;
    }
    case SIGILL:
        if (!signals->DispatchIllegalInstruction(raw_context)) {
            if (Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)]) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
            UNREACHABLE_MSG("Unhandled illegal instruction at code address {}: {}",
                            fmt::ptr(code_address), DisassembleInstruction(code_address));
        }
        break;
    default:
        if (sig == SIGSLEEP) {
            // Sleep thread until signal is received again
            sigset_t sigset;
            sigemptyset(&sigset);
            sigaddset(&sigset, SIGSLEEP);
            sigwait(&sigset, &sig);
        }
        break;
    }
}

#endif

#if defined(_WIN32)
// Last-resort crash recorder. Run 35 terminated with no device loss, no assertion, no WER report
// and a log truncated mid-write: an entire class of failure was invisible. This filter runs only
// when no VEH/guest handler claimed the exception, writes one minidump beside the log directory,
// and lets the process die normally afterwards.
static LONG WINAPI UnhandledCrashFilter(EXCEPTION_POINTERS* pointers) {
    static std::atomic_flag dumped = ATOMIC_FLAG_INIT;
    if (dumped.test_and_set()) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const auto dump_path = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "crash.dmp";
    const HANDLE file = CreateFileW(dump_path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION info{
            .ThreadId = GetCurrentThreadId(),
            .ExceptionPointers = pointers,
            .ClientPointers = FALSE,
        };
        MiniDumpWriteDump(
            GetCurrentProcess(), GetCurrentProcessId(), file,
            static_cast<MINIDUMP_TYPE>(MiniDumpWithIndirectlyReferencedMemory | MiniDumpScanMemory),
            &info, nullptr, nullptr);
        CloseHandle(file);
    }
    LOG_CRITICAL(Core, "Unhandled exception {:#x} at {}; minidump written to {}",
                 pointers->ExceptionRecord->ExceptionCode,
                 fmt::ptr(pointers->ExceptionRecord->ExceptionAddress),
                 fmt::UTF(dump_path.u8string()));
    return EXCEPTION_CONTINUE_SEARCH;
}

// Runs 35 and 64 died with no dump, no WER entry and a log truncated mid-write: paths through
// std::terminate / abort / fastfail bypass SetUnhandledExceptionFilter entirely. The first
// version of this handler raised a software exception to reach the filter - a mistake: a raised
// exception travels through every vectored and frame handler first, and in run 70 something en
// route claimed it, so the process kept EXECUTING from a dying state (the game "restarted" to
// its menu and the relaunch truncated the session log). The handler now writes the dump
// directly with a captured context and terminates immediately; nothing can intercept it.
[[noreturn]] static void TerminateDumpHandler() {
    static std::atomic_flag dumping = ATOMIC_FLAG_INIT;
    if (!dumping.test_and_set()) {
        CONTEXT context{};
        context.ContextFlags = CONTEXT_FULL;
        RtlCaptureContext(&context);
        EXCEPTION_RECORD record{};
        record.ExceptionCode = 0xE000DEADu;
        record.ExceptionAddress = _ReturnAddress();
        EXCEPTION_POINTERS pointers{
            .ExceptionRecord = &record,
            .ContextRecord = &context,
        };
        const auto dump_path = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "crash.dmp";
        const HANDLE file = CreateFileW(dump_path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                        FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            MINIDUMP_EXCEPTION_INFORMATION info{
                .ThreadId = GetCurrentThreadId(),
                .ExceptionPointers = &pointers,
                .ClientPointers = FALSE,
            };
            MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file,
                              static_cast<MINIDUMP_TYPE>(MiniDumpWithIndirectlyReferencedMemory |
                                                         MiniDumpScanMemory),
                              &info, nullptr, nullptr);
            CloseHandle(file);
        }
    }
    TerminateProcess(GetCurrentProcess(), 0xE000DEADu);
    ExitProcess(0xE000DEADu); // unreachable; satisfies [[noreturn]] analysis
}
#endif

SignalDispatch::SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(handle = AddVectoredExceptionHandler(0, SignalHandler),
               "Failed to register exception handler.");
    SetUnhandledExceptionFilter(UnhandledCrashFilter);
    std::set_terminate(TerminateDumpHandler);
    // Route abort() through the unhandled filter as well instead of fastfailing silently.
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    std::signal(SIGABRT, [](int) { TerminateDumpHandler(); });
#else
    struct sigaction action{};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to register access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to register illegal instruction signal handler.");
    ASSERT_MSG(sigaction(SIGSLEEP, &action, nullptr) == 0,
               "Failed to register sleep signal handler.");
#endif
}

SignalDispatch::~SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(RemoveVectoredExceptionHandler(handle), "Failed to remove exception handler.");
#else
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to remove access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to remove illegal instruction signal handler.");
#endif
}

bool SignalDispatch::DispatchAccessViolation(void* context, void* fault_address) const {
    for (const auto& [handler, _] : access_violation_handlers) {
        if (handler(context, fault_address)) {
            return true;
        }
    }
    return false;
}

bool SignalDispatch::DispatchIllegalInstruction(void* context) const {
    for (const auto& [handler, _] : illegal_instruction_handlers) {
        if (handler(context)) {
            return true;
        }
    }
    return false;
}

} // namespace Core
