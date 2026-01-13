# Plan

## Current Work
- otp-rs is a minimal Erlang -> BEAM pipeline in Rust (parser, compiler, assembler, BEAM chunk writer/reader).
- Code has been split into modules: beam, opcodes, asm, erl, compile, with a CLI in main.rs.
- Tests cover parsing precedence/calls and codegen for exports, labels, local calls, and module_info imports.
- tmp contains small .erl samples: min.erl, add.erl, call.erl, ext.erl.

## Goals
- Keep the compiler structure aligned with OTP sources (beam_asm/beam_dict/beam_opcodes/beam_file/genop), not just "make it work".
- Extend the BEAM encoder/decoder in a generic way (more chunks/opcodes, robust validation).
- Build and run compiled BEAMs with erl, then compare against erlc output for correctness.
- Grow tests to cover more expressions, calls, and chunk consistency.
