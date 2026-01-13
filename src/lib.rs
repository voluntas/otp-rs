pub mod asm;
pub mod beam;
pub mod compile;
pub mod erl;
pub mod opcodes;

pub use compile::{build_from_erl, compile_module};

#[cfg(test)]
mod tests {
    use crate::beam::{
        chunk_data, decode_code, find_chunk, parse_atom_chunk, parse_beam_chunks,
        parse_code_chunk, parse_export_chunk, parse_import_chunk,
    };
    use crate::compile::compile_module;
    use crate::erl::{parse_module, Expr};
    use crate::opcodes::{build_reverse_opcode_map, load_opcode_map};

    fn extract_wrapped<'a>(arg: &'a str, prefix: &str) -> Option<&'a str> {
        arg.strip_prefix(prefix).and_then(|rest| rest.strip_suffix(')'))
    }

    fn parse_u_arg(arg: &str) -> Option<i64> {
        extract_wrapped(arg, "u(")?.parse().ok()
    }

    fn parse_f_arg(arg: &str) -> Option<i64> {
        extract_wrapped(arg, "f(")?.parse().ok()
    }

    fn parse_a_arg(arg: &str) -> Option<&str> {
        extract_wrapped(arg, "a(")
    }

    #[test]
    fn parse_precedence_mul_before_add() {
        let module = parse_module(
            "-module(t).\n-export([add/2]).\nadd(X,Y) -> X + Y * 2.\n",
        )
        .unwrap();
        let func = &module.functions[0];
        match &func.body {
            Expr::Add(lhs, rhs) => {
                match &**lhs {
                    Expr::Var(name) => assert_eq!(name, "X"),
                    other => panic!("unexpected lhs {:?}", other),
                }
                match &**rhs {
                    Expr::Mul(left, right) => {
                        match &**left {
                            Expr::Var(name) => assert_eq!(name, "Y"),
                            other => panic!("unexpected mul lhs {:?}", other),
                        }
                        match &**right {
                            Expr::Integer(value) => assert_eq!(*value, 2),
                            other => panic!("unexpected mul rhs {:?}", other),
                        }
                    }
                    other => panic!("unexpected rhs {:?}", other),
                }
            }
            other => panic!("unexpected body {:?}", other),
        }
    }

    #[test]
    fn parse_call_forms() {
        let module = parse_module(
            "-module(t).\n-export([f/0]).\nf() -> m:bar(1,2) + local(3).\n",
        )
        .unwrap();
        let func = &module.functions[0];
        match &func.body {
            Expr::Add(lhs, rhs) => {
                match &**lhs {
                    Expr::Call {
                        module,
                        function,
                        args,
                    } => {
                        assert_eq!(module.as_deref(), Some("m"));
                        assert_eq!(function, "bar");
                        assert_eq!(args.len(), 2);
                    }
                    other => panic!("unexpected lhs {:?}", other),
                }
                match &**rhs {
                    Expr::Call {
                        module,
                        function,
                        args,
                    } => {
                        assert!(module.is_none());
                        assert_eq!(function, "local");
                        assert_eq!(args.len(), 1);
                    }
                    other => panic!("unexpected rhs {:?}", other),
                }
            }
            other => panic!("unexpected body {:?}", other),
        }
    }

    #[test]
    fn compile_exports_and_labels() {
        let module = parse_module(
            "-module(t).\n-export([add/2]).\nadd(X,Y) -> X + Y.\n",
        )
        .unwrap();
        let opcode_map = load_opcode_map();
        let opcodes = build_reverse_opcode_map(&opcode_map);
        let beam = compile_module(&module, &opcodes).unwrap();

        let chunks = parse_beam_chunks(&beam).unwrap();
        let atom_chunk = find_chunk(&chunks, b"AtU8")
            .or_else(|| find_chunk(&chunks, b"Atom"))
            .unwrap();
        let atoms = parse_atom_chunk(
            chunk_data(&beam, atom_chunk).unwrap(),
            atom_chunk.id == *b"AtU8",
        )
        .unwrap();
        assert_eq!(atoms.get(1).map(|s| s.as_str()), Some("t"));

        let exp_chunk = find_chunk(&chunks, b"ExpT").unwrap();
        let exports = parse_export_chunk(chunk_data(&beam, exp_chunk).unwrap(), Some(&atoms))
            .unwrap();
        assert!(exports
            .iter()
            .any(|e| e.function == "add" && e.arity == 2));
        assert!(exports
            .iter()
            .any(|e| e.function == "module_info" && e.arity == 0));
        assert!(exports
            .iter()
            .any(|e| e.function == "module_info" && e.arity == 1));

        let code_chunk = find_chunk(&chunks, b"Code").unwrap();
        let code_info = parse_code_chunk(chunk_data(&beam, code_chunk).unwrap()).unwrap();
        let expected_labels = 1 + (code_info.function_count * 2);
        assert_eq!(code_info.label_count, expected_labels);
    }

    #[test]
    fn compile_local_call_targets_entry_label() {
        let module = parse_module(
            "-module(t).\n-export([f/0, g/0]).\nf() -> g().\ng() -> 1.\n",
        )
        .unwrap();
        let opcode_map = load_opcode_map();
        let opcodes = build_reverse_opcode_map(&opcode_map);
        let beam = compile_module(&module, &opcodes).unwrap();

        let chunks = parse_beam_chunks(&beam).unwrap();
        let atom_chunk = find_chunk(&chunks, b"AtU8")
            .or_else(|| find_chunk(&chunks, b"Atom"))
            .unwrap();
        let atoms = parse_atom_chunk(
            chunk_data(&beam, atom_chunk).unwrap(),
            atom_chunk.id == *b"AtU8",
        )
        .unwrap();

        let code_chunk = find_chunk(&chunks, b"Code").unwrap();
        let code_data = chunk_data(&beam, code_chunk).unwrap();
        let code_info = parse_code_chunk(code_data).unwrap();
        let code = &code_data[code_info.code_offset..];
        let ops = decode_code(code, &opcode_map, Some(&atoms), 200).unwrap();

        let mut g_entry_label = None;
        for (idx, op) in ops.iter().enumerate() {
            if op.name != "func_info" || op.args.len() < 3 {
                continue;
            }
            let func = parse_a_arg(&op.args[1]);
            let arity = parse_u_arg(&op.args[2]);
            if func == Some("g") && arity == Some(0) {
                if let Some(next) = ops.get(idx + 1) {
                    if next.name == "label" && !next.args.is_empty() {
                        g_entry_label = parse_u_arg(&next.args[0]);
                    }
                }
            }
        }
        let g_entry_label = g_entry_label.expect("missing g/0 entry label");

        let mut saw_call = false;
        for op in &ops {
            if op.name == "call" && op.args.len() == 2 {
                if parse_u_arg(&op.args[0]) == Some(0) {
                    if let Some(label) = parse_f_arg(&op.args[1]) {
                        if label == g_entry_label {
                            saw_call = true;
                            break;
                        }
                    }
                }
            }
        }
        assert!(saw_call, "missing call to g/0 entry label");
    }

    #[test]
    fn compile_module_info_call_ext_only_imports() {
        let module = parse_module("-module(t).\n-export([f/0]).\nf() -> ok.\n").unwrap();
        let opcode_map = load_opcode_map();
        let opcodes = build_reverse_opcode_map(&opcode_map);
        let beam = compile_module(&module, &opcodes).unwrap();

        let chunks = parse_beam_chunks(&beam).unwrap();
        let atom_chunk = find_chunk(&chunks, b"AtU8")
            .or_else(|| find_chunk(&chunks, b"Atom"))
            .unwrap();
        let atoms = parse_atom_chunk(
            chunk_data(&beam, atom_chunk).unwrap(),
            atom_chunk.id == *b"AtU8",
        )
        .unwrap();

        let imp_chunk = find_chunk(&chunks, b"ImpT").unwrap();
        let imports = parse_import_chunk(chunk_data(&beam, imp_chunk).unwrap(), Some(&atoms))
            .unwrap();
        let mi1 = imports.iter().position(|entry| {
            entry.module == "erlang" && entry.function == "get_module_info" && entry.arity == 1
        });
        let mi2 = imports.iter().position(|entry| {
            entry.module == "erlang" && entry.function == "get_module_info" && entry.arity == 2
        });
        let mi1 = mi1.expect("missing get_module_info/1 import") as i64;
        let mi2 = mi2.expect("missing get_module_info/2 import") as i64;

        let code_chunk = find_chunk(&chunks, b"Code").unwrap();
        let code_data = chunk_data(&beam, code_chunk).unwrap();
        let code_info = parse_code_chunk(code_data).unwrap();
        let code = &code_data[code_info.code_offset..];
        let ops = decode_code(code, &opcode_map, Some(&atoms), 200).unwrap();

        let mut called = Vec::new();
        for op in &ops {
            if op.name == "call_ext_only" && op.args.len() == 2 {
                if let Some(index) = parse_u_arg(&op.args[1]) {
                    called.push(index);
                }
            }
        }

        assert!(called.contains(&mi1), "missing call_ext_only get_module_info/1");
        assert!(called.contains(&mi2), "missing call_ext_only get_module_info/2");
    }
}
