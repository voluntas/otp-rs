use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::asm::{encode_op, Arg, AsmBuilder};
use crate::beam::{build_atom_table, chunk, chunk_with_head, BEAM_FORMAT_NUMBER};
use crate::erl::{parse_module, ExportSpec, Expr, Function, Module};
use crate::opcodes::opcode_number;

fn build_import_chunk(asm: &AsmBuilder) -> Result<Vec<u8>, String> {
    let mut data = Vec::new();
    data.extend_from_slice(&(asm.imports.len() as i32).to_be_bytes());
    for entry in &asm.imports {
        let module_index = asm.atom_index(&entry.module)? as i32;
        let function_index = asm.atom_index(&entry.function)? as i32;
        data.extend_from_slice(&module_index.to_be_bytes());
        data.extend_from_slice(&function_index.to_be_bytes());
        data.extend_from_slice(&entry.arity.to_be_bytes());
    }
    Ok(chunk(b"ImpT", &data))
}

fn build_export_chunk(asm: &AsmBuilder) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&(asm.exports.len() as i32).to_be_bytes());
    for (atom_index, arity, label) in &asm.exports {
        data.extend_from_slice(&(*atom_index as i32).to_be_bytes());
        data.extend_from_slice(&arity.to_be_bytes());
        data.extend_from_slice(&label.to_be_bytes());
    }
    chunk(b"ExpT", &data)
}

struct RegAlloc {
    next_reg: i64,
}

impl RegAlloc {
    fn new(start: i64) -> Self {
        Self { next_reg: start }
    }

    fn alloc(&mut self) -> i64 {
        let reg = self.next_reg;
        self.next_reg += 1;
        reg
    }
}

fn compile_expr_to_reg(
    expr: &Expr,
    var_map: &HashMap<String, i64>,
    regs: &mut RegAlloc,
    local_labels: &HashMap<(String, usize), i64>,
    opcodes: &HashMap<(String, usize), u8>,
    asm: &mut AsmBuilder,
    out: &mut Vec<u8>,
) -> Result<i64, String> {
    match expr {
        Expr::Atom(name) => {
            let reg = regs.alloc();
            let op = encode_op("move", &[Arg::Atom(name.clone()), Arg::X(reg)], opcodes, asm)?;
            out.extend(op);
            Ok(reg)
        }
        Expr::Integer(value) => {
            let reg = regs.alloc();
            let op = encode_op("move", &[Arg::I(*value), Arg::X(reg)], opcodes, asm)?;
            out.extend(op);
            Ok(reg)
        }
        Expr::Var(name) => {
            if let Some(index) = var_map.get(name) {
                Ok(*index)
            } else {
                Err(format!("undefined variable {}", name))
            }
        }
        Expr::Add(lhs, rhs) => {
            let left = compile_expr_to_reg(lhs, var_map, regs, local_labels, opcodes, asm, out)?;
            let right = compile_expr_to_reg(rhs, var_map, regs, local_labels, opcodes, asm, out)?;
            let live = regs.next_reg;
            let op = encode_op(
                "gc_bif2",
                &[
                    Arg::F(0),
                    Arg::U(live),
                    Arg::ExtFunc {
                        module: "erlang".to_string(),
                        function: "+".to_string(),
                        arity: 2,
                    },
                    Arg::X(left),
                    Arg::X(right),
                    Arg::X(left),
                ],
                opcodes,
                asm,
            )?;
            out.extend(op);
            Ok(left)
        }
        Expr::Sub(lhs, rhs) => {
            let left = compile_expr_to_reg(lhs, var_map, regs, local_labels, opcodes, asm, out)?;
            let right = compile_expr_to_reg(rhs, var_map, regs, local_labels, opcodes, asm, out)?;
            let live = regs.next_reg;
            let op = encode_op(
                "gc_bif2",
                &[
                    Arg::F(0),
                    Arg::U(live),
                    Arg::ExtFunc {
                        module: "erlang".to_string(),
                        function: "-".to_string(),
                        arity: 2,
                    },
                    Arg::X(left),
                    Arg::X(right),
                    Arg::X(left),
                ],
                opcodes,
                asm,
            )?;
            out.extend(op);
            Ok(left)
        }
        Expr::Mul(lhs, rhs) => {
            let left = compile_expr_to_reg(lhs, var_map, regs, local_labels, opcodes, asm, out)?;
            let right = compile_expr_to_reg(rhs, var_map, regs, local_labels, opcodes, asm, out)?;
            let live = regs.next_reg;
            let op = encode_op(
                "gc_bif2",
                &[
                    Arg::F(0),
                    Arg::U(live),
                    Arg::ExtFunc {
                        module: "erlang".to_string(),
                        function: "*".to_string(),
                        arity: 2,
                    },
                    Arg::X(left),
                    Arg::X(right),
                    Arg::X(left),
                ],
                opcodes,
                asm,
            )?;
            out.extend(op);
            Ok(left)
        }
        Expr::Call {
            module,
            function,
            args,
        } => {
            let mut temps = Vec::new();
            for arg in args {
                let reg =
                    compile_expr_to_reg(arg, var_map, regs, local_labels, opcodes, asm, out)?;
                let temp = regs.alloc();
                if temp != reg {
                    let op = encode_op("move", &[Arg::X(reg), Arg::X(temp)], opcodes, asm)?;
                    out.extend(op);
                }
                temps.push(temp);
            }

            for (idx, temp) in temps.iter().enumerate() {
                let op = encode_op("move", &[Arg::X(*temp), Arg::X(idx as i64)], opcodes, asm)?;
                out.extend(op);
            }

            let arity = args.len();
            if let Some(module) = module {
                let op = encode_op(
                    "call_ext",
                    &[
                        Arg::U(arity as i64),
                        Arg::ExtFunc {
                            module: module.clone(),
                            function: function.clone(),
                            arity: arity as i64,
                        },
                    ],
                    opcodes,
                    asm,
                )?;
                out.extend(op);
            } else {
                let key = (function.clone(), arity);
                let label = local_labels
                    .get(&key)
                    .ok_or_else(|| format!("unknown local function {}/{}", function, arity))?;
                let op = encode_op(
                    "call",
                    &[Arg::U(arity as i64), Arg::F(*label)],
                    opcodes,
                    asm,
                )?;
                out.extend(op);
            }

            let result = regs.alloc();
            if result != 0 {
                let op = encode_op("move", &[Arg::X(0), Arg::X(result)], opcodes, asm)?;
                out.extend(op);
            }
            Ok(result)
        }
    }
}

fn compile_function(
    module_name: &str,
    func: &Function,
    func_label: i64,
    entry_label: i64,
    local_labels: &HashMap<(String, usize), i64>,
    opcodes: &HashMap<(String, usize), u8>,
    asm: &mut AsmBuilder,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let op = encode_op("label", &[Arg::U(func_label)], opcodes, asm)?;
    out.extend(op);

    let op = encode_op(
        "func_info",
        &[
            Arg::Atom(module_name.to_string()),
            Arg::Atom(func.name.clone()),
            Arg::U(func.params.len() as i64),
        ],
        opcodes,
        asm,
    )?;
    out.extend(op);

    let op = encode_op("label", &[Arg::U(entry_label)], opcodes, asm)?;
    out.extend(op);

    let mut var_map = HashMap::new();
    for (idx, name) in func.params.iter().enumerate() {
        var_map.insert(name.clone(), idx as i64);
    }

    let mut regs = RegAlloc::new(func.params.len() as i64);
    let result_reg =
        compile_expr_to_reg(&func.body, &var_map, &mut regs, local_labels, opcodes, asm, out)?;
    if result_reg != 0 {
        let op = encode_op("move", &[Arg::X(result_reg), Arg::X(0)], opcodes, asm)?;
        out.extend(op);
    }

    let op = encode_op("return", &[], opcodes, asm)?;
    out.extend(op);
    Ok(())
}

fn has_function(module: &Module, name: &str, arity: usize) -> bool {
    module
        .functions
        .iter()
        .any(|func| func.name == name && func.params.len() == arity)
}

fn has_export(exports: &[ExportSpec], name: &str, arity: usize) -> bool {
    exports
        .iter()
        .any(|export| export.name == name && export.arity == arity)
}

fn emit_module_info_0(
    module_name: &str,
    func_label: i64,
    entry_label: i64,
    opcodes: &HashMap<(String, usize), u8>,
    asm: &mut AsmBuilder,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let op = encode_op("label", &[Arg::U(func_label)], opcodes, asm)?;
    out.extend(op);

    let op = encode_op(
        "func_info",
        &[
            Arg::Atom(module_name.to_string()),
            Arg::Atom("module_info".to_string()),
            Arg::U(0),
        ],
        opcodes,
        asm,
    )?;
    out.extend(op);

    let op = encode_op("label", &[Arg::U(entry_label)], opcodes, asm)?;
    out.extend(op);

    let op = encode_op(
        "move",
        &[Arg::Atom(module_name.to_string()), Arg::X(0)],
        opcodes,
        asm,
    )?;
    out.extend(op);

    let op = encode_op(
        "call_ext_only",
        &[
            Arg::U(1),
            Arg::ExtFunc {
                module: "erlang".to_string(),
                function: "get_module_info".to_string(),
                arity: 1,
            },
        ],
        opcodes,
        asm,
    )?;
    out.extend(op);
    Ok(())
}

fn emit_module_info_1(
    module_name: &str,
    func_label: i64,
    entry_label: i64,
    opcodes: &HashMap<(String, usize), u8>,
    asm: &mut AsmBuilder,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let op = encode_op("label", &[Arg::U(func_label)], opcodes, asm)?;
    out.extend(op);

    let op = encode_op(
        "func_info",
        &[
            Arg::Atom(module_name.to_string()),
            Arg::Atom("module_info".to_string()),
            Arg::U(1),
        ],
        opcodes,
        asm,
    )?;
    out.extend(op);

    let op = encode_op("label", &[Arg::U(entry_label)], opcodes, asm)?;
    out.extend(op);

    let op = encode_op("move", &[Arg::X(0), Arg::X(1)], opcodes, asm)?;
    out.extend(op);

    let op = encode_op(
        "move",
        &[Arg::Atom(module_name.to_string()), Arg::X(0)],
        opcodes,
        asm,
    )?;
    out.extend(op);

    let op = encode_op(
        "call_ext_only",
        &[
            Arg::U(2),
            Arg::ExtFunc {
                module: "erlang".to_string(),
                function: "get_module_info".to_string(),
                arity: 2,
            },
        ],
        opcodes,
        asm,
    )?;
    out.extend(op);
    Ok(())
}

pub fn compile_module(module: &Module, opcodes: &HashMap<(String, usize), u8>) -> Result<Vec<u8>, String> {
    validate_module(module)?;
    let mut asm = AsmBuilder::new();
    asm.atom(&module.name);

    enum FunctionKind<'a> {
        User(&'a Function),
        ModuleInfo0,
        ModuleInfo1,
    }

    let mut code = Vec::new();
    let mut label = 1i64;
    let mut entry_labels = HashMap::new();
    let mut function_count = 0usize;

    let need_module_info_0 = !has_function(module, "module_info", 0);
    let need_module_info_1 = !has_function(module, "module_info", 1);

    let mut function_order = Vec::new();
    for func in &module.functions {
        let func_label = label;
        let entry_label = label + 1;
        label += 2;
        entry_labels.insert((func.name.clone(), func.params.len()), entry_label);
        function_order.push((FunctionKind::User(func), func_label, entry_label));
    }

    let mut exports = module.exports.clone();

    if need_module_info_0 {
        let func_label = label;
        let entry_label = label + 1;
        label += 2;
        entry_labels.insert(("module_info".to_string(), 0), entry_label);
        function_order.push((FunctionKind::ModuleInfo0, func_label, entry_label));
    }

    if need_module_info_1 {
        let func_label = label;
        let entry_label = label + 1;
        label += 2;
        entry_labels.insert(("module_info".to_string(), 1), entry_label);
        function_order.push((FunctionKind::ModuleInfo1, func_label, entry_label));
    }

    for (kind, func_label, entry_label) in function_order {
        match kind {
            FunctionKind::User(func) => {
                compile_function(
                    &module.name,
                    func,
                    func_label,
                    entry_label,
                    &entry_labels,
                    opcodes,
                    &mut asm,
                    &mut code,
                )?;
            }
            FunctionKind::ModuleInfo0 => {
                emit_module_info_0(
                    &module.name,
                    func_label,
                    entry_label,
                    opcodes,
                    &mut asm,
                    &mut code,
                )?;
            }
            FunctionKind::ModuleInfo1 => {
                emit_module_info_1(
                    &module.name,
                    func_label,
                    entry_label,
                    opcodes,
                    &mut asm,
                    &mut code,
                )?;
            }
        }
        function_count += 1;
    }

    if !has_export(&exports, "module_info", 0) {
        exports.push(ExportSpec {
            name: "module_info".to_string(),
            arity: 0,
        });
    }
    if !has_export(&exports, "module_info", 1) {
        exports.push(ExportSpec {
            name: "module_info".to_string(),
            arity: 1,
        });
    }

    for export in &exports {
        if let Some(label) = entry_labels.get(&(export.name.clone(), export.arity)) {
            asm.export(&export.name, export.arity as i32, *label as i32);
        } else {
            return Err(format!("exported function not found: {}/{}", export.name, export.arity));
        }
    }

    let op = encode_op("int_code_end", &[], opcodes, &mut asm)?;
    code.extend(op);

    asm.ensure_min_opcode(opcodes)?;
    if asm.atom_index(&module.name)? != 1 {
        return Err("module atom must be index 1".to_string());
    }

    let code_header = [
        (16u32).to_be_bytes(),
        (BEAM_FORMAT_NUMBER as i32).to_be_bytes(),
        (asm.highest_opcode as i32).to_be_bytes(),
        (label as i32).to_be_bytes(),
        (function_count as i32).to_be_bytes(),
    ]
    .concat();
    let code_chunk = chunk_with_head(b"Code", &code_header, &code);

    if code.is_empty() {
        return Err("code section is empty".to_string());
    }
    if code.last() != Some(&opcode_number("int_code_end", 0, opcodes)?) {
        return Err("code does not end with int_code_end".to_string());
    }

    let atom_chunk = chunk(b"AtU8", &build_atom_table(&asm.atoms));
    let imp_chunk = build_import_chunk(&asm)?;
    let exp_chunk = build_export_chunk(&asm);
    let loc_chunk = chunk_with_head(b"LocT", &(0i32).to_be_bytes(), &[]);
    let str_chunk = chunk(b"StrT", &[]);

    for (atom_index, _arity, label_value) in &asm.exports {
        if *atom_index == 0 {
            return Err("export atom index is zero".to_string());
        }
        if *label_value <= 0 || *label_value >= label as i32 {
            return Err("export label out of range".to_string());
        }
    }

    let mut chunks = Vec::new();
    chunks.extend(atom_chunk);
    chunks.extend(code_chunk);
    chunks.extend(str_chunk);
    chunks.extend(imp_chunk);
    chunks.extend(exp_chunk);
    chunks.extend(loc_chunk);

    if chunks.len() % 4 != 0 {
        return Err("chunk padding is misaligned".to_string());
    }

    let mut file = Vec::new();
    file.extend_from_slice(b"FOR1");
    file.extend_from_slice(&((chunks.len() + 4) as u32).to_be_bytes());
    file.extend_from_slice(b"BEAM");
    file.extend_from_slice(&chunks);
    Ok(file)
}

fn validate_module(module: &Module) -> Result<(), String> {
    if module.name.is_empty() {
        return Err("module name is empty".to_string());
    }
    if module.functions.is_empty() {
        return Err("module has no functions".to_string());
    }

    let mut seen_funcs = HashMap::new();
    for func in &module.functions {
        let key = (func.name.clone(), func.params.len());
        if seen_funcs.insert(key.clone(), true).is_some() {
            return Err(format!("duplicate function {}/{}", func.name, func.params.len()));
        }
        let mut seen_params = HashMap::new();
        for param in &func.params {
            if seen_params.insert(param.clone(), true).is_some() {
                return Err(format!("duplicate parameter {}", param));
            }
        }
    }

    let mut seen_exports = HashMap::new();
    for export in &module.exports {
        let key = (export.name.clone(), export.arity);
        if seen_exports.insert(key.clone(), true).is_some() {
            return Err(format!("duplicate export {}/{}", export.name, export.arity));
        }
        let defined = module
            .functions
            .iter()
            .any(|func| func.name == export.name && func.params.len() == export.arity);
        let is_module_info = export.name == "module_info"
            && (export.arity == 0 || export.arity == 1);
        if !defined && !is_module_info {
            return Err(format!(
                "exported function not defined: {}/{}",
                export.name, export.arity
            ));
        }
    }

    Ok(())
}

pub fn build_from_erl(path: &Path, opcodes: &HashMap<(String, usize), u8>) -> Result<PathBuf, String> {
    let contents = fs::read_to_string(path)
        .map_err(|e| format!("read failed: {}: {}", path.display(), e))?;
    let module = parse_module(&contents)?;
    let beam = compile_module(&module, opcodes)?;
    let mut out_path = path.to_path_buf();
    out_path.set_extension("beam");
    fs::write(&out_path, beam)
        .map_err(|e| format!("write failed: {}: {}", out_path.display(), e))?;
    Ok(out_path)
}
