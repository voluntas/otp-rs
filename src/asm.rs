use std::collections::HashMap;

use crate::beam::{encode_tag, ImportEntry, TAG_A, TAG_F, TAG_I, TAG_U, TAG_X, TAG_Y};
use crate::opcodes::opcode_number;

#[derive(Debug)]
#[allow(dead_code)]
pub enum Arg {
    U(i64),
    I(i64),
    Atom(String),
    X(i64),
    Y(i64),
    F(i64),
    ExtFunc {
        module: String,
        function: String,
        arity: i64,
    },
}

#[derive(Debug)]
pub struct AsmBuilder {
    pub(crate) atoms: Vec<String>,
    atom_index: HashMap<String, u32>,
    pub(crate) imports: Vec<ImportEntry>,
    import_index: HashMap<(String, String, i32), u32>,
    pub(crate) exports: Vec<(u32, i32, i32)>,
    pub(crate) highest_opcode: u8,
}

impl AsmBuilder {
    pub fn new() -> Self {
        Self {
            atoms: Vec::new(),
            atom_index: HashMap::new(),
            imports: Vec::new(),
            import_index: HashMap::new(),
            exports: Vec::new(),
            highest_opcode: 0,
        }
    }

    pub fn atom(&mut self, name: &str) -> u32 {
        if let Some(index) = self.atom_index.get(name) {
            return *index;
        }
        let index = (self.atoms.len() + 1) as u32;
        self.atoms.push(name.to_string());
        self.atom_index.insert(name.to_string(), index);
        index
    }

    pub fn atom_index(&self, name: &str) -> Result<u32, String> {
        self.atom_index
            .get(name)
            .copied()
            .ok_or_else(|| format!("unknown atom {}", name))
    }

    fn import(&mut self, module: &str, function: &str, arity: i32) -> u32 {
        let key = (module.to_string(), function.to_string(), arity);
        if let Some(index) = self.import_index.get(&key) {
            return *index;
        }
        self.atom(module);
        self.atom(function);
        let index = self.imports.len() as u32;
        self.imports.push(ImportEntry {
            module: module.to_string(),
            function: function.to_string(),
            arity,
        });
        self.import_index.insert(key, index);
        index
    }

    pub fn export(&mut self, name: &str, arity: i32, label: i32) {
        let index = self.atom(name);
        self.exports.push((index, arity, label));
    }

    fn record_opcode(&mut self, opcode: u8) {
        if opcode > self.highest_opcode {
            self.highest_opcode = opcode;
        }
    }

    pub fn ensure_min_opcode(
        &mut self,
        opcodes: &HashMap<(String, usize), u8>,
    ) -> Result<(), String> {
        let opcode = opcode_number("bs_create_bin", 6, opcodes)?;
        self.record_opcode(opcode);
        Ok(())
    }
}

fn encode_arg(arg: &Arg, asm: &mut AsmBuilder) -> Result<Vec<u8>, String> {
    match arg {
        Arg::U(value) => Ok(encode_tag(TAG_U, *value)),
        Arg::I(value) => Ok(encode_tag(TAG_I, *value)),
        Arg::Atom(name) => Ok(encode_tag(TAG_A, asm.atom(name) as i64)),
        Arg::X(reg) => Ok(encode_tag(TAG_X, *reg)),
        Arg::Y(reg) => Ok(encode_tag(TAG_Y, *reg)),
        Arg::F(label) => Ok(encode_tag(TAG_F, *label)),
        Arg::ExtFunc {
            module,
            function,
            arity,
        } => {
            let index = asm.import(module, function, *arity as i32);
            Ok(encode_tag(TAG_U, index as i64))
        }
    }
}

pub fn encode_op(
    name: &str,
    args: &[Arg],
    opcodes: &HashMap<(String, usize), u8>,
    asm: &mut AsmBuilder,
) -> Result<Vec<u8>, String> {
    let opcode = opcode_number(name, args.len(), opcodes)?;
    asm.record_opcode(opcode);
    let mut out = Vec::new();
    out.push(opcode);
    for arg in args {
        out.extend(encode_arg(arg, asm)?);
    }
    Ok(out)
}
