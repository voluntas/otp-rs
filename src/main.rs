use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const TAG_U: u8 = 0;
const TAG_I: u8 = 1;
const TAG_A: u8 = 2;
const TAG_X: u8 = 3;
const TAG_Y: u8 = 4;
const TAG_F: u8 = 5;
const TAG_Z: u8 = 7;

const BEAM_FORMAT_NUMBER: i32 = 0;

#[derive(Debug)]
struct BeamChunk {
    id: [u8; 4],
    size: u32,
    data_start: usize,
}

#[derive(Debug)]
struct BeamCodeInfo {
    head_size: u32,
    version: i32,
    max_opcode: i32,
    label_count: i32,
    function_count: i32,
    code_offset: usize,
    code_size: usize,
}

#[derive(Debug)]
struct ImportEntry {
    module: String,
    function: String,
    arity: i32,
}

#[derive(Debug)]
struct ExportEntry {
    function: String,
    arity: i32,
    label: i32,
}

#[derive(Debug)]
struct LambdaEntry {
    function: String,
    arity: i32,
    label: i32,
    index: i32,
    num_free: i32,
    old_uniq: i32,
}

#[derive(Debug)]
struct LineInfo {
    version: i32,
    flags: i32,
    instr_count: i32,
    item_count: i32,
    name_count: i32,
    names: Vec<String>,
    preview: Vec<(i64, i64)>,
}

#[derive(Debug)]
struct LiteralInfo {
    uncompressed_size: i32,
    payload_size: usize,
}

#[derive(Debug)]
struct TaggedValue {
    tag: u8,
    value: i64,
}

#[derive(Debug, Clone)]
struct OpInfo {
    name: String,
    arity: usize,
}

#[derive(Debug)]
struct DecodedOp {
    opcode: u8,
    name: String,
    args: Vec<String>,
}

#[derive(Debug)]
#[allow(dead_code)]
enum Arg {
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
struct AsmBuilder {
    atoms: Vec<String>,
    atom_index: HashMap<String, u32>,
    imports: Vec<ImportEntry>,
    import_index: HashMap<(String, String, i32), u32>,
    exports: Vec<(u32, i32, i32)>,
    highest_opcode: u8,
}

impl AsmBuilder {
    fn new() -> Self {
        Self {
            atoms: Vec::new(),
            atom_index: HashMap::new(),
            imports: Vec::new(),
            import_index: HashMap::new(),
            exports: Vec::new(),
            highest_opcode: 0,
        }
    }

    fn atom(&mut self, name: &str) -> u32 {
        if let Some(index) = self.atom_index.get(name) {
            return *index;
        }
        let index = (self.atoms.len() + 1) as u32;
        self.atoms.push(name.to_string());
        self.atom_index.insert(name.to_string(), index);
        index
    }

    fn atom_index(&self, name: &str) -> Result<u32, String> {
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

    fn export(&mut self, name: &str, arity: i32, label: i32) {
        let index = self.atom(name);
        self.exports.push((index, arity, label));
    }

    fn record_opcode(&mut self, opcode: u8) {
        if opcode > self.highest_opcode {
            self.highest_opcode = opcode;
        }
    }

    fn ensure_min_opcode(
        &mut self,
        opcodes: &HashMap<(String, usize), u8>,
    ) -> Result<(), String> {
        let opcode = opcode_number("bs_create_bin", 6, opcodes)?;
        self.record_opcode(opcode);
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Module {
    name: String,
    exports: Vec<ExportSpec>,
    functions: Vec<Function>,
}

#[derive(Debug, Clone)]
struct ExportSpec {
    name: String,
    arity: usize,
}

#[derive(Debug, Clone)]
struct Function {
    name: String,
    params: Vec<String>,
    body: Expr,
}

#[derive(Debug, Clone)]
enum Expr {
    Atom(String),
    Integer(i64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Call {
        module: Option<String>,
        function: String,
        args: Vec<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Atom(String),
    Var(String),
    Integer(i64),
    Dash,
    Arrow,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Dot,
    Slash,
    Plus,
    Star,
    Colon,
}

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        if self.remaining() < 1 {
            return Err("unexpected end of data".to_string());
        }
        let val = self.data[self.offset];
        self.offset += 1;
        Ok(val)
    }

    fn read_u16_be(&mut self) -> Result<u16, String> {
        if self.remaining() < 2 {
            return Err("unexpected end of data".to_string());
        }
        let b0 = self.data[self.offset] as u16;
        let b1 = self.data[self.offset + 1] as u16;
        self.offset += 2;
        Ok((b0 << 8) | b1)
    }

    fn read_i32_be(&mut self) -> Result<i32, String> {
        if self.remaining() < 4 {
            return Err("unexpected end of data".to_string());
        }
        let b0 = self.data[self.offset] as i32;
        let b1 = self.data[self.offset + 1] as i32;
        let b2 = self.data[self.offset + 2] as i32;
        let b3 = self.data[self.offset + 3] as i32;
        self.offset += 4;
        Ok((b0 << 24) | (b1 << 16) | (b2 << 8) | b3)
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], String> {
        if self.remaining() < len {
            return Err("unexpected end of data".to_string());
        }
        let start = self.offset;
        self.offset += len;
        Ok(&self.data[start..start + len])
    }
}

fn read_tagged(reader: &mut Reader<'_>) -> Result<TaggedValue, String> {
    let len_code = reader.read_u8()?;
    let tag = len_code & 0x07;

    if (len_code & 0x08) == 0 {
        return Ok(TaggedValue {
            tag,
            value: (len_code >> 4) as i64,
        });
    }

    if (len_code & 0x10) == 0 {
        let extra = reader.read_u8()? as i64;
        let value = (((len_code >> 5) as i64) << 8) | extra;
        return Ok(TaggedValue { tag, value });
    }

    let size_code = len_code >> 5;
    let count = if size_code < 7 {
        (size_code as usize) + 2
    } else {
        let size_prefix = read_tagged(reader)?;
        if size_prefix.tag != TAG_U || size_prefix.value < 0 {
            return Err("invalid tagged size prefix".to_string());
        }
        size_prefix
            .value
            .checked_add(9)
            .ok_or_else(|| "tagged size overflow".to_string())? as usize
    };

    let data = reader.read_bytes(count)?;
    if count > 8 {
        return Err("tagged value too large".to_string());
    }

    let mut value: i64 = 0;
    for &b in data {
        value = (value << 8) | (b as i64);
    }

    if tag == TAG_I {
        let shift = (8 - count) * 8;
        value = (value << shift) >> shift;
    }

    Ok(TaggedValue { tag, value })
}

fn parse_beam_chunks(bytes: &[u8]) -> Result<Vec<BeamChunk>, String> {
    if bytes.len() < 12 {
        return Err("file too short".to_string());
    }
    if &bytes[0..4] != b"FORM" && &bytes[0..4] != b"FOR1" {
        return Err("missing FORM/FOR1 header".to_string());
    }
    let form_size = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if &bytes[8..12] != b"BEAM" {
        return Err("missing BEAM header".to_string());
    }

    let form_end = 8 + form_size;
    if form_end > bytes.len() {
        return Err("FORM size exceeds file length".to_string());
    }
    let mut offset = 12;
    let mut chunks = Vec::new();

    while offset + 8 <= bytes.len() && offset + 8 <= form_end {
        let id = [bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]];
        let size = u32::from_be_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        let data_start = offset + 8;
        let data_end = data_start + size as usize;
        if data_end > bytes.len() {
            return Err("chunk exceeds file length".to_string());
        }
        if data_end > form_end {
            return Err("chunk exceeds FORM size".to_string());
        }

        chunks.push(BeamChunk {
            id,
            size,
            data_start,
        });

        let padding = (4 - (size as usize % 4)) % 4;
        offset = data_end + padding;
    }

    Ok(chunks)
}

fn chunk_data<'a>(bytes: &'a [u8], chunk: &BeamChunk) -> Result<&'a [u8], String> {
    let start = chunk.data_start;
    let end = start + chunk.size as usize;
    if end > bytes.len() {
        return Err("chunk data exceeds file length".to_string());
    }
    Ok(&bytes[start..end])
}

fn latin1_to_string(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}

fn parse_atom_chunk(data: &[u8], utf8: bool) -> Result<Vec<String>, String> {
    let mut reader = Reader::new(data);
    let mut count = reader.read_i32_be()?;
    if count == i32::MIN {
        return Err("invalid atom count".to_string());
    }
    let long_counts = if count < 0 {
        count = -count;
        true
    } else {
        false
    };
    if count < 0 {
        return Err("invalid atom count".to_string());
    }

    let count = count as usize;
    let mut atoms = Vec::with_capacity(count + 1);
    atoms.push("<nil>".to_string());

    for _ in 0..count {
        let len = if long_counts {
            let tagged = read_tagged(&mut reader)?;
            if tagged.tag != TAG_U || tagged.value < 0 {
                return Err("invalid atom length tag".to_string());
            }
            tagged.value as usize
        } else {
            reader.read_u8()? as usize
        };

        let name_bytes = reader.read_bytes(len)?;
        let name = if utf8 {
            String::from_utf8_lossy(name_bytes).into_owned()
        } else {
            latin1_to_string(name_bytes)
        };
        atoms.push(name);
    }

    Ok(atoms)
}

fn atom_name(atoms: Option<&Vec<String>>, index: i32) -> String {
    if index < 0 {
        return format!("#{}", index);
    }
    let idx = index as usize;
    if let Some(table) = atoms {
        if let Some(name) = table.get(idx) {
            return name.clone();
        }
    }
    format!("#{}", index)
}

fn parse_import_chunk(data: &[u8], atoms: Option<&Vec<String>>) -> Result<Vec<ImportEntry>, String> {
    let mut reader = Reader::new(data);
    let count = reader.read_i32_be()?;
    if count < 0 {
        return Err("invalid import count".to_string());
    }

    let mut entries = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let module_index = reader.read_i32_be()?;
        let function_index = reader.read_i32_be()?;
        let arity = reader.read_i32_be()?;
        if module_index < 0 || function_index < 0 || arity < 0 {
            return Err("invalid import entry".to_string());
        }

        entries.push(ImportEntry {
            module: atom_name(atoms, module_index),
            function: atom_name(atoms, function_index),
            arity,
        });
    }

    Ok(entries)
}

fn parse_export_chunk(data: &[u8], atoms: Option<&Vec<String>>) -> Result<Vec<ExportEntry>, String> {
    let mut reader = Reader::new(data);
    let count = reader.read_i32_be()?;
    if count < 0 {
        return Err("invalid export count".to_string());
    }

    let mut entries = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let atom_index = reader.read_i32_be()?;
        let arity = reader.read_i32_be()?;
        let label = reader.read_i32_be()?;
        if atom_index < 0 || arity < 0 || label < 0 {
            return Err("invalid export entry".to_string());
        }

        entries.push(ExportEntry {
            function: atom_name(atoms, atom_index),
            arity,
            label,
        });
    }

    Ok(entries)
}

fn parse_lambda_chunk(data: &[u8], atoms: Option<&Vec<String>>) -> Result<Vec<LambdaEntry>, String> {
    let mut reader = Reader::new(data);
    let count = reader.read_i32_be()?;
    if count < 0 {
        return Err("invalid lambda count".to_string());
    }

    let mut entries = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let atom_index = reader.read_i32_be()?;
        let arity = reader.read_i32_be()?;
        let label = reader.read_i32_be()?;
        let fun_index = reader.read_i32_be()?;
        let num_free = reader.read_i32_be()?;
        let old_uniq = reader.read_i32_be()?;

        if atom_index < 0 || arity < 0 || label < 0 || fun_index < 0 || num_free < 0 {
            return Err("invalid lambda entry".to_string());
        }

        entries.push(LambdaEntry {
            function: atom_name(atoms, atom_index),
            arity,
            label,
            index: fun_index,
            num_free,
            old_uniq,
        });
    }

    Ok(entries)
}

fn parse_code_chunk(data: &[u8]) -> Result<BeamCodeInfo, String> {
    let mut reader = Reader::new(data);
    let head_size = reader.read_i32_be()?;
    if head_size < 0 {
        return Err("invalid code header size".to_string());
    }
    let head_size = head_size as usize;
    if head_size < 16 {
        return Err("code header too small".to_string());
    }
    if head_size > reader.remaining() {
        return Err("code header exceeds chunk size".to_string());
    }

    let version = reader.read_i32_be()?;
    let max_opcode = reader.read_i32_be()?;
    let label_count = reader.read_i32_be()?;
    let function_count = reader.read_i32_be()?;

    if version != BEAM_FORMAT_NUMBER {
        return Err("unexpected code format number".to_string());
    }
    if max_opcode < 0 || label_count < 0 || function_count < 0 {
        return Err("invalid code header values".to_string());
    }

    let code_offset = 4 + head_size;
    let code_size = data
        .len()
        .checked_sub(code_offset)
        .ok_or_else(|| "invalid code size".to_string())?;

    Ok(BeamCodeInfo {
        head_size: head_size as u32,
        version,
        max_opcode,
        label_count,
        function_count,
        code_offset,
        code_size,
    })
}

fn parse_line_chunk(data: &[u8]) -> Result<LineInfo, String> {
    let mut reader = Reader::new(data);
    let version = reader.read_i32_be()?;
    let flags = reader.read_i32_be()?;
    let instr_count = reader.read_i32_be()?;
    let item_count = reader.read_i32_be()?;
    let name_count = reader.read_i32_be()?;

    if item_count < 0 || name_count < 0 {
        return Err("invalid line table counts".to_string());
    }

    if version != 0 {
        return Ok(LineInfo {
            version,
            flags,
            instr_count,
            item_count,
            name_count,
            names: Vec::new(),
            preview: Vec::new(),
        });
    }

    let mut preview = Vec::new();
    let mut current_name = 0i64;
    let mut seen_items = 0i32;

    while seen_items < item_count {
        let tagged = read_tagged(&mut reader)?;
        match tagged.tag {
            TAG_A => {
                if tagged.value < 0 || tagged.value > name_count as i64 {
                    return Err("invalid line name index".to_string());
                }
                current_name = tagged.value;
            }
            TAG_I => {
                if preview.len() < 10 {
                    preview.push((current_name, tagged.value));
                }
                seen_items += 1;
            }
            _ => return Err("unexpected line item tag".to_string()),
        }
    }

    let mut names = Vec::with_capacity(name_count as usize);
    for _ in 0..name_count {
        let len = reader.read_u16_be()? as usize;
        let bytes = reader.read_bytes(len)?;
        names.push(String::from_utf8_lossy(bytes).into_owned());
    }

    Ok(LineInfo {
        version,
        flags,
        instr_count,
        item_count,
        name_count,
        names,
        preview,
    })
}

fn parse_literal_chunk(data: &[u8]) -> Result<LiteralInfo, String> {
    let mut reader = Reader::new(data);
    let uncompressed_size = reader.read_i32_be()?;
    if uncompressed_size < 0 {
        return Err("invalid literal size".to_string());
    }
    let payload_size = reader.remaining();
    Ok(LiteralInfo {
        uncompressed_size,
        payload_size,
    })
}

fn find_chunk<'a>(chunks: &'a [BeamChunk], id: &[u8; 4]) -> Option<&'a BeamChunk> {
    chunks.iter().find(|chunk| &chunk.id == id)
}

fn find_opcode_source() -> Option<String> {
    let candidates = [
        "lib/compiler/src/beam_opcodes.erl",
        "../lib/compiler/src/beam_opcodes.erl",
    ];
    for candidate in candidates {
        if Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }
    None
}

fn parse_opcode_line(line: &str) -> Option<(usize, OpInfo)> {
    let trimmed = line.trim();
    if trimmed.starts_with('%') {
        return None;
    }
    if !trimmed.starts_with("opcode(") {
        return None;
    }

    let after = &trimmed["opcode(".len()..];
    let comma = after.find(',')?;
    let name_raw = after[..comma].trim();
    let name = if name_raw.starts_with('\'') && name_raw.ends_with('\'') {
        name_raw.trim_matches('\'').to_string()
    } else {
        name_raw.to_string()
    };

    let after_comma = after[comma + 1..].trim();
    let paren = after_comma.find(')')?;
    let arity_str = after_comma[..paren].trim();
    let arity: usize = arity_str.parse().ok()?;

    let arrow = trimmed.find("->")?;
    let after_arrow = trimmed[arrow + 2..].trim();
    let mut digits = String::new();
    for ch in after_arrow.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            break;
        }
    }
    let opcode: usize = digits.parse().ok()?;

    Some((opcode, OpInfo { name, arity }))
}

fn insert_default_opcodes(map: &mut Vec<Option<OpInfo>>) {
    let defaults = [
        (1, "label", 1),
        (2, "func_info", 3),
        (3, "int_code_end", 0),
        (4, "call", 2),
        (5, "call_last", 3),
        (6, "call_only", 2),
        (7, "call_ext", 2),
        (8, "call_ext_last", 3),
        (12, "allocate", 2),
        (13, "allocate_heap", 3),
        (18, "deallocate", 1),
        (19, "return", 0),
        (61, "jump", 1),
        (64, "move", 2),
        (153, "line", 1),
    ];

    for (opcode, name, arity) in defaults {
        if opcode < map.len() {
            map[opcode] = Some(OpInfo {
                name: name.to_string(),
                arity,
            });
        }
    }
}

fn load_opcode_map() -> Vec<Option<OpInfo>> {
    let mut map = vec![None; 256];

    if let Some(path) = find_opcode_source() {
        if let Ok(contents) = fs::read_to_string(path) {
            for line in contents.lines() {
                if let Some((opcode, info)) = parse_opcode_line(line) {
                    if opcode < map.len() {
                        map[opcode] = Some(info);
                    }
                }
            }
        }
    }

    if map.iter().all(|entry| entry.is_none()) {
        insert_default_opcodes(&mut map);
    }

    map
}

fn build_reverse_opcode_map(opcodes: &[Option<OpInfo>]) -> HashMap<(String, usize), u8> {
    let mut map = HashMap::new();
    for (opcode, info) in opcodes.iter().enumerate() {
        if let Some(info) = info {
            map.insert((info.name.clone(), info.arity), opcode as u8);
        }
    }
    map
}

fn opcode_number(
    name: &str,
    arity: usize,
    map: &HashMap<(String, usize), u8>,
) -> Result<u8, String> {
    if let Some(opcode) = map.get(&(name.to_string(), arity)) {
        return Ok(*opcode);
    }

    let fallback = [
        ("label", 1, 1u8),
        ("func_info", 3, 2u8),
        ("int_code_end", 0, 3u8),
        ("call", 2, 4u8),
        ("call_last", 3, 5u8),
        ("call_only", 2, 6u8),
        ("call_ext", 2, 7u8),
        ("call_ext_last", 3, 8u8),
        ("allocate", 2, 12u8),
        ("allocate_heap", 3, 13u8),
        ("deallocate", 1, 18u8),
        ("return", 0, 19u8),
        ("jump", 1, 61u8),
        ("move", 2, 64u8),
        ("line", 1, 153u8),
    ];

    for (fname, farity, opcode) in fallback {
        if fname == name && farity == arity {
            return Ok(opcode);
        }
    }

    Err(format!("unknown opcode {} /{}", name, arity))
}

fn format_atom(atoms: Option<&Vec<String>>, index: i64) -> String {
    if index == 0 {
        return "nil".to_string();
    }
    if index < 0 {
        return format!("atom#{}", index);
    }
    let idx = index as usize;
    if let Some(table) = atoms {
        if let Some(name) = table.get(idx) {
            return name.clone();
        }
    }
    format!("atom#{}", index)
}

fn decode_extended_arg(
    reader: &mut Reader<'_>,
    extension: i64,
    total: &mut usize,
) -> Result<String, String> {
    match extension {
        1 => {
            let count = read_tagged(reader)?;
            if count.tag != TAG_U || count.value < 0 {
                return Err("invalid list extension".to_string());
            }
            *total += count.value as usize;
            Ok(format!("list({})", count.value))
        }
        2 => {
            let index = read_tagged(reader)?;
            if index.tag != TAG_U || index.value < 0 {
                return Err("invalid float register".to_string());
            }
            Ok(format!("fr({})", index.value))
        }
        3 => {
            let count = read_tagged(reader)?;
            if count.tag != TAG_U || count.value < 0 {
                return Err("invalid alloc list".to_string());
            }
            let mut parts = Vec::new();
            let entry_count = count.value as usize;
            for _ in 0..entry_count {
                let kind = read_tagged(reader)?;
                let number = read_tagged(reader)?;
                if kind.tag != TAG_U || number.tag != TAG_U || kind.value < 0 || number.value < 0 {
                    return Err("invalid alloc list entry".to_string());
                }
                let label = match kind.value {
                    0 => "words",
                    1 => "floats",
                    2 => "funs",
                    _ => "unknown",
                };
                parts.push(format!("{}={}", label, number.value));
            }
            Ok(format!("alloc({})", parts.join(",")))
        }
        4 => {
            let index = read_tagged(reader)?;
            if index.tag != TAG_U || index.value < 0 {
                return Err("invalid literal reference".to_string());
            }
            Ok(format!("lit({})", index.value))
        }
        5 => {
            let reg = read_tagged(reader)?;
            let type_index = read_tagged(reader)?;
            if type_index.tag != TAG_U || type_index.value < 0 {
                return Err("invalid type hint".to_string());
            }
            let reg_str = match reg.tag {
                TAG_X => format!("x({})", reg.value),
                TAG_Y => format!("y({})", reg.value),
                _ => return Err("invalid type hint register".to_string()),
            };
            Ok(format!("type({},{})", reg_str, type_index.value))
        }
        _ => Err("unknown extended tag".to_string()),
    }
}

fn decode_arg(
    reader: &mut Reader<'_>,
    atoms: Option<&Vec<String>>,
    total: &mut usize,
) -> Result<String, String> {
    let tagged = read_tagged(reader)?;
    match tagged.tag {
        TAG_U => Ok(format!("u({})", tagged.value)),
        TAG_I => Ok(format!("i({})", tagged.value)),
        TAG_A => Ok(format!("a({})", format_atom(atoms, tagged.value))),
        TAG_X => Ok(format!("x({})", tagged.value)),
        TAG_Y => Ok(format!("y({})", tagged.value)),
        TAG_F => {
            if tagged.value == 0 {
                Ok("p".to_string())
            } else {
                Ok(format!("f({})", tagged.value))
            }
        }
        TAG_Z => decode_extended_arg(reader, tagged.value, total),
        _ => Err("unknown tag".to_string()),
    }
}

fn decode_code(
    data: &[u8],
    opcodes: &[Option<OpInfo>],
    atoms: Option<&Vec<String>>,
    max_ops: usize,
) -> Result<Vec<DecodedOp>, String> {
    let mut reader = Reader::new(data);
    let mut ops = Vec::new();

    while reader.remaining() > 0 && ops.len() < max_ops {
        let opcode = reader.read_u8()?;
        let info = opcodes
            .get(opcode as usize)
            .and_then(|entry| entry.as_ref())
            .ok_or_else(|| format!("unknown opcode {}", opcode))?;

        let mut args = Vec::with_capacity(info.arity);
        let mut total = info.arity;
        let mut i = 0;
        while i < total {
            let arg = decode_arg(&mut reader, atoms, &mut total)?;
            args.push(arg);
            i += 1;
        }

        let op = DecodedOp {
            opcode,
            name: info.name.clone(),
            args,
        };
        let is_end = op.name == "int_code_end";
        ops.push(op);
        if is_end {
            break;
        }
    }

    Ok(ops)
}

fn encode_tag(tag: u8, value: i64) -> Vec<u8> {
    if value < 0 {
        return encode_tag_bytes(tag, &negative_to_bytes(value));
    }
    let uval = value as u64;
    if uval < 16 {
        return vec![((uval << 4) as u8) | tag];
    }
    if uval < 0x800 {
        let first = (((uval >> 3) & 0b1110_0000) as u8) | tag | 0b0000_1000;
        let second = (uval & 0xff) as u8;
        return vec![first, second];
    }
    encode_tag_bytes(tag, &to_bytes_unsigned(uval))
}

fn encode_tag_bytes(tag: u8, bytes: &[u8]) -> Vec<u8> {
    let count = bytes.len();
    if count >= 2 && count <= 8 {
        let prefix = (((count - 2) as u8) << 5) | 0b0001_1000 | tag;
        let mut out = Vec::with_capacity(1 + count);
        out.push(prefix);
        out.extend_from_slice(bytes);
        return out;
    }
    if count > 8 {
        let mut out = Vec::new();
        out.push(0b1111_1000 | tag);
        out.extend(encode_tag(TAG_U, (count - 9) as i64));
        out.extend_from_slice(bytes);
        return out;
    }
    Vec::new()
}

fn to_bytes_unsigned(value: u64) -> Vec<u8> {
    if value == 0 {
        return vec![0];
    }
    let mut bytes = Vec::new();
    let mut tmp = value;
    while tmp > 0 {
        bytes.push((tmp & 0xff) as u8);
        tmp >>= 8;
    }
    bytes.reverse();
    if bytes[0] & 0x80 != 0 {
        let mut prefixed = Vec::with_capacity(bytes.len() + 1);
        prefixed.push(0);
        prefixed.extend_from_slice(&bytes);
        return prefixed;
    }
    bytes
}

fn negative_to_bytes(value: i64) -> Vec<u8> {
    if value >= -0x8000 {
        return (value as i16).to_be_bytes().to_vec();
    }
    let abs = (-value) as u64;
    let mut bytes = Vec::new();
    let mut tmp = abs;
    while tmp > 0 {
        bytes.push((tmp & 0xff) as u8);
        tmp >>= 8;
    }
    bytes.reverse();

    let mut twos = vec![0u8; bytes.len()];
    let mut carry = 1u8;
    for i in (0..bytes.len()).rev() {
        let inv = !bytes[i];
        let (sum, overflow) = inv.overflowing_add(carry);
        twos[i] = sum;
        carry = if overflow { 1 } else { 0 };
    }
    if twos[0] & 0x80 == 0 {
        let mut prefixed = Vec::with_capacity(twos.len() + 1);
        prefixed.push(0xff);
        prefixed.extend_from_slice(&twos);
        return prefixed;
    }
    twos
}

fn chunk(id: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(id);
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(data);
    let padding = (4 - (data.len() % 4)) % 4;
    for _ in 0..padding {
        out.push(0);
    }
    out
}

fn chunk_with_head(id: &[u8; 4], head: &[u8], data: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(head.len() + data.len());
    payload.extend_from_slice(head);
    payload.extend_from_slice(data);
    chunk(id, &payload)
}

fn build_atom_table(atoms: &[String]) -> Vec<u8> {
    let mut data = Vec::new();
    let count = atoms.len() as i32;
    data.extend_from_slice(&(-count).to_be_bytes());
    for atom in atoms {
        let bytes = atom.as_bytes();
        data.extend(encode_tag(TAG_U, bytes.len() as i64));
        data.extend_from_slice(bytes);
    }
    data
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

fn encode_op(
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

fn compile_module(module: &Module, opcodes: &HashMap<(String, usize), u8>) -> Result<Vec<u8>, String> {
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

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch == '%' {
            while let Some(c) = chars.next() {
                if c == '\n' {
                    break;
                }
            }
            continue;
        }

        match ch {
            '-' => {
                chars.next();
                if chars.peek() == Some(&'>') {
                    chars.next();
                    tokens.push(Token::Arrow);
                } else {
                    tokens.push(Token::Dash);
                }
            }
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            '[' => {
                chars.next();
                tokens.push(Token::LBracket);
            }
            ']' => {
                chars.next();
                tokens.push(Token::RBracket);
            }
            ',' => {
                chars.next();
                tokens.push(Token::Comma);
            }
            '.' => {
                chars.next();
                tokens.push(Token::Dot);
            }
            '/' => {
                chars.next();
                tokens.push(Token::Slash);
            }
            '+' => {
                chars.next();
                tokens.push(Token::Plus);
            }
            '*' => {
                chars.next();
                tokens.push(Token::Star);
            }
            ':' => {
                chars.next();
                tokens.push(Token::Colon);
            }
            '0'..='9' => {
                let mut value: i64 = 0;
                while let Some(&digit) = chars.peek() {
                    if digit.is_ascii_digit() {
                        chars.next();
                        value = value
                            .checked_mul(10)
                            .and_then(|v| v.checked_add((digit as u8 - b'0') as i64))
                            .ok_or_else(|| "integer overflow".to_string())?;
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Integer(value));
            }
            _ => {
                if ch.is_ascii_alphabetic() || ch == '_' {
                    let mut ident = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_alphanumeric() || c == '_' || c == '@' {
                            ident.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    if ident.is_empty() {
                        return Err("empty identifier".to_string());
                    }
                    let first = ident.chars().next().unwrap();
                    if first.is_ascii_uppercase() || first == '_' {
                        tokens.push(Token::Var(ident));
                    } else {
                        tokens.push(Token::Atom(ident));
                    }
                } else {
                    return Err(format!("unexpected character '{}'", ch));
                }
            }
        }
    }

    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn expect(&mut self, expected: Token) -> Result<(), String> {
        match self.next() {
            Some(tok) if tok == expected => Ok(()),
            Some(tok) => Err(format!("unexpected token {:?}", tok)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn expect_atom(&mut self) -> Result<String, String> {
        match self.next() {
            Some(Token::Atom(name)) => Ok(name),
            Some(tok) => Err(format!("expected atom, got {:?}", tok)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn expect_var(&mut self) -> Result<String, String> {
        match self.next() {
            Some(Token::Var(name)) => Ok(name),
            Some(tok) => Err(format!("expected var, got {:?}", tok)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn expect_integer(&mut self) -> Result<i64, String> {
        match self.next() {
            Some(Token::Integer(value)) => Ok(value),
            Some(tok) => Err(format!("expected integer, got {:?}", tok)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn skip_to_dot(&mut self) {
        while let Some(tok) = self.next() {
            if tok == Token::Dot {
                break;
            }
        }
    }

    fn parse_module(&mut self) -> Result<Module, String> {
        let mut module_name: Option<String> = None;
        let mut exports: Vec<ExportSpec> = Vec::new();
        let mut functions: Vec<Function> = Vec::new();

        while !self.eof() {
            match self.peek() {
                Some(Token::Dash) => {
                    self.next();
                    let attr = self.expect_atom()?;
                    match attr.as_str() {
                        "module" => {
                            self.expect(Token::LParen)?;
                            let name = self.expect_atom()?;
                            self.expect(Token::RParen)?;
                            self.expect(Token::Dot)?;
                            module_name = Some(name);
                        }
                        "export" => {
                            self.expect(Token::LParen)?;
                            self.expect(Token::LBracket)?;
                            loop {
                                let name = self.expect_atom()?;
                                self.expect(Token::Slash)?;
                                let arity = self.expect_integer()? as usize;
                                exports.push(ExportSpec { name, arity });
                                match self.peek() {
                                    Some(Token::Comma) => {
                                        self.next();
                                        continue;
                                    }
                                    Some(Token::RBracket) => break,
                                    other => {
                                        return Err(format!(
                                            "unexpected token in export list: {:?}",
                                            other
                                        ))
                                    }
                                }
                            }
                            self.expect(Token::RBracket)?;
                            self.expect(Token::RParen)?;
                            self.expect(Token::Dot)?;
                        }
                        _ => {
                            self.skip_to_dot();
                        }
                    }
                }
                Some(_) => {
                    functions.push(self.parse_function()?);
                }
                None => break,
            }
        }

        let name = module_name.ok_or_else(|| "missing -module attribute".to_string())?;
        Ok(Module {
            name,
            exports,
            functions,
        })
    }

    fn parse_function(&mut self) -> Result<Function, String> {
        let name = self.expect_atom()?;
        self.expect(Token::LParen)?;
        let mut params = Vec::new();
        match self.peek() {
            Some(Token::RParen) => {
                self.next();
            }
            _ => {
                loop {
                    let param = self.expect_var()?;
                    params.push(param);
                    match self.peek() {
                        Some(Token::Comma) => {
                            self.next();
                            continue;
                        }
                        Some(Token::RParen) => {
                            self.next();
                            break;
                        }
                        other => {
                            return Err(format!("unexpected token in params: {:?}", other));
                        }
                    }
                }
            }
        }
        self.expect(Token::Arrow)?;
        let body = self.parse_expr()?;
        self.expect(Token::Dot)?;
        Ok(Function { name, params, body })
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_sum()
    }

    fn parse_sum(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_product()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.next();
                    let rhs = self.parse_product()?;
                    expr = Expr::Add(Box::new(expr), Box::new(rhs));
                }
                Some(Token::Dash) => {
                    self.next();
                    let rhs = self.parse_product()?;
                    expr = Expr::Sub(Box::new(expr), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_product(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_factor()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.next();
                    let rhs = self.parse_factor()?;
                    expr = Expr::Mul(Box::new(expr), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some(Token::Dash) => {
                self.next();
                let expr = self.parse_factor()?;
                Ok(Expr::Sub(Box::new(Expr::Integer(0)), Box::new(expr)))
            }
            _ => self.parse_term(),
        }
    }

    fn parse_term(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(Token::Atom(name)) => {
                match self.peek() {
                    Some(Token::Colon) => {
                        self.next();
                        let func = self.expect_atom()?;
                        let args = self.parse_call_args()?;
                        Ok(Expr::Call {
                            module: Some(name),
                            function: func,
                            args,
                        })
                    }
                    Some(Token::LParen) => {
                        let args = self.parse_call_args()?;
                        Ok(Expr::Call {
                            module: None,
                            function: name,
                            args,
                        })
                    }
                    _ => Ok(Expr::Atom(name)),
                }
            }
            Some(Token::Var(name)) => Ok(Expr::Var(name)),
            Some(Token::Integer(value)) => Ok(Expr::Integer(value)),
            Some(Token::LParen) => {
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Some(tok) => Err(format!("unexpected term token {:?}", tok)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn parse_call_args(&mut self) -> Result<Vec<Expr>, String> {
        self.expect(Token::LParen)?;
        let mut args = Vec::new();
        match self.peek() {
            Some(Token::RParen) => {
                self.next();
                return Ok(args);
            }
            _ => {}
        }
        loop {
            let expr = self.parse_expr()?;
            args.push(expr);
            match self.peek() {
                Some(Token::Comma) => {
                    self.next();
                    continue;
                }
                Some(Token::RParen) => {
                    self.next();
                    break;
                }
                other => {
                    return Err(format!("unexpected token in call args: {:?}", other));
                }
            }
        }
        Ok(args)
    }
}

fn parse_module(contents: &str) -> Result<Module, String> {
    let tokens = tokenize(contents)?;
    let mut parser = Parser::new(tokens);
    let module = parser.parse_module()?;
    if !parser.eof() {
        return Err("unexpected tokens after module".to_string());
    }
    Ok(module)
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

fn build_from_erl(path: &Path, opcodes: &HashMap<(String, usize), u8>) -> Result<PathBuf, String> {
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

fn main() {
    let mut args = env::args().skip(1);
    let first = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: otp-rs <path_to_beam>");
            eprintln!("       otp-rs build <path_to_erl>");
            std::process::exit(1);
        }
    };

    let opcode_map = load_opcode_map();
    let opcode_reverse = build_reverse_opcode_map(&opcode_map);

    if first == "build" {
        let erl_path = match args.next() {
            Some(p) => p,
            None => {
                eprintln!("usage: otp-rs build <path_to_erl>");
                std::process::exit(1);
            }
        };

        let path = Path::new(&erl_path);
        match build_from_erl(path, &opcode_reverse) {
            Ok(out_path) => {
                println!("built: {}", out_path.display());
            }
            Err(e) => {
                eprintln!("build failed: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    let path = first;

    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read failed: {}: {}", path, e);
            std::process::exit(1);
        }
    };

    let chunks = match parse_beam_chunks(&bytes) {
        Ok(chunks) => chunks,
        Err(e) => {
            eprintln!("parse failed: {}", e);
            std::process::exit(1);
        }
    };

    println!("BEAM chunks:");
    for chunk in &chunks {
        let id = String::from_utf8_lossy(&chunk.id);
        println!("- {} ({} bytes)", id, chunk.size);
    }

    let mut atoms: Option<Vec<String>> = None;
    let atom_chunk = find_chunk(&chunks, b"AtU8")
        .or_else(|| find_chunk(&chunks, b"Atom"))
        .or_else(|| find_chunk(&chunks, b"ATOM"));

    if let Some(chunk) = atom_chunk {
        match chunk_data(&bytes, chunk) {
            Ok(data) => {
                let utf8 = chunk.id == *b"AtU8";
                match parse_atom_chunk(data, utf8) {
                    Ok(parsed) => atoms = Some(parsed),
                    Err(e) => eprintln!("atom parse failed: {}", e),
                }
            }
            Err(e) => eprintln!("atom chunk error: {}", e),
        }
    }

    if let Some(atoms) = atoms.as_ref() {
        println!("Atoms: {}", atoms.len().saturating_sub(1));
        let preview_count = atoms.len().saturating_sub(1).min(10);
        if preview_count > 0 {
            println!("Atom preview:");
            for atom in atoms.iter().skip(1).take(preview_count) {
                println!("- {}", atom);
            }
        }
    }

    if let Some(chunk) = find_chunk(&chunks, b"Code") {
        match chunk_data(&bytes, chunk).and_then(parse_code_chunk) {
            Ok(info) => {
                println!("Code:");
                println!("- header bytes: {}", info.head_size);
                println!("- version: {}", info.version);
                println!("- max opcode: {}", info.max_opcode);
                println!("- labels: {}", info.label_count);
                println!("- functions: {}", info.function_count);
                println!("- code bytes: {}", info.code_size);

                if let Ok(data) = chunk_data(&bytes, chunk) {
                    if info.code_offset <= data.len() {
                        let code = &data[info.code_offset..];
                        match decode_code(code, &opcode_map, atoms.as_ref(), 30) {
                            Ok(ops) => {
                                println!("Code preview:");
                                for op in ops {
                                    let args = if op.args.is_empty() {
                                        String::new()
                                    } else {
                                        format!(" {}", op.args.join(", "))
                                    };
                                    println!("- {} ({}{})", op.name, op.opcode, args);
                                }
                            }
                            Err(e) => eprintln!("code decode failed: {}", e),
                        }
                    }
                }
            }
            Err(e) => eprintln!("code parse failed: {}", e),
        }
    }

    if let Some(chunk) = find_chunk(&chunks, b"ImpT") {
        match chunk_data(&bytes, chunk).and_then(|data| parse_import_chunk(data, atoms.as_ref())) {
            Ok(entries) => {
                println!("Imports: {}", entries.len());
                for entry in entries.iter().take(10) {
                    println!("- {}:{}/{}", entry.module, entry.function, entry.arity);
                }
            }
            Err(e) => eprintln!("import parse failed: {}", e),
        }
    }

    if let Some(chunk) = find_chunk(&chunks, b"ExpT") {
        match chunk_data(&bytes, chunk).and_then(|data| parse_export_chunk(data, atoms.as_ref())) {
            Ok(entries) => {
                println!("Exports: {}", entries.len());
                for entry in entries.iter().take(10) {
                    println!("- {}/{} -> label {}", entry.function, entry.arity, entry.label);
                }
            }
            Err(e) => eprintln!("export parse failed: {}", e),
        }
    }

    if let Some(chunk) = find_chunk(&chunks, b"FunT") {
        match chunk_data(&bytes, chunk).and_then(|data| parse_lambda_chunk(data, atoms.as_ref())) {
            Ok(entries) => {
                println!("Lambdas: {}", entries.len());
                for entry in entries.iter().take(10) {
                    println!(
                        "- {}/{} label {} index {} free {} uniq {}",
                        entry.function, entry.arity, entry.label, entry.index, entry.num_free, entry.old_uniq
                    );
                }
            }
            Err(e) => eprintln!("lambda parse failed: {}", e),
        }
    }

    if let Some(chunk) = find_chunk(&chunks, b"StrT") {
        match chunk_data(&bytes, chunk) {
            Ok(data) => println!("Strings: {} bytes", data.len()),
            Err(e) => eprintln!("string chunk error: {}", e),
        }
    }

    if let Some(chunk) = find_chunk(&chunks, b"LitT") {
        match chunk_data(&bytes, chunk).and_then(parse_literal_chunk) {
            Ok(info) => {
                if info.uncompressed_size == 0 {
                    println!("Literals: uncompressed (payload {} bytes)", info.payload_size);
                } else {
                    println!(
                        "Literals: compressed (payload {} bytes, expected {} bytes)",
                        info.payload_size, info.uncompressed_size
                    );
                }
            }
            Err(e) => eprintln!("literal parse failed: {}", e),
        }
    }

    if let Some(chunk) = find_chunk(&chunks, b"Line") {
        match chunk_data(&bytes, chunk).and_then(parse_line_chunk) {
            Ok(info) => {
                println!("Lines:");
                println!("- version: {}", info.version);
                println!("- flags: {}", info.flags);
                println!("- instructions: {}", info.instr_count);
                println!("- items: {}", info.item_count);
                println!("- names: {}", info.name_count);
                if !info.preview.is_empty() {
                    println!("Line preview:");
                    for (name_index, line) in info.preview.iter().take(10) {
                        let name = if *name_index == 0 {
                            "<module>".to_string()
                        } else {
                            let idx = (*name_index as usize).saturating_sub(1);
                            info.names
                                .get(idx)
                                .cloned()
                                .unwrap_or_else(|| format!("#{}", name_index))
                        };
                        println!("- {}:{}", name, line);
                    }
                }
            }
            Err(e) => eprintln!("line parse failed: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
