use crate::opcodes::OpInfo;

pub const TAG_U: u8 = 0;
pub const TAG_I: u8 = 1;
pub const TAG_A: u8 = 2;
pub const TAG_X: u8 = 3;
pub const TAG_Y: u8 = 4;
pub const TAG_F: u8 = 5;
pub const TAG_Z: u8 = 7;

pub const BEAM_FORMAT_NUMBER: i32 = 0;

#[derive(Debug)]
pub struct BeamChunk {
    pub id: [u8; 4],
    pub size: u32,
    pub data_start: usize,
}

#[derive(Debug)]
pub struct BeamCodeInfo {
    pub head_size: u32,
    pub version: i32,
    pub max_opcode: i32,
    pub label_count: i32,
    pub function_count: i32,
    pub code_offset: usize,
    pub code_size: usize,
}

#[derive(Debug)]
pub struct ImportEntry {
    pub module: String,
    pub function: String,
    pub arity: i32,
}

#[derive(Debug)]
pub struct ExportEntry {
    pub function: String,
    pub arity: i32,
    pub label: i32,
}

#[derive(Debug)]
pub struct LambdaEntry {
    pub function: String,
    pub arity: i32,
    pub label: i32,
    pub index: i32,
    pub num_free: i32,
    pub old_uniq: i32,
}

#[derive(Debug)]
pub struct LineInfo {
    pub version: i32,
    pub flags: i32,
    pub instr_count: i32,
    pub item_count: i32,
    pub name_count: i32,
    pub names: Vec<String>,
    pub preview: Vec<(i64, i64)>,
}

#[derive(Debug)]
pub struct LiteralInfo {
    pub uncompressed_size: i32,
    pub payload_size: usize,
}

#[derive(Debug)]
pub struct TaggedValue {
    pub tag: u8,
    pub value: i64,
}

#[derive(Debug)]
pub struct DecodedOp {
    pub opcode: u8,
    pub name: String,
    pub args: Vec<String>,
}

pub struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
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

pub fn read_tagged(reader: &mut Reader<'_>) -> Result<TaggedValue, String> {
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

pub fn parse_beam_chunks(bytes: &[u8]) -> Result<Vec<BeamChunk>, String> {
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

pub fn chunk_data<'a>(bytes: &'a [u8], chunk: &BeamChunk) -> Result<&'a [u8], String> {
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

pub fn parse_atom_chunk(data: &[u8], utf8: bool) -> Result<Vec<String>, String> {
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

pub fn parse_import_chunk(data: &[u8], atoms: Option<&Vec<String>>) -> Result<Vec<ImportEntry>, String> {
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

pub fn parse_export_chunk(data: &[u8], atoms: Option<&Vec<String>>) -> Result<Vec<ExportEntry>, String> {
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

pub fn parse_lambda_chunk(data: &[u8], atoms: Option<&Vec<String>>) -> Result<Vec<LambdaEntry>, String> {
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

pub fn parse_code_chunk(data: &[u8]) -> Result<BeamCodeInfo, String> {
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

pub fn parse_line_chunk(data: &[u8]) -> Result<LineInfo, String> {
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

pub fn parse_literal_chunk(data: &[u8]) -> Result<LiteralInfo, String> {
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

pub fn find_chunk<'a>(chunks: &'a [BeamChunk], id: &[u8; 4]) -> Option<&'a BeamChunk> {
    chunks.iter().find(|chunk| &chunk.id == id)
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

pub fn decode_code(
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

pub fn encode_tag(tag: u8, value: i64) -> Vec<u8> {
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

pub fn chunk(id: &[u8; 4], data: &[u8]) -> Vec<u8> {
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

pub fn chunk_with_head(id: &[u8; 4], head: &[u8], data: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(head.len() + data.len());
    payload.extend_from_slice(head);
    payload.extend_from_slice(data);
    chunk(id, &payload)
}

pub fn build_atom_table(atoms: &[String]) -> Vec<u8> {
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
