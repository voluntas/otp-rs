use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct OpInfo {
    pub name: String,
    pub arity: usize,
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

pub fn load_opcode_map() -> Vec<Option<OpInfo>> {
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

pub fn build_reverse_opcode_map(opcodes: &[Option<OpInfo>]) -> HashMap<(String, usize), u8> {
    let mut map = HashMap::new();
    for (opcode, info) in opcodes.iter().enumerate() {
        if let Some(info) = info {
            map.insert((info.name.clone(), info.arity), opcode as u8);
        }
    }
    map
}

pub fn opcode_number(
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
