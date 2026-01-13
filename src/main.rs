use std::env;
use std::fs;

use otp_rs::beam::{
    chunk_data, decode_code, find_chunk, parse_atom_chunk, parse_beam_chunks, parse_code_chunk,
    parse_export_chunk, parse_import_chunk, parse_lambda_chunk, parse_line_chunk,
    parse_literal_chunk,
};
use otp_rs::compile::build_from_erl;
use otp_rs::opcodes::{build_reverse_opcode_map, load_opcode_map};

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
                eprintln!("missing .erl path");
                std::process::exit(1);
            }
        };

        match build_from_erl(erl_path.as_ref(), &opcode_reverse) {
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

    if let Some(chunk) = find_chunk(&chunks, b"LocT") {
        match chunk_data(&bytes, chunk).and_then(|data| parse_export_chunk(data, atoms.as_ref())) {
            Ok(entries) => {
                println!("Locals: {}", entries.len());
                for entry in entries.iter().take(10) {
                    println!("- {}/{} -> label {}", entry.function, entry.arity, entry.label);
                }
            }
            Err(e) => eprintln!("local parse failed: {}", e),
        }
    }

    if let Some(chunk) = find_chunk(&chunks, b"FunT") {
        match chunk_data(&bytes, chunk).and_then(|data| parse_lambda_chunk(data, atoms.as_ref())) {
            Ok(entries) => {
                println!("Lambdas: {}", entries.len());
                for entry in entries.iter().take(10) {
                    println!(
                        "- {}/{} -> label {} (index {}, free {})",
                        entry.function, entry.arity, entry.label, entry.index, entry.num_free
                    );
                }
            }
            Err(e) => eprintln!("lambda parse failed: {}", e),
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
