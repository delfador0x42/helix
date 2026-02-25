//! Index readers: extract structured data from the binary inverted index.

use crate::format::*;
use crate::index::{read_header, read_at};

pub fn topic_table(data: &[u8]) -> Result<Vec<(u16, String, u16)>, String> {
    let hdr = read_header(data)?;
    let top_off = { hdr.topics_off } as usize;
    let tname_off = { hdr.topic_names_off } as usize;
    let n = { hdr.num_topics } as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let te = read_at::<TopicEntry>(data, top_off + i * std::mem::size_of::<TopicEntry>())?;
        let no = tname_off + { te.name_off } as usize;
        let nl = { te.name_len } as usize;
        let name = if no + nl <= data.len() {
            std::str::from_utf8(&data[no..no + nl]).unwrap_or("?").to_string()
        } else { "?".into() };
        out.push((i as u16, name, { te.entry_count }));
    }
    Ok(out)
}

pub fn topic_recency(data: &[u8]) -> Vec<(u16, i32)> {
    let hdr = match read_header(data) { Ok(h) => h, Err(_) => return vec![] };
    let meta_off = { hdr.meta_off } as usize;
    let n = { hdr.num_entries } as usize;
    let ntop = { hdr.num_topics } as usize;
    let meta_size = std::mem::size_of::<EntryMeta>();
    let mut latest = vec![i32::MIN; ntop];
    for i in 0..n {
        if let Ok(m) = read_at::<EntryMeta>(data, meta_off + i * meta_size) {
            let tid = { m.topic_id } as usize;
            let dm = { m.date_minutes };
            if tid < ntop && dm > latest[tid] { latest[tid] = dm; }
        }
    }
    latest.into_iter().enumerate()
        .filter(|(_, d)| *d > i32::MIN)
        .map(|(i, d)| (i as u16, d)).collect()
}

pub fn topic_name(data: &[u8], topic_id: u16) -> Result<String, String> {
    let hdr = read_header(data)?;
    let top_off = { hdr.topics_off } as usize;
    let tname_off = { hdr.topic_names_off } as usize;
    if topic_id as usize >= { hdr.num_topics } as usize { return Err("topic_id out of range".into()); }
    let te = read_at::<TopicEntry>(data, top_off + topic_id as usize * std::mem::size_of::<TopicEntry>())?;
    let no = tname_off + { te.name_off } as usize;
    let nl = { te.name_len } as usize;
    if no + nl > data.len() { return Err("name out of bounds".into()); }
    Ok(std::str::from_utf8(&data[no..no + nl]).unwrap_or("?").to_string())
}

pub fn resolve_topic(data: &[u8], name: &str) -> Option<u16> {
    let hdr = read_header(data).ok()?;
    let top_off = { hdr.topics_off } as usize;
    let tname_off = { hdr.topic_names_off } as usize;
    let ntop = { hdr.num_topics } as usize;
    let nb = name.as_bytes();
    for i in 0..ntop {
        let te = read_at::<TopicEntry>(data, top_off + i * std::mem::size_of::<TopicEntry>()).ok()?;
        let no = tname_off + { te.name_off } as usize;
        let nl = { te.name_len } as usize;
        if nl == nb.len() && no + nl <= data.len() && &data[no..no + nl] == nb {
            return Some(i as u16);
        }
    }
    None
}

pub fn resolve_tag(data: &[u8], tag_name: &str) -> Option<u8> {
    let hdr = read_header(data).ok()?;
    let off = { hdr.tag_names_off } as usize;
    if off >= data.len() { return None; }
    let count = data[off] as usize;
    let mut pos = off + 1;
    let lower = tag_name.to_lowercase();
    for bit in 0..count {
        if pos >= data.len() { return None; }
        let len = data[pos] as usize;
        pos += 1;
        if pos + len > data.len() { return None; }
        if std::str::from_utf8(&data[pos..pos + len]).ok() == Some(&lower) { return Some(bit as u8); }
        pos += len;
    }
    None
}

pub fn reconstruct_tags(data: &[u8], entry_id: u32) -> Result<Option<String>, String> {
    let hdr = read_header(data)?;
    let meta_off = { hdr.meta_off } as usize;
    if entry_id as usize >= { hdr.num_entries } as usize { return Err("entry_id out of range".into()); }
    let m = read_at::<EntryMeta>(data, meta_off + entry_id as usize * std::mem::size_of::<EntryMeta>())?;
    let bitmap = { m.tag_bitmap };
    if bitmap == 0 { return Ok(None); }
    let tag_names = read_tag_names(data, &hdr)?;
    let mut out = String::with_capacity(64);
    out.push_str("[tags: ");
    let mut first = true;
    for (bit, name) in tag_names.iter().enumerate() {
        if bitmap & (1u32 << bit) != 0 {
            if !first { out.push_str(", "); }
            out.push_str(name);
            first = false;
        }
    }
    if first { return Ok(None); }
    out.push(']');
    Ok(Some(out))
}

fn read_tag_names(data: &[u8], hdr: &Header) -> Result<Vec<String>, String> {
    let off = { hdr.tag_names_off } as usize;
    if off >= data.len() { return Ok(Vec::new()); }
    let count = data[off] as usize;
    let mut pos = off + 1;
    let mut names = Vec::with_capacity(count);
    for _ in 0..count {
        if pos >= data.len() { break; }
        let len = data[pos] as usize;
        pos += 1;
        if pos + len > data.len() { break; }
        names.push(std::str::from_utf8(&data[pos..pos + len]).unwrap_or("").to_string());
        pos += len;
    }
    Ok(names)
}

pub fn xref_edges(data: &[u8]) -> Result<Vec<(u16, u16, u16)>, String> {
    let hdr = read_header(data)?;
    let off = { hdr.xref_off } as usize;
    let n = { hdr.num_xrefs } as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = read_at::<XrefEdge>(data, off + i * std::mem::size_of::<XrefEdge>())?;
        out.push(({ x.src_topic }, { x.dst_topic }, { x.mention_count }));
    }
    Ok(out)
}

pub fn entry_source_ref(data: &[u8], entry_id: u32) -> Option<&str> {
    let hdr = read_header(data).ok()?;
    let meta_off = { hdr.meta_off } as usize;
    let src_off = { hdr.source_off } as usize;
    if entry_id as usize >= { hdr.num_entries } as usize { return None; }
    let m = read_at::<EntryMeta>(data, meta_off + entry_id as usize * std::mem::size_of::<EntryMeta>()).ok()?;
    let sl = { m.source_len } as usize;
    if sl == 0 { return None; }
    let so = src_off + { m.source_off } as usize;
    if so + sl > data.len() { return None; }
    std::str::from_utf8(&data[so..so + sl]).ok()
}

pub fn entry_snippet_ref(data: &[u8], entry_id: u32) -> Result<&str, String> {
    let hdr = read_header(data)?;
    let meta_off = { hdr.meta_off } as usize;
    let snip_off = { hdr.snippet_off } as usize;
    if entry_id as usize >= { hdr.num_entries } as usize { return Err("entry_id out of range".into()); }
    let m = read_at::<EntryMeta>(data, meta_off + entry_id as usize * std::mem::size_of::<EntryMeta>())?;
    let so = snip_off + { m.snippet_off } as usize;
    let sl = { m.snippet_len } as usize;
    if so + sl > data.len() { return Ok(""); }
    Ok(std::str::from_utf8(&data[so..so + sl]).unwrap_or(""))
}

pub fn entry_topic_id(data: &[u8], entry_id: u32) -> Result<u16, String> {
    let hdr = read_header(data)?;
    let meta_off = { hdr.meta_off } as usize;
    if entry_id as usize >= { hdr.num_entries } as usize { return Err("entry_id out of range".into()); }
    let m = read_at::<EntryMeta>(data, meta_off + entry_id as usize * std::mem::size_of::<EntryMeta>())?;
    Ok(m.topic_id)
}

pub fn sourced_entries(data: &[u8]) -> Result<Vec<(u32, u16, String, i32)>, String> {
    let hdr = read_header(data)?;
    let meta_off = { hdr.meta_off } as usize;
    let src_off = { hdr.source_off } as usize;
    let n = { hdr.num_entries } as usize;
    let mut out = Vec::new();
    for i in 0..n {
        let m = read_at::<EntryMeta>(data, meta_off + i * std::mem::size_of::<EntryMeta>())?;
        let sl = { m.source_len } as usize;
        if sl == 0 { continue; }
        let so = src_off + { m.source_off } as usize;
        if so + sl > data.len() { continue; }
        let path = std::str::from_utf8(&data[so..so + sl]).unwrap_or("").to_string();
        out.push((i as u32, { m.topic_id }, path, { m.date_minutes }));
    }
    Ok(out)
}

pub fn index_info(data: &[u8]) -> Result<String, String> {
    let hdr = read_header(data)?;
    Ok(format!("index v3: {} entries, {} terms, {} topics, {} xrefs, {} tags, {} bytes",
        { hdr.num_entries }, { hdr.num_terms }, { hdr.num_topics },
        { hdr.num_xrefs }, { hdr.num_tags }, { hdr.total_len }))
}
