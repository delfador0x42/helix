//! Binary index v3 on-disk format. All structs repr(C, packed) for zero-copy access.

pub const MAGIC: [u8; 4] = *b"AMRN";
pub const VERSION: u32 = 3;

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u32,
    pub num_entries: u32,
    pub num_terms: u32,
    pub num_topics: u16,
    pub num_xrefs: u16,
    pub table_cap: u32,
    pub avgdl_x100: u32,
    pub postings_off: u32,
    pub meta_off: u32,
    pub snippet_off: u32,
    pub topics_off: u32,
    pub topic_names_off: u32,
    pub source_off: u32,
    pub xref_off: u32,
    pub total_len: u32,
    pub tag_names_off: u32,
    pub num_tags: u32,
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct TermSlot {
    pub hash: u64,
    pub postings_off: u32,
    pub postings_len: u32,
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct Posting {
    pub entry_id: u32,
    pub tf: u16,
    pub idf_x1000: u32,
    pub _pad: u16,
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct EntryMeta {
    pub topic_id: u16,
    pub word_count: u16,
    pub snippet_off: u32,
    pub snippet_len: u16,
    pub date_minutes: i32,
    pub source_off: u32,
    pub source_len: u16,
    pub log_offset: u32,
    pub tag_bitmap: u32,
    pub confidence: u8,
    pub epoch_days: u16,
    pub _pad: u8,
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct TopicEntry {
    pub name_off: u32,
    pub name_len: u16,
    pub entry_count: u16,
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct XrefEdge {
    pub src_topic: u16,
    pub dst_topic: u16,
    pub mention_count: u16,
    pub _pad: u16,
}

/// FNV-1a 64-bit hash for term table lookups.
pub fn hash_term(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
    if h == 0 { h = 1; }
    h
}

pub fn as_bytes<T: Sized>(val: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(val as *const T as *const u8, std::mem::size_of::<T>()) }
}
