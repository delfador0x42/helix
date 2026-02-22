//! Hand-rolled JSON parser + serializer. Zero dependencies.
//! Supports f64 numbers natively (needed for confidence values).
//! Single serialization path (Display impl) — no pretty printer.

use std::fmt;

#[derive(Clone)]
pub enum Value {
    Null, Bool(bool), Num(f64), Str(String), Arr(Vec<Value>), Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self { Value::Obj(p) => p.iter().find(|(k, _)| k == key).map(|(_, v)| v), _ => None }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { Value::Str(s) => Some(s), _ => None }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self { Value::Num(n) => Some(*n), _ => None }
    }
}

/// Escape string for JSON embedding (no surrounding quotes). Chunk-copy for speed.
pub fn escape_into(s: &str, buf: &mut String) {
    let bytes = s.as_bytes();
    let (mut i, mut last) = (0, 0);
    while i < bytes.len() {
        let esc = match bytes[i] {
            b'"' => "\\\"", b'\\' => "\\\\", b'\n' => "\\n", b'\r' => "\\r", b'\t' => "\\t",
            c if c < 0x20 => {
                if last < i { buf.push_str(&s[last..i]); }
                use fmt::Write; let _ = write!(buf, "\\u{:04x}", c);
                i += 1; last = i; continue;
            }
            _ => { i += 1; continue; }
        };
        if last < i { buf.push_str(&s[last..i]); }
        buf.push_str(esc); i += 1; last = i;
    }
    if last < bytes.len() { buf.push_str(&s[last..]); }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use fmt::Write;
        match self {
            Value::Null => f.write_str("null"),
            Value::Bool(b) => f.write_str(if *b { "true" } else { "false" }),
            Value::Num(n) => {
                if n.fract() == 0.0 && n.is_finite() { write!(f, "{}", *n as i64) }
                else { write!(f, "{n}") }
            }
            Value::Str(s) => { f.write_char('"')?; escape_fmt(s, f)?; f.write_char('"') }
            Value::Arr(items) => {
                f.write_char('[')?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 { f.write_char(',')?; } write!(f, "{v}")?;
                }
                f.write_char(']')
            }
            Value::Obj(pairs) => {
                f.write_char('{')?;
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 { f.write_char(',')?; }
                    f.write_char('"')?; escape_fmt(k, f)?; f.write_str("\":")?; write!(f, "{v}")?;
                }
                f.write_char('}')
            }
        }
    }
}

fn escape_fmt(s: &str, f: &mut fmt::Formatter) -> fmt::Result {
    let bytes = s.as_bytes();
    let (mut i, mut last) = (0, 0);
    while i < bytes.len() {
        let esc = match bytes[i] {
            b'"' => "\\\"", b'\\' => "\\\\", b'\n' => "\\n", b'\r' => "\\r", b'\t' => "\\t",
            c if c < 0x20 => {
                if last < i { f.write_str(&s[last..i])?; }
                write!(f, "\\u{:04x}", c)?; i += 1; last = i; continue;
            }
            _ => { i += 1; continue; }
        };
        if last < i { f.write_str(&s[last..i])?; }
        f.write_str(esc)?; i += 1; last = i;
    }
    if last < bytes.len() { f.write_str(&s[last..])?; }
    Ok(())
}

// --- Parser ---

pub fn parse(input: &str) -> Result<Value, String> {
    let mut p = Parser { b: input.as_bytes(), pos: 0 };
    p.value()
}

struct Parser<'a> { b: &'a [u8], pos: usize }

impl Parser<'_> {
    fn ws(&mut self) {
        while self.pos < self.b.len() && matches!(self.b[self.pos], b' '|b'\t'|b'\n'|b'\r') {
            self.pos += 1;
        }
    }
    fn peek(&self) -> Option<u8> { self.b.get(self.pos).copied() }
    fn next(&mut self) -> Result<u8, String> {
        self.b.get(self.pos).copied().map(|b| { self.pos += 1; b }).ok_or("unexpected end".into())
    }
    fn expect(&mut self, s: &[u8]) -> Result<(), String> {
        for &c in s { if self.next()? != c { return Err(format!("expected '{}'", c as char)); } }
        Ok(())
    }
    fn value(&mut self) -> Result<Value, String> {
        self.ws();
        match self.peek() {
            Some(b'"') => self.string().map(Value::Str),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b't') => { self.expect(b"true")?; Ok(Value::Bool(true)) }
            Some(b'f') => { self.expect(b"false")?; Ok(Value::Bool(false)) }
            Some(b'n') => { self.expect(b"null")?; Ok(Value::Null) }
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(format!("unexpected '{}'", c as char)),
            None => Err("unexpected end".into()),
        }
    }
    fn string(&mut self) -> Result<String, String> {
        self.pos += 1;
        let start = self.pos;
        let mut p = start;
        while p < self.b.len() {
            match self.b[p] {
                b'"' => {
                    let s = unsafe { std::str::from_utf8_unchecked(&self.b[start..p]) }.to_string();
                    self.pos = p + 1; return Ok(s);
                }
                b'\\' => break,
                _ => p += 1,
            }
        }
        let mut s = String::new();
        loop {
            match self.next()? {
                b'"' => return Ok(s),
                b'\\' => match self.next()? {
                    b'"' => s.push('"'), b'\\' => s.push('\\'), b'/' => s.push('/'),
                    b'n' => s.push('\n'), b'r' => s.push('\r'), b't' => s.push('\t'),
                    b'u' => {
                        let mut cp = 0u32;
                        for _ in 0..4 {
                            let h = self.next()?;
                            cp = cp * 16 + match h {
                                b'0'..=b'9' => (h - b'0') as u32,
                                b'a'..=b'f' => (h - b'a' + 10) as u32,
                                b'A'..=b'F' => (h - b'A' + 10) as u32,
                                _ => return Err("bad \\u hex".into()),
                            };
                        }
                        s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                    }
                    c => s.push(c as char),
                },
                b if b < 0x80 => s.push(b as char),
                b => {
                    let start = self.pos - 1;
                    let w = if b >= 0xF0 { 4 } else if b >= 0xE0 { 3 } else { 2 };
                    self.pos = (start + w).min(self.b.len());
                    if let Ok(u) = std::str::from_utf8(&self.b[start..self.pos]) { s.push_str(u); }
                }
            }
        }
    }
    fn number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') { self.pos += 1; }
        while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() { self.pos += 1; }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() { self.pos += 1; }
        }
        if matches!(self.peek(), Some(b'e'|b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+'|b'-')) { self.pos += 1; }
            while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() { self.pos += 1; }
        }
        std::str::from_utf8(&self.b[start..self.pos]).unwrap_or("0")
            .parse::<f64>().map(Value::Num).map_err(|e| e.to_string())
    }
    fn object(&mut self) -> Result<Value, String> {
        self.pos += 1; let mut pairs = Vec::new(); self.ws();
        if self.peek() == Some(b'}') { self.pos += 1; return Ok(Value::Obj(pairs)); }
        loop {
            self.ws(); let key = self.string()?; self.ws();
            if self.next()? != b':' { return Err("expected ':'".into()); }
            pairs.push((key, self.value()?)); self.ws();
            match self.next()? { b',' => continue, b'}' => return Ok(Value::Obj(pairs)),
                _ => return Err("expected ',' or '}'".into()) }
        }
    }
    fn array(&mut self) -> Result<Value, String> {
        self.pos += 1; let mut items = Vec::new(); self.ws();
        if self.peek() == Some(b']') { self.pos += 1; return Ok(Value::Arr(items)); }
        loop {
            items.push(self.value()?); self.ws();
            match self.next()? { b',' => continue, b']' => return Ok(Value::Arr(items)),
                _ => return Err("expected ',' or ']'".into()) }
        }
    }
}
