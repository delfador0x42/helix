//! Exclusive file lock on the data directory. Released on drop.

use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::path::Path;

extern "C" { fn flock(fd: i32, op: i32) -> i32; }

pub struct FileLock { _file: File }

impl FileLock {
    pub fn acquire(dir: &Path) -> Result<Self, String> {
        let file = OpenOptions::new().create(true).write(true)
            .open(dir.join(".lock")).map_err(|e| format!("lock: {e}"))?;
        if unsafe { flock(file.as_raw_fd(), 2) } != 0 {
            return Err("failed to acquire lock".into());
        }
        Ok(FileLock { _file: file })
    }
}
