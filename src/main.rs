//! wheelfs: convert FreeBSD python pkgs into wheels.
//!
//! A FreeBSD python pkg's payload is an unpacked wheel: the site-packages
//! tree with its dist-info (METADATA/RECORD/WHEEL) intact. Conversion is
//! extraction + re-zip: pull the site-packages payload out of the tar.zst,
//! regenerate RECORD, and stamp the local machine's platform tag so the
//! wheel matches what uv/pip compute for this host (FreeBSD tags embed the
//! patch level, so pre-built tags rarely match the client without this).
//!
//! Design + roadmap: bead mu-vwp5.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};

#[derive(Parser)]
#[command(name = "wheelfs", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert a FreeBSD .pkg file into a wheel.
    Convert {
        /// Path to the .pkg file (tar.zst).
        pkg: PathBuf,
        /// Directory to write the wheel into (default: current directory).
        #[arg(short, long, default_value = ".")]
        outdir: PathBuf,
        /// Platform tag to stamp (default: computed for this machine).
        #[arg(long)]
        platform_tag: Option<String>,
        /// Keep __pycache__/.pyc files (wheels normally omit them).
        #[arg(long)]
        keep_pyc: bool,
    },
    /// Print the platform tag this machine expects in wheel filenames.
    PlatformTag,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Cmd::Convert { pkg, outdir, platform_tag, keep_pyc } => {
            let tag = match platform_tag {
                Some(t) => t,
                None => local_platform_tag()?,
            };
            let out = convert(&pkg, &outdir, &tag, keep_pyc)?;
            println!("{}", out.display());
            Ok(())
        }
        Cmd::PlatformTag => {
            println!("{}", local_platform_tag()?);
            Ok(())
        }
    }
}

/// The platform tag uv/pip compute on this host: sysconfig.get_platform()
/// with [.-] mapped to underscores and lowercased, e.g.
/// `freebsd_15_1_release_p1_amd64`.
fn local_platform_tag() -> Result<String> {
    let out = |args: &[&str]| -> Result<String> {
        let o = Command::new("uname").args(args).output()?;
        Ok(String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let (s, r, m) = (out(&["-s"])?, out(&["-r"])?, out(&["-m"])?);
    if s.is_empty() || r.is_empty() || m.is_empty() {
        bail!("uname did not return system/release/machine");
    }
    Ok(format!("{s}-{r}-{m}").to_lowercase().replace(['.', '-'], "_"))
}

struct Payload {
    /// site-packages-relative path -> (mode, content)
    files: BTreeMap<String, (u32, Vec<u8>)>,
    /// paths in the pkg payload that are not under site-packages
    skipped: Vec<String>,
}

fn read_pkg_payload(pkg: &PathBuf, keep_pyc: bool) -> Result<Payload> {
    let mut f = File::open(pkg).with_context(|| format!("open {}", pkg.display()))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    f.seek(SeekFrom::Start(0))?;
    if magic != [0x28, 0xB5, 0x2F, 0xFD] {
        bail!(
            "{}: not a zstd-compressed pkg (magic {:02x?}); only tar.zst pkgs are supported",
            pkg.display(),
            magic
        );
    }
    let dec = zstd::stream::read::Decoder::new(f)?;
    let mut ar = tar::Archive::new(dec);

    let mut files: BTreeMap<String, (u32, Vec<u8>)> = BTreeMap::new();
    let mut skipped = Vec::new();

    for entry in ar.entries()? {
        let mut entry = entry?;
        let raw = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
        // pkg metadata files (+MANIFEST, +COMPACT_MANIFEST) are not payload.
        if raw.starts_with('+') {
            continue;
        }
        let Some(rel) = raw.split_once("/site-packages/").map(|(_, r)| r.to_string()) else {
            if entry.header().entry_type().is_file() {
                skipped.push(raw);
            }
            continue;
        };
        if rel.is_empty() {
            continue;
        }
        if !keep_pyc && (rel.contains("__pycache__/") || rel.ends_with(".pyc")) {
            continue;
        }
        let mode = entry.header().mode().unwrap_or(0o644);
        match entry.header().entry_type() {
            tar::EntryType::Regular => {
                let mut buf = Vec::with_capacity(entry.size() as usize);
                entry.read_to_end(&mut buf)?;
                files.insert(rel, (mode, buf));
            }
            // pkg tars use hardlinks for duplicate files; resolve against
            // the already-read target.
            tar::EntryType::Link => {
                let target = entry
                    .link_name_bytes()
                    .map(|b| String::from_utf8_lossy(&b).into_owned())
                    .unwrap_or_default();
                let Some(trel) = target.split_once("/site-packages/").map(|(_, r)| r.to_string())
                else {
                    skipped.push(format!("{raw} (hardlink -> {target})"));
                    continue;
                };
                match files.get(&trel) {
                    Some(v) => {
                        let v = v.clone();
                        files.insert(rel, v);
                    }
                    None => skipped.push(format!("{raw} (hardlink -> unread {target})")),
                }
            }
            tar::EntryType::Directory => {}
            other => skipped.push(format!("{raw} ({other:?})")),
        }
    }
    Ok(Payload { files, skipped })
}

/// Rewrite WHEEL's Tag lines, replacing the platform component with ours.
/// Returns (new WHEEL content, tags for the filename).
fn restamp_wheel(content: &str, platform_tag: &str) -> Result<(String, String)> {
    let mut out = String::new();
    let mut tags: Vec<String> = Vec::new();
    for line in content.lines() {
        if let Some(tag) = line.strip_prefix("Tag: ") {
            let parts: Vec<&str> = tag.trim().split('-').collect();
            if parts.len() != 3 {
                bail!("unparseable WHEEL tag: {tag}");
            }
            let plat = if parts[2] == "any" { parts[2] } else { platform_tag };
            let new = format!("{}-{}-{}", parts[0], parts[1], plat);
            out.push_str(&format!("Tag: {new}\n"));
            if !tags.contains(&new) {
                tags.push(new);
            }
        } else if !line.is_empty() {
            out.push_str(line);
            out.push('\n');
        }
    }
    if tags.is_empty() {
        bail!("WHEEL file has no Tag lines");
    }
    // Compound filename tag: py.py-abi.abi-plat.plat per PEP 425.
    let join = |i: usize| -> String {
        let mut seen: Vec<&str> = Vec::new();
        for t in &tags {
            let c = t.split('-').nth(i).unwrap();
            if !seen.contains(&c) {
                seen.push(c);
            }
        }
        seen.join(".")
    };
    let fname_tag = format!("{}-{}-{}", join(0), join(1), join(2));
    Ok((out, fname_tag))
}

fn convert(pkg: &PathBuf, outdir: &PathBuf, platform_tag: &str, keep_pyc: bool) -> Result<PathBuf> {
    let payload = read_pkg_payload(pkg, keep_pyc)?;
    if payload.files.is_empty() {
        bail!("{}: no site-packages payload found; not a python pkg?", pkg.display());
    }

    // Locate exactly one dist-info directory.
    let mut distinfos: Vec<String> = payload
        .files
        .keys()
        .filter_map(|p| p.split_once('/').map(|(d, _)| d))
        .filter(|d| d.ends_with(".dist-info"))
        .map(|d| d.to_string())
        .collect();
    distinfos.dedup();
    let distinfo = match distinfos.as_slice() {
        [one] => one.clone(),
        [] => bail!("no .dist-info directory in payload"),
        many => bail!("multiple .dist-info directories: {many:?}"),
    };
    let (name, version) = distinfo
        .trim_end_matches(".dist-info")
        .rsplit_once('-')
        .context("dist-info dirname is not {name}-{version}")?;

    let wheel_src = payload
        .files
        .get(&format!("{distinfo}/WHEEL"))
        .context("payload has no dist-info/WHEEL")?;
    let (wheel_meta, fname_tag) = restamp_wheel(&String::from_utf8_lossy(&wheel_src.1), platform_tag)?;

    let wheel_name = format!("{}-{}-{}.whl", name.replace('-', "_"), version, fname_tag);
    let out_path = outdir.join(&wheel_name);
    std::fs::create_dir_all(outdir)?;
    let mut zw = zip::ZipWriter::new(File::create(&out_path)?);

    let mut record = String::new();
    let mut write_file = |zw: &mut zip::ZipWriter<File>, path: &str, mode: u32, data: &[u8]| -> Result<()> {
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(if mode & 0o111 != 0 { 0o755 } else { 0o644 });
        zw.start_file(path, opts)?;
        zw.write_all(data)?;
        let digest = URL_SAFE_NO_PAD.encode(Sha256::digest(data));
        record.push_str(&format!("{path},sha256={digest},{}\n", data.len()));
        Ok(())
    };

    for (path, (mode, data)) in &payload.files {
        // Installer-owned files must not appear in a wheel; RECORD and WHEEL
        // are regenerated below.
        let base = path.strip_prefix(&format!("{distinfo}/")).unwrap_or("");
        if matches!(base, "RECORD" | "RECORD.jws" | "RECORD.p7s" | "INSTALLER" | "REQUESTED" | "direct_url.json")
            || base == "WHEEL"
        {
            continue;
        }
        write_file(&mut zw, path, *mode, data)?;
    }
    write_file(&mut zw, &format!("{distinfo}/WHEEL"), 0o644, wheel_meta.as_bytes())?;

    record.push_str(&format!("{distinfo}/RECORD,,\n"));
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .unix_permissions(0o644);
    zw.start_file(format!("{distinfo}/RECORD"), opts)?;
    zw.write_all(record.as_bytes())?;
    zw.finish()?;

    if !payload.skipped.is_empty() {
        eprintln!(
            "note: {} non-site-packages payload file(s) not included (bin/ wrappers are regenerated by the installer):",
            payload.skipped.len()
        );
        for s in payload.skipped.iter().take(10) {
            eprintln!("  {s}");
        }
        if payload.skipped.len() > 10 {
            eprintln!("  ... and {} more", payload.skipped.len() - 10);
        }
    }
    Ok(out_path)
}
