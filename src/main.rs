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

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
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
    /// Populate a find-links directory with wheels for a package set,
    /// following python dependencies through the pkgs' own manifests.
    Materialize {
        /// PyPI-style names (numpy, scikit-learn) or pkg names (py312-numpy).
        packages: Vec<String>,
        /// Find-links directory to populate.
        #[arg(short, long, default_value = ".")]
        outdir: PathBuf,
        /// Python version used for the pkg name prefix (pyXY-).
        #[arg(long, default_value = "3.12")]
        python: String,
        /// Directories searched for .pkg files (repeatable).
        #[arg(long, default_value = "/var/cache/pkg")]
        pkg_dir: Vec<PathBuf>,
        /// Do not attempt `pkg fetch` for pkgs missing from the pkg dirs.
        #[arg(long)]
        no_fetch: bool,
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
            let (_, out) = convert(&pkg, &outdir, &tag, keep_pyc)?;
            println!("{}", out.display());
            Ok(())
        }
        Cmd::Materialize { packages, outdir, python, pkg_dir, no_fetch, platform_tag, keep_pyc } => {
            if packages.is_empty() {
                bail!("no packages requested");
            }
            let tag = match platform_tag {
                Some(t) => t,
                None => local_platform_tag()?,
            };
            materialize(&packages, &outdir, &python, &pkg_dir, no_fetch, &tag, keep_pyc)
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

struct PkgInfo {
    name: String,
    version: String,
    /// dependency pkg name -> version, from +COMPACT_MANIFEST
    deps: BTreeMap<String, String>,
}

struct Payload {
    /// site-packages-relative path -> (mode, content)
    files: BTreeMap<String, (u32, Vec<u8>)>,
    /// paths in the pkg payload that are not under site-packages
    skipped: Vec<String>,
    /// python X.Y from the site-packages path prefix
    pyver: Option<String>,
}

fn read_pkg(pkg: &Path, keep_pyc: bool) -> Result<(PkgInfo, Payload)> {
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

    let mut info: Option<PkgInfo> = None;
    let mut files: BTreeMap<String, (u32, Vec<u8>)> = BTreeMap::new();
    let mut skipped = Vec::new();
    let mut pyver: Option<String> = None;

    for entry in ar.entries()? {
        let mut entry = entry?;
        let raw = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
        if raw == "+COMPACT_MANIFEST" || raw == "+MANIFEST" {
            if info.is_none() {
                let mut buf = Vec::with_capacity(entry.size() as usize);
                entry.read_to_end(&mut buf)?;
                let m: serde_json::Value = serde_json::from_slice(&buf)
                    .with_context(|| format!("parse {raw} in {}", pkg.display()))?;
                let deps = m["deps"]
                    .as_object()
                    .map(|o| {
                        o.iter()
                            .map(|(k, v)| {
                                (k.clone(), v["version"].as_str().unwrap_or("").to_string())
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                info = Some(PkgInfo {
                    name: m["name"].as_str().unwrap_or("").to_string(),
                    version: m["version"].as_str().unwrap_or("").to_string(),
                    deps,
                });
            }
            continue;
        }
        // other pkg metadata files are not payload
        if raw.starts_with('+') {
            continue;
        }
        if pyver.is_none() {
            if let Some((pre, _)) = raw.split_once("/site-packages/") {
                pyver = pre
                    .rsplit('/')
                    .next()
                    .and_then(|d| d.strip_prefix("python"))
                    .map(|v| v.to_string());
            }
        }
        let Some(rel) = raw.split_once("/site-packages/").map(|(_, r)| r.to_string()) else {
            // Docs/licenses/man pages never belong in a wheel; only report
            // payload whose loss could matter (bin/ scripts, headers, ...).
            let expected_loss = ["/share/doc/", "/share/licenses/", "/share/man/", "/share/examples/"]
                .iter()
                .any(|p| raw.contains(p));
            if entry.header().entry_type().is_file() && !expected_loss {
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
    let info = info.context("pkg has no +COMPACT_MANIFEST/+MANIFEST")?;
    Ok((info, Payload { files, skipped, pyver }))
}

/// Older ports install setuptools egg-info instead of PEP 517 dist-info.
/// Synthesize dist-info in place: PKG-INFO -> METADATA, generated WHEEL,
/// requires.txt -> Requires-Dist lines appended to METADATA.
fn egg_info_to_dist_info(payload: &mut Payload) -> Result<()> {
    let egg_dir = payload
        .files
        .keys()
        .filter_map(|p| p.split_once('/').map(|(d, _)| d))
        .find(|d| d.ends_with(".egg-info"))
        .map(|d| d.to_string());
    let Some(egg_dir) = egg_dir else {
        bail!("no .dist-info or .egg-info directory in payload");
    };

    // "{name}-{version}[-pyX.Y].egg-info"
    let mut stem = egg_dir.trim_end_matches(".egg-info").to_string();
    if let Some((rest, tail)) = stem.rsplit_once('-') {
        if tail.starts_with("py") && tail[2..].chars().all(|c| c.is_ascii_digit() || c == '.') {
            stem = rest.to_string();
        }
    }
    let (name, version) = stem
        .rsplit_once('-')
        .with_context(|| format!("cannot parse name-version from {egg_dir}"))?;
    let distinfo = format!("{name}-{version}.dist-info");

    let take = |payload: &mut Payload, f: &str| payload.files.remove(&format!("{egg_dir}/{f}"));
    let pkg_info = take(payload, "PKG-INFO")
        .with_context(|| format!("{egg_dir} has no PKG-INFO"))?;
    let mut metadata = String::from_utf8_lossy(&pkg_info.1).into_owned();

    // requires.txt: plain lines are deps; "[extra]" / "[extra:marker]"
    // sections scope the lines that follow.
    if let Some((_, req)) = take(payload, "requires.txt") {
        let mut extra: Option<String> = None;
        let mut marker: Option<String> = None;
        let mut lines = String::new();
        for line in String::from_utf8_lossy(&req).lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(section) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                let (e, m) = section.split_once(':').unwrap_or((section, ""));
                extra = (!e.is_empty()).then(|| e.to_string());
                marker = (!m.is_empty()).then(|| m.to_string());
                if let Some(e) = &extra {
                    lines.push_str(&format!("Provides-Extra: {e}\n"));
                }
                continue;
            }
            let mut conds = Vec::new();
            if let Some(m) = &marker {
                conds.push(m.clone());
            }
            if let Some(e) = &extra {
                conds.push(format!("extra == '{e}'"));
            }
            if conds.is_empty() {
                lines.push_str(&format!("Requires-Dist: {line}\n"));
            } else {
                lines.push_str(&format!("Requires-Dist: {line} ; {}\n", conds.join(" and ")));
            }
        }
        // Metadata headers end at the first blank line (the description
        // body follows); insert the dependency fields before it.
        match metadata.find("\n\n") {
            Some(i) => metadata.insert_str(i + 1, &lines),
            None => {
                if !metadata.ends_with('\n') {
                    metadata.push('\n');
                }
                metadata.push_str(&lines);
            }
        }
    }

    // Compiled if any extension module is present; tag conservatively for
    // this interpreter (restamp replaces the placeholder platform).
    let compiled = payload.files.keys().any(|p| p.ends_with(".so"));
    let tag = if compiled {
        let cp = payload
            .pyver
            .as_deref()
            .map(|v| format!("cp{}", v.replace('.', "")))
            .context("compiled egg-info pkg but python version unknown")?;
        format!("{cp}-{cp}-freebsd")
    } else {
        "py3-none-any".to_string()
    };
    let wheel = format!(
        "Wheel-Version: 1.0\nGenerator: wheelfs {}\nRoot-Is-Purelib: {}\nTag: {tag}\n",
        env!("CARGO_PKG_VERSION"),
        !compiled
    );

    if let Some(ep) = take(payload, "entry_points.txt") {
        payload.files.insert(format!("{distinfo}/entry_points.txt"), ep);
    }
    // Remaining egg bookkeeping (SOURCES.txt, top_level.txt, zip-safe,
    // dependency_links.txt, installed-files.txt) has no dist-info role.
    let leftovers: Vec<String> = payload
        .files
        .keys()
        .filter(|p| p.starts_with(&format!("{egg_dir}/")))
        .cloned()
        .collect();
    for p in leftovers {
        payload.files.remove(&p);
    }
    payload.files.insert(format!("{distinfo}/METADATA"), (0o644, metadata.into_bytes()));
    payload.files.insert(format!("{distinfo}/WHEEL"), (0o644, wheel.into_bytes()));
    Ok(())
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

fn convert(pkg: &Path, outdir: &Path, platform_tag: &str, keep_pyc: bool) -> Result<(PkgInfo, PathBuf)> {
    let (info, mut payload) = read_pkg(pkg, keep_pyc)?;
    if payload.files.is_empty() {
        bail!("{}: no site-packages payload found; not a python pkg?", pkg.display());
    }
    if !payload.files.keys().any(|p| {
        p.split_once('/').is_some_and(|(d, _)| d.ends_with(".dist-info"))
    }) {
        egg_info_to_dist_info(&mut payload)?;
    }
    let payload = payload;

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
    Ok((info, out_path))
}

/// Map a requested name to a pkg name: pass py3* names through, otherwise
/// prefix with pyXY-. (Ports occasionally rename — pyyaml is py312-yaml —
/// pass the pkg name directly for those.)
fn resolve_pkg_name(requested: &str, pyprefix: &str) -> String {
    if requested.starts_with("py3") {
        requested.to_string()
    } else {
        format!("{pyprefix}-{}", requested.to_lowercase())
    }
}

/// Find a .pkg file for `pkgname` in `dirs`: `{pkgname}-{version}.pkg` where
/// version starts with a digit (so py312-pandas doesn't match
/// py312-pandas-stubs). Prefers the lexically-greatest version and ignores
/// the `~hash` duplicate names pkg writes.
fn find_pkg_file(pkgname: &str, dirs: &[PathBuf]) -> Option<PathBuf> {
    let prefix = format!("{pkgname}-");
    let mut best: Option<(String, PathBuf)> = None;
    for dir in dirs {
        let Ok(rd) = std::fs::read_dir(dir) else { continue };
        for e in rd.flatten() {
            let fname = e.file_name().to_string_lossy().into_owned();
            let Some(rest) = fname.strip_prefix(&prefix) else { continue };
            let Some(verfull) = rest.strip_suffix(".pkg") else { continue };
            // "2.4.6_1,1~cae94441f8" (local duplicate) and "2.4.6_1,1~2$hash"
            // (hashed repo layout) both name the version before the '~'.
            let ver = verfull.split('~').next().unwrap_or("");
            if !ver.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                continue;
            }
            if best.as_ref().is_none_or(|(b, _)| ver > b.as_str()) {
                best = Some((ver.to_string(), e.path()));
            }
        }
    }
    best.map(|(_, p)| p)
}

fn materialize(
    packages: &[String],
    outdir: &Path,
    python: &str,
    pkg_dirs: &[PathBuf],
    no_fetch: bool,
    platform_tag: &str,
    keep_pyc: bool,
) -> Result<()> {
    let pyprefix = format!("py{}", python.replace('.', ""));
    let fetch_dir = dirs_cache_dir()?;
    let mut search_dirs: Vec<PathBuf> = pkg_dirs.to_vec();
    // pkg fetch -o writes into <dir>/All; hashed-layout repos use All/Hashed
    search_dirs.push(fetch_dir.join("All"));
    search_dirs.push(fetch_dir.join("All/Hashed"));

    let mut queue: VecDeque<String> =
        packages.iter().map(|p| resolve_pkg_name(p, &pyprefix)).collect();
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut wheels: Vec<PathBuf> = Vec::new();
    let mut native: BTreeMap<String, String> = BTreeMap::new();
    let mut missing: Vec<String> = Vec::new();
    let mut failed: Vec<String> = Vec::new();

    while let Some(pkgname) = queue.pop_front() {
        if !visited.insert(pkgname.clone()) {
            continue;
        }
        let mut file = find_pkg_file(&pkgname, &search_dirs);
        if file.is_none() && !no_fetch {
            let st = Command::new("pkg")
                .args(["fetch", "-y", "-o"])
                .arg(&fetch_dir)
                .arg(&pkgname)
                .status();
            if !st.map(|s| s.success()).unwrap_or(false) {
                eprintln!(
                    "note: `pkg fetch {pkgname}` failed (root/doas needed for the repo catalogue?)"
                );
            }
            file = find_pkg_file(&pkgname, &search_dirs);
        }
        let Some(file) = file else {
            missing.push(pkgname);
            continue;
        };
        let (info, wheel) = match convert(&file, outdir, platform_tag, keep_pyc) {
            Ok(v) => v,
            Err(e) => {
                failed.push(format!("{pkgname}: {e:#}"));
                continue;
            }
        };
        println!("{pkgname} -> {}", wheel.file_name().unwrap_or_default().to_string_lossy());
        wheels.push(wheel);
        for (dep, ver) in info.deps {
            if dep.starts_with(&format!("{pyprefix}-")) {
                queue.push_back(dep);
            } else if !dep.starts_with("python3") {
                native.entry(dep).or_insert(ver);
            }
        }
    }

    println!("\nmaterialized {} wheel(s) into {}", wheels.len(), outdir.display());
    if !native.is_empty() {
        println!("native pkg deps required at runtime (converted wheels link against /usr/local/lib):");
        for (dep, ver) in &native {
            let installed = Command::new("pkg")
                .args(["info", "-e", dep])
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            let mark = if installed { "installed" } else { "MISSING — pkg install" };
            println!("  {dep} {ver} [{mark}]");
        }
    }
    if !missing.is_empty() {
        println!("no .pkg found for: {}", missing.join(", "));
        println!("  (fetch with: doas pkg fetch -y {})", missing.join(" "));
    }
    for f in &failed {
        println!("conversion failed: {f}");
    }
    if !missing.is_empty() || !failed.is_empty() {
        bail!("{} pkg(s) could not be materialized", missing.len() + failed.len());
    }
    Ok(())
}

fn dirs_cache_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME not set")?;
    let d = PathBuf::from(home).join(".cache/wheelfs/pkgs");
    std::fs::create_dir_all(&d)?;
    Ok(d)
}
