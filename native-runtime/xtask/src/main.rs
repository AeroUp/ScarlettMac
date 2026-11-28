//! Repository maintenance. Run with `cargo xtask <command>`.

mod policy;
mod shader;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enforce the asset policy: the repository holds only small text files
    /// we wrote. Checks tracked and untracked-but-not-ignored files.
    Policy {
        /// Check only what is staged for commit (used by the pre-commit hook).
        #[arg(long)]
        staged: bool,
    },
    /// Point git at .githooks/ so every commit runs the policy check.
    InstallHooks,
    /// Debug one or more shader containers by the hash a run log names:
    /// its GFX10 disassembly and pseudo-HLSL, with the translation's outcome.
    /// Game data stays outside the repository. With `--reference`, the
    /// opcode tables are cross-checked against RageForge's reference tables.
    /// With `--probe`, the container's layout and the native reader's verdict.
    Shader {
        /// Container hashes, as log lines print them (`compute 39ef10815cb825a0`, or `39ef...`).
        #[arg(required = true)]
        hashes: Vec<String>,
        /// The folder of dumped containers (`<stage>_<hash>.bin` or `<hash>.bin`).
        /// `VICTORY_SHADER_CORPUS` names it too.
        #[arg(long)]
        corpus: Option<PathBuf>,
        /// Do not print the disassembly.
        #[arg(long)]
        no_disasm: bool,
        /// Do not print the pseudo-HLSL.
        #[arg(long)]
        no_hlsl: bool,
        /// Write each translation's DXIL (or its refusal) to this folder.
        #[arg(long)]
        dxil: Option<PathBuf>,
        /// RageForge's `Shaders/Tables.cs`: cross-check our opcode tables against its.
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Fail (exit 1) when a reference table names another instruction than ours.
        #[arg(long, requires = "reference")]
        check: bool,
        /// Print only the container's layout and the native reader's verdict:
        /// part sizes, DXIL program header, every XDAT offset/tag and whether
        /// `native_programs` finds a program. For checking a compiler's output.
        #[arg(long)]
        probe: bool,
    },
}

fn main() -> Result<()> {
    let root = workspace_root();
    match Cli::parse().command {
        Cmd::Policy { staged } => policy::run(&root, staged),
        Cmd::InstallHooks => {
            let status = Command::new("git")
                .args(["config", "core.hooksPath", "native-runtime/.githooks"])
                .current_dir(&root)
                .status()
                .context("running git")?;
            anyhow::ensure!(
                status.success(),
                "git config failed (is this a git repository?)"
            );
            println!("hooks installed: commits now run `cargo xtask policy --staged`");
            Ok(())
        }
        Cmd::Shader {
            hashes,
            corpus,
            no_disasm,
            no_hlsl,
            dxil,
            reference,
            check,
            probe,
        } => shader::run(shader::Args {
            hashes,
            corpus,
            disasm: !no_disasm,
            hlsl: !no_hlsl,
            dxil,
            reference,
            check,
            probe,
        }),
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace")
        .into()
}
