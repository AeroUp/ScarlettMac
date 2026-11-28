//! Native Metal backend for victory-gpu (macOS only; empty on other targets).
//!
//! `backend` is the platform-neutral seam, `metal` its Metal implementation, `msc` the Metal Shader
//! Converter loaded at run time, `mem` page-aligned host reservations for tests and probes.

pub use victory_backend::{backend, resources};
#[cfg(target_os = "macos")]
mod ambient;
#[cfg(target_os = "macos")]
mod archive;
#[cfg(target_os = "macos")]
pub use archive::{counts as archive_counts, is_compiler_crash};
pub mod disk;
/// `VICTORY_TRACE_FRAME`: per-frame command buffer and per-pass GPU and CPU times (macOS only).
#[cfg(target_os = "macos")]
pub mod frametime;
#[cfg(target_os = "macos")]
pub mod gfx;
#[cfg(target_os = "macos")]
pub mod mem;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "macos")]
pub mod msc;
/// Lock-free frame pacing for the window's HUD (pure std, so its tests run everywhere).
pub mod pace;
#[cfg(target_os = "macos")]
pub mod present;
