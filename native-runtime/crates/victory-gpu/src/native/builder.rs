//! The console's acceleration-structure builder: the compute programs of `XboxBvhBuilder.dll`.
//!
//! Grand Theft Auto VI builds its ray-tracing acceleration structures with a library of compute
//! programs in that DLL (52 in the build of 27 November; every one of them carries its
//! `.pdb` path in the container's `ILDN` part, and their containers lie in the DLL's own image).
//! They run on the native host like any other compute program, but their mistakes are not a
//! wrong pixel: the structures they build are walked by every ray-tracing pass
//! (`victory_shader`'s `image_bvh*_intersect_ray`), and a wrong structure takes the GPU down,
//! which macOS ends a command buffer for (`victory_metal::gfx`). So under `VICTORY_BVH_BUILDER=1`
//! the host runs them as one unit:
//!
//! * Each of their compute pipelines is registered here, and its two forms are translated
//!   ahead on a thread of their own: the plain one, and the one for dispatches counted in
//!   threads (`DispatchX` with flag 0x20, `emit::Options::thread_counts`), which the box
//!   unpackers need. Both are the native host's own translations: this host has no switch that
//!   lets one half of a console wave answer alone (a wave64 program is emulated across both
//!   halves exactly, `victory_shader::emit::lower::halves`, or refused), so what upstream's tree
//!   calls its forms' "exact options" is just `shaders::options` here.
//! * The builder's dispatches run in a frame only once every plain form is made and every
//!   thread form is made or refused. Until then they are dropped ([`held`]), and so is the
//!   builder for the rest of the run once it trips ([`trip`]): because a dispatch counted in
//!   threads found no thread form (it would write where memory is not its own), or because the
//!   host GPU ended a submission while the builder ran ([`submission_lost`]).
//! * A program that has no thread form does not stop the builder by itself: only a dispatch
//!   counted in threads that asks for one does (the title's sort step is one such program,
//!   `e63d22c4def9ba6a`: upstream refuses it under `thread_counts`, while this host translates
//!   it, its first-lane read being of a value every lane of the console wave holds alike).
//!
//! With the switch off the hold and the trip are off, but the identification and the translation
//! ahead are not: a builder pipeline's shaders go to the front of the prewarm queue as the
//! title creates them, and `glue::pipe` never puts such a pipeline off for the frame's budget.
//! Without that the world frame (which creates and binds them after its budget is spent) left
//! all of them untranslated and dropped their one-shot build, so no acceleration structure
//! existed for the ray-traced shadows to walk.
//!
//! Observed, not console behaviour: the identification by the DLL's name in the debug path and
//! by the bytecode's address inside the DLL's image, the translation ahead, and the hold and
//! trip themselves (the console has no such thing). See `docs/native-host.md`, "The BVH builder".

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::pipeline::Shader;

/// `VICTORY_BVH_BUILDER=1`: the builder runs under the hold and trip above. Unset or anything
/// else: off.
pub fn on() -> bool {
    #[cfg(test)]
    if FORCED_OFF.load(Ordering::Relaxed) > 0 {
        return false;
    }
    #[cfg(test)]
    if FORCED.load(Ordering::Relaxed) > 0 {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_BVH_BUILDER").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::info!(
                "VICTORY_BVH_BUILDER=1: the BVH builder runs from forms of its own, once all are made; a dispatch counted in threads without one and a lost submission stop it"
            );
        }
        on
    })
}

/// Whether the container was compiled from one of `XboxBvhBuilder.dll`'s programs: its `ILDN`
/// debug name (the file the program was compiled from) names the DLL's build folder. This is the
/// identification alone: with the switch off the builder still translates ahead and runs (see
/// [`note_compute`]); only the hold and the trip are `on()`. For anything that is not a container:
/// no program is the builder's.
pub fn named(container: &[u8]) -> bool {
    victory_shader::Container::parse(container)
        .ok()
        .and_then(|c| c.debug_name().map(str::to_ascii_lowercase))
        .is_some_and(|name| name.contains("xboxbvhbuilder"))
}

/// Title images the loader mapped: base, length. The builder's is its DLL's, whose image holds
/// the containers of its programs, so a pipeline made from bytecode there is its.
static IMAGES: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Notes a title image the loader mapped (`apps/victory`, where a title DLL is mapped and bound).
/// Others are of no use here and are not kept.
pub fn note_image(name: &str, base: usize, len: usize) {
    if !name.to_ascii_lowercase().contains("xboxbvhbuilder") {
        return;
    }
    let mut images = IMAGES.lock().unwrap_or_else(|e| e.into_inner());
    images.retain(|&(b, _)| b != base);
    images.push((base, len));
}

/// Whether `at` lies in one of the builder's images.
pub(crate) fn in_image(at: usize) -> bool {
    let images = IMAGES.lock().unwrap_or_else(|e| e.into_inner());
    images.iter().any(|&(base, len)| in_range(base, len, at))
}

fn in_range(base: usize, len: usize, at: usize) -> bool {
    at >= base && at - base < len
}

/// The builder's shaders, alive while the title holds their pipelines.
static SHADERS: Mutex<Vec<Weak<Shader>>> = Mutex::new(Vec::new());

/// The builder's forms to translate, ahead of the title's own shaders.
static FORMS: crate::prewarm::Queue<Arc<Shader>> = crate::prewarm::Queue::new();

/// Starts the one thread that makes the builder's forms (once).
fn start_forms_thread() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = std::thread::Builder::new()
            .name("bvh-builder-forms".into())
            .spawn(|| {
                crate::prewarm::lower_priority();
                loop {
                    let shader = FORMS.pop();
                    let _ = shader.native();
                    let _ = shader.native_threads_shader();
                }
            });
    });
}

/// A compute pipeline the title made whose programs are the builder's (their containers name
/// the DLL, or their bytecode lies in its image). Registered whatever the switch says, and pushed
/// to the front of the translation queue: the title creates and binds these pipelines back to
/// back in the world frame, after the frame's first-bind budget is spent, so a shader behind the
/// other ~1,300 new ones is put off and its one-shot build never runs (in a run on 27 November
/// the first world frame bound 35 of the programs and put them all off, so nothing built an
/// acceleration structure). Under [`on`]
/// a thread of its own also makes their thread-count forms, and the dispatches are held.
pub(crate) fn note_compute(shaders: &[Arc<Shader>]) {
    let mut list = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    list.retain(|w| w.strong_count() > 0);
    for shader in shaders {
        if !shader.builder.load(Ordering::Relaxed) {
            continue;
        }
        if !list.iter().any(|w| w.ptr_eq(&Arc::downgrade(shader))) {
            list.push(Arc::downgrade(shader));
        }
        // Queued again even if it was registered before: a pipeline made again after a run with
        // the switch off has its forms unsettled, and the queue's translations are once-locks.
        if on() {
            FORMS.push(shader.clone(), true);
        } else {
            crate::pipeline::prewarm_next(shader);
        }
    }
    drop(list);
    if on() {
        start_forms_thread();
    }
}

/// A form of a builder program: made (`true`), refused (`false`), or not yet (`None`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct FormState {
    plain: Option<bool>,
    threads: Option<bool>,
}

/// Whether the builder runs: every plain form made and every thread form made or refused (a
/// refused thread form stops the builder only when a dispatch counted in threads asks for it,
/// [`trip`]).
fn opens(forms: &[FormState]) -> bool {
    forms
        .iter()
        .all(|f| f.plain == Some(true) && f.threads.is_some())
}

/// The forms as they stand: read without waiting for a translation another thread is making.
fn forms_now() -> Vec<(Weak<Shader>, FormState)> {
    let list = {
        let mut list = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
        list.retain(|w| w.strong_count() > 0);
        list.clone()
    };
    list.into_iter()
        .filter_map(|w| {
            let shader = w.upgrade()?;
            let forms = FormState {
                plain: shader.native_state(),
                threads: shader.threads_state(),
            };
            Some((Arc::downgrade(&shader), forms))
        })
        .collect()
}

/// The frame [`runs`] last decided for, and its decision.
static DECIDED: Mutex<(u64, bool)> = Mutex::new((u64::MAX, true));

/// Tests: [`on`] whatever the variable says, while this lives.
#[cfg(test)]
static FORCED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Tests: not [`on`] whatever the variable (or [`FORCED`]) says, while this lives.
#[cfg(test)]
static FORCED_OFF: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Tests: [`on`] is true while this lives (counted, so two tests may hold it at once).
#[cfg(test)]
pub(crate) struct ForcedOn;

#[cfg(test)]
impl ForcedOn {
    pub(crate) fn new() -> ForcedOn {
        FORCED.fetch_add(1, Ordering::Relaxed);
        ForcedOn
    }
}

#[cfg(test)]
impl Drop for ForcedOn {
    fn drop(&mut self) {
        FORCED.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Tests: [`on`] is false while this lives.
#[cfg(test)]
pub(crate) struct ForcedOff;

#[cfg(test)]
impl ForcedOff {
    pub(crate) fn new() -> ForcedOff {
        FORCED_OFF.fetch_add(1, Ordering::Relaxed);
        ForcedOff
    }
}

#[cfg(test)]
impl Drop for ForcedOff {
    fn drop(&mut self) {
        FORCED_OFF.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Tests: [`runs`] decides again in this frame (a test has no frame waits to advance it).
#[cfg(test)]
pub(crate) fn forget_decision() {
    *DECIDED.lock().unwrap_or_else(|e| e.into_inner()) = (u64::MAX, true);
}

/// The frame the builder last ran in (`u64::MAX`: never).
static RAN_IN: AtomicU64 = AtomicU64::new(u64::MAX);

/// Why the builder stopped for good, once it did ([`trip`]).
static TRIPPED: OnceLock<String> = OnceLock::new();

/// Whether the builder runs this frame: not tripped, and every form settled. Off: always.
pub(crate) fn runs() -> bool {
    if !on() {
        return true;
    }
    let frame = crate::frame::current();
    if TRIPPED.get().is_some() {
        return false;
    }
    let previous = {
        let decided = DECIDED.lock().unwrap_or_else(|e| e.into_inner());
        if decided.0 == frame {
            return decided.1;
        }
        decided.1
    };
    let forms = forms_now();
    let states: Vec<FormState> = forms.iter().map(|(_, f)| *f).collect();
    let open = opens(&states);
    let (made, refused) = (
        states.iter().filter(|f| f.plain == Some(true)).count(),
        states.iter().filter(|f| f.plain == Some(false)).count(),
    );
    let (threads_made, threads_refused) = (
        states.iter().filter(|f| f.threads == Some(true)).count(),
        states.iter().filter(|f| f.threads == Some(false)).count(),
    );
    static SAID: AtomicBool = AtomicBool::new(false);
    if open != previous || !SAID.swap(true, Ordering::Relaxed) {
        tracing::info!(
            frame,
            pipelines = states.len(),
            made,
            refused,
            threads_made,
            threads_refused,
            "BVH builder {}",
            if open {
                "runs"
            } else if refused > 0 {
                "held back: a program of it has no plain form (VICTORY_BVH_BUILDER=1)"
            } else {
                "held back: not every form of it is settled (VICTORY_BVH_BUILDER=1)"
            }
        );
    }
    if open {
        static NAMED: AtomicBool = AtomicBool::new(false);
        if !NAMED.swap(true, Ordering::Relaxed) {
            let names: Vec<String> = forms
                .iter()
                .filter_map(|(w, _)| w.upgrade().map(|s| s.hash.clone()))
                .collect();
            tracing::info!(programs = ?names, "the BVH builder's programs");
        }
    }
    *DECIDED.lock().unwrap_or_else(|e| e.into_inner()) = (frame, open);
    if open {
        RAN_IN.store(frame, Ordering::Relaxed);
    }
    open
}

/// Why the builder's dispatch is held this frame, if it is ([`runs`]).
pub(crate) fn held() -> Option<String> {
    if !on() {
        return None;
    }
    if let Some(why) = TRIPPED.get() {
        return Some(why.clone());
    }
    if runs() {
        return None;
    }
    Some("the BVH builder's forms are not settled yet (VICTORY_BVH_BUILDER=1)".to_owned())
}

/// The builder runs no more in this run: `why` (logged once). From the next frame on; what this
/// frame recorded of it stays.
pub(crate) fn trip(why: String) {
    if !on() {
        return;
    }
    if TRIPPED.set(why.clone()).is_ok() {
        tracing::warn!(why, "BVH builder held back for the rest of the run");
    }
}

/// The host GPU ended a submission before it finished (macOS ends a command buffer when the GPU
/// locks up or takes a page fault; `victory_metal::gfx`). Under [`on`], if the builder ran in this
/// frame or the one before, it may be the cause: it runs no more, so a bad structure or a loop of
/// its costs one lost submission, not one a frame.
pub(crate) fn submission_lost() {
    let now = crate::frame::current();
    let ran = RAN_IN.load(Ordering::Relaxed);
    if on() && ran_lately(ran, now) {
        trip(format!(
            "the host GPU ended a submission in frame {now} while the BVH builder ran (it ran in frame {ran})"
        ));
    }
}

/// Whether a builder that last ran in frame `ran` (`u64::MAX`: never) ran in frame `now` or the
/// one before.
fn ran_lately(ran: u64, now: u64) -> bool {
    ran != u64::MAX && now <= ran.saturating_add(1)
}

/// Asks the backend how many submissions it ended before they finished (failed command buffers
/// and those it paid as lost), and trips the builder when one ended while it ran.
pub(crate) fn poll_lost() {
    if !on() {
        return;
    }
    static SEEN: AtomicU64 = AtomicU64::new(0);
    let ended = ended_submissions();
    let seen = SEEN.load(Ordering::Relaxed);
    if ended != seen && SEEN.compare_exchange(seen, ended, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        submission_lost();
    }
}

#[cfg(target_os = "macos")]
fn ended_submissions() -> u64 {
    victory_metal::gfx::ended_submissions()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The switch is process-wide: tests that force it take turns.
    static SWITCH: Mutex<()> = Mutex::new(());

    /// Under `VICTORY_BVH_BUILDER=1`: every plain form must be made and every thread form made
    /// or refused; a program still translating holds the builder.
    #[test]
    fn the_builder_runs_with_every_form_settled() {
        let form = |plain, threads| FormState { plain, threads };
        assert!(opens(&[]));
        assert!(opens(&[form(Some(true), Some(true))]));
        assert!(opens(&[form(Some(true), Some(false))]), "a refused thread form");
        assert!(!opens(&[form(Some(true), None)]), "a thread form being made");
        assert!(!opens(&[form(None, Some(true))]), "a plain form being made");
        assert!(!opens(&[form(Some(false), Some(true))]), "a plain form refused");
    }

    /// A lost submission stops the builder if it ran in that frame or the one before; never if
    /// it has not run.
    #[test]
    fn a_lost_submission_stops_a_builder_that_ran_lately() {
        assert!(!ran_lately(u64::MAX, 0));
        assert!(!ran_lately(u64::MAX, u64::MAX));
        assert!(ran_lately(7, 7));
        assert!(ran_lately(7, 8), "the frame after: its work may still run");
        assert!(!ran_lately(7, 9));
    }

    /// A container whose `ILDN` debug name is `name`.
    fn debug_container(name: &[u8]) -> Vec<u8> {
        let mut part = vec![0, 0];
        part.extend_from_slice(&(name.len() as u16 + 1).to_le_bytes());
        part.extend_from_slice(name);
        part.push(0);
        part.resize(part.len().next_multiple_of(4), 0);
        victory_shader::dxbc::build(&[(victory_shader::FourCC::ILDN, &part)])
    }

    /// The builder's programs are named by the debug path in their container, and a container
    /// without one (or from another module) is not the builder's. The identification is not the
    /// switch: with `VICTORY_BVH_BUILDER` off the builder still translates ahead and runs.
    #[test]
    fn a_builder_program_is_named_by_its_container() {
        let _switch = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
        for switch in [true, false] {
            let _on = switch.then(ForcedOn::new);
            let _off = (!switch).then(ForcedOff::new);
            assert_eq!(on(), switch);
            assert!(named(&debug_container(b"X:\\americas\\Output\\XboxBvhBuilder\\RadixAlloc.pdb")));
            assert!(named(&debug_container(br"x:\americas\output\xboxbvhbuilder\PrefixSum64.pdb")));
            assert!(!named(&debug_container(b"X:\\americas\\Output\\Other\\Program.pdb")));
            let plain = victory_shader::dxbc::build(&[(victory_shader::FourCC::SFI0, &[0; 8])]);
            assert!(!named(&plain));
            assert!(!named(b"not a container"));
        }
    }

    /// An address is in an image by its range, and the DLL's name is matched whatever its case
    /// or folder.
    #[test]
    fn an_address_is_in_the_builders_image_by_its_range() {
        assert!(in_range(0x1000, 0x200, 0x1000));
        assert!(in_range(0x1000, 0x200, 0x11FF));
        assert!(!in_range(0x1000, 0x200, 0x1200));
        assert!(!in_range(0x1000, 0x200, 0xFFF));
    }

    /// A trivial compute program, in a container whose debug name is the builder's.
    fn builder_compute_container() -> Vec<u8> {
        let base = crate::pipeline::test_compute_container(&[0x7E02_0280, 0xBF81_0000], &[], 0);
        let parsed = victory_shader::Container::parse(&base).expect("a container");
        let name = b"X:\\americas\\Output\\XboxBvhBuilder\\Sort.pdb";
        let mut ildn = vec![0, 0];
        ildn.extend_from_slice(&(name.len() as u16 + 1).to_le_bytes());
        ildn.extend_from_slice(name);
        ildn.push(0);
        ildn.resize(ildn.len().next_multiple_of(4), 0);
        let mut parts: Vec<(victory_shader::FourCC, &[u8])> =
            parsed.parts.iter().map(|p| (p.fourcc, p.data)).collect();
        parts.push((victory_shader::FourCC::ILDN, &ildn));
        victory_shader::dxbc::build(&parts)
    }

    /// A compute pipeline the title makes from one of the builder's containers is one of its
    /// own: the builder waits until both of its forms are made, then runs.
    #[test]
    fn a_builder_pipeline_is_held_until_its_forms_are_made() {
        let _switch = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
        let _on = ForcedOn::new();
        // The container must stay where the title left it: the thread form reads it again.
        let container = builder_compute_container();
        let pipeline = crate::pipeline::test_compute(&container);
        assert!(crate::pipeline::is_builder(pipeline), "named by its container");
        forget_decision();
        assert!(
            (0..600).any(|_| {
                std::thread::sleep(std::time::Duration::from_millis(10));
                forget_decision();
                runs()
            }),
            "the forms are made on their own thread"
        );
        assert!(held().is_none(), "the builder runs once both forms are made");
        let forms = forms_now();
        assert!(
            forms.iter().all(|(_, f)| f.plain == Some(true) && f.threads == Some(true)),
            "both forms settled: {forms:?}"
        );
        // One of our pipelines, released as the title releases it.
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(pipeline, 0x10)(pipeline);
    }

    /// With `VICTORY_BVH_BUILDER` off the builder's pipeline is still the builder's (so
    /// `glue::pipe` never puts its translation off), but nothing is held: the dispatches run.
    #[test]
    fn a_builder_pipeline_is_not_held_with_the_switch_off() {
        let _switch = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
        let _off = ForcedOff::new();
        let container = builder_compute_container();
        let pipeline = crate::pipeline::test_compute(&container);
        assert!(!on());
        assert!(crate::pipeline::is_builder(pipeline), "identified whatever the switch says");
        assert!(held().is_none(), "the switch off never holds a dispatch");
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(pipeline, 0x10)(pipeline);
    }
}
