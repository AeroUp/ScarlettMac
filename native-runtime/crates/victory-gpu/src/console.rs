//! Which console the title runs on. Titles pick settings (resolution,
//! memory budgets) by it, so the user chooses; Series X unless told.

use std::sync::atomic::{AtomicBool, Ordering};

const GIB: usize = 1 << 30;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Console {
    #[default]
    SeriesX,
    SeriesS,
}

impl Console {
    /// Title memory in bytes: (GPU-optimal, standard).
    ///
    /// Series X: 10 GiB of the 16 GiB is the fast pool and 6 GiB the slow one; a title gets the
    /// 10 GiB and 3.5 GiB of the slow, so (10, 3.5). Series S: 10 GiB of which the 8 GiB a title
    /// gets are the *fast* pool and the 2 GiB the system keeps are the slow one, so the title's
    /// whole range (7,776 MiB usable) is GPU-optimal and it has no standard memory to hand back.
    /// It used to be reported the other way round -- all standard, `gpu_optimal_limit` 0 -- which
    /// tells a title that buckets textures by the GPU-optimal limit that it has no fast memory.
    // shortcut: later system updates gave Series S titles more (amount not published here);
    // Grand Theft Auto VI sizes nothing from this figure in our runs, and never calls
    // XMemTransferMemory there, so the split is a figure to correct, not one it acts on yet.
    pub const fn title_memory(self) -> (usize, usize) {
        match self {
            Console::SeriesX => (10 * GIB, 7 * GIB / 2),
            Console::SeriesS => (7_776 << 20, 0),
        }
    }

    /// Compute units of the GPU.
    pub const fn compute_units(self) -> u32 {
        match self {
            Console::SeriesX => 52,
            Console::SeriesS => 20,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Console::SeriesX => "Xbox Series X",
            Console::SeriesS => "Xbox Series S",
        }
    }
}

static SERIES_S: AtomicBool = AtomicBool::new(false);

/// Set before the title starts.
pub fn set(console: Console) {
    SERIES_S.store(console == Console::SeriesS, Ordering::Relaxed);
}

pub fn current() -> Console {
    if SERIES_S.load(Ordering::Relaxed) {
        Console::SeriesS
    } else {
        Console::SeriesX
    }
}
