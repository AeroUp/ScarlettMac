//! pthread implementation of `thread` (through `std::thread`, which wraps
//! them and takes the stack size).

use std::io;
use std::os::unix::thread::JoinHandleExt;
use std::time::Duration;

unsafe extern "C" {
    fn pthread_threadid_np(thread: usize, id: *mut u64) -> i32;
}

/// A thread started by [`spawn`].
#[derive(Debug)]
pub struct Thread {
    handle: std::thread::JoinHandle<u32>,
    id: u32,
}

/// Starts a thread named `name` with a stack of `stack_size` bytes that
/// runs `f`; `f`'s result is the thread's exit code.
pub fn spawn(
    name: &str,
    stack_size: usize,
    f: impl FnOnce() -> u32 + Send + 'static,
) -> io::Result<Thread> {
    let handle = std::thread::Builder::new()
        .name(name.to_owned())
        .stack_size(stack_size)
        .spawn(f)?;
    let mut id = 0;
    // SAFETY: a live thread's pthread_t and a local out-pointer.
    unsafe { pthread_threadid_np(handle.as_pthread_t(), &mut id) };
    Ok(Thread {
        handle,
        id: id as u32,
    })
}

impl Thread {
    /// The OS thread id, as `current_id` returns on that thread.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Waits for the thread and returns its exit code (`u32::MAX` if it panicked).
    pub fn join(self) -> u32 {
        self.handle.join().unwrap_or(u32::MAX)
    }
}

/// The calling thread's OS id.
pub fn current_id() -> u32 {
    let mut id = 0;
    // SAFETY: 0 names the calling thread; local out-pointer.
    unsafe { pthread_threadid_np(0, &mut id) };
    id as u32
}

pub fn sleep(time: Duration) {
    std::thread::sleep(time);
}

/// Suspends every other thread of the process, so that nothing else runs while the process
/// ends: a title thread still in AppKit, in an audio callback or in Metal would touch objects
/// the stop has drained, and the C runtime's exit path must not run with the other threads
/// stopped in the middle of a call (their locks are never let go).
///
/// Called once, on the thread that ends the process with `_exit`; the others are never resumed.
pub fn suspend_all_others() {
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn mach_thread_self() -> u32;
        fn task_threads(task: u32, list: *mut *mut u32, count: *mut u32) -> i32;
        fn thread_suspend(thread: u32) -> i32;
        fn mach_port_deallocate(task: u32, name: u32) -> i32;
        fn vm_deallocate(task: u32, address: usize, size: usize) -> i32;
    }
    // SAFETY: the Mach calls operate on this process's own task and on thread ports the kernel
    // hands out for it; the array `task_threads` fills is released with `vm_deallocate`.
    unsafe {
        let task = mach_task_self_;
        let me = mach_thread_self();
        let (mut threads, mut count) = (std::ptr::null_mut::<u32>(), 0u32);
        if task_threads(task, &mut threads, &mut count) != 0 {
            return;
        }
        for i in 0..count as usize {
            let thread = threads.add(i).read();
            if thread != me {
                thread_suspend(thread);
            }
            mach_port_deallocate(task, thread);
        }
        vm_deallocate(
            task,
            threads as usize,
            count as usize * std::mem::size_of::<u32>(),
        );
        mach_port_deallocate(task, me);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    /// In a child process of the next test: suspends the other threads and checks a busy one
    /// stopped. It cannot report through a joining thread (`_exit` is the point), so its exit
    /// code is the result.
    #[test]
    fn suspend_child() {
        if std::env::var_os("VICTORY_SUSPEND_CHILD").is_none() {
            return;
        }
        static TICKS: AtomicU64 = AtomicU64::new(0);
        std::thread::spawn(|| {
            loop {
                TICKS.fetch_add(1, Ordering::SeqCst);
                std::thread::yield_now();
            }
        });
        while TICKS.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        super::suspend_all_others();
        let at = TICKS.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(50));
        let stopped = TICKS.load(Ordering::SeqCst) == at;
        // SAFETY: the test child ends here, as the process's stop does.
        unsafe { _exit(i32::from(!stopped)) }
    }

    unsafe extern "C" {
        fn _exit(code: i32) -> !;
    }

    #[test]
    fn suspending_the_other_threads_stops_them() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["suspend_child", "--nocapture"])
            .env("VICTORY_SUSPEND_CHILD", "1")
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
    }
}
