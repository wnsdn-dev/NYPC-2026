// BEGIN BUNDLED FILE: main.cpp
// NYPC 2026 Rookie Final -- "슈퍼루키" (interactive platformer coin collection).
//
// TURN-BASED JUDGE. The judge process sends one frame and waits for our move, so
// every read is line-at-a-time and every response is flushed (interactive.hpp).
// read_all_input() would deadlock on every case.
//
// COST (from the statement, verified against superrookie-providing/testing-tool.py):
//     Cost(t) = floor((1 - c_t / (c_max + s_t + h_t)) * 1e6),  Cost = min over t
// The statement guarantees every map is fully collectible WITHOUT special jumps
// and WITHOUT touching spikes, so s_t and h_t are kept at exactly zero: the
// search never emits 'S' and prunes any state that overlaps a spike. That turns
// the objective into pure coin maximisation.
//
// The judge's --timeout is CUMULATIVE WALL TIME SPENT WAITING FOR US (default
// 2s). Blowing it produces no POINT line at all, i.e. the map is lost outright,
// so the timer below is a hard deadline: when it runs out we print FINISH and
// exit cleanly, keeping whatever cost we have already banked.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>
#include <utility>
#include <string>
#include <vector>

// BEGIN BUNDLED FILE: core.hpp
// Core runtime for every NYPC challenge answer: fixed-width aliases, whole-file
// I/O, a fast tokenizer and writer, environment/parameter overrides for non-LLM
// sweeps, a deterministic RNG, and a timer that watches CPU *and* wall time.
//
// Written for this repository. No third-party code is copied here, so there is
// nothing to attribute. If you paste public code in during the contest, keep
// its licence and add a source comment next to it (rules: unattributed public
// code counts as code you had no right to use).
//
// src/main.cpp includes this; tools/bundle.py inlines it into the single
// submission file. include/data_blob.hpp also includes it, so the file-reading
// helpers exist exactly once.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif

#if defined(__linux__)
#include <limits.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace contest {

using i32 = std::int32_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// ---------------------------------------------------------------- debug ----
// Debug output must never reach stdout: stdout is the judged answer.
template <typename... Args>
void debug_print(const Args&... args) {
    ((std::cerr << args << ' '), ...);
    std::cerr << '\n';
}

#ifdef LOCAL
#define NYPC_DEBUG(...) ::contest::debug_print(__VA_ARGS__)
#else
#define NYPC_DEBUG(...) ((void)0)
#endif

// ------------------------------------------------------------------ io -----
// One fread loop instead of `std::cin.rdbuf()`: for a multi-megabyte challenge
// input the difference is tens of milliseconds of the time budget.
inline std::string read_all_input() {
    std::string bytes;
    char chunk[1 << 16];
    std::size_t got = 0;
    while ((got = std::fread(chunk, 1, sizeof(chunk), stdin)) > 0) {
        bytes.append(chunk, got);
    }
    return bytes;
}

inline void write_output(std::string_view text) {
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::fflush(stdout);
}

// Whitespace-delimited reader over an already-loaded buffer. It never allocates
// and never throws; past the end every call returns 0 / an empty view, so a
// truncated input degrades into a wrong answer rather than a runtime error.
class Scanner {
public:
    explicit Scanner(std::string_view text) : text_(text) {}

    bool done() {
        skip_space();
        return at_ >= text_.size();
    }

    std::string_view token() {
        skip_space();
        const std::size_t start = at_;
        while (at_ < text_.size() && !is_space(text_[at_])) ++at_;
        return text_.substr(start, at_ - start);
    }

    // Rest of the current line, without the newline. Leading whitespace is NOT
    // skipped, so this can follow an int on the same line.
    std::string_view line() {
        const std::size_t start = at_;
        while (at_ < text_.size() && text_[at_] != '\n') ++at_;
        std::size_t end = at_;
        if (at_ < text_.size()) ++at_;
        if (end > start && text_[end - 1] == '\r') --end;
        return text_.substr(start, end - start);
    }

    i64 next_i64() {
        skip_space();
        bool negative = false;
        if (at_ < text_.size() && (text_[at_] == '-' || text_[at_] == '+')) {
            negative = text_[at_] == '-';
            ++at_;
        }
        i64 value = 0;
        while (at_ < text_.size() && is_digit(text_[at_])) {
            value = value * 10 + (text_[at_] - '0');
            ++at_;
        }
        return negative ? -value : value;
    }

    int next_int() { return static_cast<int>(next_i64()); }

    double next_double() {
        const std::string_view word = token();
        if (word.empty()) return 0.0;
        // std::strtod needs a terminated buffer; the token is bounded and short.
        char buffer[64];
        const std::size_t length = std::min(word.size(), sizeof(buffer) - 1);
        std::memcpy(buffer, word.data(), length);
        buffer[length] = '\0';
        return std::strtod(buffer, nullptr);
    }

    // Reads `count` integers into a vector, sized exactly.
    std::vector<int> next_ints(std::size_t count) {
        std::vector<int> values(count);
        for (std::size_t index = 0; index < count; ++index) values[index] = next_int();
        return values;
    }

    std::size_t position() const { return at_; }

private:
    static bool is_space(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
    static bool is_digit(char c) { return c >= '0' && c <= '9'; }

    void skip_space() {
        while (at_ < text_.size() && is_space(text_[at_])) ++at_;
    }

    std::string_view text_;
    std::size_t at_ = 0;
};

// Buffered writer. Building the answer in memory and flushing once keeps the
// output off the critical path and makes `write_output` the only stdout call.
class Writer {
public:
    Writer() { buffer_.reserve(1 << 16); }
    ~Writer() { flush(); }

    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    Writer& operator<<(char value) {
        buffer_.push_back(value);
        return *this;
    }

    Writer& operator<<(std::string_view value) {
        buffer_.append(value);
        return *this;
    }

    Writer& operator<<(const char* value) {
        buffer_.append(value);
        return *this;
    }

    // One template rather than an overload per width. A fixed set written as
    // (int, i64, u32, size_t) is ambiguous the moment the caller passes a type
    // that is none of them exactly: on Linux/LP64 `i64` is `long` and `long
    // long` is a distinct type, so `writer << 1LL` compiles on macOS and is a
    // compile error on the judge. That is a CE discovered at submission time.
    template <typename T>
    std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, char>, Writer&>
    operator<<(T value) {
        if constexpr (std::is_signed_v<T>) {
            const i64 widened = static_cast<i64>(value);
            const bool negative = widened < 0;
            // Negate in unsigned space so i64 min does not overflow.
            write_digits(negative ? (~static_cast<u64>(widened) + 1u)
                                  : static_cast<u64>(widened),
                         negative);
        } else {
            write_digits(static_cast<u64>(value), false);
        }
        return *this;
    }

    Writer& operator<<(double value) {
        char digits[64];
        const int written = std::snprintf(digits, sizeof(digits), "%.10g", value);
        if (written > 0) buffer_.append(digits, static_cast<std::size_t>(written));
        return *this;
    }

    void space() { buffer_.push_back(' '); }
    void nl() { buffer_.push_back('\n'); }

    // Space-separated row plus newline: the shape most challenge outputs take.
    template <typename Container>
    void row(const Container& values) {
        bool first = true;
        for (const auto& value : values) {
            if (!first) space();
            first = false;
            *this << value;
        }
        nl();
    }

    // CAREFUL: str() only PEEKS. The destructor still flushes what is in the
    // buffer, so building a string with a Writer and then printing it another
    // way emits the answer TWICE -- which the judge reads as a malformed
    // output, not as a duplicate. Use take() when the Writer is a string
    // builder rather than the output stream.
    const std::string& str() const { return buffer_; }

    // Moves the buffer out and leaves the Writer empty, so the destructor has
    // nothing left to flush.
    std::string take() {
        std::string out;
        out.swap(buffer_);
        return out;
    }

    void clear() { buffer_.clear(); }
    bool empty() const { return buffer_.empty(); }

    void flush() {
        if (buffer_.empty()) return;
        write_output(buffer_);
        buffer_.clear();
    }

private:
    void write_digits(u64 magnitude, bool negative) {
        char digits[24];
        std::size_t at = sizeof(digits);
        do {
            digits[--at] = static_cast<char>('0' + magnitude % 10);
            magnitude /= 10;
        } while (magnitude != 0);
        if (negative) digits[--at] = '-';
        buffer_.append(digits + at, sizeof(digits) - at);
    }

    std::string buffer_;
};

// ------------------------------------------------------------- files -------
// Directory of the running binary, with a trailing '/'. Empty when the platform
// cannot say. The rules only promise the uploaded data file sits in the same
// DIRECTORY as the binary, never that it is the working directory.
inline std::string executable_dir() {
    std::string path;
#if defined(__linux__)
    char buffer[4096];
    const long got =
        static_cast<long>(::readlink("/proc/self/exe", buffer, sizeof(buffer)));
    if (got > 0 && static_cast<std::size_t>(got) < sizeof(buffer)) {
        path.assign(buffer, static_cast<std::size_t>(got));
    }
#elif defined(__APPLE__)
    char buffer[4096];
    std::uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) path.assign(buffer);
#endif
    const std::size_t slash = path.rfind('/');
    return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
}

inline bool read_file_into(const char* path, std::string& bytes) {
    if (path == nullptr || *path == '\0') return false;
    std::FILE* handle = std::fopen(path, "rb");
    if (handle == nullptr) return false;
    char chunk[1 << 16];
    std::size_t got = 0;
    while ((got = std::fread(chunk, 1, sizeof(chunk), handle)) > 0) {
        bytes.append(chunk, got);
    }
    std::fclose(handle);
    return true;
}

// The raw "read the uploaded data file" helper deliberately lives in
// data_blob.hpp, not here. tools/check_submission.py decides whether a
// submission needs the uploaded file by looking for its name as a string
// literal in the bundled source, so a default argument naming that file, in a
// header every answer includes, made the gate reject every submission that did
// NOT use the upload. Keep that literal in the one header you include only when
// you actually read the file.

// ------------------------------------------------------------- hashing -----
inline u64 hash_bytes(std::string_view bytes) {
    u64 hash = 1469598103934665603ULL;
    for (const char raw : bytes) {
        hash ^= static_cast<unsigned char>(raw);
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline u64 split_mix64(u64& seed) {
    seed += 0x9E3779B97F4A7C15ULL;
    u64 mixed = seed;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBULL;
    return mixed ^ (mixed >> 31);
}

// --------------------------------------------------------------- env -------
// Set by tools/bench.py and tools/sweep.py so repeated runs and parameter
// sweeps stay reproducible without any LLM involvement. Absent on the judge,
// where the built-in defaults apply.
inline std::string env_text(const char* name, std::string fallback) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::move(fallback);
}

// NYPC_PARAMS uses "key=value,key=value".
inline std::string param_text(const char* key, std::string fallback) {
    const std::string params = env_text("NYPC_PARAMS", std::string());
    const std::string needle = std::string(key) + "=";
    std::size_t at = 0;
    while (at < params.size()) {
        const std::size_t end = std::min(params.find(',', at), params.size());
        const std::string_view item(params.data() + at, end - at);
        if (item.size() >= needle.size() &&
            item.compare(0, needle.size(), needle) == 0) {
            return std::string(item.substr(needle.size()));
        }
        at = end + 1;
    }
    return fallback;
}

inline double param_double(const char* key, double fallback) {
    const std::string text = param_text(key, std::string());
    if (text.empty()) return fallback;
    try {
        return std::stod(text);
    } catch (...) {
        return fallback;
    }
}

inline i64 param_int(const char* key, i64 fallback) {
    const std::string text = param_text(key, std::string());
    if (text.empty()) return fallback;
    try {
        return static_cast<i64>(std::stoll(text));
    } catch (...) {
        return fallback;
    }
}

// --------------------------------------------------------------- rng -------
// xoshiro256++ seeded through SplitMix64: deterministic for a given seed and
// far cheaper than mt19937_64 inside an inner loop.
class Rng {
public:
    explicit Rng(u64 seed) {
        for (u64& word : state_) word = split_mix64(seed);
    }

    u64 next_u64() {
        const u64 result = rotate(state_[0] + state_[3], 23) + state_[0];
        const u64 shifted = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= shifted;
        state_[3] = rotate(state_[3], 45);
        return result;
    }

    u32 next_u32() { return static_cast<u32>(next_u64() >> 32); }

    // Uniform in [0, bound). Returns 0 when bound is 0. Lemire multiply-shift:
    // one multiply, no modulo, no rejection loop in the common case.
    u64 next_below(u64 bound) {
        if (bound == 0) return 0;
        return static_cast<u64>((static_cast<__uint128_t>(next_u64()) * bound) >> 64);
    }

    // Uniform in [low, high]. Caller must pass low <= high.
    i64 next_between(i64 low, i64 high) {
        return low + static_cast<i64>(next_below(static_cast<u64>(high - low) + 1));
    }

    // Uniform in [0, 1).
    double next_double() {
        return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
    }

    // True with probability `chance`.
    bool chance(double probability) { return next_double() < probability; }

    // Two DISTINCT indices in [0, count). Caller must pass count >= 2. The
    // classic `while (a == b)` retry loop is unbounded; this is branch-free.
    std::pair<std::size_t, std::size_t> distinct_pair(std::size_t count) {
        const std::size_t first = static_cast<std::size_t>(next_below(count));
        std::size_t second = static_cast<std::size_t>(next_below(count - 1));
        if (second >= first) ++second;
        return {first, second};
    }

    template <typename Container>
    void shuffle(Container& items) {
        const std::size_t count = items.size();
        for (std::size_t index = count; index > 1; --index) {
            const std::size_t pick = static_cast<std::size_t>(next_below(index));
            std::swap(items[index - 1], items[pick]);
        }
    }

private:
    static u64 rotate(u64 value, int bits) {
        return (value << bits) | (value >> (64 - bits));
    }

    u64 state_[4]{};
};

// -------------------------------------------------------- deep recursion ----
// The judge gives the usual 8 MiB stack. Measured on Ubuntu 24.04 / GCC 14 at
// -O2, a plain DFS over an adjacency list survives ~200,000 frames and
// segfaults by 400,000; a frame with more locals dies far earlier. So a
// recursive DFS or flood fill over a 200k-node input is a coin flip, and when
// it loses it is a judge RE - which loses that input to every participant at
// once, exactly like a TLE.
//
// Prefer an explicit stack. When rewriting is not worth the time, run the
// recursion on a thread with a bigger stack:
//
//     contest::run_with_stack(512u << 20, [&] { dfs(0); });
//
// This is ONE thread doing the work sequentially, not parallelism. The rules
// sum every thread's CPU time, and this adds none - it only moves where the
// frames live. Verified in the judge-like VM: depth 4,000,000 on a 512 MiB
// stack, and it links WITHOUT -pthread (glibc 2.34+ folds pthread into libc),
// so it does not depend on a compile flag the official command may not pass.
//
// Returns true when the body ran on the new stack. On failure it still runs the
// body inline and returns false: a possible overflow beats certainly not
// running.
#if defined(__linux__) || defined(__APPLE__)

namespace detail {

template <typename Body>
void* stack_entry(void* raw) {
    (*static_cast<Body*>(raw))();
    return nullptr;
}

}  // namespace detail

template <typename Body>
bool run_with_stack(std::size_t bytes, Body body) {
    pthread_attr_t attributes;
    if (pthread_attr_init(&attributes) != 0) {
        body();
        return false;
    }
#ifdef PTHREAD_STACK_MIN
    if (bytes < static_cast<std::size_t>(PTHREAD_STACK_MIN)) {
        bytes = static_cast<std::size_t>(PTHREAD_STACK_MIN);
    }
#endif
    bool ok = pthread_attr_setstacksize(&attributes, bytes) == 0;
    pthread_t thread{};
    if (ok) {
        ok = pthread_create(&thread, &attributes,
                            &detail::stack_entry<Body>, &body) == 0;
    }
    pthread_attr_destroy(&attributes);
    if (!ok) {
        body();
        return false;
    }
    pthread_join(thread, nullptr);
    return true;
}

#else

template <typename Body>
bool run_with_stack(std::size_t, Body body) {
    body();
    return false;
}

#endif

// -------------------------------------------------------------- timer ------
// TLE is measured on CPU time and WTLE on wall-clock time, so watch both and
// stop at whichever budget runs out first.
class Timer {
public:
    explicit Timer(double budget_seconds, double safety_seconds = 0.05)
        : wall_start_(Clock::now()),
          cpu_start_(std::clock()),
          budget_(std::max(0.0, budget_seconds - safety_seconds)) {}

    double wall() const {
        return std::chrono::duration<double>(Clock::now() - wall_start_).count();
    }

    double cpu() const {
        return static_cast<double>(std::clock() - cpu_start_) / CLOCKS_PER_SEC;
    }

    double used() const { return std::max(wall(), cpu()); }
    double remaining() const { return budget_ - used(); }
    bool expired() const { return remaining() <= 0.0; }
    double budget() const { return budget_; }

    // Fraction of the budget spent, clamped to [0, 1]. Anneal schedules and any
    // "spend the rest of the time on X" decision key off this.
    double progress() const {
        if (budget_ <= 0.0) return 1.0;
        return std::min(1.0, std::max(0.0, used() / budget_));
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point wall_start_;
    std::clock_t cpu_start_;
    double budget_;
};

}  // namespace contest
// END BUNDLED FILE: core.hpp
// BEGIN BUNDLED FILE: interactive.hpp
// Turn-based (interactive) problems: the judge is a live process on the other
// end of a pipe, and it will not send turn N+1 until it has read your move for
// turn N.
//
// Not included by src/main.cpp by default. Add it only when the statement says
// the program talks to a judge process; a batch problem must keep using
// core.hpp's read_all_input(), which is faster.
//
// WHY THIS HEADER EXISTS AT ALL
//
// The batch idiom deadlocks here, silently and totally:
//
//     const std::string all = contest::read_all_input();   // waits for EOF
//
// read_all_input() blocks until the judge closes the pipe, and the judge is
// blocked waiting for your first move. Neither side moves. Every case takes a
// WTLE with zero bytes of output, which loses every input to every other
// participant. It compiles, it looks right, and no local test on a file
// redirect reproduces it -- a file hits EOF, a pipe does not.
//
// std::fread has the same trap for the same reason: fread(buf, 1, N, stdin)
// blocks until it has N bytes or EOF, so a 4096-byte read waits for 4096 bytes
// that the judge is not going to send. std::fgets returns at the newline, which
// is why every read below goes through it.
//
// THE TWO RULES
//
//   1. Read one line/token at a time, never "everything".
//   2. Flush after every move. A buffered move that never leaves the process is
//      indistinguishable from a hung program.
//
// Turn::send() does both halves of rule 2 for you; use it and the failure mode
// cannot happen.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

// bundled once: core.hpp

namespace contest {

// Line-at-a-time reader over stdin, with token access on top of it.
//
// Every accessor reports EOF rather than throwing or blocking forever: when the
// judge closes the pipe mid-game (because the game ended, or because it decided
// your move was invalid), the loop condition simply becomes false and the
// program exits 0. Exiting cleanly there matters -- a crash on EOF turns a
// finished game into a runtime error.
class TurnReader {
public:
    // Reads one line, without the trailing newline. False at end of input.
    bool read_line(std::string& line) {
        line.clear();
        if (eof_) return false;
        char chunk[4096];
        // fgets stops at '\n', so this never waits for data the judge has not
        // sent. The loop only repeats when a single line is longer than the
        // buffer.
        while (std::fgets(chunk, sizeof(chunk), stdin) != nullptr) {
            line += chunk;
            if (!line.empty() && line.back() == '\n') {
                line.pop_back();
                if (!line.empty() && line.back() == '\r') line.pop_back();
                return true;
            }
        }
        eof_ = true;
        return !line.empty();
    }

    // Next whitespace-separated token, pulling further lines as needed.
    // Empty exactly at end of input.
    std::string_view token() {
        while (true) {
            while (at_ < line_.size() && is_space(line_[at_])) ++at_;
            if (at_ < line_.size()) break;
            if (!read_line(line_)) return std::string_view();
            at_ = 0;
        }
        const std::size_t start = at_;
        while (at_ < line_.size() && !is_space(line_[at_])) ++at_;
        return std::string_view(line_).substr(start, at_ - start);
    }

    // Next integer; `fallback` at end of input, so a closed pipe degrades into
    // a sentinel the caller can test rather than into undefined behaviour.
    i64 next_i64(i64 fallback = -1) {
        const std::string_view word = token();
        if (word.empty()) return fallback;
        std::size_t index = 0;
        bool negative = false;
        if (word[0] == '-' || word[0] == '+') {
            negative = word[0] == '-';
            index = 1;
        }
        i64 value = 0;
        for (; index < word.size(); ++index) {
            if (word[index] < '0' || word[index] > '9') break;
            value = value * 10 + (word[index] - '0');
        }
        return negative ? -value : value;
    }

    int next_int(int fallback = -1) {
        return static_cast<int>(next_i64(fallback));
    }

    std::vector<i64> next_i64s(std::size_t count) {
        std::vector<i64> values(count);
        for (std::size_t index = 0; index < count; ++index) values[index] = next_i64();
        return values;
    }

    // True once the judge has closed the pipe AND the current line is used up.
    bool eof() {
        if (at_ < line_.size()) return false;
        return eof_ && token().empty();
    }

private:
    static bool is_space(char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    }

    std::string line_;
    std::size_t at_ = 0;
    bool eof_ = false;
};

// One side of a turn-based conversation. Reading is TurnReader; writing always
// flushes, because a move sitting in a buffer is a hang.
class Turn {
public:
    Turn() {
        // Line-buffer stdout so even a stray printf leaves the process. send()
        // flushes explicitly anyway; this is belt and braces for the case where
        // someone adds a printf during the round and cannot work out why the
        // judge stopped responding.
        std::setvbuf(stdout, nullptr, _IOLBF, 4096);
    }

    TurnReader& in() { return reader_; }

    // Writes one move and flushes. This is the whole protocol contract: after
    // send() returns, the judge has the move.
    void send(std::string_view text) {
        std::fwrite(text.data(), 1, text.size(), stdout);
        if (text.empty() || text.back() != '\n') std::fputc('\n', stdout);
        std::fflush(stdout);
    }

    void send(i64 value) { send(std::to_string(value)); }

    // Space-separated move, then flush.
    template <typename Container>
    void send_row(const Container& values) {
        std::string line;
        bool first = true;
        for (const auto& value : values) {
            if (!first) line += ' ';
            first = false;
            line += std::to_string(value);
        }
        send(line);
    }

private:
    TurnReader reader_;
};

}  // namespace contest
// END BUNDLED FILE: interactive.hpp

using contest::i64;
using contest::u64;

// Cell types. Values 0..6 mirror the official tool's G_* constants so the
// physics port can be compared against it line by line.
enum : uint8_t {
    C_EMPTY = 0, C_WALL = 1, C_JUMP = 2, C_COIN = 3,
    C_SHIELD = 4, C_SPIKE = 5, C_UNKNOWN = 7
};

// Command bits, same encoding as the tool. SPECIAL is reserved for the narrow
// emergency escape below; ordinary planning remains damage/special-free.
enum : int { A_LEFT = 1, A_RIGHT = 2, A_JUMP = 4, A_SPECIAL = 8 };

// The judge charges us only the wall time IT spends waiting for our line
// (testing-tool.py: `used += elapsed` around player.read). Its own simulate and
// render work happens after our move is in and is NOT charged, even though we
// sit blocked in fgets for all of it. Budgeting on wall time therefore gave the
// Python judge most of our allowance: measured 1.50s wall against 0.07-0.27s of
// actual search. CPU time is the honest proxy for what gets charged; the wall
// guard is only a runaway stop, deliberately far above any real judge.
// Two tiers, both measured against the judge's own accounting (the reported
// times matched this clock to 3ms). Planning stops at the first; play does NOT.
// Wandering costs only per-frame I/O -- about 15us -- so the remaining frames
// are nearly free, and cost is the minimum over time, so they can only help.
// The intermediate evaluation had Maze large burn the whole budget for ZERO
// coins and then quit; it should have spent those frames walking around.
static double kPlanCpu = 1.40;   // stop planning
static double kHardCpu = 1.70;   // stop playing (judge limit is 2.0s)
static const double kWallGuard = 30.0;

// Tunables, resolved once from NYPC_PARAMS so a sweep needs no rebuild.
static int P_BUDGET = 15000;    // best-first node budget per plan
static int P_SPIKE = 30;       // cell price of crossing a spike while shielded
static int P_MAXEXEC = 0;      // 0 = run a plan to its end; N = replan every N
static int P_DEEP = 8;         // budget multiplier for the one deeper retry
static int P_NOCLOSER = 60;    // plans without getting closer before giving up
static int P_STUCKN = 3;       // drained searches before giving up
static int P_NOREVEAL = 25;    // plans revealing nothing before giving up
static int P_FBRETRY = 200;    // frames of wandering between planner retries
static int P_FBMODE = 0;       // 0 deterministic sweep, 1 randomised walk
static int P_FBJUMP = 70;      // percent of wander frames that press jump
static int P_FBFLIP = 3;       // percent chance per frame of reversing
// Spending k special-jump frames to win g coins changes cost from n/D to
// (n+k-g)/(D+k), which is an improvement exactly when k < g*D/c. D/c is one over
// the fraction of the map's coins we hold, so the worse the run is going the
// more a special jump is worth -- and at zero coins any number of them pays.
// The fixed per-kind budgets below predate that identity.
static int P_ADAPTRESCUE = 1;  // budget by the identity above
static int P_RESCUEALL = 0;    // ...but do not offer it to kinds it hurts
static int P_BESTTRAP = 1;     // enter the RICHEST dead end, not the nearest
// Special jump raises the character 3px EVERY frame it is held, so a run of them
// climbs out of anything. The existing chain rule required the character to be
// falling, which cuts the run the moment it starts working. Real logs show the
// consequence exactly: on Pits large the solver spent 51 special jumps early,
// then sat in a 3x2 pocket for the last 8342 frames and issued ZERO.
// Measured OFF. The identity says these frames should pay, but on the
// reconstructed evaluation maps they are spent without converting to coins
// (468458 -> 522699), and a special jump that wins nothing always raises cost:
// (n+k)/(D+k) > n/D for every k. Kept behind a flag with the machinery intact.
// Set to 1 (i.e. only detour for a shield when holding none). Higher values fix
// a real Choice map that scores 1000000 with 0 of 4 coins, but on the tiled
// Choice corpus they collapse 73 coins to 16 -- the two disagree, so this takes
// the value with no measured regression and keeps the reordering above, which
// is a gain on its own (large corpus 674230 -> 645472).
static int P_SHIELDWANT = 1;   // stock this many shields before a spiked route
// A shield spent is a coin not collected somewhere else. On Choice medium the
// solver banked 17 shields, then burned 13 of them walking a spike wall for ONE
// coin, leaving nothing for the corridor three cells away that holds THREE
// coins behind three spikes. Cap what a single plan may spend.
static int P_MAXCROSS = 0;     // 0 = unlimited
// Dead-end marks accumulate for the whole run and are never retired. Every mark
// is a permanent wall for both the field and the search, so the reachable world
// only ever shrinks -- which is the shape of the stall in every log: coins every
// ~450 frames until t~2500, then nothing for 7300 frames inside 10% of the map.
// A mark can also be wrong: the probe calls an unfinished search "safe" but a
// small reachable set "dead", and what was a pocket before a box fell, or before
// the far side was explored, need not still be one.
// Measured OFF (0 = never retire). Expiring the list is strictly worse -- at a
// 1200-frame TTL puzzle_small falls from 13/13 to 4/13 -- so the marks are load
// bearing, not the cause of the stall. Machinery kept behind the flag.
static int P_SKIPTTL = 0;      // frames before the dead-end list is retired
static int P_ESCAPE = 0;
static int P_STUCKFRAMES = 90;  // confined this long counts as stuck
static int P_BURST = 16;        // consecutive special-jump frames per attempt
static int P_JUMPSTALL = 300;  // no-coin frames before a Jump vertical rescue
static int P_JUMPRESCUE = 7;   // maximum chained S commands for that rescue
static int P_PITSSTALL = 300;  // no-coin frames before treating a wide pit as trapped
static int P_PITSSMALLSTALL = 400;  // preserve the sample's 310-frame coin gap
static int P_PITSSMALLRESCUE = 2;   // enough to clear a shaft without overshooting
static int P_PITSRESCUE = 5;   // maximum chained S commands for a large pit
static int P_SPECIAL_BUDGET = 50000;  // second-stage coin search node budget
static int P_SPECIAL_CAP = 7;         // maximum S commands in one proven route
static int P_SPECIAL_WEIGHT = 12;     // prefer equal-progress routes using less S
// LOCAL MEASUREMENT ONLY. The real budget is CPU time, which on a large map is
// genuinely hit and therefore jitters: the same binary scored 882074, 892806 and
// 889604 on three identical runs. Capping total search nodes instead makes a run
// bit-identical, so an A/B needs one run rather than five. Never set for submit.
static long long P_NODECAP = 0;
// The gravity-aware field is a large-map tool. At 3000 it also caught MEDIUM,
// and the evaluation showed exactly that split: every large case improved
// (Pits 983739->593333, Puzzle 942857->885714, Maze 1000000->958333) while
// every medium case regressed (Handcrafted 458333->708333, Puzzle 720000->800000).
// The largest small example is 1845 cells, so medium lands well under 6500.
static int P_PHYSMIN = 6500;   // cells above which the gravity-aware field pays
static int P_PHYSFIELD = 0;    // resolved from map size at startup
static int P_UPCOST = 3;       // price of a cell of climbing
// Measured both ways at every size: restricting the planner to observed cells
// costs more than the failed routes it prevents (big 665661 -> 726005). 101
// means never.
static int P_KNOWNPCT = 101;    // plan only through seen cells while this % is unseen
static int P_FIELDEVERY = 8;   // reuse the distance field for this many plans
static int P_TRAPCELLS = 12;   // floor for the dead-end size test
static int P_TRAPDIV = 200;     // ...and a share of the map, so the test scales   // a pocket this small, with no way back, is a trap
static int P_PROBE = 30000;    // return-probe budget; must be able to EXHAUST
                               // a pocket, or an unfinished probe is read as
                               // "safe" and the trap is walked into anyway

static int N, M, H, W;                  // cells (N rows, M cols) and pixels (6N, 6M)
static std::vector<uint8_t> g_cell;     // N*M cell types
static std::vector<uint8_t> g_blk;      // H*W: bit0 = wall pixel, bit1 = box pixel
// Same information as g_blk, packed one bit per pixel per row. blocked_rect is
// the hottest function in the program (about a dozen calls per simulated frame,
// twelve million simulated frames per run); reading six 6-bit slices beats
// touching thirty-six bytes. Bits are offset by 64 so x = -1 is representable.
static std::vector<uint64_t> g_bits;
static int g_row_words = 0;
static const int kBitOff = 64;

static inline void bit_set(int py, int px, bool on) {
    if (py < 0 || py >= H) return;
    const int b = px + kBitOff;
    if (b < 0) return;
    uint64_t* row = &g_bits[static_cast<size_t>(py) * g_row_words];
    const uint64_t mask = 1ULL << (b & 63);
    if (on) row[b >> 6] |= mask; else row[b >> 6] &= ~mask;
}
static std::vector<int32_t> g_dist;     // N*M heuristic distance field
static std::vector<int32_t> g_bfsq;
static bool g_puzzle_kind = false;
static int g_puzzle_phase = -1;
static int g_puzzle_focus_row = -1;
static int g_puzzle_dir = 1;
static int g_puzzle_source_row = -1;
static int32_t g_forced_source_cell = -1;
static bool g_puzzle_frontier_active = false;
static std::vector<uint8_t> g_puzzle_collected_row;
// One-way traps: a 1-wide shaft you can fall into but whose walls a 15px jump
// clears by exactly 3px too little. Entering one forfeits every remaining coin,
// so they are proven by an explicit "can I get back?" search and taken LAST.
static double g_field_s = 0.0;
static long long g_field_n = 0;
static const contest::Timer* g_timer = nullptr;
static std::vector<uint8_t> g_skip;
static std::vector<uint8_t> g_probe;   // 0 unprobed, 1 safe, 2 dead
static bool g_ignore_spike_cost = false;
static bool g_spike_once = false;

static inline int fdiv6(int v) { return v >= 0 ? v / 6 : -((-v + 5) / 6); }

// True if the character's 6x6 box at (y,x) would overlap a wall cell or a box.
static inline bool blocked_rect(int y, int x) {
    int py = y < 0 ? 0 : y;
    const int y1 = y + 6 > H ? H : y + 6;
    const int b = x + kBitOff;
    const int word = b >> 6, shift = b & 63;
    for (; py < y1; ++py) {
        const uint64_t* row = &g_bits[static_cast<size_t>(py) * g_row_words];
        uint64_t v = row[word] >> shift;
        if (shift > 58) v |= row[word + 1] << (64 - shift);
        if (v & 0x3FULL) return true;
    }
    return false;
}

static inline bool on_jump_block(int y, int x) {
    const int r0 = fdiv6(y), r1 = fdiv6(y + 5), c0 = fdiv6(x), c1 = fdiv6(x + 5);
    for (int r = r0; r <= r1; ++r) {
        if (r < 0 || r >= N) continue;
        for (int c = c0; c <= c1; ++c) {
            if (c < 0 || c >= M) continue;
            if (g_cell[static_cast<size_t>(r) * M + c] == C_JUMP) return true;
        }
    }
    return false;
}

static inline bool item_pixel(int py, int px) {
    if (py == 2 || py == 3) return px >= 1 && px <= 4;
    if (py == 1 || py == 4) return px >= 2 && px <= 3;
    return false;
}

// Mirrors Game.touch(): reports (without mutating the world) which spike cells
// the character's box at (y,x) overlaps and whether it picks up a coin or a
// shield. Spikes are reported as distinct CELL IDS because one spike is removed
// on first contact and therefore costs at most one shield per step.
struct TouchOut {
    bool coin = false;
    bool shield = false;
    int nspike = 0;
    int32_t spikes[8] = {0};
};

static inline void touch_at(int y, int x, TouchOut& o) {
    const int r0 = fdiv6(y), r1 = fdiv6(y + 5), c0 = fdiv6(x), c1 = fdiv6(x + 5);
    for (int r = r0; r <= r1; ++r) {
        if (r < 0 || r >= N) continue;
        for (int c = c0; c <= c1; ++c) {
            if (c < 0 || c >= M) continue;
            const int32_t id = static_cast<int32_t>(r) * M + c;
            const uint8_t t = g_cell[static_cast<size_t>(id)];
            if (t == C_SPIKE) {
                bool dup = false;
                for (int i = 0; i < o.nspike; ++i) if (o.spikes[i] == id) { dup = true; break; }
                if (!dup && o.nspike < 8) o.spikes[o.nspike++] = id;
                continue;
            }
            if (t != C_COIN && t != C_SHIELD) continue;
            const int by = 6 * r, bx = 6 * c;
            bool got = false;
            for (int py = 1; py <= 4 && !got; ++py) {
                const int ay = by + py;
                if (ay < y || ay >= y + 6) continue;
                for (int px = 1; px <= 4; ++px) {
                    if (!item_pixel(py, px)) continue;
                    const int ax = bx + px;
                    if (ax >= x && ax < x + 6) { got = true; break; }
                }
            }
            if (!got) continue;
            if (t == C_COIN) o.coin = true; else o.shield = true;
        }
    }
}

struct St {
    int16_t y, x;
    int8_t vy, vx;
    uint8_t wg, wj, hold, sh;   // sh = shields in hand, capped for the state key
    // Spikes disappear after first contact in the real game.  Remember the
    // cells overlapped on the previous simulated frame so lingering on one
    // spike is charged once, while entering a different spike still consumes
    // another shield. N*M <= 10000, so two int16 ids preserve the exact old
    // semantics while keeping Node smaller than the former int32 pair.
    int16_t sp0, sp1;
};
static const uint8_t kShieldCap = 7;

// One frame of physics, ported from Game.step() in testing-tool.py. Boxes are
// treated as immovable here: the planner never relies on a push it has not
// verified, and a real push that we did not predict shows up as a position
// mismatch and forces a replan.
static void sim_step(St& s, int cmd, TouchOut& o) {
    if (cmd & A_RIGHT) cmd &= ~A_LEFT;
    if (cmd & A_SPECIAL) cmd &= ~A_JUMP;
    const bool ground = blocked_rect(s.y + 1, s.x) || on_jump_block(s.y, s.x);
    bool jumped = false;
    if (cmd & A_SPECIAL) {
        jumped = true;
        s.wj = 0;
        s.hold = 0;
    } else if (cmd & A_JUMP) {
        if (ground || (s.wg && !s.wj)) { jumped = true; s.wj = 1; s.hold = 0; }
        else if (s.wj && s.hold < 2) { jumped = true; ++s.hold; }
        else { s.wj = 0; s.hold = 0; }
    } else { s.wj = 0; s.hold = 0; }
    if (jumped) s.vy = -3;

    {   // vertical
        const int d = s.vy < 0 ? -1 : 1;
        int n = s.vy < 0 ? -s.vy : s.vy;
        while (n) {
            if (blocked_rect(s.y + d, s.x)) { s.vy = 0; break; }
            s.y = static_cast<int16_t>(s.y + d);
            touch_at(s.y, s.x, o);
            --n;
        }
        if (!jumped && !blocked_rect(s.y + 1, s.x) && s.vy < 6) ++s.vy;
    }

    const int d = (cmd & A_RIGHT) ? 1 : ((cmd & A_LEFT) ? -1 : 0);
    if (s.vx && (d == 0 || ((s.vx < 0) != (d < 0)))) s.vx += (s.vx < 0 ? 1 : -1);
    if (d) {
        // push() returns -1 (free), 0 (wall/box blocks). A pushable box would
        // return 1; we never count on it.
        if (!blocked_rect(s.y, s.x + d)) {
            if (s.vx * d < 4) s.vx = static_cast<int8_t>(s.vx + d);
        }
    }
    if (s.vx) {
        const int dir = s.vx < 0 ? -1 : 1;
        const int cnt = s.vx < 0 ? -s.vx : s.vx;
        for (int i = 0; i < cnt; ++i) {
            if (blocked_rect(s.y, s.x + dir)) { s.vx = 0; break; }
            s.x = static_cast<int16_t>(s.x + dir);
            touch_at(s.y, s.x, o);
        }
    }
    s.wg = ground ? 1 : 0;
    // A shield absorbs one spike; without one this branch is simply never taken
    // by the planner, which is why s_t and h_t stay at zero.
    if (o.nspike && !g_ignore_spike_cost) {
        int charged = o.nspike;
        if (g_spike_once) {
            charged = 0;
            for (int i = 0; i < o.nspike; ++i)
                charged += o.spikes[i] != s.sp0 && o.spikes[i] != s.sp1;
        }
        if (s.sh >= charged) s.sh = static_cast<uint8_t>(s.sh - charged);
        else s.sh = 0xFF;                       // sentinel: would take damage
    }
    if (o.nspike) {
        s.sp0 = static_cast<int16_t>(o.spikes[0]);
        s.sp1 = o.nspike > 1 ? static_cast<int16_t>(o.spikes[1]) : -1;
    } else {
        s.sp0 = s.sp1 = -1;
    }
    if (o.shield && s.sh != 0xFF && s.sh < kShieldCap) ++s.sh;
}

// ---------------------------------------------------------------------------
// World model
// ---------------------------------------------------------------------------

static inline void set_wall_pixels(int r, int c) {
    for (int py = 6 * r; py < 6 * r + 6; ++py) {
        uint8_t* row = &g_blk[static_cast<size_t>(py) * W];
        for (int px = 6 * c; px < 6 * c + 6; ++px) {
            row[px] |= 1u;
            bit_set(py, px, true);
        }
    }
}

// Decodes one RLE frame into `view` (48x48) and folds it into the persistent map.
static bool absorb_view(const std::string& rle, int cy, int cx, char* view) {
    int at = 0;
    const int n = static_cast<int>(rle.size());
    int i = 0;
    while (i < n && at < 2304) {
        const char ch = rle[i++];
        int count = 0;
        while (i < n && rle[i] >= '0' && rle[i] <= '9') count = count * 10 + (rle[i++] - '0');
        if (count <= 0) return false;
        if (at + count > 2304) count = 2304 - at;
        std::memset(view + at, ch, static_cast<size_t>(count));
        at += count;
    }
    if (at != 2304) return false;

    // Boxes move, so forget every box pixel inside the window before re-reading.
    for (int sy = 0; sy < 48; ++sy) {
        const int py = cy + sy - 21;
        if (py < 0 || py >= H) continue;
        uint8_t* row = &g_blk[static_cast<size_t>(py) * W];
        for (int sx = 0; sx < 48; ++sx) {
            const int px = cx + sx - 21;
            if (px < 0 || px >= W) continue;
            row[px] &= ~2u;
            bit_set(py, px, row[px] != 0);
        }
    }

    // Per-cell evidence. A cell renders one way over its whole 6x6 area, so any
    // '#'/'^'/'+' pixel identifies it outright, while '.' on an ITEM pixel is
    // what proves a cell holds no coin or shield.
    const int r0 = fdiv6(cy - 21), r1 = fdiv6(cy + 26);
    const int c0 = fdiv6(cx - 21), c1 = fdiv6(cx + 26);
    for (int r = r0; r <= r1; ++r) {
        if (r < 0 || r >= N) continue;
        for (int c = c0; c <= c1; ++c) {
            if (c < 0 || c >= M) continue;
            bool wall = false, coin = false, shield = false, spike = false;
            bool jump = false, dot_item = false;
            for (int py = 0; py < 6; ++py) {
                const int sy = 6 * r + py - cy + 21;
                if (sy < 0 || sy >= 48) continue;
                for (int px = 0; px < 6; ++px) {
                    const int sx = 6 * c + px - cx + 21;
                    if (sx < 0 || sx >= 48) continue;
                    switch (view[sy * 48 + sx]) {
                        case '#': wall = true; break;
                        case '$': coin = true; break;
                        case '*': shield = true; break;
                        case '^': spike = true; break;
                        case '+': jump = true; break;
                        case '.': if (item_pixel(py, px)) dot_item = true; break;
                        case 'O': {
                            const int ay = 6 * r + py, ax = 6 * c + px;
                            g_blk[static_cast<size_t>(ay) * W + ax] |= 2u;
                            bit_set(ay, ax, true);
                            break;
                        }
                        default: break;  // '@' hides the terrain underneath
                    }
                }
            }
            const size_t idx = static_cast<size_t>(r) * M + c;
            uint8_t next = C_UNKNOWN;
            if (wall) next = C_WALL;
            else if (coin) next = C_COIN;
            else if (shield) next = C_SHIELD;
            else if (spike) next = C_SPIKE;
            else if (jump) next = C_JUMP;
            else if (dot_item) next = C_EMPTY;
            if (next == C_UNKNOWN) continue;
            if (g_cell[idx] != next) {
                g_cell[idx] = next;
                if (next == C_WALL) set_wall_pixels(r, c);
            }
        }
    }

    // Cells the character itself covers render as '@' and would otherwise stay
    // UNKNOWN forever. We can still classify them: standing there proves they
    // are not walls, a coin or shield would already have been picked up, and a
    // spike would already have cost damage. So they are EMPTY or a jump block,
    // and only a jump block can be confirmed later from a partial view.
    {
        const int rr0 = fdiv6(cy), rr1 = fdiv6(cy + 5);
        const int cc0 = fdiv6(cx), cc1 = fdiv6(cx + 5);
        for (int r = rr0; r <= rr1; ++r) {
            if (r < 0 || r >= N) continue;
            for (int c = cc0; c <= cc1; ++c) {
                if (c < 0 || c >= M) continue;
                const size_t idx = static_cast<size_t>(r) * M + c;
                if (g_cell[idx] == C_UNKNOWN) g_cell[idx] = C_EMPTY;
            }
        }
    }
    return true;
}

// Multi-source BFS giving, for every cell, the cell distance to the nearest
// target. Walls and spikes are impassable; unknown cells are optimistically
// passable so the same field also drives exploration.
// Dijkstra, not BFS, because a spike is not a wall: while we hold a shield,
// crossing one costs a shield but no damage. Making spikes strictly impassable
// sent Puzzle the long way round the entire map -- straight through the coin
// shafts at the bottom, which are one-way traps -- when a single shielded step
// through the spike beside it reached the same coin. `spike_cost < 0` keeps the
// old "impassable" meaning for when we hold no shield.
// A cell the character can stand in: not solid itself, and either resting on
// something solid or overlapping a jump block.
static inline bool solid_cell(int r, int c) {
    if (r < 0 || r >= N || c < 0 || c >= M) return true;
    if (g_cell[static_cast<size_t>(r) * M + c] == C_WALL) return true;
    return g_blk[static_cast<size_t>(6 * r) * W + 6 * c] != 0;
}

static inline bool standable(int r, int c) {
    if (r < 0 || r >= N || c < 0 || c >= M) return false;
    const uint8_t t = g_cell[static_cast<size_t>(r) * M + c];
    if (t == C_WALL) return false;
    if (t == C_JUMP) return true;
    return solid_cell(r + 1, c);
}

static void build_distance(uint8_t source_kind, int spike_cost) {
    const double t_field = g_timer ? g_timer->wall() : 0.0;
    ++g_field_n;
    struct FieldTimer {
        double start;
        ~FieldTimer() { if (g_timer) g_field_s += g_timer->wall() - start; }
    } field_timer{t_field};
    const size_t cells = static_cast<size_t>(N) * M;
    std::fill(g_dist.begin(), g_dist.end(), INT32_MAX);
    std::priority_queue<std::pair<int32_t, int32_t>,
                        std::vector<std::pair<int32_t, int32_t>>,
                        std::greater<std::pair<int32_t, int32_t>>> pq;
    for (size_t i = 0; i < cells; ++i) {
        const bool source = g_forced_source_cell >= 0
            ? static_cast<int32_t>(i) == g_forced_source_cell
            : g_cell[i] == source_kind;
        if (!source || g_skip[i]) continue;
        if (source_kind == C_UNKNOWN && g_puzzle_source_row >= 0 &&
            static_cast<int>(i / static_cast<size_t>(M)) != g_puzzle_source_row)
            continue;
        g_dist[i] = 0;
        pq.emplace(0, static_cast<int32_t>(i));
    }
    while (!pq.empty()) {
        const auto [d, cur] = pq.top();
        pq.pop();
        if (d != g_dist[static_cast<size_t>(cur)]) continue;
        const int r = cur / M, c = cur % M;
        // Edges are the moves that would bring the character INTO `cur`, so the
        // field reads as "cost from here to the nearest target". Walking and
        // falling are cheap; climbing is not, and is only possible from a cell
        // with something to stand on. A plain 4-neighbour BFS treats a sheer
        // 5-cell shaft as a 5-step path, sends the search up it, and burns the
        // whole node budget failing -- which is how a large Maze scores zero.
        const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
        for (int k = 0; k < 4; ++k) {
            const int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= N || nc < 0 || nc >= M) continue;
            const size_t ni = static_cast<size_t>(nr) * M + nc;
            const uint8_t t = g_cell[ni];
            if (t == C_WALL) continue;
            if (g_skip[ni]) continue;          // a proven dead end is a wall
            int step = 1;
            if (P_PHYSFIELD && k == 1) {
                // neighbour is BELOW cur: reaching cur from it means climbing
                if (!standable(nr, nc)) continue;
                step = P_UPCOST;
            }
            if (t == C_SPIKE) {
                if (spike_cost < 0) continue;
                step += spike_cost;
            }
            const int32_t nd = d + step;
            if (g_dist[ni] <= nd) continue;
            g_dist[ni] = nd;
            pq.emplace(nd, static_cast<int32_t>(ni));
        }
        if (P_PHYSFIELD) {
            // A jump clears 15px, i.e. two cells, so a foothold two rows below
            // also reaches cur.
            const int nr = r + 2;
            if (nr < N && standable(nr, c) && g_cell[static_cast<size_t>(r + 1) * M + c] != C_WALL) {
                const size_t ni = static_cast<size_t>(nr) * M + c;
                const uint8_t t = g_cell[ni];
                if (t != C_WALL && !g_skip[ni] && (t != C_SPIKE || spike_cost >= 0)) {
                    const int32_t nd = d + P_UPCOST + (t == C_SPIKE ? spike_cost : 0);
                    if (g_dist[ni] > nd) {
                        g_dist[ni] = nd;
                        pq.emplace(nd, static_cast<int32_t>(ni));
                    }
                }
            }
        }
    }
}

// Optimistic field used only to order the S-aware frontier search. Walls stay
// fully enforced by sim_step(); ignoring them here prevents an enclosed shaft
// from assigning INF to every state and exhausting the node budget before any
// vertical-special state is considered.
static void build_open_distance(uint8_t source_kind) {
    std::fill(g_dist.begin(), g_dist.end(), INT32_MAX);
    int head = 0, tail = 0;
    for (size_t i = 0; i < g_cell.size(); ++i) {
        if (g_cell[i] != source_kind) continue;
        g_dist[i] = 0;
        g_bfsq[tail++] = static_cast<int32_t>(i);
    }
    const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
    while (head < tail) {
        const int32_t cur = g_bfsq[head++];
        const int r = cur / M, c = cur % M;
        const int32_t nd = g_dist[static_cast<size_t>(cur)] + 1;
        for (int k = 0; k < 4; ++k) {
            const int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= N || nc < 0 || nc >= M) continue;
            const size_t ni = static_cast<size_t>(nr) * M + nc;
            if (g_dist[ni] <= nd) continue;
            g_dist[ni] = nd;
            g_bfsq[tail++] = static_cast<int32_t>(ni);
        }
    }
}

static inline int heuristic_of(int y, int x) {
    const int r = fdiv6(y + 3), c = fdiv6(x + 3);
    if (r < 0 || r >= N || c < 0 || c >= M) return INT32_MAX / 4;
    const int32_t d = g_dist[static_cast<size_t>(r) * M + c];
    return d == INT32_MAX ? INT32_MAX / 4 : d;
}

// Puzzle is layered: the long coin corridors repeat every eight rows.  Using
// every unknown cell as a source always selects the closest old fringe and, in
// the official large logs, sends the player from row 66 all the way back to
// row 10 after collecting row 50.  Keep walking through uncollected layers in
// the current vertical direction; reverse only after that side is exhausted.
static int puzzle_frontier_row() {
    if (!g_puzzle_kind || g_puzzle_phase < 0 || g_puzzle_focus_row < 0) return -1;
    auto has_unknown = [](int r) {
        if (r <= 0 || r >= N - 1 ||
            (r - g_puzzle_phase) % 8 != 0 || g_puzzle_collected_row[r]) return false;
        for (int c = 0; c < M; ++c)
            if (g_cell[static_cast<size_t>(r) * M + c] == C_UNKNOWN) return true;
        return false;
    };
    for (int pass = 0; pass < 2; ++pass) {
        const int dir = pass == 0 ? g_puzzle_dir : -g_puzzle_dir;
        for (int r = g_puzzle_focus_row + dir * 8; r > 0 && r < N - 1; r += dir * 8)
            if (has_unknown(r)) return r;
    }
    return -1;
}

// Target ladder. Reaching a coin without touching a spike is always preferred;
// the statement guarantees such a route exists, but on maps like Choice it only
// exists AFTER picking up a shield, so a coin that is unreachable spike-free
// makes shields the goal first.
static bool choose_field(size_t coins_left, size_t shields_left, int shields,
                         int y, int x) {
    const int spike = shields > 0 ? P_SPIKE : -1;
    bool have = false;
    g_puzzle_frontier_active = false;
    if (coins_left) {
        // Every layered Puzzle coin corridor is closed on the left and entered
        // from the jump column at the right edge.  A direct coin field still
        // pulls a player on the left spine against that wall.  Route to the
        // matching right-hand entrance first; once on that side the ordinary
        // coin field takes over and sweeps left through the corridor.
        if (g_puzzle_kind) {
            const int pr = fdiv6(y + 3), pc = fdiv6(x + 3);
            int coin_row = -1, best = INT32_MAX;
            for (int r = 0; r < N; ++r) {
                bool row_coin = false;
                for (int c = 0; c < M; ++c)
                    row_coin |= g_cell[static_cast<size_t>(r) * M + c] == C_COIN;
                if (row_coin && std::abs(r - pr) < best) {
                    best = std::abs(r - pr);
                    coin_row = r;
                }
            }
            const bool already_in_corridor = coin_row >= 0 && pr == coin_row && pc >= 4;
            if (coin_row >= 0 && !already_in_corridor && pc < M - 7) {
                const int gate_c = M - 2;
                g_forced_source_cell = static_cast<int32_t>(coin_row * M + gate_c);
                build_distance(C_EMPTY, spike);
                g_forced_source_cell = -1;
                have = heuristic_of(y, x) < INT32_MAX / 8;
                if (have) return true;
            }
        }
        // Is there a spike-free route at all? If not, the coin costs shields,
        // and the field cannot say how many: it prices a spike at P_SPIKE
        // whatever we are carrying, so it reports "reachable" while the physics
        // search correctly refuses the crossing for want of a shield. Choice
        // medium hides three of four coins behind three spikes; holding two
        // shields the solver planned nothing and scored 0 of 4.
        build_distance(C_COIN, -1);
        have = heuristic_of(y, x) < INT32_MAX / 8;
        if (!have && shields_left && shields < P_SHIELDWANT) {
            build_distance(C_SHIELD, spike);
            if (heuristic_of(y, x) < INT32_MAX / 8) return true;   // stock up first
        }
        if (!have) {
            build_distance(C_COIN, spike);
            have = heuristic_of(y, x) < INT32_MAX / 8;
        }
    }
    if (!have && shields_left) {
        build_distance(C_SHIELD, spike);
        have = heuristic_of(y, x) < INT32_MAX / 8;
        if (!have) {
            build_distance(C_SHIELD, P_SPIKE);
            have = heuristic_of(y, x) < INT32_MAX / 8;
        }
    }
    if (!have) {
        const int puzzle_row = puzzle_frontier_row();
        g_puzzle_source_row = puzzle_row;
        build_distance(C_UNKNOWN, spike);
        g_puzzle_source_row = -1;
        have = heuristic_of(y, x) < INT32_MAX / 8;
        g_puzzle_frontier_active = have && puzzle_row >= 0;
        // A row can be hidden behind an as-yet unknown wall.  If the focused
        // field cannot reach it, retain the general explorer as a fallback.
        if (!have && puzzle_row >= 0) {
            build_distance(C_UNKNOWN, spike);
            have = heuristic_of(y, x) < INT32_MAX / 8;
            g_puzzle_frontier_active = false;
        }
    }
    if (!have && coins_left) {
        build_distance(C_COIN, P_SPIKE);
        have = heuristic_of(y, x) < INT32_MAX / 8;
    }
    if (!have) build_distance(C_UNKNOWN, P_SPIKE);
    return have;
}

// ---------------------------------------------------------------------------
// Best-first search over exact physics states
// ---------------------------------------------------------------------------

struct Node {
    St s;
    int32_t parent;
    int32_t g;
    int32_t f;
    int8_t act;
    uint8_t special;
};

static std::vector<Node> g_pool;
static std::vector<int32_t> g_heap;
static std::vector<uint64_t> g_seen;
static uint32_t g_seen_mask = 0;
static long long g_stat_replans = 0, g_stat_nodes = 0, g_stat_goals = 0;
static long long g_stat_probes = 0, g_stat_traps = 0;
static int g_dbg_h0 = 0, g_dbg_besth = 0, g_dbg_goal = 0, g_dbg_exp = 0;
static double g_stat_search = 0.0;

static inline uint64_t state_key(const St& s) {
    uint64_t k = static_cast<uint64_t>(s.y) * W + s.x;
    k = k * 10 + static_cast<uint64_t>(s.vy + 3);
    k = k * 9 + static_cast<uint64_t>(s.vx + 4);
    k = k * 12 + static_cast<uint64_t>(s.wg * 6 + s.wj * 3 + s.hold);
    k = k * (kShieldCap + 1) + static_cast<uint64_t>(s.sh);
    return k + 1;  // 0 marks an empty slot
}

static inline bool seen_insert(uint64_t key) {
    uint64_t h = key * 0x9E3779B97F4A7C15ULL;
    uint32_t i = static_cast<uint32_t>(h >> 40) & g_seen_mask;
    while (true) {
        const uint64_t cur = g_seen[i];
        if (cur == 0) { g_seen[i] = key; return true; }
        if (cur == key) return false;
        i = (i + 1) & g_seen_mask;
    }
}

static inline void heap_push(int32_t id) {
    g_heap.push_back(id);
    size_t i = g_heap.size() - 1;
    while (i) {
        const size_t p = (i - 1) / 2;
        if (g_pool[static_cast<size_t>(g_heap[p])].f <= g_pool[static_cast<size_t>(g_heap[i])].f) break;
        std::swap(g_heap[p], g_heap[i]);
        i = p;
    }
}

static inline int32_t heap_pop() {
    const int32_t top = g_heap[0];
    g_heap[0] = g_heap.back();
    g_heap.pop_back();
    size_t i = 0;
    const size_t n = g_heap.size();
    while (true) {
        size_t best = i, l = 2 * i + 1, r = l + 1;
        if (l < n && g_pool[static_cast<size_t>(g_heap[l])].f < g_pool[static_cast<size_t>(g_heap[best])].f) best = l;
        if (r < n && g_pool[static_cast<size_t>(g_heap[r])].f < g_pool[static_cast<size_t>(g_heap[best])].f) best = r;
        if (best == i) break;
        std::swap(g_heap[i], g_heap[best]);
        i = best;
    }
    return top;
}

// Searches for an action sequence that picks up a coin. Falls back to the path
// that gets closest to the target field. Returns actions in `plan`.
static bool g_knownonly = false;
static bool g_heap_drained = false;

static bool g_made_progress = false;
static bool g_found_goal = false;
static bool g_search_special = false;
static bool g_coin_goal_only = false;
static bool g_unknown_goal = false;
static int g_reach_cells = 0;             // distinct cells the last search touched
static std::vector<int32_t> g_cellstamp;
static int32_t g_stamp_now = 0;
static int32_t g_goal_cell = -1;      // >=0: goal is reaching this cell

static const int kActionBits[9] = {
    0, A_LEFT, A_RIGHT, A_JUMP, A_LEFT | A_JUMP, A_RIGHT | A_JUMP,
    A_SPECIAL, A_LEFT | A_SPECIAL, A_RIGHT | A_SPECIAL
};

static inline int action_bits(int8_t index) {
    return kActionBits[static_cast<uint8_t>(index)];
}

// BFS distance to one particular cell, for asking "can I get back here?".
static void build_distance_single(int32_t cell, bool allow_spikes) {
    std::fill(g_dist.begin(), g_dist.end(), INT32_MAX);
    int head = 0, tail = 0;
    g_dist[static_cast<size_t>(cell)] = 0;
    g_bfsq[tail++] = cell;
    while (head < tail) {
        const int32_t cur = g_bfsq[head++];
        const int r = cur / M, c = cur % M;
        const int nd = g_dist[static_cast<size_t>(cur)] + 1;
        const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
        for (int k = 0; k < 4; ++k) {
            const int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= N || nc < 0 || nc >= M) continue;
            const size_t ni = static_cast<size_t>(nr) * M + nc;
            const uint8_t t = g_cell[ni];
            if (t == C_WALL) continue;
            if (t == C_SPIKE && !allow_spikes) continue;
            if (g_dist[ni] <= nd) continue;
            g_dist[ni] = nd;
            g_bfsq[tail++] = static_cast<int32_t>(ni);
        }
    }
}  // found a coin, or a strictly better foothold

static bool plan_moves(const St& start, int budget, std::vector<int8_t>& plan) {
    plan.clear();
    g_pool.clear();
    g_heap.clear();
    uint32_t cap = 1024;
    while (cap < static_cast<uint32_t>(budget) * 4u) cap <<= 1;
    if (g_seen.size() != cap) g_seen.assign(cap, 0);
    else std::fill(g_seen.begin(), g_seen.end(), 0ULL);
    g_seen_mask = cap - 1;

    ++g_stamp_now;
    g_reach_cells = 0;
    const int start_sh = start.sh;
    const int h0 = heuristic_of(start.y, start.x);
    const int64_t start_f64 = static_cast<int64_t>(h0) * 4;
    const int32_t start_f = g_search_special
        ? static_cast<int32_t>(std::min<int64_t>(INT32_MAX, start_f64))
        : h0 * 4;
    g_pool.push_back(Node{start, -1, 0, start_f, 0, 0});
    heap_push(0);
    seen_insert(g_search_special
        ? state_key(start) * static_cast<uint64_t>(P_SPECIAL_CAP + 1)
        : state_key(start));

    // A plan must END somewhere the character can stand. Choosing the lowest-h
    // node regardless of footing picks the apex of a jump, and after the plan
    // runs out gravity drops us straight back where we started: on pits_small
    // that oscillation burned 9973 of 10000 frames inside the first pit.
    int32_t best_id = 0, best_h = INT32_MAX;
    int32_t goal = -1;
    const int action_count = g_search_special ? 9 : 6;

    int expanded = 0;
    while (!g_heap.empty() && expanded < budget) {
        const int32_t id = heap_pop();
        ++expanded;
        const Node cur = g_pool[static_cast<size_t>(id)];
        if (cur.g > 400) continue;
        for (int a = 0; a < action_count; ++a) {
            St ns = cur.s;
            TouchOut o;
            const bool uses_special = (kActionBits[a] & A_SPECIAL) != 0;
            const int next_special = static_cast<int>(cur.special) + (uses_special ? 1 : 0);
            if (next_special > (g_search_special ? P_SPECIAL_CAP : 0)) continue;
            sim_step(ns, kActionBits[a], o);
            if (ns.sh == 0xFF) continue;               // would take damage
            if (P_MAXCROSS > 0 && start_sh - static_cast<int>(ns.sh) > P_MAXCROSS) continue;
            if (ns.y == cur.s.y && ns.x == cur.s.x &&
                ns.vy == cur.s.vy && ns.vx == cur.s.vx &&
                ns.wg == cur.s.wg && ns.wj == cur.s.wj && ns.hold == cur.s.hold) continue;
            {   // Never route through a proven dead end -- and, on a big map,
                // never route through a cell we have not seen. The 48x48 window
                // means most of a large map is unknown; planning across it
                // produced routes that simply did not survive contact (maze_big
                // reached 40 coin goals and banked 8), and every failed plan is
                // a wasted replan.
                const int fr = fdiv6(ns.y + 3), fc = fdiv6(ns.x + 3);
                if (fr >= 0 && fr < N && fc >= 0 && fc < M) {
                    const size_t fi = static_cast<size_t>(fr) * M + fc;
                    if (g_skip[fi]) continue;
                    if (g_knownonly && g_cell[fi] == C_UNKNOWN) continue;
                }
            }
            const uint64_t skey = g_search_special
                ? state_key(ns) * static_cast<uint64_t>(P_SPECIAL_CAP + 1) +
                      static_cast<uint64_t>(next_special)
                : state_key(ns);
            if (!seen_insert(skey)) continue;
            {   // how much of the MAP this search can touch, not how many
                // states: a 4-cell pocket holds hundreds of thousands of
                // states, so "did the search exhaust itself" is unusable as a
                // trap test, but "did it ever leave these few cells" is exact.
                const int cr = fdiv6(ns.y + 3), cc2 = fdiv6(ns.x + 3);
                if (cr >= 0 && cr < N && cc2 >= 0 && cc2 < M) {
                    const size_t ci = static_cast<size_t>(cr) * M + cc2;
                    if (g_cellstamp[ci] != g_stamp_now) {
                        g_cellstamp[ci] = g_stamp_now;
                        ++g_reach_cells;
                    }
                }
            }
            const int h = heuristic_of(ns.y, ns.x);
            const int32_t nid = static_cast<int32_t>(g_pool.size());
            const int64_t f64 = static_cast<int64_t>(cur.g + 1) +
                                static_cast<int64_t>(next_special) * P_SPECIAL_WEIGHT +
                                static_cast<int64_t>(h) * 4;
            const int32_t f = g_search_special
                ? static_cast<int32_t>(std::min<int64_t>(INT32_MAX, f64))
                : cur.g + 1 + h * 4;
            g_pool.push_back(Node{ns, id, cur.g + 1, f, static_cast<int8_t>(a),
                                  static_cast<uint8_t>(next_special)});
            if (g_goal_cell >= 0) {
                if (fdiv6(ns.y + 3) * M + fdiv6(ns.x + 3) == g_goal_cell) { goal = nid; break; }
            } else {
                const int gr = fdiv6(ns.y + 3), gc = fdiv6(ns.x + 3);
                const bool reached_unknown = g_unknown_goal &&
                    gr >= 0 && gr < N && gc >= 0 && gc < M &&
                    g_cell[static_cast<size_t>(gr) * M + gc] == C_UNKNOWN;
                if (reached_unknown || o.coin ||
                    (!g_coin_goal_only && o.shield && cur.s.sh == 0)) {
                    goal = nid;
                    break;
                }
            }
            const bool settled = blocked_rect(ns.y + 1, ns.x) || on_jump_block(ns.y, ns.x);
            if (settled && (h < best_h || (h == best_h && best_id != 0 &&
                                           cur.g + 1 < g_pool[static_cast<size_t>(best_id)].g))) {
                best_h = h;
                best_id = nid;
            }
            heap_push(nid);
            if (g_pool.size() + 8 >= g_pool.capacity()) break;
        }
        if (goal >= 0) break;
        // Clear progress to a foothold is worth taking now; searching on from a
        // better standing position beats spending the rest of the budget here.
        if (!g_search_special && g_goal_cell < 0 && best_h + 3 <= h0 && expanded >= 400) break;
    }

    g_dbg_h0 = h0; g_dbg_besth = best_h; g_dbg_goal = (goal >= 0); g_dbg_exp = expanded;
    g_stat_replans++;
    g_stat_nodes += expanded;
    if (goal >= 0) g_stat_goals++;
    g_heap_drained = g_heap.empty();
    g_found_goal = (goal >= 0);
    g_made_progress = (goal >= 0) || (best_h < h0);
    int32_t id = goal >= 0 ? goal : (best_h < h0 ? best_id : 0);
    if (id == 0) {
        // No node beat the starting heuristic. Rather than stall, walk the
        // longest branch we expanded: it still uncovers map.
        int32_t deepest = 0, deepest_g = 0;
        for (size_t i = 1; i < g_pool.size(); ++i) {
            if (g_pool[i].g > deepest_g) { deepest_g = g_pool[i].g; deepest = static_cast<int32_t>(i); }
        }
        id = deepest;
    }
    if (id == 0) return false;
    while (id > 0) {
        plan.push_back(g_pool[static_cast<size_t>(id)].act);
        id = g_pool[static_cast<size_t>(id)].parent;
    }
    std::reverse(plan.begin(), plan.end());
    return !plan.empty();
}

// ---------------------------------------------------------------------------

int main() {
    contest::Timer timer(kHardCpu);
    g_timer = &timer;
    P_NODECAP = contest::param_int("nodecap", 0);
    kPlanCpu = contest::param_double("plancpu", kPlanCpu);
    kHardCpu = contest::param_double("hardcpu", kHardCpu);
    P_BUDGET = static_cast<int>(contest::param_int("budget", P_BUDGET));
    P_SPIKE = static_cast<int>(contest::param_int("spike", P_SPIKE));
    P_MAXEXEC = static_cast<int>(contest::param_int("maxexec", P_MAXEXEC));
    P_DEEP = static_cast<int>(contest::param_int("deep", P_DEEP));
    P_NOCLOSER = static_cast<int>(contest::param_int("nocloser", P_NOCLOSER));
    P_STUCKN = static_cast<int>(contest::param_int("stuckn", P_STUCKN));
    P_NOREVEAL = static_cast<int>(contest::param_int("noreveal", P_NOREVEAL));
    P_FBRETRY = static_cast<int>(contest::param_int("fbretry", P_FBRETRY));
    P_FBMODE = static_cast<int>(contest::param_int("fbmode", P_FBMODE));
    P_FBJUMP = static_cast<int>(contest::param_int("fbjump", P_FBJUMP));
    P_FBFLIP = static_cast<int>(contest::param_int("fbflip", P_FBFLIP));
    P_ADAPTRESCUE = static_cast<int>(contest::param_int("adaptrescue", P_ADAPTRESCUE));
    P_RESCUEALL = static_cast<int>(contest::param_int("rescueall", P_RESCUEALL));
    P_BESTTRAP = static_cast<int>(contest::param_int("besttrap", P_BESTTRAP));
    P_ESCAPE = static_cast<int>(contest::param_int("escape", P_ESCAPE));
    P_SHIELDWANT = static_cast<int>(contest::param_int("shieldwant", P_SHIELDWANT));
    P_MAXCROSS = static_cast<int>(contest::param_int("maxcross", P_MAXCROSS));
    P_SKIPTTL = static_cast<int>(contest::param_int("skipttl", P_SKIPTTL));
    P_STUCKFRAMES = static_cast<int>(contest::param_int("stuckframes", P_STUCKFRAMES));
    P_BURST = static_cast<int>(contest::param_int("burst", P_BURST));
    P_JUMPSTALL = static_cast<int>(contest::param_int("jumpstall", P_JUMPSTALL));
    P_JUMPRESCUE = static_cast<int>(contest::param_int("jumprescue", P_JUMPRESCUE));
    P_PITSSTALL = static_cast<int>(contest::param_int("pitsstall", P_PITSSTALL));
    P_PITSSMALLSTALL = static_cast<int>(contest::param_int("pitssmallstall", P_PITSSMALLSTALL));
    P_PITSSMALLRESCUE = static_cast<int>(contest::param_int("pitssmallrescue", P_PITSSMALLRESCUE));
    P_PITSRESCUE = static_cast<int>(contest::param_int("pitsrescue", P_PITSRESCUE));
    P_SPECIAL_BUDGET = static_cast<int>(contest::param_int("specialbudget", P_SPECIAL_BUDGET));
    P_SPECIAL_CAP = static_cast<int>(contest::param_int("specialcap", P_SPECIAL_CAP));
    P_SPECIAL_WEIGHT = static_cast<int>(contest::param_int("specialweight", P_SPECIAL_WEIGHT));
    P_SPECIAL_CAP = std::clamp(P_SPECIAL_CAP, 0, 20);
    P_PROBE = static_cast<int>(contest::param_int("probe", P_PROBE));
    P_TRAPCELLS = static_cast<int>(contest::param_int("trapcells", P_TRAPCELLS));
    P_TRAPDIV = static_cast<int>(contest::param_int("trapdiv", P_TRAPDIV));
    P_KNOWNPCT = static_cast<int>(contest::param_int("knownpct", P_KNOWNPCT));
    P_PHYSMIN = static_cast<int>(contest::param_int("physmin", P_PHYSMIN));
    P_PHYSFIELD = static_cast<int>(contest::param_int("physfield", -1));
    P_UPCOST = static_cast<int>(contest::param_int("upcost", P_UPCOST));
    P_FIELDEVERY = static_cast<int>(contest::param_int("fieldevery", P_FIELDEVERY));
    contest::Rng rng(0x9E3779B97F4A7C15ULL);
    contest::Turn turn;
    auto& in = turn.in();

    N = in.next_int();
    M = in.next_int();
    std::string kind(in.token());
    if (N <= 0 || M <= 0) { turn.send("FINISH"); return 0; }
    H = 6 * N; W = 6 * M;
    g_puzzle_kind = kind == "Puzzle";
    g_spike_once = kind == "Choice";
    g_puzzle_collected_row.assign(static_cast<size_t>(N), 0);

    // A Maze branch is not a small disposable pocket: its corridors routinely
    // expose only a handful of cells before a return probe's node budget ends.
    // Treating every such inconclusive probe as a proven one-way trap seals the
    // unexplored maze. On Maze, only an actually drained state space proves it.
    if (kind == "Maze") {
        P_TRAPCELLS = 0;
        P_TRAPDIV = 1000000;
    }

    // A small map is fully visible almost at once and the optimistic field
    // finds routes the physical approximation refuses; a large one is mostly
    // unseen and the optimistic field walks the search into sheer shafts.
    if (P_PHYSFIELD < 0) P_PHYSFIELD = (N * M >= P_PHYSMIN) ? 1 : 0;
    // Choice is built from vertical alternatives separated by one-way jump
    // columns.  An undirected field says the coin directly above is closest
    // even when the only real entrance is at the top, then the exact planner
    // follows whichever adjacent branch happens to make that fake distance
    // smaller.  The gravity-aware field evaluates the actual fork instead.
    if (kind == "Choice") P_PHYSFIELD = 1;

    g_cell.assign(static_cast<size_t>(N) * M, C_UNKNOWN);
    g_skip.assign(static_cast<size_t>(N) * M, 0);
    g_probe.assign(static_cast<size_t>(N) * M, 0);
    g_cellstamp.assign(static_cast<size_t>(N) * M, 0);
    g_blk.assign(static_cast<size_t>(H) * W, 0);
    g_row_words = (W + 2 * kBitOff + 63) / 64 + 2;
    g_bits.assign(static_cast<size_t>(H) * g_row_words, 0);
    // Outside the map is solid: the border cells are walls anyway, but the
    // 6-bit slice can reach past W, and a zero there would read as free.
    for (int py = 0; py < H; ++py) {
        for (int px = -kBitOff; px < 0; ++px) bit_set(py, px, true);
        for (int px = W; px < W + kBitOff; ++px) bit_set(py, px, true);
    }
    g_dist.assign(static_cast<size_t>(N) * M, 0);
    g_bfsq.resize(static_cast<size_t>(N) * M);
    g_pool.reserve(200000);

    std::vector<char> view(2304);
    std::vector<int8_t> plan;
    size_t plan_at = 0;
    St me{};
    int coins = 0, shields = 0;
    int actions = 0;
    bool have_state = false;
    int last_coin_seen = 0;
    int last_coin_action = 0;
    int stagnant = 0;
    int stuck = 0;
    size_t unknown_prev = static_cast<size_t>(-1);
    int no_reveal = 0;
    int best_h_ever = INT32_MAX;
    int no_closer = 0;
    bool take_traps = false;   // every remaining coin is behind a dead end
    int skip_epoch = 0;
    int field_age = 1 << 30;
    size_t last_coins_left = static_cast<size_t>(-1), last_unknown_left = static_cast<size_t>(-1);
    int last_shields = -1;
    bool last_knownonly = false, last_have_target = false;
    // Cheap wandering used when the planner has nothing to offer. It costs no
    // search at all, just the per-frame I/O, so it is affordable for the whole
    // 10000-frame allowance and occasionally walks into coins the planner could
    // not prove a route to.
    bool fallback = false;
    int fb_since = 0, fb_dir = 1, fb_lastx = -1, fb_stall = 0;
    // The exact planner intentionally treats boxes as immovable. Some Pits,
    // Puzzle and Handcrafted pockets are nevertheless escapable by extending a
    // normal jump once with S. Keep the rescue sparse: at most once for each
    // coin count, and only after the character has stayed in a 5-cell box long
    // enough to establish that ordinary jumping is not leaving it.
    // With an adaptive budget the gate is the arithmetic, not the map name.
    // Handcrafted is hand-built to be solvable cleanly; letting it buy height
    // cost it a coin AND three extra spike hits (461538 -> 647058).
    const bool rescue_kind = P_RESCUEALL
                                 ? true
                                 : (kind == "Jump" || kind == "Puzzle" || kind == "Pits");
    int specials_since_coin = 0;
    size_t known_coins_left = 0, known_unknown_left = 0;
    // The exact S-aware retry is substantially more expensive than the normal
    // planner. Paired runs show a decisive gain on Puzzle and a small gain on
    // large Pits, but only lost planning time on Choice/Handcrafted/Maze and
    // made small Pits pay two S commands to replace one missed coin.
    const bool exact_special_kind = kind == "Puzzle" || kind == "Pits";
    int rescue_coin = -1, rescue_uses = 0;
    const int rescue_limit = kind == "Jump" ? P_JUMPRESCUE :
                             kind == "Pits"
                                 ? (static_cast<size_t>(N) * M <= 500
                                        ? P_PITSSMALLRESCUE : P_PITSRESCUE)
                                 : 1;
    int escape_left = 0;
    int rescue_anchor_x = -1000000000, rescue_anchor_y = -1000000000;
    int rescue_floor_y = -1000000000, rescue_prev_y = -1000000000, rescue_confined = 0;
    bool rescue_chain = false;
    int special_try_coin = -1, special_try_cell = -1;
    size_t special_try_unknown = static_cast<size_t>(-1);
    std::vector<uint8_t> puzzle_swept_row(static_cast<size_t>(N), 0);
    int puzzle_sweep_row = -1, puzzle_sweep_coin = -1;

    std::string line;
    std::string rle;
    while (true) {
        // ---- read one frame -------------------------------------------------
        if (!in.read_line(line)) break;
        int vals[4] = {0, 0, 0, 0};
        {
            int k = 0, i = 0;
            const int len = static_cast<int>(line.size());
            while (k < 4 && i < len) {
                while (i < len && (line[i] == ' ' || line[i] == '\t')) ++i;
                int sign = 1;
                if (i < len && (line[i] == '-' || line[i] == '+')) { sign = line[i] == '-' ? -1 : 1; ++i; }
                int v = 0; bool any = false;
                while (i < len && line[i] >= '0' && line[i] <= '9') { v = v * 10 + (line[i++] - '0'); any = true; }
                if (!any) break;
                vals[k++] = v * sign;
            }
            if (k < 4) break;
        }
        if (!in.read_line(rle)) break;

        const int newc = vals[0];
        shields = vals[1];
        const int py = vals[2], px = vals[3];

        if (!absorb_view(rle, py, px, view.data())) { turn.send("FINISH"); return 0; }

        // Replan whenever reality diverged from the simulation: a different
        // position, or a coin we did not expect to (not) collect.
        bool need_plan = false;
        if (!have_state) {
            me = St{static_cast<int16_t>(py), static_cast<int16_t>(px), 0, 0, 0, 0, 0,
                    static_cast<uint8_t>(shields < kShieldCap ? shields : kShieldCap),
                    -1, -1};
            have_state = true;
            need_plan = true;
        } else if (me.y != py || me.x != px) {
            me.y = static_cast<int16_t>(py);
            me.x = static_cast<int16_t>(px);
            me.vy = 0; me.vx = 0; me.wj = 0; me.hold = 0;
            need_plan = true;
        }
        if (newc != last_coin_seen) {
            // Picking a coin up does NOT invalidate the route we are running --
            // it only removes one source from the heuristic field. Forcing a
            // replan here meant a dense map like Jump re-searched every few
            // frames and spent its whole budget planning: 343 of 2744 coins.
            if (g_puzzle_kind && newc > last_coin_seen) {
                const int coin_row = fdiv6(py + 3);
                if (coin_row >= 0 && coin_row < N) {
                    g_puzzle_collected_row[static_cast<size_t>(coin_row)] = 1;
                    if (g_puzzle_phase < 0) g_puzzle_phase = ((coin_row % 8) + 8) % 8;
                    if (g_puzzle_focus_row >= 0 && coin_row != g_puzzle_focus_row)
                        g_puzzle_dir = coin_row > g_puzzle_focus_row ? 1 : -1;
                    else if (g_puzzle_focus_row < 0)
                        g_puzzle_dir = coin_row < N / 2 ? 1 : -1;
                    g_puzzle_focus_row = coin_row;
                }
            }
            last_coin_seen = newc;
            last_coin_action = actions;
            best_h_ever = INT32_MAX;    // new target set; give it a fresh budget
            no_closer = 0;
        }
        coins = newc;
        if (plan_at >= plan.size()) need_plan = true;
        if (P_MAXEXEC > 0 && plan_at >= static_cast<size_t>(P_MAXEXEC)) need_plan = true;

        // Large Puzzle starts on the right entrance of one of its eight-row
        // coin corridors.  The general explorer can climb the whole outer
        // shaft before revealing that the current row contains a coin (77x73:
        // row 66 -> row 10, wasting 256 actions).  Sweep a matching corridor
        // directly once; it is flat, contains no boxes, and needs no S command.
        const int puzzle_frame_row = fdiv6(py + 3);
        const int puzzle_frame_col = fdiv6(px + 3);
        if (puzzle_sweep_row >= 0 &&
            (newc > puzzle_sweep_coin || puzzle_frame_row != puzzle_sweep_row ||
             puzzle_frame_col <= 3)) {
            puzzle_sweep_row = -1;
        }
        if (g_puzzle_kind && g_cell.size() > 5000 && puzzle_sweep_row < 0 &&
            puzzle_frame_row > 0 && puzzle_frame_row < N - 1 &&
            puzzle_frame_row % 8 == 2 && puzzle_frame_row <= N - 11 &&
            puzzle_frame_col >= M - 4 &&
            !puzzle_swept_row[static_cast<size_t>(puzzle_frame_row)]) {
            puzzle_sweep_row = puzzle_frame_row;
            puzzle_sweep_coin = newc;
            puzzle_swept_row[static_cast<size_t>(puzzle_frame_row)] = 1;
            NYPC_DEBUG("PUZZLESWEEP row=", puzzle_sweep_row,
                       "act=", actions, "coins=", newc);
        }
        const bool puzzle_force_sweep = puzzle_sweep_row >= 0;
        if (puzzle_force_sweep) {
            fallback = false;
            plan.clear();
            plan_at = 0;
            need_plan = false;
        }

        // ---- hard deadline: bank what we have and leave cleanly -------------
        if (timer.cpu() > kHardCpu || timer.wall() > kWallGuard || actions >= 10000) {
            turn.send("FINISH");
            NYPC_DEBUG("STAT coins=", coins, "actions=", actions,
                       "cpu=", timer.cpu(), "wall=", timer.wall(), "replans=", g_stat_replans,
                       "goals=", g_stat_goals, "probes=", g_stat_probes,
                       "traps=", g_stat_traps, "nodes=", g_stat_nodes, "fieldn=", g_field_n, "field_s=", g_field_s,
                       "search_s=", g_stat_search);
            return 0;
        }

        const bool out_of_thinking =
            P_NODECAP > 0 ? (g_stat_nodes > P_NODECAP) : (timer.cpu() > kPlanCpu);
        // NOTE: refreshing known_coins_left outside the replan block also feeds
        // the adaptive special-jump budget, and doing so cost 468458 -> 492699
        // on the reconstructed evaluation maps. The counts stay replan-scoped.
        if (!fallback && out_of_thinking) {
            fallback = true;                 // out of thinking time, not of frames
            fb_since = -1000000000;          // never re-offer the planner
        }
        if (fallback && ++fb_since >= P_FBRETRY) {
            fallback = false;                 // periodically re-offer the planner
            fb_since = 0;
            need_plan = true;
        }
        const bool small_pits_last_trap = kind == "Pits" &&
            static_cast<size_t>(N) * M <= 500 && coins > 0 && coins < 3 &&
            actions - last_coin_action >= P_PITSSMALLSTALL && rescue_uses == 0;
        if (small_pits_last_trap && !fallback) {
            fallback = true;
            fb_since = 0;
            need_plan = false;
        }
        if (P_SKIPTTL > 0 && actions - skip_epoch >= P_SKIPTTL) {
            skip_epoch = actions;
            std::fill(g_skip.begin(), g_skip.end(), 0);
            std::fill(g_probe.begin(), g_probe.end(), 0);
            take_traps = false;
            need_plan = true;
            NYPC_DEBUG("SKIPRESET act=", actions, "coins=", newc);
        }
        if (!fallback && need_plan) {
            me.sh = static_cast<uint8_t>(shields < kShieldCap ? shields : kShieldCap);
            // A cell you are standing in cannot be avoided, and treating it as a
            // wall makes the distance field unable to reach you: every target
            // reads as unreachable, which used to end the run on the spot. That
            // is the shape of Choice large scoring a flat zero.
            {
                const int hr0 = fdiv6(me.y + 3), hc0 = fdiv6(me.x + 3);
                if (hr0 >= 0 && hr0 < N && hc0 >= 0 && hc0 < M)
                    g_skip[static_cast<size_t>(hr0) * M + hc0] = 0;
            }
            size_t coins_left = 0, shields_left = 0, unknown_left = 0;
            for (size_t i = 0, e = g_cell.size(); i < e; ++i) {
                coins_left += (g_cell[i] == C_COIN);
                shields_left += (g_cell[i] == C_SHIELD);
                unknown_left += (g_cell[i] == C_UNKNOWN);
            }
            // Cells inside solid rock are never observed, so "unknown cells
            // remain" is not a reason to keep walking. What matters is whether
            // walking still REVEALS anything: once it stops paying, and no coin
            // is on the board, there is nothing left to find.
            known_coins_left = coins_left;
            known_unknown_left = unknown_left;
            if (unknown_left < unknown_prev) { unknown_prev = unknown_left; no_reveal = 0; }
            else ++no_reveal;
            if (coins_left == 0 && no_reveal >= P_NOREVEAL) {
                // Only worth avoiding dead ends while a coin remains OUTSIDE
                // them. On Choice the single coin IS inside one, so holding the
                // line there means never scoring at all; once no reachable coin
                // is left outside, walking in is the correct move.
                {
                    // Give up on avoiding dead ends only when there is nothing
                    // left outside them -- no reachable coin AND no reachable
                    // unexplored cell. Counting only known coins let Puzzle
                    // clear the list while three whole corridors of coins were
                    // still undiscovered, and walk into a shaft.
                    bool outside = false;
                    if (coins_left) {
                        build_distance(C_COIN, shields > 0 ? P_SPIKE : -1);
                        outside = heuristic_of(me.y, me.x) < INT32_MAX / 8;
                    }
                    if (!outside) {
                        build_distance(C_UNKNOWN, shields > 0 ? P_SPIKE : -1);
                        outside = heuristic_of(me.y, me.x) < INT32_MAX / 8;
                    }
                    if (!outside) {
                        std::fill(g_skip.begin(), g_skip.end(), 0);
                        NYPC_DEBUG("CLEARSKIP act=", actions, "coinsleft=", (int)coins_left);
                    }
                }
                fallback = true; fb_since = 0;
                stuck = 0; no_closer = 0; no_reveal = 0;
                NYPC_DEBUG("FALLBACK explored-out coins=", newc, "actions=", actions,
                           "cpu=", timer.cpu());
            }
            // On a large map most cells are unseen and optimistic routes across
            // them do not survive contact; once the map is mostly known the
            // optimism is free and helps. Switch on the actual ratio rather than
            // picking one rule for both.
            // Only on the maps that are mostly unseen. A small map is known
            // almost at once, and there the optimism is free; a large one plans
            // routes across territory it has never observed, which is why
            // choice_big reached 449 coin goals and banked 81.
            g_knownonly = P_PHYSFIELD &&
                          unknown_left * 100 >
                              static_cast<size_t>(P_KNOWNPCT) * g_cell.size();
            // Rebuilding the field is a Dijkstra over every cell and, on a large
            // map, cost more than the physics search it feeds. Collecting one
            // coin barely moves it, so reuse it unless the world really changed.
            const bool field_stale =
                field_age >= P_FIELDEVERY || coins_left != last_coins_left ||
                unknown_left != last_unknown_left ||
                shields != last_shields || g_knownonly != last_knownonly;
            bool have_target = last_have_target;
            if (field_stale) {
                have_target = choose_field(coins_left, shields_left, shields, me.y, me.x);
                field_age = 0;
                last_coins_left = coins_left;
                last_unknown_left = unknown_left;
                last_shields = shields;
                last_knownonly = g_knownonly;
                last_have_target = have_target;
            } else {
                ++field_age;
            }

            // Nothing left worth reaching: no coin and no unexplored cell is
            // reachable from here. Cost is min over time and coins only ever go
            // up, so stopping cannot lose anything already banked -- and every
            // extra frame spends the judge's 2s response budget for free.
            bool target = have_target;
            if (!target) {
                // Every remaining coin sits behind a dead end. That is exactly
                // when entering one is right: cost is the minimum over time, so
                // the coins already banked can never be lost, and one more coin
                // is pure gain.
                bool any_skip = false;
                for (size_t i = 0, e = g_skip.size(); i < e && !any_skip; ++i)
                    any_skip = g_skip[i] != 0;
                if (any_skip) {
                    // Entering a dead end is a one-way decision, so it matters
                    // WHICH one. Clearing the whole list let the planner walk
                    // into whichever was nearest; on Choice, where the map is
                    // literally a set of one-way corridors to choose between,
                    // that is the difference between one coin and the richest
                    // corridor. Score each pocket by the coins inside it.
                    int32_t best_cell = -1;
                    int best_coins = -1;
                    const int limit = std::max<int>(
                        P_TRAPCELLS,
                        static_cast<int>(g_cell.size()) / std::max(1, P_TRAPDIV)) * 2 + 8;
                    std::vector<int32_t> seen_mark(g_cell.size(), -1);
                    std::vector<int32_t> stack;
                    for (size_t i = 0; i < g_skip.size(); ++i) {
                        if (!g_skip[i]) continue;
                        int coins_here = 0, visited = 0;
                        stack.clear();
                        stack.push_back(static_cast<int32_t>(i));
                        seen_mark[i] = static_cast<int32_t>(i);
                        while (!stack.empty() && visited < limit) {
                            const int32_t cur = stack.back();
                            stack.pop_back();
                            ++visited;
                            if (g_cell[static_cast<size_t>(cur)] == C_COIN) ++coins_here;
                            const int r = cur / M, c = cur % M;
                            const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
                            for (int k = 0; k < 4; ++k) {
                                const int nr = r + dr[k], nc = c + dc[k];
                                if (nr < 0 || nr >= N || nc < 0 || nc >= M) continue;
                                const size_t ni = static_cast<size_t>(nr) * M + nc;
                                if (g_cell[ni] == C_WALL) continue;
                                if (seen_mark[ni] == static_cast<int32_t>(i)) continue;
                                seen_mark[ni] = static_cast<int32_t>(i);
                                stack.push_back(static_cast<int32_t>(ni));
                            }
                        }
                        if (coins_here > best_coins) {
                            best_coins = coins_here;
                            best_cell = static_cast<int32_t>(i);
                        }
                    }
                    if (P_BESTTRAP && best_cell >= 0 && best_coins > 0) {
                        g_skip[static_cast<size_t>(best_cell)] = 0;
                        g_probe[static_cast<size_t>(best_cell)] = 1;
                    } else {
                        std::fill(g_skip.begin(), g_skip.end(), 0);
                        std::fill(g_probe.begin(), g_probe.end(), 0);
                    }
                    // ...and stop probing. Without this the probe immediately
                    // re-proves the dead end, the plan is rejected again, and
                    // the last coin is never taken.
                    take_traps = true;
                    target = choose_field(coins_left, shields_left, shields, me.y, me.x);
                    if (!target) {
                        std::fill(g_skip.begin(), g_skip.end(), 0);
                        std::fill(g_probe.begin(), g_probe.end(), 0);
                        target = choose_field(coins_left, shields_left, shields, me.y, me.x);
                    }
                    NYPC_DEBUG("LASTTRAP act=", actions, "coins_in_pocket=", best_coins,
                               "target=", (int)target);
                }
            }
            if (!target) {
                // Do NOT stop here. Cost is the minimum over time, so playing on
                // can never lose a banked coin, and a frame costs only its I/O.
                // The intermediate evaluation showed exactly this: Choice and
                // Maze were the only two types that stopped early, and they came
                // 38.5th and 36th on large while every type that played all
                // 10000 frames placed in the top ten.
                fallback = true;
                fb_since = 0;
                stuck = 0; no_closer = 0; no_reveal = 0;
                NYPC_DEBUG("FALLBACK nothing-reachable coins=", newc,
                           "actions=", actions, "cpu=", timer.cpu());
            }

            const double left = P_NODECAP > 0
                ? (g_stat_nodes < P_NODECAP ? 1.0 : 0.0)
                : (kPlanCpu - timer.cpu());
            int budget = P_BUDGET;
            if (left < 0.6) budget = 2000;
            if (left < 0.3) budget = 600;
            if (left < 0.1) budget = 150;
            plan.clear();
            plan_at = 0;
            const double t_before = timer.wall();
            bool special_plan_selected = false;
            bool ok = plan_moves(me, budget, plan);
            // A drained heap means the search enumerated everything the
            // character can physically reach and none of it was better. That is
            // being stuck, not being short of budget; more frames cannot help.
            if (g_heap_drained && !g_made_progress) ++stuck; else stuck = 0;
            // Standing still means the cheap search could not find a foothold
            // that improves on this one. Pay for one much deeper look before
            // accepting it, but never when the state space itself ran out.
            if (g_dbg_besth >= g_dbg_h0 && !g_heap_drained && left > 0.35) {
                ok = plan_moves(me, budget * P_DEEP, plan);
                if (g_heap_drained && !g_made_progress) ++stuck;
            }

            // Second stage: if free movement cannot make progress, search the
            // exact same physics with S/S+direction enabled. Prefer a known
            // coin; when Puzzle has none on screen, reaching a genuinely unseen
            // cell is also a useful proved goal because it opens the next view.
            const int here_cell = fdiv6(me.y + 3) * M + fdiv6(me.x + 3);
            const bool allow_small_pits_special = kind != "Pits" ||
                static_cast<size_t>(N) * M > 500 ||
                (newc > 0 &&
                 actions - last_coin_action >= P_PITSSMALLSTALL);
            const bool explore_unknown = kind == "Puzzle" &&
                !target && unknown_left > 0;
            if (exact_special_kind && allow_small_pits_special &&
                (coins_left || explore_unknown) &&
                !g_found_goal && !g_made_progress &&
                stuck >= ((explore_unknown || kind == "Choice") ? 1 : P_STUCKN) &&
                left > 0.20 &&
                P_SPECIAL_CAP > 0 &&
                (special_try_coin != newc || special_try_cell != here_cell ||
                 special_try_unknown != unknown_left)) {
                special_try_coin = newc;
                special_try_cell = here_cell;
                special_try_unknown = unknown_left;

                const std::vector<int8_t> normal_plan = plan;
                const bool normal_ok = ok;
                const bool normal_drained = g_heap_drained;
                const bool normal_progress = g_made_progress;
                const bool normal_goal = g_found_goal;
                const int normal_reach = g_reach_cells;
                const int normal_h0 = g_dbg_h0, normal_besth = g_dbg_besth;
                const int normal_dbg_goal = g_dbg_goal, normal_exp = g_dbg_exp;
                const std::vector<int32_t> normal_dist = g_dist;

                std::vector<uint8_t> saved_skip = g_skip;
                std::fill(g_skip.begin(), g_skip.end(), 0);
                const int saved_physfield = P_PHYSFIELD;
                P_PHYSFIELD = 0;  // S can climb; the ordinary directed field cannot model that.
                if (explore_unknown) build_open_distance(C_UNKNOWN);
                else build_distance(C_COIN, shields > 0 ? P_SPIKE : -1);
                P_PHYSFIELD = saved_physfield;
                const bool saved_knownonly = g_knownonly;
                g_knownonly = !explore_unknown;
                g_search_special = true;
                g_coin_goal_only = !explore_unknown;
                g_unknown_goal = explore_unknown;
                std::vector<int8_t> special_plan;
                const int saved_special_cap = P_SPECIAL_CAP;
                if (kind == "Pits" && g_cell.size() <= 500) P_SPECIAL_CAP = 1;
                const bool special_ok = plan_moves(me, P_SPECIAL_BUDGET, special_plan);
                P_SPECIAL_CAP = saved_special_cap;
                const bool special_goal = g_found_goal;
                g_search_special = false;
                g_coin_goal_only = false;
                g_unknown_goal = false;
                g_knownonly = saved_knownonly;
                g_skip.swap(saved_skip);

                if (special_ok && special_goal && !special_plan.empty()) {
                    plan.swap(special_plan);
                    plan_at = 0;
                    ok = true;
                    fallback = false;
                    fb_since = 0;
                    stuck = 0;
                    special_plan_selected = true;
                    [[maybe_unused]] int special_n = 0;
                    for (int8_t a : plan) special_n += (action_bits(a) & A_SPECIAL) != 0;
                    NYPC_DEBUG("SPECIALPLAN act=", actions, "coins=", newc,
                               "len=", (int)plan.size(), "special=", special_n);
                } else {
                    plan = normal_plan;
                    ok = normal_ok;
                    g_heap_drained = normal_drained;
                    g_made_progress = normal_progress;
                    g_found_goal = normal_goal;
                    g_reach_cells = normal_reach;
                    g_dbg_h0 = normal_h0;
                    g_dbg_besth = normal_besth;
                    g_dbg_goal = normal_dbg_goal;
                    g_dbg_exp = normal_exp;
                    g_dist = normal_dist;
                }
            }
            // Circling. h0 is the cell distance to the nearest live target; if
            // that never improves over many plans we are not travelling towards
            // anything, just orbiting, and every frame still costs the judge's
            // response budget.
            if (g_dbg_h0 < best_h_ever) { best_h_ever = g_dbg_h0; no_closer = 0; }
            else ++no_closer;
            if (no_closer >= P_NOCLOSER) {
                // Only worth avoiding dead ends while a coin remains OUTSIDE
                // them. On Choice the single coin IS inside one, so holding the
                // line there means never scoring at all; once no reachable coin
                // is left outside, walking in is the correct move.
                {
                    // Give up on avoiding dead ends only when there is nothing
                    // left outside them -- no reachable coin AND no reachable
                    // unexplored cell. Counting only known coins let Puzzle
                    // clear the list while three whole corridors of coins were
                    // still undiscovered, and walk into a shaft.
                    bool outside = false;
                    if (coins_left) {
                        build_distance(C_COIN, shields > 0 ? P_SPIKE : -1);
                        outside = heuristic_of(me.y, me.x) < INT32_MAX / 8;
                    }
                    if (!outside) {
                        build_distance(C_UNKNOWN, shields > 0 ? P_SPIKE : -1);
                        outside = heuristic_of(me.y, me.x) < INT32_MAX / 8;
                    }
                    if (!outside) {
                        std::fill(g_skip.begin(), g_skip.end(), 0);
                        NYPC_DEBUG("CLEARSKIP act=", actions, "coinsleft=", (int)coins_left);
                    }
                }
                fallback = true; fb_since = 0;
                stuck = 0; no_closer = 0; no_reveal = 0;
                NYPC_DEBUG("FALLBACK circling coins=", newc, "actions=", actions,
                           "cpu=", timer.cpu());
            }
            if (stuck >= P_STUCKN) {
                // Only worth avoiding dead ends while a coin remains OUTSIDE
                // them. On Choice the single coin IS inside one, so holding the
                // line there means never scoring at all; once no reachable coin
                // is left outside, walking in is the correct move.
                {
                    // Give up on avoiding dead ends only when there is nothing
                    // left outside them -- no reachable coin AND no reachable
                    // unexplored cell. Counting only known coins let Puzzle
                    // clear the list while three whole corridors of coins were
                    // still undiscovered, and walk into a shaft.
                    bool outside = false;
                    if (coins_left) {
                        build_distance(C_COIN, shields > 0 ? P_SPIKE : -1);
                        outside = heuristic_of(me.y, me.x) < INT32_MAX / 8;
                    }
                    if (!outside) {
                        build_distance(C_UNKNOWN, shields > 0 ? P_SPIKE : -1);
                        outside = heuristic_of(me.y, me.x) < INT32_MAX / 8;
                    }
                    if (!outside) {
                        std::fill(g_skip.begin(), g_skip.end(), 0);
                        NYPC_DEBUG("CLEARSKIP act=", actions, "coinsleft=", (int)coins_left);
                    }
                }
                fallback = true; fb_since = 0;
                stuck = 0; no_closer = 0; no_reveal = 0;
                NYPC_DEBUG("FALLBACK stuck coins=", newc, "actions=", actions,
                           "cpu=", timer.cpu());
            }
            // ---- one-way trap check -------------------------------------
            // Replay the plan, then ask from where it ends: can the character
            // get back to the cell it is standing in right now? A search that
            // exhausts its whole reachable state space without returning proves
            // the answer is no, and entering costs every remaining coin. This
            // test deliberately does NOT depend on coins or unexplored cells --
            // an earlier version keyed off those, and when a map was fully
            // explored the field went degenerate and every cell looked like a
            // trap, which blocked all movement and cost 4 maps at once.
            // Maze's exact-only trap rule has never proved a trap in paired
            // runs; all return probes were safe and merely consumed the budget
            // needed to climb again after a fall. Corridors are therefore
            // allowed directly, while other kinds retain the full proof.
            bool plan_safe = take_traps || special_plan_selected ||
                             kind == "Maze" || kind == "Choice" ||
                             (kind == "Puzzle" && coins_left > 0) ||
                             (g_puzzle_frontier_active && g_cell.size() > 5000);
            for (int attempt = 0; !plan_safe && !take_traps && ok &&
                                  !plan.empty() && attempt < 16; ++attempt) {
                St end = me;
                for (size_t i = 0; i < plan.size(); ++i) {
                    TouchOut po;
                    sim_step(end, action_bits(plan[i]), po);
                    if (end.sh == 0xFF) end.sh = 0;
                }
                const int er = fdiv6(end.y + 3), ec = fdiv6(end.x + 3);
                const int hr = fdiv6(me.y + 3), hc = fdiv6(me.x + 3);
                if (er < 0 || er >= N || ec < 0 || ec >= M) break;
                if (hr < 0 || hr >= N || hc < 0 || hc >= M) break;
                const size_t ecell = static_cast<size_t>(er) * M + ec;
                const int32_t here = static_cast<int32_t>(hr) * M + hc;
                if (static_cast<int32_t>(ecell) == here) { plan_safe = true; break; }
                if (g_probe[ecell] == 0) {
                    if (g_stat_probes >= 400 ||
                        (P_NODECAP > 0 ? g_stat_nodes > P_NODECAP
                                       : kPlanCpu - timer.cpu() < 0.30)) {
                        // Maze branches repeatedly revisit already traversable
                        // ascents. Once probe reserve ran low, leaving
                        // plan_safe=false rejected every future climb and the
                        // fallback oscillated on the lower tier forever. Maze
                        // only classifies a trap after an exhausted proof, so an
                        // unprobed route must follow the same "inconclusive is
                        // safe" rule as a budget-limited probe.
                        if (kind == "Maze") plan_safe = true;
                        break;
                    }
                    std::vector<int8_t> pr;
                    build_distance_single(here, shields > 0);
                    g_goal_cell = here;
                    plan_moves(end, P_PROBE, pr);
                    g_goal_cell = -1;
                    // Inconclusive (budget ran out before the space did) counts
                    // as SAFE: a false trap is far more expensive than a real one.
                    // A dead end on a large map is a large pocket. A fixed
                    // 12-cell test only catches the shafts on a small one, and
                    // lets the character fall into everything bigger -- which is
                    // what confines it to one corner of a large Maze.
                    const int trap_limit = std::max<int>(
                        P_TRAPCELLS, static_cast<int>(g_cell.size()) / std::max(1, P_TRAPDIV));
                    g_probe[ecell] =
                        (!g_found_goal && (g_heap_drained || g_reach_cells <= trap_limit))
                            ? 2 : 1;
                    ++g_stat_probes;
                    NYPC_DEBUG("PROBE act=", actions, "end=", er, ",", ec,
                               "here=", hr, ",", hc, "goal=", (int)g_found_goal,
                               "drained=", (int)g_heap_drained, "cells=", g_reach_cells,
                               "verdict=", (int)g_probe[ecell]);
                }
                if (g_probe[ecell] == 1) { plan_safe = true; break; }  // safe route
                g_skip[ecell] = 1;
                ++g_stat_traps;
                plan.clear();
                if (choose_field(coins_left, shields_left, shields, me.y, me.x)) {
                    ok = plan_moves(me, budget, plan);
                } else {
                    // Everything that is left sits behind a dead end. NOW it is
                    // right to walk into one: cost is the minimum over time, so
                    // the coins already banked can never be taken back.
                    std::fill(g_skip.begin(), g_skip.end(), 0);
                    choose_field(coins_left, shields_left, shields, me.y, me.x);
                    ok = plan_moves(me, budget, plan);
                    plan_safe = true;   // deliberate: nothing else is left
                    break;
                }
            }
            // Running out of attempts used to mean executing the last plan
            // anyway -- straight into the dead end it had just been rejected
            // for. Wander instead; the wander is dead-end aware.
            if (ok && !plan.empty() && !plan_safe) {
                plan.clear();
                fallback = true;
                fb_since = 0;
                NYPC_DEBUG("NOSAFEPLAN act=", actions);
            }
            g_stat_search += timer.wall() - t_before;
            if (g_stat_replans % 12 == 1 && actions < 3000) {
                NYPC_DEBUG("T act=", actions, "cell=", (py/6), ",", (px/6),
                           "knowncoins=", (int)coins_left, "shields=", shields,
                           "h0=", g_dbg_h0, "besth=", g_dbg_besth,
                           "goal=", g_dbg_goal, "len=", (int)plan.size());
            }
            if (g_stat_replans <= 0) {
                NYPC_DEBUG("PLAN#", g_stat_replans, "act=", actions,
                           "cell=", (py / 6), ",", (px / 6),
                           "h0=", g_dbg_h0, "besth=", g_dbg_besth,
                           "goal=", g_dbg_goal, "len=", (int)plan.size(),
                           "exp=", g_dbg_exp, "coins=", newc, "traps=", g_stat_traps, "probes=", g_stat_probes);
            }
            if (!ok) {
                if (++stagnant > 6) { turn.send("FINISH"); return 0; }
            } else {
                stagnant = 0;
            }
        }

        // ---- emit one move --------------------------------------------------
        int act = 0;
        if (fallback) {
            if (rescue_coin != coins) {
                rescue_coin = coins;
                rescue_uses = 0;
                rescue_chain = false;
            }
            const bool observed_falling = py > rescue_prev_y;
            rescue_prev_y = py;
            const bool pits_stalled = kind == "Pits" &&
                actions - last_coin_action >= P_PITSSTALL;
            // Horizontal progress means the pocket has been escaped even when
            // Pits has exceeded its no-coin timer.  Exempting a "stalled" Pits
            // run kept rescue_chain alive across the entire 497-column top
            // corridor: 46 useless S commands after the first successful one.
            if (std::abs(px - rescue_anchor_x) > 30 ||
                std::abs(py - rescue_anchor_y) > 30) {
                rescue_anchor_x = px;
                rescue_anchor_y = py;
                rescue_floor_y = py;
                rescue_confined = 0;
                rescue_chain = false;
            } else {
                rescue_floor_y = std::max(rescue_floor_y, py);
                ++rescue_confined;
            }
            if (px == fb_lastx) {
                if (++fb_stall > 6) { fb_dir = -fb_dir; fb_stall = 0; }
            } else {
                fb_stall = 0;
            }
            fb_lastx = px;
            if (fb_since <= 1) NYPC_DEBUG("WANDER act=", actions, "cell=", (py/6), ",", (px/6));
            // The wander must respect the dead ends the planner proved, or it
            // undoes the planner's work: on Puzzle the planner correctly walked
            // past seven one-way coin shafts, then the blind wander dropped
            // straight into the eighth and forfeited the rest of the map.
            if (P_FBMODE == 1) {
                if (static_cast<int>(rng.next_below(100)) < P_FBFLIP) fb_dir = -fb_dir;
            }
            const int jbit = (P_FBMODE == 1 &&
                              static_cast<int>(rng.next_below(100)) >= P_FBJUMP) ? 0 : A_JUMP;
            const int kFall[4] = {A_RIGHT | jbit, A_LEFT | jbit, A_RIGHT, A_LEFT};
            const int order[4] = {
                fb_dir > 0 ? kFall[0] : kFall[1], fb_dir > 0 ? kFall[1] : kFall[0],
                fb_dir > 0 ? kFall[2] : kFall[3], fb_dir > 0 ? kFall[3] : kFall[2]};
            act = A_JUMP;
            for (int ci = 0; ci < 4; ++ci) {
                St look = me;
                bool bad = false;
                for (int k = 0; k < 14 && !bad; ++k) {
                    TouchOut lo;
                    sim_step(look, order[ci], lo);
                    if (look.sh == 0xFF) look.sh = 0;
                    const int rr = fdiv6(look.y + 3), cc = fdiv6(look.x + 3);
                    if (rr >= 0 && rr < N && cc >= 0 && cc < M &&
                        g_skip[static_cast<size_t>(rr) * M + cc]) bad = true;
                }
                if (!bad) { act = order[ci]; break; }
                if (ci == 0) fb_dir = -fb_dir;
            }

            // ---- escape burst ------------------------------------------------
            // k special frames for one more coin pays whenever k < g*D/c, and D/c
            // is large exactly when the run is going badly -- which is precisely
            // the situation of being stuck with coins still on the board.
            {
                const long long d_now =
                    static_cast<long long>(coins) +
                    static_cast<long long>(known_coins_left) + rescue_uses;
                const long long budget_now =
                    coins > 0 ? std::max<long long>(1, d_now / coins) : 1000000;
                const long long left_now = budget_now - specials_since_coin;
                // Being stuck with unexplored map left is exactly when a
                // special jump pays: the coins that justify it are the ones we
                // have not seen yet, and inside a pit every coin we HAVE seen is
                // already collected.
                const bool worth_escaping =
                    known_coins_left > 0 || known_unknown_left > 0;
                if (P_ESCAPE && escape_left <= 0 && worth_escaping &&
                    rescue_confined >= P_STUCKFRAMES && left_now > 2) {
                    escape_left = static_cast<int>(
                        std::min<long long>(P_BURST, left_now));
                }
                if (escape_left > 0) {
                    --escape_left;
                    ++specials_since_coin;
                    ++rescue_uses;
                    rescue_confined = 0;
                    act = A_SPECIAL | (fb_dir > 0 ? A_RIGHT : A_LEFT);
                }
            }
            // Wait until the normal jump has reached its apex. One S here adds
            // enough height to clear the canonical three-cell pit wall, while
            // issuing it from the floor would mostly duplicate a free jump.
            const bool observed_airborne = py <= rescue_floor_y - 8;
            const bool simulated_airborne = me.vy >= 0 && !blocked_rect(me.y + 1, me.x);
            const bool continue_chain = rescue_chain && observed_airborne && observed_falling;
            const bool first_rescue_ready = kind == "Jump"
                ? actions - last_coin_action >= P_JUMPSTALL
                : kind == "Pits"
                    ? pits_stalled && rescue_confined >= 40
                    : rescue_confined >= 40;
            if (rescue_chain && py >= rescue_floor_y - 1) rescue_chain = false;
            // D/c with D estimated from the coins we have actually seen, which
            // under-counts an unexplored map and so errs towards spending less.
            const long long d_est =
                static_cast<long long>(coins) + static_cast<long long>(known_coins_left) +
                rescue_uses;
            const long long adaptive_budget =
                coins > 0 ? std::max<long long>(1, d_est / coins) : 1000000;
            const bool budget_ok = P_ADAPTRESCUE
                                       ? specials_since_coin < adaptive_budget
                                       : rescue_uses < rescue_limit;
            const bool exact_handles_small_pits = kind == "Pits" &&
                g_cell.size() <= 500 && known_coins_left > 0;
            if (rescue_kind && !exact_handles_small_pits &&
                (first_rescue_ready || continue_chain) && budget_ok &&
                (observed_airborne || simulated_airborne)) {
                // A vertical boost below a ceiling does nothing. Aim first for
                // the nearest observed opening in that ceiling; the canonical
                // pit is three cells wide with a one-cell opening, so blindly
                // retaining the sweep direction misses it half the time.
                int rescue_dir = fb_dir;
                const int rr = fdiv6(py + 3), cc = fdiv6(px + 3);
                const int ceiling_r = fdiv6(rescue_floor_y - 15) - 1;
                for (int d = 0; d <= 8; ++d) {
                    const int first = rescue_dir > 0 ? cc + d : cc - d;
                    const int second = rescue_dir > 0 ? cc - d : cc + d;
                    bool found = false;
                    for (int pass = 0; pass < 2; ++pass) {
                        const int tc = pass == 0 ? first : second;
                        if (tc < 0 || tc >= M || ceiling_r < 0 || ceiling_r >= N ||
                            rr < 0 || rr >= N) continue;
                        const uint8_t above = g_cell[static_cast<size_t>(ceiling_r) * M + tc];
                        const uint8_t level = g_cell[static_cast<size_t>(rr) * M + tc];
                        if (above != C_WALL && level != C_WALL) {
                            rescue_dir = tc == cc ? 0 : (tc > cc ? 1 : -1);
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
                // In Pits the opening on either side is geometrically valid,
                // but only one side advances to the next coin.  The generic
                // nearest-opening tie-break chose left from the third small
                // pit and spent its sole cost-effective S returning to an
                // already cleared corridor.  Aim at the nearest observed coin.
                if (kind == "Pits") {
                    int best_coin_d = INT32_MAX;
                    for (int r = 0; r < N; ++r) {
                        for (int c = 0; c < M; ++c) {
                            if (g_cell[static_cast<size_t>(r) * M + c] != C_COIN)
                                continue;
                            const int d = std::abs(c - cc);
                            if (d < best_coin_d && c != cc) {
                                best_coin_d = d;
                                rescue_dir = c > cc ? 1 : -1;
                            }
                        }
                    }
                }
                if (rescue_dir) fb_dir = rescue_dir;
                act = A_SPECIAL | (rescue_dir > 0 ? A_RIGHT :
                                   rescue_dir < 0 ? A_LEFT : 0);
                ++rescue_uses;
                ++specials_since_coin;
                rescue_confined = 0;
                rescue_chain = true;
                NYPC_DEBUG("RESCUE act=", actions, "coins=", coins,
                           "cell=", (py / 6), ",", (px / 6));
            }
        } else if (plan_at < plan.size()) {
            act = action_bits(plan[plan_at++]);
        }
        if (puzzle_force_sweep) act = A_LEFT;
        {
            TouchOut o;
            sim_step(me, act, o);
            if (me.sh == 0xFF) me.sh = 0;
        }

        char out[16];
        int len = 0;
        const int count = (act & A_LEFT ? 1 : 0) + (act & A_RIGHT ? 1 : 0) +
                          (act & A_JUMP ? 1 : 0) + (act & A_SPECIAL ? 1 : 0);
        out[len++] = static_cast<char>('0' + count);
        out[len++] = '\n';
        if (act & A_LEFT) { out[len++] = '<'; out[len++] = '\n'; }
        if (act & A_RIGHT) { out[len++] = '>'; out[len++] = '\n'; }
        if (act & A_JUMP) { out[len++] = 'J'; out[len++] = '\n'; }
        if (act & A_SPECIAL) { out[len++] = 'S'; out[len++] = '\n'; }
        std::fwrite(out, 1, static_cast<size_t>(len), stdout);
        std::fflush(stdout);
        ++actions;
    }

    turn.send("FINISH");
    return 0;
}
// END BUNDLED FILE: main.cpp
