#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace fs = std::filesystem;

using Clock = std::chrono::steady_clock;

struct Config {
    std::string root;
    std::string op = "lookup_txn";
    int threads = 4;
    int warmup_sec = 2;
    int duration_sec = 10;
    int files = 10000;
    int fanout = 256;
    int payload_bytes = 4096;
    uint64_t seed = 12345;
    bool sync_file = false;
    bool sync_dir = false;
    bool reset = false;
    bool cleanup = false;
    std::string json_out;
};

struct ThreadResult {
    uint64_t operations = 0;
    uint64_t errors = 0;
    uint64_t bytes_written = 0;
    uint64_t bytes_read = 0;
    std::vector<uint32_t> latency_us;
    std::vector<std::string> error_samples;
};

struct AggregateResult {
    uint64_t operations = 0;
    uint64_t errors = 0;
    uint64_t bytes_written = 0;
    uint64_t bytes_read = 0;
    double elapsed_sec = 0.0;
    double setup_sec = 0.0;
    std::vector<uint32_t> latency_us;
    std::vector<std::string> error_samples;
};

static std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::setfill(' ');
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

static bool parse_bool_flag(const std::string& arg, const std::string& name) {
    return arg == name;
}

static void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --root PATH [options]\n"
        << "\n"
        << "Options:\n"
        << "  --op NAME             stat | access | open_close | create_unlink | rename\n"
        << "                        | readdir | lookup_txn | publish_txn | evict_txn | mixed\n"
        << "  --threads N           Worker threads (default: 4)\n"
        << "  --warmup-sec N        Warmup seconds before timing (default: 2)\n"
        << "  --duration-sec N      Timed benchmark seconds (default: 10)\n"
        << "  --files N             Number of tracked files / working-set entries (default: 10000)\n"
        << "  --fanout N            Number of data directories (default: 256)\n"
        << "  --payload-bytes N     Tiny payload size for create/publish/evict (default: 4096)\n"
        << "  --seed N              RNG seed (default: 12345)\n"
        << "  --sync-file           fsync files after writes\n"
        << "  --sync-dir            fsync parent directory after rename/unlink/create\n"
        << "  --reset              Remove existing benchmark contents under --root before setup\n"
        << "  --cleanup            Delete benchmark contents under --root after the run\n"
        << "  --json-out PATH      Also write JSON summary to a file\n"
        << "  --help               Show this help\n";
}

static Config parse_args(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--root") {
            config.root = next_value("--root");
        } else if (arg == "--op") {
            config.op = next_value("--op");
        } else if (arg == "--threads") {
            config.threads = std::stoi(next_value("--threads"));
        } else if (arg == "--warmup-sec") {
            config.warmup_sec = std::stoi(next_value("--warmup-sec"));
        } else if (arg == "--duration-sec") {
            config.duration_sec = std::stoi(next_value("--duration-sec"));
        } else if (arg == "--files") {
            config.files = std::stoi(next_value("--files"));
        } else if (arg == "--fanout") {
            config.fanout = std::stoi(next_value("--fanout"));
        } else if (arg == "--payload-bytes") {
            config.payload_bytes = std::stoi(next_value("--payload-bytes"));
        } else if (arg == "--seed") {
            config.seed = std::stoull(next_value("--seed"));
        } else if (arg == "--json-out") {
            config.json_out = next_value("--json-out");
        } else if (parse_bool_flag(arg, "--sync-file")) {
            config.sync_file = true;
        } else if (parse_bool_flag(arg, "--sync-dir")) {
            config.sync_dir = true;
        } else if (parse_bool_flag(arg, "--reset")) {
            config.reset = true;
        } else if (parse_bool_flag(arg, "--cleanup")) {
            config.cleanup = true;
        } else if (parse_bool_flag(arg, "--help")) {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (config.root.empty()) {
        throw std::runtime_error("--root is required");
    }
    if (config.threads <= 0 || config.duration_sec <= 0 || config.files <= 0 || config.fanout <= 0) {
        throw std::runtime_error("threads, duration-sec, files, and fanout must be positive");
    }
    if (config.payload_bytes < 0 || config.warmup_sec < 0) {
        throw std::runtime_error("payload-bytes and warmup-sec must be non-negative");
    }
    return config;
}

static std::string zero_pad(int value, int width) {
    std::ostringstream out;
    out << std::setw(width) << std::setfill('0') << value;
    return out.str();
}

static fs::path data_root(const Config& config) {
    return fs::path(config.root) / "data";
}

static fs::path rename_root(const Config& config) {
    return fs::path(config.root) / "rename";
}

static fs::path data_dir_for_index(const Config& config, int index) {
    const int bucket = index % config.fanout;
    return data_root(config) / ("dir_" + zero_pad(bucket, 4));
}

static fs::path data_file_for_index(const Config& config, int index) {
    return data_dir_for_index(config, index) / ("entry_" + zero_pad(index, 8) + ".bin");
}

static fs::path publish_tmp_for_index(const Config& config, int index, int thread_id, uint64_t seq) {
    return data_dir_for_index(config, index) /
           ("tmp_" + zero_pad(index, 8) + "_t" + zero_pad(thread_id, 3) + "_" + std::to_string(seq) + ".bin");
}

static fs::path rename_dir_for_thread(const Config& config, int thread_id) {
    return rename_root(config) / ("thread_" + zero_pad(thread_id, 3));
}

static void ensure_dir_fsync(const fs::path& dir_path) {
    int fd = ::open(dir_path.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) {
        throw std::runtime_error("open(dir) failed for fsync: " + dir_path.string() + ": " + std::strerror(errno));
    }
    if (::fsync(fd) != 0) {
        int saved = errno;
        ::close(fd);
        throw std::runtime_error("fsync(dir) failed for " + dir_path.string() + ": " + std::strerror(saved));
    }
    ::close(fd);
}

static void write_payload_to_fd(int fd, int payload_bytes, const std::string& payload) {
    if (payload_bytes == 0) {
        return;
    }

    constexpr size_t kDirectAlignment = 4096;
    const size_t requested = static_cast<size_t>(payload_bytes);
    const size_t rounded = ((requested + kDirectAlignment - 1) / kDirectAlignment) * kDirectAlignment;

    void* raw_buffer = nullptr;
    if (::posix_memalign(&raw_buffer, kDirectAlignment, rounded) != 0) {
        throw std::runtime_error("posix_memalign failed for O_DIRECT write buffer");
    }

    auto* buffer = static_cast<char*>(raw_buffer);
    for (size_t offset = 0; offset < rounded; offset += payload.size()) {
        const size_t chunk = std::min(payload.size(), rounded - offset);
        std::memcpy(buffer + offset, payload.data(), chunk);
    }

    size_t written_total = 0;
    while (written_total < rounded) {
        ssize_t written = ::write(fd, buffer + written_total, rounded - written_total);
        if (written < 0) {
            const int saved = errno;
            std::free(raw_buffer);
            throw std::runtime_error(std::string("O_DIRECT write failed: ") + std::strerror(saved));
        }
        if (written == 0) {
            std::free(raw_buffer);
            throw std::runtime_error("O_DIRECT write returned 0");
        }
        written_total += static_cast<size_t>(written);
    }

    std::free(raw_buffer);

    if (rounded != requested && ::ftruncate(fd, static_cast<off_t>(requested)) != 0) {
        throw std::runtime_error(std::string("ftruncate after O_DIRECT write failed: ") + std::strerror(errno));
    }
}

static void create_or_replace_file(const fs::path& path, int payload_bytes, bool sync_file, bool sync_dir) {
    int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT, 0644);
    if (fd < 0) {
        throw std::runtime_error("open for write failed: " + path.string() + ": " + std::strerror(errno));
    }
    static const std::string payload(4096, 'x');
    if (payload_bytes > 0) {
        write_payload_to_fd(fd, payload_bytes, payload);
    }
    if (sync_file && ::fsync(fd) != 0) {
        int saved = errno;
        ::close(fd);
        throw std::runtime_error("fsync(file) failed: " + path.string() + ": " + std::strerror(saved));
    }
    if (::close(fd) != 0) {
        throw std::runtime_error("close failed: " + path.string() + ": " + std::strerror(errno));
    }
    if (sync_dir) {
        ensure_dir_fsync(path.parent_path());
    }
}

static void prepare_workspace(const Config& config) {
    const fs::path root_path(config.root);
    if (config.reset && fs::exists(root_path)) {
        fs::remove_all(root_path);
    }

    fs::create_directories(data_root(config));
    fs::create_directories(rename_root(config));

    for (int bucket = 0; bucket < config.fanout; ++bucket) {
        fs::create_directories(data_root(config) / ("dir_" + zero_pad(bucket, 4)));
    }
    for (int thread_id = 0; thread_id < config.threads; ++thread_id) {
        const fs::path dir = rename_dir_for_thread(config, thread_id);
        fs::create_directories(dir);
        create_or_replace_file(dir / "rename_a.bin", config.payload_bytes, false, false);
        create_or_replace_file(dir / "rename_b.bin", config.payload_bytes, false, false);
    }

    for (int index = 0; index < config.files; ++index) {
        create_or_replace_file(data_file_for_index(config, index), config.payload_bytes, false, false);
    }
}

static void cleanup_workspace(const Config& config) {
    fs::remove_all(fs::path(config.root));
}

static double percentile_us(std::vector<uint32_t> values, double pct) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double rank = pct / 100.0 * static_cast<double>(values.size() - 1);
    const size_t low = static_cast<size_t>(rank);
    const size_t high = std::min(values.size() - 1, low + 1);
    const double fraction = rank - static_cast<double>(low);
    return static_cast<double>(values[low]) + (static_cast<double>(values[high]) - static_cast<double>(values[low])) * fraction;
}

static std::string build_json(const Config& config, const AggregateResult& result) {
    const auto& lat = result.latency_us;
    uint32_t min_us = 0;
    uint32_t max_us = 0;
    double mean_us = 0.0;
    if (!lat.empty()) {
        auto minmax = std::minmax_element(lat.begin(), lat.end());
        min_us = *minmax.first;
        max_us = *minmax.second;
        uint64_t sum = std::accumulate(lat.begin(), lat.end(), uint64_t{0});
        mean_us = static_cast<double>(sum) / static_cast<double>(lat.size());
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "{\n";
    out << "  \"config\": {\n";
    out << "    \"root\": \"" << json_escape(config.root) << "\",\n";
    out << "    \"op\": \"" << json_escape(config.op) << "\",\n";
    out << "    \"threads\": " << config.threads << ",\n";
    out << "    \"warmup_sec\": " << config.warmup_sec << ",\n";
    out << "    \"duration_sec\": " << config.duration_sec << ",\n";
    out << "    \"files\": " << config.files << ",\n";
    out << "    \"fanout\": " << config.fanout << ",\n";
    out << "    \"payload_bytes\": " << config.payload_bytes << ",\n";
    out << "    \"seed\": " << config.seed << ",\n";
    out << "    \"sync_file\": " << (config.sync_file ? "true" : "false") << ",\n";
    out << "    \"sync_dir\": " << (config.sync_dir ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"setup_sec\": " << result.setup_sec << ",\n";
    out << "  \"elapsed_sec\": " << result.elapsed_sec << ",\n";
    out << "  \"attempts\": " << (result.operations + result.errors) << ",\n";
    out << "  \"operations\": " << result.operations << ",\n";
    out << "  \"errors\": " << result.errors << ",\n";
    out << "  \"ops_per_sec\": " << (result.elapsed_sec > 0.0 ? result.operations / result.elapsed_sec : 0.0) << ",\n";
    out << "  \"bytes_written\": " << result.bytes_written << ",\n";
    out << "  \"bytes_read\": " << result.bytes_read << ",\n";
    out << "  \"error_samples\": [";
    for (size_t i = 0; i < result.error_samples.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << "\"" << json_escape(result.error_samples[i]) << "\"";
    }
    out << "],\n";
    out << "  \"latency_us\": {\n";
    out << "    \"count\": " << lat.size() << ",\n";
    out << "    \"min\": " << min_us << ",\n";
    out << "    \"mean\": " << mean_us << ",\n";
    out << "    \"p50\": " << percentile_us(lat, 50.0) << ",\n";
    out << "    \"p95\": " << percentile_us(lat, 95.0) << ",\n";
    out << "    \"p99\": " << percentile_us(lat, 99.0) << ",\n";
    out << "    \"max\": " << max_us << "\n";
    out << "  }\n";
    out << "}\n";
    return out.str();
}

class Runner {
  public:
    explicit Runner(const Config& config) : config_(config) {}

    AggregateResult run() {
        const auto setup_begin = Clock::now();
        prepare_workspace(config_);
        const auto setup_end = Clock::now();

        start_time_ = Clock::now() + std::chrono::milliseconds(250);
        warmup_end_ = start_time_ + std::chrono::seconds(config_.warmup_sec);
        bench_end_ = warmup_end_ + std::chrono::seconds(config_.duration_sec);

        std::vector<std::thread> threads;
        std::vector<ThreadResult> results(static_cast<size_t>(config_.threads));
        threads.reserve(static_cast<size_t>(config_.threads));

        for (int thread_id = 0; thread_id < config_.threads; ++thread_id) {
            threads.emplace_back([&, thread_id]() { worker(thread_id, results[thread_id]); });
        }
        for (auto& thread : threads) {
            thread.join();
        }

        AggregateResult aggregate;
        aggregate.setup_sec = std::chrono::duration<double>(setup_end - setup_begin).count();
        aggregate.elapsed_sec = std::chrono::duration<double>(std::chrono::seconds(config_.duration_sec)).count();
        for (auto& result : results) {
            aggregate.operations += result.operations;
            aggregate.errors += result.errors;
            aggregate.bytes_written += result.bytes_written;
            aggregate.bytes_read += result.bytes_read;
            for (const auto& sample : result.error_samples) {
                if (aggregate.error_samples.size() >= 8) {
                    break;
                }
                aggregate.error_samples.push_back(sample);
            }
            aggregate.latency_us.insert(
                aggregate.latency_us.end(), result.latency_us.begin(), result.latency_us.end());
        }

        if (config_.cleanup) {
            cleanup_workspace(config_);
        }
        return aggregate;
    }

  private:
    void worker(int thread_id, ThreadResult& result) {
        std::mt19937_64 rng(config_.seed + static_cast<uint64_t>(thread_id) * 1000003ULL);
        std::uniform_int_distribution<int> file_dist(0, config_.files - 1);
        std::uniform_int_distribution<int> dir_dist(0, config_.fanout - 1);
        std::discrete_distribution<int> mixed_dist({80.0, 12.0, 7.0, 1.0});
        uint64_t local_seq = 0;
        bool rename_toggle = false;

        const int shard_begin = (config_.files * thread_id) / config_.threads;
        const int shard_end = (config_.files * (thread_id + 1)) / config_.threads;
        const int shard_size = std::max(0, shard_end - shard_begin);
        std::uniform_int_distribution<int> shard_dist(0, std::max(0, shard_size - 1));

        std::this_thread::sleep_until(start_time_);

        while (Clock::now() < bench_end_) {
            const auto op_begin = Clock::now();
            bool ok = true;
            uint64_t bytes_written = 0;
            uint64_t bytes_read = 0;

            try {
                const int file_index = shard_size > 0 ? (shard_begin + shard_dist(rng)) : file_dist(rng);
                const int dir_index = dir_dist(rng);
                std::string selected = config_.op;
                if (selected == "mixed") {
                    switch (mixed_dist(rng)) {
                        case 0: selected = "lookup_txn"; break;
                        case 1: selected = "publish_txn"; break;
                        case 2: selected = "evict_txn"; break;
                        default: selected = "readdir"; break;
                    }
                }

                if (selected == "stat") {
                    do_stat(file_index);
                } else if (selected == "access") {
                    do_access(file_index);
                } else if (selected == "open_close") {
                    do_open_close(file_index);
                } else if (selected == "create_unlink") {
                    bytes_written = do_create_unlink(thread_id, local_seq++);
                } else if (selected == "rename") {
                    do_rename(thread_id, rename_toggle);
                    rename_toggle = !rename_toggle;
                } else if (selected == "readdir") {
                    bytes_read = do_readdir(dir_index);
                } else if (selected == "lookup_txn") {
                    do_stat(file_index);
                    do_access(file_index);
                    do_open_close(file_index);
                } else if (selected == "publish_txn") {
                    bytes_written = do_publish(file_index, thread_id, local_seq++);
                } else if (selected == "evict_txn") {
                    bytes_written = do_evict(file_index);
                } else {
                    throw std::runtime_error("unsupported op: " + selected);
                }
            } catch (const std::exception& ex) {
                ok = false;
                if (result.error_samples.size() < 4) {
                    result.error_samples.push_back(ex.what());
                }
            }

            const auto op_end = Clock::now();
            if (op_end < warmup_end_) {
                continue;
            }
            if (op_begin >= bench_end_) {
                break;
            }

            result.operations += ok ? 1 : 0;
            result.errors += ok ? 0 : 1;
            result.bytes_written += bytes_written;
            result.bytes_read += bytes_read;
            const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(op_end - op_begin).count();
            result.latency_us.push_back(static_cast<uint32_t>(std::min<int64_t>(micros, std::numeric_limits<uint32_t>::max())));
        }
    }

    void do_stat(int index) const {
        struct stat st {};
        const fs::path path = data_file_for_index(config_, index);
        if (::stat(path.c_str(), &st) != 0) {
            throw std::runtime_error("stat failed: " + path.string() + ": " + std::strerror(errno));
        }
    }

    void do_access(int index) const {
        const fs::path path = data_file_for_index(config_, index);
        if (::access(path.c_str(), F_OK) != 0) {
            throw std::runtime_error("access failed: " + path.string() + ": " + std::strerror(errno));
        }
    }

    void do_open_close(int index) const {
        const fs::path path = data_file_for_index(config_, index);
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("open failed: " + path.string() + ": " + std::strerror(errno));
        }
        if (::close(fd) != 0) {
            throw std::runtime_error("close failed: " + path.string() + ": " + std::strerror(errno));
        }
    }

    uint64_t do_create_unlink(int thread_id, uint64_t seq) const {
        const int index = static_cast<int>(seq % static_cast<uint64_t>(config_.fanout));
        const fs::path dir = data_root(config_) / ("dir_" + zero_pad(index, 4));
        const fs::path path = dir / ("create_unlink_t" + zero_pad(thread_id, 3) + "_" + std::to_string(seq) + ".bin");
        create_or_replace_file(path, config_.payload_bytes, config_.sync_file, false);
        if (::unlink(path.c_str()) != 0) {
            throw std::runtime_error("unlink failed: " + path.string() + ": " + std::strerror(errno));
        }
        if (config_.sync_dir) {
            ensure_dir_fsync(dir);
        }
        return static_cast<uint64_t>(config_.payload_bytes);
    }

    void do_rename(int thread_id, bool toggle) const {
        const fs::path dir = rename_dir_for_thread(config_, thread_id);
        const fs::path a = dir / "rename_a.bin";
        const fs::path b = dir / "rename_b.bin";
        const fs::path src = toggle ? b : a;
        const fs::path dst = toggle ? a : b;
        if (::rename(src.c_str(), dst.c_str()) != 0) {
            throw std::runtime_error("rename failed: " + src.string() + " -> " + dst.string() + ": " + std::strerror(errno));
        }
        if (::rename(dst.c_str(), src.c_str()) != 0) {
            throw std::runtime_error("rename restore failed: " + dst.string() + " -> " + src.string() + ": " + std::strerror(errno));
        }
        if (config_.sync_dir) {
            ensure_dir_fsync(dir);
        }
    }

    uint64_t do_readdir(int dir_index) const {
        const fs::path dir = data_root(config_) / ("dir_" + zero_pad(dir_index, 4));
        DIR* handle = ::opendir(dir.c_str());
        if (!handle) {
            throw std::runtime_error("opendir failed: " + dir.string() + ": " + std::strerror(errno));
        }
        while (::readdir(handle) != nullptr) {
        }
        if (::closedir(handle) != 0) {
            throw std::runtime_error("closedir failed: " + dir.string() + ": " + std::strerror(errno));
        }
        return 0;
    }

    uint64_t do_publish(int index, int thread_id, uint64_t seq) const {
        const fs::path tmp = publish_tmp_for_index(config_, index, thread_id, seq);
        const fs::path final = data_file_for_index(config_, index);
        create_or_replace_file(tmp, config_.payload_bytes, config_.sync_file, false);
        if (::rename(tmp.c_str(), final.c_str()) != 0) {
            throw std::runtime_error("rename publish failed: " + tmp.string() + " -> " + final.string() + ": " + std::strerror(errno));
        }
        if (config_.sync_dir) {
            ensure_dir_fsync(final.parent_path());
        }
        return static_cast<uint64_t>(config_.payload_bytes);
    }

    uint64_t do_evict(int index) const {
        const fs::path final = data_file_for_index(config_, index);
        if (::unlink(final.c_str()) != 0) {
            throw std::runtime_error("unlink evict failed: " + final.string() + ": " + std::strerror(errno));
        }
        create_or_replace_file(final, config_.payload_bytes, config_.sync_file, false);
        if (config_.sync_dir) {
            ensure_dir_fsync(final.parent_path());
        }
        return static_cast<uint64_t>(config_.payload_bytes);
    }

    const Config& config_;
    Clock::time_point start_time_;
    Clock::time_point warmup_end_;
    Clock::time_point bench_end_;
};

int main(int argc, char** argv) {
    try {
        const Config config = parse_args(argc, argv);
        Runner runner(config);
        const AggregateResult result = runner.run();
        const std::string json = build_json(config, result);
        std::cout << json;
        if (!config.json_out.empty()) {
            std::ofstream out(config.json_out);
            if (!out) {
                throw std::runtime_error("failed to open json output path: " + config.json_out);
            }
            out << json;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
