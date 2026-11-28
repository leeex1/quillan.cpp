// safetensors_to_qbin.cpp - native C++17 converter:
//   sovereign packed safetensors -> Quillan .qbin v2
//
// Hardened diagnostic build:
//   - token-string based owner-token-ID -> donor-row remap
//   - strict JSON numeric validation and depth cap
//   - diagnostics-only output allowlist, no --force, temp file + atomic rename

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    const Json* find(const std::string& key) const {
        if (type != Type::Object) return nullptr;
        for (const auto& kv : object) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& input) : s_(input) {}

    Json parse() {
        skip_ws();
        Json v = parse_value();
        skip_ws();
        if (i_ != s_.size()) throw std::runtime_error("trailing JSON content");
        return v;
    }

private:
    const std::string& s_;
    std::size_t i_ = 0;
    std::size_t depth_ = 0;

    struct DepthGuard {
        JsonParser& parser;
        explicit DepthGuard(JsonParser& p) : parser(p) {
            if (++parser.depth_ > 64) {
                --parser.depth_;
                throw std::runtime_error("JSON depth limit exceeded");
            }
        }
        ~DepthGuard() { --parser.depth_; }
    };

    void skip_ws() {
        while (i_ < s_.size()) {
            const char c = s_[i_];
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') ++i_;
            else break;
        }
    }

    char peek() const {
        if (i_ >= s_.size()) throw std::runtime_error("unexpected end of JSON");
        return s_[i_];
    }

    void expect(char c) {
        if (i_ >= s_.size() || s_[i_] != c)
            throw std::runtime_error(std::string("expected JSON character '") + c + "'");
        ++i_;
    }

    Json parse_value() {
        DepthGuard guard(*this);
        skip_ws();
        const char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '\"') {
            Json v;
            v.type = Json::Type::String;
            v.text = parse_string();
            return v;
        }
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

    static void append_utf8(std::string& out, unsigned code) {
        if (code <= 0x7F) out.push_back(static_cast<char>(code));
        else if (code <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    unsigned parse_hex4() {
        if (i_ + 4 > s_.size()) throw std::runtime_error("short unicode escape");
        unsigned value = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[i_++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<unsigned>(c - 'A' + 10);
            else throw std::runtime_error("bad unicode escape");
        }
        return value;
    }

    std::string parse_string() {
        expect('\"');
        std::string out;
        while (i_ < s_.size()) {
            char c = s_[i_++];
            if (c == '\"') return out;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (i_ >= s_.size()) throw std::runtime_error("bad string escape");
            char e = s_[i_++];
            switch (e) {
                case '\"': out.push_back('\"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    unsigned cp = parse_hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 6 <= s_.size() &&
                        s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                        i_ += 2;
                        unsigned lo = parse_hex4();
                        if (lo >= 0xDC00 && lo <= 0xDFFF)
                            cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: throw std::runtime_error("unsupported string escape");
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }

    Json parse_number() {
        const char* begin = s_.c_str() + i_;
        char* end = nullptr;
        double value = std::strtod(begin, &end);
        if (end == begin) throw std::runtime_error("bad JSON number");
        if (!std::isfinite(value)) throw std::runtime_error("non-finite JSON number");
        i_ += static_cast<std::size_t>(end - begin);
        Json v;
        v.type = Json::Type::Number;
        v.number = value;
        return v;
    }

    Json parse_bool() {
        Json v;
        v.type = Json::Type::Bool;
        if (s_.compare(i_, 4, "true") == 0) {
            v.boolean = true;
            i_ += 4;
        } else if (s_.compare(i_, 5, "false") == 0) {
            v.boolean = false;
            i_ += 5;
        } else {
            throw std::runtime_error("bad JSON boolean");
        }
        return v;
    }

    Json parse_null() {
        if (s_.compare(i_, 4, "null") != 0) throw std::runtime_error("bad JSON null");
        i_ += 4;
        Json v;
        v.type = Json::Type::Null;
        return v;
    }

    Json parse_array() {
        expect('[');
        Json v;
        v.type = Json::Type::Array;
        skip_ws();
        if (peek() == ']') {
            ++i_;
            return v;
        }
        while (true) {
            v.array.push_back(parse_value());
            skip_ws();
            char c = peek();
            if (c == ']') {
                ++i_;
                return v;
            }
            expect(',');
        }
    }

    Json parse_object() {
        expect('{');
        Json v;
        v.type = Json::Type::Object;
        skip_ws();
        if (peek() == '}') {
            ++i_;
            return v;
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            v.object.emplace_back(key, parse_value());
            skip_ws();
            char c = peek();
            if (c == '}') {
                ++i_;
                return v;
            }
            expect(',');
        }
    }
};

static uint64_t json_u64(const Json& j, const std::string& what) {
    if (j.type != Json::Type::Number) throw std::runtime_error(what + ": expected number");
    const double v = j.number;
    if (!std::isfinite(v) || v < 0.0 || std::floor(v) != v ||
        v > static_cast<double>(std::numeric_limits<uint64_t>::max()))
        throw std::runtime_error(what + ": invalid uint64");
    return static_cast<uint64_t>(v);
}

static int64_t json_i64(const Json& j, const std::string& what) {
    if (j.type != Json::Type::Number) throw std::runtime_error(what + ": expected number");
    const double v = j.number;
    if (!std::isfinite(v) || std::floor(v) != v ||
        v < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
        v > static_cast<double>(std::numeric_limits<int64_t>::max()))
        throw std::runtime_error(what + ": invalid int64");
    return static_cast<int64_t>(v);
}

static uint64_t dtype_size_safetensors(const std::string& dt) {
    if (dt == "BOOL" || dt == "U8" || dt == "I8" || dt == "F8_E4M3" || dt == "F8_E5M2") return 1;
    if (dt == "I16" || dt == "F16" || dt == "BF16") return 2;
    if (dt == "I32" || dt == "F32") return 4;
    if (dt == "I64" || dt == "F64") return 8;
    return 0;
}

static int qbin_dtype(const std::string& dt) {
    if (dt == "F32") return 0;
    if (dt == "F16") return 1;
    if (dt == "I8") return 2;
    if (dt == "U8") return 3;
    if (dt == "I32") return 4;
    return -1;
}

static uint64_t qbin_dtype_size(int dt) {
    if (dt == 0 || dt == 4) return 4;
    if (dt == 1) return 2;
    if (dt == 2 || dt == 3) return 1;
    return 0;
}

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t nbytes = 0;
    int qdtype = -1;
};

static uint64_t shape_elements(const Json& shape_json, const std::string& what) {
    if (shape_json.type != Json::Type::Array) throw std::runtime_error(what + ": shape is not array");
    uint64_t n = 1;
    for (std::size_t k = 0; k < shape_json.array.size(); ++k) {
        const uint64_t dim = json_u64(shape_json.array[k], what + ":shape dim");
        if (dim == 0) return 0;
        if (n > UINT64_MAX / dim) throw std::runtime_error(what + ": shape overflow");
        n *= dim;
    }
    return n;
}

static bool load_safetensors_index(const std::string& path,
                                   std::map<std::string, TensorInfo>& tensors,
                                   uint64_t& payload_start) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open input safetensors: " + path);
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    if (file_size < 8) throw std::runtime_error("input too small for safetensors header");

    uint64_t header_len64 = 0;
    f.seekg(0, std::ios::beg);
    if (!f.read(reinterpret_cast<char*>(&header_len64), 8))
        throw std::runtime_error("cannot read header length");
    const uint64_t header_len = header_len64;
    payload_start = 8ull + header_len;
    if (header_len == 0 || header_len > (64ull << 20) || payload_start > file_size)
        throw std::runtime_error("bad safetensors header length");

    std::vector<char> header_buf(static_cast<std::size_t>(header_len) + 1, 0);
    f.seekg(8, std::ios::beg);
    if (!f.read(header_buf.data(), static_cast<std::streamsize>(header_len)))
        throw std::runtime_error("cannot read header JSON");

    Json root = JsonParser(std::string(header_buf.data())).parse();
    if (root.type != Json::Type::Object)
        throw std::runtime_error("safetensors header root is not an object");
    const uint64_t payload_bytes = file_size - payload_start;

    for (const auto& kv : root.object) {
        if (kv.first == "__metadata__") continue;
        const Json* dtype_j = kv.second.find("dtype");
        const Json* shape_j = kv.second.find("shape");
        const Json* offsets_j = kv.second.find("data_offsets");
        if (!dtype_j || dtype_j->type != Json::Type::String ||
            !shape_j || shape_j->type != Json::Type::Array ||
            !offsets_j || offsets_j->type != Json::Type::Array || offsets_j->array.size() != 2)
            throw std::runtime_error("malformed tensor entry: " + kv.first);

        TensorInfo t;
        t.name = kv.first;
        t.dtype = dtype_j->text;
        for (const auto& d : shape_j->array)
            t.shape.push_back(json_i64(d, "tensor shape: " + t.name));
        t.start = json_u64(offsets_j->array[0], "tensor offset start: " + t.name);
        t.end = json_u64(offsets_j->array[1], "tensor offset end: " + t.name);
        t.nbytes = t.end >= t.start ? t.end - t.start : 0;
        t.qdtype = qbin_dtype(t.dtype);
        const uint64_t ds = dtype_size_safetensors(t.dtype);
        const uint64_t elements = shape_elements(*shape_j, "tensor: " + t.name);
        if (ds == 0 || elements == 0 || t.end <= t.start || t.end > payload_bytes ||
            ds > UINT64_MAX / elements || elements * ds != t.nbytes)
            throw std::runtime_error("invalid tensor layout: " + t.name);
        tensors[t.name] = t;
    }
    return true;
}

struct Profile {
    std::string key;
    std::string arch_name;
    int arch_id = 0;
    int vocab = 151648;
    int hidden = 0;
    int ffn = 0;
    int layers = 0;
    int heads = 0;
    int kv_heads = 0;
    int head_dim = 64;
    int group_size = 32;
    int planes = 3;
    int eggroll_rank = 8;
};

static Profile profile_for(const std::string& key) {
    Profile p;
    p.key = key;
    p.vocab = 151648;
    p.head_dim = 64;
    p.group_size = 32;
    p.planes = 3;
    p.eggroll_rank = 8;
    if (key == "mini") {
        p.arch_name = "Quillan-Sovereign-Mini";
        p.arch_id = 1;
        p.hidden = 896;
        p.ffn = 4864;
        p.layers = 24;
        p.heads = 14;
        p.kv_heads = 2;
    } else if (key == "main" || key == "flagship") {
        p.arch_name = "Quillan-Sovereign-Flagship";
        p.arch_id = 2;
        p.hidden = 2048;
        p.ffn = 5120;
        p.layers = 16;
        p.heads = 32;
        p.kv_heads = 8;
    } else {
        throw std::runtime_error("unknown profile: " + key);
    }
    return p;
}

static std::unordered_map<std::string, int64_t> load_tokenizer_vocab(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open tokenizer: " + path);
    const std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Json root = JsonParser(data).parse();
    const Json* model = root.find("model");
    if (!model) throw std::runtime_error("tokenizer missing model object: " + path);
    const Json* vocab = model->find("vocab");
    if (!vocab || vocab->type != Json::Type::Object)
        throw std::runtime_error("tokenizer missing model.vocab object: " + path);

    std::unordered_map<std::string, int64_t> out;
    std::unordered_set<int64_t> seen_ids;
    out.reserve(vocab->object.size());
    seen_ids.reserve(vocab->object.size());
    for (const auto& kv : vocab->object) {
        const int64_t id = json_i64(kv.second, "tokenizer vocab id: " + kv.first);
        if (id < 0) throw std::runtime_error("negative tokenizer id: " + kv.first);
        if (!seen_ids.insert(id).second)
            throw std::runtime_error("duplicate tokenizer id: " + std::to_string(id));
        out.emplace(kv.first, id);
    }
    return out;
}

static std::vector<int64_t> build_owner_to_donor_rows(const std::string& owner_path,
                                                      const std::string& donor_path,
                                                      int owner_vocab) {
    const auto owner = load_tokenizer_vocab(owner_path);
    const auto donor = load_tokenizer_vocab(donor_path);
    std::vector<int64_t> remap(static_cast<std::size_t>(owner_vocab), -1);
    std::unordered_set<int64_t> seen_owner_ids;

    for (const auto& kv : owner) {
        const int64_t owner_id = kv.second;
        if (owner_id < 0 || owner_id >= owner_vocab)
            throw std::runtime_error("owner tokenizer id out of locked range: " + std::to_string(owner_id));
        if (!seen_owner_ids.insert(owner_id).second)
            throw std::runtime_error("duplicate owner id after validation");
        const auto it = donor.find(kv.first);
        if (it != donor.end()) remap[static_cast<std::size_t>(owner_id)] = it->second;
    }
    if (seen_owner_ids.size() != static_cast<std::size_t>(owner_vocab))
        throw std::runtime_error("owner tokenizer does not cover locked vocab exactly");

    std::size_t mapped = 0;
    for (int64_t d : remap) if (d >= 0) ++mapped;
    if (mapped == 0) throw std::runtime_error("owner→donor token map is empty");
    std::cout << "token_mapping_owner_vocab=" << owner_vocab << "\n";
    std::cout << "token_mapping_donor_vocab=" << donor.size() << "\n";
    std::cout << "token_mapping_mapped=" << mapped << "\n";
    std::cout << "token_mapping_unmapped=" << (remap.size() - mapped) << "\n";
    return remap;
}

enum CopyMode { COPY_EXACT = 0, COPY_EMB_I8 = 1, COPY_EMB_SCALE = 2 };

struct OutTensor {
    std::string name;
    int dtype = -1;
    std::vector<int> dims;
    uint64_t src_start = 0;
    uint64_t src_nbytes = 0;
    uint64_t out_offset = 0;
    uint64_t out_nbytes = 0;
    CopyMode mode = COPY_EXACT;
};

static void write_i32(std::ostream& o, int32_t v) { o.write(reinterpret_cast<const char*>(&v), 4); }
static void write_u32(std::ostream& o, uint32_t v) { o.write(reinterpret_cast<const char*>(&v), 4); }
static void write_u64(std::ostream& o, uint64_t v) { o.write(reinterpret_cast<const char*>(&v), 8); }
static void write_f32(std::ostream& o, float v) { o.write(reinterpret_cast<const char*>(&v), 4); }

static void write_cstr64(std::ostream& o, const std::string& s) {
    char buf[64];
    std::memset(buf, 0, sizeof(buf));
    const std::size_t n = std::min<std::size_t>(s.size(), 63);
    std::memcpy(buf, s.data(), n);
    o.write(buf, 64);
}

static void write_zeros(std::ostream& o, uint64_t bytes) {
    static const char zeros[65536] = {0};
    while (bytes) {
        const std::size_t chunk = static_cast<std::size_t>(std::min<uint64_t>(bytes, sizeof(zeros)));
        o.write(zeros, static_cast<std::streamsize>(chunk));
        bytes -= chunk;
    }
}

static void copy_source_bytes(std::istream& src, std::ostream& dst,
                              uint64_t absolute_start, uint64_t bytes) {
    if (bytes == 0) return;
    src.clear();
    src.seekg(static_cast<std::streamoff>(absolute_start), std::ios::beg);
    static const std::size_t chunk_size = 1u << 20;
    std::vector<char> buf(chunk_size);
    uint64_t left = bytes;
    while (left) {
        const std::size_t want = static_cast<std::size_t>(std::min<uint64_t>(left, chunk_size));
        src.read(buf.data(), static_cast<std::streamsize>(want));
        if (src.gcount() != static_cast<std::streamsize>(want))
            throw std::runtime_error("short read while streaming tensor payload");
        dst.write(buf.data(), static_cast<std::streamsize>(want));
        left -= want;
    }
}

static void copy_rows_remapped(std::istream& src, std::ostream& dst,
                               uint64_t src_start, const std::vector<int64_t>& remap,
                               uint64_t row_bytes, bool is_scale) {
    std::vector<char> rowbuf(static_cast<std::size_t>(row_bytes));
    const char one[2] = {0x00, 0x3C};
    for (std::size_t owner_row = 0; owner_row < remap.size(); ++owner_row) {
        const int64_t donor_row = remap[owner_row];
        if (donor_row < 0) {
            if (is_scale) dst.write(one, 2);
            else write_zeros(dst, row_bytes);
            continue;
        }
        src.clear();
        const uint64_t off = src_start + static_cast<uint64_t>(donor_row) * row_bytes;
        src.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        src.read(rowbuf.data(), static_cast<std::streamsize>(row_bytes));
        if (src.gcount() != static_cast<std::streamsize>(row_bytes))
            throw std::runtime_error("short read in remapped embedding row " + std::to_string(owner_row));
        dst.write(rowbuf.data(), static_cast<std::streamsize>(row_bytes));
    }
}

static void reject_symlink_components(const fs::path& p) {
    fs::path cur;
    for (const auto& part : p) {
        cur /= part;
        std::error_code ec;
        const fs::file_status st = fs::symlink_status(cur, ec);
        if (ec) continue;
        if (fs::is_symlink(st)) throw std::runtime_error("symlink/reparse component rejected: " + cur.string());
    }
}

static fs::path validate_diagnostic_output(const std::string& out_path, const fs::path& input_canonical) {
    fs::path out = fs::absolute(fs::path(out_path)).lexically_normal();
    if (out.extension() != ".qbin") throw std::runtime_error("diagnostic output must end in .qbin");
    const std::string fname = out.filename().string();
    if (fname.find("_NONCANONICAL") == std::string::npos)
        throw std::runtime_error("diagnostic output filename must contain _NONCANONICAL");

    const fs::path root_abs = fs::absolute(fs::path("F:/02_QUILLAN/exports/diagnostics"));
    std::error_code ec;
    fs::create_directories(root_abs, ec);
    if (ec) throw std::runtime_error("cannot create diagnostics root: " + ec.message());
    const fs::path root = fs::weakly_canonical(root_abs);
    reject_symlink_components(root);
    reject_symlink_components(out.parent_path());

    fs::path parent;
    try {
        parent = fs::weakly_canonical(out.parent_path());
    } catch (const std::exception&) {
        throw std::runtime_error("output parent does not exist: " + out.parent_path().string());
    }
    if (parent != root) throw std::runtime_error("output must be directly inside F:/02_QUILLAN/exports/diagnostics");

    if (fs::exists(out)) throw std::runtime_error("output already exists; overwrite is disabled: " + out.string());
    reject_symlink_components(out);
    if (input_canonical == fs::weakly_canonical(out))
        throw std::runtime_error("input and output resolve to the same path");
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: safetensors_to_qbin.exe <input.safetensors> <output.qbin> <mini|main> "
                     "--owner-tokenizer <path> --donor-tokenizer <path> --donor <name>\n";
        return 2;
    }

    std::string owner_tokenizer;
    std::string donor_tokenizer;
    std::string donor_name;
    for (int i = 4; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--force") {
            std::cerr << "ERROR: --force is disabled\n";
            return 2;
        }
        if (i + 1 >= argc) {
            std::cerr << "ERROR: missing value for " << flag << "\n";
            return 2;
        }
        const std::string value = argv[++i];
        if (flag == "--owner-tokenizer") owner_tokenizer = value;
        else if (flag == "--donor-tokenizer") donor_tokenizer = value;
        else if (flag == "--donor") donor_name = value;
        else {
            std::cerr << "ERROR: unknown flag: " << flag << "\n";
            return 2;
        }
    }
    if (owner_tokenizer.empty() || donor_tokenizer.empty() || donor_name.empty()) {
        std::cerr << "ERROR: --owner-tokenizer, --donor-tokenizer, and --donor are required\n";
        return 2;
    }

    fs::path temp_path;
    try {
        const std::string in_path = argv[1];
        const std::string out_path = argv[2];
        const std::string profile_key = argv[3];

        const fs::path input_canonical = fs::weakly_canonical(fs::absolute(fs::path(in_path)));
        reject_symlink_components(input_canonical);
        const fs::path out = validate_diagnostic_output(out_path, input_canonical);

        Profile p = profile_for(profile_key);
        std::map<std::string, TensorInfo> tensors;
        uint64_t payload_start = 0;
        load_safetensors_index(input_canonical.string(), tensors, payload_start);

        auto require = [&](const std::string& name) -> const TensorInfo& {
            auto it = tensors.find(name);
            if (it == tensors.end()) throw std::runtime_error("missing required tensor: " + name);
            return it->second;
        };

        const TensorInfo& emb = require("embed_tokens.weight_int8");
        const TensorInfo& emb_scale = require("embed_tokens.weight_scale");
        const TensorInfo& norm = require("norm.weight");
        if (emb.shape.size() != 2 || emb.shape[1] != p.hidden)
            throw std::runtime_error("embedding hidden dimension mismatch");
        if (emb_scale.shape.size() != 2 || emb_scale.shape[1] != 1 || emb_scale.dtype != "F16")
            throw std::runtime_error("embedding scale must be F16 [rows,1]");
        if (norm.dtype != "F32" || norm.shape.size() != 1 || norm.shape[0] != p.hidden)
            throw std::runtime_error("norm.weight must be F32 [hidden_dim]");

        const uint64_t src_rows = static_cast<uint64_t>(emb.shape[0]);
        const std::vector<int64_t> remap = build_owner_to_donor_rows(
            owner_tokenizer, donor_tokenizer, p.vocab);
        std::size_t bad_rows = 0;
        for (int64_t d : remap) if (d >= static_cast<int64_t>(src_rows)) ++bad_rows;
        if (bad_rows)
            throw std::runtime_error("donor row map exceeds embedding rows for " + std::to_string(bad_rows) + " tokens");

        std::vector<OutTensor> out_tensors;
        for (const auto& kv : tensors) {
            const TensorInfo& t = kv.second;
            OutTensor o;
            o.name = t.name;
            o.dtype = t.qdtype;
            o.src_start = payload_start + t.start;
            o.src_nbytes = t.nbytes;

            if (t.name == "embed_tokens.weight_int8") {
                if (t.dtype != "I8") throw std::runtime_error("embed_tokens.weight_int8 is not I8");
                o.mode = COPY_EMB_I8;
                o.dims = {p.vocab, p.hidden};
                o.out_nbytes = static_cast<uint64_t>(p.vocab) * p.hidden;
            } else if (t.name == "embed_tokens.weight_scale") {
                o.mode = COPY_EMB_SCALE;
                o.dims = {p.vocab, 1};
                o.out_nbytes = static_cast<uint64_t>(p.vocab) * 2;
            } else {
                if (o.dtype < 0 || t.name.size() >= 64 || t.shape.empty() || t.shape.size() > 4)
                    continue;
                o.mode = COPY_EXACT;
                o.dims.assign(t.shape.begin(), t.shape.end());
                for (int d : o.dims) if (d <= 0) { o.dims.clear(); break; }
                if (o.dims.empty()) continue;
                uint64_t expected = qbin_dtype_size(o.dtype);
                for (int d : o.dims) {
                    if (expected > UINT64_MAX / static_cast<uint64_t>(d)) {
                        expected = UINT64_MAX;
                        break;
                    }
                    expected *= static_cast<uint64_t>(d);
                }
                if (expected != t.nbytes) continue;
                o.out_nbytes = t.nbytes;
            }
            out_tensors.push_back(o);
        }

        const uint64_t table_off = 428;
        const uint64_t payload_off = table_off + static_cast<uint64_t>(out_tensors.size()) * 104;
        uint64_t cursor = 0;
        for (auto& o : out_tensors) {
            o.out_offset = cursor;
            cursor += o.out_nbytes;
        }

        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        temp_path = out;
        temp_path += ".tmp-" + std::to_string(stamp);
        if (fs::exists(temp_path)) throw std::runtime_error("temp output already exists");

        std::ofstream fo(temp_path, std::ios::binary | std::ios::trunc);
        if (!fo) throw std::runtime_error("cannot open temp output qbin: " + temp_path.string());

        write_u32(fo, 0x4E4C4C51u);
        write_u32(fo, 2);

        write_i32(fo, p.vocab);
        write_i32(fo, p.hidden);
        write_i32(fo, p.ffn);
        write_i32(fo, p.layers);
        write_i32(fo, 34);
        write_i32(fo, 34);
        write_i32(fo, p.eggroll_rank);
        write_i32(fo, 2097152);
        write_i32(fo, 3);
        write_f32(fo, 0.92f);
        write_f32(fo, 0.85f);

        write_i32(fo, p.arch_id);
        write_i32(fo, p.head_dim);
        write_i32(fo, p.heads);
        write_i32(fo, p.kv_heads);
        write_i32(fo, p.layers);
        write_i32(fo, p.planes);
        write_i32(fo, p.planes);
        write_i32(fo, p.group_size);
        write_i32(fo, 1);
        write_i32(fo, 2);
        for (int i = 0; i < 64; ++i) write_i32(fo, i < p.layers ? i : -1);
        write_f32(fo, 1e-6f);
        write_f32(fo, 1e6f);
        write_i32(fo, 2097152);
        write_cstr64(fo, p.arch_name);

        write_u32(fo, static_cast<uint32_t>(out_tensors.size()));
        for (const auto& o : out_tensors) {
            write_cstr64(fo, o.name);
            write_u32(fo, static_cast<uint32_t>(o.dtype));
            write_u32(fo, static_cast<uint32_t>(o.dims.size()));
            int32_t dims[4] = {0, 0, 0, 0};
            for (std::size_t i = 0; i < o.dims.size() && i < 4; ++i) dims[i] = o.dims[i];
            for (int d : dims) write_i32(fo, d);
            write_u64(fo, o.out_offset);
            write_u64(fo, o.out_nbytes);
        }
        if (static_cast<uint64_t>(fo.tellp()) != payload_off)
            throw std::runtime_error("internal error: qbin header size mismatch");

        std::ifstream fin(input_canonical, std::ios::binary);
        if (!fin) throw std::runtime_error("cannot reopen input safetensors for payload streaming");

        for (const auto& o : out_tensors) {
            if (o.mode == COPY_EXACT) {
                copy_source_bytes(fin, fo, o.src_start, o.out_nbytes);
            } else {
                const bool is_scale = (o.mode == COPY_EMB_SCALE);
                const uint64_t row_bytes = is_scale ? 2ull : static_cast<uint64_t>(p.hidden);
                copy_rows_remapped(fin, fo, o.src_start, remap, row_bytes, is_scale);
            }
        }

        if (static_cast<uint64_t>(fo.tellp()) != payload_off + cursor)
            throw std::runtime_error("internal error: qbin payload size mismatch");

        fo.flush();
        if (!fo) throw std::runtime_error("failed to flush temp output");
        fo.close();
        if (fo.fail()) throw std::runtime_error("failed to close temp output");
        fs::rename(temp_path, out);
        temp_path.clear();

        std::cout << "input=" << input_canonical.string() << "\n";
        std::cout << "output=" << out.string() << "\n";
        std::cout << "profile=" << p.key << "\n";
        std::cout << "donor=" << donor_name << "\n";
        std::size_t mapped_count = 0;
        for (int64_t d : remap) if (d >= 0) ++mapped_count;
        std::cout << "token_mapping_mapped=" << mapped_count << "\n";
        std::cout << "sections=" << out_tensors.size() << "\n";
        std::cout << "header_bytes=" << payload_off << "\n";
        std::cout << "payload_bytes=" << cursor << "\n";
        std::cout << "embedding_source_rows=" << src_rows << "\n";
        std::cout << "embedding_target_rows=" << p.vocab << "\n";
        std::cout << "verdict=QBIN_V2_WRITTEN\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        if (!temp_path.empty()) {
            std::error_code ec;
            fs::remove(temp_path, ec);
        }
        return 1;
    }
}
