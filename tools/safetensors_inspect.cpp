// safetensors_inspect.cpp — native C++17 safetensors header inspector.
// No Python, no third-party JSON dependency.
//
// Reads:
//   [0..8)   uint64 little-endian header length
//   [8..8+N) UTF-8 JSON header
//   payload  raw tensor bytes
//
// Verifies tensor dtype/shape byte counts and data_offsets bounds.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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
        for (const auto& kv : object) {
            if (kv.first == key) return &kv.second;
        }
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
        if (i_ != s_.size()) {
            throw std::runtime_error("trailing JSON content");
        }
        return v;
    }

private:
    const std::string& s_;
    std::size_t i_ = 0;

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
        if (i_ >= s_.size() || s_[i_] != c) {
            throw std::runtime_error(std::string("expected JSON character '") + c + "'");
        }
        ++i_;
    }

    Json parse_value() {
        skip_ws();
        const char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
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
        if (code <= 0x7F) {
            out.push_back(static_cast<char>(code));
        } else if (code <= 0x7FF) {
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
            const char c = s_[i_++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<unsigned>(c - 'A' + 10);
            else throw std::runtime_error("bad unicode escape");
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (i_ < s_.size()) {
            const char c = s_[i_++];
            if (c == '"') return out;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (i_ >= s_.size()) throw std::runtime_error("bad string escape");
            const char e = s_[i_++];
            switch (e) {
                case '"': out.push_back('"'); break;
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
                        const unsigned lo = parse_hex4();
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                        }
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
        const double value = std::strtod(begin, &end);
        if (end == begin) throw std::runtime_error("bad JSON number");
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
            const char c = peek();
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
            const std::string key = parse_string();
            skip_ws();
            expect(':');
            v.object.emplace_back(key, parse_value());
            skip_ws();
            const char c = peek();
            if (c == '}') {
                ++i_;
                return v;
            }
            expect(',');
        }
    }
};

static uint64_t dtype_size(const std::string& dtype) {
    if (dtype == "BOOL") return 1;
    if (dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3" || dtype == "F8_E5M2") return 1;
    if (dtype == "I16" || dtype == "F16" || dtype == "BF16") return 2;
    if (dtype == "I32" || dtype == "F32") return 4;
    if (dtype == "I64" || dtype == "F64") return 8;
    return 0;
}

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::string shape;
    uint64_t expected = 0;
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t nbytes = 0;
    bool bounds_ok = false;
    bool size_ok = false;
};

static std::string shape_string(const Json& shape_json) {
    std::string out;
    if (shape_json.type != Json::Type::Array) return "?";
    for (std::size_t k = 0; k < shape_json.array.size(); ++k) {
        if (k) out.push_back('x');
        const double d = shape_json.array[k].number;
        if (d < 0) out += "?";
        else out += std::to_string(static_cast<uint64_t>(d));
    }
    return out.empty() ? "scalar" : out;
}

static uint64_t shape_elements(const Json& shape_json) {
    if (shape_json.type != Json::Type::Array) return 0;
    uint64_t n = 1;
    for (const auto& dim_json : shape_json.array) {
        const double d = dim_json.number;
        if (d < 0) return 0;
        const uint64_t dim = static_cast<uint64_t>(d);
        if (dim && n > UINT64_MAX / dim) return UINT64_MAX;
        n *= dim;
    }
    return n;
}

int main(int argc, char** argv) {
    if (argc < 2 || (argc % 2) != 0) {
        std::cerr << "usage: safetensors_inspect.exe <file.safetensors> [--limit N] [--find SUBSTR]\n";
        return 2;
    }
    const std::string path = argv[1];
    std::size_t limit = 8;
    std::string find_text;
    for (int i = 2; i < argc; i += 2) {
        const std::string flag = argv[i];
        const std::string value = argv[i + 1];
        if (flag == "--limit") {
            limit = static_cast<std::size_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (flag == "--find") {
            find_text = value;
        } else {
            std::cerr << "unknown flag: " << flag << "\n";
            return 2;
        }
    }

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        return 2;
    }
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    if (file_size < 8) {
        std::cerr << "ERROR: file shorter than safetensors header\n";
        return 1;
    }

    uint64_t header_len64 = 0;
    f.seekg(0, std::ios::beg);
    if (!f.read(reinterpret_cast<char*>(&header_len64), 8)) {
        std::cerr << "ERROR: cannot read header length\n";
        return 1;
    }
    const uint64_t header_len = header_len64;
    const uint64_t payload_start = 8ull + header_len;
    if (header_len == 0 || payload_start > file_size || header_len > (64ull << 20)) {
        std::cerr << "ERROR: bad header length " << header_len << "\n";
        return 1;
    }

    std::vector<char> header_buf(static_cast<std::size_t>(header_len) + 1, 0);
    f.seekg(8, std::ios::beg);
    if (!f.read(header_buf.data(), static_cast<std::streamsize>(header_len))) {
        std::cerr << "ERROR: cannot read header JSON\n";
        return 1;
    }

    Json root;
    try {
        root = JsonParser(std::string(header_buf.data())).parse();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: JSON parse: " << e.what() << "\n";
        return 1;
    }
    if (root.type != Json::Type::Object) {
        std::cerr << "ERROR: safetensors header root is not an object\n";
        return 1;
    }

    const uint64_t payload_bytes = file_size - payload_start;
    std::vector<TensorInfo> tensors;
    std::size_t metadata_keys = 0;

    for (const auto& kv : root.object) {
        if (kv.first == "__metadata__") {
            if (kv.second.type == Json::Type::Object) metadata_keys = kv.second.object.size();
            continue;
        }
        const Json* dtype_json = kv.second.find("dtype");
        const Json* shape_json = kv.second.find("shape");
        const Json* offsets_json = kv.second.find("data_offsets");
        if (!dtype_json || !shape_json || !offsets_json ||
            dtype_json->type != Json::Type::String ||
            shape_json->type != Json::Type::Array ||
            offsets_json->type != Json::Type::Array ||
            offsets_json->array.size() != 2) {
            std::cerr << "ERROR: malformed tensor entry: " << kv.first << "\n";
            return 1;
        }

        TensorInfo t;
        t.name = kv.first;
        t.dtype = dtype_json->text;
        t.shape = shape_string(*shape_json);
        t.start = static_cast<uint64_t>(offsets_json->array[0].number);
        t.end = static_cast<uint64_t>(offsets_json->array[1].number);
        t.nbytes = (t.end >= t.start) ? (t.end - t.start) : 0;
        const uint64_t ds = dtype_size(t.dtype);
        t.expected = ds ? shape_elements(*shape_json) * ds : 0;
        t.bounds_ok = (t.end >= t.start && t.end <= payload_bytes);
        t.size_ok = (ds != 0 && t.expected == t.nbytes);
        tensors.push_back(t);
    }

    std::vector<std::pair<uint64_t, uint64_t>> intervals;
    for (const auto& t : tensors) intervals.emplace_back(t.start, t.end);
    std::sort(intervals.begin(), intervals.end());
    bool overlap = false;
    for (std::size_t k = 1; k < intervals.size(); ++k) {
        if (intervals[k].first < intervals[k - 1].second) overlap = true;
    }

    bool bounds_all = true;
    bool sizes_all = true;
    for (const auto& t : tensors) {
        bounds_all = bounds_all && t.bounds_ok;
        sizes_all = sizes_all && t.size_ok;
    }
    const bool payload_exact = (payload_bytes >= tensors.size()) &&
                               (!tensors.empty() ? intervals.back().second == payload_bytes : payload_bytes == 0);
    const bool ok = bounds_all && !overlap && sizes_all && payload_exact;

    std::cout << "file=" << path << "\n";
    std::cout << "bytes=" << file_size << "\n";
    std::cout << "header_bytes=" << header_len << "\n";
    std::cout << "payload_start=" << payload_start << "\n";
    std::cout << "payload_bytes=" << payload_bytes << "\n";
    std::cout << "metadata_keys=" << metadata_keys << "\n";
    std::cout << "tensors=" << tensors.size() << "\n";
    std::cout << "bounds_ok=" << (bounds_all ? "true" : "false") << "\n";
    std::cout << "overlap=" << (overlap ? "true" : "false") << "\n";
    std::cout << "shape_dtype_size_ok=" << (sizes_all ? "true" : "false") << "\n";
    std::cout << "payload_exact=" << (payload_exact ? "true" : "false") << "\n";
    std::vector<std::size_t> selected;
    for (std::size_t k = 0; k < tensors.size(); ++k) {
        if (find_text.empty() || tensors[k].name.find(find_text) != std::string::npos) {
            selected.push_back(k);
        }
    }
    const std::size_t shown = std::min(limit, selected.size());
    std::cout << "filter=\"" << find_text << "\"\n";
    std::cout << "matched=" << selected.size() << "\n";
    for (std::size_t s = 0; s < shown; ++s) {
        const std::size_t k = selected[s];
        const auto& t = tensors[k];
        std::cout << "tensor[" << k << "] name=" << t.name
                  << " dtype=" << t.dtype
                  << " shape=" << t.shape
                  << " bytes=" << t.nbytes
                  << " expected=" << t.expected
                  << " offsets=[" << t.start << "," << t.end << "]"
                  << " ok=" << (t.bounds_ok && t.size_ok ? "true" : "false") << "\n";
    }
    if (selected.size() > shown) std::cout << "... omitted=" << (selected.size() - shown) << "\n";
    std::cout << "verdict=" << (ok ? "SAFETENSORS_OK" : "SAFETENSORS_BAD") << "\n";
    return ok ? 0 : 1;
}
