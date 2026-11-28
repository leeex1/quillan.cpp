// qbin_inspect.cpp — standalone C++17 inspector for Quillan .qbin v1/v2 files.
// No Python. Prints header, config, architecture descriptor, tensor sections,
// and validates that every section fits inside the file payload.

#include "quillan.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static std::string fixed_string(const char* s, std::size_t n) {
    std::string out(s, n);
    const std::size_t z = out.find('\0');
    if (z != std::string::npos) out.resize(z);
    return out;
}

template <typename T>
static bool read_at(std::ifstream& f, std::streamoff off, T& out) {
    f.seekg(off, std::ios::beg);
    return static_cast<bool>(f.read(reinterpret_cast<char*>(&out), sizeof(T)));
}

static const char* dtype_name(std::uint32_t d) {
    switch (d) {
        case QLLN_DT_F32: return "F32";
        case QLLN_DT_F16: return "F16";
        case QLLN_DT_I8:  return "I8";
        case QLLN_DT_U8:  return "U8";
        case QLLN_DT_I32: return "I32";
        default: return "UNKNOWN";
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: qbin_inspect.exe <model.qbin>\n";
        return 2;
    }
    const std::string path = argv[1];
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        return 2;
    }
    const std::uint64_t file_size = static_cast<std::uint64_t>(f.tellg());

    std::uint32_t magic = 0, version = 0;
    if (!read_at(f, 0, magic) || !read_at(f, 4, version)) {
        std::cerr << "ERROR: cannot read qbin header\n";
        return 2;
    }

    std::cout << "file=" << path << "\n";
    std::cout << "bytes=" << file_size << "\n";
    std::cout << std::hex << std::setw(8) << std::setfill('0')
              << "magic=0x" << magic << std::dec << "\n";
    std::cout << "version=" << version << "\n";

    if (magic != QUILLAN_MAGIC) {
        std::cerr << "ERROR: bad magic, expected 0x" << std::hex << QUILLAN_MAGIC << std::dec << "\n";
        return 1;
    }

    quillan_config cfg{};
    if (!read_at(f, 8, cfg)) {
        std::cerr << "ERROR: cannot read config\n";
        return 2;
    }
    std::cout << "config.vocab_size=" << cfg.vocab_size << "\n"
              << "config.hidden_dim=" << cfg.hidden_dim << "\n"
              << "config.ffn_dim=" << cfg.ffn_dim << "\n"
              << "config.num_layers=" << cfg.num_layers << "\n"
              << "config.num_experts=" << cfg.num_experts << "\n"
              << "config.active_experts=" << cfg.active_experts << "\n"
              << "config.eggroll_rank=" << cfg.eggroll_rank << "\n"
              << "config.max_seq_len=" << cfg.max_seq_len << "\n"
              << "config.diffusion_steps=" << cfg.diffusion_steps << "\n"
              << "config.diffusion_halt=" << cfg.diffusion_halt << "\n"
              << "config.ccrl_threshold=" << cfg.ccrl_threshold << "\n";

    if (version == 1) {
        std::cout << "sections=0\n";
        std::cout << "verdict=CONFIG_ONLY_V1\n";
        return 0;
    }
    if (version != QUILLAN_QBIN_VERSION) {
        std::cerr << "ERROR: unsupported qbin version " << version << "\n";
        return 1;
    }

    quillan_arch_desc arch{};
    std::uint32_t section_count = 0;
    if (!read_at(f, 52, arch) || !read_at(f, 424, section_count)) {
        std::cerr << "ERROR: cannot read arch descriptor or section count\n";
        return 2;
    }
    std::cout << "arch.id=" << arch.arch_id << "\n"
              << "arch.name=" << fixed_string(arch.arch_name, sizeof(arch.arch_name)) << "\n"
              << "arch.head_dim=" << arch.head_dim << "\n"
              << "arch.attention_heads=" << arch.num_attention_heads << "\n"
              << "arch.kv_heads=" << arch.num_kv_heads << "\n"
              << "arch.moe_layers=" << arch.num_moe_layers << "\n"
              << "arch.group_size=" << arch.group_size << "\n"
              << "arch.planes_attn=" << arch.num_planes_attn << "\n"
              << "arch.planes_ffn=" << arch.num_planes_ffn << "\n"
              << "arch.flags=" << arch.flags << "\n"
              << "sections=" << section_count << "\n";

    const std::uint64_t payload_start = 428ull + static_cast<std::uint64_t>(section_count) * sizeof(quillan_tensor_section);
    if (payload_start > file_size) {
        std::cerr << "ERROR: section table exceeds file size\n";
        return 1;
    }

    bool ok = true;
    std::vector<quillan_tensor_section> sections(section_count);
    for (std::uint32_t i = 0; i < section_count; ++i) {
        if (!read_at(f, static_cast<std::streamoff>(428ull + static_cast<std::uint64_t>(i) * sizeof(quillan_tensor_section)), sections[i])) {
            std::cerr << "ERROR: cannot read section " << i << "\n";
            return 2;
        }
        const auto& s = sections[i];
        const std::uint64_t end = payload_start + s.offset + s.nbytes;
        if (end > file_size) ok = false;
        std::cout << "tensor[" << i << "] name=" << fixed_string(s.name, sizeof(s.name))
                  << " dtype=" << dtype_name(s.dtype)
                  << " rank=" << s.rank
                  << " dims=" << s.dims[0] << "x" << s.dims[1] << "x" << s.dims[2] << "x" << s.dims[3]
                  << " offset=" << s.offset
                  << " nbytes=" << s.nbytes
                  << " bounds=" << (end <= file_size ? "OK" : "BAD") << "\n";
    }

    std::cout << "payload_start=" << payload_start << "\n";
    std::cout << "verdict=" << (ok ? "QBIN_V2_SECTIONS_OK" : "QBIN_V2_BAD_BOUNDS") << "\n";
    return ok ? 0 : 1;
}
