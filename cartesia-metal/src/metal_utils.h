#pragma once

#include <dlfcn.h>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace cartesia::mlx_ext {

inline const std::string& metallib_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&metallib_dir), &info)) {
      throw std::runtime_error(
          "cartesia-metal: unable to locate the extension binary directory.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

} // namespace cartesia::mlx_ext
