cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(ppo)
# ✅ 开启调试构建模式并禁用优化（断点可用）
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O2")
# 编译标准 & 架构（适配 macOS M 系列）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)

# 为生成 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LibTorch 和 Eigen
set(CMAKE_PREFIX_PATH "/Users/abc/libtorch/libtorch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories("/opt/homebrew/include/eigen3")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 源文件列表（你也可以用 GLOB，但推荐显式列出）
set(SOURCES
    TrainSAC.cpp
    TestPPO.cpp
    ReplayBuffer.cpp
)

# 头文件（用于 IntelliSense，非编译）
set(HEADERS
    Models.h
    ProximalPolicyOptimization.h
    TestEnvironment.h
    ReplayBuffer.h
)

# 可执行文件：Train
add_executable(train_sac TrainSAC.cpp ReplayBuffer.cpp)
target_link_libraries(train_sac ${TORCH_LIBRARIES})
target_include_directories(train_sac PRIVATE ${CMAKE_CURRENT_SOURCE})

# # 可执行文件：Test
# add_executable(test_ppo TestPPO.cpp ReplayBuffer.cpp)
# target_link_libraries(test_ppo ${TORCH_LIBRARIES})
# target_include_directories(test_ppo PRIVATE ${CMAKE_CURRENT_SOURCE})

# # 添加虚拟 target，让 CMake 解析头文件（对 VSCode 有帮助）
# add_custom_target(headers SOURCES ${HEADERS})
# 让头文件参与 CMake 的 dependency graph
add_custom_target(force_headers_indexing SOURCES
    Models.h
    ProximalPolicyOptimization.h
    SoftActorCritic.h
    TestEnvironment.h
    ReplayBuffer.h
)
