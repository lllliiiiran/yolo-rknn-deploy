cmake_minimum_required(VERSION 3.10)
project(yolo_rknn_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RGA 兼容性（RK3566 裁剪 NV12→RGB 时报 Invalid argument）
add_definitions(-DDISABLE_RGA)

# ======================== 选项 ========================
option(ENABLE_VISUALIZE "启用 OpenCV 可视化" ON)

# ======================== 第三方库路径 ========================
set(3RDPARTY_PATH ${CMAKE_SOURCE_DIR}/3rdparty)

# RKNN API
set(RKNN_API_PATH ${3RDPARTY_PATH}/librknn_api)
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)

# RGA
set(RGA_DIR ${3RDPARTY_PATH}/rga)
set(RGA_LIB ${RGA_DIR}/lib/aarch64/librga.so)

# STB
set(STB_DIR ${3RDPARTY_PATH}/stb)

# ======================== 查找依赖 ========================
find_package(OpenCV REQUIRED)

# ======================== 源文件 ========================
set(SOURCES
    src/main.cpp
    src/yolo_rknn.cpp
    src/decode.cpp
    src/preprocess.cpp
)

add_executable(yolo_rknn ${SOURCES})

target_include_directories(yolo_rknn PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${RKNN_API_INCLUDE_PATH}
    ${RGA_DIR}/include
    ${STB_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo_rknn PRIVATE
    ${RKNN_RT_LIB}
    ${RGA_LIB}
    ${OpenCV_LIBS}
    pthread
)

if(ENABLE_VISUALIZE)
    target_compile_definitions(yolo_rknn PRIVATE ENABLE_VISUALIZE)
endif()

# ======================== 安装 ========================
install(TARGETS yolo_rknn DESTINATION bin)
install(FILES ${RKNN_RT_LIB} ${RGA_LIB} DESTINATION lib)
install(DIRECTORY ${RKNN_API_INCLUDE_PATH}/ DESTINATION include FILES_MATCHING PATTERN "*.h")
