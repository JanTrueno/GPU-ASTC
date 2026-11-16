cmake_minimum_required(VERSION 3.13)
project(GPU-ASTC CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Embed the compute shader as a string, regenerated on reconfigure.
set(SHADER_FILES shaders/astc_encode.comp)
set(SHADER_SOURCE "")
foreach(f ${SHADER_FILES})
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/${f} content)
  string(APPEND SHADER_SOURCE "${content}\n")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${f})
endforeach()
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/shader_source.h
  "#pragma once\nnamespace gpuastc { static const char* const kShaderSource = R\"GLSL(${SHADER_SOURCE})GLSL\"; }\n")

find_package(SDL2 REQUIRED)
find_package(Threads REQUIRED)
find_library(GLESV2_LIBRARY NAMES GLESv2 libGLESv2 REQUIRED)

# PNG decoding: libdeflate gives a much faster decoder than stb_image alone (optional).
find_path(LIBDEFLATE_INCLUDE_DIR libdeflate.h)
find_library(LIBDEFLATE_LIBRARY NAMES deflate libdeflate)
add_library(gpuastc_png STATIC src/png_decode.cpp)
target_include_directories(gpuastc_png PUBLIC src PRIVATE third_party)
if(LIBDEFLATE_INCLUDE_DIR AND LIBDEFLATE_LIBRARY)
  message(STATUS "libdeflate found: fast PNG decoder enabled")
  target_compile_definitions(gpuastc_png PUBLIC GPUASTC_HAVE_LIBDEFLATE)
  target_include_directories(gpuastc_png PRIVATE ${LIBDEFLATE_INCLUDE_DIR})
  target_link_libraries(gpuastc_png PUBLIC ${LIBDEFLATE_LIBRARY})
else()
  message(WARNING "libdeflate not found (apt install libdeflate-dev): PNG decoding falls back to stb_image, which is slower")
endif()

add_executable(gpuastc src/main.cpp src/astc_gles.cpp)
target_include_directories(gpuastc PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(gpuastc PRIVATE gpuastc_png Threads::Threads ${GLESV2_LIBRARY})
if(TARGET SDL2::SDL2)
  target_link_libraries(gpuastc PRIVATE SDL2::SDL2)
else()
  target_include_directories(gpuastc PRIVATE ${SDL2_INCLUDE_DIRS})
  target_link_libraries(gpuastc PRIVATE ${SDL2_LIBRARIES})
endif()

# compares the fast PNG decoder with stb_image: build/png_compare *.png
add_executable(png_compare tests/png_compare.cpp)
target_link_libraries(png_compare PRIVATE gpuastc_png)
